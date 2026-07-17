"""
EcoNetMap - Data Filtering Module
==================================
This module provides comprehensive filtering capabilities for vegetation monitoring data.
It enables filtering by taxonomic groups, habitat types, time periods, geographic regions,
and species characteristics (native/invasive status). The module supports reproducible
random sampling, creates merged datasets combining species and occurrence data, and exports
filtered results to SQLite databases for efficient downstream processing.

Supports NOVANA data format and compatible vegetation monitoring datasets.

Part of the EcoNetMap toolkit (Data Management 2/2)
Author: Flemming Skov (fs@ecos.au.dk)
Last Updated: January 2026
"""

# Import packages for web applications
import streamlit as st

# Import packages for data manipulation and analysis
import pandas as pd
import numpy as np
import sqlite3
import traceback

# Import packages for file and system operations
from pathlib import Path
from datetime import datetime, date

# Import packages for type hints
from typing import Optional, List, Tuple, Dict

# Column-role mapping (lets users adapt the toolkit to their own column names)
from column_config import VEGETATION_ROLES, TAXA_ROLES, rename_to_canonical

# Constants
PREVIEW_LIMIT = 1000
DEFAULT_MIN_OCCURRENCES = 5
DEFAULT_SEED = 42
MAX_OCCURRENCE_THRESHOLD = 50
MAX_PLOTS_PER_HABITAT_MIN = 50
MAX_PLOTS_PER_HABITAT_MAX = 1000
MAX_PLOTS_PER_HABITAT_DEFAULT = 300
DEFAULT_UTM_X = 550000
DEFAULT_UTM_Y = 6230000
PARQUET_COMPRESSION = 'snappy'

# Page configuration
st.set_page_config(
    page_title="Data Filtering - EcoNetMap",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for consistent styling
st.markdown("""
<style>
    .stTextInput > label {
        font-weight: bold;
        color: #2c3e50;
    }
    .filter-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #3498db;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    div[data-testid="stExpander"] > details {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title and progress indicator
col1, col2 = st.columns([4, 1])
with col1:
    st.header("Data management")
    st.subheader("🔍 Filtering & selection")
    st.markdown("*Apply filters, refine selection, and export curated datasets for network analysis*")
with col2:
    pass

st.markdown("---")

# UTILITY FUNCTIONS
###################################################################################

def get_dataframe_info(df: pd.DataFrame) -> dict:
    """Get comprehensive dataframe information"""
    memory_usage_bytes = df.memory_usage(deep=True).sum()  
    memory_usage_mb = memory_usage_bytes / (1024 * 1024)
    
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_mb': round(memory_usage_mb, 2)
    }

def ensure_datetime_compatibility(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Ensure date column is properly formatted for filtering"""
    if date_column in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            # Try different date formats
            try:
                # First try standard datetime conversion
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            except (ValueError, TypeError) as e:
                try:
                    # Try integer format (YYYYMMDD)
                    df[date_column] = pd.to_datetime(df[date_column].astype(str), format='%Y%m%d', errors='coerce')
                except (ValueError, TypeError) as e:
                    st.warning(f"Could not convert date column '{date_column}': {str(e)}")
    return df

def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance and convert to kilometers"""
    distance_meters = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distance_meters / 1000

def summarize_data(df: pd.DataFrame) -> dict:
    """Create comprehensive summary statistics"""
    date_range = (None, None)
    if 'date' in df.columns:
        try:
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            date_range = (df['date'].min(), df['date'].max())
        except (ValueError, TypeError, KeyError):
            date_range = (None, None)
    
    return {
        'unique_plots': df['plot_id'].nunique() if 'plot_id' in df.columns else 0,
        'unique_nature_types': df['habitat_type'].nunique() if 'habitat_type' in df.columns else 0,
        'unique_species': df['species_key'].nunique() if 'species_key' in df.columns else 0,
        'total_records': len(df),
        'date_range': date_range
    }

# SAMPLING FUNCTIONS
###################################################################################

def safe_sample(group: pd.DataFrame, max_sample_size: int, random_seed: Optional[int] = None) -> np.ndarray:
    """Safely sample unique plots from a group with optional seed for reproducibility"""
    unique_plots = group['plot_id'].unique()
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    sample_size = min(max_sample_size, len(unique_plots))
    return np.random.choice(unique_plots, size=sample_size, replace=False)

def create_stratified_sample(df: pd.DataFrame, max_per_habitat: int, min_occurrences: int, 
                            random_seed: Optional[int] = None) -> pd.DataFrame:
    """Create stratified sample based on habitat types with reproducibility"""
    sampled_plots = []
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    for habitat, group in df.groupby('habitat_type'):
        sampled_plots.extend(safe_sample(group, max_per_habitat, random_seed))
    
    # Filter to sampled plots
    df_sampled = df[df['plot_id'].isin(sampled_plots)]
    
    # Apply minimum occurrence filter
    if 'species_key' in df_sampled.columns:
        species_counts = df_sampled['species_key'].value_counts()
        species_to_keep = species_counts[species_counts >= min_occurrences].index
        df_sampled = df_sampled[df_sampled['species_key'].isin(species_to_keep)]
    
    return df_sampled

def apply_random_sample(df: pd.DataFrame, sample_percentage: int, 
                       random_seed: Optional[int] = None) -> pd.DataFrame:
    """Apply random sampling to plots with optional seed"""
    if sample_percentage >= 100 or 'plot_id' not in df.columns:
        return df
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    unique_plots = df['plot_id'].unique()
    sample_size = int(sample_percentage * len(unique_plots) / 100)
    sampled_plots = np.random.choice(unique_plots, size=sample_size, replace=False)
    
    return df[df['plot_id'].isin(sampled_plots)]

# EXPORT FUNCTIONS
###################################################################################

def export_to_sqlite(df: pd.DataFrame, file_path: Path, metadata: dict,
                    filter_settings: dict) -> bool:
    """Export dataframe to SQLite with comprehensive table structure"""
    try:
        with sqlite3.connect(file_path) as conn:
            # Table 1: 'data' - Main filtered merged dataset
            df.to_sql('data', conn, if_exists='replace', index=False)

            # Table 2: 'plot_id' - Unique plot information
            if 'plot_id' in df.columns:
                plot_columns = ['plot_id']

                standard_plot_cols = [
                    'stationsNr', 'plotNr', 'year', 'progid', 'date',
                    'habitat_type', 'stedid', 'status', 'x', 'y',
                    'stednavn', 'subregion', 'region', 'major_type', 'taxonomi'
                ]

                species_specific_cols = [
                    'species_key', 'species',
                    'abundance', 'frekvens', 'forekomst', 'taxonomi_x', 'taxonomi_y'
                ]

                # Add available standard columns
                for col in standard_plot_cols:
                    if col in df.columns and col not in plot_columns:
                        plot_columns.append(col)

                # Add other non-species columns
                for col in df.columns:
                    if col not in plot_columns and col not in species_specific_cols:
                        plot_columns.append(col)

                plot_data = df[plot_columns].drop_duplicates('plot_id').reset_index(drop=True)
                plot_data.to_sql('plot_id', conn, if_exists='replace', index=False)

            # Table 3: 'species_list' - Species with occurrence counts
            if 'species_key' in df.columns:
                species_data = df['species_key'].value_counts().reset_index()
                species_data.columns = ['species_key', 'occurrences']
                species_data.to_sql('species_list', conn, if_exists='replace', index=False)

            # Table 4: 'metadata' - Export information
            metadata_df = pd.DataFrame([metadata])
            metadata_df.to_sql('metadata', conn, if_exists='replace', index=False)

            # Table 5: 'filter_settings' - Complete filter configuration for reproducibility
            filter_df = pd.DataFrame([filter_settings])
            filter_df.to_sql('filter_settings', conn, if_exists='replace', index=False)

            conn.commit()
        return True

    except PermissionError as e:
        st.error(f"❌ Permission denied: Cannot write to {file_path.parent}")
        return False
    except sqlite3.Error as e:
        st.error(f"❌ SQLite error: {str(e)}")
        return False
    except Exception as e:
        st.error(f"❌ Unexpected error in SQLite export: {str(e)}")
        return False

def export_to_parquet(df: pd.DataFrame, file_path: Path, metadata: dict) -> bool:
    """Export dataframe to Parquet file with metadata"""
    import json

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(file_path, compression=PARQUET_COMPRESSION, index=False)

        metadata_path = file_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        return True
    except PermissionError as e:
        st.error(f"❌ Permission denied: Cannot write to {file_path.parent}")
        return False
    except OSError as e:
        st.error(f"❌ File system error: {str(e)}")
        return False
    except Exception as e:
        st.error(f"❌ Unexpected error in Parquet export: {str(e)}")
        return False

def export_to_csv(df: pd.DataFrame, file_path: Path, metadata: dict) -> bool:
    """Export dataframe to CSV with metadata"""
    import json

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)

        metadata_path = file_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        return True
    except PermissionError as e:
        st.error(f"❌ Permission denied: Cannot write to {file_path.parent}")
        return False
    except OSError as e:
        st.error(f"❌ File system error: {str(e)}")
        return False
    except Exception as e:
        st.error(f"❌ Unexpected error in CSV export: {str(e)}")
        return False

# CHECK DATA AVAILABILITY
###################################################################################

if 'df_vegetation' not in st.session_state or 'df_taxa' not in st.session_state:
    st.error("⚠️ Required data not found. Please complete Step 1: Import Data first.")
    if st.button("← Go to Import Data"):
        st.switch_page("pages/01_data_import.py")
    st.stop()

# Validate that session state data is not None
if st.session_state.get('df_vegetation') is None or st.session_state.get('df_taxa') is None:
    st.error("⚠️ Data loaded but is empty. Please reload data in Import Data page.")
    st.stop()

# MAIN INTERFACE
###################################################################################

# Create tabs for workflow
tab1, tab2 = st.tabs([
    "🎯 Filter Data", 
    "🔧 Refine & Export"
])

# TAB 1: FILTER DATA
###################################################################################
with tab1:
    st.markdown("### Apply filters to your dataset")
    st.markdown("*Select species and plots based on taxonomy, habitat, time, and geography*")
    
    # Initialize data
    df_taxa_all = rename_to_canonical(st.session_state.df_taxa, 'columns_taxa', TAXA_ROLES)
    df_vegetation = rename_to_canonical(st.session_state.df_vegetation.copy(), 'columns_vegetation', VEGETATION_ROLES)

    # Check that the required columns were mapped in Data Import
    missing_required = [role for role in VEGETATION_ROLES['required'] if role not in df_vegetation.columns]
    if missing_required:
        st.error(f"⚠️ Missing required column mapping: {', '.join(missing_required)}. "
                 "Please map these columns on the Data Import page first.")
        if st.button("← Go to Import Data", key="missing_mapping_goto_import"):
            st.switch_page("pages/01_data_import.py")
        st.stop()

    # Ensure proper date handling
    df_vegetation = ensure_datetime_compatibility(df_vegetation, 'date')
    df_vegetation = ensure_datetime_compatibility(df_vegetation, 'senesteRegistrering')
    
    # Create two main columns
    col1, col2 = st.columns([1, 1])
    
    # LEFT COLUMN: SPECIES FILTERS
    ###################################################################################
    with col1:
        st.markdown("#### 📋 Species filters")
        
        # Taxonomic category
        st.markdown("##### Taxonomic rank")
        if 'taxonomi' in df_taxa_all.columns:
            available_categories = sorted(df_taxa_all['taxonomi'].dropna().unique())
            default_categories = ["Art"] if "Art" in available_categories else available_categories[:1]
        else:
            available_categories = []
            default_categories = []
            st.warning("⚠️ 'taxonomi' column not found in taxonomy data")
        
        selected_categories = st.multiselect(
            "Select taxonomic ranks:",
            options=available_categories,
            default=default_categories,
            help="Filter species by taxonomic category (e.g., Art = species level)"
        )
        
        # Major taxonomic group
        st.markdown("##### Major taxonomic group")
        if 'raekke' in df_taxa_all.columns:
            available_phyla = sorted(df_taxa_all['raekke'].dropna().unique())
            default_phyla = ["Tracheophyta"] if "Tracheophyta" in available_phyla else available_phyla[:1]
        else:
            available_phyla = []
            default_phyla = []
            st.warning("⚠️ 'raekke' column not found in taxonomy data")
        
        selected_phyla = st.multiselect(
            "Select major groups:",
            options=available_phyla,
            default=default_phyla,
            help="Filter by major taxonomic divisions"
        )
        
        # Species origin
        st.markdown("##### Species origin")
        has_origin_columns = ('isHjemmehoerende' in df_taxa_all.columns) and ('isInvasiv' in df_taxa_all.columns)
        
        if has_origin_columns:
            origin_filter = st.selectbox(
                "Filter by origin:",
                options=[
                    "All species (no filter)",
                    "Native species only",
                    "Non-native species only",
                    "Invasive species only",
                    "Native + Non-native + Invasive"
                ],
                help="Filter species based on native/invasive status"
            )
        else:
            st.warning("⚠️ Origin columns not found in taxonomy data")
            origin_filter = "All species (no filter)"
    
    # RIGHT COLUMN: PLOT FILTERS
    ###################################################################################
    with col2:
        st.markdown("#### 🗺️ Plot filters")
        
        # Habitat type selection
        st.markdown("##### Habitat types")
        
        if 'major_type' in df_vegetation.columns and 'habitat_type' in df_vegetation.columns:
            # Step 1: Select major habitat categories
            unique_major_types = sorted(df_vegetation['major_type'].dropna().unique())
            
            select_all_major = st.checkbox("Select all major habitat types", value=True, key="select_all_major")
            
            selected_major_types = st.multiselect(
                "Major habitat categories:",
                options=unique_major_types,
                default=unique_major_types if select_all_major else [],
                help="Select broad habitat categories"
            )
            
            # Step 2: Fine-grained selection of specific habitat types
            if selected_major_types:
                all_possible_habitats = df_vegetation[
                    df_vegetation['major_type'].isin(selected_major_types)
                ]['habitat_type'].unique()
                
                st.markdown("**Step 2: Fine-tune specific habitat types** (optional)")
                
                with st.expander("🔧 Select/deselect specific habitat IDs", expanded=False):
                    st.caption("Uncheck habitat types you want to exclude")
                    
                    # Group by major type for organized display
                    for major_type in selected_major_types:
                        habitat_ids = sorted(df_vegetation[df_vegetation['major_type'] == major_type]['habitat_type'].unique())
                        plot_counts = df_vegetation[df_vegetation['major_type'] == major_type].groupby('habitat_type')['plot_id'].nunique()
                        
                        st.markdown(f"**{major_type}**")
                        
                        # Create checkboxes for each habitat
                        cols = st.columns(3)
                        for idx, hab_id in enumerate(habitat_ids):
                            count = plot_counts.get(hab_id, 0)
                            with cols[idx % 3]:
                                # Use session state to track selections
                                key = f"habitat_{hab_id}"
                                if key not in st.session_state:
                                    st.session_state[key] = True
                                
                                st.checkbox(
                                    f"{hab_id} ({count:,} plots)",
                                    value=st.session_state[key],
                                    key=key
                                )
                
                # Collect selected habitats from checkboxes
                selected_habitats = [
                    hab_id for hab_id in all_possible_habitats 
                    if st.session_state.get(f"habitat_{hab_id}", True)
                ]
                
                st.info(f"📊 {len(selected_habitats)} habitat types selected from {len(all_possible_habitats)} available")
            else:
                selected_habitats = []
        elif 'habitat_type' in df_vegetation.columns:
            # Simple single-tier selection when there's no major_type grouping
            unique_habitats = sorted(df_vegetation['habitat_type'].dropna().unique())
            select_all_habitats = st.checkbox("Select all habitat types", value=True, key="select_all_habitats")
            selected_habitats = st.multiselect(
                "Habitat types:",
                options=unique_habitats,
                default=unique_habitats if select_all_habitats else [],
                help="Select habitat types to include"
            )
            st.info(f"📊 {len(selected_habitats)} habitat types selected from {len(unique_habitats)} available")
        else:
            st.warning("⚠️ Habitat type columns not found")
            selected_habitats = []
        
        # Sampling method
        st.markdown("##### Sampling methods")
        
        # Define the actual sampling method columns
        method_columns = ['isPinpoint', 'isProvefelt', 'is5m', 'is15m']
        available_methods = [col for col in method_columns if col in df_vegetation.columns and df_vegetation[col].sum() > 0]
        
        if available_methods:
            # Add "select all" checkbox
            select_all_methods = st.checkbox("Select all sampling methods", value=True, key="select_all_methods")
            
            # Create user-friendly labels
            method_labels = {
                'isPinpoint': 'Pinpoint method',
                'isProvefelt': 'Provefelt method',
                'is5m': '5m method',
                'is15m': '15m method'
            }
            
            selected_methods = st.multiselect(
                "Select sampling methods to include:",
                options=available_methods,
                default=available_methods if select_all_methods else [],
                format_func=lambda x: method_labels.get(x, x),
                help="Filter plots by sampling methodology (plots with value=1 for selected methods)"
            )
            
            # Show how many plots use each method
            if len(available_methods) > 0:
                with st.expander("📊 View sampling method distribution", expanded=False):
                    for method in available_methods:
                        plot_count = df_vegetation[df_vegetation[method] == 1]['plot_id'].nunique() if 'plot_id' in df_vegetation.columns else 0
                        st.caption(f"{method_labels.get(method, method)}: {plot_count:,} plots")
        else:
            st.warning("⚠️ No sampling method columns found (isPinpoint, isProvefelt, is5m, is15m)")
            selected_methods = []
        
        # Year range
        st.markdown("##### Year range")
        if 'year' in df_vegetation.columns:
            valid_years = df_vegetation['year'].dropna()
            if len(valid_years) > 0:
                data_min_year = int(valid_years.min())
                data_max_year = int(valid_years.max())
            else:
                data_min_year = 2000
                data_max_year = date.today().year

            if data_min_year == data_max_year:
                st.caption(f"All data is from {data_min_year}")
                year_range = (data_min_year, data_max_year)
            else:
                year_range = st.slider(
                    "Select year range:",
                    min_value=data_min_year,
                    max_value=data_max_year,
                    value=(data_min_year, data_max_year),
                    help="Filter data by sampling year",
                    key="year_range_filter"
                )
                st.caption(f"Data available from {data_min_year} to {data_max_year}")
        else:
            st.warning("⚠️ 'year' column not found")
            year_range = None
    
    # SPECIAL FILTERS (Full Width)
    ###################################################################################
    st.markdown("---")
    st.markdown("#### 🎯 Special filters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🚨 Invasive species plots")
        filter_invasive_plots = st.checkbox(
            "Include only plots containing invasive species",
            value=False,
            help="Filter to plots that contain at least one invasive species (keeps all species from those plots)"
        )
        
        if filter_invasive_plots and has_origin_columns:
            # Preview invasive plot count
            invasive_species_ids = df_taxa_all[df_taxa_all['isInvasiv'] == 1]['species_key'].unique()
            if len(invasive_species_ids) > 0 and 'species_key' in df_vegetation.columns:
                plots_with_invasives = df_vegetation[df_vegetation['species_key'].isin(invasive_species_ids)]['plot_id'].unique()
                st.caption(f"✅ {len(plots_with_invasives):,} plots contain invasive species ({len(plots_with_invasives)/df_vegetation['plot_id'].nunique()*100:.1f}% of total)")
            else:
                st.caption("⚠️ No invasive species found in dataset")
    
    with col2:
        st.markdown("##### 🎲 Random sampling")
        use_random_sample = st.checkbox("Apply random plot sampling", value=False)
        
        if use_random_sample:
            col_a, col_b = st.columns(2)
            with col_a:
                sample_percentage = st.slider("Sample %:", 1, 100, 50, help="Percentage of plots to randomly sample")
            with col_b:
                random_seed = st.number_input("Random seed:", value=42, min_value=0, help="For reproducibility")
        else:
            sample_percentage = 100
            random_seed = None
    
    # APPLY FILTERS BUTTON
    ###################################################################################
    st.markdown("---")
    if st.button("🔥 Apply All Filters", type="primary", use_container_width=True):
        with st.spinner("Applying filters..."):
            try:
                progress_bar = st.progress(0)
                
                # Step 1: Apply taxonomic filters (20%)
                progress_bar.progress(0.2)
                df_taxa_filtered = df_taxa_all.copy()
                
                if selected_categories and 'taxonomi' in df_taxa_filtered.columns:
                    df_taxa_filtered = df_taxa_filtered[df_taxa_filtered['taxonomi'].isin(selected_categories)]
                
                if selected_phyla and 'raekke' in df_taxa_filtered.columns:
                    df_taxa_filtered = df_taxa_filtered[df_taxa_filtered['raekke'].isin(selected_phyla)]
                
                # Apply origin filter
                if has_origin_columns and origin_filter != "All species (no filter)":
                    if origin_filter == "Native species only":
                        df_taxa_filtered = df_taxa_filtered[df_taxa_filtered['isHjemmehoerende'] == 1]
                    elif origin_filter == "Non-native species only":
                        df_taxa_filtered = df_taxa_filtered[df_taxa_filtered['isHjemmehoerende'] == 0]
                    elif origin_filter == "Invasive species only":
                        df_taxa_filtered = df_taxa_filtered[df_taxa_filtered['isInvasiv'] == 1]
                    elif origin_filter == "Native + Non-native + Invasive":
                        df_taxa_filtered = df_taxa_filtered[
                            (df_taxa_filtered['isHjemmehoerende'].isin([0, 1])) | 
                            (df_taxa_filtered['isInvasiv'].isin([0, 1]))
                        ]
                
                # Step 2: Apply plot filters (40%)
                progress_bar.progress(0.4)
                df_vegetation_filtered = df_vegetation.copy()
                
                if selected_habitats and 'habitat_type' in df_vegetation_filtered.columns:
                    df_vegetation_filtered = df_vegetation_filtered[df_vegetation_filtered['habitat_type'].isin(selected_habitats)]
                
                if selected_methods:
                    # For binary columns (value 0 or 1), keep plots where at least one selected method = 1
                    mask = df_vegetation_filtered[selected_methods].eq(1).any(axis=1)
                    df_vegetation_filtered = df_vegetation_filtered[mask]
                
                # Apply year filter
                if year_range and 'year' in df_vegetation_filtered.columns and len(year_range) == 2:
                    start_year, end_year = year_range
                    df_vegetation_filtered = df_vegetation_filtered[
                        (df_vegetation_filtered['year'] >= start_year) &
                        (df_vegetation_filtered['year'] <= end_year)
                    ]
                
                # Step 3: Apply invasive species plot filter (60%)
                progress_bar.progress(0.6)
                if filter_invasive_plots and has_origin_columns:
                    invasive_species_ids = df_taxa_all[df_taxa_all['isInvasiv'] == 1]['species_key'].unique()
                    
                    if len(invasive_species_ids) > 0 and 'species_key' in df_vegetation_filtered.columns:
                        plots_with_invasives = df_vegetation_filtered[
                            df_vegetation_filtered['species_key'].isin(invasive_species_ids)
                        ]['plot_id'].unique()
                        
                        df_vegetation_filtered = df_vegetation_filtered[
                            df_vegetation_filtered['plot_id'].isin(plots_with_invasives)
                        ]
                        
                        st.info(f"🚨 Invasive filter applied: {len(plots_with_invasives):,} plots contain invasive species")
                
                # Step 4: Apply random sampling (80%)
                progress_bar.progress(0.8)
                if use_random_sample and sample_percentage < 100:
                    df_vegetation_filtered = apply_random_sample(df_vegetation_filtered, sample_percentage, random_seed)
                    st.info(f"🎲 Random sampling applied: {sample_percentage}% of plots (seed: {random_seed})")
                
                # Step 5: Merge datasets (100%)
                progress_bar.progress(1.0)
                # species_key is required in both files, so it's always the join key --
                # it's on the user to have mapped columns whose values actually match
                # between the vegetation and taxa files (see column_config.py)
                if 'species_key' in df_vegetation_filtered.columns and 'species_key' in df_taxa_filtered.columns:
                    # Merge with FILTERED taxa (not all taxa) - this preserves taxonomic filters
                    taxa_merge_cols = ['species_key']
                    if 'taxonomi' in df_taxa_filtered.columns:
                        taxa_merge_cols.append('taxonomi')
                    df_merged = pd.merge(
                        df_vegetation_filtered,
                        df_taxa_filtered[taxa_merge_cols],
                        on='species_key',
                        how='inner'
                    )
                else:
                    st.error("Cannot merge: species_key column missing from vegetation or taxa data")
                    df_merged = pd.DataFrame()
                
                # Store results in session state
                st.session_state.df_taxa_filtered = df_taxa_filtered
                st.session_state.df_vegetation_filtered = df_vegetation_filtered
                st.session_state.df_merged = df_merged
                
                # Store filter settings for export
                st.session_state.filter_settings = {
                    'selected_categories': str(selected_categories),
                    'selected_phyla': str(selected_phyla),
                    'origin_filter': origin_filter,
                    'selected_habitats': str(selected_habitats),
                    'selected_methods': str(selected_methods),
                    'year_range': str(year_range),
                    'filter_invasive_plots': filter_invasive_plots,
                    'use_random_sample': use_random_sample,
                    'sample_percentage': sample_percentage if use_random_sample else None,
                    'random_seed': random_seed if use_random_sample else None,
                    'filter_timestamp': datetime.now().isoformat()
                }
                
                progress_bar.empty()
                
                # Display results
                st.success("✅ Filters applied successfully!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Plots Selected",
                        f"{df_merged['plot_id'].nunique() if 'plot_id' in df_merged.columns else 0:,}",
                        f"{df_merged['plot_id'].nunique()/df_vegetation['plot_id'].nunique()*100:.1f}% of total"
                    )
                
                with col2:
                    st.metric(
                        "Species Selected",
                        f"{df_merged['species_key'].nunique() if 'species_key' in df_merged.columns else 0:,}",
                        f"{df_merged['species_key'].nunique()/len(df_taxa_all)*100:.1f}% of total"
                    )
                
                with col3:
                    st.metric(
                        "Final Records",
                        f"{len(df_merged):,}",
                        f"{len(df_merged)/(len(df_vegetation)*df_taxa_all.shape[0])*100:.3f}% of all possible"
                    )
                
                # Preview filtered data
                with st.expander("🔍 Preview Filtered Data", expanded=False):
                    preview_tab1, preview_tab2 = st.tabs(["Merged Dataset", "Summary Statistics"])
                    
                    with preview_tab1:
                        st.dataframe(df_merged.head(1000), use_container_width=True, height=400)
                        st.caption(f"Showing first 1,000 of {len(df_merged):,} total records")
                    
                    with preview_tab2:
                        summary = summarize_data(df_merged)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("📍 Unique Plots", f"{summary['unique_plots']:,}")
                        col2.metric("🌲 Habitat Types", f"{summary['unique_nature_types']:,}")
                        col3.metric("🌿 Species", f"{summary['unique_species']:,}")
            
            except Exception as e:
                st.error(f"❌ Error applying filters: {str(e)}")
                import traceback
                st.error(f"Details: {traceback.format_exc()}")

# TAB 2: REFINE SELECTION & OVERVIEW
###################################################################################
with tab2:
    st.markdown("### Data overview & refinement")
    st.markdown("*Review statistics and apply additional refinements*")
    
    if 'df_merged' not in st.session_state:
        st.info("👈 Please apply filters in the 'Filter Data' tab first")
    else:
        df_current = st.session_state.df_merged
        
        # Summary metrics
        st.markdown("#### 📊 Current selection summary")
        
        col1, col2, col3, col4 = st.columns(4)
        summary = summarize_data(df_current)
        
        with col1:
            st.metric("📍 Unique Plots", f"{summary['unique_plots']:,}")
        with col2:
            st.metric("🌲 Habitat Types", f"{summary['unique_nature_types']:,}")
        with col3:
            st.metric("🌿 Species", f"{summary['unique_species']:,}")
        with col4:
            st.metric("📄 Total Records", f"{summary['total_records']:,}")
        
        # Additional Refinements section - MOVED UP
        st.markdown("---")
        st.markdown("#### 🔧 Additional refinements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🌿 Minimum species occurrence**")
            st.caption("Remove species below occurrence threshold")
            
            # Need species counts for preview
            if 'species_key' in df_current.columns:
                species_counts = df_current['species_key'].value_counts()
            else:
                species_counts = pd.Series()
            
            min_occurrences = st.slider(
                "Minimum occurrences:",
                min_value=1,
                max_value=50,
                value=5,
                key="refine_min_occurrences",
                help="Exclude species occurring fewer times"
            )
            
            # Preview effect
            if len(species_counts) > 0:
                species_after = species_counts[species_counts >= min_occurrences]
                st.metric("Species after filter", f"{len(species_after):,}", 
                         delta=f"-{len(species_counts) - len(species_after):,}")
        
        with col2:
            st.markdown("**📊 Stratified sampling**")
            st.caption("Balance sample across habitats")
            
            use_stratified = st.checkbox("Apply stratified sampling", value=False)
            
            if use_stratified:
                max_plots_habitat = st.slider(
                    "Max plots per habitat:",
                    min_value=50,
                    max_value=1000,
                    value=300,
                    step=50
                )
                
                stratified_seed = st.number_input(
                    "Random seed:",
                    value=42,
                    min_value=0
                )
        
        # Apply refinements button
        st.markdown("---")
        if st.button("✨ Apply Refinements", type="primary", use_container_width=True):
            with st.spinner("Applying refinements..."):
                df_refined = df_current.copy()
                
                # Apply occurrence filter
                if 'species_key' in df_refined.columns and min_occurrences > 1:
                    species_counts_refine = df_refined['species_key'].value_counts()
                    species_to_keep = species_counts_refine[species_counts_refine >= min_occurrences].index
                    df_refined = df_refined[df_refined['species_key'].isin(species_to_keep)]
                    st.info(f"🌿 Removed {len(species_counts_refine) - len(species_to_keep)} rare species")
                
                # Apply stratified sampling
                if use_stratified and 'habitat_type' in df_refined.columns:
                    df_refined = create_stratified_sample(
                        df_refined, 
                        max_plots_habitat, 
                        min_occurrences,
                        stratified_seed
                    )
                    st.info(f"📊 Stratified sampling applied (seed: {stratified_seed})")
                
                # Update session state
                st.session_state.df_merged = df_refined
                
                # Update filter settings
                if 'filter_settings' in st.session_state:
                    st.session_state.filter_settings.update({
                        'min_occurrences': min_occurrences,
                        'use_stratified': use_stratified,
                        'max_plots_habitat': max_plots_habitat if use_stratified else None,
                        'stratified_seed': stratified_seed if use_stratified else None
                    })
                
                st.success("✅ Refinements applied!")
                
                # Show new summary
                col1, col2, col3 = st.columns(3)
                col1.metric("📍 Final Plots", f"{df_refined['plot_id'].nunique() if 'plot_id' in df_refined.columns else 0:,}")
                col2.metric("🌿 Final Species", f"{df_refined['species_key'].nunique() if 'species_key' in df_refined.columns else 0:,}")
                col3.metric("📄 Final Records", f"{len(df_refined):,}")
                
                st.rerun()
        
        # Detailed statistics - NOW APPEARS AFTER REFINEMENTS
        st.markdown("---")
        st.markdown("#### 📊 Updated dataset statistics")

        st.markdown("##### 🌲 Top habitat types")
        if 'habitat_type' in df_current.columns and 'plot_id' in df_current.columns:
            habitat_stats = df_current.groupby('habitat_type').agg({
                'plot_id': 'nunique',
                'species_key': 'nunique'
            }).reset_index()
            habitat_stats.columns = ['Habitat ID', 'Plot Count', 'Species Count']
            habitat_stats = habitat_stats.sort_values('Plot Count', ascending=False).head(50)

            st.dataframe(
                habitat_stats,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("Required columns not found")

        # Species frequency analysis with visualization
        st.markdown("---")
        st.markdown("##### 🌿 Species occurrence distribution")
        
        if 'species_key' in df_current.columns:
            # Import matplotlib locally (only used in this section)
            import matplotlib.pyplot as plt

            species_counts = df_current['species_key'].value_counts()

            col1, col2 = st.columns([2, 1])

            with col1:
                # Create frequency categories
                frequency_categories = {
                    'Very Rare (1-2)': len(species_counts[(species_counts >= 1) & (species_counts <= 2)]),
                    'Rare (3-5)': len(species_counts[(species_counts >= 3) & (species_counts <= 5)]),
                    'Uncommon (6-10)': len(species_counts[(species_counts >= 6) & (species_counts <= 10)]),
                    'Frequent (11-25)': len(species_counts[(species_counts >= 11) & (species_counts <= 25)]),
                    'Common (26-50)': len(species_counts[(species_counts >= 26) & (species_counts <= 50)]),
                    'Very Common (51-100)': len(species_counts[(species_counts >= 51) & (species_counts <= 100)]),
                    'Dominant (>100)': len(species_counts[species_counts > 100])
                }
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(12, 6))
                
                categories = list(frequency_categories.keys())
                values = list(frequency_categories.values())
                colors = ['#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60', '#3498db']
                
                bars = ax.bar(range(len(categories)), values, color=colors, alpha=0.8)
                
                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, values)):
                    if value > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                               str(value), ha='center', va='bottom', fontweight='bold')
                
                ax.set_xlabel('Frequency Category', fontsize=10)
                ax.set_ylabel('Number of Species', fontsize=10)
                ax.set_title('Species Distribution by Occurrence Frequency', fontsize=12, fontweight='bold')
                ax.set_xticks(range(len(categories)))
                ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.markdown("**📊 Key statistics**")
                st.metric("Total species", f"{len(species_counts):,}")
                st.metric("Median occurrences", f"{species_counts.median():.0f}")
                st.metric("Mean occurrences", f"{species_counts.mean():.1f}")
                st.metric("Max occurrences", f"{species_counts.max():,}")
                
                # Most common species
                st.markdown("**🏆 Most Common:**")
                top_species = species_counts.head(10)
                for i, (species, count) in enumerate(top_species.items(), 1):
                    st.caption(f"{i}. {species}: {count:,}")
        
        # EXPORT SECTION
        ###################################################################################
        st.markdown("---")
        st.markdown("### 💾 Export your dataset")
        st.markdown("*Choose export strategy and save for network analysis*")
        df_export_base = st.session_state.df_merged
        
        # Current selection summary
        st.markdown("#### 📊 Current selection summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📍 Plots", f"{df_export_base['plot_id'].nunique() if 'plot_id' in df_export_base.columns else 0:,}")
        with col2:
            st.metric("🌿 Species", f"{df_export_base['species_key'].nunique() if 'species_key' in df_export_base.columns else 0:,}")
        with col3:
            st.metric("🌲 Habitats", f"{df_export_base['habitat_type'].nunique() if 'habitat_type' in df_export_base.columns else 0:,}")
        with col4:
            st.metric("📄 Records", f"{len(df_export_base):,}")
        
        st.markdown("---")
        
        # Export type selection
        st.markdown("#### 🎯 Choose export strategy")
        
        export_type = st.radio(
            "Select export method:",
            options=[
                "📦 Basic Export (use refined data as-is)",
                "📍 Geographic Subset (distance-based filtering)"
            ],
            help="Choose how to process data before export"
        )
        
        st.markdown("---")
        
        # Common export settings
        st.markdown("#### ⚙️ Export settings")
        
        col1, col2 = st.columns(2)
        with col1:
            export_path = st.text_input(
                "Export directory:",
                value=st.session_state.get('queries_path', './exports'),
                help="Directory for saved file"
            )
            
            current_date = date.today()
            if "Basic Export" in export_type:
                default_name = f"{current_date.strftime('%b%d_%Y')}_basic"
            else:  # Geographic
                default_name = f"{current_date.strftime('%b%d_%Y')}_geographic"
            
            file_name = st.text_input(
                "File name:",
                value=default_name,
                help="Name without extension"
            )
        
        with col2:
            export_format = st.selectbox(
                "Export format:",
                options=["SQLite (.db)", "Parquet (.parquet)", "CSV (.csv)"],
                help="SQLite recommended for network analysis"
            )
            
            comments = st.text_area(
                "Comments:",
                value="Filtered vegetation dataset",
                height=100,
                help="Notes about this export"
            )
        
        # Type-specific settings - only for Geographic subset
        if "Geographic" in export_type:
            st.markdown("---")
            st.markdown("#### 🔧 Processing options")
            st.markdown("**Filter by distance from center point:**")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                center_x = st.number_input(
                    "Center X (UTM):",
                    value=550000,
                    step=10000
                )
                center_y = st.number_input(
                    "Center Y (UTM):",
                    value=6230000,
                    step=10000
                )
                max_distance = st.slider(
                    "Max distance (km):",
                    min_value=1.0,
                    max_value=100.0,
                    value=10.0,
                    step=0.5
                )
            
            with col2:
                st.markdown("**📍 Geographic Preview:**")
                
                if 'x' in df_export_base.columns and 'y' in df_export_base.columns:
                    try:
                        unique_plots = df_export_base.drop_duplicates('plot_id')[['plot_id', 'x', 'y']]
                        unique_plots['distance'] = unique_plots.apply(
                            lambda row: calculate_distance(center_x, center_y, row['x'], row['y']),
                            axis=1
                        )
                        
                        # Simple map
                        fig, ax = plt.subplots(figsize=(8, 8))
                        
                        ax.scatter(unique_plots['x'], unique_plots['y'], 
                                  c='lightgray', alpha=0.5, s=8, label='All plots')
                        
                        filtered_plots = unique_plots[unique_plots['distance'] <= max_distance]
                        if len(filtered_plots) > 0:
                            ax.scatter(filtered_plots['x'], filtered_plots['y'], 
                                      c='red', s=15, alpha=0.8, label=f'Within {max_distance} km')
                        
                        ax.scatter(center_x, center_y, c='blue', s=200, marker='*', 
                                  label='Center', edgecolors='white', linewidth=2)
                        
                        circle = plt.Circle((center_x, center_y), max_distance * 1000, 
                                          fill=False, edgecolor='blue', linestyle='--', linewidth=2)
                        ax.add_patch(circle)
                        
                        ax.set_xlabel('UTM X')
                        ax.set_ylabel('UTM Y')
                        ax.set_title(f'Filter: {max_distance} km radius')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        ax.set_aspect('equal')
                        
                        st.pyplot(fig)
                        plt.close()
                        
                        st.info(f"📍 {len(filtered_plots):,} plots within {max_distance} km")
                    except Exception as e:
                        st.error(f"Map error: {str(e)}")
        
        # Export button
        st.markdown("---")
        
        if st.button("💾 Export Dataset", type="primary", use_container_width=True):
            if not file_name:
                st.error("❌ Please provide a file name")
            else:
                try:
                    with st.spinner("Processing and exporting..."):
                        # Process based on export type
                        if "Basic Export" in export_type:
                            # For basic export, use data as-is (refinements already applied)
                            df_final = df_export_base.copy()
                        
                        else:  # Geographic
                            if 'x' in df_export_base.columns and 'y' in df_export_base.columns:
                                unique_plots = df_export_base.drop_duplicates('plot_id')[['plot_id', 'x', 'y']]
                                unique_plots['distance'] = unique_plots.apply(
                                    lambda row: calculate_distance(center_x, center_y, row['x'], row['y']),
                                    axis=1
                                )
                                filtered_plot_ids = unique_plots[unique_plots['distance'] <= max_distance]['plot_id'].values
                                df_final = df_export_base[df_export_base['plot_id'].isin(filtered_plot_ids)]
                            else:
                                st.error("Missing UTM coordinates")
                                df_final = df_export_base
                        
                        # Create metadata
                        metadata = {
                            'export_date': datetime.now().isoformat(),
                            'export_type': export_type,
                            'export_format': export_format,
                            'comments': comments,
                            'total_records': len(df_final),
                            'unique_plots': df_final['plot_id'].nunique() if 'plot_id' in df_final.columns else 0,
                            'unique_species': df_final['species_key'].nunique() if 'species_key' in df_final.columns else 0
                        }
                        
                        # Get filter settings
                        filter_settings = st.session_state.get('filter_settings', {})
                        
                        # Add export-specific settings (only for Geographic)
                        if "Geographic" in export_type:
                            filter_settings['export_center_x'] = center_x
                            filter_settings['export_center_y'] = center_y
                            filter_settings['export_max_distance'] = max_distance
                        
                        # Export
                        export_path_obj = Path(export_path)
                        export_path_obj.mkdir(parents=True, exist_ok=True)
                        
                        if "SQLite" in export_format:
                            file_path = export_path_obj / f"{file_name}.db"
                            success = export_to_sqlite(df_final, file_path, metadata, filter_settings)
                            
                            if success:
                                st.success(f"✅ Exported to `{file_path.name}`")
                                st.info("📋 **Database tables:**\n"
                                       "- `data`: Complete dataset\n"
                                       "- `plot_id`: Plot metadata\n"
                                       "- `species_list`: Species counts\n"
                                       "- `metadata`: Export info\n"
                                       "- `filter_settings`: Full configuration")
                        
                        elif "Parquet" in export_format:
                            file_path = export_path_obj / f"{file_name}.parquet"
                            success = export_to_parquet(df_final, file_path, metadata)
                            if success:
                                st.success(f"✅ Exported to `{file_path.name}`")
                        
                        else:  # CSV
                            file_path = export_path_obj / f"{file_name}.csv"
                            success = export_to_csv(df_final, file_path, metadata)
                            if success:
                                st.success(f"✅ Exported to `{file_path.name}`")
                        
                        if success:
                            col1, col2, col3 = st.columns(3)
                            col1.metric("📍 Plots", f"{metadata['unique_plots']:,}")
                            col2.metric("🌿 Species", f"{metadata['unique_species']:,}")
                            col3.metric("📄 Records", f"{metadata['total_records']:,}")
                            
                            st.info(f"📁 Saved to: `{file_path}`")
                
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")
                    import traceback
                    st.error(f"Details: {traceback.format_exc()}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d; font-size: 0.9em;'>
    EcoNetMap - Data Filtering Module
    </div>
    """, 
    unsafe_allow_html=True
)
