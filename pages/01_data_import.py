"""
EcoNetMap - Data Import Module
===============================
This module handles the initial data import for the ecological network mapping pipeline.
It loads three key datasets: vegetation monitoring data (species occurrences), taxa information
(species taxonomy and traits including Ellenberg values), and optionally regional species pool data.
The module provides efficient caching using Parquet format for faster subsequent loads,
validates data integrity, and prepares the foundation for all downstream analyses.

Supports NOVANA data format and compatible vegetation monitoring datasets.

Part of the EcoNetMap toolkit (Data Management 1/2)
Author: Flemming Skov (fs@ecos.au.dk)
Last Updated: January 2026
"""

# Import packages for web applications
import streamlit as st

# Import packages for data manipulation and analysis
import pandas as pd
import numpy as np

# Import packages for file and system operations
from pathlib import Path
import datetime

# Import packages for type hints
from typing import Optional, Tuple

# Column-role mapping (lets users adapt the toolkit to their own column names)
from column_config import (
    ROLE_LABELS, guess_default_column, load_column_mapping, save_column_mapping,
    VEGETATION_ROLES, TAXA_ROLES, REGIONAL_POOL_ROLES
)

# Constants
CACHE_DIR_NAME = 'cache'
PREVIEW_ROWS = 20
PARQUET_COMPRESSION = 'snappy'
DATE_COLUMNS = ['dato', 'senesteRegistrering']

# Page configuration
st.set_page_config(
    page_title="Data Import - EcoNetMap",
    page_icon="📥",
    layout="wide"
)

# Custom CSS for consistent styling
st.markdown("""
<style>
    .stTextInput > label {
        font-weight: bold;
        color: #2c3e50;
    }
    .success-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 10px 0;
    }
    .info-metric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 10px 0;
    }
    div[data-testid="stExpander"] > details {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 5px;
    }
    .cache-status {
        background-color: #e8f4f8;
        border: 1px solid #b8e0ea;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and progress indicator
col1, col2 = st.columns([4, 1])
with col1:
    st.header("Data management")
    st.subheader("📥 Import data")
    st.markdown("*Import vegetation monitoring and species taxonomy data from cached or raw files*")
with col2:
    pass

st.markdown("---")

# FUNCTIONS
###################################################################################

@st.cache_data(show_spinner=False)
def load_csv_file(file_path: str, encoding: str = 'utf-8', sep: str = ',') -> Optional[pd.DataFrame]:
    """Load CSV file with caching and flexible encoding/separator options"""
    try:
        return pd.read_csv(file_path, encoding=encoding, sep=sep, low_memory=False)
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}")
        return None
    except PermissionError:
        st.error(f"Error: Permission denied accessing {file_path}")
        return None
    except pd.errors.ParserError as e:
        st.error(f"Error parsing CSV file: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error loading CSV: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def load_excel_sheet(file_path: str, sheet_name: str) -> Optional[pd.DataFrame]:
    """Load Excel sheet with caching"""
    try:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        st.error(f"Error: Excel file not found at {file_path}")
        return None
    except PermissionError:
        st.error(f"Error: Permission denied accessing {file_path}")
        return None
    except ValueError as e:
        st.error(f"Error: Sheet '{sheet_name}' not found in Excel file. {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error loading Excel: {str(e)}")
        return None

def get_dataframe_info(df: pd.DataFrame) -> dict:
    """Get comprehensive dataframe information"""
    memory_usage_bytes = df.memory_usage(deep=True).sum()  
    memory_usage_mb = memory_usage_bytes / (1024 * 1024)
    
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_mb': round(memory_usage_mb, 2),
        'dtypes': df.dtypes.value_counts().to_dict()
    }


def check_file_exists(file_path: Path) -> Tuple[bool, str]:
    """Check if file exists and return status message"""
    if file_path.exists():
        file_size = file_path.stat().st_size / (1024 * 1024)  # Size in MB
        return True, f"✅ File found ({file_size:.1f} MB)"
    return False, "❌ File not found"

def clean_dataframe_for_parquet(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Clean dataframe for Parquet storage (handle mixed types and dates)"""
    df_clean = df.copy()

    for col in df_clean.columns:
        # Handle object columns with mixed types
        if df_clean[col].dtype == 'object':
            # Check if it's a date column
            if 'date' in col.lower() or 'dato' in col.lower():
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                    continue
                except (ValueError, TypeError) as e:
                    st.warning(f"Could not convert date column '{col}': {str(e)}")
                    pass

            # For all other object columns, convert to string
            df_clean[col] = df_clean[col].astype(str)
            # Replace 'nan' strings with actual NaN
            df_clean[col] = df_clean[col].replace('nan', pd.NA)

        # Handle integer date columns (YYYYMMDD format like 20090929)
        elif col.lower() in ['dato', 'senesteregistrering'] and df_clean[col].dtype in ['int64', 'float64']:
            try:
                # Convert integer format (e.g., 20090929) to datetime
                df_clean[col] = pd.to_datetime(df_clean[col].astype(str), format='%Y%m%d', errors='coerce')
            except (ValueError, TypeError) as e:
                st.warning(f"Could not convert integer date column '{col}': {str(e)}")
                pass

    return df_clean

def save_to_parquet(df: pd.DataFrame, file_path: Path, name: str) -> bool:
    """Save dataframe to Parquet format with error handling"""
    try:
        df_clean = clean_dataframe_for_parquet(df, name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_parquet(file_path, compression=PARQUET_COMPRESSION, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving {name} to Parquet: {str(e)}")
        return False

def load_from_parquet(file_path: Path) -> Optional[pd.DataFrame]:
    """Load dataframe from Parquet format"""
    try:
        if file_path.exists():
            return pd.read_parquet(file_path)
        return None
    except Exception as e:
        st.error(f"Error loading from Parquet: {str(e)}")
        return None

def render_column_mapping_ui(df: pd.DataFrame, section: str, roles: dict, key_prefix: str) -> None:
    """Let the user map their file's columns onto canonical roles and save to settings.txt"""
    st.markdown("#### 🔗 Map your columns")
    st.caption("Tell EcoNetMap which column in your file corresponds to each concept. "
               "Optional roles can be left as *(not present)* if your data doesn't have them.")

    existing_mapping = load_column_mapping(section)
    columns = df.columns.tolist()
    selections = {}

    required_roles = roles.get('required', [])
    optional_roles = roles.get('optional', [])

    for role in required_roles:
        label = f"{ROLE_LABELS.get(role, role)} (required)"
        default_col = existing_mapping.get(role) or guess_default_column(role, columns)
        default_index = columns.index(default_col) if default_col in columns else 0
        selections[role] = st.selectbox(label, options=columns, index=default_index,
                                        key=f"{key_prefix}_{role}")

    if optional_roles:
        with st.expander("Optional columns", expanded=False):
            options = ["(not present)"] + columns
            for role in optional_roles:
                label = ROLE_LABELS.get(role, role)
                default_col = existing_mapping.get(role) or guess_default_column(role, columns)
                default_index = options.index(default_col) if default_col in options else 0
                choice = st.selectbox(label, options=options, index=default_index,
                                      key=f"{key_prefix}_{role}")
                selections[role] = None if choice == "(not present)" else choice

    # Warn if two roles point at the same source column (each role still gets its
    # own copy on save, but this is almost always a sign one selection is wrong)
    by_source: dict = {}
    for role, column in selections.items():
        if column:
            by_source.setdefault(column, []).append(role)
    collisions = {col: roles_ for col, roles_ in by_source.items() if len(roles_) > 1}
    if collisions:
        for col, roles_ in collisions.items():
            st.warning(f"⚠️ Column **{col}** is mapped to multiple roles: "
                       f"{', '.join(ROLE_LABELS.get(r, r) for r in roles_)}. "
                       "Double-check this is intentional before saving.")

    if st.button("💾 Save column mapping", key=f"{key_prefix}_save"):
        save_column_mapping(section, selections)
        st.success("✅ Column mapping saved to settings.txt")

# MAIN UI
###################################################################################

# Initialize paths from session state with validation
if 'data_path' not in st.session_state or 'external_data_path' not in st.session_state:
    st.error("⚠️ Project paths not initialized. Please run the home page first to set up your project.")
    st.info("Navigate to the home page to configure project settings.")
    st.stop()

data_path = Path(st.session_state['data_path'])
external_data_path = Path(st.session_state['external_data_path'])

# Validate paths exist
if not data_path.exists():
    st.warning(f"⚠️ Data path does not exist: {data_path}")
    st.info("The path will be created when you load data.")
if not external_data_path.exists():
    st.warning(f"⚠️ External data path does not exist: {external_data_path}")
    st.info("The path will be created when you import custom data.")

# Cache directory for Parquet files
cache_dir = data_path / CACHE_DIR_NAME
cache_dir.mkdir(parents=True, exist_ok=True)

# Create tabs for different import types
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Vegetation Monitoring Data",
    "🌿 Species Taxonomy & Traits", 
    "🗺️ Regional Species Pool",
    "📤 Custom Data Import"
])

# TAB 1: Vegetation Monitoring Data (formerly NOVANA data)
###################################################################################
with tab1:
    st.markdown("### 📊 Import vegetation monitoring data")
    st.markdown("*Species occurrence data from plot-based monitoring programs*")
    
    # Check for cached data
    cache_file = cache_dir / 'monitoring_data.parquet'
    cache_exists, cache_msg = check_file_exists(cache_file)
    
    # Auto-load from cache if exists and not already loaded
    if cache_exists and 'df_vegetation' not in st.session_state:
        with st.spinner("Auto-loading cached vegetation monitoring data..."):
            df = load_from_parquet(cache_file)
            if df is not None:
                st.session_state['df_vegetation'] = df
                st.success(f"✅ Auto-loaded {len(df):,} records from cache")
    
    if cache_exists:
        st.markdown('<div class="cache-status">', unsafe_allow_html=True)
        st.markdown("**⚡ Cached data available**")
        st.markdown(cache_msg)
        if 'df_vegetation' in st.session_state:
            st.markdown("✅ *Currently loaded in session*")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reload from Cache", type="secondary", use_container_width=True):
                with st.spinner("Reloading cached vegetation monitoring data..."):
                    df = load_from_parquet(cache_file)
                    if df is not None:
                        st.session_state['df_vegetation'] = df
                        st.success(f"✅ Reloaded {len(df):,} records from cache")
                        st.rerun()
        
        with col2:
            if st.button("🔄 Reload from Source", use_container_width=True):
                cache_file.unlink()
                if 'df_vegetation' in st.session_state:
                    del st.session_state['df_vegetation']
                st.rerun()
    
    else:
        st.info("📁 No cached data found. Please load from source file.")
    
    # Load from source file
    with st.expander("📂 Load from Source File", expanded=not cache_exists):
        st.markdown("**Select vegetation monitoring data file**")
        st.caption("Expected format: CSV with columns for plot ID, species names, coordinates, and metadata")
        
        source_file = st.text_input(
            "File path:",
            value=str(data_path / "monitoring_data.csv"),
            help="Path to vegetation monitoring CSV file"
        )
        
        source_path = Path(source_file)
        exists, status = check_file_exists(source_path)
        st.markdown(status)
        
        # File format options
        col1, col2 = st.columns(2)
        with col1:
            separator = st.selectbox("Separator:", [',', '\t', ';'], index=0)
        with col2:
            encoding = st.selectbox("Encoding:", ['utf-8', 'latin1', 'iso-8859-1'], index=0)
        
            if st.button("📥 Load from Source", disabled=not exists, type="primary", use_container_width=True):
                with st.spinner("Loading vegetation monitoring data..."):
                    df = load_csv_file(str(source_path), encoding=encoding, sep=separator)
                    
                    if df is not None:
                        # CONVERT DATE COLUMNS IMMEDIATELY AFTER LOADING
                        date_columns = ['dato', 'senesteRegistrering']
                        for date_col in date_columns:
                            if date_col in df.columns:
                                try:
                                    # Check if dates are stored as integers (YYYYMMDD format)
                                    if df[date_col].dtype in ['int64', 'float64']:
                                        df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d', errors='coerce')
                                    else:
                                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                                    st.info(f"✅ Converted {date_col} to datetime format")
                                except Exception as e:
                                    st.warning(f"⚠️ Could not convert {date_col}: {str(e)}")

                    # Save to session state
                    st.session_state['df_vegetation'] = df
                    
                    # Save to cache
                    if save_to_parquet(df, cache_file, "monitoring data"):
                        st.success("💾 Saved to cache for faster loading")
                    
                    # Show summary
                    st.success(f"✅ Loaded {len(df):,} records")
                    
                    info = get_dataframe_info(df)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Records", f"{info['rows']:,}")
                    col2.metric("Columns", info['columns'])
                    col3.metric("Memory", f"{info['memory_mb']:.1f} MB")

                    # Show preview
                    st.markdown("**Data Preview:**")
                    st.dataframe(df.head(PREVIEW_ROWS), use_container_width=True)
    
    # Show current data status
    if 'df_vegetation' in st.session_state and st.session_state['df_vegetation'] is not None:
        st.markdown("---")
        st.markdown("### ✅ Current data status")
        
        df = st.session_state['df_vegetation']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        
        if 'aktId' in df.columns:
            col2.metric("Unique Plots", f"{df['aktId'].nunique():,}")
        
        if 'almindeligtNavn' in df.columns:
            col3.metric("Unique Species", f"{df['almindeligtNavn'].nunique():,}")
        
        if 'naturtypeId' in df.columns:
            col4.metric("Habitat Types", f"{df['naturtypeId'].nunique():,}")

        st.markdown("---")
        render_column_mapping_ui(df, section='columns_vegetation', roles=VEGETATION_ROLES,
                                 key_prefix='veg')

# TAB 2: Species Taxonomy & Traits
###################################################################################
with tab2:
    st.markdown("### 🌿 Import species taxonomy & traits")
    st.markdown("*Species names, taxonomy, and ecological indicator values (e.g., Ellenberg)*")
    
    # Check for cached data
    cache_file = cache_dir / 'taxa_data.parquet'
    cache_exists, cache_msg = check_file_exists(cache_file)
    
    # Auto-load from cache if exists and not already loaded
    if cache_exists and 'df_taxa' not in st.session_state:
        with st.spinner("Auto-loading cached taxonomy data..."):
            df = load_from_parquet(cache_file)
            if df is not None:
                st.session_state['df_taxa'] = df
                st.success(f"✅ Auto-loaded {len(df):,} species records from cache")
    
    if cache_exists:
        st.markdown('<div class="cache-status">', unsafe_allow_html=True)
        st.markdown("**⚡ Cached data available**")
        st.markdown(cache_msg)
        if 'df_taxa' in st.session_state:
            st.markdown("✅ *Currently loaded in session*")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reload from Cache", type="secondary", use_container_width=True, key="taxa_reload_cache"):
                with st.spinner("Reloading cached taxonomy data..."):
                    df = load_from_parquet(cache_file)
                    if df is not None:
                        st.session_state['df_taxa'] = df
                        st.success(f"✅ Reloaded {len(df):,} species records from cache")
                        st.rerun()
        
        with col2:
            if st.button("🔄 Reload from Source", use_container_width=True, key="taxa_reload_source"):
                cache_file.unlink()
                if 'df_taxa' in st.session_state:
                    del st.session_state['df_taxa']
                st.rerun()
    
    else:
        st.info("📁 No cached taxonomy data found. Please load from source file.")
    
    # Load from source file
    with st.expander("📂 Load from Source File", expanded=not cache_exists):
        st.markdown("**Select species taxonomy file**")
        st.caption("Expected format: Excel or CSV with species names, taxonomy, and traits (Ellenberg values, etc.)")
        
        source_file = st.text_input(
            "Taxa file path:",
            value=str(data_path / "taxa_data.xlsx"),
            help="Path to species taxonomy file (Excel or CSV)",
            key="taxa_file"
        )
        
        source_path = Path(source_file)
        exists, status = check_file_exists(source_path)
        st.markdown(status)
        
        # File format options
        if source_path.suffix in ['.xlsx', '.xls']:
            sheet_name = st.text_input("Sheet name:", value="Sheet1")
            
            if st.button("📥 Load from Source (Excel)", disabled=not exists, type="primary", use_container_width=True):
                with st.spinner("Loading taxonomy data from Excel..."):
                    df = load_excel_sheet(str(source_path), sheet_name)
                    
                    if df is not None:
                        st.session_state['df_taxa'] = df
                        
                        if save_to_parquet(df, cache_file, "taxonomy data"):
                            st.success("💾 Saved to cache for faster loading")
                        
                        st.success(f"✅ Loaded {len(df):,} species records")
                        
                        info = get_dataframe_info(df)
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Species", f"{info['rows']:,}")
                        col2.metric("Attributes", info['columns'])
                        col3.metric("Memory", f"{info['memory_mb']:.1f} MB")
                        
                        st.markdown("**Data Preview:**")
                        st.dataframe(df.head(PREVIEW_ROWS), use_container_width=True)
        
        else:  # CSV
            col1, col2 = st.columns(2)
            with col1:
                separator = st.selectbox("Separator:", [',', '\t', ';'], index=0, key="taxa_sep")
            with col2:
                encoding = st.selectbox("Encoding:", ['utf-8', 'latin1', 'iso-8859-1'], index=0, key="taxa_enc")
            
            if st.button("📥 Load from Source (CSV)", disabled=not exists, type="primary", use_container_width=True):
                with st.spinner("Loading taxonomy data from CSV..."):
                    df = load_csv_file(str(source_path), encoding=encoding, sep=separator)
                    
                    if df is not None:
                        st.session_state['df_taxa'] = df
                        
                        if save_to_parquet(df, cache_file, "taxonomy data"):
                            st.success("💾 Saved to cache for faster loading")
                        
                        st.success(f"✅ Loaded {len(df):,} species records")
                        
                        info = get_dataframe_info(df)
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Species", f"{info['rows']:,}")
                        col2.metric("Attributes", info['columns'])
                        col3.metric("Memory", f"{info['memory_mb']:.1f} MB")
                        
                        st.markdown("**Data Preview:**")
                        st.dataframe(df.head(PREVIEW_ROWS), use_container_width=True)
    
    # Show current data status
    if 'df_taxa' in st.session_state and st.session_state['df_taxa'] is not None:
        st.markdown("---")
        st.markdown("### ✅ Current taxonomy data status")
        
        df = st.session_state['df_taxa']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Species", f"{len(df):,}")
        col2.metric("Data Columns", f"{len(df.columns)}")
        
        # Check for Ellenberg values
        ellenberg_cols = [col for col in df.columns if 'ellenberg' in col.lower() or col.startswith('E_')]
        if ellenberg_cols:
            col3.metric("Ellenberg Indicators", len(ellenberg_cols))
            with st.expander("📊 Available Ellenberg Indicators"):
                st.write(ellenberg_cols)

        st.markdown("---")
        render_column_mapping_ui(df, section='columns_taxa', roles=TAXA_ROLES,
                                 key_prefix='taxa')

# TAB 3: Regional Species Pool (formerly Atlas Flora Danica)
###################################################################################
with tab3:
    st.markdown("### 🗺️ Import regional species pool data")
    st.markdown("*Regional occurrence data for dark diversity analysis (e.g., Atlas Flora Danica)*")
    
    # Check for cached data
    cache_file = cache_dir / 'regional_pool.parquet'
    cache_exists, cache_msg = check_file_exists(cache_file)
    
    # Auto-load from cache if exists and not already loaded
    if cache_exists and 'df_regional_pool' not in st.session_state:
        with st.spinner("Auto-loading cached regional pool data..."):
            df = load_from_parquet(cache_file)
            if df is not None:
                st.session_state['df_regional_pool'] = df
                st.success(f"✅ Auto-loaded {len(df):,} regional records from cache")
    
    if cache_exists:
        st.markdown('<div class="cache-status">', unsafe_allow_html=True)
        st.markdown("**⚡ Cached data available**")
        st.markdown(cache_msg)
        if 'df_regional_pool' in st.session_state:
            st.markdown("✅ *Currently loaded in session*")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reload from Cache", type="secondary", use_container_width=True, key="atlas_reload_cache"):
                with st.spinner("Reloading cached regional pool data..."):
                    df = load_from_parquet(cache_file)
                    if df is not None:
                        st.session_state['df_regional_pool'] = df
                        st.success(f"✅ Reloaded {len(df):,} regional records from cache")
                        st.rerun()
        
        with col2:
            if st.button("🔄 Reload from Source", use_container_width=True, key="atlas_reload_source"):
                cache_file.unlink()
                if 'df_regional_pool' in st.session_state:
                    del st.session_state['df_regional_pool']
                st.rerun()
    
    else:
        st.info("📁 No cached regional data found. Loading is optional but enhances dark diversity analysis.")
    
    # Load from source file
    with st.expander("📂 Load from Source File (Optional)", expanded=False):
        st.markdown("**Select regional species pool file**")
        st.caption("Expected format: CSV with species occurrences across regional grid cells or locations")
        
        source_file = st.text_input(
            "Regional pool file path:",
            value=str(data_path / "regional_species_pool.csv"),
            help="Path to regional species distribution data",
            key="atlas_file"
        )
        
        source_path = Path(source_file)
        exists, status = check_file_exists(source_path)
        st.markdown(status)
        
        # File format options
        col1, col2 = st.columns(2)
        with col1:
            separator = st.selectbox("Separator:", [',', '\t', ';'], index=1, key="atlas_sep")
        with col2:
            encoding = st.selectbox("Encoding:", ['utf-8', 'latin1', 'iso-8859-1'], index=1, key="atlas_enc")
        
        if st.button("📥 Load Regional Pool Data", disabled=not exists, type="primary", use_container_width=True):
            with st.spinner("Loading regional pool data..."):
                df = load_csv_file(str(source_path), encoding=encoding, sep=separator)
                
                if df is not None:
                    st.session_state['df_regional_pool'] = df
                    
                    if save_to_parquet(df, cache_file, "regional pool data"):
                        st.success("💾 Saved to cache for faster loading")
                    
                    st.success(f"✅ Loaded {len(df):,} regional records")
                    
                    info = get_dataframe_info(df)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Records", f"{info['rows']:,}")
                    col2.metric("Columns", info['columns'])
                    col3.metric("Memory", f"{info['memory_mb']:.1f} MB")
                    
                    st.markdown("**Data Preview:**")
                    st.dataframe(df.head(PREVIEW_ROWS), use_container_width=True)
    
    # Show current data status
    if 'df_regional_pool' in st.session_state and st.session_state['df_regional_pool'] is not None:
        st.markdown("---")
        st.markdown("### ✅ Current regional pool status")
        
        df = st.session_state['df_regional_pool']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", f"{len(df):,}")
        
        # Try to identify species and location columns
        possible_species_cols = ['species', 'taxon', 'almindeligtNavn', 'videnskabeligtNavn']
        species_col = next((col for col in possible_species_cols if col in df.columns), None)
        if species_col:
            col2.metric("Unique Species", f"{df[species_col].nunique():,}")
        
        possible_location_cols = ['grid_cell', 'location', 'region', 'UTM10']
        location_col = next((col for col in possible_location_cols if col in df.columns), None)
        if location_col:
            col3.metric("Locations", f"{df[location_col].nunique():,}")

        st.markdown("---")
        render_column_mapping_ui(df, section='columns_regional_pool', roles=REGIONAL_POOL_ROLES,
                                 key_prefix='regpool')

# TAB 4: Custom Data Import
###################################################################################
with tab4:
    st.markdown("### 📤 Import custom vegetation plot data")
    st.markdown("*Import external vegetation monitoring data for overlay mapping*")
    
    st.info("""
    📋 **Required columns for compatibility:**
    - `aktId`: Unique plot identifier
    - `almindeligtNavn`: Species name (Danish common name or adapt to your naming)
    - `videnskabeligtNavn`: Scientific name
    - `UTMx`, `UTMy`: Plot coordinates
    
    Optional columns: `naturtypeId`, `major_type`, `aarstal` (year), etc.
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Excel file with vegetation plot data",
        type=['xlsx', 'xls'],
        help="Excel file with plot-level species occurrence data. Recommended maximum: 100 MB for optimal performance."
    )
    
    # Dataset name and import button
    col1, col2 = st.columns([3, 1])
    with col1:
        dataset_name = st.text_input(
            "Dataset name (for saving):",
            placeholder="e.g., my_forest_plots_2024",
            help="Used as filename in external_data folder"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        import_button = st.button("💾 Import & Save", type="primary", use_container_width=True)
    
    # Process uploaded file
    if uploaded_file is not None:
        # Import sqlite3 locally (only needed for this tab)
        import sqlite3

        try:
            df_custom = pd.read_excel(uploaded_file)
            
            with st.expander("👀 Preview uploaded data", expanded=True):
                st.info(f"Uploaded file contains {len(df_custom):,} rows and {len(df_custom.columns)} columns")
                st.dataframe(df_custom.head(PREVIEW_ROWS), use_container_width=True)
                
                # Show column names
                st.markdown("**Columns in file:**")
                st.code(", ".join(df_custom.columns.tolist()))
            
            # Required columns (basic set for compatibility)
            required_cols = ['aktId', 'almindeligtNavn', 'videnskabeligtNavn', 'UTMx', 'UTMy']
            missing_cols = [col for col in required_cols if col not in df_custom.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Required columns: aktId (plot ID), almindeligtNavn (species name), videnskabeligtNavn (scientific name), UTMx, UTMy (coordinates)")
            else:
                st.success("✅ All required columns found - data is compatible!")
                
                # Species name validation against taxa database
                st.markdown("---")
                st.markdown("#### 🔍 Species name validation")
                
                if 'df_taxa' in st.session_state and st.session_state['df_taxa'] is not None:
                    df_taxa = st.session_state['df_taxa']
                    
                    # Get unique species from uploaded data
                    uploaded_species = set(df_custom['almindeligtNavn'].dropna().unique())
                    
                    # Get species from taxa database
                    taxa_species_col = None
                    for col in ['almindeligtNavn', 'species', 'taxon_name', 'scientific_name']:
                        if col in df_taxa.columns:
                            taxa_species_col = col
                            break
                    
                    if taxa_species_col:
                        taxa_species = set(df_taxa[taxa_species_col].dropna().unique())
                        
                        # Find matches and mismatches
                        matched_species = uploaded_species & taxa_species
                        unmatched_species = uploaded_species - taxa_species
                        
                        # Calculate match rate
                        match_rate = 100 * len(matched_species) / len(uploaded_species) if uploaded_species else 0
                        
                        # Show match statistics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Uploaded Species", len(uploaded_species))
                        col2.metric("Matched", len(matched_species), 
                                   delta=f"{match_rate:.1f}%", delta_color="normal")
                        col3.metric("Unmatched", len(unmatched_species),
                                   delta=f"{100-match_rate:.1f}%", delta_color="inverse")
                        
                        # Show warnings or success based on match rate
                        if match_rate == 100:
                            st.success("✅ All species names match the taxonomy database! Data can be mapped to ecological space.")
                        elif match_rate >= 80:
                            st.warning(f"⚠️ {len(unmatched_species)} species ({100-match_rate:.1f}%) not found in taxonomy database. These will not be positioned in ecological space.")
                        else:
                            st.error(f"❌ Only {match_rate:.1f}% species match! {len(unmatched_species)} species not found in taxonomy database.")
                            st.info("💡 Consider checking species names for spelling variations or taxonomic differences.")
                        
                        # Show list of unmatched species if any
                        if unmatched_species:
                            with st.expander(f"📋 Unmatched species list ({len(unmatched_species)})", expanded=False):
                                st.markdown("**These species are in your data but NOT in the taxonomy database:**")
                                st.markdown("*Common causes: different taxonomy, typos, or species not in reference database*")
                                
                                # Sort and show with record counts
                                unmatched_sorted = sorted(list(unmatched_species))
                                unmatched_df = pd.DataFrame({
                                    'Species name': unmatched_sorted,
                                    'Records': [len(df_custom[df_custom['almindeligtNavn'] == sp]) for sp in unmatched_sorted]
                                })
                                unmatched_df = unmatched_df.sort_values('Records', ascending=False)
                                st.dataframe(unmatched_df, use_container_width=True, height=300)
                                
                                # Download option
                                csv = unmatched_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download unmatched species list",
                                    data=csv,
                                    file_name=f"unmatched_species_{dataset_name}.csv",
                                    mime="text/csv"
                                )
                    else:
                        st.warning("⚠️ Could not identify species column in taxonomy database")
                else:
                    st.info("ℹ️ Taxonomy database not loaded. Species name validation skipped.")
                    st.markdown("*Import taxonomy data first to enable validation*")
                
                st.markdown("---")
                
                # Import and save
                if import_button and dataset_name:
                    with st.spinner("Importing and saving custom vegetation data..."):
                        try:
                            # Use session state path for external data
                            external_folder = external_data_path
                            external_folder.mkdir(parents=True, exist_ok=True)

                            # Create SQLite database
                            db_path = external_folder / f"{dataset_name}.db"

                            # Save to database using context manager
                            with sqlite3.connect(db_path) as conn:
                                # Main data table
                                df_custom.to_sql('data', conn, if_exists='replace', index=False)

                                # Species table
                                if 'almindeligtNavn' in df_custom.columns:
                                    species_cols = ['almindeligtNavn']
                                    if 'videnskabeligtNavn' in df_custom.columns:
                                        species_cols.append('videnskabeligtNavn')

                                    species_df = df_custom[species_cols].drop_duplicates().reset_index(drop=True)
                                    species_df.columns = [col if col == 'videnskabeligtNavn' else 'species'
                                                         for col in species_df.columns]
                                    species_df.to_sql('species', conn, if_exists='replace', index=False)

                                # Plot table
                                if 'aktId' in df_custom.columns:
                                    aktId_cols = ['aktId', 'UTMx', 'UTMy']
                                    optional_cols = ['naturtypeId', 'major_type', 'novanareg', 'bioreg', 'aarstal']
                                    for col in optional_cols:
                                        if col in df_custom.columns and col not in aktId_cols:
                                            aktId_cols.append(col)

                                    aktId_df = df_custom[aktId_cols].drop_duplicates(subset=['aktId']).reset_index(drop=True)
                                    aktId_df.to_sql('aktId', conn, if_exists='replace', index=False)
                            
                            # Success message
                            st.success(f"✅ Successfully imported {len(df_custom):,} vegetation records!")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total Records", f"{len(df_custom):,}")
                            col2.metric("Unique Plots", f"{df_custom['aktId'].nunique():,}")
                            
                            if 'almindeligtNavn' in df_custom.columns:
                                col3.metric("Unique Species", f"{df_custom['almindeligtNavn'].nunique():,}")
                            
                            col4.metric("File Size", f"{db_path.stat().st_size / 1024:.1f} KB")
                            
                            st.info(f"📁 Saved to: `{db_path.relative_to(external_data_path.parent)}`")
                            
                            # Preview saved data
                            with st.expander("📋 Saved data preview", expanded=False):
                                st.dataframe(df_custom.head(), use_container_width=True)
                            
                        except PermissionError as e:
                            st.error(f"❌ Permission denied: Cannot write to {external_folder}")
                            st.info("💡 Check folder permissions or choose a different location")
                        except sqlite3.Error as e:
                            st.error(f"❌ Database error: {str(e)}")
                            st.info("💡 The database file may be locked or corrupted")
                        except Exception as e:
                            st.error(f"❌ Unexpected error saving data: {str(e)}")
                            st.exception(e)
                
                elif import_button and not dataset_name:
                    st.warning("⚠️ Please provide a dataset name")
        
        except pd.errors.ParserError as e:
            st.error(f"❌ Error parsing Excel file: {str(e)}")
            st.info("💡 Check if the file is a valid Excel format")
        except Exception as e:
            st.error(f"❌ Unexpected error reading Excel file: {str(e)}")
            st.exception(e)

# Summary Section
###################################################################################
st.markdown("---")
st.markdown("### 📊 Data import summary")

data_status = []

if 'df_vegetation' in st.session_state and st.session_state['df_vegetation'] is not None:
    df = st.session_state['df_vegetation']
    data_status.append({
        'Dataset': 'Vegetation Monitoring',
        'Status': '✅ Loaded',
        'Records': f"{len(df):,}",
        'Memory': f"{get_dataframe_info(df)['memory_mb']:.1f} MB"
    })
else:
    data_status.append({
        'Dataset': 'Vegetation Monitoring',
        'Status': '❌ Not loaded',
        'Records': '-',
        'Memory': '-'
    })

if 'df_taxa' in st.session_state and st.session_state['df_taxa'] is not None:
    df = st.session_state['df_taxa']
    data_status.append({
        'Dataset': 'Species Taxonomy',
        'Status': '✅ Loaded',
        'Records': f"{len(df):,}",
        'Memory': f"{get_dataframe_info(df)['memory_mb']:.1f} MB"
    })
else:
    data_status.append({
        'Dataset': 'Species Taxonomy',
        'Status': '❌ Not loaded',
        'Records': '-',
        'Memory': '-'
    })

if 'df_regional_pool' in st.session_state and st.session_state['df_regional_pool'] is not None:
    df = st.session_state['df_regional_pool']
    data_status.append({
        'Dataset': 'Regional Species Pool',
        'Status': '✅ Loaded',
        'Records': f"{len(df):,}",
        'Memory': f"{get_dataframe_info(df)['memory_mb']:.1f} MB"
    })
else:
    data_status.append({
        'Dataset': 'Regional Species Pool',
        'Status': '⚪ Optional',
        'Records': '-',
        'Memory': '-'
    })

st.dataframe(pd.DataFrame(data_status), use_container_width=True, hide_index=True)

# Next steps
if 'df_vegetation' in st.session_state and 'df_taxa' in st.session_state:
    st.success("✅ **Core data loaded!** Proceed to Data Filtering (2/2) to prepare datasets for network analysis.")
else:
    st.info("⚠️ Load vegetation monitoring data and species taxonomy to proceed with the workflow.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d; font-size: 0.9em;'>
    EcoNetMap - Data Import Module
    </div>
    """, 
    unsafe_allow_html=True
)
