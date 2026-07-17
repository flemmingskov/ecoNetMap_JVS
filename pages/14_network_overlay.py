"""
EcoNetMap - Map Preparation Module
==================================================
Combines network coordinates with query data to create unified overlay databases.
Merges species positions from network analysis with plot-level occurrence data,
calculates plot coordinates as weighted averages, and adds ecological indicators
(Ellenberg values) when available. Validates data quality and exports to SQLite
databases ready for visualization.

Part of the EcoNetMap toolkit - Network-based Ecological Cartography
Author: Flemming Skov
Updated: January 2026
"""

# Import packages for web applications
import streamlit as st

# Import packages for data manipulation and analysis
import pandas as pd
import numpy as np
import sqlite3

# Import packages for file and system operations
from pathlib import Path
import datetime

# Import packages for type hints
from typing import Optional, Tuple, Dict, List

# Column-role mapping (lets users adapt the toolkit to their own column names)
from column_config import TAXA_ROLES, rename_to_canonical

# Import packages for visualization
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Map Preparation - EcoNetMap", 
    page_icon="🗺️",
    layout="wide"
)

# Title and description
col1, col2 = st.columns([4, 1])
with col1:
    st.header("Network overlay")
    st.subheader("🗺️ Map preparation")
    st.markdown("*Create overlay maps by combining reference coordinates with query data*")
with col2:
    pass

st.markdown("---")

# Session state validation
###################################################################################
if 'queries_path' not in st.session_state or not st.session_state.queries_path:
    st.error("⚠️ Project paths not initialized. Please run the **Home** page first to set up your project directory.")
    st.stop()

if 'reference_map_path' not in st.session_state or not st.session_state.reference_map_path:
    st.error("⚠️ Reference map path not set. Please run the **Home** page first to set up your project directory.")
    st.stop()

# FUNCTIONS
###################################################################################

@st.cache_data(show_spinner=False)
def load_coordinate_data(db_path: str, table_name: str = 'keyword_coordinates') -> Optional[pd.DataFrame]:
    """Load coordinate data from reference map database"""
    try:
        conn = sqlite3.connect(db_path)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        
        if table_name in tables['name'].values:
            df = pd.read_sql_query(f'SELECT * FROM {table_name}', conn)
        elif 'keyword_raw_coordinates' in tables['name'].values:
            df = pd.read_sql_query('SELECT * FROM keyword_raw_coordinates', conn)
        else:
            conn.close()
            return None
            
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading coordinate data: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def load_overlay_data(db_path: str) -> Dict[str, pd.DataFrame]:
    """Load all relevant tables from overlay database"""
    try:
        conn = sqlite3.connect(db_path)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        
        data_dict = {}
        
        # Load main data table
        if 'data' in tables['name'].values:
            data_dict['data'] = pd.read_sql_query('SELECT * FROM data', conn)
        
        # Load species table
        if 'species_list' in tables['name'].values:
            data_dict['species'] = pd.read_sql_query('SELECT * FROM species_list', conn)
        
        # Load plot_id table
        if 'plot_id' in tables['name'].values:
            data_dict['plot_id'] = pd.read_sql_query('SELECT * FROM plot_id', conn)
        
        conn.close()
        return data_dict
    except Exception as e:
        st.error(f"Error loading overlay data: {str(e)}")
        return {}

def merge_species_coordinates(species_df: pd.DataFrame, coordinates_df: pd.DataFrame,
                            include_unmatched: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """Merge species data with reference coordinates on species_key -> node_key."""
    coord_cols = set(coordinates_df.columns)

    merge_report = {
        'strategy_used': None,
        'total_species': len(species_df),
        'matched_species': 0,
        'unmatched_species': 0,
        'match_rate': 0.0,
        'columns_merged': []
    }

    if 'species_key' in species_df.columns and 'node_key' in coord_cols:
        merged = pd.merge(
            species_df,
            coordinates_df,
            left_on='species_key',
            right_on='node_key',
            how='left',
            indicator=True,
            suffixes=('', '_coord')
        )

        matched = (merged['_merge'] == 'both').sum()
        total = len(merged)
        match_rate = matched/total*100 if total > 0 else 0

        merge_report['strategy_used'] = 'species_key → node_key'
        merge_report['matched_species'] = matched
        merge_report['unmatched_species'] = total - matched
        merge_report['match_rate'] = match_rate
        merge_report['columns_merged'] = ['species_key', 'node_key']

        if not include_unmatched:
            merged = merged[merged['_merge'] == 'both']

        return merged.drop('_merge', axis=1), merge_report

    # If merge isn't possible
    merge_report['strategy_used'] = 'No suitable columns - merge failed'

    # Add empty coordinate columns
    for col in ['xcoor', 'ycoor', 'keyword', 'wdegree']:
        if col in coord_cols:
            species_df[col] = np.nan

    return species_df, merge_report

def calculate_plot_coordinates(data_df: pd.DataFrame, coordinates_df: pd.DataFrame, 
                             weighting: str = 'equal', coord_precision: int = 6,
                             taxa_df: pd.DataFrame = None, 
                             plot_id_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Calculate plot coordinates with different weighting options
    
    Returns: (plot_summary, calculation_report)
    """
    calc_report = {
        'merge_method': None,
        'species_with_coords': 0,
        'species_without_coords': 0,
        'coord_coverage': 0.0,
        'ellenberg_available': [],
        'ellenberg_coverage': 0,
        'columns_preserved': 0,
        'warnings': []
    }
    
    # Merge data with coordinates on species_key -> node_key
    if 'species_key' in data_df.columns and 'node_key' in coordinates_df.columns:
        merged = pd.merge(data_df, coordinates_df, left_on='species_key', right_on='node_key', how='left', suffixes=('', '_coord'))
        calc_report['merge_method'] = 'species_key → node_key'
    else:
        calc_report['warnings'].append('No suitable matching columns found')
        return pd.DataFrame(), calc_report

    # Merge with taxa data (containing Ellenberg indicators) if available
    ellenberg_cols = ['M', 'N', 'L', 'R', 'T']
    if taxa_df is not None and not taxa_df.empty and 'species_key' in taxa_df.columns:
        available_ellenberg = [col for col in ellenberg_cols if col in taxa_df.columns]

        if available_ellenberg:
            merged = pd.merge(merged, taxa_df[['species_key'] + available_ellenberg], on='species_key', how='left', suffixes=('', '_taxa'))
            calc_report['ellenberg_available'] = available_ellenberg
            calc_report['ellenberg_coverage'] = merged[available_ellenberg].notna().any(axis=1).sum()
    
    # Remove rows without coordinates
    merged_valid = merged.dropna(subset=['xcoor', 'ycoor'])
    
    if len(merged_valid) == 0:
        calc_report['warnings'].append('No species with valid coordinates')
        return pd.DataFrame(), calc_report
    
    # Report coordinate coverage
    calc_report['species_with_coords'] = len(merged_valid)
    calc_report['species_without_coords'] = len(merged) - len(merged_valid)
    calc_report['coord_coverage'] = len(merged_valid) / len(merged) * 100 if len(merged) > 0 else 0
    
    # Calculate based on weighting method
    if weighting == 'equal':
        plot_summary = merged_valid.groupby('plot_id').agg({
            'xcoor': 'mean',
            'ycoor': 'mean',
            'species_key': 'count'
        }).reset_index()
        plot_summary.rename(columns={'species_key': 'speciesNum'}, inplace=True)
        
    elif weighting == 'abundance' and 'abundance' in merged_valid.columns:
        merged_valid['weighted_x'] = merged_valid['xcoor'] * merged_valid['abundance']
        merged_valid['weighted_y'] = merged_valid['ycoor'] * merged_valid['abundance']
        
        plot_summary = merged_valid.groupby('plot_id').agg({
            'weighted_x': 'sum',
            'weighted_y': 'sum',
            'abundance': 'sum',
            'species_key': 'count'
        }).reset_index()
        
        plot_summary['xcoor'] = plot_summary['weighted_x'] / plot_summary['abundance']
        plot_summary['ycoor'] = plot_summary['weighted_y'] / plot_summary['abundance']
        plot_summary.rename(columns={'species_key': 'speciesNum'}, inplace=True)
        plot_summary = plot_summary.drop(['weighted_x', 'weighted_y'], axis=1)
        
    elif weighting == 'degree' and 'wdegree' in merged_valid.columns:
        merged_valid['weighted_x'] = merged_valid['xcoor'] * merged_valid['wdegree']
        merged_valid['weighted_y'] = merged_valid['ycoor'] * merged_valid['wdegree']
        
        plot_summary = merged_valid.groupby('plot_id').agg({
            'weighted_x': 'sum',
            'weighted_y': 'sum',
            'wdegree': 'sum',
            'species_key': 'count'
        }).reset_index()
        
        plot_summary['xcoor'] = plot_summary['weighted_x'] / plot_summary['wdegree']
        plot_summary['ycoor'] = plot_summary['weighted_y'] / plot_summary['wdegree']
        plot_summary.rename(columns={'species_key': 'speciesNum'}, inplace=True)
        plot_summary = plot_summary.drop(['weighted_x', 'weighted_y'], axis=1)
    else:
        # Fallback to equal weighting
        calc_report['warnings'].append(f"Weighting '{weighting}' not available, using equal weighting")
        plot_summary = merged_valid.groupby('plot_id').agg({
            'xcoor': 'mean',
            'ycoor': 'mean',
            'species_key': 'count'
        }).reset_index()
        plot_summary.rename(columns={'species_key': 'speciesNum'}, inplace=True)
    
    # Round coordinates
    plot_summary['xcoor'] = plot_summary['xcoor'].round(coord_precision)
    plot_summary['ycoor'] = plot_summary['ycoor'].round(coord_precision)
    
    # Calculate Ellenberg indicators means
    available_eco_cols = [col for col in ellenberg_cols if col in merged_valid.columns]
    
    if available_eco_cols:
        if weighting == 'abundance' and 'abundance' in merged_valid.columns:
            # Weight Ellenberg by abundance
            weighted_eco = {}
            for col in available_eco_cols:
                if col in merged_valid.columns and merged_valid[col].notna().sum() > 0:
                    merged_valid[f'weighted_{col}'] = merged_valid[col] * merged_valid['abundance']
                    weighted_eco[f'weighted_{col}'] = 'sum'
            
            if weighted_eco:
                eco_summary = merged_valid.groupby('plot_id').agg(weighted_eco).reset_index()
                abundance_summary = merged_valid.groupby('plot_id')['abundance'].sum().reset_index()
                
                for col in available_eco_cols:
                    if f'weighted_{col}' in eco_summary.columns:
                        eco_summary[col] = (eco_summary[f'weighted_{col}'] / abundance_summary['abundance']).round(2)
                        eco_summary = eco_summary.drop(f'weighted_{col}', axis=1)
                
                plot_summary = pd.merge(plot_summary, eco_summary, on='plot_id', how='left')
            
        elif weighting == 'degree' and 'wdegree' in merged_valid.columns:
            # Weight Ellenberg by degree
            weighted_eco = {}
            for col in available_eco_cols:
                if col in merged_valid.columns and merged_valid[col].notna().sum() > 0:
                    merged_valid[f'weighted_{col}'] = merged_valid[col] * merged_valid['wdegree']
                    weighted_eco[f'weighted_{col}'] = 'sum'
            
            if weighted_eco:
                eco_summary = merged_valid.groupby('plot_id').agg(weighted_eco).reset_index()
                degree_summary = merged_valid.groupby('plot_id')['wdegree'].sum().reset_index()
                
                for col in available_eco_cols:
                    if f'weighted_{col}' in eco_summary.columns:
                        eco_summary[col] = (eco_summary[f'weighted_{col}'] / degree_summary['wdegree']).round(2)
                        eco_summary = eco_summary.drop(f'weighted_{col}', axis=1)
                
                plot_summary = pd.merge(plot_summary, eco_summary, on='plot_id', how='left')
        else:
            # Simple unweighted means
            valid_eco_cols = [col for col in available_eco_cols 
                            if col in merged_valid.columns and merged_valid[col].notna().sum() > 0]
            
            if valid_eco_cols:
                ecological_means = merged_valid.groupby('plot_id')[valid_eco_cols].mean().round(2).reset_index()
                plot_summary = pd.merge(plot_summary, ecological_means, on='plot_id', how='left')
    
    # Calculate mean Mahalanobis distance per plot (species-level quality → plot-level)
    # Uses the same weighting as plot coordinates for methodological consistency.
    # Gracefully skipped if mahal_dist column is absent (older reference databases).
    if 'mahal_dist' in merged_valid.columns and merged_valid['mahal_dist'].notna().any():
        if weighting == 'abundance' and 'abundance' in merged_valid.columns:
            merged_valid['_mahal_w'] = merged_valid['mahal_dist'] * merged_valid['abundance']
            mahal_num = merged_valid.groupby('plot_id')['_mahal_w'].sum().reset_index()
            mahal_den = merged_valid.groupby('plot_id')['abundance'].sum().reset_index()
            mahal_summary = pd.merge(mahal_num, mahal_den, on='plot_id')
            mahal_summary['mean_mahal_dist'] = (mahal_summary['_mahal_w'] / mahal_summary['abundance']).round(4)
            mahal_summary = mahal_summary[['plot_id', 'mean_mahal_dist']]

        elif weighting == 'degree' and 'wdegree' in merged_valid.columns:
            merged_valid['_mahal_w'] = merged_valid['mahal_dist'] * merged_valid['wdegree']
            mahal_num = merged_valid.groupby('plot_id')['_mahal_w'].sum().reset_index()
            mahal_den = merged_valid.groupby('plot_id')['wdegree'].sum().reset_index()
            mahal_summary = pd.merge(mahal_num, mahal_den, on='plot_id')
            mahal_summary['mean_mahal_dist'] = (mahal_summary['_mahal_w'] / mahal_summary['wdegree']).round(4)
            mahal_summary = mahal_summary[['plot_id', 'mean_mahal_dist']]

        else:
            mahal_summary = merged_valid.groupby('plot_id')['mahal_dist'].mean().round(4).reset_index()
            mahal_summary.rename(columns={'mahal_dist': 'mean_mahal_dist'}, inplace=True)

        plot_summary = pd.merge(plot_summary, mahal_summary, on='plot_id', how='left')

    # Merge with original plot_id data to preserve ALL columns
    if plot_id_df is not None and not plot_id_df.empty:
        plot_summary = pd.merge(plot_id_df, plot_summary, on='plot_id', how='left', suffixes=('', '_calc'))
        calc_report['columns_preserved'] = len(plot_id_df.columns)
    
    return plot_summary, calc_report

def validate_map_data(taxa_df: Optional[pd.DataFrame], 
                     plot_df: Optional[pd.DataFrame], 
                     species_df: Optional[pd.DataFrame]) -> List[Dict]:
    """Validate data quality before saving"""
    issues = []
    
    # Check for missing coordinates in taxa
    if taxa_df is not None and not taxa_df.empty and 'xcoor' in taxa_df.columns:
        missing_coords = taxa_df[taxa_df['xcoor'].isna() | taxa_df['ycoor'].isna()]
        if len(missing_coords) > 0:
            issues.append({
                'severity': 'error',
                'message': f'{len(missing_coords)} taxa missing coordinates',
                'count': len(missing_coords)
            })
    
    # Check for missing coordinates in plots
    if plot_df is not None and not plot_df.empty and 'xcoor' in plot_df.columns:
        missing_plot_coords = plot_df[plot_df['xcoor'].isna() | plot_df['ycoor'].isna()]
        if len(missing_plot_coords) > 0:
            issues.append({
                'severity': 'warning',
                'message': f'{len(missing_plot_coords)} plots missing coordinates',
                'count': len(missing_plot_coords)
            })
    
    # Check coordinate ranges
    for df, name in [(taxa_df, 'taxa'), (plot_df, 'plots'), (species_df, 'species')]:
        if df is not None and not df.empty and 'xcoor' in df.columns and 'ycoor' in df.columns:
            out_of_range = df[
                (df['xcoor'].notna()) & (df['ycoor'].notna()) & 
                ((df['xcoor'] < 0) | (df['xcoor'] > 1) | 
                 (df['ycoor'] < 0) | (df['ycoor'] > 1))
            ]
            if len(out_of_range) > 0:
                issues.append({
                    'severity': 'error',
                    'message': f'{len(out_of_range)} {name} with coordinates outside [0,1] range',
                    'count': len(out_of_range)
                })
    
    # Check for duplicate node_keys in taxa
    if taxa_df is not None and 'node_key' in taxa_df.columns:
        duplicates = taxa_df['node_key'].duplicated().sum()
        if duplicates > 0:
            issues.append({
                'severity': 'warning',
                'message': f'{duplicates} duplicate node_keys in taxa table',
                'count': duplicates
            })
    
    return issues

def get_database_info(db_path: str) -> dict:
    """Get information about database tables and records"""
    try:
        conn = sqlite3.connect(db_path)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        
        info = {'tables': tables['name'].tolist(), 'record_counts': {}}
        
        for table in info['tables']:
            count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table}", conn)
            info['record_counts'][table] = count['count'][0]
        
        conn.close()
        return info
    except Exception as e:
        return {'tables': [], 'record_counts': {}, 'error': str(e)}

# SESSION STATE INITIALIZATION
###################################################################################

if 'coordinates_df' not in st.session_state:
    st.session_state.coordinates_df = None
if 'overlay_data' not in st.session_state:
    st.session_state.overlay_data = None
if 'species_merged' not in st.session_state:
    st.session_state.species_merged = None
if 'plot_id_merged' not in st.session_state:
    st.session_state.plot_id_merged = None
if 'taxa_merged' not in st.session_state:
    st.session_state.taxa_merged = None
if 'merge_report' not in st.session_state:
    st.session_state.merge_report = {}
if 'calc_report' not in st.session_state:
    st.session_state.calc_report = {}
if 'validation_issues' not in st.session_state:
    st.session_state.validation_issues = []

# MAIN INTERFACE
###################################################################################

# Explanation section
with st.expander("ℹ️ How does map preparation work?", expanded=False):
    st.markdown("""
    **Process Overview:**
    
    1. **Load reference coordinates:** Species positions from network analysis (Step 3)
    2. **Load query data:** Filtered species/plot data from your query (Step 2)
    3. **Merge species:** Match query species with reference coordinates
    4. **Calculate plot positions:** Weighted average of species coordinates
    5. **Add ecological indicators:** Ellenberg values from trait database (if available)
    6. **Validate & save:** Quality checks and export to map database
    
    **Plot Coordinate Calculation:**
    
    - **Equal weighting (recommended):** All species contribute equally
      - Use for: General ecological positioning
      - Matches network construction methodology (Jaccard-based)
      
    - **Abundance weighting:** Common species influence position more
      - Use for: When abundance matters (e.g., biomass, cover patterns)
      - Note: Inconsistent with network (which ignores abundance)
      
    - **Degree weighting (experimental):** Network-important species matter more
      - Use with caution: May favor generalists over specialists
      - Higher degree = more connections = typically generalist species
      - May mask contributions from rare specialist species
    
    **Output:**
    - SQLite database with taxa coordinates, plot coordinates, and metadata
    - Ready for visualization and spatial analysis
    - Includes Ellenberg indicators if available in trait database
    """)

st.markdown("---")
st.markdown("### 📁 Select input databases")

# Input database selection
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🗺️ Reference coordinate database")
    
    # Use proper path resolution
    default_output_path = None
    
    if 'reference_map_path' in st.session_state and st.session_state['reference_map_path']:
        default_output_path = Path(st.session_state['reference_map_path'])
    elif 'project_base_path' in st.session_state and st.session_state['project_base_path']:
        default_output_path = Path(st.session_state['project_base_path']) / 'reference_maps'
    elif 'queries_path' in st.session_state and st.session_state['queries_path']:
        queries_path = Path(st.session_state['queries_path'])
        default_output_path = queries_path.parent / 'reference_maps'
    else:
        default_output_path = Path('./reference_maps')
    
    if not default_output_path.exists():
        st.error(f"Reference maps directory not found: {default_output_path}")
        st.info("Please complete network analysis (Step 3) first")
        st.stop()
    
    ref_db_files = sorted([f.name for f in default_output_path.glob("*.db")])
    
    if ref_db_files:
        # Auto-select if coming from previous step
        default_ref_idx = 0
        if 'enhanced_coordinates_path' in st.session_state:
            default_name = Path(st.session_state.enhanced_coordinates_path).name
            if default_name in ref_db_files:
                default_ref_idx = ref_db_files.index(default_name)
        elif 'coordinates_db_path' in st.session_state:
            default_name = Path(st.session_state.coordinates_db_path).name
            if default_name in ref_db_files:
                default_ref_idx = ref_db_files.index(default_name)
        
        selected_ref_db = st.selectbox(
            "Select coordinate database:",
            options=ref_db_files,
            index=default_ref_idx,
            help="Database containing species coordinates from network analysis"
        )
        ref_db_path = default_output_path / selected_ref_db
        
        st.caption(f"📂 From: `{default_output_path.name}`")
        
        # Show database info
        ref_info = get_database_info(str(ref_db_path))
        with st.expander("📊 Database Info", expanded=False):
            for table, count in ref_info['record_counts'].items():
                st.text(f"{table}: {count:,} records")
    else:
        st.error("No reference coordinate databases found")
        st.stop()

with col2:
    st.markdown("#### 📍 Query/Overlay Database")
    
    queries_path = Path(st.session_state.get('queries_path', '.'))
    if not queries_path.exists():
        st.error(f"Queries directory not found: {queries_path}")
        st.info("Please complete data filtering (Step 2) first")
        st.stop()
    
    query_db_files = sorted([f.name for f in queries_path.glob("*.db")])
    
    if query_db_files:
        selected_query_db = st.selectbox(
            "Select query database:",
            options=query_db_files,
            help="Database containing filtered species/plot data to overlay"
        )
        query_db_path = queries_path / selected_query_db
        
        st.caption(f"📂 From: `{queries_path.name}`")
        
        # Show database info
        query_info = get_database_info(str(query_db_path))
        with st.expander("📊 Database Info", expanded=False):
            for table, count in query_info['record_counts'].items():
                st.text(f"{table}: {count:,} records")
    else:
        st.error("No query databases found")
        st.stop()

# Configuration options
st.markdown("---")
st.markdown("### ⚙️ Configuration options")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Merging Options:**")
    
    include_unmatched = st.checkbox(
        "Include unmatched species",
        value=True,
        help="Keep species without coordinates in output (with NA coordinates)"
    )
    
    validate_data = st.checkbox(
        "Validate data before saving",
        value=True,
        help="Check for data quality issues"
    )

with col2:
    st.markdown("**Plot Calculation:**")
    
    plot_weighting = st.selectbox(
        "Plot coordinate weighting:",
        options=["equal", "abundance", "degree"],
        help="How to weight species when calculating plot positions"
    )
    
    if plot_weighting == "degree":
        st.caption("⚠️ Degree weighting may favor generalists")
    
    coord_precision = st.number_input(
        "Coordinate precision:",
        min_value=2,
        max_value=10,
        value=6,
        help="Decimal places for coordinates"
    )

with col3:
    st.markdown("**Output Options:**")
    
    create_summary_stats = st.checkbox(
        "Generate summary statistics",
        value=True,
        help="Create summary statistics table"
    )

# Process button
st.markdown("---")

if st.button("🔨 Load and Process Databases", type="primary", use_container_width=True):
    
    # Load reference coordinates
    with st.spinner("Loading reference coordinates..."):
        coordinates_df = load_coordinate_data(str(ref_db_path))
        
        if coordinates_df is None:
            st.error("❌ Failed to load coordinate data")
            st.stop()
        
        st.session_state.coordinates_df = coordinates_df
    
    # Load query data
    with st.spinner("Loading query data..."):
        overlay_data = load_overlay_data(str(query_db_path))
        
        if not overlay_data:
            st.error("❌ Failed to load overlay data")
            st.stop()
        
        st.session_state.overlay_data = overlay_data
    
    # Load additional data from session state (canonical names, per settings.txt mapping)
    raw_taxa_df = st.session_state.get('df_taxa', None)
    taxa_df = rename_to_canonical(raw_taxa_df, 'columns_taxa', TAXA_ROLES) if raw_taxa_df is not None else None
    
    # Process species data
    with st.spinner("Merging species with coordinates..."):
        if 'species' in overlay_data:
            species_merged, merge_report = merge_species_coordinates(
                overlay_data['species'], 
                coordinates_df,
                include_unmatched=include_unmatched
            )
            st.session_state.species_merged = species_merged
            st.session_state.merge_report = merge_report
        else:
            st.session_state.species_merged = None
            st.session_state.merge_report = {}
    
    # Calculate plot coordinates
    with st.spinner("Calculating plot coordinates..."):
        if 'data' in overlay_data:
            plot_id_df = overlay_data.get('plot_id', None)
            
            plot_summary, calc_report = calculate_plot_coordinates(
                overlay_data['data'],
                coordinates_df,
                weighting=plot_weighting,
                coord_precision=coord_precision,
                taxa_df=taxa_df,
                plot_id_df=plot_id_df
            )
            
            if plot_summary.empty:
                st.session_state.plot_id_merged = None
                st.session_state.calc_report = calc_report
            else:
                st.session_state.plot_id_merged = plot_summary
                st.session_state.calc_report = calc_report
        else:
            st.session_state.plot_id_merged = None
            st.session_state.calc_report = {}
    
    # Merge taxa information. coordinates_df's 'keyword' column duplicates
    # node_key (species_key), so match it against taxa_df's species_key.
    with st.spinner("Preparing taxa table..."):
        if taxa_df is not None and 'species_key' in taxa_df.columns:
            taxa_merged = pd.merge(coordinates_df, taxa_df, left_on='keyword', right_on='species_key', how='left', suffixes=('', '_taxa'))
        else:
            taxa_merged = coordinates_df
        st.session_state.taxa_merged = taxa_merged
    
    # Validate if requested
    if validate_data:
        with st.spinner("Validating data quality..."):
            validation_issues = validate_map_data(
                st.session_state.taxa_merged,
                st.session_state.plot_id_merged,
                st.session_state.species_merged
            )
            st.session_state.validation_issues = validation_issues
    else:
        st.session_state.validation_issues = []
    
    st.success("✅ Processing complete!")
    st.rerun()

# Display results if processed
if st.session_state.coordinates_df is not None:
    
    st.markdown("---")
    st.markdown("### 📊 Processing results")
    
    merge_report = st.session_state.merge_report
    calc_report = st.session_state.calc_report
    
    # Quality Metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Taxa with coordinates", f"{len(st.session_state.coordinates_df):,}")
        if st.session_state.species_merged is not None:
            match_rate = merge_report.get('match_rate', 0)
            st.metric("Species match rate", f"{match_rate:.1f}%")
    
    with col2:
        if st.session_state.species_merged is not None:
            matched = merge_report.get('matched_species', 0)
            total = merge_report.get('total_species', 0)
            st.metric("Species matched", f"{matched}/{total}")
    
    with col3:
        if st.session_state.plot_id_merged is not None:
            plot_count = len(st.session_state.plot_id_merged)
            st.metric("Plots processed", f"{plot_count:,}")
            
            if 'speciesNum' in st.session_state.plot_id_merged.columns:
                avg_species = st.session_state.plot_id_merged['speciesNum'].mean()
                st.metric("Avg species/plot", f"{avg_species:.1f}")
    
    with col4:
        coord_cov = calc_report.get('coord_coverage', 0)
        st.metric("Coordinate coverage", f"{coord_cov:.1f}%")
        
        ellenberg = calc_report.get('ellenberg_available', [])
        if ellenberg:
            st.metric("Ellenberg indicators", len(ellenberg))
    
    # Merge & calculation details
    with st.expander("🔍 Processing Details", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Species Merge:**")
            st.write(f"Strategy: {merge_report.get('strategy_used', 'N/A')}")
            st.write(f"Matched: {merge_report.get('matched_species', 0):,}")
            st.write(f"Unmatched: {merge_report.get('unmatched_species', 0):,}")
        
        with col2:
            st.markdown("**Plot Calculation:**")
            st.write(f"Method: {calc_report.get('merge_method', 'N/A')}")
            st.write(f"Weighting: {plot_weighting}")
            if calc_report.get('columns_preserved', 0) > 0:
                st.write(f"Columns preserved: {calc_report['columns_preserved']}")
        
        if calc_report.get('warnings'):
            st.markdown("**Warnings:**")
            for warning in calc_report['warnings']:
                st.caption(f"⚠️ {warning}")
    
    # Validation results
    if st.session_state.validation_issues:
        with st.expander("✅ Validation Results", expanded=False):
            errors = [i for i in st.session_state.validation_issues if i['severity'] == 'error']
            warnings = [i for i in st.session_state.validation_issues if i['severity'] == 'warning']
            
            if errors:
                st.markdown("**Errors:**")
                for issue in errors:
                    st.error(f"🔴 {issue['message']}")
            
            if warnings:
                st.markdown("**Warnings:**")
                for issue in warnings:
                    st.warning(f"🟡 {issue['message']}")
    else:
        if validate_data and st.session_state.validation_issues is not None:
            st.success("✅ All validation checks passed")
    
    # Preview data
    st.markdown("---")
    st.markdown("### 👁️ Data Preview")
    
    preview_tabs = st.tabs(["Taxa", "Plots", "Species", "Validation"])
    
    with preview_tabs[0]:
        if st.session_state.taxa_merged is not None:
            st.markdown("**Taxa Coordinates:**")
            
            preview_rows = st.number_input("Preview rows:", 10, 500, 100, key="taxa_preview")
            st.dataframe(
                st.session_state.taxa_merged.head(preview_rows), 
                use_container_width=True,
                hide_index=True
            )
            st.caption(f"Showing {min(preview_rows, len(st.session_state.taxa_merged))} of {len(st.session_state.taxa_merged):,} taxa")
    
    with preview_tabs[1]:
        if st.session_state.plot_id_merged is not None:
            st.markdown("**Plot Summary:**")
            
            preview_rows = st.number_input("Preview rows:", 10, 500, 100, key="plot_preview")
            st.dataframe(
                st.session_state.plot_id_merged.head(preview_rows), 
                use_container_width=True,
                hide_index=True
            )
            st.caption(f"Showing {min(preview_rows, len(st.session_state.plot_id_merged))} of {len(st.session_state.plot_id_merged):,} plots")
            
            # Show columns
            with st.expander("📋 Available columns", expanded=False):
                st.write(list(st.session_state.plot_id_merged.columns))
    
    with preview_tabs[2]:
        if st.session_state.species_merged is not None:
            st.markdown("**Species List:**")
            
            col1, col2 = st.columns(2)
            with col1:
                show_matched = st.checkbox("Show matched", value=True, key="show_match")
            with col2:
                show_unmatched = st.checkbox("Show unmatched", value=True, key="show_unmatch")
            
            # Filter species
            if 'xcoor' in st.session_state.species_merged.columns:
                if show_matched and not show_unmatched:
                    species_display = st.session_state.species_merged[
                        st.session_state.species_merged['xcoor'].notna()
                    ]
                elif show_unmatched and not show_matched:
                    species_display = st.session_state.species_merged[
                        st.session_state.species_merged['xcoor'].isna()
                    ]
                else:
                    species_display = st.session_state.species_merged
            else:
                species_display = st.session_state.species_merged
            
            preview_rows = st.number_input("Preview rows:", 10, 500, 100, key="species_preview")
            st.dataframe(species_display.head(preview_rows), use_container_width=True, hide_index=True)
            st.caption(f"Showing {min(preview_rows, len(species_display))} of {len(species_display):,} species")
            
            # Download unmatched species
            if 'xcoor' in st.session_state.species_merged.columns:
                species_no_coords = st.session_state.species_merged[
                    st.session_state.species_merged['xcoor'].isna()
                ]
                
                if len(species_no_coords) > 0:
                    st.markdown("---")
                    st.markdown(f"**📋 Species without coordinates: {len(species_no_coords):,}**")
                    
                    csv = species_no_coords.to_csv(index=False)
                    st.download_button(
                        label="📥 Download unmatched species",
                        data=csv,
                        file_name=f"species_without_coordinates_{datetime.date.today()}.csv",
                        mime="text/csv"
                    )
    
    with preview_tabs[3]:
        st.markdown("**Data Quality Report:**")
        
        if st.session_state.validation_issues:
            errors = [i for i in st.session_state.validation_issues if i['severity'] == 'error']
            warnings = [i for i in st.session_state.validation_issues if i['severity'] == 'warning']
            
            if errors:
                st.markdown("**Errors:**")
                for issue in errors:
                    st.error(f"🔴 {issue['message']}")
            
            if warnings:
                st.markdown("**Warnings:**")
                for issue in warnings:
                    st.warning(f"🟡 {issue['message']}")
        else:
            st.success("✅ No data quality issues detected")
        
        # Coverage statistics
        st.markdown("---")
        st.markdown("**Coordinate Coverage:**")
        
        coverage_data = []
        
        if st.session_state.taxa_merged is not None and 'xcoor' in st.session_state.taxa_merged.columns:
            taxa_cov = st.session_state.taxa_merged['xcoor'].notna().sum() / len(st.session_state.taxa_merged) * 100
            coverage_data.append({"Dataset": "Taxa", "Coverage %": taxa_cov})
        
        if st.session_state.plot_id_merged is not None and 'xcoor' in st.session_state.plot_id_merged.columns:
            plot_cov = st.session_state.plot_id_merged['xcoor'].notna().sum() / len(st.session_state.plot_id_merged) * 100
            coverage_data.append({"Dataset": "Plots", "Coverage %": plot_cov})
        
        if st.session_state.species_merged is not None and 'xcoor' in st.session_state.species_merged.columns:
            species_cov = st.session_state.species_merged['xcoor'].notna().sum() / len(st.session_state.species_merged) * 100
            coverage_data.append({"Dataset": "Species", "Coverage %": species_cov})
        
        if coverage_data:
            coverage_df = pd.DataFrame(coverage_data)
            st.dataframe(coverage_df, use_container_width=True, hide_index=True)
    
    # Save section
    st.markdown("---")
    st.markdown("### 💾 Save map database")
    
    # Determine output path
    if 'overlay_maps_path' in st.session_state and st.session_state['overlay_maps_path']:
        output_path = Path(st.session_state['overlay_maps_path'])
    elif 'project_base_path' in st.session_state and st.session_state['project_base_path']:
        output_path = Path(st.session_state['project_base_path']) / 'overlay_maps'
    elif 'queries_path' in st.session_state:
        output_path = Path(st.session_state['queries_path']).parent / 'overlay_maps'
    else:
        output_path = Path('./overlay_maps')
    
    if not output_path.exists():
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            st.error(f"Could not create output directory: {str(e)}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        output_db_name = st.text_input(
            "Output database name:",
            value=f"map_{selected_query_db.replace('.db', '')}_{datetime.date.today().strftime('%Y%m%d')}",
            help="Name for the new map database (without .db)"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(f"📂 Saving to: `{output_path.name}/`")
    
    # Save button
    if st.button("💾 Save Map Database", type="primary", use_container_width=True):
            
            with st.spinner("Creating map database..."):
                try:
                    output_db_path = output_path / f"{output_db_name}.db"
                    conn = sqlite3.connect(str(output_db_path))
                    
                    tables_saved = []
                    
                    # Save taxa coordinates
                    st.session_state.taxa_merged.to_sql('taxa', conn, if_exists='replace', index=False)
                    tables_saved.append('taxa')
                    
                    # Save plot summary
                    if st.session_state.plot_id_merged is not None:
                        st.session_state.plot_id_merged.to_sql('plot_id', conn, if_exists='replace', index=False)
                        tables_saved.append('plot_id')
                    
                    # Save species list
                    if st.session_state.species_merged is not None:
                        st.session_state.species_merged.to_sql('localSpecies', conn, if_exists='replace', index=False)
                        tables_saved.append('localSpecies')
                    
                    # Save original overlay data
                    if 'data' in st.session_state.overlay_data:
                        st.session_state.overlay_data['data'].to_sql('data', conn, if_exists='replace', index=False)
                        tables_saved.append('data')
                    
                    # Create and save summary statistics
                    if create_summary_stats:
                        summary_stats = {
                            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'merge_strategy': st.session_state.merge_report.get('strategy_used', 'N/A'),
                            'total_species_in_query': st.session_state.merge_report.get('total_species', 0),
                            'species_with_coordinates': st.session_state.merge_report.get('matched_species', 0),
                            'species_without_coordinates': st.session_state.merge_report.get('unmatched_species', 0),
                            'species_match_rate': f"{st.session_state.merge_report.get('match_rate', 0):.1f}%",
                            'total_taxa': len(st.session_state.taxa_merged) if st.session_state.taxa_merged is not None else 0,
                            'total_plots': len(st.session_state.plot_id_merged) if st.session_state.plot_id_merged is not None else 0,
                            'plots_with_coordinates': st.session_state.plot_id_merged['xcoor'].notna().sum() if st.session_state.plot_id_merged is not None else 0,
                            'avg_species_per_plot': st.session_state.plot_id_merged['speciesNum'].mean() if st.session_state.plot_id_merged is not None and 'speciesNum' in st.session_state.plot_id_merged.columns else 0
                        }
                        summary_df = pd.DataFrame([summary_stats])
                        summary_df.to_sql('summary_statistics', conn, if_exists='replace', index=False)
                        tables_saved.append('summary_statistics')
                    
                    # Create metadata
                    metadata = pd.DataFrame([{
                        'creation_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'reference_database': selected_ref_db,
                        'query_database': selected_query_db,
                        'plot_weighting': plot_weighting,
                        'include_unmatched': include_unmatched,
                        'coord_precision': coord_precision,
                        'num_taxa': len(st.session_state.taxa_merged) if st.session_state.taxa_merged is not None else 0,
                        'num_plots': len(st.session_state.plot_id_merged) if st.session_state.plot_id_merged is not None else 0,
                        'num_local_species': len(st.session_state.species_merged) if st.session_state.species_merged is not None else 0,
                        'validation_issues': len(st.session_state.validation_issues),
                        'merge_strategy': st.session_state.merge_report.get('strategy_used', 'N/A'),
                        'calc_method': st.session_state.calc_report.get('merge_method', 'N/A'),
                        'ellenberg_indicators': ','.join(st.session_state.calc_report.get('ellenberg_available', [])),
                        'has_mahal_dist': 'mean_mahal_dist' in (st.session_state.plot_id_merged.columns if st.session_state.plot_id_merged is not None else [])
                    }])
                    metadata.to_sql('metadata', conn, if_exists='replace', index=False)
                    tables_saved.append('metadata')
                    
                    conn.close()
                    
                    # Store in session state
                    st.session_state.map_db_path = str(output_db_path)
                    
                    st.success(f"✅ **Map database created successfully!**")
                    
                    # Final summary
                    st.markdown("---")
                    st.markdown("### 📊 Database summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Taxa", f"{len(st.session_state.taxa_merged):,}")
                        st.metric("Plots", f"{len(st.session_state.plot_id_merged):,}" if st.session_state.plot_id_merged is not None else "0")
                    
                    with col2:
                        st.metric("Species", f"{len(st.session_state.species_merged):,}" if st.session_state.species_merged is not None else "0")
                        st.metric("Match rate", f"{st.session_state.merge_report.get('match_rate', 0):.1f}%")
                    
                    with col3:
                        file_size = output_db_path.stat().st_size / 1024 / 1024
                        st.metric("Database size", f"{file_size:.1f} MB")
                        st.metric("Tables created", len(tables_saved))
                    
                    with col4:
                        st.metric("Validation issues", len(st.session_state.validation_issues))
                        st.metric("Weighting", plot_weighting.capitalize())
                    
                    st.info(f"📁 Saved as: `{output_db_path.name}`")
                    st.caption(f"📂 Location: `{output_path}`")
                    
                except Exception as e:
                    st.error(f"❌ Error creating overlay database: {str(e)}")
                    import traceback
                    with st.expander("🔍 Error details"):
                        st.code(traceback.format_exc())

# Next steps
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if 'map_db_path' in st.session_state:
        st.success("✅ **Map preparation complete!**")
        st.info("➡️ Database ready for visualization and analysis")
    else:
        st.info("Process and save data to create map database")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d; font-size: 0.9em;'>
    EcoNetMap - Map Preparation Module
    </div>
    """, 
    unsafe_allow_html=True
)
