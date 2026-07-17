"""
EcoNetMap - Rare Species Coordinate Assignment
==============================================================
Assigns network coordinates to species excluded from main graph analysis
due to low occurrence thresholds. Uses Jaccard-weighted positioning based on
co-occurrence patterns with already-positioned core species.

Part of the EcoNetMap toolkit - Network-based Ecological Cartography
Author: Flemming Skov
Updated: January 2026

METHODOLOGY:
- Identifies missing species by comparing full dataset with network species
- Calculates Jaccard similarity with positioned species
- Uses only core/specialist species (top X% farthest from center) as references
- Assigns coordinates as Jaccard-weighted average of neighbor positions
- Includes quality metrics for assignment confidence
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
import math

# Import packages for type hints
from typing import Optional, Tuple, Dict, List, Set, Any

# Import packages for visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# Page configuration
st.set_page_config(
    page_title="Rare Species Assignment - EcoNetMap", 
    page_icon="🕸️",
    layout="wide"
)

# Title and progress indicator
col1, col2 = st.columns([4, 1])
with col1:
    st.header("Network enhancement")
    st.subheader("🎯 Rare species coordinate assignment")
    st.markdown("*Position rare species using Jaccard-weighted co-occurrence with core network species*")
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

def map_bearing(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate bearing from point 1 to point 2"""
    dx = x2 - x1
    dy = y2 - y1
    bearing = math.degrees(math.atan2(dy, dx))
    return (bearing + 360) % 360

@st.cache_data(show_spinner=False)
def load_reference_coordinates(db_path: str) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Load coordinate data from reference database"""
    try:
        conn = sqlite3.connect(db_path)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        
        info = {
            'tables': tables['name'].tolist(),
            'species_count': 0,
            'has_coordinates': False
        }
        
        # Try to find coordinate table
        coord_table = None
        if 'keyword_coordinates' in tables['name'].values:
            coord_table = 'keyword_coordinates'
        elif 'keyword_raw_coordinates' in tables['name'].values:
            coord_table = 'keyword_raw_coordinates'
        
        if coord_table:
            df = pd.read_sql_query(f'SELECT * FROM {coord_table}', conn)
            info['species_count'] = len(df)
            info['has_coordinates'] = True
            info['coordinate_table'] = coord_table
            
            # Get metadata if available
            if 'metadata' in tables['name'].values:
                metadata = pd.read_sql_query('SELECT * FROM metadata', conn)
                info['metadata'] = metadata.to_dict('records')[0] if len(metadata) > 0 else {}
            
            conn.close()
            return df, info
        else:
            conn.close()
            return None, info
            
    except Exception as e:
        return None, {'error': str(e)}

@st.cache_data(show_spinner=False)
def load_source_data(db_path: str) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Load data from the SAME database used for graph construction"""
    try:
        conn = sqlite3.connect(db_path)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        
        info = {
            'tables': tables['name'].tolist(),
            'record_count': 0,
            'species_count': 0,
            'plot_count': 0
        }
        
        # Load data table
        if 'data' in tables['name'].values:
            data = pd.read_sql_query('SELECT * FROM data', conn)
            info['record_count'] = len(data)
            info['species_count'] = data['species_key'].nunique() if 'species_key' in data.columns else 0
            info['plot_count'] = data['plot_id'].nunique()
            
            # Get metadata if available
            if 'metadata' in tables['name'].values:
                metadata = pd.read_sql_query('SELECT * FROM metadata', conn)
                info['metadata'] = metadata.to_dict('records')[0] if len(metadata) > 0 else {}
            
            conn.close()
            return data, info
        else:
            conn.close()
            return None, info
            
    except Exception as e:
        return None, {'error': str(e)}

def extract_species_list(data_df: pd.DataFrame, key_col: str = 'species_key') -> pd.DataFrame:
    """Extract unique species list from data, identified by key_col (species_key).
    Output's identity column is always named 'node_key' to match the reference
    coordinate table saved by network_layout; 'species_name' duplicates it since
    species_key is the only species identifier available."""
    species_list = data_df.groupby(key_col).agg({
        'plot_id': 'nunique'
    }).reset_index()
    species_list.columns = ['node_key', 'plot_count']
    species_list['species_name'] = species_list['node_key']
    return species_list

def build_plots_by_species(data_df: pd.DataFrame, key_col: str = 'species_key') -> Dict[Any, set]:
    """Precompute {species_key: set(plot_ids)} once, so Jaccard comparisons look values up
    instead of re-scanning the full occurrence table for every (species, core species) pair."""
    return data_df.groupby(key_col)['plot_id'].apply(set).to_dict()

def compare_species_lists(source_species: pd.DataFrame, reference_coords: pd.DataFrame) -> Dict:
    """Compare species between source data and reference to find missing ones"""
    source_keys = set(source_species['node_key'].unique())
    reference_keys = set(reference_coords['node_key'].unique())

    missing_in_reference = source_keys - reference_keys

    missing_species_df = source_species[source_species['node_key'].isin(missing_in_reference)].copy()
    missing_species_df = missing_species_df.sort_values('plot_count', ascending=False)

    comparison = {
        'source_total': len(source_keys),
        'reference_total': len(reference_keys),
        'overlap': len(source_keys & reference_keys),
        'missing_in_reference': len(missing_in_reference),
        'missing_species_df': missing_species_df
    }
    
    return comparison

def calculate_distance_from_center(coords_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate distance and bearing from center for each species"""
    df = coords_df.copy()
    
    # Distance from center (0.5, 0.5)
    df['distance'] = np.sqrt((df['xcoor'] - 0.5)**2 + (df['ycoor'] - 0.5)**2)
    
    # Bearing from center
    df['bearing'] = df.apply(lambda row: map_bearing(0.5, 0.5, row['xcoor'], row['ycoor']), axis=1)
    
    return df

def build_mahal_params(reference_coords: pd.DataFrame) -> Dict:
    """
    Build Mahalanobis distance parameters from all reference species.
    Returns centroid (weighted by wdegree or occurrence_count) and
    inverse covariance matrix. Used to assign consistent mahal_dist
    to both core and newly added rare species.
    """
    valid = reference_coords.dropna(subset=['xcoor', 'ycoor'])
    coords = valid[['xcoor', 'ycoor']].values

    try:
        cov = np.cov(coords.T)
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return {'valid': False}

    if 'wdegree' in valid.columns and valid['wdegree'].sum() > 0:
        w = valid['wdegree'].fillna(0).values
    elif 'occurrence_count' in valid.columns and valid['occurrence_count'].sum() > 0:
        w = valid['occurrence_count'].fillna(0).values
    else:
        w = np.ones(len(valid))

    cx = np.average(coords[:, 0], weights=w)
    cy = np.average(coords[:, 1], weights=w)

    return {'valid': True, 'centroid': np.array([cx, cy]), 'cov_inv': cov_inv}

def mahal_dist_for_point(x: float, y: float, mahal_params: Dict) -> float:
    """Calculate Mahalanobis distance for a single point given prebuilt params"""
    if not mahal_params.get('valid', False):
        return math.hypot(x - 0.5, y - 0.5)  # fallback to geometric
    v = np.array([x, y]) - mahal_params['centroid']
    return float(np.sqrt(np.dot(np.dot(v, mahal_params['cov_inv']), v)))

def calculate_jaccard_similarity(species_plots: set, positioned_plots: set) -> float:
    """Calculate Jaccard similarity between two species' plot sets (intersection / union)"""
    intersection = len(species_plots & positioned_plots)
    union = len(species_plots | positioned_plots)

    if union > 0:
        return intersection / union
    return 0.0

def assign_coordinates_to_species(species_key_value,
                                  plots_by_species: Dict[Any, set],
                                  reference_coords: pd.DataFrame,
                                  distance_percentile: float = 0.7,
                                  min_jaccard: float = 0.025,
                                  max_neighbors: int = 10,
                                  random_seed: Optional[int] = None,
                                  add_jitter: bool = True,
                                  mahal_params: Optional[Dict] = None) -> Dict:
    """
    Assign coordinates using Jaccard-weighted average of core species positions

    Parameters:
    - species_key_value: Species to assign coordinates to
    - plots_by_species: precomputed {species_key: set(plot_ids)} lookup (see build_plots_by_species) --
      avoids re-scanning the full occurrence table for every (species, core species) pair
    - reference_coords: Species already positioned in network (identified by 'node_key')
    - distance_percentile: Use species above this distance percentile (0.7 = top 30% farthest)
    - min_jaccard: Minimum Jaccard similarity required
    - max_neighbors: Maximum neighbor species to use
    - random_seed: Seed for reproducible jitter
    - add_jitter: Add small random offset to prevent exact overlaps
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Find plots where this species occurs
    species_plots = plots_by_species.get(species_key_value, set())

    if len(species_plots) == 0:
        return {'success': False, 'reason': 'No occurrence data'}

    # Use mahal_dist if available (ecologically meaningful); fall back to geometric distance
    dist_col = 'mahal_dist' if 'mahal_dist' in reference_coords.columns else 'distance'

    # Calculate distance threshold based on percentile
    distance_threshold = reference_coords[dist_col].quantile(distance_percentile)

    # Get core species (those far from density centre = specialists)
    core_species = reference_coords[reference_coords[dist_col] >= distance_threshold]

    if len(core_species) == 0:
        return {'success': False, 'reason': f'No core species at {distance_percentile:.0%} percentile'}

    # Calculate Jaccard with each core species
    neighbors = []
    for _, species_data in core_species.iterrows():
        positioned_plots = plots_by_species.get(species_data['node_key'], set())
        jaccard = calculate_jaccard_similarity(species_plots, positioned_plots)

        if jaccard >= min_jaccard:
            neighbors.append({
                'node_key': species_data['node_key'],
                'xcoor': species_data['xcoor'],
                'ycoor': species_data['ycoor'],
                'jaccard': jaccard,
                'distance': species_data['distance']
            })
    
    if len(neighbors) == 0:
        return {'success': False, 'reason': f'No core species with Jaccard ≥ {min_jaccard}'}
    
    # Sort by Jaccard and take top N
    neighbors = sorted(neighbors, key=lambda x: x['jaccard'], reverse=True)[:max_neighbors]
    
    # Calculate Jaccard-weighted average position
    total_jaccard = sum(n['jaccard'] for n in neighbors)
    x_weighted = sum(n['xcoor'] * n['jaccard'] for n in neighbors) / total_jaccard
    y_weighted = sum(n['ycoor'] * n['jaccard'] for n in neighbors) / total_jaccard
    
    # Quality metrics
    jaccard_values = [n['jaccard'] for n in neighbors]
    avg_jaccard = np.mean(jaccard_values)
    std_jaccard = np.std(jaccard_values)
    
    # Coordinate variance (stability measure)
    x_coords = [n['xcoor'] for n in neighbors]
    y_coords = [n['ycoor'] for n in neighbors]
    coord_variance = np.var(x_coords) + np.var(y_coords)
    
    # Add jitter if requested
    if add_jitter:
        x_final = x_weighted + np.random.normal(0, 0.01)
        y_final = y_weighted + np.random.normal(0, 0.01)
    else:
        x_final = x_weighted
        y_final = y_weighted
    
    # Ensure within bounds
    x_final = np.clip(x_final, 0, 1)
    y_final = np.clip(y_final, 0, 1)
    
    # Calculate distance and bearing for assigned position
    distance_final = math.hypot(x_final - 0.5, y_final - 0.5)
    bearing_final = map_bearing(0.5, 0.5, x_final, y_final)
    mahal_final = mahal_dist_for_point(x_final, y_final, mahal_params) if mahal_params else distance_final
    
    return {
        'success': True,
        'xcoor': x_final,
        'ycoor': y_final,
        'distance': distance_final,
        'bearing': bearing_final,
        'mahal_dist': mahal_final,
        'neighbors_used': len(neighbors),
        'neighbor_keys': [n['node_key'] for n in neighbors],
        'avg_jaccard': avg_jaccard,
        'std_jaccard': std_jaccard,
        'max_jaccard': max(jaccard_values),
        'min_jaccard': min(jaccard_values),
        'coord_variance': coord_variance,
        'distance_threshold': distance_threshold
    }

def create_comparison_plot(reference_coords: pd.DataFrame,
                          new_assignments: pd.DataFrame,
                          distance_threshold: float) -> plt.Figure:
    """Create visualization of reference species and new assignments"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Use mahal_dist if available, fall back to geometric distance
    dist_col = 'mahal_dist' if 'mahal_dist' in reference_coords.columns else 'distance'
    dist_label = 'Mahalanobis Distance' if dist_col == 'mahal_dist' else 'Distance from Center'

    # Normalise for marker sizing (both columns have different scales)
    dist_max = reference_coords[dist_col].max()
    dist_norm = reference_coords[dist_col] / dist_max if dist_max > 0 else reference_coords[dist_col]

    # Left plot: Reference species by distance
    ax1.set_title(f"Reference Species Network\n(colored by {dist_label})", fontsize=14)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect('equal')

    # Add guide circles
    for radius in [0.125, 0.25, 0.375, 0.5]:
        circle = Circle((0.5, 0.5), radius, linewidth=0.5, color='gray', fill=False, alpha=0.5)
        ax1.add_patch(circle)

    # Add grid lines
    ax1.add_line(Line2D([0.5, 0.5], [0, 1], color='gray', linewidth=0.5, alpha=0.5))
    ax1.add_line(Line2D([0, 1], [0.5, 0.5], color='gray', linewidth=0.5, alpha=0.5))

    # Plot reference species
    scatter1 = ax1.scatter(
        reference_coords['xcoor'],
        reference_coords['ycoor'],
        s=50 + 100 * dist_norm,
        c=reference_coords[dist_col],
        cmap='viridis',
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )

    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label(dist_label, rotation=270, labelpad=20)

    # Right plot: All species
    ax2.set_title("Complete Species Map\n(with rare species added)", fontsize=14)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_aspect('equal')

    # Add guide circles
    for radius in [0.125, 0.25, 0.375, 0.5]:
        circle = Circle((0.5, 0.5), radius, linewidth=0.5, color='gray', fill=False, alpha=0.5)
        ax2.add_patch(circle)

    # Note: no threshold circle drawn — Mahalanobis distance threshold does not
    # correspond to a circle in coordinate space (it follows the covariance ellipse)

    # Add grid lines
    ax2.add_line(Line2D([0.5, 0.5], [0, 1], color='gray', linewidth=0.5, alpha=0.5))
    ax2.add_line(Line2D([0, 1], [0.5, 0.5], color='gray', linewidth=0.5, alpha=0.5))

    # Plot reference species (core vs peripheral based on dist_col)
    core_species = reference_coords[reference_coords[dist_col] >= distance_threshold]
    peripheral_species = reference_coords[reference_coords[dist_col] < distance_threshold]
    
    if len(peripheral_species) > 0:
        ax2.scatter(
            peripheral_species['xcoor'], 
            peripheral_species['ycoor'],
            c='lightblue',
            s=20,
            alpha=0.4,
            label=f'Peripheral species ({len(peripheral_species)})'
        )
    
    if len(core_species) > 0:
        ax2.scatter(
            core_species['xcoor'], 
            core_species['ycoor'],
            c='steelblue',
            s=65,
            edgecolors='black',
            alpha=0.7,
            linewidths=0.5,
            label=f'Core species ({len(core_species)})'
        )
    
    # Plot new assignments
    if len(new_assignments) > 0:
        successful = new_assignments[new_assignments['assigned'] == True]
        if len(successful) > 0:
            ax2.scatter(
                successful['xcoor'], 
                successful['ycoor'],
                c='red',
                s=80,
                alpha=0.8,
                marker='^',
                edgecolors='darkred',
                linewidths=1.5,
                label=f'Newly assigned ({len(successful)})'
            )
    
    ax2.legend(loc='upper right', fontsize=10)
    
    # Add labels
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xlabel('X coordinate', fontsize=10)
        ax.set_ylabel('Y coordinate', fontsize=10)
    
    # Add threshold annotation
    percentile = (1 - (reference_coords[dist_col] >= distance_threshold).sum() / len(reference_coords)) * 100
    ax2.text(0.02, 0.98,
             f'{dist_label} threshold: {distance_threshold:.3f}\n({100-percentile:.0f}% of species used as references)',
             transform=ax2.transAxes,
             verticalalignment='top',
             fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def save_figure(fig: plt.Figure, filename: str, output_dir: Path) -> bool:
    """Save matplotlib figure to file"""
    try:
        output_path = output_dir / f"{filename}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        return True
    except Exception as e:
        st.error(f"Error saving figure: {e}")
        return False

def check_for_duplicates(conn: sqlite3.Connection, coord_table: str,
                        new_species_keys: List) -> List:
    """Check for existing node_keys to prevent duplicates"""
    try:
        if len(new_species_keys) == 0:
            return []
        placeholders = ','.join('?' for _ in new_species_keys)
        existing_query = f"SELECT DISTINCT node_key FROM {coord_table} WHERE node_key IN ({placeholders})"
        existing_keys = pd.read_sql_query(existing_query, conn, params=list(new_species_keys))['node_key'].tolist()
        return existing_keys
    except Exception:
        return []

# SESSION STATE INITIALIZATION
###################################################################################

if 'ref_coords' not in st.session_state:
    st.session_state.ref_coords = None
if 'source_data' not in st.session_state:
    st.session_state.source_data = None
if 'comparison' not in st.session_state:
    st.session_state.comparison = None
if 'assignments_df' not in st.session_state:
    st.session_state.assignments_df = None

# MAIN INTERFACE - 3 TABS
###################################################################################

tab1, tab2 = st.tabs([
    "📁 Load & Compare Databases",
    "🎯 Assign, Review & Save"
])

# TAB 1: LOAD & COMPARE
###################################################################################

with tab1:
    st.markdown("### 📁 Load databases and compare species lists")
    
    st.markdown("#### 🗄️ Source database")
    st.markdown("*The same database used for network construction in Step 1*")
    
    # Try to get path from previous step
    if 'enhanced_db_path' in st.session_state:
        source_db_path = Path(st.session_state['enhanced_db_path'])
        st.success(f"✅ Using source database from previous step: `{source_db_path.name}`")
    elif 'queries_path' in st.session_state:
        queries_path = Path(st.session_state['queries_path'])
        if queries_path.exists():
            query_db_files = sorted([f.name for f in queries_path.glob("*.db")])
            if query_db_files:
                selected_source_db = st.selectbox(
                    "Select source database:",
                    options=query_db_files,
                    help="The database used to create the network graph"
                )
                source_db_path = queries_path / selected_source_db
            else:
                st.error("No database files found in queries directory")
                st.stop()
        else:
            st.error("Queries directory not found")
            st.stop()
    else:
        st.error("No source database path available. Please run graph construction first.")
        st.stop()
    
    st.markdown("---")
    st.markdown("#### 🗺️ Reference coordinate database")
    st.markdown("*The network coordinates created in Step 2*")
    
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
        st.stop()
    
    ref_db_files = sorted([f.name for f in default_output_path.glob("*.db")])
    
    if not ref_db_files:
        st.error(f"No database files found in: {default_output_path}")
        st.stop()
    
    # Auto-select if coming from previous step
    default_ref_idx = 0
    if 'coordinates_db_path' in st.session_state:
        default_name = Path(st.session_state.coordinates_db_path).name
        if default_name in ref_db_files:
            default_ref_idx = ref_db_files.index(default_name)
    
    selected_ref_db = st.selectbox(
        "Select reference coordinate database:",
        options=ref_db_files,
        index=default_ref_idx,
        help="Database containing network coordinates from Step 2"
    )
    ref_db_path = default_output_path / selected_ref_db
    
    st.caption(f"📂 Loading from: `{default_output_path}`")
    
    # Load button
    if st.button("📊 Load and Analyze Databases", type="primary", use_container_width=True):
        
        # Load reference coordinates
        with st.spinner("Loading reference coordinates..."):
            ref_coords, ref_info = load_reference_coordinates(str(ref_db_path))
            
            if ref_coords is None:
                st.error(f"Failed to load reference coordinates: {ref_info.get('error', 'No coordinate table found')}")
                st.stop()
            
            # Calculate distances from center
            ref_coords = calculate_distance_from_center(ref_coords)
            st.session_state.ref_coords = ref_coords
            st.session_state.ref_info = ref_info
            st.session_state.ref_db_path = str(ref_db_path)
        
        # Load source data
        with st.spinner("Loading source data..."):
            source_data, source_info = load_source_data(str(source_db_path))
            
            if source_data is None:
                st.error(f"Failed to load source data: {source_info.get('error', 'No data table found')}")
                st.stop()
            
            st.session_state.source_data = source_data
            st.session_state.source_info = source_info
            st.session_state.source_db_path = str(source_db_path)
        
        # Extract species list
        with st.spinner("Extracting species list..."):
            source_species = extract_species_list(source_data, key_col='species_key')
            st.session_state.source_species = source_species
        
        # Compare species lists
        with st.spinner("Comparing species lists..."):
            comparison = compare_species_lists(source_species, ref_coords)
            st.session_state.comparison = comparison
        
        st.success("✅ Databases loaded and analyzed successfully!")
        st.rerun()
    
    # Display results if loaded
    if st.session_state.ref_coords is not None:
        
        st.markdown("---")
        st.markdown("### 📊 Database information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Reference Database (network coordinates):**")
            st.metric("Species with coordinates", f"{st.session_state.ref_info['species_count']:,}")
            if 'metadata' in st.session_state.ref_info:
                meta = st.session_state.ref_info['metadata']
                if 'creation_date' in meta:
                    st.caption(f"Created: {meta['creation_date']}")
        
        with col2:
            st.markdown("**Source Database (vegetation data):**")
            st.metric("Unique species", f"{st.session_state.source_info['species_count']:,}")
            st.metric("Unique plots", f"{st.session_state.source_info['plot_count']:,}")
        
        # Comparison results
        st.markdown("---")
        st.markdown("### 🔍 Species comparison")
        
        comparison = st.session_state.comparison
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Source species", f"{comparison['source_total']:,}")
        
        with col2:
            st.metric("Network species", f"{comparison['reference_total']:,}")
        
        with col3:
            st.metric("Overlap", f"{comparison['overlap']:,}")
        
        with col4:
            st.metric("Missing (rare)", f"{comparison['missing_in_reference']:,}")
        
        # Show missing species
        if comparison['missing_in_reference'] > 0:
            with st.expander(f"🔍 Rare species needing coordinates ({comparison['missing_in_reference']:,})", expanded=False):
                missing_df = comparison['missing_species_df']
                st.dataframe(
                    missing_df[['node_key', 'species_name', 'plot_count']],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Summary statistics
                st.markdown("**Occurrence summary:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max plots", int(missing_df['plot_count'].max()))
                with col2:
                    st.metric("Mean plots", f"{missing_df['plot_count'].mean():.1f}")
                with col3:
                    st.metric("Min plots", int(missing_df['plot_count'].min()))
        else:
            st.success("✅ All species from source data already have coordinates!")

# TAB 2: ASSIGN COORDINATES
###################################################################################

with tab2:
    st.markdown("### 🎯 Assign coordinates to rare species")
    
    if st.session_state.comparison is None:
        st.info("👈 Please load databases in Tab 1 first")
    
    elif st.session_state.comparison['missing_in_reference'] == 0:
        st.success("✅ All species already have coordinates! No assignment needed.")
    
    else:
        # Parameters section
        with st.expander("ℹ️ How does coordinate assignment work?", expanded=False):
            st.markdown("""
            **Methodology:**
            
            1. **Identify core species:** Select positioned species that are far from center (specialists, not generalists)
            2. **Calculate Jaccard similarity:** Measure association strength between rare species and each core species
            3. **Filter by minimum Jaccard:** Exclude very weak associations
            4. **Weight by Jaccard:** Strong associations influence position more than weak ones
            5. **Calculate weighted average:** Final position is Jaccard-weighted average of neighbor positions
            6. **Add optional jitter:** Small random offset prevents exact overlaps
            
            **Why use core species only?**
            - Generalist species near center are less informative about habitat preferences
            - Specialist species at periphery provide stronger ecological signal
            - Distance percentile controls how selective we are (70% = top 30% farthest species)
            
            **Quality metrics:**
            - Average Jaccard: Mean association strength with neighbors
            - Coordinate variance: How spread out the neighbors are (lower = more stable)
            - Number of neighbors: How many core species used for positioning
            """)
        
        st.markdown("---")
        st.markdown("#### ⚙️ Assignment parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Core Species Selection:**")
            
            distance_percentile = st.slider(
                "Distance percentile threshold:",
                min_value=0.50,
                max_value=0.95,
                value=0.70,
                step=0.05,
                help="Use species above this percentile (0.70 = top 30% farthest from center)"
            )
            
            # Show what this means
            if st.session_state.ref_coords is not None:
                _rc = st.session_state.ref_coords
                _dist_col = 'mahal_dist' if 'mahal_dist' in _rc.columns else 'distance'
                threshold = _rc[_dist_col].quantile(distance_percentile)
                core_count = (_rc[_dist_col] >= threshold).sum()
                total_count = len(_rc)
                pct_used = (core_count / total_count) * 100
                _dist_label = 'Mahalanobis' if _dist_col == 'mahal_dist' else 'Distance'

                st.caption(f"💡 {core_count:,}/{total_count:,} species ({pct_used:.0f}%) will be used as references")
                st.caption(f"📏 {_dist_label} threshold: {threshold:.3f}")
        
        with col2:
            st.markdown("**Jaccard Filtering:**")
            
            min_jaccard = st.number_input(
                "Minimum Jaccard similarity:",
                min_value=0.001,
                max_value=0.5,
                value=0.025,
                step=0.005,
                format="%.3f",
                help="Exclude very weak associations (0.025 is very permissive)"
            )
            
            max_neighbors = st.number_input(
                "Max neighbors to use:",
                min_value=3,
                max_value=50,
                value=10,
                help="Maximum number of core species to include in weighted average"
            )
        
        with col3:
            st.markdown("**Reproducibility:**")
            
            use_seed = st.checkbox(
                "Use fixed seed",
                value=True,
                help="For reproducible assignments"
            )
            
            if use_seed:
                random_seed = st.number_input(
                    "Random seed:",
                    min_value=0,
                    max_value=999999,
                    value=42,
                    help="Same seed = same coordinates"
                )
            else:
                random_seed = None
            
            add_jitter = st.checkbox(
                "Add random jitter",
                value=True,
                help="Small offset prevents exact overlaps"
            )
        
        # Run assignment
        if st.button("🎯 Assign Coordinates to Rare Species", type="primary", use_container_width=True):
            
            # Show distance distribution
            with st.spinner("Analyzing species distribution..."):
                ref_coords = st.session_state.ref_coords
                _dist_col = 'mahal_dist' if 'mahal_dist' in ref_coords.columns else 'distance'
                _dist_label = 'Mahalanobis Distance from Density Centre' if _dist_col == 'mahal_dist' else 'Distance from Center (0=center, ~0.7=edge)'
                threshold = ref_coords[_dist_col].quantile(distance_percentile)

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(ref_coords[_dist_col], bins=40, alpha=0.7, edgecolor='black', color='steelblue')
                ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                          label=f'Core species threshold: {threshold:.3f} ({100-distance_percentile*100:.0f}% used)')
                ax.set_xlabel(_dist_label, fontsize=10)
                ax.set_ylabel('Number of Species', fontsize=10)
                ax.set_title(f'Distribution of Species {_dist_label}', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            # Assign coordinates
            with st.spinner("Assigning coordinates to rare species..."):
                missing_species = st.session_state.comparison['missing_species_df']
                source_data = st.session_state.source_data
                species_key = 'species_key'
                assignments = []

                # Build Mahalanobis params once from all reference species
                mahal_params = build_mahal_params(ref_coords)

                # Precompute species -> plot set once, instead of re-scanning source_data
                # for every (missing species, core species) pair inside the loop below
                plots_by_species = build_plots_by_species(source_data, species_key)

                progress_bar = st.progress(0)
                progress_text = st.empty()

                for idx, (_, species) in enumerate(missing_species.iterrows()):
                    # Update progress
                    progress = (idx + 1) / len(missing_species)
                    progress_bar.progress(progress)
                    progress_text.text(f"Processing: {species['species_name']} ({idx + 1}/{len(missing_species)})")

                    # Assign coordinates
                    result = assign_coordinates_to_species(
                        species['node_key'],
                        plots_by_species,
                        ref_coords,
                        distance_percentile=distance_percentile,
                        min_jaccard=min_jaccard,
                        max_neighbors=max_neighbors,
                        random_seed=random_seed,
                        add_jitter=add_jitter,
                        mahal_params=mahal_params
                    )

                    # Store assignment
                    assignment = {
                        'node_key': species['node_key'],
                        'keyword': species['species_name'],
                        'label': species['species_name'],
                        'assigned': result['success'],
                        'plot_count': species['plot_count']
                    }
                    
                    if result['success']:
                        assignment.update({
                            'xcoor': result['xcoor'],
                            'ycoor': result['ycoor'],
                            'distance': result['distance'],
                            'bearing': result['bearing'],
                            'mahal_dist': result['mahal_dist'],
                            'neighbors_used': result['neighbors_used'],
                            'avg_jaccard': result['avg_jaccard'],
                            'std_jaccard': result['std_jaccard'],
                            'max_jaccard': result['max_jaccard'],
                            'min_jaccard': result['min_jaccard'],
                            'coord_variance': result['coord_variance'],
                            'distance_threshold': result['distance_threshold']
                        })
                    else:
                        assignment.update({
                            'xcoor': np.nan,
                            'ycoor': np.nan,
                            'distance': np.nan,
                            'bearing': np.nan,
                            'mahal_dist': np.nan,
                            'neighbors_used': 0,
                            'failure_reason': result['reason']
                        })
                    
                    assignments.append(assignment)
                
                progress_bar.progress(1.0)
                progress_text.text("✅ Assignment complete!")
                
                # Convert to dataframe
                assignments_df = pd.DataFrame(assignments)
                st.session_state.assignments_df = assignments_df
                
                # Store parameters
                st.session_state.assignment_params = {
                    'distance_percentile': distance_percentile,
                    'min_jaccard': min_jaccard,
                    'max_neighbors': max_neighbors,
                    'random_seed': random_seed,
                    'add_jitter': add_jitter,
                    'distance_threshold': threshold
                }
                
                # Summary
                successful = assignments_df['assigned'].sum()
                failed = len(assignments_df) - successful
                
                st.success(f"✅ Successfully assigned coordinates to {successful:,} rare species")
                if failed > 0:
                    st.warning(f"⚠️ Could not assign coordinates to {failed:,} species")
            
            # Create visualization
            with st.spinner("Creating visualization..."):
                fig = create_comparison_plot(
                    ref_coords,
                    assignments_df,
                    threshold
                )
                st.pyplot(fig)
                
                # Save option
                col1, col2 = st.columns([2, 1])
                with col1:
                    save_name = st.text_input(
                        "Save visualization as:", 
                        value=f"core_and_rare_species",
                        key="save_viz"
                    )
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("💾 Save Visualization", key="save_viz_btn"):
                        figures_path = Path(st.session_state.get('figures_path', default_output_path))
                        if save_figure(fig, save_name, figures_path):
                            st.success(f"✅ Saved as {save_name}.png")
                
                plt.close()
            
            # Show quality metrics
            st.markdown("---")
            st.markdown("#### 📊 Assignment quality metrics")
            
            successful_df = assignments_df[assignments_df['assigned'] == True]
            
            if len(successful_df) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Avg Jaccard", f"{successful_df['avg_jaccard'].mean():.3f}")
                    st.caption("Mean association strength")
                
                with col2:
                    st.metric("Avg neighbors", f"{successful_df['neighbors_used'].mean():.1f}")
                    st.caption("Core species used")
                
                with col3:
                    st.metric("Coord variance", f"{successful_df['coord_variance'].mean():.4f}")
                    st.caption("Position stability")
                
                with col4:
                    success_rate = (successful / len(assignments_df)) * 100
                    st.metric("Success rate", f"{success_rate:.0f}%")
                    st.caption("Assignments completed")
            
            # Show assignment details
            with st.expander("📋 Assignment Details", expanded=False):
                failed_df = assignments_df[assignments_df['assigned'] == False]
                
                tab1, tab2 = st.tabs(["✅ Successful", "❌ Failed"])
                
                with tab1:
                    if len(successful_df) > 0:
                        display_cols = ['node_key', 'keyword', 'plot_count', 'xcoor', 'ycoor', 
                                      'neighbors_used', 'avg_jaccard', 'coord_variance']
                        display_df = successful_df[display_cols].copy()
                        display_df['xcoor'] = display_df['xcoor'].round(4)
                        display_df['ycoor'] = display_df['ycoor'].round(4)
                        display_df['avg_jaccard'] = display_df['avg_jaccard'].round(3)
                        display_df['coord_variance'] = display_df['coord_variance'].round(5)
                        
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No successful assignments")
                
                with tab2:
                    if len(failed_df) > 0:
                        st.dataframe(
                            failed_df[['node_key', 'keyword', 'plot_count', 'failure_reason']],
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Show failure reasons summary
                        st.markdown("**Failure reasons:**")
                        reason_counts = failed_df['failure_reason'].value_counts()
                        for reason, count in reason_counts.items():
                            st.caption(f"• {reason}: {count} species")
                    else:
                        st.info("No failed assignments")

        # REVIEW & SAVE SECTION (Combined with assignment)
        ###################################################################################

        st.markdown("---")
        st.markdown("### 💾 Review & save results")

        if st.session_state.assignments_df is not None:
            assignments_df = st.session_state.assignments_df
            successful_assignments = assignments_df[assignments_df['assigned'] == True]

            if len(successful_assignments) == 0:
                st.warning("⚠️ No successful assignments to save")

            else:
                # Configuration summary
                st.markdown("---")
                st.markdown("#### ⚙️ Assignment configuration")
            
                params = st.session_state.assignment_params
            
                config_data = {
                    'Distance Percentile': f"{params['distance_percentile']:.0%}",
                    'Distance Threshold': f"{params['distance_threshold']:.3f}",
                    'Min Jaccard': f"{params['min_jaccard']:.3f}",
                    'Max Neighbors': str(params['max_neighbors']),
                    'Random Seed': str(params['random_seed']) if params['random_seed'] is not None else 'None',
                    'Jitter Applied': 'Yes' if params['add_jitter'] else 'No'
                }
            
                config_df = pd.DataFrame(list(config_data.items()), columns=['Parameter', 'Value'])
                st.dataframe(config_df, use_container_width=True, hide_index=True)
            
                # Save options
                st.markdown("---")
                st.markdown("#### 💾 Save options")
            
                col1, col2 = st.columns(2)
            
                with col1:
                    update_existing = st.checkbox(
                        "Update existing database",
                        value=True,
                        help="Add to existing coordinate database (unchecked = create new file)"
                    )
                
                    if not update_existing:
                        new_db_name = st.text_input(
                            "New database name:",
                            value=f"{Path(st.session_state.ref_db_path).stem}_with_rare.db",
                            help="Name for new database file"
                        )
                    else:
                        st.info(f"Will add to: `{Path(st.session_state.ref_db_path).name}`")
            
                with col2:
                    save_log = st.checkbox(
                        "Save assignment log",
                        value=True,
                        help="Save detailed assignment information and quality metrics"
                    )
                
                    st.markdown("**Summary:**")
                    st.metric("Species to add", f"{len(successful_assignments):,}")
            
                # Save button
                if st.button("💾 Save Enhanced Coordinates", type="primary", use_container_width=True):
                    try:
                        # Determine output path
                        if update_existing:
                            output_path = Path(st.session_state.ref_db_path)
                        else:
                            output_path = Path(st.session_state.ref_db_path).parent / new_db_name
                    
                        # Connect to database
                        conn = sqlite3.connect(str(output_path))
                    
                        # Get existing coordinate table name
                        coord_table = st.session_state.ref_info.get('coordinate_table', 'keyword_coordinates')
                    
                        # Check for duplicates
                        new_keys = successful_assignments['node_key'].tolist()
                        existing_keys = check_for_duplicates(conn, coord_table, new_keys)
                    
                        if existing_keys:
                            st.warning(f"⚠️ Found {len(existing_keys)} species already in database - skipping duplicates")
                            successful_assignments = successful_assignments[~successful_assignments['node_key'].isin(existing_keys)]
                        
                            if len(successful_assignments) == 0:
                                st.error("❌ All species already exist. Nothing to add.")
                                conn.close()
                                st.stop()
                    
                        # Get existing table structure
                        existing_cols = pd.read_sql_query(f"PRAGMA table_info({coord_table})", conn)
                        existing_col_names = existing_cols['name'].tolist()
                    
                        # Prepare new species data
                        new_species = successful_assignments.copy()
                    
                        # Fill missing columns with appropriate defaults
                        for col in existing_col_names:
                            if col not in new_species.columns:
                                if col in ['degree', 'wdegree', 'betweenness', 'closeness']:
                                    new_species[col] = 0  # Network metrics = 0 (not in original network)
                                elif col in ['leiden', 'multi_level']:
                                    new_species[col] = -1  # Community = -1 (indicates assigned species)
                                elif col == 'occurrence_count':
                                    new_species[col] = new_species['plot_count']
                                elif col not in ['xcoor', 'ycoor', 'distance', 'bearing', 'node_key', 'keyword', 'label']:
                                    new_species[col] = None
                    
                        # Ensure we only include columns that exist in the original table
                        cols_to_insert = [col for col in existing_col_names if col in new_species.columns]
                    
                        # Insert new species
                        new_species[cols_to_insert].to_sql(
                            coord_table, 
                            conn, 
                            if_exists='append', 
                            index=False
                        )
                    
                        st.success(f"✅ Added {len(new_species):,} rare species to coordinate table")
                    
                        # Save assignment log if requested
                        if save_log:
                            log_df = assignments_df.copy()
                            log_df['assignment_date'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            log_df['distance_percentile'] = params['distance_percentile']
                            log_df['distance_threshold'] = params['distance_threshold']
                            log_df['min_jaccard'] = params['min_jaccard']
                            log_df['max_neighbors'] = params['max_neighbors']
                            log_df['random_seed'] = params['random_seed']
                            log_df['jitter_applied'] = params['add_jitter']
                            log_df['source_database'] = Path(st.session_state.source_db_path).name
                        
                            log_df.to_sql(
                                'rare_species_assignments',
                                conn,
                                if_exists='replace',
                                index=False
                            )
                            st.info("📋 Saved assignment log with quality metrics")
                    
                        # Update metadata
                        metadata = pd.DataFrame([{
                            'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'update_type': 'rare_species_assignment',
                            'rare_species_added': len(new_species),
                            'source_database': Path(st.session_state.source_db_path).name,
                            'reference_database': Path(st.session_state.ref_db_path).name,
                            'assignment_method': 'jaccard_weighted_average',
                            'distance_percentile': params['distance_percentile'],
                            'distance_threshold': params['distance_threshold'],
                            'min_jaccard': params['min_jaccard'],
                            'max_neighbors': params['max_neighbors'],
                            'random_seed': params['random_seed'],
                            'jitter_applied': params['add_jitter'],
                            'duplicates_prevented': len(existing_keys) if existing_keys else 0,
                            'success_rate': (successful_assignments.shape[0] / assignments_df.shape[0]) * 100
                        }])
                    
                        # Check if metadata_updates table exists and has correct schema
                        tables = pd.read_sql_query(
                            "SELECT name FROM sqlite_master WHERE type='table'", conn
                        )['name'].values
                    
                        if 'metadata_updates' in tables:
                            # Check if schema matches
                            try:
                                existing_schema = pd.read_sql_query(
                                    "PRAGMA table_info(metadata_updates)", conn
                                )
                                existing_cols = set(existing_schema['name'].tolist())
                                new_cols = set(metadata.columns.tolist())
                            
                                # If schemas don't match, recreate table
                                if existing_cols != new_cols:
                                    conn.execute("DROP TABLE metadata_updates")
                                    metadata.to_sql('metadata_updates', conn, if_exists='replace', index=False)
                                else:
                                    # Schema matches, safe to append
                                    metadata.to_sql('metadata_updates', conn, if_exists='append', index=False)
                            except Exception:
                                # If any issue, recreate table
                                conn.execute("DROP TABLE metadata_updates")
                                metadata.to_sql('metadata_updates', conn, if_exists='replace', index=False)
                        else:
                            # Table doesn't exist, create it
                            metadata.to_sql('metadata_updates', conn, if_exists='replace', index=False)
                    
                        conn.close()
                    
                        # Final summary
                        st.markdown("---")
                        st.markdown("#### 📊 Final summary")
                    
                        col1, col2, col3, col4 = st.columns(4)
                    
                        with col1:
                            original_species = st.session_state.ref_info['species_count']
                            st.metric("Original species", f"{original_species:,}")
                    
                        with col2:
                            st.metric("Rare species added", f"{len(new_species):,}")
                    
                        with col3:
                            total_species = original_species + len(new_species)
                            st.metric("Total species", f"{total_species:,}")
                    
                        with col4:
                            if existing_keys:
                                st.metric("Duplicates skipped", f"{len(existing_keys):,}")
                            else:
                                st.metric("Database", "✅ Complete")
                    
                        st.success("✅ **Successfully saved enhanced coordinate database!**")
                    
                        if not update_existing:
                            st.info(f"📁 Created new file: `{new_db_name}`")
                    
                        # Update session state
                        st.session_state.enhanced_coordinates_path = str(output_path)
                    
                    except Exception as e:
                        st.error(f"❌ Error saving database: {str(e)}")
                        import traceback
                        with st.expander("🔍 Error details"):
                            st.code(traceback.format_exc())

# Next steps
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
        if 'enhanced_coordinates_path' in st.session_state:
            st.success("✅ **Rare species coordinate assignment complete!**")
            st.info("➡️ Ready to proceed to next step")
        elif st.session_state.comparison is not None and st.session_state.comparison['missing_in_reference'] == 0:
            st.success("✅ **All species already have coordinates!**")
            st.info("➡️ Ready to proceed to next step")
        else:
            st.info("Complete assignment in Tab 2 to proceed")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d; font-size: 0.9em;'>
    EcoNetMap - Rare Species Coordinate Assignment
    </div>
    """, 
    unsafe_allow_html=True
)
