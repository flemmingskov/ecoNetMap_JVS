"""
EcoNetMap - Network Analysis & Reference Map Generation
=======================================================
This module creates species co-occurrence networks from filtered vegetation data.
It calculates species associations using Jaccard similarity, builds network graphs
using igraph, and generates optimized 2D reference maps using either:
  - Fruchterman-Reingold layout (topology emphasis)
  - Multidimensional Scaling (distance preservation emphasis)

Part of the EcoNetMap toolkit (Graph Construction & Reference Map 1/3)
Author: Flemming Skov (fs@ecos.au.dk)
Last Updated: January 2026
"""

# Import packages for web applications
import streamlit as st

# Import packages for data manipulation and analysis
import pandas as pd
import numpy as np
import sqlite3

# Import packages for network analysis
import igraph as ig
import leidenalg as la

# Import packages for file and system operations
from pathlib import Path
import datetime
import random

# Import packages for mathematical operations
import math
from itertools import combinations
from sklearn.manifold import MDS
from scipy.optimize import minimize_scalar

# Import packages for type hints
from typing import Optional, Tuple, List, Dict

# Import packages for visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# Constants
DEFAULT_MIN_OCCURRENCES = 1
DEFAULT_MIN_JACCARD = 0.0
DEFAULT_SEED = 42
DEFAULT_FR_ITERATIONS = 2000
MIN_FR_ITERATIONS = 100
MAX_FR_ITERATIONS = 2000
DEFAULT_EXPANSION_FACTOR = 1.0
MIN_EXPANSION_FACTOR = 0.5
MAX_EXPANSION_FACTOR = 3.0
DEFAULT_EXPANSION_ITERATIONS = 1
MIN_EXPANSION_ITERATIONS = 1
MAX_EXPANSION_ITERATIONS = 10

# Concave (centre-priority) expansion constants
DEFAULT_EXPANSION_ALPHA = 0.6
MIN_EXPANSION_ALPHA = 0.1   # Very aggressive central expansion
MAX_EXPANSION_ALPHA = 0.99  # Nearly identity (just below 1.0)

# Page configuration
st.set_page_config(
    page_title="Network Analysis - EcoNetMap", 
    page_icon="🕸️",
    layout="wide"
)

# Custom CSS for consistent styling
st.markdown("""
<style>
    .stTextInput > label {
        font-weight: bold;
        color: #2c3e50;
    }
    .info-box {
        background-color: #e8f4f8;
        border: 1px solid #b8e0ea;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    div[data-testid="stExpander"] > details {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Title and progress indicator
col1, col2 = st.columns([4, 1])
with col1:
    st.header("Reference map generation")
    st.subheader("🕸️ Network-based ecological cartography")
    st.markdown("*Create 2D reference landscape from species co-occurrence patterns*")
with col2:
    pass

st.markdown("---")

# Validate required session state
if 'queries_path' not in st.session_state or not st.session_state.queries_path:
    st.error("⚠️ **Project paths not initialized.** Please configure your project on the home page first.")
    st.info("👉 Go to the **home page** to set up your project paths before using this module.")
    st.stop()

if 'reference_map_path' not in st.session_state or not st.session_state.reference_map_path:
    st.error("⚠️ **Reference maps path not initialized.** Please configure your project on the home page first.")
    st.info("👉 Go to the **home page** to set up your project paths before using this module.")
    st.stop()

# PERSISTENT NETWORK STATUS (Visible in both tabs)
###################################################################################
if 'graph_data' in st.session_state and st.session_state.graph_data is not None:
    with st.container():
        st.markdown("### 📊 Current network status")

        # Get network info
        g = st.session_state.graph_data
        edges_df = st.session_state.get('edges_df', pd.DataFrame())
        metrics = st.session_state.get('graph_metrics', {})

        # Compact summary - database and layout only
        col1, col2 = st.columns(2)
        col1.metric("📁 Database", st.session_state.get('selected_db', 'N/A'))
        col2.metric("🎯 Layout", st.session_state.get('layout_method', 'Not generated'))

        # Configuration details in expander
        with st.expander("⚙️ View Network Configuration", expanded=False):
            config_cols = st.columns(4)
            with config_cols[0]:
                st.caption("**Min Occurrences:**")
                st.write(st.session_state.get('min_occurrences', 'N/A'))
            with config_cols[1]:
                st.caption("**Min Jaccard:**")
                st.write(st.session_state.get('min_jaccard', 'N/A'))
            with config_cols[2]:
                st.caption("**Random Seed:**")
                st.write(st.session_state.get('layout_seed', 'Not set'))
            with config_cols[3]:
                st.caption("**FR Iterations:**")
                st.write(st.session_state.get('fr_iterations', 'N/A') if 'Fruchterman' in st.session_state.get('layout_method', '') else 'N/A')

        st.markdown("---")
else:
    st.info("💡 **Status:** No network created yet. Start in Tab 1 to load data and build your network.")
    st.markdown("---")

# UTILITY FUNCTIONS
###################################################################################

@st.cache_data(show_spinner=False)
def load_sqlite_data(db_path: str, table_name: str = 'data') -> Optional[pd.DataFrame]:
    """Load data from SQLite database with caching"""
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        return df
    except (sqlite3.Error, pd.errors.DatabaseError) as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_database_info(db_path: str) -> dict:
    """Get information about SQLite database"""
    try:
        with sqlite3.connect(db_path) as conn:
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
            try:
                metadata = pd.read_sql_query("SELECT * FROM metadata", conn).to_dict('records')[0]
            except (sqlite3.Error, pd.errors.DatabaseError, IndexError):
                metadata = {}
        return {'tables': tables['name'].tolist(), 'metadata': metadata}
    except (sqlite3.Error, pd.errors.DatabaseError) as e:
        return {'tables': [], 'metadata': {}}

def map_bearing(x: float, y: float, center_x: float, center_y: float) -> float:
    """Calculate the bearing (direction angle) from center to target point"""
    angle = math.degrees(math.atan2(y - center_y, x - center_x))
    bearing = (90 - angle) % 360
    return bearing

# NETWORK CREATION FUNCTIONS
###################################################################################

def create_species_nodes(df: pd.DataFrame, min_occurrences: int = 1, key_col: str = 'species_key') -> pd.DataFrame:
    """Create node list from species data with minimum occurrence filter.

    key_col identifies each species uniquely (species_key, the user-mapped
    identity column). The output's identity column is always renamed to
    'node_key' so downstream code doesn't need to know the original name.
    """
    species_counts = df.groupby(key_col).size().reset_index(name='occurrence_count')

    species_info = df[[key_col]].drop_duplicates().copy()
    species_info['label'] = species_info[key_col]

    species_nodes = species_info.merge(species_counts, on=key_col, how='left')

    original_count = len(species_nodes)
    species_nodes_filtered = species_nodes[species_nodes['occurrence_count'] >= min_occurrences].copy()
    filtered_count = original_count - len(species_nodes_filtered)

    if filtered_count > 0:
        st.info(f"📊 Filtered out {filtered_count} species with occurrence count < {min_occurrences}")

    species_nodes_filtered = species_nodes_filtered.rename(columns={key_col: 'node_key'})
    species_nodes_filtered = species_nodes_filtered.sort_values('node_key').reset_index(drop=True)
    return species_nodes_filtered

def create_co_occurrence_edges_jaccard(df: pd.DataFrame, valid_species: set,
                                       min_jaccard: float = 0.0, key_col: str = 'species_key') -> pd.DataFrame:
    """
    Create edge list using Jaccard similarity index instead of raw co-occurrence counts.

    Jaccard similarity normalizes for species prevalence:
    J(A,B) = |A ∩ B| / |A ∪ B|

    Where:
    - |A ∩ B| = number of plots where both species occur (co-occurrences)
    - |A ∪ B| = total plots where either species occurs (union)

    This gives proper weight to rare specialist co-occurrences vs common generalist pairs.
    key_col identifies each species (species_key, the user-mapped identity column).
    """
    # Filter to valid species
    df_filtered = df[df[key_col].isin(valid_species)].copy()

    with st.spinner("Calculating Jaccard similarity for species pairs..."):
        # Step 1: Count individual species occurrences (for union calculation)
        species_occurrences = df_filtered.groupby(key_col)['plot_id'].nunique().to_dict()

        # Step 2: Count co-occurrences (intersection)
        co_occurrence = df_filtered.groupby('plot_id')[key_col].unique()
        total_plots = len(co_occurrence)
        
        intersection_counts = {}
        
        if total_plots > 100:
            progress_bar = st.progress(0)
            progress_text = st.empty()
        
        for idx, (plot_id, species_list) in enumerate(co_occurrence.items()):
            if len(species_list) > 1:
                for sp1, sp2 in combinations(sorted(species_list), 2):
                    pair = (sp1, sp2)
                    intersection_counts[pair] = intersection_counts.get(pair, 0) + 1
            
            if total_plots > 100 and idx % 100 == 0:
                progress = idx / total_plots
                progress_bar.progress(progress)
                progress_text.text(f"Processing plot {idx}/{total_plots}")
        
        if total_plots > 100:
            progress_bar.progress(1.0)
            progress_text.text("Calculating Jaccard indices...")
        
        # Step 3: Calculate Jaccard similarity
        edges_jaccard = []
        for (sp1, sp2), intersection in intersection_counts.items():
            # Union = occurrences(A) + occurrences(B) - intersection
            union = species_occurrences[sp1] + species_occurrences[sp2] - intersection
            jaccard = intersection / union if union > 0 else 0
            
            if jaccard >= min_jaccard:
                edges_jaccard.append({
                    'source': sp1,
                    'target': sp2,
                    'weight': jaccard,
                    'co_occurrences': intersection
                })
        
        if total_plots > 100:
            progress_bar.empty()
            progress_text.empty()

    if edges_jaccard:
        edges_df = pd.DataFrame(edges_jaccard)
        return edges_df
    else:
        return pd.DataFrame(columns=['source', 'target', 'weight', 'co_occurrences'])

# GRAPH CONSTRUCTION FUNCTIONS
###################################################################################

def create_graph_from_data(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> Optional[ig.Graph]:
    """Create an igraph Graph object from node and edge dataframes"""
    try:
        g = ig.Graph()

        # Add vertices
        node_keys = nodes_df['node_key'].tolist()
        labels = nodes_df['label'].tolist()
        g.add_vertices(len(node_keys))

        # Add vertex attributes
        g.vs["label"] = labels
        g.vs["node_key"] = node_keys
        if 'occurrence_count' in nodes_df.columns:
            g.vs["occurrence_count"] = nodes_df['occurrence_count'].tolist()

        # Map node_keys to vertex indices
        node_key_to_index = {node_key: i for i, node_key in enumerate(node_keys)}

        # Convert edge node_keys to vertex indices
        edges_indices = []
        weights = []
        for _, edge in edges_df.iterrows():
            try:
                idx1 = node_key_to_index[edge['source']]
                idx2 = node_key_to_index[edge['target']]
                edges_indices.append((idx1, idx2))
                weights.append(edge['weight'])
            except KeyError:
                continue
        
        # Add edges and weights
        if edges_indices:
            g.add_edges(edges_indices)
            g.es["weight"] = weights

        return g

    except (KeyError, AttributeError, TypeError, ValueError) as e:
        st.error(f"Error creating graph: {str(e)}")
        return None

def calculate_graph_metrics(g: ig.Graph) -> Dict:
    """Calculate graph metrics and community detection"""
    metrics = {}
    
    # Basic metrics
    metrics['degree'] = g.degree()
    metrics['weighted_degree'] = g.strength(mode="ALL", loops=True, 
                                           weights=g.es['weight'] if g.ecount() > 0 else None)
    metrics['betweenness'] = g.betweenness()
    metrics['closeness'] = g.closeness()
    
    # Community detection only if there are edges
    if g.ecount() > 0:
        # Multilevel (Louvain)
        multi_partition = g.community_multilevel(weights=g.es['weight'])
        metrics['multi_level_membership'] = multi_partition.membership
        metrics['multi_level_clusters'] = len(multi_partition)
        metrics['multi_level_modularity'] = g.modularity(multi_partition.membership)
        
        # Leiden
        leiden_partition = la.find_partition(g, la.ModularityVertexPartition, 
                                            weights=g.es['weight'])
        metrics['leiden_membership'] = leiden_partition.membership
        metrics['leiden_clusters'] = len(leiden_partition)
        metrics['leiden_modularity'] = g.modularity(leiden_partition.membership)
    else:
        # No communities if no edges
        metrics['multi_level_membership'] = [0] * g.vcount()
        metrics['multi_level_clusters'] = 1
        metrics['multi_level_modularity'] = 0
        metrics['leiden_membership'] = [0] * g.vcount()
        metrics['leiden_clusters'] = 1
        metrics['leiden_modularity'] = 0
    
    return metrics

# LAYOUT GENERATION FUNCTIONS
###################################################################################

def create_layout_fruchterman_reingold(g: ig.Graph, iterations: int, metrics: Dict, 
                                       seed: Optional[int] = None) -> pd.DataFrame:
    """Create graph layout using Fruchterman-Reingold algorithm with proper seeding for reproducibility"""
    
    # CRITICAL: Set ALL random number generators for full reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        # igraph uses Python's random module internally, but we need to ensure
        # the same random state for initial layout positions
        ig.set_random_number_generator(random.Random(seed))
    
    # Create layout - igraph will now use seeded random for initial positions
    if g.ecount() > 0:
        layout = g.layout_fruchterman_reingold(weights=g.es['weight'], niter=iterations)
    else:
        layout = g.layout_random()
    
    # Extract and normalize coordinates
    coords = np.array(layout.coords)
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Normalize to 0-1 range
    x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x
    y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() != y.min() else y
    
    # Create dataframe
    df = pd.DataFrame({
        'keyword': g.vs["label"],
        'node_key': g.vs["node_key"],
        'xcoor': x_norm,
        'ycoor': y_norm,
        'degree': metrics['degree'],
        'wdegree': metrics['weighted_degree'],
        'betweenness': metrics['betweenness'],
        'closeness': metrics['closeness'],
        'multi_level': metrics['multi_level_membership'],
        'leiden': metrics['leiden_membership']
    })
    
    # Add occurrence count if available
    if 'occurrence_count' in g.vertex_attributes():
        df['occurrence_count'] = g.vs["occurrence_count"]
    
    # Sort by weighted degree
    df = df.sort_values("wdegree", ascending=False)
    
    # Calculate distance from center and bearing
    df['distance'] = df.apply(lambda row: math.hypot(row['xcoor'] - 0.5, row['ycoor'] - 0.5), axis=1)
    df['bearing'] = df.apply(lambda row: map_bearing(row['xcoor'], row['ycoor'], 0.5, 0.5), axis=1)
    
    return df


def create_layout_mds(g: ig.Graph, metrics: Dict,
                      seed: Optional[int] = None) -> pd.DataFrame:
    """
    Create graph layout using Multidimensional Scaling (MDS).
    
    MDS positions species to preserve graph-theoretic distances through the 
    Jaccard-weighted co-occurrence network:
    - Network edges represent Jaccard similarity (co-occurrence strength)
    - Shortest path distances capture both direct and indirect relationships
    - Species closely connected (few hops) are positioned near each other
    - Species with no path between them are positioned far apart
    - Optimizes global distance preservation rather than local topology
    
    This differs from Fruchterman-Reingold which emphasizes network topology
    and community structure over strict distance preservation.
    """
    
    n_species = g.vcount()
    
    # Calculate shortest path distances
    with st.spinner("Calculating shortest path distances for MDS..."):
        # Option A: Unweighted paths (each edge = 1 step)
        distances = np.array(g.shortest_paths(weights=None))
        
        # Option B: Weighted by inverse Jaccard (uncomment to use)
        # Higher Jaccard = shorter distance (stronger connection)
        # weights_inverted = [1.0 / max(w, 0.01) for w in g.es['weight']]
        # distances = np.array(g.shortest_paths(weights=weights_inverted))
        
        # Handle disconnected components
        # Replace infinite distances with maximum finite distance + 1
        finite_distances = distances[np.isfinite(distances)]
        if len(finite_distances) > 0:
            max_finite = np.max(finite_distances)
            distances[np.isinf(distances)] = max_finite + 1
            st.info(f"📊 Distance matrix: {n_species}×{n_species}, "
                   f"mean distance: {np.mean(finite_distances):.2f}, "
                   f"max distance: {max_finite:.0f}")
        else:
            max_finite = 1.0
            distances[:] = 1.0
            np.fill_diagonal(distances, 0.0)
            st.warning("⚠️ Fully disconnected graph — all distances set to 1.0")
    
    # Apply MDS with proper settings
    with st.spinner("Applying MDS layout..."):
        mds = MDS(n_components=2, 
                 dissimilarity='precomputed',
                 random_state=seed if seed is not None else 42, 
                 max_iter=300,
                 n_init=4,  # Multiple initializations to find best solution
                 eps=1e-6)  # Convergence tolerance
        
        mds_coords = mds.fit_transform(distances)
        
        # Normalize to 0-1 range
        x = mds_coords[:, 0]
        y = mds_coords[:, 1]
        x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x
        y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() != y.min() else y
        
        # Create dataframe
        df = pd.DataFrame({
            'keyword': g.vs["label"],
            'node_key': g.vs["node_key"],
            'xcoor': x_norm,
            'ycoor': y_norm,
            'degree': metrics['degree'],
            'wdegree': metrics['weighted_degree'],
            'betweenness': metrics['betweenness'],
            'closeness': metrics['closeness'],
            'multi_level': metrics['multi_level_membership'],
            'leiden': metrics['leiden_membership']
        })
        
        # Add occurrence count if available
        if 'occurrence_count' in g.vertex_attributes():
            df['occurrence_count'] = g.vs["occurrence_count"]
        
        # Sort by weighted degree
        df = df.sort_values("wdegree", ascending=False)
        
        # Calculate distance from center and bearing
        df['distance'] = df.apply(lambda row: math.hypot(row['xcoor'] - 0.5, row['ycoor'] - 0.5), axis=1)
        df['bearing'] = df.apply(lambda row: map_bearing(row['xcoor'], row['ycoor'], 0.5, 0.5), axis=1)
        
        st.success(f"✅ MDS layout created (final stress: {mds.stress_:.2f})")
    
    return df


def expand_nodes(x: np.ndarray, y: np.ndarray, expansion_factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """Expand nodes radially from center point"""
    dist_from_center = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
    dist_from_center = np.where(dist_from_center == 0, 0.0001, dist_from_center)
    dist_expanded = dist_from_center * expansion_factor
    max_dist = 0.5
    dist_expanded = np.minimum(dist_expanded, max_dist)
    
    angle = np.arctan2(y - 0.5, x - 0.5)
    x_new = 0.5 + dist_expanded * np.cos(angle)
    y_new = 0.5 + dist_expanded * np.sin(angle)
    
    x_new = np.clip(x_new, 0, 1)
    y_new = np.clip(y_new, 0, 1)
    
    return x_new, y_new

def expand_nodes_concave(x: np.ndarray, y: np.ndarray, alpha: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand nodes radially using a concave power transform.

    Species closer to the centre are displaced proportionally MORE than
    species near the periphery, directly addressing the dense central
    cluster that forms in both FR and MDS layouts.

    Unlike the original expand_nodes() which pushes peripheral species
    furthest, this function opens up the crowded centre while leaving
    well-spread peripheral specialists largely in place.

    Rank order of distances from centre is strictly preserved:
    no species can overtake another (monotonically increasing transform).

    Formula:
        d_norm     = d / 0.5                   # normalise to [0, 1]
        d_expanded = (d_norm ** alpha) * 0.5   # concave transform, rescale

    Parameters
    ----------
    x, y  : np.ndarray
        Species coordinates in [0, 1] x [0, 1].
    alpha : float, default 0.6
        Concavity parameter, must be in (0, 1).
        Lower values = more aggressive central opening.
        alpha = 0.5  → square root (strong)
        alpha = 0.6  → recommended default (moderate)
        alpha = 0.75 → mild expansion
        alpha → 1.0  → approaches identity (no change)

    Returns
    -------
    x_new, y_new : np.ndarray
        Expanded coordinates, still within [0, 1] x [0, 1].
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha:.3f}")

    # Distance from centre (0.5, 0.5)
    dist = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)

    # Avoid division by zero for species exactly at centre
    dist_safe = np.where(dist == 0, 1e-9, dist)

    # Normalise to [0, 1], apply concave power transform, rescale to [0, 0.5]
    dist_norm     = dist_safe / 0.5
    dist_expanded = (dist_norm ** alpha) * 0.5

    # Preserve radial angle exactly
    angle = np.arctan2(y - 0.5, x - 0.5)

    # Reconstruct coordinates
    x_new = 0.5 + dist_expanded * np.cos(angle)
    y_new = 0.5 + dist_expanded * np.sin(angle)

    # Clip to unit square (defensive — should not be needed)
    x_new = np.clip(x_new, 0, 1)
    y_new = np.clip(y_new, 0, 1)

    return x_new, y_new


def calculate_node_statistics(df: pd.DataFrame) -> dict:
    """Calculate statistics about node distribution"""
    distances = np.sqrt((df['xcoor'] - 0.5)**2 + (df['ycoor'] - 0.5)**2)
    return {
        'min_distance': distances.min(),
        'max_distance': distances.max(),
        'mean_distance': distances.mean(),
        'median_distance': np.median(distances),
        'nodes_in_center': len(distances[distances < 0.1]),
        'nodes_at_edge': len(distances[distances > 0.4])
    }

def optimize_expansion_factor(df: pd.DataFrame, criterion: str = 'max_spread') -> float:
    """
    Automatically find optimal expansion factor based on criterion.
    
    Criteria:
    - 'max_spread': Maximize average distance from center (moderate)
    - 'target_edge': Get 30% of nodes at edge (distance > 0.4)
    - 'minimize_center': Reduce nodes in center to <5%
    """
    
    def objective(factor):
        x_exp, y_exp = expand_nodes(df['xcoor'].values, df['ycoor'].values, factor)
        distances = np.sqrt((x_exp - 0.5)**2 + (y_exp - 0.5)**2)
        
        if criterion == 'max_spread':
            # Maximize mean distance but with penalty for extreme spreading
            mean_dist = np.mean(distances)
            edge_penalty = (np.sum(distances > 0.48) / len(distances)) * 0.5  # Penalize pushing to edge
            return -(mean_dist - edge_penalty)
        
        elif criterion == 'target_edge':
            # Target 30% of nodes at edge (more reasonable than 50%)
            edge_fraction = np.sum(distances > 0.4) / len(distances)
            return abs(edge_fraction - 0.30)
        
        elif criterion == 'minimize_center':
            # Get center occupancy to ~5% (more reasonable than minimizing)
            center_fraction = np.sum(distances < 0.1) / len(distances)
            return abs(center_fraction - 0.05)
        
        else:
            return 0
    
    # Optimize within more conservative bounds (was 0.5-3.0, now 0.8-2.0)
    result = minimize_scalar(objective, bounds=(0.8, 2.0), method='bounded')
    
    # Apply damping factor to avoid extreme values
    optimal = result.x
    damped = 1.0 + (optimal - 1.0) * 0.7  # Damp by 30%
    
    return damped


def optimize_expansion_alpha(df: pd.DataFrame, criterion: str = 'minimize_center') -> float:
    """
    Automatically find the optimal alpha for concave (centre-priority) expansion.

    Criteria:
    - 'minimize_center': Reduce nodes in centre (distance < 0.1) to ~5%
    - 'target_mean':     Push mean distance from centre to ~0.30
    """
    from scipy.optimize import minimize_scalar

    def objective(alpha):
        x_exp, y_exp = expand_nodes_concave(df['xcoor'].values, df['ycoor'].values, alpha)
        distances = np.sqrt((x_exp - 0.5)**2 + (y_exp - 0.5)**2)

        if criterion == 'minimize_center':
            center_fraction = np.sum(distances < 0.1) / len(distances)
            return abs(center_fraction - 0.05)
        elif criterion == 'target_mean':
            return abs(np.mean(distances) - 0.30)
        else:
            return 0

    result = minimize_scalar(objective, bounds=(0.1, 0.99), method='bounded')
    # Clamp to valid range
    return float(np.clip(result.x, MIN_EXPANSION_ALPHA, MAX_EXPANSION_ALPHA))

# VISUALIZATION FUNCTIONS
###################################################################################

def create_network_visualization(df: pd.DataFrame, title: str, 
                                 color_by: str = 'Fixed color',
                                 show_labels: bool = False,
                                 show_kde: bool = False,
                                 figsize: Tuple[int, int] = (10, 10)) -> plt.Figure:
    """Create network visualization with various options"""
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    
    # Draw guide circles
    for radius in [0.125, 0.25, 0.375, 0.5]:
        circle = Circle((0.5, 0.5), radius, linewidth=0.5, color='gray', 
                       fill=False, alpha=0.5)
        ax.add_patch(circle)
    
    # Draw guide lines
    ax.add_line(Line2D([0.5, 0.5], [0, 1], color='gray', linewidth=0.5, alpha=0.5))
    ax.add_line(Line2D([0, 1], [0.5, 0.5], color='gray', linewidth=0.5, alpha=0.5))
    ax.add_line(Line2D([0, 1], [0, 1], color='gray', linewidth=0.5, alpha=0.5))
    ax.add_line(Line2D([0, 1], [1, 0], color='gray', linewidth=0.5, alpha=0.5))
    
    # Prepare colors
    if color_by == "Fixed color":
        colors = 'steelblue'
    elif color_by == "Community (Leiden)":
        colors = df['leiden']
    elif color_by == "Community (Louvain)":
        colors = df['multi_level']
    else:  # Weighted degree
        colors = df['wdegree']
    
    # Plot nodes
    scatter = ax.scatter(
        df['xcoor'], 
        df['ycoor'],
        s=30,
        alpha=0.6, 
        c=colors,
        cmap='tab20' if color_by != "Fixed color" else None,
        edgecolors='darkgray', 
        linewidths=0.5
    )
    
    # Add KDE if requested
    if show_kde and len(df) > 1:
        sns.kdeplot(
            data=df, x='xcoor', y='ycoor',
            fill=True, alpha=0.2, cmap='Blues', 
            thresh=0.01, ax=ax
        )
    
    # Add labels if requested
    if show_labels and len(df) < 100:
        top_nodes = df.nlargest(min(10, len(df)), 'wdegree')
        for _, node in top_nodes.iterrows():
            ax.annotate(
                node['keyword'][:15], 
                (node['xcoor'], node['ycoor']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
            )
    
    ax.set_xlabel('X coordinate', fontsize=10, alpha=0.7)
    ax.set_ylabel('Y coordinate', fontsize=10, alpha=0.7)
    ax.set_title(title, fontsize=14, pad=20)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    return fig

# SESSION STATE INITIALIZATION
###################################################################################

if 'network_data' not in st.session_state:
    st.session_state.network_data = None
if 'nodes_df' not in st.session_state:
    st.session_state.nodes_df = None
if 'edges_df' not in st.session_state:
    st.session_state.edges_df = None
if 'graph_data' not in st.session_state:
    st.session_state.graph_data = None
if 'graph_metrics' not in st.session_state:
    st.session_state.graph_metrics = None
if 'df_fr_layout' not in st.session_state:
    st.session_state.df_fr_layout = None
if 'df_mds_layout' not in st.session_state:
    st.session_state.df_mds_layout = None
if 'df_current_layout' not in st.session_state:
    st.session_state.df_current_layout = None
if 'df_final' not in st.session_state:
    st.session_state.df_final = None

# MAIN INTERFACE - 2 TABS
###################################################################################

# st.tabs() has no way to remember which tab was active across a script rerun
# (every button click triggers one), so clicking "Generate Layout" in tab 2 would
# silently snap the view back to tab 1 even though it worked -- st.radio() with a
# key persists its selection across reruns, so we use it as a tab bar instead.
tab_labels = ["📁 1. Load & Configure Network", "🎯 2. Generate, Review & Save"]
if 'network_layout_active_tab' not in st.session_state:
    st.session_state.network_layout_active_tab = tab_labels[0]

active_tab = st.radio(
    "Navigation",
    tab_labels,
    key='network_layout_active_tab',
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("---")

# TAB 1: LOAD & CONFIGURE
###################################################################################
if active_tab == tab_labels[0]:
    st.markdown("### 📁 Load data & configure network")
    
    # Database selection
    st.markdown("#### Step 1: Select database")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        queries_path = Path(st.session_state.get('queries_path', '.'))
        if not queries_path.exists():
            st.error(f"Queries directory not found: {queries_path}")
            st.stop()
        
        db_files = sorted([f.name for f in queries_path.glob("*.db")])
        
        if not db_files:
            st.warning("No database files found in the queries directory.")
            st.info("Please complete Step 2 (Data Filtering) first.")
            st.stop()
        
        selected_db = st.selectbox(
            "Select filtered database:",
            options=db_files,
            help="Choose a database created in the filtering step"
        )
        
        db_path = queries_path / selected_db
    
    with col2:
        pass

    # Auto-load when selection changes
    if selected_db != st.session_state.get('selected_db'):
        with st.spinner(f"Loading {selected_db}..."):
            df = load_sqlite_data(str(db_path))
            if df is not None:
                st.session_state.network_data = df
                st.session_state.selected_db = selected_db
                st.session_state.db_load_metrics = {
                    'records': len(df),
                    'species': df['species_key'].nunique() if 'species_key' in df.columns else 'N/A',
                    'plots':   df['plot_id'].nunique() if 'plot_id' in df.columns else 'N/A',
                }

    # Show metrics whenever a database is loaded
    if st.session_state.get('db_load_metrics'):
        m = st.session_state.db_load_metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Records", f"{m['records']:,}" if isinstance(m['records'], int) else "N/A")
        col2.metric("Species", f"{m['species']:,}" if isinstance(m['species'], int) else "N/A")
        col3.metric("Plots",   f"{m['plots']:,}"   if isinstance(m['plots'],   int) else "N/A")
    
    # Network configuration (only if data is loaded)
    if st.session_state.network_data is not None:
        st.markdown("---")
        st.markdown("#### Step 2: Network parameters")
        
        st.markdown("""
        **Configure network construction parameters:**
        - **Min occurrences:** Remove rare species appearing in fewer plots (reduces noise)
        - **Min Jaccard:** Remove weak associations (0.0 = include all connections, higher = stricter)
        """)
        
        df = st.session_state.network_data

        species_key = 'species_key'

        col1, col2, col3 = st.columns(3)

        with col1:
            species_counts = df.groupby(species_key).size()
            max_occurrence = int(species_counts.max())
            median_occurrence = int(species_counts.median())
            
            min_occurrences = st.number_input(
                "Minimum species occurrences:",
                min_value=DEFAULT_MIN_OCCURRENCES,
                max_value=max_occurrence,
                value=max(DEFAULT_MIN_OCCURRENCES, median_occurrence // 2),
                help=f"Species must occur in at least this many plots. Max: {max_occurrence}, Median: {median_occurrence}. Start conservative (median/2)."
            )

            st.caption(f"💡 Recommendation: Start with {median_occurrence // 2}, increase to reduce network size")

        with col2:
            min_jaccard = st.number_input(
                "Minimum Jaccard similarity:",
                min_value=DEFAULT_MIN_JACCARD,
                max_value=1.0,
                value=DEFAULT_MIN_JACCARD,
                step=0.01,
                help="Minimum association strength. 0.0 = include all co-occurrences, 0.1 = moderately associated, 0.5+ = strongly associated"
            )
            
            st.caption("💡 Recommendation: Start with 0.0, increase if network too dense")
        
        with col3:
            st.markdown("**Preview:**")
            total_species = df[species_key].nunique()
            filtered_species = (species_counts >= min_occurrences).sum()
            
            st.metric("Species after filter", f"{filtered_species:,}")
            st.metric("Species removed", f"{total_species - filtered_species:,}")
            
            if min_jaccard > 0:
                st.caption(f"⚠️ Edge filter: Jaccard ≥ {min_jaccard:.2f}")
        
        st.markdown("---")
        st.markdown("#### Step 3: Create network")
        
        if st.button("🔨 Build Network with Jaccard Index", type="primary", use_container_width=True):
            with st.spinner("Creating node list..."):
                nodes_df = create_species_nodes(df, min_occurrences, key_col=species_key)

                if len(nodes_df) == 0:
                    st.error("❌ No species meet the minimum occurrence threshold")
                    st.stop()

                st.success(f"✅ Created {len(nodes_df)} nodes")

            with st.spinner("Calculating Jaccard similarities..."):
                valid_species = set(nodes_df['node_key'])
                edges_df = create_co_occurrence_edges_jaccard(df, valid_species, min_jaccard, key_col=species_key)
                
                if len(edges_df) == 0:
                    st.warning("⚠️ No edges created - try lowering thresholds")
                else:
                    st.success(f"✅ Created {len(edges_df)} edges")
            
            # Store in session state
            st.session_state.nodes_df = nodes_df
            st.session_state.edges_df = edges_df
            st.session_state.min_occurrences = min_occurrences
            st.session_state.min_jaccard = min_jaccard
            
            # Preview of Jaccard matrix
            if len(edges_df) > 0:
                st.markdown("---")
                st.markdown("#### 🔍 Jaccard similarity preview")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Top Species Associations (Highest Jaccard):**")
                    
                    # Get species names for display
                    species_names = dict(zip(nodes_df['node_key'], nodes_df['label']))
                    
                    # Show top 10 pairs
                    top_pairs = edges_df.nlargest(10, 'weight').copy()
                    top_pairs['Species A'] = top_pairs['source'].map(species_names)
                    top_pairs['Species B'] = top_pairs['target'].map(species_names)
                    top_pairs['Jaccard'] = top_pairs['weight'].round(3)
                    top_pairs['Co-occurrences'] = top_pairs['co_occurrences']
                    
                    display_df = top_pairs[['Species A', 'Species B', 'Jaccard', 'Co-occurrences']]
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    st.caption("These species pairs have the strongest associations (high Jaccard = often found together relative to their total occurrences)")
                
                with col2:
                    st.markdown("**Jaccard Distribution:**")
                    
                    # Create histogram
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.hist(edges_df['weight'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
                    ax.set_xlabel('Jaccard Similarity', fontsize=10)
                    ax.set_ylabel('Number of Species Pairs', fontsize=10)
                    ax.set_title('Distribution of Edge Weights', fontsize=11)
                    ax.grid(True, alpha=0.3)
                    
                    # Add median line
                    median_jaccard = edges_df['weight'].median()
                    ax.axvline(median_jaccard, color='red', linestyle='--', linewidth=2, 
                              label=f'Median: {median_jaccard:.3f}')
                    ax.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                # Additional statistics
                col1, col2, col3 = st.columns(3)
                col1.metric("Weak associations (<0.1)", f"{(edges_df['weight'] < 0.1).sum():,}")
                col2.metric("Moderate (0.1-0.3)", f"{((edges_df['weight'] >= 0.1) & (edges_df['weight'] < 0.3)).sum():,}")
                col3.metric("Strong (≥0.3)", f"{(edges_df['weight'] >= 0.3).sum():,}")
            
            # Create graph
            with st.spinner("Building graph structure..."):
                g = create_graph_from_data(nodes_df, edges_df)
                
                if g is None:
                    st.error("Failed to create graph")
                    st.stop()
                
                st.session_state.graph_data = g
                st.success("✅ Graph created")
            
            # Calculate metrics
            with st.spinner("Analyzing network..."):
                metrics = calculate_graph_metrics(g)
                st.session_state.graph_metrics = metrics
            
            # Display additional network metrics (node/edge counts shown in persistent banner above)
            st.markdown("#### 📊 Network quality")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Density", f"{g.density():.4f}")
            col2.metric("Leiden Modularity", f"{metrics['leiden_modularity']:.3f}")
            col3.metric("Louvain Communities", metrics['multi_level_clusters'])
            col4.metric("Louvain Modularity", f"{metrics['multi_level_modularity']:.3f}")

            # Next step guidance
            st.markdown("---")
            st.success("✅ **Network successfully created!** Switch to the **'🎯 2. Generate, Review & Save'** tab above to create your reference map.")

# TAB 2: GENERATE REFERENCE MAP
###################################################################################
if active_tab == tab_labels[1]:
    st.markdown("### 🎯 Generate 2D reference map")

    if st.session_state.graph_data is None:
        st.error("⚠️ **No network found!** Please complete Tab 1 first:")
        st.markdown("""
        <div style='background-color: #fff3cd; padding: 15px; border-radius: 10px; border-left: 4px solid #ffa726;'>
            <b>Required Steps in Tab 1:</b><br>
            1️⃣ Load a database<br>
            2️⃣ Configure network parameters<br>
            3️⃣ Click "🔨 Build Network"<br><br>
            Then return here to generate your reference map.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    else:
        g = st.session_state.graph_data

        # METHOD SELECTION
        st.markdown("#### Choose layout method")

        with st.expander("ℹ️ About the Methods", expanded=False):
            st.markdown("""
            **Fruchterman-Reingold (FR):**
            - Emphasizes network topology and community structure
            - Uses physical simulation (attractive and repulsive forces)
            - Connected species cluster together, communities are well-separated
            - Best for: Community analysis, identifying ecological associations

            **Multidimensional Scaling (MDS):**
            - Emphasizes distance preservation and gradient structure
            - Positions species to preserve Jaccard dissimilarities
            - All pairwise distances reflect co-occurrence patterns
            - Best for: Gradient analysis, environmental interpretation

            **Recommendation:** Create both reference maps and compare them using validation
            analysis (script 14) to determine which better captures environmental gradients for
            your specific vegetation data.
            """)

        layout_method = st.radio(
            "Select layout algorithm:",
            options=["Fruchterman-Reingold", "Multidimensional Scaling (MDS)"],
            help="Choose based on your analysis focus: topology vs. distances"
        )

        st.markdown("---")

        # LAYOUT GENERATION
        if layout_method == "Fruchterman-Reingold":

            st.markdown("#### Step 1: Initial layout (Fruchterman-Reingold)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                max_iterations = st.slider(
                    "FR iterations:",
                    min_value=MIN_FR_ITERATIONS,
                    max_value=MAX_FR_ITERATIONS,
                    value=DEFAULT_FR_ITERATIONS,
                    step=100,
                    help="More iterations = better layout quality"
                )

            with col2:
                use_random_seed = st.checkbox(
                    "Use fixed seed",
                    value=True,
                    help="CRITICAL for reproducibility: Same seed + iterations = identical layout every time. Uncheck for random layouts."
                )

                if use_random_seed:
                    random_seed = st.number_input("Random seed:", min_value=1, value=DEFAULT_SEED, key='seed_input_fr')
                else:
                    random_seed = None
            
            # GENERATE FR
            if st.button("🚀 Generate Layout", type="primary", use_container_width=True, key="generate_fr"):
                with st.spinner(f"Creating {layout_method} layout..."):
                    start_time = datetime.datetime.now()

                    df_fr = create_layout_fruchterman_reingold(
                        g, max_iterations, st.session_state.graph_metrics, random_seed
                    )
                    
                    layout_time = (datetime.datetime.now() - start_time).total_seconds()
                    
                    # Store FR layout and reset to clean state
                    st.session_state.df_fr_layout = df_fr
                    st.session_state.df_current_layout = df_fr.copy()
                    st.session_state.layout_method = layout_method
                    st.session_state.fr_iterations = max_iterations
                    st.session_state.fr_seed = random_seed
                    st.session_state.layout_seed = random_seed
                    
                    st.success(f"✅ Fruchterman-Reingold layout created in {layout_time:.1f} seconds")
                    if random_seed is not None:
                        st.info(f"🔒 Reproducible layout (seed: {random_seed})")

                    # Guide user to next step
                    st.markdown("---")
                    st.info("📍 **Layout generated!** Scroll down to **Step 2: Layout Expansion** to adjust node distribution, then visualize and save.")


        # MDS Layout
        if layout_method == "Multidimensional Scaling (MDS)":

            st.markdown("#### Step 1: Initial layout (MDS)")

            col1, col2 = st.columns(2)

            with col1:
                use_random_seed = st.checkbox(
                    "Use fixed seed",
                    value=True,
                    key='random_seed_mds',
                    help="CRITICAL for reproducibility: Same seed = identical layout every time. Uncheck for random layouts."
                )

            with col2:
                if use_random_seed:
                    random_seed = st.number_input("Random seed:", min_value=1, value=DEFAULT_SEED, key='seed_input_mds')
                else:
                    random_seed = None
            
            # GENERATE MDS
            if st.button("🚀 Generate Layout", type="primary", use_container_width=True, key="generate_mds"):
                with st.spinner(f"Creating {layout_method} layout..."):
                    start_time = datetime.datetime.now()

                    df_mds = create_layout_mds(
                        g, st.session_state.graph_metrics, random_seed
                    )

                    layout_time = (datetime.datetime.now() - start_time).total_seconds()

                    # Store MDS layout and set as current
                    st.session_state.df_mds_layout = df_mds
                    st.session_state.df_current_layout = df_mds.copy()
                    st.session_state.layout_method = layout_method
                    st.session_state.layout_seed = random_seed

                    st.success(f"✅ {layout_method} layout created in {layout_time:.1f} seconds")
                    if random_seed is not None:
                        st.info(f"🔒 Reproducible layout (seed: {random_seed})")

                # Guide user to next step
                st.markdown("---")
                st.info("📍 **Layout generated!** Scroll down to **Step 2: Layout Expansion** to adjust node distribution, then visualize and save.")

        # STEP 2: Expansion
        if st.session_state.df_current_layout is not None:
            st.markdown("---")
            st.markdown("#### Step 2: Layout expansion")

            # ── Expansion method selector ──────────────────────────────────
            expansion_method = st.radio(
                "Expansion method:",
                options=["Peripheral Priority", "Centre Priority"],
                help=(
                    "**Original:** pushes species outward proportionally to their distance "
                    "from centre — peripheral species move most. "
                    "**Concave:** pushes central species outward more than peripheral ones, "
                    "opening up the dense central cluster. Rank order of distances is "
                    "preserved in both methods."
                )
            )

            # ── ORIGINAL method ────────────────────────────────────────────
            if expansion_method == "Peripheral Priority":

                expansion_mode = st.radio(
                    "Expansion mode:",
                    options=["Manual", "Automated"],
                    help="Manual: set factor yourself. Automated: optimize based on criterion",
                    key="orig_mode"
                )

                if expansion_mode == "Manual":
                    expansion_factor = st.slider(
                        "Expansion factor:",
                        min_value=MIN_EXPANSION_FACTOR,
                        max_value=MAX_EXPANSION_FACTOR,
                        value=DEFAULT_EXPANSION_FACTOR,
                        step=0.05,
                        help="Expand nodes radially from center (>1 = expand, 1 = no change)"
                    )
                    expansion_iterations = 1

                else:  # Automated original
                    optimization_criterion = st.selectbox(
                            "Optimization criterion:",
                            options=["max_spread", "target_edge", "minimize_center"],
                            format_func=lambda x: {
                                'max_spread': 'Moderate spread (balanced)',
                                'target_edge': 'Target 30% at edge',
                                'minimize_center': 'Reduce center crowding to 5%'
                            }[x],
                            help="Conservative optimization to avoid pushing all nodes to extremes"
                        )

                    expansion_iterations = 1

                    if st.button("🔍 Find Optimal Expansion", type="secondary", key="opt_orig"):
                        with st.spinner("Optimizing expansion factor..."):
                            optimal_factor = optimize_expansion_factor(
                                st.session_state.df_current_layout,
                                optimization_criterion
                            )
                            expansion_factor = optimal_factor
                            st.success(f"✅ Optimal expansion factor: {optimal_factor:.2f}")
                            st.session_state.optimal_expansion_factor = optimal_factor

                    if 'optimal_expansion_factor' in st.session_state:
                        expansion_factor = st.session_state.optimal_expansion_factor
                        st.info(f"Using optimized factor: {expansion_factor:.2f}")
                    else:
                        expansion_factor = 1.0

                # Apply original expansion
                df_expanded = st.session_state.df_current_layout.copy()
                x_coords = df_expanded['xcoor'].values
                y_coords = df_expanded['ycoor'].values

                for _ in range(expansion_iterations):
                    x_coords, y_coords = expand_nodes(x_coords, y_coords, expansion_factor)

                expansion_label = f"peripheral ×{expansion_factor:.2f}"

            # ── CONCAVE method ─────────────────────────────────────────────
            else:  # Centre Priority

                expansion_mode = st.radio(
                    "Expansion mode:",
                    options=["Manual", "Automated"],
                    help="Manual: set alpha yourself. Automated: optimize based on criterion",
                    key="conc_mode"
                )

                if expansion_mode == "Manual":
                    expansion_alpha = st.slider(
                        "Alpha (concavity):",
                        min_value=float(MIN_EXPANSION_ALPHA),
                        max_value=float(MAX_EXPANSION_ALPHA),
                        value=0.99,
                        step=0.01,
                        help=(
                            "Controls how aggressively the centre is opened up. "
                            "Lower = more expansion of central species; 1.0 = no change. "
                            "Default 0.99 = no change; recommended range: 0.5–0.8"
                        )
                    )

                else:  # Automated concave
                    optimization_criterion_c = st.selectbox(
                        "Optimization criterion:",
                        options=["minimize_center", "target_mean"],
                        format_func=lambda x: {
                            'minimize_center': 'Reduce centre crowding to ~5%',
                            'target_mean': 'Push mean distance to ~0.30'
                        }[x],
                        help="Automatically finds best alpha for the chosen goal"
                    )

                    if st.button("🔍 Find Optimal Alpha", type="secondary", key="opt_conc"):
                        with st.spinner("Optimizing alpha..."):
                            optimal_alpha = optimize_expansion_alpha(
                                st.session_state.df_current_layout,
                                optimization_criterion_c
                            )
                            st.success(f"✅ Optimal alpha: {optimal_alpha:.3f}")
                            st.session_state.optimal_expansion_alpha = optimal_alpha

                    if 'optimal_expansion_alpha' in st.session_state:
                        expansion_alpha = st.session_state.optimal_expansion_alpha
                        st.info(f"Using optimized alpha: {expansion_alpha:.3f}")
                    else:
                        expansion_alpha = 0.99

                # Apply concave expansion (single pass — applied once by design)
                df_expanded = st.session_state.df_current_layout.copy()
                x_coords = df_expanded['xcoor'].values
                y_coords = df_expanded['ycoor'].values
                x_coords, y_coords = expand_nodes_concave(x_coords, y_coords, expansion_alpha)

                # Set these so the save/summary blocks below have consistent variables
                expansion_factor = expansion_alpha
                expansion_iterations = 1
                expansion_label = f"concave α={expansion_alpha:.2f}"

            # ── Common: update coordinates ─────────────────────────────────
            df_expanded['xcoor'] = x_coords
            df_expanded['ycoor'] = y_coords
            df_expanded['distance'] = np.sqrt((df_expanded['xcoor'] - 0.5)**2 +
                                              (df_expanded['ycoor'] - 0.5)**2)
            df_expanded['bearing'] = df_expanded.apply(
                lambda row: map_bearing(row['xcoor'], row['ycoor'], 0.5, 0.5), axis=1
            )

            # Mahalanobis distance from the weighted ecological centroid
            # Uses wdegree as weight (rewards generalists), falls back to
            # occurrence_count then unweighted if unavailable
            try:
                _coords = df_expanded[['xcoor', 'ycoor']].values
                _cov = np.cov(_coords.T)
                _cov_inv = np.linalg.inv(_cov)

                if 'wdegree' in df_expanded.columns and df_expanded['wdegree'].sum() > 0:
                    _w = df_expanded['wdegree'].fillna(0).values
                elif 'occurrence_count' in df_expanded.columns and df_expanded['occurrence_count'].sum() > 0:
                    _w = df_expanded['occurrence_count'].fillna(0).values
                else:
                    _w = np.ones(len(df_expanded))

                _cx = np.average(_coords[:, 0], weights=_w)
                _cy = np.average(_coords[:, 1], weights=_w)
                _centroid = np.array([_cx, _cy])

                df_expanded['mahal_dist'] = df_expanded.apply(
                    lambda row: float(np.sqrt(
                        np.dot(np.dot(
                            np.array([row['xcoor'], row['ycoor']]) - _centroid,
                            _cov_inv),
                            np.array([row['xcoor'], row['ycoor']]) - _centroid)
                    )), axis=1
                )
                st.info(f"📐 Ecological centroid: ({_cx:.3f}, {_cy:.3f}) — Mahalanobis distance added")
            except Exception as e:
                st.warning(f"Could not calculate Mahalanobis distance: {e}")
                df_expanded['mahal_dist'] = df_expanded['distance']  # fallback to geometric

            st.session_state.df_final = df_expanded
            st.session_state.expansion_factor = expansion_factor
            st.session_state.expansion_iterations = expansion_iterations
            st.session_state.expansion_method = expansion_method
            st.session_state.expansion_label = expansion_label
            
            # Visualization controls
            st.markdown("---")
            st.markdown("#### 🎨 Visualization options")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                show_labels = st.checkbox("Show labels", value=False)
                if show_labels:
                    st.caption("Shows top 10 species by weighted degree (works best with <100 species)")
            
            with col2:
                show_kde = st.checkbox("Show density", value=False)
                if show_kde:
                    st.caption("Kernel density estimation showing species clustering patterns")
            
            with col3:
                color_by = st.selectbox(
                    "Color by:",
                    options=["Fixed color", "Community (Leiden)", "Community (Louvain)", 
                            "Weighted degree"]
                )
            
            with col4:
                max_wdegree = int(df_expanded['wdegree'].max())
                if max_wdegree > 0:
                    wdegree_threshold = st.slider(
                        "Min weighted degree:",
                        min_value=0,
                        max_value=max_wdegree,
                        value=0
                    )
                else:
                    wdegree_threshold = 0
                    st.caption("No edges - all nodes shown")
            
            # Filter for visualization
            df_viz = df_expanded[df_expanded['wdegree'] >= wdegree_threshold]
            
            # Create visualization
            st.markdown("---")
            st.markdown("#### 📊 Reference map visualization")
            
            # Single visualization
            method_name = st.session_state.get('layout_method', 'Layout')
            fig = create_network_visualization(
                df_viz,
                f"{method_name} (expansion: {st.session_state.get('expansion_label', 'none')})",
                color_by,
                show_labels,
                show_kde,
                figsize=(12, 12)
            )
            st.pyplot(fig)
            plt.close()
            
            # Statistics
            st.markdown("---")
            st.markdown("#### 📊 Layout statistics")
            
            stats = calculate_node_statistics(df_expanded)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Distance", f"{stats['mean_distance']:.3f}")
            col2.metric("Median Distance", f"{stats['median_distance']:.3f}")
            col3.metric("Nodes in Center", stats['nodes_in_center'])
            col4.metric("Nodes at Edge", stats['nodes_at_edge'])


    # SAVE & EXPORT
    ###################################################################################

    # Only show save section if layout exists
    if st.session_state.df_final is None:
        st.markdown("---")
        st.info("💡 **No layout generated yet.** Complete Steps 1-2 above to generate and visualize your reference map before saving.")
        st.stop()

    st.markdown("---")
    st.markdown("### 💾 Review & save results")

    # Debug: Show session state paths and working directory
    with st.expander("🔧 Debug: View Current Paths & Working Directory", expanded=False):
        st.markdown("**Current working directory (where script runs):**")
        st.code(str(Path.cwd()))
        
        st.markdown("**Script location:**")
        st.code(str(Path(__file__).parent.resolve()))
        
        st.markdown("**Session state paths from settings.txt:**")
        for key in ['project_base_path', 'reference_map_path', 'queries_path', 'data_path']:
            value = st.session_state.get(key, 'Not set')
            st.code(f"{key}: {value}")
        
        st.markdown("**Expected save location:**")
        if 'project_base_path' in st.session_state:
            expected = Path(st.session_state['project_base_path']).resolve() / 'reference_maps'
            st.code(str(expected))
        else:
            st.warning("project_base_path not set in session state!")
        
        st.caption("⚠️ If files appear in 'Current working directory' instead of 'Expected save location', the path resolution is failing.")
        
        st.markdown("---")
        st.markdown("**Reproducibility troubleshooting:**")
        st.markdown("""
        If FR layouts differ with same seed + iterations:
        1. Verify you're using the **same network** (same min_occurrences, min_jaccard)
        2. Check the seed is actually set (not None)
        3. Clear Streamlit cache and restart app completely
        4. igraph version differences can cause minor variations
        """)

    df_final = st.session_state.df_final

    # Configuration summary
    st.markdown("#### ⚙️ Configuration used")

    # Convert all values to strings to avoid Arrow serialization errors
    config_data = {
        'Source Database': st.session_state.get('selected_db', 'unknown'),
        'Min Occurrences': str(st.session_state.get('min_occurrences', 'N/A')),
        'Min Jaccard': str(st.session_state.get('min_jaccard', 'N/A')),
        'Layout Method': st.session_state.get('layout_method', 'N/A')
    }

    # Only show FR iterations if using FR method
    layout_method = st.session_state.get('layout_method', '')
    if 'Fruchterman' in layout_method:
        config_data['FR Iterations'] = str(st.session_state.get('fr_iterations', 'N/A'))

    config_data['Random Seed'] = str(st.session_state.get('layout_seed', 'N/A'))

    # Add expansion parameters if available
    expansion_factor = st.session_state.get('expansion_factor')
    expansion_iterations = st.session_state.get('expansion_iterations')
    if expansion_factor is not None and expansion_iterations is not None:
        config_data['Expansion'] = st.session_state.get('expansion_label', f"{expansion_factor:.2f} × {expansion_iterations}")

    config_df = pd.DataFrame(list(config_data.items()), columns=['Parameter', 'Value'])
    st.dataframe(config_df, use_container_width=True, hide_index=True)

    # Save options
    st.markdown("---")
    st.markdown("#### 💾 Save options")

    col1, col2 = st.columns(2)

    with col1:
        # Determine output path with proper fallback chain - ALWAYS use absolute paths
        default_output_path = None

        # Try getting reference_map_path directly from session state
        if 'reference_map_path' in st.session_state and st.session_state['reference_map_path']:
            default_output_path = str(Path(st.session_state['reference_map_path']).resolve())

        # Try constructing from project_base_path (most reliable)
        elif 'project_base_path' in st.session_state and st.session_state['project_base_path']:
            base_path = Path(st.session_state['project_base_path']).resolve()
            default_output_path = str(base_path / 'reference_maps')

        # Try constructing from queries_path (go up one level, then to reference_maps)
        elif 'queries_path' in st.session_state and st.session_state['queries_path']:
            queries_path = Path(st.session_state['queries_path']).resolve()
            default_output_path = str(queries_path.parent / 'reference_maps')

        # Last resort - use current directory
        else:
            default_output_path = str(Path.cwd() / 'reference_maps')
            st.warning("⚠️ Using current directory - check paths in settings.txt")

        output_path = st.text_input(
            "Output directory:",
            value=default_output_path,
            help="Absolute path where files will be saved"
        )

        # Always show absolute path
        abs_output_path = Path(output_path).resolve()
        st.info(f"📂 **Files will be saved to:** `{abs_output_path}`")

        # Verify it's not the script directory
        script_dir = Path(__file__).parent.resolve()
        if abs_output_path == script_dir or abs_output_path.is_relative_to(script_dir):
            st.error("❌ WARNING: This will save files in the script directory! Please use the reference_maps folder in your project.")

        save_network = st.checkbox(
            "Save network files (nodes.csv, edges.csv)",
            value=False
        )

        if save_network:
            export_format = st.radio(
                "Network file format:",
                options=["CSV (with headers)", "CSV (no headers)"]
            )

    with col2:
        db_name = st.text_input(
            "Coordinate database name:",
            value="coordinates.db",
            help="SQLite database for coordinates"
        )

        has_fr_layout = st.session_state.get('df_fr_layout') is not None
        if has_fr_layout:
            save_original_fr = st.checkbox(
                "Include original FR coordinates",
                value=True,
                help="Save both FR and final coordinates"
            )
        else:
            save_original_fr = False

    # Save button
    if st.button("💾 Save All Files", type="primary", use_container_width=True):
        try:
            # ALWAYS use absolute path to avoid saving in wrong location
            output_dir = Path(output_path).resolve()

            # Double-check we're not saving in script directory
            script_dir = Path(__file__).parent.resolve()
            if output_dir == script_dir or output_dir.is_relative_to(script_dir):
                st.error("❌ STOPPED: Would save files in script directory! Change output path to reference_maps folder.")
                st.stop()

            # Show where we're saving
            st.info(f"📁 **Saving to:** `{output_dir}`")

            # Create directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            st.success(f"✅ Output directory created/verified")

            # Save network files
            if save_network:
                nodes_path = output_dir / "nodes.csv"
                edges_path = output_dir / "edges.csv"

                if export_format == "CSV (no headers)":
                    st.session_state.nodes_df.to_csv(nodes_path, index=False, header=False)
                    st.session_state.edges_df.to_csv(edges_path, index=False, header=False)
                else:
                    st.session_state.nodes_df.to_csv(nodes_path, index=False)
                    st.session_state.edges_df.to_csv(edges_path, index=False)

                st.success(f"✅ Network files saved")

            # Save coordinate database
            db_path = output_dir / db_name

            df_to_save = df_final.copy()

            if save_original_fr and st.session_state.get('df_fr_layout') is not None:
                df_to_save['x_fr'] = st.session_state.df_fr_layout['xcoor']
                df_to_save['y_fr'] = st.session_state.df_fr_layout['ycoor']

            with sqlite3.connect(str(db_path)) as conn:
                # Save coordinates
                df_to_save.to_sql('keyword_coordinates', conn, if_exists='replace', index=False)

                # Save metadata
                metadata = pd.DataFrame([{
                    'creation_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source_database': st.session_state.get('selected_db', 'unknown'),
                    'min_occurrences': st.session_state.get('min_occurrences', 1),
                    'min_jaccard': st.session_state.get('min_jaccard', 0.0),
                    'layout_method': st.session_state.get('layout_method', 'Unknown'),
                    'layout_algorithm': 'Fruchterman-Reingold' if 'Fruchterman' in st.session_state.get('layout_method', '') else 'MDS',
                    'fr_iterations': st.session_state.get('fr_iterations', None),
                    'random_seed': st.session_state.get('layout_seed', None),
                    'expansion_method': st.session_state.get('expansion_method', 'Original (peripheral priority)'),
                    'expansion_label': st.session_state.get('expansion_label', 'none'),
                    'expansion_factor': st.session_state.get('expansion_factor', 1.0),
                    'expansion_iterations': st.session_state.get('expansion_iterations', 1),
                    'num_nodes': len(df_to_save),
                    'num_edges': len(st.session_state.edges_df),
                    'leiden_clusters': st.session_state.graph_metrics['leiden_clusters'],
                    'louvain_clusters': st.session_state.graph_metrics['multi_level_clusters']
                }])
                metadata.to_sql('metadata', conn, if_exists='replace', index=False)

            st.success(f"✅ Coordinate database saved: {db_name}")

            # Update session state with absolute paths
            st.session_state.coordinates_db_path = str(db_path.resolve())
            st.session_state.reference_map_complete = True

            # Comprehensive save summary
            st.markdown("---")
            st.markdown("#### ✅ Save complete - file summary")

            saved_files = []

            if save_network and 'nodes_path' in locals():
                saved_files.append(f"📄 {nodes_path.name} → `{nodes_path.resolve()}`")
                saved_files.append(f"📄 {edges_path.name} → `{edges_path.resolve()}`")
                st.session_state.network_nodes_path = str(nodes_path.resolve())
                st.session_state.network_edges_path = str(edges_path.resolve())

            saved_files.append(f"💾 {db_name} → `{db_path.resolve()}`")

            st.markdown("**Files saved:**")
            for file_info in saved_files:
                st.markdown(file_info)

            # Summary metrics
            col1, col2 = st.columns(2)
            col1.metric("Database Size", f"{db_path.stat().st_size / 1024:.1f} KB")
            col2.metric("Files Created", len(saved_files))

            st.success("🎉 All files saved successfully!")

        except (PermissionError, OSError, sqlite3.Error, KeyError, AttributeError) as e:
            st.error(f"❌ Error saving files: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d; font-size: 0.9em;'>
    EcoNetMap - Network-based Reference Map Generation
    </div>
    """, 
    unsafe_allow_html=True
)
