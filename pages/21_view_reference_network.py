"""
EcoNetMap - Multi-Species Distribution Viewer
================================================================
This module creates various visualizations of species distributions in ecological space.
Features interactive Plotly maps with hover tooltips for species identification.

Part of the EcoNetMap toolkit (mapping 1/2)
Author: Flemming Skov (fs@ecos.au.dk)
Last Updated: January 2025
"""

# Import packages for web applications
import streamlit as st

# Import packages for data manipulation and analysis
import pandas as pd
import numpy as np
import sqlite3

# Import packages for file and system operations
from pathlib import Path
import warnings
import traceback

# Import packages for type hints
from typing import Optional, Tuple, List, Dict

# Import packages for visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import plotly.express as px

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Page configuration
st.set_page_config(
    page_title="Species Mapping - EcoNetMap", 
    page_icon="🗺️",
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
    div[data-testid="stExpander"] > details {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-left: 20px;
        padding-right: 20px;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and progress indicator
col1, col2 = st.columns([4, 1])
with col1:
    st.header("Maps")
    st.subheader("🌿 Reference maps")
    st.markdown("*Species in ecological reference space, gradients and distribution of groups*")
with col2:
    pass

st.markdown("---")

# FUNCTIONS
###################################################################################

@st.cache_data(show_spinner=False)
def load_map_data(db_path: str) -> Dict[str, pd.DataFrame]:
    """Load all relevant tables from map database with error handling"""
    try:
        if not Path(db_path).exists():
            st.error(f"Database file not found: {db_path}")
            return {}
        
        conn = sqlite3.connect(db_path)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        
        data_dict = {}
        for table in tables['name'].values:
            if not table.startswith('_'):  # Skip metadata tables
                try:
                    data_dict[table] = pd.read_sql_query(f'SELECT * FROM {table}', conn)
                except Exception as e:
                    st.warning(f"Could not load table '{table}': {str(e)}")
        
        conn.close()
        
        if not data_dict:
            st.error("No valid tables found in database")
            return {}
        
        return data_dict
        
    except Exception as e:
        st.error(f"Error loading map data: {str(e)}")
        return {}

def validate_required_columns(df: pd.DataFrame, required_cols: List[str], table_name: str = "") -> bool:
    """Validate that required columns exist in dataframe"""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing columns in {table_name}: {missing_cols}")
        return False
    return True

def create_base_map(title: str = '', figsize: Tuple[int, int] = (12, 12)) -> Tuple[plt.Figure, plt.Axes]:
    """Create a base map with guide circles and lines (matplotlib version)"""
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set up the plot
        ax.set_xlim(-0.010, 1.01)
        ax.set_ylim(-0.010, 1.01)
        ax.set_aspect('equal')
        
        # Draw guide circles
        center = (0.5, 0.5)
        radii = [0.125, 0.25, 0.375, 0.5]
        for radius in radii:
            circle = Circle(center, radius, linewidth=0.5, color='gray', fill=False, alpha=0.3)
            ax.add_patch(circle)
        
        # Draw guide lines
        ax.add_line(Line2D([0.5, 0.5], [0, 1], color='gray', linewidth=0.5, alpha=0.3))
        ax.add_line(Line2D([0, 1], [0.5, 0.5], color='gray', linewidth=0.5, alpha=0.3))
        ax.add_line(Line2D([0, 1], [0, 1], color='gray', linewidth=0.5, alpha=0.3))
        ax.add_line(Line2D([0, 1], [1, 0], color='gray', linewidth=0.5, alpha=0.3))
        
        # Labels and title
        ax.set_xlabel('X coordinate', fontsize=10, alpha=0.7)
        ax.set_ylabel('Y coordinate', fontsize=10, alpha=0.7)
        ax.set_title(title, fontsize=12, pad=20)
        ax.grid(False)
        
        return fig, ax
    except Exception as e:
        st.error(f"Error creating base map: {str(e)}")
        return plt.subplots(figsize=figsize)

def create_plotly_base_map(title: str = '', zoom: float = -0.05) -> go.Figure:
    """Create a base map with guide circles and lines (Plotly version)"""
    fig = go.Figure()
    
    # Calculate axis limits based on zoom
    x_min, x_max = zoom, 1 - zoom
    y_min, y_max = zoom, 1 - zoom
    
    # Guide circles
    center = (0.5, 0.5)
    radii = [0.125, 0.25, 0.375, 0.5]
    for radius in radii:
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = center[0] + radius * np.cos(theta)
        y_circle = center[1] + radius * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=x_circle, y=y_circle,
            mode='lines',
            line=dict(color='gray', width=0.5),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Guide lines
    guide_lines = [
        ([0.5, 0.5], [0, 1]),      # Vertical center
        ([0, 1], [0.5, 0.5]),      # Horizontal center
        ([0, 1], [0, 1]),          # Diagonal
        ([0, 1], [1, 0])           # Anti-diagonal
    ]
    for x_coords, y_coords in guide_lines:
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            mode='lines',
            line=dict(color='gray', width=0.5),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis=dict(
            title='X coordinate',
            range=[x_min, x_max],
            scaleanchor='y',
            scaleratio=1,
            showgrid=False
        ),
        yaxis=dict(
            title='Y coordinate',
            range=[y_min, y_max],
            showgrid=False
        ),
        plot_bgcolor='white',
        width=800,
        height=800,
        hovermode='closest'
    )
    
    return fig

def get_color_palette(n_colors: int, palette_name: str = 'Set2') -> List:
    """Get a color palette with the specified number of colors"""
    try:
        if n_colors <= 0:
            return ['blue']
        elif n_colors <= 8:
            return sns.color_palette(palette_name, n_colors).as_hex()
        else:
            return sns.color_palette('husl', n_colors).as_hex()
    except Exception:
        return ['blue'] * max(1, n_colors)

def safe_figure_save(fig: plt.Figure, filename: str, figures_path: Path) -> bool:
    """Save figure to file with comprehensive error handling"""
    try:
        if not figures_path.exists():
            figures_path.mkdir(parents=True, exist_ok=True)
        
        # Clean filename
        clean_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
        if not clean_filename:
            clean_filename = "unnamed_figure"
        
        filepath = figures_path / f"{clean_filename}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.1, 
                   facecolor='white', edgecolor='none')
        return True
    except Exception as e:
        st.error(f"Error saving figure: {str(e)}")
        return False

def safe_plotly_save(fig: go.Figure, filename: str, figures_path: Path) -> bool:
    """Save Plotly figure to file with comprehensive error handling"""
    try:
        if not figures_path.exists():
            figures_path.mkdir(parents=True, exist_ok=True)
        
        # Clean filename
        clean_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
        if not clean_filename:
            clean_filename = "unnamed_figure"
        
        filepath = figures_path / f"{clean_filename}.png"
        fig.write_image(str(filepath), scale=2)
        return True
    except Exception as e:
        st.error(f"Error saving figure: {str(e)}")
        return False

def safe_kde_plot(data: pd.DataFrame, x_col: str, y_col: str, ax: plt.Axes, **kwargs) -> bool:
    """Safely create KDE plot with error handling"""
    try:
        if len(data) < 3:
            return False
        
        # Check for valid coordinates
        valid_data = data.dropna(subset=[x_col, y_col])
        if len(valid_data) < 3:
            return False
        
        # Remove 'weights' from kwargs if present - seaborn 2D kdeplot doesn't support it
        kwargs.pop('weights', None)
        
        sns.kdeplot(data=valid_data, x=x_col, y=y_col, ax=ax, **kwargs)
        return True
    except Exception as e:
        st.warning(f"Could not create KDE plot: {str(e)}")
        return False

def calculate_weighted_centroid(df: pd.DataFrame, n_nearest: int = 10) -> dict:
    """
    Calculate the abundance-weighted centroid of the full species distribution.
    Always uses the complete dataset - never affected by filters.
    Weights: wdegree (primary) → occurrence_count (fallback) → unweighted.
    """
    valid = df.dropna(subset=['xcoor', 'ycoor']).copy()
    valid = valid[(valid['xcoor'] >= 0) & (valid['xcoor'] <= 1) &
                  (valid['ycoor'] >= 0) & (valid['ycoor'] <= 1)]

    if len(valid) == 0:
        return {}

    if 'wdegree' in valid.columns and valid['wdegree'].notna().any():
        weights = valid['wdegree'].fillna(0)
        weight_label = 'wdegree'
    elif 'occurrence_count' in valid.columns and valid['occurrence_count'].notna().any():
        weights = valid['occurrence_count'].fillna(0)
        weight_label = 'occurrence_count'
    else:
        weights = pd.Series(np.ones(len(valid)), index=valid.index)
        weight_label = 'unweighted'

    total_weight = weights.sum()
    if total_weight == 0:
        cx, cy = valid['xcoor'].mean(), valid['ycoor'].mean()
        weight_label = 'unweighted (zero weights)'
    else:
        cx = (valid['xcoor'] * weights).sum() / total_weight
        cy = (valid['ycoor'] * weights).sum() / total_weight

    valid['_dist_to_centroid'] = np.sqrt((valid['xcoor'] - cx)**2 + (valid['ycoor'] - cy)**2)
    nearest = valid.nsmallest(n_nearest, '_dist_to_centroid')

    return {'cx': cx, 'cy': cy, 'weight_label': weight_label, 'nearest': nearest}

def safe_scatter_plot(ax: plt.Axes, x_data, y_data, **kwargs) -> bool:
    """Safely create scatter plot with error handling"""
    try:
        if len(x_data) == 0 or len(y_data) == 0:
            return False
        
        ax.scatter(x_data, y_data, **kwargs)
        return True
    except Exception as e:
        st.warning(f"Could not create scatter plot: {str(e)}")
        return False

# Main interface
###################################################################################

# Database selection
st.markdown("#### 🔍 Select map database")

try:
    overlay_path = Path(st.session_state.get('overlay_map_path', '.'))
    if not overlay_path.exists():
        st.error(f"Overlay map directory not found: {overlay_path}")
        st.info("Please ensure the overlay_map_path is set correctly in session state.")
        st.stop()

    db_files = sorted([f.name for f in overlay_path.glob("*.db")])

    if not db_files:
        st.error("No map databases found.")
        st.info("Please complete the Prepare Map step first.")
        st.stop()

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_db = st.selectbox(
            "Select map database:",
            options=db_files,
            help="Choose a prepared map database"
        )
        db_path = overlay_path / selected_db

    with col2:
        if st.button("📊 Load Map Data", type="primary", use_container_width=True):
            map_data = load_map_data(str(db_path))
            if map_data:
                st.session_state.map_data = map_data
                st.success(f"✅ Loaded {len(map_data)} tables")

except Exception as e:
    st.error(f"Error in database selection: {str(e)}")
    st.stop()

# Show data info
if 'map_data' in st.session_state:
    try:
        with st.expander("📊 Database Contents", expanded=False):
            for table_name, df in st.session_state.map_data.items():
                st.text(f"Table '{table_name}': {len(df):,} records × {len(df.columns)} columns")
    except Exception as e:
        st.warning(f"Error displaying database contents: {str(e)}")

# Main mapping interface
if 'map_data' in st.session_state:
    try:
        map_data = st.session_state.map_data
        
        # Check for required tables
        if 'taxa' not in map_data:
            st.error("Taxa table not found in database")
            st.stop()
        
        taxa_df = map_data['taxa']
        plot_df = map_data.get('plot_id', pd.DataFrame())
        occurrence_df = map_data.get('data', pd.DataFrame())
        
        # Validate taxa table has required columns
        if not validate_required_columns(taxa_df, ['keyword'], 'taxa'):
            st.error("Taxa table missing required columns")
            st.stop()
        
        st.markdown("---")
        
        # Create tabs for different map types
        tab1, tab2, tab3 = st.tabs([
            "🌿 Species",  
            "📊 Environmental gradients",
            "🌲 Categories"
        ])
        
###############################################################################################################################################################################
        # TAB 1: SPECIES DISTRIBUTION (PLOTLY INTERACTIVE)
###############################################################################################################################################################################

        with tab1:
            try:
                st.markdown("### Species distribution")
                st.caption("*Hover over points to see species names*")

                # Compute wdegree-weighted centroid early for distance filter and hover
                centroid_info = calculate_weighted_centroid(taxa_df)
                if centroid_info:
                    _cx, _cy = centroid_info['cx'], centroid_info['cy']
                    taxa_df = taxa_df.copy()
                    taxa_df['dist_centroid'] = np.sqrt(
                        (taxa_df['xcoor'] - _cx)**2 + (taxa_df['ycoor'] - _cy)**2
                    ).round(3)

                # Filtering options
                col1, col2 = st.columns(2)
                
                with col1:
                    # Filter by taxonomic group
                    if 'gruppe' in taxa_df.columns:
                        unique_groups = taxa_df['gruppe'].dropna().unique().tolist()
                        selected_groups = st.multiselect(
                            "Select taxonomic groups:",
                            options=unique_groups,
                            default=unique_groups[:5] if len(unique_groups) > 5 else unique_groups
                        )
                        filtered_taxa = taxa_df[taxa_df['gruppe'].isin(selected_groups)] if selected_groups else taxa_df
                    else:
                        filtered_taxa = taxa_df
                        selected_groups = []
                    
                    # Filter by network clusters
                    if 'leiden' in taxa_df.columns:
                        unique_clusters = sorted(taxa_df['leiden'].dropna().unique())
                        selected_clusters = st.multiselect(
                            "Select network clusters:",
                            options=unique_clusters,
                            default=unique_clusters
                        )
                        if selected_clusters:
                            filtered_taxa = filtered_taxa[filtered_taxa['leiden'].isin(selected_clusters)]
                
                with col2:
                    # Distance filter — uses wdegree-weighted centroid
                    if 'dist_centroid' in filtered_taxa.columns:
                        _valid_dists = filtered_taxa['dist_centroid'].dropna()
                        if len(_valid_dists) > 0:
                            min_dist = float(_valid_dists.min())
                            max_dist = float(_valid_dists.max())
                            distance_range = st.slider(
                                "Distance from centroid:",
                                min_value=min_dist,
                                max_value=max_dist,
                                value=(min_dist, max_dist),
                                step=0.01,
                                help="Filter species by distance from the wdegree-weighted ecological centroid"
                            )
                            filtered_taxa = filtered_taxa[
                                (filtered_taxa['dist_centroid'] >= distance_range[0]) &
                                (filtered_taxa['dist_centroid'] <= distance_range[1])
                            ]
                    elif 'distance' in filtered_taxa.columns:
                        min_dist, max_dist = filtered_taxa['distance'].min(), filtered_taxa['distance'].max()
                        distance_range = st.slider(
                            "Distance from center:",
                            min_value=float(min_dist),
                            max_value=float(max_dist),
                            value=(float(min_dist), float(max_dist)),
                            step=0.01,
                            help="Filter species by their distance from the map center"
                        )
                        filtered_taxa = filtered_taxa[
                            (filtered_taxa['distance'] >= distance_range[0]) &
                            (filtered_taxa['distance'] <= distance_range[1])
                        ]
                
                st.info(f"Showing {len(filtered_taxa)} species after filtering")
                
                # Validate coordinate columns
                if not validate_required_columns(filtered_taxa, ['xcoor', 'ycoor'], 'filtered taxa'):
                    st.error("Coordinate data not available")
                else:
                    # Remove rows with invalid coordinates
                    valid_coords = filtered_taxa.dropna(subset=['xcoor', 'ycoor'])
                    valid_coords = valid_coords[
                        (valid_coords['xcoor'] >= 0) & (valid_coords['xcoor'] <= 1) &
                        (valid_coords['ycoor'] >= 0) & (valid_coords['ycoor'] <= 1)
                    ]
                    
                    if len(valid_coords) == 0:
                        st.warning("No species with valid coordinates found")
                    else:
                        with st.expander("Show selected species:"):
                            st.dataframe(valid_coords)
                        
                        # Map settings
                        with st.expander("🎨 Map Settings", expanded=True):
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                point_size = st.slider("Point size:", 5, 20, 8)
                                point_opacity = st.slider("Point opacity:", 0.3, 1.0, 0.7)
                                num_labels = st.slider(
                                    "Number of labels (for saved image):",
                                    min_value=0,
                                    max_value=min(100, len(valid_coords)),
                                    value=min(20, len(valid_coords)),
                                    help="Labels are shown in saved PNG; hover for names in interactive view"
                                )

                            with col2:
                                color_options = ['None']
                                if 'gruppe' in valid_coords.columns:
                                    color_options.append('gruppe')
                                if 'leiden' in valid_coords.columns:
                                    color_options.append('leiden')

                                color_by = st.selectbox("Color points by:", options=color_options, index=0)

                            with col3:
                                zoom_level = st.slider("Zoom:", -0.1, 0.25, -0.05)
                                show_centroid = st.checkbox(
                                    "Show ecological centroid",
                                    value=True,
                                    help="Show the abundance-weighted centroid of all species"
                                )
                        
                        # Create the Plotly map
                        fig = create_plotly_base_map("Species distribution", zoom=zoom_level)
                        
                        # Build hover text
                        hover_columns = ['keyword']
                        if 'gruppe' in valid_coords.columns:
                            hover_columns.append('gruppe')
                        if 'leiden' in valid_coords.columns:
                            hover_columns.append('leiden')
                        if 'dist_centroid' in valid_coords.columns:
                            hover_columns.append('dist_centroid')
                        elif 'distance' in valid_coords.columns:
                            hover_columns.append('distance')
                        
                        # Plot points
                        if color_by != 'None' and color_by in valid_coords.columns:
                            unique_values = valid_coords[color_by].dropna().unique()
                            colors = get_color_palette(len(unique_values))
                            
                            for i, value in enumerate(unique_values):
                                subset = valid_coords[valid_coords[color_by] == value].copy()
                                
                                # Build hover text for this subset
                                hover_text = subset.apply(
                                    lambda row: '<br>'.join([f"<b>{col}</b>: {row[col]}" for col in hover_columns if col in row.index and pd.notna(row[col])]),
                                    axis=1
                                )
                                
                                fig.add_trace(go.Scatter(
                                    x=subset['xcoor'],
                                    y=subset['ycoor'],
                                    mode='markers',
                                    marker=dict(
                                        size=point_size,
                                        color=colors[i],
                                        opacity=point_opacity,
                                        line=dict(width=0.5, color='black')
                                    ),
                                    name=str(value)[:20],
                                    text=hover_text,
                                    hoverinfo='text',
                                    showlegend=True
                                ))
                        else:
                            # Single color for all points
                            hover_text = valid_coords.apply(
                                lambda row: '<br>'.join([f"<b>{col}</b>: {row[col]}" for col in hover_columns if col in row.index and pd.notna(row[col])]),
                                axis=1
                            )
                            
                            fig.add_trace(go.Scatter(
                                x=valid_coords['xcoor'],
                                y=valid_coords['ycoor'],
                                mode='markers',
                                marker=dict(
                                    size=point_size,
                                    color='darkblue',
                                    opacity=point_opacity,
                                    line=dict(width=0.5, color='black')
                                ),
                                name='Species',
                                text=hover_text,
                                hoverinfo='text',
                                showlegend=False
                            ))
                        
                        # Update layout for legend
                        fig.update_layout(
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=1.02
                            )
                        )
                        
                        # Ecological centroid — always on full taxa_df, unaffected by filters
                        if show_centroid:
                            if centroid_info:
                                cx, cy = centroid_info['cx'], centroid_info['cy']

                                # Store for script 23
                                st.session_state.ecological_centroid = centroid_info

                                hover_lines = [
                                    f"<b>Ecological centroid</b>",
                                    f"X: {cx:.3f}, Y: {cy:.3f}",
                                    f"Weighted by: {centroid_info['weight_label']}",
                                ]

                                fig.add_trace(go.Scatter(
                                    x=[cx], y=[cy],
                                    mode='markers',
                                    marker=dict(
                                        symbol='star',
                                        size=18,
                                        color='gold',
                                        line=dict(width=1.5, color='darkred')
                                    ),
                                    name='Ecological centroid',
                                    text=['<br>'.join(hover_lines)],
                                    hoverinfo='text',
                                    showlegend=True
                                ))

                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Save option - create matplotlib version with labels for export
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            save_name = st.text_input("Save as:", value="species_distribution", key="save1")
                        with col2:
                            if st.button("💾 Save Map", key="save_btn1"):
                                figures_path = Path(st.session_state.get('figures_path', '.'))
                                
                                # Create matplotlib version for saving (with labels)
                                save_fig, save_ax = create_base_map("Species distribution")
                                
                                # Plot all points
                                if color_by != 'None' and color_by in valid_coords.columns:
                                    unique_values = valid_coords[color_by].dropna().unique()
                                    colors = get_color_palette(len(unique_values))
                                    for i, value in enumerate(unique_values):
                                        subset = valid_coords[valid_coords[color_by] == value]
                                        safe_scatter_plot(
                                            save_ax, subset['xcoor'], subset['ycoor'],
                                            s=point_size*3, alpha=point_opacity, c=[colors[i]],
                                            label=str(value)[:20], edgecolors='black', linewidths=0.5
                                        )
                                    if len(unique_values) <= 10:
                                        save_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                                else:
                                    safe_scatter_plot(
                                        save_ax, valid_coords['xcoor'], valid_coords['ycoor'],
                                        s=point_size*3, alpha=point_opacity, c='darkblue',
                                        edgecolors='black', linewidths=0.5
                                    )
                                
                                # Add labels - deterministic selection by distance from center
                                if num_labels > 0 and len(valid_coords) > 0:
                                    num_to_label = min(num_labels, len(valid_coords))
                                    
                                    # Select species furthest from centroid for labels
                                    if 'dist_centroid' in valid_coords.columns:
                                        label_species = valid_coords.nlargest(num_to_label, 'dist_centroid')
                                    elif 'distance' in valid_coords.columns:
                                        label_species = valid_coords.nlargest(num_to_label, 'distance')
                                    else:
                                        temp_dist = np.sqrt((valid_coords['xcoor'] - 0.5)**2 + (valid_coords['ycoor'] - 0.5)**2)
                                        label_indices = temp_dist.nlargest(num_to_label).index
                                        label_species = valid_coords.loc[label_indices]
                                    
                                    for _, species in label_species.iterrows():
                                        save_ax.annotate(
                                            species['keyword'][:20] if 'keyword' in species else '',
                                            (species['xcoor'], species['ycoor']),
                                            xytext=(5, 5), textcoords='offset points',
                                            fontsize=8, alpha=0.8
                                        )
                                
                                save_ax.set_xlim(zoom_level, 1 - zoom_level)
                                save_ax.set_ylim(zoom_level, 1 - zoom_level)
                                
                                if safe_figure_save(save_fig, save_name, figures_path):
                                    st.success(f"Map saved as {save_name}.png")
                                
                                plt.close(save_fig)
            
            except Exception as e:
                st.error(f"Error in Species Distribution tab: {str(e)}")
                st.code(traceback.format_exc())
        
 
###############################################################################################################################################################################
        # TAB 2: ENVIRONMENTAL GRADIENTS
###############################################################################################################################################################################

        with tab2:
            try:
                st.markdown("### 📊 Environmental Gradient Mapping")
                
                env_vars = [col for col in ['L', 'M', 'N', 'R', 'T'] if col in taxa_df.columns]
                
                if not env_vars:
                    st.info("No environmental indicator data available in this dataset")
                else:
                    env_labels = {
                        'L': 'Light', 'M': 'Moisture', 'N': 'Nitrogen',
                        'R': 'Reaction (pH)', 'T': 'Temperature'
                    }
                    
                    selected_env = st.selectbox(
                        "Select environmental indicator:",
                        options=env_vars,
                        format_func=lambda x: f"{env_labels.get(x, x)} ({x})"
                    )
                    
                    # Filter by indicator values
                    env_data = taxa_df.dropna(subset=[selected_env])
                    
                    if len(env_data) == 0:
                        st.warning(f"No data available for {env_labels.get(selected_env, selected_env)}")
                    else:
                        # Validate coordinates
                        if not validate_required_columns(env_data, ['xcoor', 'ycoor']):
                            st.error("Coordinate data not available for environmental mapping")
                        else:
                            env_data = env_data.dropna(subset=['xcoor', 'ycoor'])
                            env_data = env_data[
                                (env_data['xcoor'] >= 0) & (env_data['xcoor'] <= 1) &
                                (env_data['ycoor'] >= 0) & (env_data['ycoor'] <= 1)
                            ]
                            
                            if len(env_data) == 0:
                                st.warning("No valid coordinate data for environmental mapping")
                            else:
                                value_range = st.slider(
                                    f"Select {env_labels.get(selected_env, selected_env)} value range:",
                                    min_value=float(env_data[selected_env].min()),
                                    max_value=float(env_data[selected_env].max()),
                                    value=(float(env_data[selected_env].min()), float(env_data[selected_env].max())),
                                    step=0.5
                                )
                                
                                filtered_env = env_data[
                                    (env_data[selected_env] >= value_range[0]) & 
                                    (env_data[selected_env] <= value_range[1])
                                ]
                                
                                st.info(f"Showing {len(filtered_env)} species with {env_labels.get(selected_env, selected_env)} values between {value_range[0]} and {value_range[1]}")
                                
                                if len(filtered_env) > 0:
                                    with st.expander("Show selected species:"):
                                        st.dataframe(filtered_env[['keyword', selected_env, 'xcoor', 'ycoor']])
                                    
                                    # Create environmental map
                                    fig, ax = create_base_map(f"{env_labels.get(selected_env, selected_env)} Gradient")
                                    
                                    # KDE plot (density shading without weighting)
                                    if st.checkbox("Show density shading", value=True, key="env_kde"):
                                        safe_kde_plot(
                                            filtered_env, 'xcoor', 'ycoor', ax,
                                            fill=True, cmap='RdYlBu_r', levels=20,
                                            thresh=0.05, alpha=0.5
                                        )
                                    
                                    # Scatter plot
                                    try:
                                        scatter = ax.scatter(
                                            filtered_env['xcoor'], filtered_env['ycoor'],
                                            c=filtered_env[selected_env], cmap='RdYlBu_r',
                                            s=30, alpha=0.8, edgecolors='black', linewidths=0.5
                                        )
                                        
                                        # Add colorbar
                                        cbar = plt.colorbar(scatter, ax=ax)
                                        cbar.set_label(f"{env_labels.get(selected_env, selected_env)} Value", 
                                                     rotation=270, labelpad=20)
                                    except Exception as e:
                                        st.warning(f"Could not create scatter plot: {str(e)}")
                                    
                                    zoom = st.slider("Zoom:", -0.1, 0.25, -0.05, key="env_zoom")
                                    ax.set_xlim(zoom, 1 - zoom)
                                    ax.set_ylim(zoom, 1 - zoom)
                                    
                                    st.pyplot(fig)
                                    
                                    # Save option
                                    col1, col2 = st.columns([2, 1])
                                    with col1:
                                        save_name = st.text_input("Save as:", value=f"{selected_env}_gradient", key="save3")
                                    with col2:
                                        if st.button("💾 Save Map", key="save_btn3"):
                                            figures_path = Path(st.session_state.get('figures_path', '.'))
                                            if safe_figure_save(fig, save_name, figures_path):
                                                st.success(f"Map saved as {save_name}.png")
                                    
                                    plt.close(fig)
            
            except Exception as e:
                st.error(f"Error in Environmental Gradients tab: {str(e)}")
                st.code(traceback.format_exc())
        
###############################################################################################################################################################################
        # TAB 3: PLANT CATEGORIES
###############################################################################################################################################################################

        with tab3:
            try:
                st.markdown("### 🌲 Plant Categories")
                
                # Define available categorical columns
                categorical_columns = {
                    'forC': 'Forest Categories',
                    'gruppe': 'Groups',
                    'livsform': 'Life Forms'
                }
                
                # Column selection dropdown
                selected_column = st.selectbox(
                    "Select category type:",
                    options=list(categorical_columns.keys()),
                    format_func=lambda x: categorical_columns[x],
                    key="category_column_select"
                )
                
                if selected_column not in taxa_df.columns:
                    st.info(f"No {categorical_columns[selected_column].lower()} data available in this dataset")
                else:
                    # Get unique categories from selected column
                    categories = taxa_df[selected_column].dropna().unique()
                    
                    if len(categories) == 0:
                        st.warning(f"No {categorical_columns[selected_column].lower()} found in data")
                    else:
                        # Define labels for forC (keep existing labels for backward compatibility)
                        category_labels = {}
                        if selected_column == 'forC':
                            category_labels = {
                                '1_1': 'Shade-tolerant forest species',
                                '1_2': 'Moderate shade forest species',
                                '2_1': 'Forest edge species',
                                '2_2': 'Forest clearing species',
                                'O': 'Open habitat species'
                            }
                        
                        # Dynamic multiselect based on selected column
                        selected_categories = st.multiselect(
                            f"Select {categorical_columns[selected_column].lower()}:",
                            options=sorted(categories),
                            format_func=lambda x: category_labels.get(x, str(x)),
                            default=[categories[0]] if len(categories) > 0 else []
                        )
                        
                        if not selected_categories:
                            st.info(f"Please select at least one {categorical_columns[selected_column].lower()}")
                        else:
                            # Filter data based on selected column and categories
                            filtered_data = taxa_df[taxa_df[selected_column].isin(selected_categories)]
                            
                            # Validate coordinates
                            if not validate_required_columns(filtered_data, ['xcoor', 'ycoor']):
                                st.error("Coordinate data not available for mapping")
                            else:
                                filtered_data = filtered_data.dropna(subset=['xcoor', 'ycoor'])
                                filtered_data = filtered_data[
                                    (filtered_data['xcoor'] >= 0) & (filtered_data['xcoor'] <= 1) &
                                    (filtered_data['ycoor'] >= 0) & (filtered_data['ycoor'] <= 1)
                                ]
                                
                                if len(filtered_data) == 0:
                                    st.warning(f"No valid coordinate data for selected {categorical_columns[selected_column].lower()}")
                                else:
                                    with st.expander("Show selected species:"):
                                        display_columns = ['keyword', selected_column, 'xcoor', 'ycoor']
                                        st.dataframe(filtered_data[display_columns])
                                    
                                    # Create dynamic map title
                                    map_title = f"Plant {categorical_columns[selected_column]}"
                                    fig, ax = create_base_map(map_title)
                                    
                                    # KDE shading
                                    if st.checkbox("Show density shading", value=True, key=f"{selected_column}_kde"):
                                        safe_kde_plot(
                                            filtered_data, 'xcoor', 'ycoor', ax,
                                            fill=True, cmap='Greens', levels=20,
                                            thresh=0.05, alpha=0.5
                                        )
                                    
                                    # Plot points by category
                                    colors = get_color_palette(len(selected_categories), 'Set2')
                                    
                                    for i, category in enumerate(selected_categories):
                                        cat_data = filtered_data[filtered_data[selected_column] == category]
                                        if len(cat_data) > 0:
                                            safe_scatter_plot(
                                                ax, cat_data['xcoor'], cat_data['ycoor'],
                                                s=25, alpha=0.8, c=[colors[i]],
                                                label=str(category),
                                                edgecolors='black', linewidths=0.5
                                            )
                                    
                                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                                    
                                    zoom = st.slider("Zoom:", -0.1, 0.25, -0.05, key=f"{selected_column}_zoom")
                                    ax.set_xlim(zoom, 1 - zoom)
                                    ax.set_ylim(zoom, 1 - zoom)
                                    
                                    st.pyplot(fig)
                                    
                                    # Save option with dynamic naming
                                    col1, col2 = st.columns([2, 1])
                                    with col1:
                                        default_name = f"{selected_column}_categories"
                                        save_name = st.text_input("Save as:", value=default_name, key=f"save_{selected_column}")
                                    with col2:
                                        if st.button("💾 Save Map", key=f"save_btn_{selected_column}"):
                                            figures_path = Path(st.session_state.get('figures_path', '.'))
                                            if safe_figure_save(fig, save_name, figures_path):
                                                st.success(f"Map saved as {save_name}.png")
                                    
                                    plt.close(fig)
            
            except Exception as e:
                st.error(f"Error in Categories tab: {str(e)}")
                st.code(traceback.format_exc())
      
      
    except Exception as e:
        st.error(f"Critical error in main mapping interface: {str(e)}")
        st.code(traceback.format_exc())

else:
    st.info("👆 Please load a map database to begin creating visualizations")

# Memory cleanup
try:
    plt.close('all')
except:
    pass

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d; font-size: 0.9em;'>
    EcoNetMap - Network-based ecological mapping
    </div>
    """, 
    unsafe_allow_html=True
)
