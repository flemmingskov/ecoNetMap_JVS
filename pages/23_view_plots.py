"""
EcoNetMap - Plot Mapping Module
===============================================
This module visualizes vegetation plots in both ecological and geographic space.
It creates distribution maps of plots colored by habitat types or environmental
conditions, analyzes temporal changes in plot positions, and generates geographic
maps with UTM coordinates. The module includes density analysis, spatial pattern
detection, and environmental gradient visualization at the plot level.

Part of the EcoNetMap toolkit (mapping 2/2)
Author: Flemming Skov (fs@ecos.au.dk)
Last Updated: July 2025
"""

# Import packages for web applications
import streamlit as st

# Import packages for data manipulation and analysis
import pandas as pd
import numpy as np
import sqlite3

# Import packages for file and system operations
from pathlib import Path
from matplotlib.path import Path as MplPath 
import datetime
import warnings

# Import packages for type hints
from typing import Optional, Tuple, List, Dict

# Import packages for visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

# Import packages for GIS functionality
import contextily as ctx
ctx.set_cache_dir("./map_cache")  # Creates local cache folder

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.filterwarnings('ignore', message='.*Arrow.*')

# Page configuration
st.set_page_config(
    page_title="Mapping Survey plots", 
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
</style>
""", unsafe_allow_html=True)

# Title and progress indicator
col1, col2 = st.columns([4, 1])
with col1:
    st.header("Maps")
    st.subheader("🌿 Habitat maps")
    st.markdown("*Mapping habitat distribution in ecological and geographic space and Ellemberg profiles*")
with col2:
    pass
st.markdown("---")


###################################################################################
# FUNCTIONS
###################################################################################

@st.cache_data(show_spinner=False)
def load_map_data(db_path: str) -> Dict[str, pd.DataFrame]:
    """Load all relevant tables from map database"""
    try:
        conn = sqlite3.connect(db_path)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        
        data_dict = {}
        for table in tables['name'].values:
            if not table.startswith('_'):  # Skip metadata tables
                data_dict[table] = pd.read_sql_query(f'SELECT * FROM {table}', conn)
        
        conn.close()
        return data_dict
    except Exception as e:
        st.error(f"Error loading map data: {str(e)}")
        return {}

def create_base_map(title: str = '', figsize: Tuple[int, int] = (12, 12)) -> Tuple[plt.Figure, plt.Axes]:
    """Create a base map with guide circles and lines"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up the plot
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
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
    ax.set_title(title, fontsize=16, pad=20)
    ax.grid(False)
    
    return fig, ax

def get_color_palette(n_colors: int, palette_name: str = 'viridis') -> List:
    """Get a color palette with the specified number of colors"""
    if n_colors <= 20:
        return sns.color_palette(palette_name, n_colors)
    else:
        return sns.color_palette('husl', n_colors)

def save_figure(fig: plt.Figure, filename: str, figures_path: Path) -> bool:
    """Save figure to file"""
    try:
        if not figures_path.exists():
            figures_path.mkdir(parents=True, exist_ok=True)
        
        filepath = figures_path / f"{filename}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.1)
        return True
    except Exception as e:
        st.error(f"Error saving figure: {str(e)}")
        return False

#### TEST AREA

from scipy.stats import gaussian_kde
import numpy as np

def draw_ecological_contour(ax, subset_df, label, color,
                             percentile=20, alpha=0.4,
                             label_alpha=0.6, rotation=0,
                             repel_from=None, linewidth=1.8, linestyle='dashed'):
    """
    Draw an outer KDE contour around plots meeting an ecological condition.
    
    Parameters:
    -----------
    repel_from : tuple (x, y) or None
        If provided, the label is placed at the contour boundary point 
        furthest from this coordinate — useful to separate overlapping labels.
    """
    if len(subset_df) < 15:
        return
    
    x = subset_df['xcoor'].values
    y = subset_df['ycoor'].values
    
    kde = gaussian_kde(np.vstack([x, y]), bw_method=0.25)
    
    xi = np.linspace(0, 1, 150)
    yi = np.linspace(0, 1, 150)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
    
    point_densities = kde(np.vstack([x, y]))
    threshold = np.percentile(point_densities, percentile)
    
    cs = ax.contour(Xi, Yi, Zi, levels=[threshold],
                    colors=[color], alpha=alpha, linewidths=linewidth,
                    linestyles=linestyle)
    
    # Label placement: furthest contour point from repel_from, 
    # or centroid fallback
    label_x, label_y = x.mean(), y.mean()  # default fallback
    
    if repel_from is not None:
        paths = cs.get_paths()
        if paths:
            all_verts = np.concatenate([p.vertices for p in paths], axis=0)
            dists = np.sqrt((all_verts[:, 0] - repel_from[0])**2 + 
                            (all_verts[:, 1] - repel_from[1])**2)
            furthest = all_verts[np.argmax(dists)]
            label_x, label_y = furthest
    else:
        # Centroid of the contour region
        mask = Zi >= threshold
        if mask.any():
            label_x = Xi[mask].mean()
            label_y = Yi[mask].mean()
    
    ax.text(label_x, label_y, label,
            fontsize=10, color=color, fontweight='bold',
            ha='center', va='center', alpha=label_alpha,
            rotation=rotation,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                      alpha=0.4, edgecolor='none'))


def create_annotated_background(ax: plt.Axes, plot_df: pd.DataFrame, text_alpha: float = 0.1, line_alpha: float = 0.2, nr_percentile: float = 15, l_percentile: float = 12.5, salt_distance: float = 65) -> None:
    """
    Render a multi-layer ecological backdrop onto an existing matplotlib axes object.

    The background consists of five stacked layers, drawn in order from bottom to top:
      1. Moisture zones      — hexbin cells shaded by Ellenberg M (dry / mesic / wet)
      2. Low-light overlay   — dark green hexbin borders for low Ellenberg L plots,
                               fading outward from the distribution centroid
      3. Halophytic overlay  — steel blue hexbin borders for nature type 1330,
                               intensity scaled by local plot density
      4. Ecological contours — KDE contours marking nutrient-poor (N) and low-pH (R) zones
      5. Text labels         — 'Shade - Forest', 'Salty', 'Wet', 'Dry' placed at distribution centres

    Parameters:
    -----------
    ax : plt.Axes
        The matplotlib axes to draw on. Must already have xlim/ylim set to [0, 1].
    plot_df : pd.DataFrame
        Full (unfiltered) plot data. Required columns: xcoor, ycoor, M.
        Optional columns used when present: N, R, L, habitat_type.
    text_alpha : float
        Opacity for all text labels and contour label boxes (default 0.1).
        Exposed as a Streamlit slider so the user can dim all text together.
    line_alpha : float
        Opacity for KDE contour lines and hexagon overlay borders (default 0.2).
        Exposed as a Streamlit slider so the user can dim all overlays together.
    nr_percentile : float
        Percentile threshold for defining nutrient-poor (N) and low-pH (R) zones (default 15).
        Lower values = more extreme/rare conditions only. Exposed as a Streamlit slider.
    l_percentile : float
        Percentile threshold for defining low-light (shade/forest) zones (default 12.5).
        Lower values = darker/more extreme shade only. Exposed as a Streamlit slider.
    salt_distance : float
        Percentile distance cutoff for halophytic zone display (default 65).
        Controls how far from core the salt zone extends. Higher = tighter boundaries.
    """

    # -------------------------------------------------------------------------
    # SETUP & VALIDATION
    # -------------------------------------------------------------------------

    # Minimum columns needed to proceed
    required_cols = ['xcoor', 'ycoor', 'M']
    if not all(col in plot_df.columns for col in required_cols):
        st.warning("Missing required columns for annotated background (xcoor, ycoor, M)")
        return

    # Drop rows with missing coordinates or moisture values
    valid_data = plot_df.dropna(subset=['xcoor', 'ycoor', 'M'])
    if len(valid_data) == 0:
        st.warning("No valid data for annotated background")
        return

    # Shared hexbin parameters
    GRIDSIZE    = 20                    # Hexagon grid resolution
    EXTENT      = (0.0, 1.0, 0.0, 1.0) # Match the ecological space coordinate range
    MINCNT      = 10                     # Minimum plots per hex cell for wet/dry overdraws
    MINCNT_RARE = 10                     # Lower threshold for rare/sparse distributions

    # RGBA fill colours for moisture zones (subtle pastels)
    moisture_colors = {
        'wet':   (0.78, 0.87, 0.93, 1.0),  # Blue tint
        'mesic': (0.91, 0.94, 0.97, 1.0),  # Near-neutral blue-grey
        'dry':   (0.97, 0.94, 0.89, 1.0),  # Warm beige
    }

    # -------------------------------------------------------------------------
    # THRESHOLDS
    # -------------------------------------------------------------------------

    # Moisture (M): bottom 10% = dry, middle 80% = mesic, top 10% = wet
    M_low  = np.percentile(valid_data['M'].dropna(), 10)
    M_high = np.percentile(valid_data['M'].dropna(), 90)

    # -------------------------------------------------------------------------
    # LAYER 1: MOISTURE BACKGROUND ZONES
    # A full mesic base layer is drawn first with mincnt=MINCNT so that every
    # hex cell containing at least one plot is coloured — this eliminates white
    # cells entirely. Wet and dry zones are then overdrawn on top using the
    # stricter MINCNT threshold so only well-sampled cells change colour.
    # vmin=0.5 ensures zero-count cells fall below the colour range and are
    # rendered transparent rather than white via set_under/set_bad.
    # -------------------------------------------------------------------------

    wet_df  = valid_data[valid_data['M'] > M_high]
    dry_df  = valid_data[valid_data['M'] < M_low]

    # Base layer: ALL valid plots in mesic colour, mincnt=MINCNT to leave no gaps
    cmap_base = ListedColormap([moisture_colors['mesic']])
    cmap_base.set_bad(color=(0, 0, 0, 0))
    cmap_base.set_under(color=(0, 0, 0, 0))
    ax.hexbin(
        valid_data['xcoor'], valid_data['ycoor'],
        gridsize=GRIDSIZE, extent=EXTENT, mincnt=MINCNT,
        cmap=cmap_base, vmin=0.5,
        edgecolors='lightgrey', alpha=1.0, linewidth=0.2
    )

    # Overdraw wet and dry zones on top of the mesic base
    for subset, key in [(wet_df, 'wet'), (dry_df, 'dry')]:
        if len(subset) > 0:
            cmap = ListedColormap([moisture_colors[key]])
            cmap.set_bad(color=(0, 0, 0, 0))
            cmap.set_under(color=(0, 0, 0, 0))
            ax.hexbin(
                subset['xcoor'], subset['ycoor'],
                gridsize=GRIDSIZE, extent=EXTENT, mincnt=MINCNT,
                cmap=cmap, vmin=0.5,
                edgecolors='lightgrey', alpha=1.0, linewidth=0.2
            )

    # -------------------------------------------------------------------------
    # LAYER 2: LOW-LIGHT OVERLAY (Ellenberg L)
    # Hexagons occupied by low-L plots get a dark green border. The border
    # alpha and linewidth both decrease with distance from the distribution
    # centroid, creating a vignette effect — denser core = stronger signal.
    # Fill is transparent so the moisture colours beneath remain visible.
    # The per-hexagon alphas are scaled by line_alpha so the overlay slider
    # dims these borders together with the contours.
    # -------------------------------------------------------------------------

    if 'L' in valid_data.columns:
        # Select the lowest N% of L values to define the low-light zone (user-configurable)
        threshold_L = np.percentile(valid_data['L'].dropna(), l_percentile)
        low_L = valid_data[valid_data['L'] <= threshold_L]
        centroid_L = (low_L['xcoor'].mean(), low_L['ycoor'].mean())

        if len(low_L) > 0:
            hb_light = ax.hexbin(
                low_L['xcoor'], low_L['ycoor'],
                gridsize=GRIDSIZE, extent=EXTENT, mincnt=MINCNT_RARE,
                linewidth=1.2
            )
            hb_light.set_facecolor('none')

            # Compute normalised distance of each hex centre from the centroid
            offsets   = hb_light.get_offsets()
            cx, cy    = centroid_L
            distances = np.sqrt((offsets[:, 0] - cx)**2 + (offsets[:, 1] - cy)**2)
            norm_dist = distances / distances.max()

            # Alpha fades outward, then scaled globally by line_alpha slider
            alphas     = (0.5 - 0.4 * norm_dist) * line_alpha / 0.75
            linewidths = 2.5 - 1.7 * norm_dist                        # 2.5 → 0.8

            # Build per-hexagon RGBA edge colours (darkgreen = 0.0, 0.392, 0.0)
            r, g, b = 0.0, 0.392, 0.0
            edge_colors = np.column_stack([
                np.full(len(alphas), r),
                np.full(len(alphas), g),
                np.full(len(alphas), b),
                np.clip(alphas, 0.0, 1.0)
            ])
            hb_light.set_edgecolors(edge_colors)
            hb_light.set_linewidths(linewidths)

    # -------------------------------------------------------------------------
    # LAYER 3: HALOPHYTIC VEGETATION OVERLAY (nature type 1330)
    # Steel blue borders mark hexagons containing salt-tolerant coastal plots.
    # Border intensity scales with local plot density (from hexbin counts)
    # rather than distance, so denser clusters read as more prominent.
    # The per-hexagon alphas are scaled by line_alpha so the overlay slider
    # dims these borders together with the contours.
    # -------------------------------------------------------------------------

    if 'habitat_type' in valid_data.columns:
        # habitat_type may be stored as int or string depending on the database
        halo_df = valid_data[
            (valid_data['habitat_type'] == 1330) |
            (valid_data['habitat_type'] == '1330')
        ]

        if len(halo_df) > 0:
            centroid_halo = (halo_df['xcoor'].mean(), halo_df['ycoor'].mean())

            hb_halo = ax.hexbin(
                halo_df['xcoor'], halo_df['ycoor'],
                gridsize=GRIDSIZE, extent=EXTENT, mincnt=MINCNT_RARE,
                linewidth=1.2
            )
            hb_halo.set_facecolor('none')

            # Compute distance of each hex centre from the distribution centroid
            offsets   = hb_halo.get_offsets()
            cx, cy    = centroid_halo
            distances = np.sqrt((offsets[:, 0] - cx)**2 + (offsets[:, 1] - cy)**2)

            # Only keep hexagons within the core N% of the distribution (user-configurable)
            dist_threshold = np.percentile(distances, salt_distance)
            core_mask = distances <= dist_threshold

            # Normalise counts to [0, 1] for scaling alpha and linewidth
            counts      = hb_halo.get_array()
            norm_counts = counts / counts.max()

            # Dense hexagons get strong borders, scaled globally by line_alpha slider
            # Outer hexagons (beyond 75th percentile) are made fully transparent
            alphas     = (0.15 + 0.75 * norm_counts) * line_alpha / 0.8
            alphas     = np.where(core_mask, alphas, 0.0)
            linewidths = np.where(core_mask, 0.8 + 1.7 * norm_counts, 0.0)

            # Steel blue RGB (0.27, 0.51, 0.71) — coastal / saline feel
            r, g, b = 0.27, 0.51, 0.71
            edge_colors = np.column_stack([
                np.full(len(alphas), r),
                np.full(len(alphas), g),
                np.full(len(alphas), b),
                np.clip(alphas, 0.0, 1.0)
            ])
            hb_halo.set_edgecolors(edge_colors)
            hb_halo.set_linewidths(linewidths)    

    # -------------------------------------------------------------------------
    # LAYER 4: ECOLOGICAL GRADIENT CONTOURS (N and R)
    # KDE contours outline areas dominated by nutrient-poor and low-pH plots.
    # Label positions are repelled away from the opposite corner of the map
    # to guarantee separation. Both line opacity (line_alpha) and label opacity
    # (text_alpha) are controlled by the Streamlit sliders.
    # -------------------------------------------------------------------------

    if all(col in valid_data.columns for col in ['N', 'R']):
        # Select the lowest N% of N and R values (user-configurable)
        threshold_N = np.percentile(valid_data['N'].dropna(), nr_percentile)
        threshold_R = np.percentile(valid_data['R'].dropna(), nr_percentile)

        low_N = valid_data[valid_data['N'] <= threshold_N]
        low_R = valid_data[valid_data['R'] <= threshold_R]

        # Nutrient-poor: solid brown contour, label pushed toward top-left
        draw_ecological_contour(ax, low_N,
            label='Nutrient\npoor', color='saddlebrown',
            percentile=40, alpha=line_alpha, label_alpha=text_alpha,
            rotation=0, repel_from=(1.0, 0.0), linewidth=3.5, linestyle='solid')

        # Low pH: dashed dark blue contour, label pushed toward bottom-right
        draw_ecological_contour(ax, low_R,
            label='Low pH', color='darkblue',
            percentile=40, alpha=line_alpha, label_alpha=text_alpha,
            rotation=0, repel_from=(0.0, 1.0))

    # -------------------------------------------------------------------------
    # LAYER 5: TEXT LABELS
    # Simple text placed at the median/centroid of ecologically extreme plots.
    # All labels are black for a clean, consistent look.
    # All labels share text_alpha so they can be dimmed together.
    # -------------------------------------------------------------------------

    # Shade - Forest — placed at the centroid of the low-L distribution
    if 'L' in valid_data.columns and 'centroid_L' in dir():
        ax.text(
            centroid_L[0], centroid_L[1]+0.08, 'Shade',
            fontsize=12, color='black', fontweight='bold',
            ha='center', va='center', alpha=text_alpha
        )

    # Salty — placed at the centroid of the halophytic (1330) distribution
    if 'habitat_type' in valid_data.columns and 'centroid_halo' in dir():
        ax.text(
            centroid_halo[0], centroid_halo[1], 'Salty',
            fontsize=12, color='black', fontweight='bold',
            ha='center', va='center', alpha=text_alpha
        )

    # Wet — placed at the median position of very wet plots (M > 8), nudged 25% toward centre
    high_M = valid_data[valid_data['M'] > 8]
    if len(high_M) > 10:
        cx, cy = high_M['xcoor'].median(), high_M['ycoor'].median()
        cx = cx + (0.5 - cx) * 0.25
        cy = cy + (0.5 - cy) * 0.25
        ax.text(cx, cy, 'Wet',
                fontsize=12, color='black', fontweight='bold',
                ha='center', va='center', alpha=text_alpha)

    # Dry — placed at the median position of very dry plots (M < 3.5)
    low_M = valid_data[valid_data['M'] < 3.5]
    if len(low_M) > 10:
        cx, cy = low_M['xcoor'].median(), low_M['ycoor'].median()
        ax.text(cx, cy, 'Dry',
                fontsize=12, color='black', fontweight='bold',
                ha='center', va='center', alpha=text_alpha)


###################################################################################
# Main interface
###################################################################################

# Database selection
st.markdown("### 📁 Select Map Databases")

overlay_path = Path(st.session_state.get('overlay_map_path', '.'))
if not overlay_path.exists():
    st.error(f"Overlay map directory not found: {overlay_path}")
    st.stop()

db_files = sorted([f.name for f in overlay_path.glob("*.db")])

if not db_files:
    st.error("No map databases found.")
    st.info("Please complete Step 6 (Prepare Map) first.")
    st.stop()

# Database selectors and single load button
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    selected_overlay_db = st.selectbox(
        "Overlay map (plot analysis):",
        options=db_files,
        key="overlay_db_select",
        help="Choose the map database for analyzing plot distributions"
    )
    overlay_db_path = overlay_path / selected_overlay_db

with col2:
    selected_reference_db = st.selectbox(
        "Reference map (background annotation):",
        options=db_files,
        key="reference_db_select",
        help="Choose the map database for background annotations (typically a complete map)"
    )
    reference_db_path = overlay_path / selected_reference_db

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("📂 Load Maps", type="primary", use_container_width=True, key="load_both_maps"):
        map_data = load_map_data(str(overlay_db_path))
        if map_data:
            st.session_state.plot_map_data = map_data
            st.session_state.plot_map_name = selected_overlay_db
        ref_data = load_map_data(str(reference_db_path))
        if ref_data:
            st.session_state.reference_map_data = ref_data
            st.session_state.reference_map_name = selected_reference_db

# Show analysis map info if loaded
if 'plot_map_data' in st.session_state:
    map_data = st.session_state.plot_map_data

    # Check for required table
    if 'plot_id' not in map_data:
        st.error("Plot data (plot_id table) not found in analysis map database")
        st.stop()

    plot_df = map_data['plot_id']

    # Display summary statistics for analysis map
    with st.expander(f"📊 Analysis Map Summary: {st.session_state.get('plot_map_name', 'Unknown')}", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Plots", f"{plot_df['plot_id'].nunique():,}")
        with col2:
            if 'habitat_type' in plot_df.columns:
                st.metric("Habitat Types", f"{plot_df['habitat_type'].nunique():,}")
            else:
                st.metric("Habitat Types", "N/A")
        with col3:
            if 'major_type' in plot_df.columns:
                st.metric("Major Types", f"{plot_df['major_type'].nunique():,}")
            else:
                st.metric("Major Types", "N/A")
        with col4:
            if 'speciesNum' in plot_df.columns:
                st.metric("Avg Species/Plot", f"{plot_df['speciesNum'].mean():.1f}")
            else:
                st.metric("Avg Species/Plot", "N/A")

# Show reference map info if loaded
if 'reference_map_data' in st.session_state:
    ref_data = st.session_state.reference_map_data

    if 'plot_id' in ref_data:
        ref_plot_df = ref_data['plot_id']

        with st.expander(f"🗺️ Reference Map Summary: {st.session_state.get('reference_map_name', 'Unknown')}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Plots", f"{ref_plot_df['plot_id'].nunique():,}")
            with col2:
                if 'habitat_type' in ref_plot_df.columns:
                    st.metric("Habitat Types", f"{ref_plot_df['habitat_type'].nunique():,}")
                else:
                    st.metric("Habitat Types", "N/A")
            with col3:
                if 'major_type' in ref_plot_df.columns:
                    st.metric("Major Types", f"{ref_plot_df['major_type'].nunique():,}")
                else:
                    st.metric("Major Types", "N/A")
            with col4:
                if 'speciesNum' in ref_plot_df.columns:
                    st.metric("Avg Species/Plot", f"{ref_plot_df['speciesNum'].mean():.1f}")
                else:
                    st.metric("Avg Species/Plot", "N/A")
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs([
        "📍 Plot Distribution",
        "📊 Temporal Changes"
    ])    
        
    
    with tab1:
        st.markdown("### 📍 Plot Distribution Analysis")
        
        # Filtering options
        st.markdown("#### 🔍 Filter plots")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Regional filters in expander
            with st.expander("🗺️ Regional Filters", expanded=False):
                # NOVANA Region filter
                if 'region' in plot_df.columns:
                    regions = plot_df['region'].dropna().unique()
                    selected_regions = st.multiselect(
                        "Select monitoring regions:",
                        options=sorted(regions),
                        default=list(regions)
                    )
                    filtered_plots = plot_df[plot_df['region'].isin(selected_regions)] if selected_regions else plot_df
                else:
                    filtered_plots = plot_df

                # Bioregion filter
                if 'subregion' in filtered_plots.columns:
                    bioregions = filtered_plots['subregion'].dropna().unique()
                    selected_bioregions = st.multiselect(
                        "Select bioregions:",
                        options=sorted(bioregions),
                        default=list(bioregions)
                    )
                    if selected_bioregions:
                        filtered_plots = filtered_plots[filtered_plots['subregion'].isin(selected_bioregions)]
        
        with col2:
            # Habitat filters in expander
            with st.expander("🏞️ Vegetation & Habitat Type Filters", expanded=False):
                # Initialise filter variables — overwritten below if columns are present
                selected_major_types = []
                selected_habitats = []
                # Major type filter - default to ALL major types
                if 'major_type' in filtered_plots.columns:
                    major_types = filtered_plots['major_type'].dropna().unique()
                    selected_major_types = st.multiselect(
                        "Select major vegetation types:",
                        options=sorted(major_types),
                        default=list(major_types)  # Default to ALL major types
                    )
                    if selected_major_types:
                        filtered_plots = filtered_plots[filtered_plots['major_type'].isin(selected_major_types)]

                # Habitat type filter - filtered based on selected major types if available
                if 'habitat_type' in filtered_plots.columns:
                    # Get habitat types that belong to selected major types
                    if selected_major_types:
                        available_habitats = filtered_plots[
                            filtered_plots['major_type'].isin(selected_major_types)
                        ]['habitat_type'].dropna().unique()
                    else:
                        available_habitats = filtered_plots['habitat_type'].dropna().unique()

                    habitat_types = sorted(available_habitats)
                    selected_habitats = st.multiselect(
                        "Select specific habitat types:",
                        options=habitat_types,
                        default=habitat_types  # Default to all available habitats
                    )
                    if selected_habitats:
                        filtered_plots = filtered_plots[filtered_plots['habitat_type'].isin(selected_habitats)]
        
        st.info(f"Showing {len(filtered_plots)} plots after filtering")
        
        # Map settings
        st.markdown("#### 🎨 Map Settings")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Background Layers:**")
            show_background = st.checkbox("Show background (all plots)", value=True)
            show_background_annotated = st.checkbox("Show background (annotated)", value=False)

        with col2:
            st.markdown("**Plot Layers:**")
            show_scatter = st.checkbox("Scatter plot", value=True)
            show_density_contours = st.checkbox("Density contours (50/90)", value=False)
            show_density_contours2 = st.checkbox("Density contours all", value=False)
            show_density_shading = st.checkbox("Density shading", value=False)

        with col3:
            st.markdown("**Style Options:**")
            if 'major_type' in filtered_plots.columns and 'habitat_type' in filtered_plots.columns:
                legend_by = st.radio(
                    "Color/Legend by:",
                    options=["habitat_type", "major_type"],
                    index=1
                )
            elif 'habitat_type' in filtered_plots.columns:
                legend_by = 'habitat_type'
            elif 'major_type' in filtered_plots.columns:
                legend_by = 'major_type'
            else:
                legend_by = None

            if 'speciesNum' in filtered_plots.columns:
                size_by_species = st.checkbox("Size by species count", value=True)
            else:
                size_by_species = False

        # Advanced settings in expander
        with st.expander("⚙️ Advanced Map Settings", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Transparency Controls:**")
                text_alpha = st.slider("Text transparency:", 0.0, 1.0, 0.7, 0.05, key="text_alpha1")
                line_alpha = st.slider("Overlay transparency:", 0.0, 1.0, 0.2, 0.05, key="line_alpha1")
                scatter_alpha = st.slider("Scatter transparency:", 0.1, 1.0, 0.7, 0.05, key="scatter_alpha_tab1")

            with col2:
                st.markdown("**View Controls:**")
                zoom = st.slider("Zoom:", -0.1, 0.25, 0.0)
                kde_thresh = st.slider("Density threshold:", 0.01, 0.05, 0.005)

            with col3:
                st.markdown("**Annotation Thresholds:**")
                nr_percentile = st.slider("N/R threshold (%):", 5, 50, 15, 1, key="nr_percentile1",
                    help="Percentile for nutrient-poor/low-pH zones. Lower = more extreme conditions only.")
                l_percentile = st.slider("Light threshold (%):", 5.0, 25.0, 12.5, 0.5, key="l_percentile1",
                    help="Percentile for shade/forest zones. Lower = darker shade only.")
                salt_distance = st.slider("Salt zone extent (%):", 50, 90, 65, 5, key="salt_distance1",
                    help="Distance cutoff for halophytic zones. Higher = tighter boundaries around core.")
         
        # Create the map
        fig, ax = create_base_map("Plot distribution")

        # Get reference data if available, otherwise use plot data
        if 'reference_map_data' in st.session_state and 'plot_id' in st.session_state.reference_map_data:
            reference_df = st.session_state.reference_map_data['plot_id']
        else:
            reference_df = plot_df

        # Plot background layer first (all plots in light grey) if requested
        if show_background:
            ax.scatter(
                reference_df['xcoor'], reference_df['ycoor'],
                s=40, alpha=0.45, c='lightgrey',
                edgecolors='lightgrey', linewidths=0.05
            )


        # Add annotated background if requested (after basic background, before other layers)
        if show_background_annotated:
            create_annotated_background(ax, reference_df, text_alpha, line_alpha, nr_percentile, l_percentile, salt_distance)
        
        # Apply visualizations based on checkboxes
        if show_density_shading and len(filtered_plots) > 2:
            try:
                sns.kdeplot(
                    data=filtered_plots, x='xcoor', y='ycoor',
                    fill=True, cmap='Blues', levels=20,
                    thresh=kde_thresh, alpha=0.5, ax=ax
                )
            except:
                st.warning("Could not create density shading")
        
        if show_density_contours and len(filtered_plots) > 2:
            try:
                sns.kdeplot(
                    data=filtered_plots, x='xcoor', y='ycoor',
                    fill=False, levels=2,
                    thresh=0.1, alpha=1, color='blue', ax=ax
                )
                sns.kdeplot(
                    data=filtered_plots, x='xcoor', y='ycoor',
                    fill=False, levels=2,
                    thresh=0.5, alpha=1, color='red', ax=ax
                )
            except:
                st.warning("Could not create density contours")


        if show_density_contours2 and len(filtered_plots) > 2:
            try:
                sns.kdeplot(
                    data=filtered_plots, x='xcoor', y='ycoor',
                    fill=False, levels=24, linewidths=0.56,
                    thresh=0.05, alpha=0.8, color='black', ax=ax
                )
            except:
                st.warning("Could not create density contours")


      
        if show_scatter:
            # Determine sizes
            if size_by_species and 'speciesNum' in filtered_plots.columns:
                sizes = filtered_plots['speciesNum'] * 5
            else:
                sizes = 12
            
            # Plot by category if specified
            if legend_by and legend_by in filtered_plots.columns:
                unique_values = filtered_plots[legend_by].dropna().unique()
                colors = get_color_palette(len(unique_values))
                
                for i, value in enumerate(unique_values):
                    subset = filtered_plots[filtered_plots[legend_by] == value]
                    if size_by_species and 'speciesNum' in subset.columns:
                        subset_sizes = subset['speciesNum'] * 4
                    else:
                        subset_sizes = 12
                    
                    ax.scatter(
                        subset['xcoor'], subset['ycoor'],
                        s=subset_sizes, alpha=scatter_alpha, c=[colors[i]],
                        label=str(value)[:30], linewidths=0.5
                    )
                
                if len(unique_values) <= 20:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                             ncol=1 if len(unique_values) <= 10 else 2)
            else:
                ax.scatter(
                    filtered_plots['xcoor'], filtered_plots['ycoor'],
                    s=sizes, alpha=scatter_alpha, c='darkgreen',
                    edgecolors='black', linewidths=0.5
                )
        
        # Center star: wdegree-weighted centroid of all species coordinates
        # This is the same reference point used to compute mahal_dist in 11_network_layout.py
        if 'reference_map_data' in st.session_state:
            ref_taxa = st.session_state.reference_map_data.get('taxa')
            if ref_taxa is not None:
                valid_species = ref_taxa.dropna(subset=['xcoor', 'ycoor'])
                if len(valid_species) > 0:
                    if 'wdegree' in valid_species.columns and valid_species['wdegree'].sum() > 0:
                        _w = valid_species['wdegree'].fillna(0).values
                        cx = np.average(valid_species['xcoor'].values, weights=_w)
                        cy = np.average(valid_species['ycoor'].values, weights=_w)
                    else:
                        cx = valid_species['xcoor'].mean()
                        cy = valid_species['ycoor'].mean()
                    ax.scatter([cx], [cy], s=150, color='red', marker='*',
                               edgecolor='darkred', linewidth=2, zorder=10)

        ax.set_xlim(zoom, 1 - zoom)
        ax.set_ylim(zoom, 1 - zoom)

        st.pyplot(fig)

        # Save option
        col1, col2 = st.columns([2, 1])
        with col1:
            save_name = st.text_input("Save as:", value="plot_distribution", key="save_plot1")
        with col2:
            if st.button("💾 Save Map", key="save_btn_plot1"):
                figures_path = Path(st.session_state.get('figures_path', '.'))
                if save_figure(fig, save_name, figures_path):
                    st.success(f"Map saved as {save_name}.png")
        
        plt.close()
        
        # Geographic map section
        st.markdown("---")
        st.markdown("### 🌍 Geographic Distribution")
        
        # Initialise geographic settings — overwritten below if UTM data is present
        use_fixed_extent = True
        geo_show_background = False

        # Check if UTM coordinates are available
        if 'x' in filtered_plots.columns and 'y' in filtered_plots.columns:
            # Remove any plots with missing coordinates
            geo_plots = filtered_plots.dropna(subset=['x', 'y'])
            dk_dot_map = st.session_state.get('df_regional_pool', None)          
            
            if len(geo_plots) > 0:
                # Create geographic map
                fig_geo, ax_geo = plt.subplots(figsize=(12, 10))
                
                # Map settings for geographic view
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    geo_show_background = st.checkbox(
                        "Show geographic background", 
                        value=True,
                        key="geo_background",
                        help="Show all available geographic points as background"
                    )
                    
                    use_fixed_extent = st.checkbox(
                        "Fixed map extent (full extent)",
                        value=True,
                        key="fixed_extent",
                        help="Always show same area vs zoom to data"
                    )
                    
                    geo_color_options = [
                        col for col in ['habitat_type', 'major_type', 'region', 'subregion']
                        if col in geo_plots.columns
                    ] + ['None']
                    geo_color_by = st.selectbox(
                        "Color geographic points by:",
                        options=geo_color_options,
                        index=0,
                        key="geo_color"
                    )
                
                with col2:
                    geo_point_size = st.slider("Point size:", 5, 50, 15, key="geo_size")
                    
                with col3:
                    show_geo_density = st.checkbox("Show density overlay", value=False, key="geo_density")
                
                # Plot based on coloring choice
                if geo_color_by != 'None' and geo_color_by in geo_plots.columns:
                    unique_values = geo_plots[geo_color_by].dropna().unique()
                    colors = get_color_palette(len(unique_values))
                    
                    for i, value in enumerate(unique_values):
                        subset = geo_plots[geo_plots[geo_color_by] == value]
                        ax_geo.scatter(
                            subset['x'], subset['y'],
                            s=geo_point_size, alpha=0.7, c=[colors[i]],
                            label=str(value)[:30], edgecolors='black', linewidths=0.5
                        )
                    
                    if len(unique_values) <= 20:
                        ax_geo.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                                     ncol=1 if len(unique_values) <= 10 else 2)
                else:
                    ax_geo.scatter(
                        geo_plots['x'], geo_plots['y'],
                        s=geo_point_size, alpha=0.7, c='darkgreen',
                        edgecolors='black', linewidths=0.5
                    )
                
                
                if use_fixed_extent:
                    ax_geo.set_xlim(425000, 910000)
                    ax_geo.set_ylim(6040000, 6415000)
                    
                
                if geo_show_background:
                    try:
                        ctx.add_basemap(
                            ax_geo,
                            crs="EPSG:25832",
                            source=ctx.providers.CartoDB.Positron,
                            zoom=9,
                            alpha=0.75,
                            attribution=False
                        )
                    except Exception as e:
                        st.warning(f"Could not load basemap: {e}")
                
                # Add density overlay if requested
                if show_geo_density and len(geo_plots) > 2:
                    try:                  
                        sns.kdeplot(
                            data=geo_plots, x='x', y='y',
                            fill=False, cmap='Reds', levels=12,
                            thresh=0.05, alpha=0.95, ax=ax_geo
                        )
                    except:
                        st.warning("Could not create density overlay")
                
                # Set labels and title
                ax_geo.set_xlabel('utm x', fontsize=12)
                ax_geo.set_ylabel('utm y', fontsize=12)
                ax_geo.set_title('Distribution of plots', fontsize=16, pad=20)
                ax_geo.grid(True, alpha=0.3)
                
                # Set aspect ratio to equal for proper geographic representation
                ax_geo.set_aspect('equal', adjustable='box')
                
                # Add plot count
                ax_geo.text(0.02, 0.98, f"Plots shown: {len(geo_plots)}",
                           transform=ax_geo.transAxes, verticalalignment='top', 
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                                                 facecolor="white", alpha=0.8))
                
                st.pyplot(fig_geo)
                
                # Save option for geographic map
                col1, col2 = st.columns([2, 1])
                with col1:
                    save_name_geo = st.text_input("Save as:", value="plot_geographic", key="save_geo")
                with col2:
                    if st.button("💾 Save Map", key="save_btn_geo"):
                        figures_path = Path(st.session_state.get('figures_path', '.'))
                        if save_figure(fig_geo, save_name_geo, figures_path):
                            st.success(f"Map saved as {save_name_geo}.png")
                
                plt.close()
                
                # Summary statistics for geographic distribution
                with st.expander("📊 Geographic Statistics"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("X Range (m)", f"{geo_plots['x'].max() - geo_plots['x'].min():,.0f}")
                        st.metric("Min X", f"{geo_plots['x'].min():,.0f}")
                    with col2:
                        st.metric("Y Range (m)", f"{geo_plots['y'].max() - geo_plots['y'].min():,.0f}")
                        st.metric("Min Y", f"{geo_plots['y'].min():,.0f}")
                    with col3:
                        st.metric("Coverage Area", f"~{((geo_plots['x'].max() - geo_plots['x'].min()) * (geo_plots['y'].max() - geo_plots['y'].min()) / 1e6):,.1f} km²")
                        st.metric("Max Y", f"{geo_plots['y'].max():,.0f}")
            else:
                st.warning("No valid geographic coordinates found in filtered data")
        else:
            st.info("Geographic coordinates (x, y) not available in this dataset")
            

# Combined Geographic and Ecological Distribution Map
        st.markdown("---")
        st.markdown("### 🗺️ Combined distribution view")

        # Combined map settings
        col1, col2, col3 = st.columns(3)

        with col1:
            show_combined_contours = st.selectbox(
                "Show KDE contours & hull:",
                options=["On", "Off"],
                index=0,  # Default to "On"
                key="combined_contours"
            ) == "On"

        with col2:
            show_combined_legend = st.selectbox(
                "Show legend box:",
                options=["On", "Off"], 
                index=0,  # Default to "On"
                key="combined_legend"
            ) == "On"

        with col3:
            # Default title based on current filter description
            default_title = f"{len(filtered_plots)} plots"
            if selected_habitats and 'habitat_type' in filtered_plots.columns:
                unique_habitats = sorted(filtered_plots['habitat_type'].dropna().unique())
                if len(unique_habitats) <= 5:
                    habitat_list = ', '.join(map(str, unique_habitats))
                else:
                    habitat_list = ', '.join(map(str, unique_habitats[:3])) + f' + {len(unique_habitats)-3} more'
                default_title = f"Habitat type(s): {habitat_list}"
            elif selected_major_types:
                default_title += f" - {', '.join(selected_major_types)}"
            
            combined_figure_title = st.text_input(
                "Figure title:",
                value=default_title,
                key="combined_title"
            )

        # Create new combined figure (A4 width)
        #fig_combined, (ax_geo_combined, ax_eco_combined) = plt.subplots(1, 2, figsize=(11.7, 6))
        
        fig_combined, (ax_geo_combined, ax_eco_combined) = plt.subplots(
                                                1, 2, 
                                                figsize=(11.7, 6),
                                                gridspec_kw={'width_ratios': [56.5, 43.5]}  # 60% left, 40% right
                                            )

        # Add border around entire figure
        for spine in fig_combined.patch.get_children():
            if hasattr(spine, 'set_linewidth'):
                spine.set_linewidth(0.8)

        # --- LEFT: GEOGRAPHIC PLOT ---
        # Background
        # if 'df_regional_pool' in st.session_state:
        #     dk_dot_map = st.session_state['df_regional_pool']
        #     if not dk_dot_map.empty and 'utm_easting' in dk_dot_map.columns and 'utm_northing' in dk_dot_map.columns:
        #         valid_background = dk_dot_map.dropna(subset=['utm_easting', 'utm_northing'])
        #         if len(valid_background) > 0:
        #             ax_geo_combined.scatter(
        #                 valid_background['utm_easting'], valid_background['utm_northing'],
        #                 s=20, alpha=0.20, c='lightgrey', edgecolors='none'
        #             )

        # Plot the filtered plots geographically
        if 'x' in filtered_plots.columns and 'y' in filtered_plots.columns:
            geo_data_combined = filtered_plots.dropna(subset=['x', 'y'])
            
            if len(geo_data_combined) > 0:
                # Color by the same variable as the main maps if specified
                if legend_by and legend_by in geo_data_combined.columns:
                    unique_values = geo_data_combined[legend_by].dropna().unique()
                    colors = get_color_palette(len(unique_values))
                    
                    for i, value in enumerate(unique_values):
                        subset = geo_data_combined[geo_data_combined[legend_by] == value]
                        if len(subset) > 0:
                            ax_geo_combined.scatter(
                                subset['x'], subset['y'],
                                s=25, alpha=0.75, c=[colors[i]], 
                                edgecolors='black', linewidths=0.1
                            )
                else:
                    ax_geo_combined.scatter(
                        geo_data_combined['x'], geo_data_combined['y'],
                        s=25, alpha=0.75, c='red', 
                        edgecolors='black', linewidths=0.1
                    )
                    
        if use_fixed_extent:
                    ax_geo_combined.set_xlim(425000, 910000)
                    ax_geo_combined.set_ylim(6040000, 6415000)
                    
        if geo_show_background:
            try:
                ctx.add_basemap(
                    ax_geo_combined,
                    crs="EPSG:25832",
                    source=ctx.providers.CartoDB.Positron,
                    zoom=9,
                    alpha=0.75,
                    attribution=False
                )
            except Exception as e:
                st.warning(f"Could not load basemap: {e}")

        ax_geo_combined.set_xlabel('utm x', fontsize=10)
        ax_geo_combined.set_ylabel('utm y', fontsize=10)
        ax_geo_combined.set_title('Geographic distribution', fontsize=11)
        ax_geo_combined.grid(True, alpha=0.3)
        ax_geo_combined.set_aspect('equal', adjustable='box')

        # Add legend to geographic map if coloring by categories and if enabled
        if show_combined_legend and legend_by and legend_by in geo_data_combined.columns and len(unique_values) <= 10:
            # Create legend entries
            legend_elements = []
            for i, value in enumerate(unique_values):
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=colors[i], markersize=8,
                            label=str(value)[:25])  # Truncate long labels
                )
            
            # Add legend in upper right corner
            ax_geo_combined.legend(handles=legend_elements, loc='upper right', 
                                framealpha=0.9, fontsize=8, 
                                title=legend_by.replace('_', ' ').title())

        # --- RIGHT: ECOLOGICAL PLOT ---
        # Background elements
        center = (0.5, 0.5)
        radii = [0.125, 0.25, 0.375, 0.5]
        for radius in radii:
            circle = Circle(center, radius, linewidth=0.5, color='gray', fill=False, alpha=0.3)
            ax_eco_combined.add_patch(circle)

        # Add guide lines
        ax_eco_combined.add_line(Line2D([0.5, 0.5], [0, 1], color='gray', linewidth=0.5, alpha=0.3))
        ax_eco_combined.add_line(Line2D([0, 1], [0.5, 0.5], color='gray', linewidth=0.5, alpha=0.3))
        ax_eco_combined.add_line(Line2D([0, 1], [0, 1], color='gray', linewidth=0.5, alpha=0.3))
        ax_eco_combined.add_line(Line2D([0, 1], [1, 0], color='gray', linewidth=0.5, alpha=0.3))

        # Add background context - all plots
        if len(plot_df) > 0:
            all_plots_valid = plot_df.dropna(subset=['xcoor', 'ycoor'])
            all_plots_valid = all_plots_valid[
                (all_plots_valid['xcoor'] >= 0) & (all_plots_valid['xcoor'] <= 1) &
                (all_plots_valid['ycoor'] >= 0) & (all_plots_valid['ycoor'] <= 1)
            ]
            
            if len(all_plots_valid) > 0:
                ax_eco_combined.scatter(
                    all_plots_valid['xcoor'], all_plots_valid['ycoor'],
                    alpha=0.15, s=20, color='lightgrey', zorder=1
                )

        # Plot filtered plots in ecological space (plotted first so contours appear on top)
        if len(filtered_plots) > 0:
            # Color by same variable as main maps
            if legend_by and legend_by in filtered_plots.columns:
                unique_values = filtered_plots[legend_by].dropna().unique()
                colors = get_color_palette(len(unique_values))
                
                for i, value in enumerate(unique_values):
                    subset = filtered_plots[filtered_plots[legend_by] == value]
                    if len(subset) > 0:
                        ax_eco_combined.scatter(
                            subset['xcoor'], subset['ycoor'],
                            alpha=0.7, s=20, c=[colors[i]], zorder=3
                        )
            else:
                ax_eco_combined.scatter(
                    filtered_plots['xcoor'], filtered_plots['ycoor'],
                    alpha=0.7, s=20, color='darkgreen', zorder=3
                )

        # Add Mahalanobis-weighted centroid if enough data
        if len(filtered_plots) > 0:
            try:
                valid_for_centroid = filtered_plots.dropna(subset=['xcoor', 'ycoor'])
                if len(valid_for_centroid) > 0:
                    coords_centroid = valid_for_centroid[['xcoor', 'ycoor']].values
                    if 'mean_mahal_dist' in valid_for_centroid.columns:
                        weights = valid_for_centroid['mean_mahal_dist'].fillna(0).values
                        centroid = np.average(coords_centroid, weights=weights, axis=0) if weights.sum() > 0 else np.mean(coords_centroid, axis=0)
                    else:
                        centroid = np.mean(coords_centroid, axis=0)
                    ax_eco_combined.scatter([centroid[0]], [centroid[1]],
                                        s=150, color='red', marker='*',
                                        edgecolor='darkred', linewidth=2, zorder=8)
            except Exception as e:
                st.warning(f"Could not draw centroid: {e}")

        # Add KDE contours for filtered plots if enough data and if enabled (plotted after scatter to appear on top)
        if show_combined_contours and len(filtered_plots) >= 3:
            try:
                
                # Create evaluation grid
                x_grid = np.linspace(-0.1, 1.1, 60)
                y_grid = np.linspace(-0.1, 1.1, 60)
                xx, yy = np.meshgrid(x_grid, y_grid)
                
                # Calculate KDE for filtered plots
                from scipy.stats import gaussian_kde
                
                coords_combined = filtered_plots[['xcoor', 'ycoor']].values
                kde = gaussian_kde(coords_combined.T)
                positions = np.vstack([xx.ravel(), yy.ravel()])
                density = kde(positions).reshape(xx.shape)
                
                # Find 90% and 50% contour levels
                sorted_density = np.sort(density.ravel())[::-1]
                cumsum = np.cumsum(sorted_density)
                total = cumsum[-1]
                level_90 = sorted_density[np.where(cumsum >= 0.9 * total)[0][0]]
                level_50 = sorted_density[np.where(cumsum >= 0.5 * total)[0][0]]
                
                # Plot 90% KDE contour (on top of scatter points)
                ax_eco_combined.contour(xx, yy, density, levels=[level_90], 
                                    colors=['blue'], linewidths=2, linestyles='-', alpha=0.8, zorder=10)
                
                # Plot 50% KDE contour (on top of scatter points)
                ax_eco_combined.contour(xx, yy, density, levels=[level_50], 
                                    colors=['red'], linewidths=2.3, linestyles='-', alpha=0.99, zorder=10)
                
            except Exception as e:
                pass  # Skip if KDE fails

        # Add convex hull for filtered plots if enabled (plotted after scatter to appear on top)
        if show_combined_contours and len(filtered_plots) >= 3:
            try:
                from scipy.spatial import ConvexHull
                from matplotlib.patches import Polygon
                
                coords_hull = filtered_plots[['xcoor', 'ycoor']].values
                hull = ConvexHull(coords_hull)
                hull_points = coords_hull[hull.vertices]
                hull_polygon = Polygon(hull_points, fill=False, 
                                    edgecolor='blue', linewidth=1.5, 
                                    linestyle='--', alpha=0.5, zorder=11)
                ax_eco_combined.add_patch(hull_polygon)
            except Exception as e:
                pass  # Skip if hull fails

        # Center marker (highest zorder to always be on top)
        ax_eco_combined.scatter([0.5], [0.5], s=75, color='green', marker='+', linewidth=3, zorder=15)

        ax_eco_combined.set_xlim(-0.05, 1.05)
        ax_eco_combined.set_ylim(-0.05, 1.05)
        ax_eco_combined.set_xlabel('x', fontsize=10, alpha=0.7)
        ax_eco_combined.set_ylabel('y', fontsize=10, alpha=0.7)
        ax_eco_combined.set_title('Ecological distribution', fontsize=11)
        ax_eco_combined.grid(True, alpha=0.2)
        ax_eco_combined.set_aspect('equal')

        plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.95])  # Leave space for border and title

        # Add custom title at bottom
        fig_combined.text(0.08, 0.02, combined_figure_title, fontsize=12, fontweight='bold', 
                        transform=fig_combined.transFigure, verticalalignment='bottom')

        st.pyplot(fig_combined)

        # Save option for combined figure
        col1, col2 = st.columns([3, 1])
        with col1:
            # Create filename based on filters
            filename_parts = ["plots_combined"]
            if selected_major_types and len(selected_major_types) <= 3:
                clean_types = [t.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '') for t in selected_major_types]
                filename_parts.extend(clean_types)
            
            default_combined_name = "_".join(filename_parts)
            save_name_combined = st.text_input("Save combined figure as:", 
                                            value=default_combined_name, 
                                            key="save_plots_combined")
        with col2:
            if st.button("💾 Save Combined Figure", key="save_btn_plots_combined"):
                figures_path = Path(st.session_state.get('figures_path', '.'))
                if save_figure(fig_combined, save_name_combined, figures_path):
                    st.success(f"Combined figure saved as {save_name_combined}.png")

        plt.close(fig_combined)
        
    
    with tab2:
        st.markdown("### 📊 Temporal change analysis")

        if 'year' not in plot_df.columns:
            st.warning("No temporal data (year) available in the plot data")
        else:
            # Filtering options (same as tab1)
            st.markdown("#### 🔍 Filter plots")

            col1, col2 = st.columns(2)

            with col1:
                # Regional filters in expander
                with st.expander("🗺️ Regional Filters", expanded=False):
                    # NOVANA Region filter
                    if 'region' in plot_df.columns:
                        regions = plot_df['region'].dropna().unique()
                        selected_regions_t2 = st.multiselect(
                            "Select monitoring regions:",
                            options=sorted(regions),
                            default=list(regions),
                            key="regions_t2"
                        )
                        filtered_plots_t2 = plot_df[plot_df['region'].isin(selected_regions_t2)] if selected_regions_t2 else plot_df
                    else:
                        filtered_plots_t2 = plot_df

                    # Bioregion filter
                    if 'subregion' in filtered_plots_t2.columns:
                        bioregions = filtered_plots_t2['subregion'].dropna().unique()
                        selected_bioregions_t2 = st.multiselect(
                            "Select bioregions:",
                            options=sorted(bioregions),
                            default=list(bioregions),
                            key="bioregions_t2"
                        )
                        if selected_bioregions_t2:
                            filtered_plots_t2 = filtered_plots_t2[filtered_plots_t2['subregion'].isin(selected_bioregions_t2)]

            with col2:
                # Habitat filters in expander
                with st.expander("🏞️ Vegetation & Habitat Type Filters", expanded=False):
                    # Initialise filter variables — overwritten below if columns are present
                    selected_major_types_t2 = []
                    selected_habitats_t2 = []
                    # Major type filter
                    if 'major_type' in filtered_plots_t2.columns:
                        major_types = filtered_plots_t2['major_type'].dropna().unique()
                        selected_major_types_t2 = st.multiselect(
                            "Select major vegetation types:",
                            options=sorted(major_types),
                            default=list(major_types),
                            key="major_types_t2"
                        )
                        if selected_major_types_t2:
                            filtered_plots_t2 = filtered_plots_t2[filtered_plots_t2['major_type'].isin(selected_major_types_t2)]

                    # Habitat type filter
                    if 'habitat_type' in filtered_plots_t2.columns:
                        if selected_major_types_t2:
                            available_habitats = filtered_plots_t2[
                                filtered_plots_t2['major_type'].isin(selected_major_types_t2)
                            ]['habitat_type'].dropna().unique()
                        else:
                            available_habitats = filtered_plots_t2['habitat_type'].dropna().unique()

                        habitat_types = sorted(available_habitats)
                        selected_habitats_t2 = st.multiselect(
                            "Select specific habitat types:",
                            options=habitat_types,
                            default=habitat_types,
                            key="habitats_t2"
                        )
                        if selected_habitats_t2:
                            filtered_plots_t2 = filtered_plots_t2[filtered_plots_t2['habitat_type'].isin(selected_habitats_t2)]

            st.info(f"Showing {len(filtered_plots_t2)} plots after filtering")

            # Temporal Comparison Settings
            st.markdown("---")
            st.markdown("#### ⏱️ Temporal comparison settings")

            if len(filtered_plots_t2) > 0:
                # Time range settings
                min_year = int(filtered_plots_t2['year'].min())
                max_year = int(filtered_plots_t2['year'].max())

                if min_year < max_year:
                    # Calculate median year
                    median_year = int(filtered_plots_t2['year'].median())
                    default_min = median_year - 1
                    default_max = median_year

                    year_range = st.slider(
                        "Select time periods for comparison:",
                        min_value=min_year,
                        max_value=max_year,
                        value=(default_min, default_max),
                        help="Adjust to compare early vs late periods. Default splits data roughly in half.",
                        key="year_range_t2"
                    )

                    # Split data by time
                    early_data = filtered_plots_t2[filtered_plots_t2['year'] <= year_range[0]]
                    late_data = filtered_plots_t2[filtered_plots_t2['year'] > year_range[1]]

                    # Display plot counts
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        st.metric("🔵 Early period plots", f"{len(early_data):,}",
                                 help=f"Plots from year ≤ {year_range[0]}")
                    with col2:
                        st.metric("🔴 Late period plots", f"{len(late_data):,}",
                                 help=f"Plots from year > {year_range[1]}")
                    with col3:
                        excluded = len(filtered_plots_t2) - len(early_data) - len(late_data)
                        st.metric("⚪ Excluded plots", f"{excluded:,}",
                                 help=f"Plots between {year_range[0]} and {year_range[1]}")

                    st.markdown("---")

                    # Map settings (similar to tab1)
                    st.markdown("#### 🎨 Map Settings")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**Background Layers:**")
                        show_background_t2 = st.checkbox("Show background (all plots)", value=False, key="bg_t2")
                        show_background_annotated_t2 = st.checkbox("Show background (annotated)", value=False, key="bg_ann_t2")

                    with col2:
                        st.markdown("**Visualization Options:**")
                        show_contours_t2 = st.checkbox("Show density contours", value=True, key="contours_t2")
                        show_centroids_t2 = st.checkbox("Show period centroids", value=True, key="centroids_t2")

                    with col3:
                        st.markdown("**Display Options:**")
                        show_periods = st.radio(
                            "Show plots from:",
                            options=["Both periods", "Early only", "Late only", "None"],
                            index=0,
                            key="show_periods_t2"
                        )

                    # Advanced settings in expander
                    with st.expander("⚙️ Advanced Map Settings", expanded=False):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown("**Transparency Controls:**")
                            text_alpha_t2 = st.slider("Text transparency:", 0.0, 1.0, 0.7, 0.05, key="text_alpha_t2")
                            line_alpha_t2 = st.slider("Overlay transparency:", 0.0, 1.0, 0.2, 0.05, key="line_alpha_t2")
                            scatter_alpha_t2 = st.slider("Scatter transparency:", 0.1, 1.0, 0.6, 0.05, key="scatter_alpha_t2")

                        with col2:
                            st.markdown("**View Controls:**")
                            zoom_t2 = st.slider("Zoom:", -0.1, 0.25, 0.0, key="zoom_t2")
                            kde_thresh_t2 = st.slider("KDE threshold:", 0.01, 0.5, 0.05, 0.01, key="kde_t2",
                                help="Lower values show tighter contours")

                        with col3:
                            st.markdown("**Annotation Thresholds:**")
                            nr_percentile_t2 = st.slider("N/R threshold (%):", 5, 50, 15, 1, key="nr_percentile_t2",
                                help="Percentile for nutrient-poor/low-pH zones. Lower = more extreme conditions only.")
                            l_percentile_t2 = st.slider("Light threshold (%):", 5.0, 25.0, 12.5, 0.5, key="l_percentile_t2",
                                help="Percentile for shade/forest zones. Lower = darker shade only.")
                            salt_distance_t2 = st.slider("Salt zone extent (%):", 50, 90, 65, 5, key="salt_distance_t2",
                                help="Distance cutoff for halophytic zones. Higher = tighter boundaries around core.")

                    # Create the map
                    fig, ax = create_base_map('Temporal Change Analysis')

                    # Get reference data if available
                    if 'reference_map_data' in st.session_state and 'plot_id' in st.session_state.reference_map_data:
                        reference_df = st.session_state.reference_map_data['plot_id']
                    else:
                        reference_df = plot_df

                    # Plot background layer if requested
                    if show_background_t2:
                        ax.scatter(
                            reference_df['xcoor'], reference_df['ycoor'],
                            s=40, alpha=0.45, c='lightgrey',
                            edgecolors='lightgrey', linewidths=0.05
                        )

                    # Add annotated background if requested
                    if show_background_annotated_t2:
                        create_annotated_background(ax, reference_df, text_alpha_t2, line_alpha_t2,
                                                   nr_percentile_t2, l_percentile_t2, salt_distance_t2)

                    # Plot early period points in blue (if selected)
                    if show_periods in ["Both periods", "Early only"]:
                        ax.scatter(
                            early_data['xcoor'], early_data['ycoor'],
                            s=30, alpha=scatter_alpha_t2, c='blue',
                            label=f'Early (≤{year_range[0]}): n = {len(early_data)}',
                            edgecolors='black', linewidths=0.5
                        )

                    # Plot late period points in red (if selected)
                    if show_periods in ["Both periods", "Late only"]:
                        ax.scatter(
                            late_data['xcoor'], late_data['ycoor'],
                            s=30, alpha=scatter_alpha_t2, c='red',
                            label=f'Late (>{year_range[1]}): n = {len(late_data)}',
                            edgecolors='black', linewidths=0.5
                        )

                    # Plot KDE contours if requested (always show both periods)
                    if show_contours_t2:
                        if len(early_data) > 2:
                            try:
                                sns.kdeplot(
                                    data=early_data, x='xcoor', y='ycoor',
                                    levels=2, fill=False, thresh=kde_thresh_t2,
                                    alpha=1, linewidths=1.5, color='blue', ax=ax
                                )
                            except:
                                pass

                        if len(late_data) > 2:
                            try:
                                sns.kdeplot(
                                    data=late_data, x='xcoor', y='ycoor',
                                    levels=2, fill=False, thresh=kde_thresh_t2,
                                    alpha=1, linewidths=3, color='red', ax=ax
                                )
                            except:
                                pass

                    # Plot centroids if requested (always show both periods)
                    if show_centroids_t2 and len(early_data) > 0 and len(late_data) > 0:
                        if 'mean_mahal_dist' in early_data.columns:
                            ew = early_data['mean_mahal_dist'].fillna(0).values
                            early_center_x = np.average(early_data['xcoor'], weights=ew) if ew.sum() > 0 else early_data['xcoor'].mean()
                            early_center_y = np.average(early_data['ycoor'], weights=ew) if ew.sum() > 0 else early_data['ycoor'].mean()
                            lw = late_data['mean_mahal_dist'].fillna(0).values
                            late_center_x = np.average(late_data['xcoor'], weights=lw) if lw.sum() > 0 else late_data['xcoor'].mean()
                            late_center_y = np.average(late_data['ycoor'], weights=lw) if lw.sum() > 0 else late_data['ycoor'].mean()
                        else:
                            early_center_x = early_data['xcoor'].mean()
                            early_center_y = early_data['ycoor'].mean()
                            late_center_x = late_data['xcoor'].mean()
                            late_center_y = late_data['ycoor'].mean()

                        ax.scatter([early_center_x], [early_center_y], s=200, c='blue',
                                 marker='X', edgecolors='black', linewidths=2, zorder=5)
                        ax.scatter([late_center_x], [late_center_y], s=200, c='red',
                                 marker='X', edgecolors='black', linewidths=2, zorder=5)

                        # Draw arrow between centroids
                        ax.annotate('', xy=(late_center_x, late_center_y),
                                  xytext=(early_center_x, early_center_y),
                                  arrowprops=dict(arrowstyle='->', lw=2, color='black'))

                    ax.legend(loc='upper right')

                    ax.set_xlim(zoom_t2, 1 - zoom_t2)
                    ax.set_ylim(zoom_t2, 1 - zoom_t2)

                    st.pyplot(fig)

                    # Save option
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        save_name = st.text_input("Save as:",
                                                value=f"temporal_change",
                                                key="save_temporal")
                    with col2:
                        if st.button("💾 Save Map", key="save_btn_temporal"):
                            figures_path = Path(st.session_state.get('figures_path', '.'))
                            if save_figure(fig, save_name, figures_path):
                                st.success(f"Map saved as {save_name}.png")

                    plt.close()

                    # Summary statistics
                    with st.expander("📊 Temporal Movement Analysis", expanded=False):
                        if len(early_data) > 0 and len(late_data) > 0:
                            # Calculate Mahalanobis-weighted centroids
                            if 'mean_mahal_dist' in early_data.columns:
                                ew = early_data['mean_mahal_dist'].fillna(0).values
                                early_center_x = np.average(early_data['xcoor'], weights=ew) if ew.sum() > 0 else early_data['xcoor'].mean()
                                early_center_y = np.average(early_data['ycoor'], weights=ew) if ew.sum() > 0 else early_data['ycoor'].mean()
                                lw = late_data['mean_mahal_dist'].fillna(0).values
                                late_center_x = np.average(late_data['xcoor'], weights=lw) if lw.sum() > 0 else late_data['xcoor'].mean()
                                late_center_y = np.average(late_data['ycoor'], weights=lw) if lw.sum() > 0 else late_data['ycoor'].mean()
                            else:
                                early_center_x = early_data['xcoor'].mean()
                                early_center_y = early_data['ycoor'].mean()
                                late_center_x = late_data['xcoor'].mean()
                                late_center_y = late_data['ycoor'].mean()

                            # Calculate movement metrics
                            movement_dist = np.sqrt((late_center_x - early_center_x)**2 +
                                                  (late_center_y - early_center_y)**2)
                            angle = np.degrees(np.arctan2(late_center_y - early_center_y,
                                                         late_center_x - early_center_x))

                            # Calculate spread (standard deviation)
                            early_spread = np.sqrt(early_data['xcoor'].std()**2 + early_data['ycoor'].std()**2)
                            late_spread = np.sqrt(late_data['xcoor'].std()**2 + late_data['ycoor'].std()**2)
                            spread_change = late_spread - early_spread

                            # Movement as percentage of map
                            movement_pct = movement_dist * 100

                            # Interpret direction
                            if -22.5 <= angle < 22.5:
                                direction_txt = "→ East"
                            elif 22.5 <= angle < 67.5:
                                direction_txt = "↗ Northeast"
                            elif 67.5 <= angle < 112.5:
                                direction_txt = "↑ North"
                            elif 112.5 <= angle < 157.5:
                                direction_txt = "↖ Northwest"
                            elif angle >= 157.5 or angle < -157.5:
                                direction_txt = "← West"
                            elif -157.5 <= angle < -112.5:
                                direction_txt = "↙ Southwest"
                            elif -112.5 <= angle < -67.5:
                                direction_txt = "↓ South"
                            else:
                                direction_txt = "↘ Southeast"

                            # Display metrics in columns
                            st.markdown("#### Movement vector")
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Distance", f"{movement_dist:.3f}",
                                         delta=f"{movement_pct:.1f}% of map")
                            with col2:
                                st.metric("Direction", direction_txt,
                                         delta=f"{angle:.1f}°")
                            with col3:
                                spread_emoji = "📈" if spread_change > 0 else "📉"
                                st.metric("Spread Change", f"{spread_emoji}",
                                         delta=f"{spread_change:.3f} units")

                            st.markdown("---")
                            st.markdown("#### Centroid positions")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**🔵 Early Period**")
                                st.metric("X coordinate", f"{early_center_x:.3f}")
                                st.metric("Y coordinate", f"{early_center_y:.3f}")
                                st.metric("Spread (σ)", f"{early_spread:.3f}")

                            with col2:
                                st.markdown("**🔴 Late Period**")
                                st.metric("X coordinate", f"{late_center_x:.3f}",
                                         delta=f"{late_center_x - early_center_x:+.3f}")
                                st.metric("Y coordinate", f"{late_center_y:.3f}",
                                         delta=f"{late_center_y - early_center_y:+.3f}")
                                st.metric("Spread (σ)", f"{late_spread:.3f}",
                                         delta=f"{spread_change:+.3f}")

                            # Ellenberg indicator changes
                            st.markdown("---")
                            st.markdown("#### Ellenberg indicator changes")

                            # Check which Ellenberg values are available
                            ellenberg_cols = {'M': 'Moisture', 'L': 'Light', 'N': 'Nitrogen', 'R': 'pH (Reaction)'}
                            available_indicators = {col: name for col, name in ellenberg_cols.items()
                                                   if col in early_data.columns and col in late_data.columns}

                            if available_indicators:
                                # Calculate means for each available indicator
                                col1, col2, col3, col4 = st.columns(4)
                                cols = [col1, col2, col3, col4]

                                for idx, (col, name) in enumerate(available_indicators.items()):
                                    early_mean = early_data[col].mean()
                                    late_mean = late_data[col].mean()
                                    change = late_mean - early_mean

                                    # Determine delta color based on change
                                    if abs(change) < 0.1:
                                        delta_color = "off"  # Minimal change
                                    else:
                                        delta_color = "normal"

                                    with cols[idx]:
                                        if col == 'M':
                                            emoji = "💧"
                                        elif col == 'L':
                                            emoji = "☀️"
                                        elif col == 'N':
                                            emoji = "🌱"
                                        elif col == 'R':
                                            emoji = "⚗️"
                                        else:
                                            emoji = ""

                                        st.metric(f"{emoji} {name}",
                                                f"{late_mean:.2f}",
                                                delta=f"{change:+.2f}",
                                                delta_color=delta_color,
                                                help=f"Early: {early_mean:.2f} → Late: {late_mean:.2f}")

                                # Interpretation of changes
                                st.markdown("")
                                interpretations = []

                                if 'M' in available_indicators:
                                    m_change = late_data['M'].mean() - early_data['M'].mean()
                                    if m_change > 0.2:
                                        interpretations.append("💧 **Wetter conditions** - Movement toward moister habitats")
                                    elif m_change < -0.2:
                                        interpretations.append("💧 **Drier conditions** - Movement toward drier habitats")

                                if 'L' in available_indicators:
                                    l_change = late_data['L'].mean() - early_data['L'].mean()
                                    if l_change > 0.2:
                                        interpretations.append("☀️ **More light** - Movement toward open/sunny habitats")
                                    elif l_change < -0.2:
                                        interpretations.append("☀️ **Less light** - Movement toward shaded habitats")

                                if 'N' in available_indicators:
                                    n_change = late_data['N'].mean() - early_data['N'].mean()
                                    if n_change > 0.2:
                                        interpretations.append("🌱 **Nutrient enrichment** - Movement toward nutrient-rich sites")
                                    elif n_change < -0.2:
                                        interpretations.append("🌱 **Nutrient depletion** - Movement toward nutrient-poor sites")

                                if 'R' in available_indicators:
                                    r_change = late_data['R'].mean() - early_data['R'].mean()
                                    if r_change > 0.2:
                                        interpretations.append("⚗️ **More basic** - Movement toward calcareous/basic soils")
                                    elif r_change < -0.2:
                                        interpretations.append("⚗️ **More acidic** - Movement toward acidic soils")

                                if interpretations:
                                    st.info("\n\n".join(interpretations))
                                else:
                                    st.success("✅ **Stable environmental conditions** - Minimal changes in Ellenberg indicators")
                            else:
                                st.warning("No Ellenberg indicator data (M, L, N, R) available in plot data")

                            # Interpretation
                            st.markdown("---")
                            st.markdown("#### Ecological interpretation")

                            if movement_dist > 0.1:
                                interpretation = "🔴 **Large shift** - Significant change in habitat positioning"
                            elif movement_dist > 0.05:
                                interpretation = "🟡 **Moderate shift** - Notable change in habitat location"
                            else:
                                interpretation = "🟢 **Stable** - Minimal centroid movement"

                            if spread_change > 0.05:
                                spread_interp = "📈 **Expansion** - Habitat occupies broader niche space"
                            elif spread_change < -0.05:
                                spread_interp = "📉 **Contraction** - Habitat occupies narrower niche space"
                            else:
                                spread_interp = "➡️ **Stable** - Niche breadth unchanged"

                            st.info(f"{interpretation}\n\n{spread_interp}")
                        else:
                            st.warning("Need both early and late period data for movement analysis")
                else:
                    st.warning("Not enough temporal variation for analysis")
            else:
                st.warning("No plots found with current filters")
    
