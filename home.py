"""
EcoNetMap - Ecological Network Mapping Toolkit
==============================================
A comprehensive toolkit for network-based ecological cartography. Transform species 
co-occurrence patterns into interpretable 2D reference landscapes that reveal community 
structures, ecological gradients, and temporal dynamics.

Uses Jaccard similarity networks with force-directed layouts and MDS distance correction
to position species in ecological space based purely on co-occurrence patterns. Supports
any vegetation monitoring dataset with plot-level species occurrence data.

The workflow consists of four main components: 
 - Data Handling (import, filtering, and validation)
 - Graph Construction (network analysis with dual layout approach)
 - Mapping (species and plot visualizations in ecological space)
 - Analysis (dark diversity, species profiles, temporal changes)

Developed for the Danish NOVANA vegetation monitoring program, applicable to any 
plot-based species occurrence dataset.

Main entry point for EcoNetMap
Author: Flemming Skov (fs@ecos.au.dk)
Last Updated: January 2026
"""

# Import packages for web applications
import streamlit as st

# Import packages for file and system operations
from pathlib import Path
import configparser

from column_config import (
    get_project_base_path, set_project_base_path, SETTINGS_FILENAME,
    VEGETATION_ROLES, TAXA_ROLES, all_roles,
)

# App info - permanent in the script
APP_INFO = {
    'title': 'EcoNetMap - Ecological Network Mapping Toolkit',
    'version': '1.05',
    'author': 'Flemming Skov',
    'email': 'fs@ecos.au.dk',
    'github_url': 'https://github.com/flemmingskov/EcoNetMap'
}

# Constants
PROJECT_FOLDERS = {
    'data': 'Raw vegetation monitoring data files',
    'queries': 'Filtered data and query results',
    'reference_maps': 'Reference map outputs and coordinates',
    'overlay_maps': 'Overlay map outputs',
    'external_data': 'Additional external datasets',
    'figures': 'Generated figures and plots'
}


def activate_project(project_dir: Path) -> bool:
    """Point the app at project_dir, creating it if needed.

    Never overwrites an existing settings.txt: if project_dir already has one
    (e.g. switching back to a project used before), its saved paths and column
    mappings are kept as-is. Only writes a fresh settings.txt when there isn't
    one already. Returns True if a new settings.txt was created, False if an
    existing one was reused.
    """
    project_dir.mkdir(parents=True, exist_ok=True)
    for folder in PROJECT_FOLDERS.keys():
        (project_dir / folder).mkdir(exist_ok=True)

    settings_file_path = project_dir / SETTINGS_FILENAME
    is_new = not settings_file_path.exists()

    if is_new:
        settings_config = configparser.ConfigParser()
        settings_config['project'] = {
            'project_base_path': str(project_dir),
            **{f"{folder}_folder": folder for folder in PROJECT_FOLDERS.keys()}
        }
        # Seed identity column mappings (role name == column name) as a generic
        # starting point. Works out of the box if the user's file already uses
        # canonical names; otherwise gets overwritten via the Map Your Columns UI.
        settings_config['columns_vegetation'] = {role: role for role in all_roles(VEGETATION_ROLES)}
        settings_config['columns_taxa'] = {role: role for role in all_roles(TAXA_ROLES)}
        with open(settings_file_path, 'w') as f:
            settings_config.write(f)

    set_project_base_path(project_dir)
    return is_new

# Simple settings loader - no Streamlit commands
def load_settings():
    """Load and validate settings.txt from the configured project workspace, or return None"""
    project_base = get_project_base_path()
    if project_base is None:
        return None

    settings_file = project_base / SETTINGS_FILENAME
    if not settings_file.exists():
        return None

    try:
        config = configparser.ConfigParser()
        config.read(settings_file)

        # Validate required section exists
        if not config.has_section('project'):
            return None

        return config
    except configparser.Error:
        return None

# Load config (might be None)
config = load_settings()

# Page configuration 
if config:
    app_title = APP_INFO['title']
else:
    app_title = APP_INFO['title'] + ' - Setup'

st.set_page_config(
    page_title=app_title, 
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NOW we can use Streamlit caching decorators
@st.cache_data
def load_settings_cached():
    """Cached version of settings loader"""
    return load_settings()

@st.cache_data
def get_project_paths():
    """Get project paths from settings - pure function without side effects"""
    config = load_settings_cached()
    if not config:
        return None, None

    # The workspace location is authoritative from the pointer file, not settings.txt itself
    base_path = get_project_base_path()

    paths = {
        'data_path': base_path / config.get('project', 'data_folder', fallback='data'),
        'queries_path': base_path / config.get('project', 'queries_folder', fallback='queries'),
        'reference_map_path': base_path / config.get('project', 'reference_maps_folder', fallback='reference_maps'),
        'overlay_map_path': base_path / config.get('project', 'overlay_maps_folder', fallback='overlay_maps'),
        'external_data_path': base_path / config.get('project', 'external_data_folder', fallback='external_data'),
        'figures_path': base_path / config.get('project', 'figures_folder', fallback='figures')
    }

    return base_path, paths

# Custom CSS
st.markdown("""
<style>
    .stTextInput > label {
        font-weight: bold;
        color: #2c3e50;
    }
    .main-header {
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
        margin-bottom: 30px;
    }
    .setup-box {
        background-color: #f0f8ff;
        border: 2px solid #4a90e2;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    .feature-box {
        background-color: #f8f9fa;
        border-left: 4px solid #27ae60;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">EcoNetMap</h1>', unsafe_allow_html=True)
st.subheader("Revealing ecological landscapes from species co-occurrence")

# Check if we need to set up project
if not config:
    st.markdown('<div class="setup-box">', unsafe_allow_html=True)
    st.markdown("### 🚀 Welcome! Let's set up your EcoNetMap project")
    
    st.markdown("""
    This appears to be your first time running the app. Let's configure your project location.
    
    **What we'll do:**
    1. You choose a project folder location
    2. We'll create the necessary subfolders
    3. We'll save these settings for future use
    """)
    
    # Project folder selection
    st.markdown("#### 📁 Choose Your Project Location")
    
    existing_pointer = get_project_base_path()
    default_project = str(existing_pointer) if existing_pointer else str(Path.home() / "Desktop" / "EcoNetMap_project")

    if 'project_path_input' not in st.session_state:
        st.session_state['project_path_input'] = default_project

    project_path = st.text_input(
        "Project folder path:",
        key='project_path_input',
        help="Full path where you want to store your EcoNetMap project files"
    )

    project_dir = Path(project_path)
    
    # Show what will be created
    st.markdown("#### 📋 Folder Structure")
    st.markdown("The following subfolders will be created in your project:")

    for folder, description in PROJECT_FOLDERS.items():
        subfolder_path = project_dir / folder
        if subfolder_path.exists():
            st.markdown(f"📁 `{folder}/` - {description} ✅ *exists*")
        else:
            st.markdown(f"📁 `{folder}/` - {description} ⚪ *will be created*")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create project button
    if st.button("🔨 Create Project", type="primary", use_container_width=True):
        try:
            # Validate path is writable before creating
            parent_dir = project_dir.parent
            if not parent_dir.exists():
                st.error(f"❌ Parent directory does not exist: {parent_dir}")
                st.info("💡 Please create the parent directory first or choose a different location")
                st.stop()

            # Test write permissions
            try:
                test_file = parent_dir / ".EcoNetMap_write_test"
                test_file.touch()
                test_file.unlink()
            except (PermissionError, OSError) as e:
                st.error(f"❌ Cannot write to this location: {parent_dir}")
                st.info("💡 Please choose a location where you have write permissions")
                st.stop()

            is_new = activate_project(project_dir)
            st.success(f"✅ Project directory ready: {project_dir}")
            if is_new:
                st.success(f"✅ Created {SETTINGS_FILENAME} file in {project_dir}")
            else:
                st.info(f"📁 Found an existing {SETTINGS_FILENAME} in {project_dir} — reusing its saved settings")

            st.markdown("### 🎉 Setup Complete!")
            st.info("Please **refresh the page** (F5 or Ctrl+R) to continue with your project.")

        except PermissionError as e:
            st.error(f"❌ Permission denied: {str(e)}")
            st.info("💡 Please choose a location where you have write permissions")
        except OSError as e:
            st.error(f"❌ System error creating directories: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
    
    # Stop here - don't show the rest of the interface
    st.stop()

# If we get here, settings exist - load paths
project_base, paths = get_project_paths()

# Update session state with current paths
if paths:
    st.session_state['project_base_path'] = str(project_base)
    for key, path in paths.items():
        st.session_state[key] = str(path)

# Show current project info
st.markdown("---")
st.markdown("### 📁 Current Project")
st.info(f"**Project Location:** `{project_base}`")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f"*Settings loaded from `settings.txt` inside your project folder (`{project_base}`)* ")
    st.caption("💡 Edit settings.txt and click reload to update paths")
with col2:
    if st.button("🔄 Reload Settings", type="primary"):
        # Clear the cached functions
        load_settings_cached.clear()
        get_project_paths.clear()
        # Force complete reload
        st.rerun()

with st.expander("🔀 Switch to a Different Project", expanded=False):
    st.markdown("""
    Point to a different project folder. If it already contains a `settings.txt`
    (a project you've used before), its saved paths and column mappings are kept
    as-is. If it's empty, a new project is created there.
    """)

    if 'switch_path_input' not in st.session_state:
        st.session_state['switch_path_input'] = str(project_base)

    st.text_input(
        "Project folder path:",
        key='switch_path_input',
        label_visibility="collapsed"
    )

    if st.button("🔀 Switch to This Project", type="primary"):
        try:
            new_dir = Path(st.session_state['switch_path_input'])
            is_new = activate_project(new_dir)
            load_settings_cached.clear()
            get_project_paths.clear()
            if is_new:
                st.success(f"✅ Created new project at {new_dir}")
            else:
                st.success(f"✅ Switched to existing project at {new_dir}")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Could not switch project: {e}")

# Key Features section
# st.markdown("---")
with st.expander("🌟 Key Features", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("""
        **🕸️ Network-Based Positioning**
        - Species positioned by co-occurrence patterns
        - Jaccard similarity for proper ecological weighting
        - Community detection (Leiden & Louvain algorithms)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("""
        **📊 Dual Layout Approach**
        - Force-directed layouts for clustering visualization
        - MDS distance correction for interpretable distances
        - Both topological and metric interpretations
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("""
        **🗺️ Ecological Reference Landscapes**
        - 2D space preserving species relationships
        - Environmental gradients emerge from data
        - Intuitive spatial metaphors for stakeholders
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("""
        **🔬 Comprehensive Analysis**
        - Temporal change detection
        - Dark diversity assessment
        - Species profiling & habitat quality metrics
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# About section
# st.markdown("---")
with st.expander("ℹ️ About & Workflow", expanded=False):
    st.markdown("""
    ### 📖 About EcoNetMap

    EcoNetMap implements **ecological cartography** — a network-based approach to positioning
    species in two-dimensional ecological space. Unlike traditional ordination methods that
    compress multidimensional relationships into orthogonal axes, this approach preserves
    pairwise species co-occurrence patterns directly.

    **Applicable to:** Any plot-based species occurrence dataset
    **Data requirements:** Species identities and plot identifiers

    ### 🔄 Analysis Workflow

    Use the **sidebar navigation** to work through the following stages in order:

    #### 🗂️ Phase 1 — Data management (01–03)
    **01 Import data**
    - Load vegetation monitoring data (species × plots)
    - Load species taxonomy and Ellenberg indicator values
    - Optionally load a regional species pool dataset

    **02 Filter data**
    - Filter by habitat types, taxonomic groups, time periods, and geographic regions
    - Apply data quality thresholds and stratified sampling
    - Export filtered datasets as SQLite databases

    **03 Data diagnostics**
    - Examine temporal and spatial sampling patterns
    - Identify biases that could affect network stability

    #### 🕸️ Phase 2 — Network construction (11–14)
    **11 Network layout**
    - Calculate Jaccard similarity from co-occurrence data
    - Build species association network
    - Generate dual layouts: force-directed (clustering) + MDS (distances)
    - Perform community detection (Leiden / Louvain)

    **12 Network enhancement**
    - Assign coordinates to rare species below the occurrence threshold
    - Uses Jaccard-weighted averaging of positioned neighbours

    **13 Network validation**
    - Assess network quality and layout stability
    - Compare runs across different random seeds

    **14 Network overlay**
    - Combine network coordinates with occurrence data from a query dataset
    - Creates unified databases in `overlay_maps/` ready for visualisation

    #### 🗺️ Phase 3 — Visualisation (21–23)
    **21 View reference network**
    - Interactive visualisation of the full species network
    - Community colouring and Ellenberg gradient overlays

    **22 View species**
    - Individual species maps in ecological and geographic space
    - Temporal change analysis and regional species pool overlay

    **23 View plots**
    - Plot-level maps coloured by habitat type or environmental indicators
    - Temporal trajectory mapping

    ### 🛠️ Technical notes

    **Network construction:**
    - Edges weighted by Jaccard similarity (not raw co-occurrence counts)
    - Minimum occurrence thresholds prevent noise
    - Community detection reveals habitat structure

    **Coordinate systems:**
    - Force-directed (`_x` / `_y`): emphasises clustering patterns
    - MDS-corrected (`_mds_x` / `_mds_y`): preserves ecological distances
    - Both layouts stored in the same database

    **Reproducibility:**
    - Network layouts are stochastic — set a random seed for exact reproduction
    - Stability assessed through multiple runs with different seeds
    - Environmental gradients validate ecological meaning

    ### 📚 Citation

    If you use EcoNetMap in your research, please cite:

    *Skov, F. (2026). EcoNetMap: Network-based ecological cartography for vegetation analysis.
    Aarhus University. https://github.com/flemmingskov/EcoNetMap*

    Manuscript in preparation for Journal of Vegetation Science.
    """)

# Current paths display
st.markdown("---")
st.markdown("### 📂 Project Folders")

# Check and create missing directories button
if st.button("🔧 Create Missing Directories", type="secondary"):
    created_dirs = []
    failed_dirs = []
    
    for key in ['data_path', 'queries_path', 'reference_map_path', 'overlay_map_path', 'figures_path', 'external_data_path']:
        path = Path(st.session_state[key])
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(path.name)
            except Exception as e:
                failed_dirs.append(f"{path.name}: {str(e)}")
    
    if created_dirs:
        st.success(f"✅ Created directories: {', '.join(created_dirs)}")
    if failed_dirs:
        st.error(f"❌ Failed to create: {', '.join(failed_dirs)}")
    if not created_dirs and not failed_dirs:
        st.info("All directories already exist")

# Show folder status
folder_info = [
    ('📊 Data', 'data_path'),
    ('🔍 Queries', 'queries_path'),
    ('🗺️ Reference Maps', 'reference_map_path'),
    ('📍 Overlay Maps', 'overlay_map_path'),
    ('📊 External Data', 'external_data_path'),
    ('🖼️ Figures', 'figures_path')
]

col1, col2 = st.columns(2)
for i, (label, key) in enumerate(folder_info):
    path = Path(st.session_state[key])
    status = "✅" if path.exists() and path.is_dir() else "❌"
    display_path = path.name  # Just show folder name, not full path

    if i < 3:
        col1.markdown(f"{status} **{label}**: `{display_path}`")
    else:
        col2.markdown(f"{status} **{label}**: `{display_path}`")

# Sidebar
st.sidebar.success("📖 Select a page above")
st.sidebar.markdown("---")

st.sidebar.title("🌿 EcoNetMap")
st.sidebar.info(f"**Version {APP_INFO['version']}** - February 19, 2026")

st.sidebar.markdown("### 🎯 Quick Start")
st.sidebar.markdown("""
1. **Import** your data
2. **Filter** to focus analysis
3. **Create** network layout
4. **Map** species & plots
5. **Analyze** patterns
""")

st.sidebar.markdown("---")

st.sidebar.title("💻 GitHub Repository")
st.sidebar.info(f"""
This project is under active development. Contributions, comments, 
and questions are welcome on [GitHub]({APP_INFO['github_url']}).
""")

st.sidebar.markdown("---")

st.sidebar.title("👤 About")
st.sidebar.info(f"""
**{APP_INFO['author']}**  
Aarhus University

[University Profile](https://pure.au.dk/portal/da/persons/flemming-skov(d16e357d-aa51-4bd3-ae16-9059110a3fe8).html)

📧 [{APP_INFO['email']}](mailto:{APP_INFO['email']})
""")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #7f8c8d; font-size: 0.9em;'>
{APP_INFO['title']} © 2026 | Aarhus University
</div>
""", unsafe_allow_html=True)