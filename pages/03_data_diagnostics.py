"""
EcoNetMap - Sampling Diagnostics Module
========================================
Diagnostic tool to examine temporal and spatial sampling patterns in vegetation 
monitoring data. Identifies potential biases that could affect validation studies.

This script analyzes:
- Temporal sampling distribution (plots per year)
- Habitat representation over time
- Spatial sampling patterns
- Species accumulation curves
- Sampling intensity metrics

Helps determine if temporal stratification or weighting is needed before creating
reference landscapes for validation.

Part of the EcoNetMap toolkit (Diagnostic Tool)
Author: Flemming Skov (fs@ecos.au.dk)
Created: February 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from pathlib import Path
from typing import Dict, Optional
from io import BytesIO
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Column-role mapping (lets users adapt the toolkit to their own column names)
from column_config import VEGETATION_ROLES, rename_to_canonical

# Constants
###################################################################################
# Display limits
MAX_HABITATS_HEATMAP = 30
MAX_HABITATS_CV_PLOT = 20
TOP_HABITAT_SHIFTS = 15

# CV thresholds for interpretation
CV_THRESHOLD_LOW = 0.15
CV_THRESHOLD_MODERATE = 0.30
CV_THRESHOLD_HIGH = 0.50

# Correlation thresholds for period stability
CORR_EXCELLENT = 0.90
CORR_GOOD = 0.75
CORR_MODERATE = 0.60

# Species accumulation thresholds
SPECIES_CAPTURE_EXCELLENT = 95.0
SPECIES_CAPTURE_GOOD = 90.0

# Proportion shift threshold
PROPORTION_SHIFT_THRESHOLD = 0.02

# Default split year
DEFAULT_SPLIT_YEAR = 2015

# Figure DPI
FIGURE_DPI = 300

# Habitat CV issue thresholds
HIGH_CV_HABITAT_COUNT_CRITICAL = 5

# Page configuration
st.set_page_config(
    page_title="Sampling Diagnostics - EcoNetMap",
    page_icon="🔍",
    layout="wide"
)

# Title
col1, col2 = st.columns([4, 1])
with col1:
    st.header("Data management")
    st.subheader("🔍 Sampling diagnostics")
    st.markdown("*Examine temporal and spatial patterns in vegetation monitoring data*")
with col2:
    pass

st.markdown("---")

# Session state validation
###################################################################################
if 'data_path' not in st.session_state or st.session_state.data_path is None:
    st.error("⚠️ Project paths not initialized. Please run the **Home** page first to set up your project directory.")
    st.stop()

if 'df_vegetation' not in st.session_state or st.session_state.get('df_vegetation') is None:
    st.error("⚠️ No vegetation data loaded. Please complete **01 Import Data** first.")
    st.stop()

# UTILITY FUNCTIONS
###################################################################################

def save_figure_to_buffer(fig, filename: str, dpi: int = FIGURE_DPI) -> BytesIO:
    """Save matplotlib figure to BytesIO buffer for download"""
    buf = BytesIO()
    try:
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        return buf
    except (OSError, ValueError) as e:
        st.error(f"Error saving figure: {e}")
        return BytesIO()

def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Detect canonical column names in dataframe (see column_config.py)"""
    columns = {}

    if 'year' in df.columns:
        columns['year'] = 'year'
    else:
        st.warning("Could not find year column")

    if 'plot_id' in df.columns:
        columns['plot_id'] = 'plot_id'
    else:
        st.warning("Could not find plot ID column")

    if 'species_key' in df.columns:
        columns['species'] = 'species_key'
    else:
        st.warning("Could not find species column")

    if 'habitat_type' in df.columns:
        columns['habitat'] = 'habitat_type'
    else:
        st.info("No habitat column found - habitat-specific analyses will be skipped")

    return columns

def calculate_temporal_stats(df: pd.DataFrame, cols: Dict[str, str]) -> Optional[pd.DataFrame]:
    """Calculate temporal sampling statistics"""
    if 'year' not in cols or 'plot_id' not in cols:
        return None

    try:
        yearly_stats = df.groupby(cols['year']).agg({
            cols['plot_id']: 'nunique',
            cols['species']: 'nunique' if 'species' in cols else 'count'
        }).reset_index()

        # Count total records
        yearly_records = df.groupby(cols['year']).size().reset_index(name='Total_Records')
        yearly_stats = yearly_stats.merge(yearly_records, on=cols['year'])

        yearly_stats.columns = ['Year', 'Unique_Plots', 'Unique_Species', 'Total_Records']

        return yearly_stats
    except (KeyError, ValueError, TypeError) as e:
        st.error(f"Error calculating temporal statistics: {e}")
        return None

def calculate_habitat_temporal_matrix(df: pd.DataFrame, cols: Dict[str, str]) -> Optional[pd.DataFrame]:
    """Create habitat × year matrix of plot counts"""
    if 'habitat' not in cols or 'year' not in cols or 'plot_id' not in cols:
        return None

    try:
        # Count unique plots per habitat per year
        habitat_temporal = df.groupby([cols['habitat'], cols['year']])[cols['plot_id']].nunique().unstack(fill_value=0)
        return habitat_temporal
    except (KeyError, ValueError, TypeError) as e:
        st.error(f"Error calculating habitat temporal matrix: {e}")
        return None

def calculate_sampling_cv(df: pd.DataFrame, cols: Dict[str, str]) -> Optional[pd.DataFrame]:
    """Calculate coefficient of variation in sampling across years per habitat"""
    if 'habitat' not in cols or 'year' not in cols or 'plot_id' not in cols:
        return None

    try:
        habitat_yearly = df.groupby([cols['habitat'], cols['year']])[cols['plot_id']].nunique().reset_index()

        def safe_cv(x):
            """Calculate CV with proper NaN handling"""
            mean_val = x.mean()
            std_val = x.std()
            if pd.isna(mean_val) or pd.isna(std_val) or mean_val == 0:
                return 0.0
            return std_val / mean_val

        cv_stats = habitat_yearly.groupby(cols['habitat'])[cols['plot_id']].agg([
            ('Mean_Plots', 'mean'),
            ('SD_Plots', 'std'),
            ('CV', safe_cv),
            ('Min_Year_Plots', 'min'),
            ('Max_Year_Plots', 'max')
        ]).reset_index()

        cv_stats.columns = ['Habitat', 'Mean_Plots', 'SD_Plots', 'CV', 'Min_Year_Plots', 'Max_Year_Plots']
        cv_stats = cv_stats.sort_values('CV', ascending=False)

        return cv_stats
    except (KeyError, ValueError, TypeError) as e:
        st.error(f"Error calculating sampling CV: {e}")
        return None

def calculate_species_accumulation(df: pd.DataFrame, years: list, cols: Dict[str, str]) -> Optional[pd.DataFrame]:
    """Calculate cumulative species richness over years"""
    if 'species' not in cols or 'year' not in cols:
        return None

    try:
        cumulative = []

        for year in years:
            species_so_far = df[df[cols['year']] <= year][cols['species']].nunique()
            cumulative.append({'Year': year, 'Cumulative_Species': species_so_far})

        return pd.DataFrame(cumulative)
    except (KeyError, ValueError, TypeError) as e:
        st.error(f"Error calculating species accumulation: {e}")
        return None

def compare_periods(df: pd.DataFrame, split_year: int, cols: Dict[str, str]) -> Optional[Dict]:
    """Compare habitat representation between early and late periods"""
    if 'habitat' not in cols or 'year' not in cols or 'plot_id' not in cols:
        return None

    try:
        early = df[df[cols['year']] <= split_year]
        late = df[df[cols['year']] > split_year]

        early_habitats = early.groupby(cols['habitat'])[cols['plot_id']].nunique()
        late_habitats = late.groupby(cols['habitat'])[cols['plot_id']].nunique()

        # Combine
        comparison = pd.DataFrame({
            'Early_Plots': early_habitats,
            'Late_Plots': late_habitats
        }).fillna(0)

        # Calculate proportions
        comparison['Early_Prop'] = comparison['Early_Plots'] / comparison['Early_Plots'].sum()
        comparison['Late_Prop'] = comparison['Late_Plots'] / comparison['Late_Plots'].sum()
        comparison['Difference'] = comparison['Late_Prop'] - comparison['Early_Prop']
        comparison['Abs_Difference'] = comparison['Difference'].abs()

        # Correlation with NaN handling
        valid_habitats = comparison[(comparison['Early_Plots'] > 0) & (comparison['Late_Plots'] > 0)]
        if len(valid_habitats) > 2:
            # Use pandas corr() which handles NaN better than np.corrcoef
            correlation = valid_habitats['Early_Plots'].corr(valid_habitats['Late_Plots'])
        else:
            correlation = np.nan

        return {
            'comparison': comparison,
            'correlation': correlation,
            'total_early': comparison['Early_Plots'].sum(),
            'total_late': comparison['Late_Plots'].sum()
        }
    except (KeyError, ValueError, TypeError) as e:
        st.error(f"Error comparing periods: {e}")
        return None

# MAIN APPLICATION
###################################################################################

# DATA SOURCE SELECTION
###################################################################################

# Build list of available query .db files
queries_path = Path(st.session_state.data_path).parent / "queries"
db_files = sorted(queries_path.glob("*.db")) if queries_path.exists() else []
db_names = [f.name for f in db_files]

source_options = ["All data (session)"] + db_names

st.markdown("#### 📂 Data source")
selected_source = st.selectbox(
    "Choose the dataset to analyse:",
    options=source_options,
    help="'All data (session)' uses the full vegetation dataset. "
         "Query files are pre-filtered subsets from the queries folder."
)

df = None

if selected_source == "All data (session)":
    df = rename_to_canonical(st.session_state.df_vegetation.copy(), 'columns_vegetation', VEGETATION_ROLES)
    st.success(f"✅ Using session data  —  {len(df):,} records")
else:
    db_path = queries_path / selected_source
    try:
        con = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM data", con)
        con.close()
        st.success(f"✅ Loaded **{selected_source}**  —  {len(df):,} records")
    except Exception as e:
        st.error(f"Could not load {selected_source}: {e}")
        st.stop()

if 'year' not in df.columns:
    st.error("⚠️ No 'year' column mapped. Please map this column on the Data Import page first.")
    st.stop()

st.markdown("---")

# ANALYSIS SECTIONS
###################################################################################

# Detect columns
cols = detect_columns(df)

if 'year' not in cols or 'plot_id' not in cols:
    st.error("Cannot proceed: Missing required columns (year and plot ID)")
    st.stop()

# Data summary
st.header("📊 Dataset overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", f"{len(df):,}")
with col2:
    st.metric("Unique Plots", f"{df[cols['plot_id']].nunique():,}")
with col3:
    if 'species' in cols:
        st.metric("Unique Species", f"{df[cols['species']].nunique():,}")
    else:
        st.metric("Unique Species", "N/A")
with col4:
    year_range = f"{df[cols['year']].min():.0f}-{df[cols['year']].max():.0f}"
    st.metric("Year Range", year_range)

st.markdown("---")

# SECTION 1: TEMPORAL SAMPLING DISTRIBUTION
###################################################################################

st.header("📅 1. Temporal sampling distribution")
st.markdown("*Check if sampling effort was consistent across years*")

yearly_stats = calculate_temporal_stats(df, cols)

if yearly_stats is None:
    st.error("Cannot calculate temporal statistics - missing required columns")
else:
    # Plot: Plots per year
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ax.bar(yearly_stats['Year'], yearly_stats['Unique_Plots'], 
               color='steelblue', alpha=0.7, edgecolor='black')
        
        # Add mean line
        mean_plots = yearly_stats['Unique_Plots'].mean()
        ax.axhline(mean_plots, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_plots:.0f}')
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Unique Plots', fontsize=12)
        ax.set_title('Sampling Effort Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Download button
        buf = save_figure_to_buffer(fig, 'temporal_sampling.png')
        st.download_button(
            label="📥 Download Figure",
            data=buf,
            file_name="temporal_sampling.png",
            mime="image/png",
            key="download_temporal"
        )
        
        plt.close()
    
    with col2:
        st.markdown("**📈 Summary Statistics**")
        
        cv = yearly_stats['Unique_Plots'].std() / yearly_stats['Unique_Plots'].mean()
        min_year = yearly_stats.loc[yearly_stats['Unique_Plots'].idxmin(), 'Year']
        max_year = yearly_stats.loc[yearly_stats['Unique_Plots'].idxmax(), 'Year']
        
        st.metric("Mean Plots/Year", f"{mean_plots:.0f}")
        st.metric("CV (Variation)", f"{cv:.2f}")
        st.metric("Min Year", f"{min_year:.0f} ({yearly_stats['Unique_Plots'].min():,} plots)")
        st.metric("Max Year", f"{max_year:.0f} ({yearly_stats['Unique_Plots'].max():,} plots)")
        
        # Interpretation
        if cv < CV_THRESHOLD_LOW:
            st.success(f"✅ **Low variation** - consistent sampling (CV < {CV_THRESHOLD_LOW})")
        elif cv < CV_THRESHOLD_MODERATE:
            st.warning(f"⚠️ **Moderate variation** - consider temporal stratification (CV = {CV_THRESHOLD_LOW}-{CV_THRESHOLD_MODERATE})")
        else:
            st.error(f"❌ **High variation** - temporal stratification recommended (CV > {CV_THRESHOLD_MODERATE})")
    
    # Show data table
    with st.expander("📋 View Yearly Statistics"):
        st.dataframe(yearly_stats, use_container_width=True, hide_index=True)

st.markdown("---")

# SECTION 2: HABITAT REPRESENTATION OVER TIME
###################################################################################

st.header("🌲 2. Habitat representation over time")
st.markdown("*Check if habitat types were sampled evenly across years*")

habitat_temporal = calculate_habitat_temporal_matrix(df, cols)

if habitat_temporal is not None and len(habitat_temporal) > 0:
    
    # Heatmap
    st.markdown("#### Habitat × year sampling matrix")
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(habitat_temporal) * 0.3)))
    
    # Sort habitats by total plots
    habitat_totals = habitat_temporal.sum(axis=1).sort_values(ascending=False)
    habitat_temporal_sorted = habitat_temporal.loc[habitat_totals.index]
    
    # Take top N habitats for visibility
    if len(habitat_temporal_sorted) > MAX_HABITATS_HEATMAP:
        habitat_temporal_plot = habitat_temporal_sorted.head(MAX_HABITATS_HEATMAP)
        st.info(f"Showing top {MAX_HABITATS_HEATMAP} of {len(habitat_temporal_sorted)} habitat types by total plots")
    else:
        habitat_temporal_plot = habitat_temporal_sorted
    
    sns.heatmap(habitat_temporal_plot, cmap='YlOrRd', annot=False, 
               fmt='d', cbar_kws={'label': 'Number of Plots'}, ax=ax)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Habitat Type', fontsize=12)
    ax.set_title('Sampling Intensity: Habitat Types × Years', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Download button
    buf = save_figure_to_buffer(fig, 'habitat_temporal_heatmap.png')
    st.download_button(
        label="📥 Download Heatmap",
        data=buf,
        file_name="habitat_temporal_heatmap.png",
        mime="image/png",
        key="download_heatmap"
    )
    
    plt.close()

    with st.expander("📋 View heatmap data table"):
        st.dataframe(habitat_temporal_sorted, use_container_width=True)
    
    # Coefficient of Variation by Habitat
    st.markdown("#### Sampling consistency by habitat type")
    
    cv_stats = calculate_sampling_cv(df, cols)
    
    if cv_stats is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bar plot of CV
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot habitats of concern
            plot_data = cv_stats[cv_stats['CV'] <= CV_THRESHOLD_HIGH].sort_values('CV', ascending=False)

            bars = ax.barh(range(len(plot_data)), plot_data['CV'],
                          color=['red' if cv > CV_THRESHOLD_HIGH else 'orange' if cv > CV_THRESHOLD_MODERATE else 'green'
                                 for cv in plot_data['CV']])

            ax.set_yticks(range(len(plot_data)))
            ax.set_yticklabels(plot_data['Habitat'])
            ax.set_xlabel('Coefficient of Variation (CV)', fontsize=12)
            ax.set_ylabel('Habitat Type', fontsize=12)
            ax.set_title(f'Most Variable Habitat Sampling (Top {MAX_HABITATS_CV_PLOT})', fontsize=14, fontweight='bold')
            ax.axvline(CV_THRESHOLD_MODERATE, color='orange', linestyle='--', alpha=0.5, label=f'Moderate ({CV_THRESHOLD_MODERATE})')
            ax.axvline(CV_THRESHOLD_HIGH, color='red', linestyle='--', alpha=0.5, label=f'High ({CV_THRESHOLD_HIGH})')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Download button
            buf = save_figure_to_buffer(fig, 'habitat_cv.png')
            st.download_button(
                label="📥 Download Figure",
                data=buf,
                file_name="habitat_cv.png",
                mime="image/png",
                key="download_cv"
            )
            
            plt.close()
        
        with col2:
            st.markdown("**🎯 Problematic habitats**")

            high_cv = cv_stats[cv_stats['CV'] > CV_THRESHOLD_HIGH]
            moderate_cv = cv_stats[(cv_stats['CV'] > CV_THRESHOLD_MODERATE) & (cv_stats['CV'] <= CV_THRESHOLD_HIGH)]

            st.metric(f"High CV (>{CV_THRESHOLD_HIGH})", len(high_cv))
            st.metric(f"Moderate CV ({CV_THRESHOLD_MODERATE}-{CV_THRESHOLD_HIGH})", len(moderate_cv))
            st.metric(f"Low CV (<{CV_THRESHOLD_MODERATE})", len(cv_stats[cv_stats['CV'] <= CV_THRESHOLD_MODERATE]))
            
            low_cv = cv_stats[cv_stats['CV'] <= CV_THRESHOLD_LOW]

            if len(high_cv) > 0:
                st.info(f"ℹ️ **{len(high_cv)} habitats** with very high CV — reflects planned monitoring rotation")
            if len(moderate_cv) > 0:
                st.warning(f"⚠️ **{len(moderate_cv)} habitats** with moderate sampling variation")
            if len(low_cv) > 0:
                st.warning(f"⚠️ **{len(low_cv)} habitats** with very low or zero CV — likely sampled in too few years to calculate meaningful variation. These habitats may be severely under-represented and should be interpreted with caution.")
            if len(high_cv) == 0 and len(moderate_cv) == 0 and len(low_cv) == 0:
                st.success("✅ All habitats sampled consistently")
        
        # Show detailed table
        with st.expander("📋 View Habitat Sampling Statistics"):
            st.dataframe(cv_stats, use_container_width=True, hide_index=True)

else:
    st.info("No habitat information found in dataset")

st.markdown("---")

# SECTION 3: SPECIES RICHNESS DISTRIBUTION BY HABITAT
###################################################################################

st.header("🌱 3. Species richness distribution by habitat type")
st.markdown("*Distribution of species counts per plot within each habitat type*")

if 'habitat' in cols and 'species' in cols:
    # Calculate species richness per plot
    richness = df.groupby([cols['plot_id'], cols['habitat']])[cols['species']].nunique().reset_index()
    richness.columns = ['plot_id', 'Habitat', 'Species_Richness']

    # Order habitats by median richness
    habitat_order = (richness.groupby('Habitat')['Species_Richness']
                     .median()
                     .sort_values(ascending=False)
                     .index.tolist())

    fig, ax = plt.subplots(figsize=(14, max(6, len(habitat_order) * 0.4)))

    richness_grouped = [richness[richness['Habitat'] == h]['Species_Richness'].values
                        for h in habitat_order]

    bp = ax.boxplot(richness_grouped, vert=False, patch_artist=True,
                    flierprops=dict(marker='o', markersize=2, alpha=0.3, linestyle='none'),
                    medianprops=dict(color='black', linewidth=2))

    # Colour boxes by median richness
    medians = [data.median() for data in [richness[richness['Habitat'] == h]['Species_Richness']
                                           for h in habitat_order]]
    cmap = plt.cm.YlGn
    norm = plt.Normalize(min(medians), max(medians))
    for patch, med in zip(bp['boxes'], medians):
        patch.set_facecolor(cmap(norm(med)))
        patch.set_alpha(0.8)

    ax.set_yticks(range(1, len(habitat_order) + 1))
    ax.set_yticklabels(habitat_order, fontsize=9)
    ax.set_xlabel('Species richness per plot', fontsize=12)
    ax.set_title('Species Richness Distribution by Habitat Type', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    st.pyplot(fig)

    buf = save_figure_to_buffer(fig, 'species_richness_by_habitat.png')
    st.download_button(label="📥 Download Figure", data=buf,
                       file_name="species_richness_by_habitat.png", mime="image/png",
                       key="download_richness")
    plt.close()

    with st.expander("📋 View richness statistics by habitat"):
        richness_stats = (richness.groupby('Habitat')['Species_Richness']
                          .agg(Plots='count', Min='min', Q25=lambda x: x.quantile(0.25),
                               Median='median', Q75=lambda x: x.quantile(0.75), Max='max',
                               Mean='mean')
                          .round(1)
                          .sort_values('Median', ascending=False)
                          .reset_index())
        st.dataframe(richness_stats, use_container_width=True, hide_index=True)
else:
    st.info("Habitat or species column not found - skipping species richness analysis")

st.markdown("---")

# SECTION 4: EARLY VS LATE PERIOD COMPARISON
###################################################################################

st.header("⏱️ 4. Early vs. late period comparison")
st.markdown("*Compare habitat representation between reference and validation periods*")

split_year = st.slider(
    "Split year (early ≤ year, late > year):",
    min_value=int(df[cols['year']].min()),
    max_value=int(df[cols['year']].max()),
    value=DEFAULT_SPLIT_YEAR,
    help="Year that divides early (reference) from late (validation) period"
)

period_comparison = compare_periods(df, split_year, cols)

if period_comparison is not None:
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Early Period Plots", f"{period_comparison['total_early']:,.0f}")
    with col2:
        st.metric("Late Period Plots", f"{period_comparison['total_late']:,.0f}")
    with col3:
        corr = period_comparison['correlation']
        st.metric("Habitat Correlation", f"{corr:.3f}" if not np.isnan(corr) else "N/A")
    
    # Scatter plot: Early vs Late
    comparison_df = period_comparison['comparison']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Remove zeros for log scale
        plot_data = comparison_df[(comparison_df['Early_Plots'] > 0) & 
                                  (comparison_df['Late_Plots'] > 0)]
        
        ax.scatter(plot_data['Early_Plots'], plot_data['Late_Plots'], 
                  alpha=0.6, s=80, edgecolor='black')
        
        # Add 1:1 line
        max_val = max(plot_data['Early_Plots'].max(), plot_data['Late_Plots'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='1:1 line')
        
        ax.set_xlabel(f'Early Period Plots (≤{split_year})', fontsize=12)
        ax.set_ylabel(f'Late Period Plots (>{split_year})', fontsize=12)
        ax.set_title('Habitat Representation: Early vs Late', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Log scale if large range
        if max_val > 1000:
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Download button
        buf = save_figure_to_buffer(fig, 'early_vs_late_scatter.png')
        st.download_button(
            label="📥 Download Scatter Plot",
            data=buf,
            file_name="early_vs_late_scatter.png",
            mime="image/png",
            key="download_scatter"
        )
        
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 8))

        # Proportion shift
        shift_data = comparison_df.sort_values('Abs_Difference', ascending=False).head(TOP_HABITAT_SHIFTS)
        
        colors = ['red' if d > 0 else 'blue' for d in shift_data['Difference']]
        
        ax.barh(range(len(shift_data)), shift_data['Difference'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(shift_data)))
        ax.set_yticklabels(shift_data.index)
        ax.set_xlabel('Proportion Shift (Late - Early)', fontsize=12)
        ax.set_ylabel('Habitat Type', fontsize=12)
        ax.set_title('Largest Habitat Proportion Shifts', fontsize=14, fontweight='bold')
        ax.axvline(0, color='black', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Download button
        buf = save_figure_to_buffer(fig, 'habitat_proportion_shifts.png')
        st.download_button(
            label="📥 Download Shifts Chart",
            data=buf,
            file_name="habitat_proportion_shifts.png",
            mime="image/png",
            key="download_shifts"
        )
        
        plt.close()
    
    # Interpretation
    st.markdown("#### 🔍 Interpretation")

    if not np.isnan(corr):
        if corr > CORR_EXCELLENT:
            st.success(f"✅ **Excellent stability** (r = {corr:.3f}) - Habitat proportions very similar between periods")
        elif corr > CORR_GOOD:
            st.success(f"✅ **Good stability** (r = {corr:.3f}) - Habitat proportions reasonably similar")
        elif corr > CORR_MODERATE:
            st.warning(f"⚠️ **Moderate stability** (r = {corr:.3f}) - Some habitat shifts occurred")
        else:
            st.error(f"❌ **Poor stability** (r = {corr:.3f}) - Substantial habitat composition changes")

    # Show largest shifts
    large_shifts = comparison_df[comparison_df['Abs_Difference'] > PROPORTION_SHIFT_THRESHOLD].sort_values('Abs_Difference', ascending=False)
    
    if len(large_shifts) > 0:
        st.warning(f"⚠️ **{len(large_shifts)} habitats** show >{PROPORTION_SHIFT_THRESHOLD*100:.0f}% proportion shift between periods")
        
        with st.expander("📋 View Habitats with Large Shifts"):
            display_cols = ['Early_Plots', 'Late_Plots', 'Early_Prop', 'Late_Prop', 'Difference']
            st.dataframe(large_shifts[display_cols], use_container_width=True)

st.markdown("---")

# SECTION 5: SPECIES ACCUMULATION
###################################################################################

st.header("🌿 5. Species accumulation over time")
st.markdown("*Check if early years captured most species diversity*")

if 'species' in cols:
    years = sorted(df[cols['year']].unique())
    accumulation = calculate_species_accumulation(df, years, cols)

    if accumulation is not None:
        # Initialize variables for split year analysis
        split_species = None
        pct_by_split = None

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(accumulation['Year'], accumulation['Cumulative_Species'],
                linewidth=3, marker='o', markersize=6, color='darkgreen')

        # Mark split year if in range
        if split_year in years:
            split_species = accumulation[accumulation['Year'] == split_year]['Cumulative_Species'].values[0]
            final_species = accumulation['Cumulative_Species'].max()
            pct_by_split = (split_species / final_species) * 100

            ax.axvline(split_year, color='red', linestyle='--', linewidth=2,
                      label=f'Split year ({split_year}): {pct_by_split:.1f}% of total species')
            ax.axhline(split_species, color='red', linestyle=':', alpha=0.5)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Cumulative Species', fontsize=12)
        ax.set_title('Species Accumulation Curve', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Download button
        buf = save_figure_to_buffer(fig, 'species_accumulation.png')
        st.download_button(
            label="📥 Download Accumulation Curve",
            data=buf,
            file_name="species_accumulation.png",
            mime="image/png",
            key="download_accumulation"
        )

        plt.close()

        # Summary
        col1, col2, col3 = st.columns(3)

        final_species = accumulation['Cumulative_Species'].max()

        with col1:
            st.metric("Total Species", f"{final_species:,}")

        with col2:
            if split_species is not None:
                st.metric(f"Species by {split_year}", f"{split_species:,}")

        with col3:
            if pct_by_split is not None:
                st.metric("% Captured by Split", f"{pct_by_split:.1f}%")

        if pct_by_split is not None:
            if pct_by_split > SPECIES_CAPTURE_EXCELLENT:
                st.success(f"✅ Early period captured >{SPECIES_CAPTURE_EXCELLENT:.0f}% of total species diversity")
            elif pct_by_split > SPECIES_CAPTURE_GOOD:
                st.success(f"✅ Early period captured >{SPECIES_CAPTURE_GOOD:.0f}% of total species diversity")
            else:
                st.warning(f"⚠️ Early period captured only {pct_by_split:.1f}% of total species - many new species in late period")
else:
    st.info("Species column not found - skipping species accumulation analysis")

st.markdown("---")

# SECTION 6: RECOMMENDATIONS
###################################################################################

st.header("💡 6. Recommendations")
st.markdown("*Summary and suggested actions based on diagnostics*")

# Collect issues
issues = []
recommendations = []

# Check temporal CV
if yearly_stats is not None:
    temporal_cv = yearly_stats['Unique_Plots'].std() / yearly_stats['Unique_Plots'].mean()
    if temporal_cv > CV_THRESHOLD_MODERATE:
        issues.append(f"❌ High temporal variation in sampling (CV > {CV_THRESHOLD_MODERATE})")
        recommendations.append("Implement **temporal stratification**: Sample equally from each year when creating reference sets")
    elif temporal_cv > CV_THRESHOLD_LOW:
        issues.append(f"⚠️ Moderate temporal variation (CV = {CV_THRESHOLD_LOW}-{CV_THRESHOLD_MODERATE})")
        recommendations.append("Consider **temporal weighting** or stratification to balance year representation")
    else:
        st.success(f"✅ **Temporal sampling**: Consistent across years (CV < {CV_THRESHOLD_LOW})")

# Check habitat CV
if 'habitat' in cols:
    cv_stats = calculate_sampling_cv(df, cols)
    if cv_stats is not None:
        high_cv_habitats = len(cv_stats[cv_stats['CV'] > CV_THRESHOLD_HIGH])
        if high_cv_habitats > HIGH_CV_HABITAT_COUNT_CRITICAL:
            issues.append(f"❌ {high_cv_habitats} habitats with highly uneven temporal sampling (CV > {CV_THRESHOLD_HIGH})")
            recommendations.append("Consider **excluding** or **downweighting** problematic habitats in analysis")
        elif high_cv_habitats > 0:
            issues.append(f"⚠️ {high_cv_habitats} habitats with uneven sampling")

# Check period correlation
if period_comparison is not None and not np.isnan(period_comparison['correlation']):
    corr = period_comparison['correlation']
    if corr < CORR_GOOD:
        issues.append(f"❌ Poor habitat stability between periods (r = {corr:.2f})")
        recommendations.append("**Test sensitivity**: Validate with different temporal splits to assess robustness")
    elif corr < CORR_EXCELLENT:
        issues.append(f"⚠️ Moderate habitat shifts between periods (r = {corr:.2f})")
    else:
        st.success(f"✅ **Habitat stability**: Excellent correlation between periods (r = {corr:.2f})")

# Check species accumulation
if 'species' in cols:
    years_check = sorted(df[cols['year']].unique())
    accumulation_check = calculate_species_accumulation(df, years_check, cols)

    if accumulation_check is not None and split_year in years_check:
        split_species_check = accumulation_check[accumulation_check['Year'] == split_year]['Cumulative_Species'].values[0]
        final_species_check = accumulation_check['Cumulative_Species'].max()
        pct_by_split_check = (split_species_check / final_species_check) * 100

        if pct_by_split_check < SPECIES_CAPTURE_GOOD:
            issues.append(f"⚠️ Only {pct_by_split_check:.1f}% of species captured by split year")
            recommendations.append("Consider extending early period or documenting **novel species** in validation")
        else:
            st.success(f"✅ **Species coverage**: Early period captures {pct_by_split_check:.1f}% of total diversity")

# Display issues and recommendations
if len(issues) > 0:
    st.markdown("#### ⚠️ Issues detected:")
    for issue in issues:
        st.markdown(f"- {issue}")
    
    st.markdown("#### 🔧 Recommended actions:")
    for rec in recommendations:
        st.markdown(f"- {rec}")
else:
    st.success("✅ **No major issues detected** - data appears suitable for temporal validation without additional stratification")

# Final summary box
st.markdown("---")
st.info("""
**📋 Next Steps:**

1. Review all diagnostic plots and statistics above
2. If issues detected, implement recommended sampling strategies
3. Document findings in Methods section of manuscript
4. Proceed with creating reference datasets using appropriate stratification
""")


# Footer
st.markdown("---")
st.caption("EcoNetMap Sampling Diagnostics | Flemming Skov | Aarhus University | 2026")
