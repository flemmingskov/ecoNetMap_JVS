"""
EcoNetMap - Reference Map Validation Analysis (Corrected Final Version)
=========================================================================
Comprehensive empirical validation of network-based ecological cartography.

Validation Analyses:
1. Species Position Stability (Procrustes R²)
2. Environmental Gradient Emergence (Ellenberg R²)
3. Distance Preservation with Spatial Variation (colorblind-safe)
4. Predictive Accuracy (k-NN Cross-Validation)
5. Habitat Environmental Differentiation (ANOVA)
6. Habitat Spatial Clustering (Discrimination Index)

Individual map scoring and ranking to identify optimal reference map.

Part of the EcoNetMap toolkit - Network-Based Ecological Cartography
Author: Flemming Skov
Last Updated: February 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import datetime
from itertools import combinations
from io import BytesIO

from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, f_oneway
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# Page configuration
st.set_page_config(
    page_title="Validation - EcoNetMap", 
    page_icon="🔬", 
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 5px 0;
    }
    .best-map {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
col1, col2 = st.columns([4, 1])
with col1:
    st.header("Network validation")
    st.subheader("🔬 Reference map validation analysis")
    st.markdown("*Comprehensive empirical assessment with individual map ranking*")
with col2:
    pass

st.markdown("---")

# HELPER FUNCTIONS
###################################################################################

def load_overlay_db(path: Path) -> dict:
    """Load overlay database tables"""
    try:
        conn = sqlite3.connect(str(path))
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", 
            conn
        )['name'].tolist()
        
        result = {}
        if 'taxa' in tables:
            result['taxa'] = pd.read_sql_query("SELECT * FROM taxa", conn)
        if 'plot_id' in tables:
            result['plot_id'] = pd.read_sql_query("SELECT * FROM plot_id", conn)
        if 'metadata' in tables:
            result['metadata'] = pd.read_sql_query("SELECT * FROM metadata", conn)
        
        conn.close()
        return result if result else None
    except Exception as e:
        st.error(f"Error loading {path.name}: {e}")
        return None

def detect_method(name: str, metadata: pd.DataFrame = None) -> str:
    """Detect layout method from overlay name or metadata"""
    name_lower = name.lower()
    
    if metadata is not None and 'layout_method' in metadata.columns:
        method = metadata['layout_method'].iloc[0]
        if 'Fruchterman' in str(method):
            return 'FR'
        elif 'MDS' in str(method) or 'Multidimensional' in str(method):
            return 'MDS'
    
    if 'fr' in name_lower and 'mds' not in name_lower:
        return 'FR'
    elif 'mds' in name_lower:
        return 'MDS'
    else:
        return 'Unknown'

def calc_mds_stress(coords_2d: np.ndarray, original_dists: np.ndarray) -> dict:
    """
    Calculate MDS stress and create data for Shepard diagram.

    Stress measures how well 2D distances match original ecological distances.
    Lower stress = better preservation of original relationships.

    Stress formula: sqrt(sum((d_2D - d_original)²) / sum(d_original²))

    Args:
        coords_2d: 2D coordinates (n_species x 2)
        original_dists: Original distance vector (from pdist) - ecological dissimilarity

    Returns:
        dict with stress value, R², and data for Shepard diagram
    """
    # Calculate 2D Euclidean distances
    dists_2d = pdist(coords_2d, metric='euclidean')

    # Stress (normalized RMSE)
    numerator = np.sum((dists_2d - original_dists)**2)
    denominator = np.sum(original_dists**2)
    stress = np.sqrt(numerator / denominator) if denominator > 0 else np.nan

    # R² (how well 2D distances predict original distances)
    r2 = r2_score(original_dists, dists_2d)

    # Correlation for Shepard diagram
    corr, p_value = pearsonr(original_dists, dists_2d)

    return {
        'stress': stress,
        'r2': r2,
        'correlation': corr,
        'p_value': p_value,
        'original_dists': original_dists,
        'dists_2d': dists_2d
    }

def calc_procrustes_r2(c1: np.ndarray, c2: np.ndarray) -> float:
    """
    Calculate Procrustes R² (variance explained after optimal alignment).
    
    scipy.spatial.procrustes standardizes both matrices to unit Frobenius norm
    (centered, then divided by root-sum-of-squares). After standardization,
    the total sum of squares of the reference matrix is exactly 1.0.
    The returned 'disparity' = sum((m1 - m2)²), so R² = 1 - disparity.
    
    Note: Previous implementation used ss_total from the ORIGINAL (unstandardized)
    coordinates while ss_residual came from the STANDARDIZED outputs, creating a
    scale mismatch that inflated R² toward 1.0 regardless of actual fit quality.
    """
    m1, m2, disparity = procrustes(c1, c2)
    # disparity = sum of squared differences between standardized, aligned matrices
    # ss_total of standardized reference = 1.0 (by definition of Frobenius normalization)
    # Therefore: R² = 1 - disparity / 1.0 = 1 - disparity
    r2 = max(0.0, min(1.0, 1.0 - disparity))
    return r2

def calc_gradient_null_model(df: pd.DataFrame, indicator: str, n_permutations: int = 999) -> dict:
    """
    Test gradient significance using permutation null model.

    Permutes indicator values and recalculates R² to generate null distribution.
    This tests whether observed R² is better than random.

    Args:
        df: Plot-level data with coordinates and indicator values
        indicator: Which Ellenberg indicator to test
        n_permutations: Number of permutations

    Returns:
        dict with observed R², null distribution, and p-value
    """
    data = df[['xcoor', 'ycoor', indicator]].dropna()
    if len(data) < 30:  # Match calc_gradient minimum
        return None

    X = data[['xcoor', 'ycoor']].values
    y_obs = data[indicator].values

    # Calculate observed R²
    X_poly = np.column_stack([
        X[:, 0], X[:, 1],
        X[:, 0]*X[:, 1],
        X[:, 0]**2, X[:, 1]**2
    ])
    model_obs = LinearRegression().fit(X_poly, y_obs)
    r2_obs = r2_score(y_obs, model_obs.predict(X_poly))

    # Permutation test
    null_r2s = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y_obs)
        model_perm = LinearRegression().fit(X_poly, y_perm)
        r2_perm = r2_score(y_perm, model_perm.predict(X_poly))
        null_r2s.append(r2_perm)

    null_r2s = np.array(null_r2s)

    # One-tailed p-value (observed should be higher than null)
    # Uses (count + 1)/(n + 1) to avoid p = 0 (Phipson & Smyth, 2010)
    p_value = (np.sum(null_r2s >= r2_obs) + 1) / (n_permutations + 1)

    return {
        'r2_observed': r2_obs,
        'r2_null_mean': null_r2s.mean(),
        'r2_null_95': np.percentile(null_r2s, 95),
        'null_distribution': null_r2s,
        'p_value': p_value,
        'n_permutations': n_permutations
    }

def calc_gradient(df: pd.DataFrame, indicator: str) -> dict:
    """
    Calculate environmental gradient metrics using polynomial regression.
    
    Input: Plot coordinates (x, y) and indicator values
    Model: indicator = f(x, y, x*y, x², y²)  [5 predictors]
    Output: R², adjusted R², gradient direction, sample size
    """
    data = df[['xcoor', 'ycoor', indicator]].dropna()
    if len(data) < 30:  # Increased from 10: need sufficient df for 5 predictors + intercept
        return None
    
    X = data[['xcoor', 'ycoor']].values
    y = data[indicator].values
    
    # Polynomial model (captures non-linear structure)
    X_poly = np.column_stack([
        X[:, 0], X[:, 1], 
        X[:, 0]*X[:, 1], 
        X[:, 0]**2, X[:, 1]**2
    ])
    model = LinearRegression().fit(X_poly, y)
    r2 = r2_score(y, model.predict(X_poly))
    
    # Adjusted R² (corrects for number of predictors)
    n = len(y)
    p = X_poly.shape[1]  # 5 predictors
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # Linear gradient direction
    model_lin = LinearRegression().fit(X, y)
    grad = model_lin.coef_
    grad_norm = np.linalg.norm(grad)
    grad_unit = grad / grad_norm if grad_norm > 0 else grad
    
    return {
        'r2': r2, 
        'adj_r2': adj_r2,
        'gradient': grad, 
        'gradient_unit': grad_unit, 
        'n': len(data)
    }

def calc_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate angle between two vectors in degrees"""
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return np.nan
    
    v1_unit = v1 / v1_norm
    v2_unit = v2 / v2_norm
    
    cos_angle = np.clip(np.dot(v1_unit, v2_unit), -1, 1)
    angle = np.degrees(np.arccos(cos_angle))
    
    return angle

def calc_all_angles(gradients: dict) -> dict:
    """Calculate all pairwise angles between gradients"""
    indicators = list(gradients.keys())
    angles = {}
    
    for ind1, ind2 in combinations(indicators, 2):
        if gradients[ind1] and gradients[ind2]:
            angle = calc_angle(
                gradients[ind1]['gradient'], 
                gradients[ind2]['gradient']
            )
            angles[f"{ind1}⊥{ind2}"] = angle
    
    return angles

def mantel_test(dist1: np.ndarray, dist2: np.ndarray, n_permutations: int = 999) -> dict:
    """
    Mantel test for correlation between two distance matrices.

    Standard approach in ecology for testing distance matrix correlations.
    Uses permutation test to assess significance.

    Args:
        dist1: First distance vector (from pdist)
        dist2: Second distance vector (from pdist)
        n_permutations: Number of permutations for significance testing

    Returns:
        dict with correlation, p_value, and null distribution
    """
    # Observed correlation
    obs_corr, _ = pearsonr(dist1, dist2)

    # Permutation test
    n = len(dist1)
    # Convert to square matrices for permutation
    mat1_square = squareform(dist1)
    mat2 = dist2.copy()

    null_corrs = []
    for _ in range(n_permutations):
        # Permute rows/columns of first matrix
        perm_idx = np.random.permutation(len(mat1_square))
        mat1_perm = mat1_square[perm_idx][:, perm_idx]
        # Convert back to distance vector
        dist1_perm = squareform(mat1_perm, checks=False)
        # Calculate correlation
        null_corr, _ = pearsonr(dist1_perm, mat2)
        null_corrs.append(null_corr)

    null_corrs = np.array(null_corrs)

    # Two-tailed p-value (corrected: Phipson & Smyth, 2010)
    p_value = (np.sum(np.abs(null_corrs) >= np.abs(obs_corr)) + 1) / (n_permutations + 1)

    return {
        'correlation': obs_corr,
        'p_value': p_value,
        'null_corrs': null_corrs,
        'n_permutations': n_permutations
    }

def calc_distance_preservation_global(plot_id: pd.DataFrame, use_mantel: bool = True) -> dict:
    """
    Calculate global distance preservation using Mantel test.
    Tests if 2D Euclidean distances correlate with ecological dissimilarities.

    Args:
        plot_id: Plot-level data with coordinates and indicators
        use_mantel: If True, use Mantel test (with permutations); if False, use simple Pearson
    """
    plots = plot_id[['xcoor', 'ycoor']].dropna()

    if len(plots) < 10:
        return None

    # Sample if too many plots (for computational efficiency)
    if len(plots) > 300:
        plots = plots.sample(n=300, random_state=42)

    plot_indices = plots.index.tolist()

    # Calculate pairwise 2D Euclidean distances
    coords = plots[['xcoor', 'ycoor']].values
    euclidean_dists = pdist(coords, metric='euclidean')

    # Calculate ecological dissimilarities using indicator values
    # Note: L, M, R, N are Ellenberg indicators (M=Moisture, L=Light, R=Reaction/pH, N=Nitrogen)
    indicators = ['L', 'M', 'R', 'N']
    available_indicators = [ind for ind in indicators if ind in plot_id.columns]

    if len(available_indicators) < 2:
        return None

    plot_features = plot_id.loc[plot_indices, available_indicators].dropna()
    # Re-align: only keep plots that have all indicator values
    valid_indices = plot_features.index.tolist()
    if len(valid_indices) < 10:
        return None
    coords = plot_id.loc[valid_indices, ['xcoor', 'ycoor']].values
    euclidean_dists = pdist(coords, metric='euclidean')
    plot_features = plot_features.values
    feature_dists = pdist(plot_features, metric='euclidean')

    # Normalize to 0-1
    max_feature_dist = feature_dists.max()
    if max_feature_dist > 0:
        feature_dists = feature_dists / max_feature_dist

    # Use Mantel test or simple correlation
    if len(euclidean_dists) > 0 and len(feature_dists) > 0:
        if use_mantel:
            result = mantel_test(euclidean_dists, feature_dists, n_permutations=999)
            return {
                'correlation': result['correlation'],
                'p_value': result['p_value'],
                'n_plots': len(plots),
                'n_pairs': len(euclidean_dists),
                'method': 'Mantel test'
            }
        else:
            corr, p_value = pearsonr(euclidean_dists, feature_dists)
            return {
                'correlation': corr,
                'p_value': p_value,
                'n_plots': len(plots),
                'n_pairs': len(euclidean_dists),
                'method': 'Pearson'
            }

    return None

def calc_distance_preservation_spatial(plot_id: pd.DataFrame, n_regions=16) -> dict:
    """
    Calculate distance preservation in 16 spatial regions (4x4 grid).
    Tests if preservation varies across landscape (dense vs sparse areas).
    """
    plots = plot_id[['xcoor', 'ycoor']].copy()
    
    if len(plots) < 50:
        return None
    
    # Divide into 4x4 grid
    x_edges = np.linspace(plots['xcoor'].min(), plots['xcoor'].max(), 5)
    y_edges = np.linspace(plots['ycoor'].min(), plots['ycoor'].max(), 5)
    
    plots['x_bin'] = pd.cut(plots['xcoor'], bins=x_edges, labels=False, include_lowest=True)
    plots['y_bin'] = pd.cut(plots['ycoor'], bins=y_edges, labels=False, include_lowest=True)
    plots['region'] = plots['y_bin'] * 4 + plots['x_bin']
    
    # Get indicator columns
    indicators = ['M', 'L', 'N', 'R']
    available_indicators = [ind for ind in indicators if ind in plot_id.columns]
    
    if len(available_indicators) < 2:
        return None
    
    # Calculate preservation for each region
    regional_results = np.full((4, 4), np.nan)
    
    for region in range(16):
        region_indices = plots[plots['region'] == region].index.tolist()
        
        if len(region_indices) >= 10:
            # Get coordinates and features (drop NaN instead of filling with 0)
            region_features = plot_id.loc[region_indices, available_indicators].dropna()
            valid_region_indices = region_features.index.tolist()
            
            if len(valid_region_indices) < 10:
                continue
                
            region_coords = plot_id.loc[valid_region_indices, ['xcoor', 'ycoor']].values
            region_features = region_features.values
            
            # Calculate distances
            euclidean_dists = pdist(region_coords, metric='euclidean')
            feature_dists = pdist(region_features, metric='euclidean')
            
            # Normalize
            max_feature = feature_dists.max()
            if max_feature > 0:
                feature_dists = feature_dists / max_feature
            
            # Correlate
            if len(euclidean_dists) > 5 and len(feature_dists) > 5:
                try:
                    corr, _ = pearsonr(euclidean_dists, feature_dists)
                    y_idx = region // 4
                    x_idx = region % 4
                    regional_results[y_idx, x_idx] = corr
                except:
                    pass
    
    return {
        'regional_matrix': regional_results,
        'mean_correlation': np.nanmean(regional_results),
        'std_correlation': np.nanstd(regional_results)
    }

def cross_validate_predictions(plot_id: pd.DataFrame, indicator: str, 
                               k: int = 5, n_samples: int = 100) -> dict:
    """
    k-NN cross-validation for predicting indicator values from position.
    
    For each plot (up to n_samples):
    1. Remove it from dataset
    2. Find k nearest neighbors by Euclidean distance
    3. Predict indicator as mean of neighbors
    4. Calculate prediction accuracy
    """
    data = plot_id[['xcoor', 'ycoor', indicator]].dropna()
    
    if len(data) < k + 5:
        return None
    
    # Sample if requested
    if len(data) > n_samples:
        data = data.sample(n=n_samples, random_state=42)
    
    coords = data[['xcoor', 'ycoor']].values
    values = data[indicator].values
    
    predictions = []
    actuals = []
    
    # Leave-one-out prediction
    for i in range(len(data)):
        train_coords = np.delete(coords, i, axis=0)
        train_values = np.delete(values, i)
        test_coord = coords[i].reshape(1, -1)
        test_value = values[i]
        
        # Find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
        nbrs.fit(train_coords)
        distances, indices = nbrs.kneighbors(test_coord)
        
        # Predict as mean of neighbors
        neighbor_values = train_values[indices[0]]
        prediction = np.mean(neighbor_values)
        
        predictions.append(prediction)
        actuals.append(test_value)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    r2 = r2_score(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    correlation, p_value = pearsonr(actuals, predictions)
    
    return {
        'r2': r2,
        'rmse': rmse,
        'correlation': correlation,
        'p_value': p_value,
        'n': len(data),
        'k': k
    }

def interpret_prediction(r2: float, rmse: float, indicator: str) -> str:
    """Generate interpretation text for prediction results"""
    # R² interpretation
    if r2 > 0.7:
        quality = "Strong predictive power"
    elif r2 > 0.5:
        quality = "Good predictive power"
    elif r2 > 0.3:
        quality = "Moderate predictive power"
    else:
        quality = "Weak predictive power"
    
    # RMSE interpretation
    if rmse < 1.0:
        accuracy = "Predictions accurate within ~1 unit"
    elif rmse < 2.0:
        accuracy = "Predictions accurate within ~2 units"
    else:
        accuracy = "Predictions have >2 unit error"
    
    return f"{quality}. {accuracy}."

def analyze_habitat_differentiation(plot_id: pd.DataFrame, 
                                    habitat_col: str,
                                    indicators: list = ['M', 'L', 'N', 'R']) -> dict:
    """
    Test if habitat types differ in environmental conditions using ANOVA.
    
    Uses plot-level data grouped by habitat classification (habitat_type or major_type).
    Tests whether field-classified habitats correspond to environmental differences.
    """
    if habitat_col not in plot_id.columns:
        return None
    
    # Get plots with habitat classification
    data = plot_id[[habitat_col] + indicators].dropna(subset=[habitat_col])
    
    if len(data) < 20:
        return None
    
    # Get unique habitats
    habitats = data[habitat_col].unique()
    
    if len(habitats) < 2:
        return None
    
    results = {}
    
    for indicator in indicators:
        if indicator not in data.columns:
            continue
        
        # Remove NaN for this indicator
        indicator_data = data[[habitat_col, indicator]].dropna()
        
        if len(indicator_data) < 10:
            continue
        
        # Calculate means per habitat
        habitat_means = indicator_data.groupby(habitat_col)[indicator].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('n', 'count')
        ])
        
        # ANOVA test
        habitat_groups = indicator_data[habitat_col].unique()
        if len(habitat_groups) < 2:
            continue
        
        groups = [indicator_data[indicator_data[habitat_col] == h][indicator].values 
                  for h in habitat_groups]
        
        try:
            f_stat, p_value = f_oneway(*groups)
            
            results[indicator] = {
                'habitat_means': habitat_means,
                'f_stat': f_stat,
                'p_value': p_value,
                'n_habitats': len(habitat_groups)
            }
        except:
            pass
    
    return results if results else None

def analyze_habitat_spatial_clustering(plot_id: pd.DataFrame, 
                                       habitat_col: str) -> dict:
    """
    Analyze spatial clustering of habitat types in the map.
    
    Tests whether plots of the same habitat type cluster together spatially,
    indicating good discriminative power of the reference map.
    
    Metrics:
    - Within-habitat variance: Spread of plots around habitat centroid
    - Between-habitat distance: Separation of habitat centroids
    - Discrimination index: Ratio of between/within distances
    
    Lower within-variance and higher discrimination index = better map discrimination.
    """
    if habitat_col not in plot_id.columns:
        return None
    
    # Get plots with coordinates and habitat
    data = plot_id[[habitat_col, 'xcoor', 'ycoor']].dropna()
    
    if len(data) < 10:
        return None
    
    habitats = data[habitat_col].unique()
    
    if len(habitats) < 2:
        return None
    
    habitat_stats = {}
    
    for habitat in habitats:
        habitat_plots = data[data[habitat_col] == habitat]
        
        if len(habitat_plots) < 2:
            continue
        
        # Centroid (mean position)
        centroid_x = habitat_plots['xcoor'].mean()
        centroid_y = habitat_plots['ycoor'].mean()
        
        # Variance (mean squared distance from centroid)
        distances_sq = ((habitat_plots['xcoor'] - centroid_x)**2 + 
                       (habitat_plots['ycoor'] - centroid_y)**2)
        variance = distances_sq.mean()
        
        habitat_stats[habitat] = {
            'n': len(habitat_plots),
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'variance': variance,
            'std_dev': np.sqrt(variance)
        }
    
    if len(habitat_stats) < 2:
        return None
    
    # Calculate between-habitat distances (pairwise centroid distances)
    centroids = [(s['centroid_x'], s['centroid_y']) for s in habitat_stats.values()]
    between_distances = []
    
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 + 
                          (centroids[i][1] - centroids[j][1])**2)
            between_distances.append(dist)
    
    # Summary metrics
    mean_within_variance = np.mean([s['variance'] for s in habitat_stats.values()])
    mean_within_std = np.sqrt(mean_within_variance)
    mean_between_distance = np.mean(between_distances) if between_distances else 0
    
    # Discrimination index (higher = better separation)
    # Ratio of between-habitat distance to within-habitat spread
    discrimination_index = mean_between_distance / mean_within_std if mean_within_std > 0 else 0
    
    # Permutation test: shuffle habitat labels to test significance
    n_permutations = 199  # Fewer than Mantel (less critical, computationally lighter)
    null_disc_indices = []
    habitat_labels = data[habitat_col].values.copy()
    coords_array = data[['xcoor', 'ycoor']].values
    
    for _ in range(n_permutations):
        perm_labels = np.random.permutation(habitat_labels)
        perm_within_vars = []
        perm_centroids = []
        
        for habitat in habitats:
            mask = perm_labels == habitat
            if mask.sum() < 2:
                continue
            hab_coords = coords_array[mask]
            centroid = hab_coords.mean(axis=0)
            var = np.mean(np.sum((hab_coords - centroid)**2, axis=1))
            perm_within_vars.append(var)
            perm_centroids.append(centroid)
        
        if len(perm_centroids) >= 2:
            perm_between = []
            for ci in range(len(perm_centroids)):
                for cj in range(ci+1, len(perm_centroids)):
                    perm_between.append(np.linalg.norm(
                        perm_centroids[ci] - perm_centroids[cj]))
            perm_mean_within_std = np.sqrt(np.mean(perm_within_vars))
            perm_mean_between = np.mean(perm_between)
            if perm_mean_within_std > 0:
                null_disc_indices.append(perm_mean_between / perm_mean_within_std)
    
    null_disc_indices = np.array(null_disc_indices)
    disc_p_value = ((np.sum(null_disc_indices >= discrimination_index) + 1) 
                    / (len(null_disc_indices) + 1)) if len(null_disc_indices) > 0 else np.nan
    
    return {
        'habitat_stats': habitat_stats,
        'mean_within_variance': mean_within_variance,
        'mean_within_std': mean_within_std,
        'mean_between_distance': mean_between_distance,
        'discrimination_index': discrimination_index,
        'disc_p_value': disc_p_value,
        'n_habitats': len(habitat_stats)
    }

def save_figure(fig, filename: str, figures_folder: Path):
    """
    Save a matplotlib figure to the figures folder.
    
    Args:
        fig: matplotlib figure object
        filename: name for the saved file (without extension)
        figures_folder: Path to figures directory
    """
    try:
        # Ensure folder exists
        figures_folder.mkdir(parents=True, exist_ok=True)
        
        # Save as PNG with high DPI
        filepath = figures_folder / f"{filename}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        
        return True
    except Exception as e:
        st.warning(f"Could not save figure {filename}: {e}")
        return False

def calculate_map_scores(data: dict, all_results: dict, weights: dict = None) -> pd.DataFrame:
    """
    Calculate overall performance score for each individual map.
    Returns ranked dataframe with best map highlighted.

    Args:
        data: Dictionary of loaded overlay data
        all_results: Dictionary of validation results
        weights: Optional dict of metric weights. If None, uses recommended weights.
                Keys: 'procrustes', 'gradient', 'distance', 'prediction', 'habitat', 'clustering'
                Default weights prioritize Ellenberg correlation (primary goal).

    Returns:
        DataFrame with scores for each map, sorted by overall score
    """
    # Default weights based on research priorities:
    # PRIMARY: Ellenberg correlation (gradient + prediction)
    # SECONDARY: Habitat clustering
    # TERTIARY: Distance preservation (supports gradient interpretation)
    # DIAGNOSTIC ONLY: Procrustes (consistency check, not quality measure)
    # EXCLUDED: Habitat ANOVA (validates classification, not map)
    if weights is None:
        weights = {
            'gradient': 0.35,      # PRIMARY: Environmental gradient emergence
            'prediction': 0.25,    # PRIMARY: Predictive power for environment
            'clustering': 0.25,    # SECONDARY: Habitat spatial clustering
            'distance': 0.15,      # Supports gradient interpretation
            'procrustes': 0.00,    # Excluded: measures consensus, not quality
            'habitat': 0.00        # Excluded: validates data, not map
        }

    scores = []
    
    for name in data.keys():
        score = {
            'map': name,
            'method': data[name]['method']
        }
        
        # 1. Procrustes (mean similarity with other maps)
        # NOTE: Reported for diagnostics but excluded from overall score by default
        # (measures consensus, not quality — no ground truth to compare against)
        if 'procrustes_matrix' in all_results:
            sim_matrix = all_results['procrustes_matrix']
            if name in sim_matrix.index:
                row = sim_matrix.loc[name]
                score['procrustes'] = row[row.index != name].mean()
        
        # 2. Gradient (mean R² across indicators)
        if 'gradients' in all_results:
            grad_data = all_results['gradients']
            grad_row = grad_data[grad_data['overlay'] == name]
            if not grad_row.empty:
                r2_cols = [c for c in grad_row.columns if c.endswith('_R2')]
                if r2_cols:
                    r2_values = [grad_row[col].values[0] for col in r2_cols]
                    score['gradient'] = np.nanmean([v for v in r2_values if not np.isnan(v)])
        
        # 3. Distance preservation (global correlation)
        if 'distance_global' in all_results:
            if name in all_results['distance_global']:
                score['distance'] = all_results['distance_global'][name]['correlation']
        
        # 4. Prediction (mean R² across indicators)
        if 'predictions' in all_results:
            pred_data = all_results['predictions']
            pred_rows = pred_data[pred_data['overlay'] == name]
            if not pred_rows.empty:
                score['prediction'] = pred_rows['R2'].mean()
        
        # 5. Habitat differentiation (mean -log10(p-value) across indicators)
        # NOTE: This validates habitat classification, not the map itself
        # Excluded from overall score by default (can be included via weights)
        if 'habitat' in all_results:
            if name in all_results['habitat']:
                hab_data = all_results['habitat'][name]
                if hab_data:
                    p_values = [v['p_value'] for v in hab_data.values() if v['p_value'] > 0]
                    if p_values:
                        sig_scores = [-np.log10(max(p, 1e-10)) for p in p_values]
                        score['habitat'] = min(1.0, np.mean(sig_scores) / 10.0)

        # 6. Habitat spatial clustering (discrimination index, normalized)
        if 'spatial_clustering' in all_results:
            if name in all_results['spatial_clustering']:
                cluster_data = all_results['spatial_clustering'][name]
                if cluster_data:
                    # Normalize discrimination index to 0-1 scale
                    # Typical values range from 2-10, so divide by 10
                    score['clustering'] = min(1.0, cluster_data['discrimination_index'] / 10.0)
        
        # Calculate overall score (weighted mean of available metrics)
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in score and not np.isnan(score[metric]):
                weighted_sum += score[metric] * weight
                total_weight += weight

        if total_weight > 0:
            score['overall'] = weighted_sum / total_weight
        else:
            score['overall'] = 0.0
        
        scores.append(score)
    
    # Create dataframe and sort by overall score
    scores_df = pd.DataFrame(scores)
    scores_df = scores_df.sort_values('overall', ascending=False)
    
    return scores_df

# INTERFACE
###################################################################################

with st.expander("ℹ️ About This Analysis", expanded=False):
    st.markdown("""
    ## Comprehensive Validation Framework

    **Six independent validation approaches:**

    1. **Species Position Stability** - Procrustes R² measures structural consistency across maps
    2. **Environmental Gradient Emergence** - R² from polynomial regression of Ellenberg indicators
       - *Optional: Null model testing validates gradients are better than random*
    3. **Distance Preservation** - Mantel test correlation between 2D and ecological distances
       - *Includes spatial variation analysis (4×4 grid)*
       - *For MDS: Stress analysis and Shepard diagrams*
    4. **Predictive Accuracy** - Cross-validated k-NN prediction from position
       - *Caveat: May be inflated by spatial autocorrelation*
    5. **Habitat Environmental Differentiation** - ANOVA testing if habitat types differ environmentally
       - *Note: Validates habitat classification quality, not map structure*
    6. **Habitat Spatial Clustering** - Tests if habitats cluster spatially in the map

    **Individual Map Ranking:**

    All maps are scored across metrics and ranked to identify the optimal reference map.
    - **Weighted scoring (recommended):** Prioritizes Ellenberg correlation (primary goal)
    - **Equal weights (optional):** All metrics contribute equally

    **Scientific Rigor:**

    - Multiple lines of evidence (convergent validation)
    - Statistical significance testing (Mantel test, permutation tests)
    - Null model comparisons to rule out chance patterns
    - Spatial autocorrelation caveats clearly stated
    - Empirical method comparison (FR vs MDS)
    - Clear, data-driven recommendations

    **Key Innovation:**

    This validates that networks built ONLY from species co-occurrence
    (without trait data) successfully capture environmental gradients.
    This is the critical test for network-based ecological cartography.
    """)

# PATH CONFIGURATION
if 'overlay_map_path' not in st.session_state:
    st.error("⚠️ Run Home page first to initialize paths")
    st.stop()

overlay_path = Path(st.session_state['overlay_map_path'])
if not overlay_path.exists():
    st.error(f"⚠️ Overlay directory not found: {overlay_path}")
    st.stop()

# LOAD OVERLAYS
st.markdown("### 📁 Select overlay maps")

files = sorted([f.name for f in overlay_path.glob("*.db")])
if not files:
    st.error("❌ No overlay .db files found")
    st.stop()

st.info(f"Found {len(files)} overlay databases")

selected = st.multiselect(
    "Select overlay maps to compare:",
    files,
    default=files[:min(3, len(files))],
    help="Select at least 2 overlays for comparison"
)

if len(selected) < 2:
    st.warning("⚠️ Please select at least 2 overlay maps")
    st.stop()

st.info(f"📊 Selected {len(selected)} overlays")

# ANALYSIS OPTIONS
st.markdown("---")
st.markdown("### ⚙️ Analysis configuration")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Core Analyses:**")
    do_procrustes = st.checkbox("Species position stability", True)
    do_gradients = st.checkbox("Environmental gradients", True)
    do_distance = st.checkbox("Distance preservation", True)

with col2:
    st.markdown("**Advanced Analyses:**")
    do_prediction = st.checkbox("Predictive validation", True)
    do_habitat = st.checkbox("Habitat differentiation", True)
    do_null_models = st.checkbox("Null model testing", True,
                                  help="Test if gradients are better than random (permutation test)")

with col3:
    st.markdown("**Settings:**")

    scoring_method = st.radio(
        "Overall scoring:",
        ["Weighted (Recommended)", "Equal weights"],
        help="Weighted scoring prioritizes Ellenberg correlation (35%), prediction (25%), and habitat clustering (25%)"
    )

    if do_gradients or do_prediction:
        test_indicators = st.multiselect(
            "Ellenberg indicators:",
            ['L', 'M', 'R', 'N'],
            default=['L', 'M', 'R', 'N'],
            help="L=Light, M=Moisture, R=Reaction (pH), N=Nitrogen"
        )
    
    if do_prediction:
        n_samples_pred = st.slider(
            "Prediction samples:",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="More = accurate but slower"
        )
        est_time = n_samples_pred * len(selected) * len(test_indicators) * 0.003
        st.caption(f"⏱️ Est. time: ~{est_time:.1f} min")
    
    if do_habitat:
        habitat_column = st.selectbox(
            "Habitat classification:",
            ["habitat_type", "major_type"],
            help="Choose habitat classification to test"
        )
    
    # Figure saving option
    save_figures = st.checkbox(
        "💾 Save all figures",
        value=False,
        help="Save all plots to /figures folder"
    )

st.markdown("---")

# RUN ANALYSIS
if st.button("🔬 Run Comprehensive Validation", type="primary", use_container_width=True):

    # Create figures folder if saving is enabled
    saved_figures = []  # Initialize for all cases
    if save_figures:
        figures_folder = Path(st.session_state.get('figures_path', '.'))
        figures_folder.mkdir(parents=True, exist_ok=True)
        st.info(f"💾 Figures will be saved to: {figures_folder}")

    # Storage for all results
    all_results = {}
    
    # LOAD DATA
    st.markdown("## 📊 Loading data")
    
    data = {}
    progress_bar = st.progress(0, text="Loading overlays...")
    
    for i, filename in enumerate(selected):
        progress_bar.progress((i + 1) / len(selected), text=f"Loading {filename}...")
        
        overlay_data = load_overlay_db(overlay_path / filename)
        if overlay_data:
            metadata = overlay_data.get('metadata')
            method = detect_method(filename, metadata)
            
            name = filename.replace('.db', '')
            data[name] = {
                **overlay_data,
                'method': method,
                'filename': filename
            }
    
    progress_bar.empty()
    
    if len(data) < 2:
        st.error("❌ Could not load at least 2 valid overlays")
        st.stop()
    
    st.success(f"✅ Loaded {len(data)} overlay maps")
    
    # Display method distribution
    methods = {}
    for name, d in data.items():
        method = d['method']
        if method not in methods:
            methods[method] = []
        methods[method].append(name)
    
    cols = st.columns(len(methods))
    for i, (method, names) in enumerate(methods.items()):
        with cols[i]:
            st.metric(method, len(names))
    
    # 1. SPECIES POSITION STABILITY
    if do_procrustes:
        st.markdown("---")
        st.markdown("## 1️⃣ Species Position Stability")
        
        with st.expander("ℹ️ Method Explanation", expanded=False):
            st.markdown("""
            **Procrustes Analysis** measures structural similarity by optimally aligning 
            two coordinate sets (rotation, translation, scaling) and calculating remaining variance.
            R² = 1 − disparity, where disparity is measured after standardizing both 
            configurations to unit Frobenius norm.
            
            **High R² (>0.85):** Species maintain very similar relative positions  
            **Moderate R² (0.60-0.85):** Some variation in positioning  
            **Low R² (<0.60):** Substantial differences in structure
            
            **Important:** High similarity between FR and MDS is expected and desirable - it shows
            both capture the same ecological structure. They differ in which environmental 
            gradients align with coordinate axes, not in fundamental species relationships.
            """)
        
        common_species = None
        for d in data.values():
            if d.get('taxa') is not None:
                species_set = set(d['taxa']['node_key'])
                common_species = (species_set if common_species is None 
                                else common_species & species_set)
        
        if not common_species or len(common_species) < 10:
            st.error(f"❌ Too few common species")
        else:
            common_species = sorted(list(common_species))
            st.info(f"📊 Analyzing {len(common_species):,} common species")
            
            names = list(data.keys())
            n_overlays = len(names)
            similarity_matrix = np.ones((n_overlays, n_overlays))
            
            with st.spinner("Calculating Procrustes similarities..."):
                for i in range(n_overlays):
                    for j in range(i + 1, n_overlays):
                        taxa_i = data[names[i]]['taxa']
                        taxa_i = taxa_i[taxa_i['node_key'].isin(common_species)].sort_values('node_key')
                        
                        taxa_j = data[names[j]]['taxa']
                        taxa_j = taxa_j[taxa_j['node_key'].isin(common_species)].sort_values('node_key')
                        
                        coords_i = taxa_i[['xcoor', 'ycoor']].values
                        coords_j = taxa_j[['xcoor', 'ycoor']].values
                        
                        r2 = calc_procrustes_r2(coords_i, coords_j)
                        similarity_matrix[i, j] = r2
                        similarity_matrix[j, i] = r2
            
            sim_df = pd.DataFrame(similarity_matrix, index=names, columns=names)
            off_diagonal = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            
            all_results['procrustes_matrix'] = sim_df
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean R²", f"{off_diagonal.mean():.3f}")
            with col2:
                st.metric("Std Dev", f"{off_diagonal.std():.3f}")
            with col3:
                st.metric("Min R²", f"{off_diagonal.min():.3f}")
            with col4:
                st.metric("Max R²", f"{off_diagonal.max():.3f}")
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                sim_df,
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',  # Colorblind-friendly: red=low, blue=high
                vmin=0.4,
                vmax=1.0,
                center=0.7,
                square=True,
                ax=ax,
                cbar_kws={'label': 'Procrustes R²'},
                linewidths=0.5
            )
            ax.set_title('Species Position Similarity (Colorblind-friendly)', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            if save_figures:
                save_figure(fig, "01_procrustes_similarity_matrix", figures_folder)
                saved_figures.append("Procrustes similarity matrix")
            
            st.pyplot(fig)
            plt.close()

            # MDS STRESS ANALYSIS (for MDS maps only)
            mds_maps = [name for name, d in data.items() if d['method'] == 'MDS']
            if mds_maps:
                st.markdown("---")
                st.markdown("### 📐 MDS stress analysis")
                st.caption("**Validates distance preservation for MDS method**")

                with st.expander("ℹ️ Method Explanation", expanded=False):
                    st.markdown("""
                    **Stress** measures how well MDS preserves original ecological distances in 2D space.

                    **Formula:** Stress = √(Σ(d_2D - d_original)² / Σd_original²)

                    **Interpretation:**
                    - **Stress < 0.05:** Excellent fit (2D representation almost perfect)
                    - **Stress < 0.10:** Good fit (2D representation adequate)
                    - **Stress < 0.20:** Acceptable fit (some distortion)
                    - **Stress > 0.20:** Poor fit (high distortion)

                    **Shepard Diagram** shows original ecological distances vs 2D distances.
                    Points close to diagonal = good preservation.

                    **Note:** FR (force-directed) layouts don't optimize distance preservation,
                    so stress values are not meaningful for FR maps. Only MDS is evaluated here.
                    """)

                # We need to calculate ecological distances from species data
                # This requires species-level Ellenberg values
                st.info(f"Found {len(mds_maps)} MDS map(s) for stress analysis")

                # For now, calculate stress based on plot-level distances
                stress_results = {}

                with st.spinner("Calculating MDS stress..."):
                    for name in mds_maps:
                        if data[name].get('plot_id') is None:
                            continue

                        plot_id = data[name]['plot_id']
                        plots = plot_id[['xcoor', 'ycoor']].dropna()

                        if len(plots) < 20:
                            continue

                        # Sample for computational efficiency
                        if len(plots) > 300:
                            plots = plots.sample(n=300, random_state=42)

                        plot_indices = plots.index.tolist()
                        coords_2d = plots[['xcoor', 'ycoor']].values

                        # Calculate ecological distances
                        indicators = ['L', 'M', 'R', 'N']
                        available_indicators = [ind for ind in indicators if ind in plot_id.columns]

                        if len(available_indicators) >= 2:
                            plot_features = plot_id.loc[plot_indices, available_indicators].dropna()
                            valid_stress_indices = plot_features.index.tolist()
                            
                            if len(valid_stress_indices) < 20:
                                continue
                            
                            coords_2d = plot_id.loc[valid_stress_indices, ['xcoor', 'ycoor']].values
                            ecological_dists = pdist(plot_features.values, metric='euclidean')

                            # Normalize
                            max_dist = ecological_dists.max()
                            if max_dist > 0:
                                ecological_dists = ecological_dists / max_dist

                            # Calculate stress
                            stress_res = calc_mds_stress(coords_2d, ecological_dists)
                            stress_results[name] = stress_res

                if stress_results:
                    # Summary table
                    stress_summary = []
                    for name, res in stress_results.items():
                        stress_summary.append({
                            'Map': name,
                            'Stress': res['stress'],
                            'R²': res['r2'],
                            'Correlation': res['correlation'],
                            'p-value': res['p_value']
                        })

                    stress_df = pd.DataFrame(stress_summary)

                    st.dataframe(
                        stress_df.style.format({
                            'Stress': '{:.4f}',
                            'R²': '{:.3f}',
                            'Correlation': '{:.3f}',
                            'p-value': '{:.2e}'
                        }).background_gradient(subset=['Stress'], cmap='RdYlGn_r', vmin=0, vmax=0.2)
                         .background_gradient(subset=['R²'], cmap='RdYlGn', vmin=0, vmax=1),
                        use_container_width=True,
                        hide_index=True
                    )

                    # Interpretation
                    best_stress = stress_df.loc[stress_df['Stress'].idxmin()]
                    if best_stress['Stress'] < 0.10:
                        quality = "Excellent"
                        emoji = "✅"
                    elif best_stress['Stress'] < 0.20:
                        quality = "Good"
                        emoji = "✓"
                    else:
                        quality = "Acceptable"
                        emoji = "⚠️"

                    st.success(f"""
                    {emoji} **Best MDS Stress:** {best_stress['Map']}
                    (Stress = {best_stress['Stress']:.4f}, {quality} fit)

                    MDS preserves {best_stress['R²']*100:.1f}% of variance in ecological distances.
                    """)

                    # Shepard diagrams
                    st.markdown("### Shepard diagrams")
                    st.caption("**Original ecological distances vs 2D distances**")

                    n_maps = len(stress_results)
                    n_cols = min(3, n_maps)
                    n_rows = (n_maps + n_cols - 1) // n_cols

                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                    if n_maps == 1:
                        axes = np.array([axes])
                    axes = axes.flatten()

                    for idx, (name, res) in enumerate(stress_results.items()):
                        ax = axes[idx]

                        # Subsample points for visualization (too many = too dense)
                        n_points = len(res['original_dists'])
                        if n_points > 5000:
                            sample_idx = np.random.choice(n_points, 5000, replace=False)
                            x = res['original_dists'][sample_idx]
                            y = res['dists_2d'][sample_idx]
                        else:
                            x = res['original_dists']
                            y = res['dists_2d']

                        # Scatter plot with transparency
                        ax.scatter(x, y, alpha=0.3, s=1, color='steelblue')

                        # Diagonal line (perfect preservation)
                        max_val = max(x.max(), y.max())
                        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2,
                               label='Perfect preservation')

                        # Regression line
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        ax.plot(x, p(x), 'g-', linewidth=2, alpha=0.7,
                               label=f'Fit (R²={res["r2"]:.3f})')

                        ax.set_xlabel('Ecological Distance', fontsize=10)
                        ax.set_ylabel('2D Distance', fontsize=10)
                        ax.set_title(f'{name}\nStress = {res["stress"]:.4f}',
                                   fontsize=11, fontweight='bold')
                        ax.legend(fontsize=8)
                        ax.grid(alpha=0.3)
                        ax.set_aspect('equal', adjustable='box')

                    # Hide extra subplots
                    for idx in range(n_maps, len(axes)):
                        axes[idx].axis('off')

                    plt.tight_layout()

                    if save_figures:
                        save_figure(fig, "01b_mds_stress_shepard", figures_folder)
                        saved_figures.append("MDS stress Shepard diagrams")

                    st.pyplot(fig)
                    plt.close()

                    st.info("""
                    **Interpretation:** Points close to the red diagonal line indicate good distance
                    preservation. Scatter around the line = distortion. Lower stress = better fit.
                    """)

                else:
                    st.warning("⚠️ Could not calculate stress for MDS maps")

    # 2. ENVIRONMENTAL GRADIENTS
    if do_gradients and test_indicators:
        st.markdown("---")
        st.markdown("## 2️⃣ Environmental Gradient Analysis")
        
        with st.expander("ℹ️ Method Explanation", expanded=False):
            st.markdown("""
            **Polynomial Regression** tests how well 2D position predicts Ellenberg indicators.
            
            **Model:** Indicator = f(x, y, x×y, x², y²)
            
            **Input:** Plot coordinates and mean Ellenberg values (averaged across species)
            
            **R² Interpretation:**
            - **R² > 0.7:** Strong gradient (position strongly predicts environment)
            - **R² = 0.5-0.7:** Good gradient (clear environmental structure)
            - **R² = 0.3-0.5:** Moderate gradient (some spatial pattern)
            - **R² < 0.3:** Weak gradient (little spatial structure)
            """)
        
        gradient_results = {}
        summary_rows = []
        
        with st.spinner("Calculating gradients..."):
            for name, d in data.items():
                if d.get('plot_id') is None:
                    continue
                
                plot_id = d['plot_id']
                row = {'overlay': name, 'method': d['method']}
                gradient_results[name] = {}
                
                for indicator in test_indicators:
                    if indicator in plot_id.columns:
                        grad = calc_gradient(plot_id, indicator)
                        if grad:
                            gradient_results[name][indicator] = grad
                            row[f'{indicator}_R2'] = grad['r2']
                            row[f'{indicator}_adjR2'] = grad['adj_r2']
                
                if len(gradient_results[name]) >= 2:
                    angles = calc_all_angles(gradient_results[name])
                    for angle_name, angle_value in angles.items():
                        row[angle_name] = angle_value
                
                summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        all_results['gradients'] = summary_df

        # Store gradient results in session state for null model testing
        st.session_state['gradient_results'] = gradient_results
        st.session_state['test_indicators'] = test_indicators
        st.session_state['validation_data'] = data
        st.session_state['do_null_models'] = do_null_models
        st.session_state['save_figures'] = save_figures
        if save_figures:
            st.session_state['figures_folder'] = figures_folder
            st.session_state['saved_figures'] = saved_figures

        if not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Visualization
            r2_cols = [c for c in summary_df.columns if c.endswith('_R2')]
            if r2_cols:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                x = np.arange(len(summary_df))
                width = 0.8 / len(r2_cols)
                colors = {'M': '#3498db', 'L': '#2ecc71', 'N': '#f39c12', 'R': '#e74c3c'}
                
                for i, indicator in enumerate(test_indicators):
                    col_name = f'{indicator}_R2'
                    if col_name in summary_df.columns:
                        values = summary_df[col_name].fillna(0)
                        offset = width * (i - len(test_indicators)/2 + 0.5)
                        ax.bar(
                            x + offset,
                            values,
                            width,
                            label=indicator,
                            color=colors.get(indicator, 'gray'),
                            edgecolor='black',
                            linewidth=0.5
                        )
                
                ax.set_xlabel('Overlay Map', fontsize=11)
                ax.set_ylabel('R² (Variance Explained)', fontsize=11)
                ax.set_title('Environmental Gradient Strength', fontsize=13, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [f"{row['overlay'][:15]}\n({row['method']})" 
                     for _, row in summary_df.iterrows()],
                    rotation=0,
                    ha='center',
                    fontsize=8
                )
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                ax.set_ylim(0, 1)
                plt.tight_layout()
                
                if save_figures:
                    save_figure(fig, "02_environmental_gradients", figures_folder)
                    saved_figures.append("Environmental gradients")
                
                st.pyplot(fig)
                plt.close()

            # NULL MODEL TESTING
            if do_null_models and gradient_results:
                st.markdown("---")
                st.markdown("### 🎲 Null model testing")
                st.caption("**Tests if observed gradients are significantly better than random**")

                with st.expander("ℹ️ Method Explanation", expanded=False):
                    st.markdown("""
                    **Permutation Test** validates that observed R² values are not due to chance.

                    **Process:**
                    1. Permute indicator values 999 times (breaks gradient structure)
                    2. Recalculate R² for each permutation
                    3. Compare observed R² to null distribution
                    4. p-value = proportion of null R² ≥ observed R²

                    **Interpretation:**
                    - **p < 0.001:** Gradient highly significant (much better than random)
                    - **Observed R² >> null mean:** Strong evidence for real gradient

                    This is a critical validation that co-occurrence networks capture real
                    environmental structure, not spurious patterns.
                    """)

                # Run null model testing for first map automatically (avoids widget interaction issues)
                st.info("💡 **Running null model testing for the first map** (avoids page reset issues)")
                null_test_map = list(gradient_results.keys())[0]
                st.write(f"**Map being tested:** `{null_test_map}`")

                if null_test_map:
                    with st.spinner(f"Running permutation test for {null_test_map} (999 permutations)..."):
                        null_results = {}

                        for indicator in test_indicators:
                            if indicator in data[null_test_map]['plot_id'].columns:
                                null_res = calc_gradient_null_model(
                                    data[null_test_map]['plot_id'],
                                    indicator,
                                    n_permutations=999
                                )
                                if null_res:
                                    null_results[indicator] = null_res

                        if null_results:
                            # Summary table
                            null_summary = []
                            for ind, res in null_results.items():
                                null_summary.append({
                                    'Indicator': ind,
                                    'Observed R²': res['r2_observed'],
                                    'Null Mean R²': res['r2_null_mean'],
                                    'Null 95th %ile': res['r2_null_95'],
                                    'p-value': res['p_value']
                                })

                            null_df = pd.DataFrame(null_summary)

                            st.dataframe(
                                null_df.style.format({
                                    'Observed R²': '{:.3f}',
                                    'Null Mean R²': '{:.3f}',
                                    'Null 95th %ile': '{:.3f}',
                                    'p-value': '{:.4f}'
                                }).background_gradient(subset=['p-value'], cmap='RdYlGn_r', vmin=0, vmax=0.05),
                                use_container_width=True,
                                hide_index=True
                            )

                            # Check significance
                            all_sig = all(res['p_value'] < 0.05 for res in null_results.values())
                            if all_sig:
                                st.success("✅ **All gradients are significantly better than random** (p < 0.05)")
                            else:
                                st.warning("⚠️ Some gradients not significantly different from random")

                            # Visualization: Null distributions
                            n_indicators = len(null_results)
                            fig, axes = plt.subplots(1, n_indicators, figsize=(5*n_indicators, 4))
                            if n_indicators == 1:
                                axes = [axes]

                            colors = {'L': '#2ecc71', 'F': '#3498db', 'R': '#e74c3c', 'N': '#f39c12',
                                     'M': '#3498db'}

                            for ax, (ind, res) in zip(axes, null_results.items()):
                                # Histogram of null distribution
                                ax.hist(res['null_distribution'], bins=30, alpha=0.6,
                                       color='gray', edgecolor='black', label='Null distribution')

                                # Observed value
                                ax.axvline(res['r2_observed'], color=colors.get(ind, 'red'),
                                         linewidth=3, label=f'Observed R²', linestyle='--')

                                # 95th percentile
                                ax.axvline(res['r2_null_95'], color='black',
                                         linewidth=2, label='Null 95th %ile', linestyle=':')

                                ax.set_xlabel('R²', fontsize=11)
                                ax.set_ylabel('Frequency', fontsize=11)
                                ax.set_title(f'{ind} Indicator\n(p = {res["p_value"]:.4f})',
                                           fontsize=12, fontweight='bold')
                                ax.legend(fontsize=9)
                                ax.grid(alpha=0.3)

                            plt.suptitle(f'Null Model Test: {null_test_map}',
                                       fontsize=14, fontweight='bold', y=1.02)
                            plt.tight_layout()

                            if save_figures:
                                save_figure(fig, f"02b_null_model_gradients_{null_test_map.replace('/', '_')}", figures_folder)
                                saved_figures.append(f"Null model test ({null_test_map})")

                            st.pyplot(fig)
                            plt.close()

                            st.info("""
                            **Interpretation:** If observed R² (dashed line) is far to the right of the
                            null distribution (gray histogram), the gradient is real and not due to chance.
                            p < 0.05 indicates statistical significance.
                            """)
                        else:
                            st.error("❌ Could not calculate null models")

    # 3. DISTANCE PRESERVATION
    if do_distance:
        st.markdown("---")
        st.markdown("## 3️⃣ Distance Preservation Analysis")
        
        with st.expander("ℹ️ Method Explanation", expanded=False):
            st.markdown("""
            **Distance Preservation (Mantel Test)** tests if 2D Euclidean distances correlate with
            ecological dissimilarities (based on Ellenberg indicators).

            **Method:** Mantel test with 999 permutations (standard in ecology for distance matrices).
            This is more robust than simple Pearson correlation.

            **Interpretation:**
            - **High correlation (r > 0.7):** Plots close together in 2D space have similar
              Ellenberg values. Spatial proximity = ecological similarity.
            - **Low correlation (r < 0.3):** Distance doesn't reflect ecological similarity.
              Plots can be close but ecologically different.

            **Expected:** MDS should show higher r (MDS explicitly optimizes distance preservation)
            while FR focuses on clustering.

            **Spatial Variation (4×4 grid):** Tests if preservation varies across landscape.
            Dense areas with high turnover may show different patterns than sparse isolated regions.

            **Color interpretation:** Yellow/bright = stronger preservation, Purple/dark = weaker preservation
            """)
        
        st.markdown("### Global distance preservation")
        
        distance_global = {}
        distance_spatial = {}
        dist_results = []
        
        with st.spinner("Analyzing distance preservation..."):
            for name, d in data.items():
                if d.get('plot_id') is None:
                    continue
                
                # Global (using Mantel test)
                result_global = calc_distance_preservation_global(d['plot_id'], use_mantel=True)
                if result_global:
                    distance_global[name] = result_global
                    dist_results.append({
                        'overlay': name,
                        'method': d['method'],
                        'correlation': result_global['correlation'],
                        'p_value': result_global['p_value'],
                        'n_plots': result_global['n_plots'],
                        'test_method': result_global.get('method', 'Mantel test')
                    })
                
                # Spatial
                result_spatial = calc_distance_preservation_spatial(d['plot_id'])
                if result_spatial:
                    distance_spatial[name] = result_spatial
        
        all_results['distance_global'] = distance_global
        all_results['distance_spatial'] = distance_spatial
        
        if dist_results:
            dist_df = pd.DataFrame(dist_results)
            
            st.dataframe(
                dist_df.style.format({
                    'correlation': '{:.3f}',
                    'p_value': '{:.2e}'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Highlight best
            best_idx = dist_df['correlation'].idxmax()
            best_map = dist_df.loc[best_idx]
            
            st.success(f"""
            **Highest Distance Preservation:** {best_map['overlay']}
            (r = {best_map['correlation']:.3f}, p = {best_map['p_value']:.3e}, {best_map['method']} layout)

            This map best preserves ecological relationships in 2D space (Mantel test).
            """)
            
            # Spatial variation heatmaps (COLORBLIND-SAFE)
            if distance_spatial:
                st.markdown("### Spatial variation (4×4 grid)")
                st.caption("**Bright (yellow) = stronger preservation | Dark (purple) = weaker preservation**")
                
                n_maps = len(distance_spatial)
                n_cols = min(3, n_maps)
                n_rows = (n_maps + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                if n_maps == 1:
                    axes = np.array([axes])
                axes = axes.flatten()
                
                for idx, (name, result) in enumerate(distance_spatial.items()):
                    ax = axes[idx]
                    matrix = result['regional_matrix']
                    
                    # Use viridis colormap (colorblind-safe)
                    im = ax.imshow(matrix, cmap='viridis', vmin=-0.2, vmax=1.0, aspect='auto')
                    
                    # Annotate cells
                    for i in range(4):
                        for j in range(4):
                            if not np.isnan(matrix[i, j]):
                                # Use white text for dark background, black for bright
                                text_color = 'white' if matrix[i, j] < 0.5 else 'black'
                                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                                             ha="center", va="center",
                                             color=text_color,
                                             fontsize=9, fontweight='bold')
                    
                    ax.set_title(f'{name[:20]}\n({data[name]["method"]}, mean={result["mean_correlation"]:.2f})',
                               fontsize=10, fontweight='bold')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlabel('←West    East→', fontsize=8)
                    ax.set_ylabel('←South    North→', fontsize=8)
                    
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Correlation (r)')
                
                # Hide extra subplots
                for idx in range(n_maps, len(axes)):
                    axes[idx].axis('off')
                
                plt.tight_layout()
                
                if save_figures:
                    save_figure(fig, "03_distance_preservation_spatial", figures_folder)
                    saved_figures.append("Distance preservation spatial variation")
                
                st.pyplot(fig)
                plt.close()
                
                st.info("""
                **Interpretation Guide:**
                - **High values (bright yellow, r > 0.7):** Plots close together are ecologically similar
                - **Low values (dark purple, r < 0.3):** Spatial distance doesn't reflect ecology
                - **Variation across regions** reflects differences in species turnover rates
                """)
    
    # 4. PREDICTIVE VALIDATION
    if do_prediction:
        st.markdown("---")
        st.markdown("## 4️⃣ Predictive Validation")
        
        with st.expander("ℹ️ Method Explanation", expanded=False):
            st.markdown("""
            **k-Nearest Neighbor Cross-Validation** tests if we can predict environment from position.

            **Process:**
            1. For each plot: Remove it, find k=5 nearest plots, predict as their mean
            2. Calculate R², RMSE, correlation

            **R² Interpretation:**
            - **R² > 0.7:** Strong - position highly informative
            - **R² = 0.5-0.7:** Good - useful predictive power
            - **R² < 0.5:** Weak - limited predictive value

            **Per-Map Performance:** Shows which map encodes most environmental information.

            **⚠️ Important Caveat - Spatial Autocorrelation:**
            Ecological data typically show spatial autocorrelation (nearby plots are more similar).
            This means k-NN performance may be partially inflated by finding spatially close plots
            that are inherently similar, not just because the map captures environmental structure.
            However, if the map positions plots based on co-occurrence (not geography), then high
            k-NN accuracy validates that ecological similarity (co-occurrence) predicts environment.
            """)
            st.warning("⚠️ R² values may be inflated due to spatial autocorrelation in ecological data")
        
        st.warning(f"⏱️ This analysis may take several minutes (testing {n_samples_pred} plots per overlay)...")
        
        pred_indicators = test_indicators
        prediction_results = []
        
        progress_total = len(data) * len(pred_indicators)
        progress_current = 0
        progress_bar = st.progress(0, text="Running predictions...")
        
        for name, d in data.items():
            if d.get('plot_id') is None:
                continue
            
            for indicator in pred_indicators:
                if indicator in d['plot_id'].columns:
                    progress_current += 1
                    progress_bar.progress(
                        progress_current / progress_total,
                        text=f"Predicting {indicator} for {name[:20]}..."
                    )
                    
                    result = cross_validate_predictions(
                        d['plot_id'], 
                        indicator, 
                        k=5, 
                        n_samples=n_samples_pred
                    )
                    
                    if result:
                        interpretation = interpret_prediction(
                            result['r2'], 
                            result['rmse'], 
                            indicator
                        )
                        
                        prediction_results.append({
                            'overlay': name,
                            'method': d['method'],
                            'indicator': indicator,
                            'R2': result['r2'],
                            'RMSE': result['rmse'],
                            'correlation': result['correlation'],
                            'n_plots': result['n'],
                            'interpretation': interpretation
                        })
        
        progress_bar.empty()
        
        if prediction_results:
            pred_df = pd.DataFrame(prediction_results)
            all_results['predictions'] = pred_df
            
            # KEY CHANGE: Show per-MAP performance
            st.markdown("### Per-map predictive performance")
            st.caption("**Which map encodes the most environmental information?**")
            
            map_performance = pred_df.groupby(['overlay', 'method']).agg({
                'R2': 'mean',
                'RMSE': 'mean'
            }).round(3)
            map_performance = map_performance.sort_values('R2', ascending=False)
            map_performance = map_performance.reset_index()
            
            st.dataframe(
                map_performance.style.background_gradient(subset=['R2'], cmap='RdYlGn', vmin=0, vmax=1),
                use_container_width=True,
                hide_index=True
            )
            
            best_pred_map = map_performance.iloc[0]
            st.success(f"""
            **Best Predictive Performance:** {best_pred_map['overlay']} 
            (Mean R² = {best_pred_map['R2']:.3f}, {best_pred_map['method']} method)
            
            This map encodes the most environmental information in its spatial structure.
            """)
            
            # Detail by indicator
            st.markdown("### Performance by indicator")
            st.caption("*Average across all maps*")
            
            indicator_summary = pred_df.groupby('indicator').agg({
                'R2': ['mean', 'std'],
                'RMSE': ['mean', 'std']
            }).round(3)
            st.dataframe(indicator_summary, use_container_width=True)
            
            # Full detailed results
            with st.expander("📋 Detailed Results (All Maps × All Indicators)"):
                display_df = pred_df[['overlay', 'method', 'indicator', 'R2', 'RMSE', 'interpretation']].copy()
                st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # 5. HABITAT ENVIRONMENTAL DIFFERENTIATION
    if do_habitat:
        st.markdown("---")
        st.markdown("## 5️⃣ Habitat Environmental Differentiation")
        
        with st.expander("ℹ️ Method Explanation", expanded=False):
            st.markdown("""
            **ANOVA Tests** determine if field-classified habitat types differ in 
            environmental conditions (Ellenberg values).
            
            **This tests:** Do your habitat classifications (habitat_type or major_type) 
            correspond to real environmental differences?
            
            **Process:**
            1. Group plots by habitat classification
            2. Calculate mean Ellenberg values per habitat
            3. ANOVA: Do habitats differ significantly?
            
            **Interpretation:**
            - **p < 0.001:** Highly significant - habitats are environmentally distinct
            - **F-statistic:** Higher = stronger differentiation
            
            This validates that habitat classifications reflect environmental reality.
            """)
        
        test_indicators_hab = test_indicators if do_gradients else ['M', 'L', 'N', 'R']
        habitat_results = {}
        spatial_clustering_results = {}
        
        with st.spinner("Testing habitat differentiation and spatial clustering..."):
            for name, d in data.items():
                if d.get('plot_id') is None:
                    continue
                
                # Environmental differentiation (ANOVA)
                result = analyze_habitat_differentiation(
                    d['plot_id'],
                    habitat_column,
                    test_indicators_hab
                )
                
                if result:
                    habitat_results[name] = result
                
                # Spatial clustering
                cluster_result = analyze_habitat_spatial_clustering(
                    d['plot_id'],
                    habitat_column
                )
                
                if cluster_result:
                    spatial_clustering_results[name] = cluster_result
        
        all_results['habitat'] = habitat_results
        all_results['spatial_clustering'] = spatial_clustering_results
        
        if habitat_results:
            # Summary table
            summary_rows = []
            for name, results in habitat_results.items():
                row = {'overlay': name, 'method': data[name]['method']}
                for indicator, result in results.items():
                    row[f'{indicator}_F'] = result['f_stat']
                    row[f'{indicator}_p'] = result['p_value']
                    row[f'{indicator}_n_habitats'] = result['n_habitats']
                summary_rows.append(row)
            
            summary_df_hab = pd.DataFrame(summary_rows)
            
            st.markdown(f"### Statistical Tests: {habitat_column}")
            st.dataframe(
                summary_df_hab.style.format({
                    c: '{:.2f}' if c.endswith('_F') else '{:.2e}' if c.endswith('_p') else '{:.0f}'
                    for c in summary_df_hab.columns if c.endswith(('_F', '_p', '_n_habitats'))
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Let user select which map to display details for
            st.markdown("### Habitat environmental profiles")
            
            available_maps = list(habitat_results.keys())
            selected_map_display = st.selectbox(
                "Select map to display detailed profiles:",
                available_maps,
                help="Choose which map's habitat profiles to view"
            )
            
            if selected_map_display:
                st.markdown(f"#### Map: {selected_map_display}")
                
                for indicator in test_indicators_hab:
                    if indicator in habitat_results[selected_map_display]:
                        result = habitat_results[selected_map_display][indicator]
                        
                        st.markdown(f"**{indicator}** (F={result['f_stat']:.2f}, p={result['p_value']:.2e})")
                        
                        means_df = result['habitat_means'].reset_index()
                        means_df.columns = ['Habitat', 'Mean', 'Std Dev', 'n']
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.dataframe(
                                means_df.style.format({
                                    'Mean': '{:.2f}',
                                    'Std Dev': '{:.2f}'
                                }),
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        with col2:
                            if result['p_value'] < 0.001:
                                st.success("✅ p < 0.001")
                            elif result['p_value'] < 0.05:
                                st.success("✅ p < 0.05")
                            else:
                                st.info("ℹ️ n.s.")
        
        # SPATIAL CLUSTERING ANALYSIS
        if spatial_clustering_results:
            st.markdown("---")
            st.markdown("### Habitat spatial clustering")
            
            with st.expander("ℹ️ What This Tests", expanded=False):
                st.markdown("""
                **Spatial clustering analysis** tests whether plots of the same habitat type 
                cluster together in the reference map, indicating good discriminative power.
                
                **Metrics:**
                - **Within-habitat variance:** Spatial spread of plots around habitat centroid  
                  (Lower = tighter clustering)
                - **Between-habitat distance:** Separation of habitat centroids  
                  (Higher = better separation)
                - **Discrimination index:** Ratio of between/within distances  
                  (Higher = better habitat discrimination)
                
                **Interpretation:**
                Maps with higher discrimination indices better separate habitat types spatially,
                indicating that the map structure reflects habitat classifications.
                
                **Note:** Some habitats are naturally more heterogeneous than others, so 
                interpret variance in ecological context.
                """)
            
            # Summary table across maps
            st.markdown("#### Discrimination performance by map")
            
            cluster_summary = []
            for name, result in spatial_clustering_results.items():
                cluster_summary.append({
                    'overlay': name,
                    'method': data[name]['method'],
                    'mean_within_variance': result['mean_within_variance'],
                    'mean_between_distance': result['mean_between_distance'],
                    'discrimination_index': result['discrimination_index'],
                    'disc_p_value': result.get('disc_p_value', np.nan),
                    'n_habitats': result['n_habitats']
                })
            
            cluster_df = pd.DataFrame(cluster_summary)
            cluster_df = cluster_df.sort_values('discrimination_index', ascending=False)
            
            st.dataframe(
                cluster_df.style.format({
                    'mean_within_variance': '{:.4f}',
                    'mean_between_distance': '{:.3f}',
                    'discrimination_index': '{:.2f}',
                    'disc_p_value': '{:.3f}'
                }).background_gradient(subset=['discrimination_index'], cmap='RdYlGn', vmin=0, vmax=10),
                use_container_width=True,
                hide_index=True
            )
            
            # Highlight best
            best_clustering = cluster_df.iloc[0]
            st.success(f"""
            **Best Habitat Discrimination:** {best_clustering['overlay']} 
            (Discrimination Index = {best_clustering['discrimination_index']:.2f})
            
            This map shows the tightest habitat clustering and best spatial separation.
            """)
            
            # Detailed habitat statistics for selected map
            st.markdown("#### Per-habitat spatial statistics")
            
            selected_map_cluster = st.selectbox(
                "Select map to display habitat clustering details:",
                list(spatial_clustering_results.keys()),
                key='cluster_map_select',
                help="View spatial distribution of each habitat type"
            )
            
            if selected_map_cluster:
                cluster_data = spatial_clustering_results[selected_map_cluster]
                
                st.markdown(f"**Map: {selected_map_cluster}**")
                
                # Create detailed table
                habitat_detail = []
                for habitat, stats in cluster_data['habitat_stats'].items():
                    habitat_detail.append({
                        'Habitat': str(habitat)[:30],
                        'n plots': stats['n'],
                        'Centroid X': stats['centroid_x'],
                        'Centroid Y': stats['centroid_y'],
                        'Variance': stats['variance'],
                        'Std Dev': stats['std_dev']
                    })
                
                habitat_detail_df = pd.DataFrame(habitat_detail)
                habitat_detail_df = habitat_detail_df.sort_values('Variance')
                
                st.dataframe(
                    habitat_detail_df.style.format({
                        'Centroid X': '{:.3f}',
                        'Centroid Y': '{:.3f}',
                        'Variance': '{:.4f}',
                        'Std Dev': '{:.3f}'
                    }).background_gradient(subset=['Variance'], cmap='RdYlGn_r', vmin=0),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.caption("""
                **Interpretation:** Lower variance indicates tighter spatial clustering. 
                Habitats with low variance are well-discriminated in this map.
                """)
                
                # Always show bar chart (most informative)
                st.markdown("#### Habitat clustering strength")
                
                n_habitats_total = len(cluster_data['habitat_stats'])
                
                fig, ax = plt.subplots(figsize=(12, max(6, n_habitats_total * 0.3)))
                
                # Sort by variance
                sorted_stats = sorted(cluster_data['habitat_stats'].items(), 
                                     key=lambda x: x[1]['variance'])
                
                habitat_names = [str(h[0])[:45] for h in sorted_stats]
                variances = [h[1]['variance'] for h in sorted_stats]
                n_plots = [h[1]['n'] for h in sorted_stats]
                
                # Color by variance (green=good, red=poor)
                colors_bar = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(variances)))
                
                bars = ax.barh(habitat_names, variances, color=colors_bar, 
                              edgecolor='black', linewidth=0.5)
                
                # Add sample size labels
                for i, (var, n) in enumerate(zip(variances, n_plots)):
                    ax.text(var, i, f'  n={n}', va='center', fontsize=8, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                
                ax.set_xlabel('Spatial Variance (lower = tighter clustering)', fontsize=11, fontweight='bold')
                ax.set_ylabel('Habitat Type', fontsize=10)
                ax.set_title(f'Spatial Clustering by Habitat: {selected_map_cluster}', 
                           fontsize=12, fontweight='bold', pad=15)
                ax.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
                
                plt.tight_layout()
                
                if save_figures:
                    map_name_clean = selected_map_cluster.replace('/', '_').replace(' ', '_')
                    save_figure(fig, f"05_habitat_clustering_bars_{map_name_clean}", figures_folder)
                    saved_figures.append(f"Habitat clustering bars ({selected_map_cluster})")
                
                st.pyplot(fig)
                plt.close()
                
                st.caption("**Green** = well-discriminated (tight clustering), **Red** = poorly-discriminated (dispersed)")
                
                # Centroid map - ONLY if reasonable number of habitats
                if n_habitats_total <= 12:
                    st.markdown("#### Habitat centroid positions")
                    
                    fig, ax = plt.subplots(figsize=(10, 9))
                    
                    colors = plt.cm.tab20(np.linspace(0, 1, n_habitats_total))
                    
                    for idx, (habitat, stats) in enumerate(cluster_data['habitat_stats'].items()):
                        cx, cy = stats['centroid_x'], stats['centroid_y']
                        
                        # Plot spread circle FIRST (smaller, behind)
                        # Scale down radius significantly (20% of std_dev)
                        circle = plt.Circle((cx, cy), stats['std_dev'] * 0.2, 
                                          color=colors[idx], alpha=0.25, 
                                          linestyle='--', linewidth=1.5, fill=False, zorder=5)
                        ax.add_patch(circle)
                        
                        # Plot centroid on top
                        ax.scatter(cx, cy, s=stats['n']*20, c=[colors[idx]], 
                                 alpha=0.7, edgecolors='black', linewidth=2,
                                 label=f"{str(habitat)[:30]} (n={stats['n']})", zorder=10)
                        
                        # Small label
                        ax.text(cx, cy, str(habitat)[:8], 
                               fontsize=8, ha='center', va='center',
                               fontweight='bold', color='black',
                               bbox=dict(boxstyle='round,pad=0.2', 
                                       facecolor='white', alpha=0.8, edgecolor='none'))
                    
                    ax.set_xlabel('X Coordinate', fontsize=11, fontweight='bold')
                    ax.set_ylabel('Y Coordinate', fontsize=11, fontweight='bold')
                    ax.set_title('Habitat Spatial Distribution', fontsize=12, fontweight='bold')
                    ax.set_xlim(-0.05, 1.05)
                    ax.set_ylim(-0.05, 1.05)
                    ax.set_aspect('equal')
                    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
                    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, framealpha=0.95)
                    
                    plt.tight_layout()
                    
                    if save_figures:
                        map_name_clean = selected_map_cluster.replace('/', '_').replace(' ', '_')
                        save_figure(fig, f"05_habitat_centroids_{map_name_clean}", figures_folder)
                        saved_figures.append(f"Habitat centroids ({selected_map_cluster})")
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    st.caption("""
                    **Points:** Centroids (size = n plots) | **Small dashed circles:** Spread indicator (scaled)  
                    Well-separated centroids with small circles = good discrimination
                    """)
                    
                else:
                    st.info(f"""
                    📊 **{n_habitats_total} habitat types** - too many for clear centroid visualization.  
                    The bar chart above shows discrimination quality for all habitat types.  
                    Consider using **major_type** instead of **habitat_type** for clearer spatial patterns.
                    """)
                
        else:
            st.warning(f"⚠️ Could not analyze habitat differentiation. Check if '{habitat_column}' column exists.")
    
    # 6. OVERALL RANKING
    st.markdown("---")
    st.markdown("## 🏆 Overall Map Ranking")
    
    with st.expander("ℹ️ How Scores Are Calculated", expanded=False):
        st.markdown("""
        **Individual Map Scores:**

        Each map receives scores for:
        1. **Procrustes:** Mean similarity with other maps (0-1) - *Diagnostic only, excluded from score*
        2. **Gradient:** Mean R² across indicators (0-1) - *Primary validation*
        3. **Distance:** Mantel test correlation (0-1) - *Distance preservation*
        4. **Prediction:** Mean R² across indicators (0-1) - *Primary validation*
        5. **Habitat ANOVA:** Mean significance (0-1) - *Data quality check (excluded from overall)*
        6. **Clustering:** Discrimination index (0-1) - *Habitat discrimination*

        **Overall Score Weighting (Default):**
        - **35%** Environmental Gradient (R²) - PRIMARY goal
        - **25%** Predictive Accuracy (k-NN R²) - PRIMARY goal
        - **25%** Habitat Clustering - SECONDARY goal
        - **15%** Distance Preservation (Mantel test)
        - **0%** Position Stability (Procrustes) - diagnostic only, not a quality measure
        - **0%** Habitat ANOVA (validates classification, not map)

        **Rationale:** Weighting reflects that the primary goal is to validate that
        co-occurrence-based networks capture environmental gradients (Ellenberg values),
        even though the network construction uses no trait data. Procrustes is excluded
        from scoring because it measures consensus (similarity to other maps), not quality —
        there is no ground truth configuration to compare against.

        Maps are ranked from highest to lowest overall score.
        """)
    
    # Determine weights based on user selection
    if scoring_method == "Weighted (Recommended)":
        scoring_weights = None  # Use default weights from function
    else:
        # Equal weights
        scoring_weights = {
            'procrustes': 1.0,
            'gradient': 1.0,
            'distance': 1.0,
            'prediction': 1.0,
            'habitat': 1.0,
            'clustering': 1.0
        }

    scores_df = calculate_map_scores(data, all_results, weights=scoring_weights)
    
    if not scores_df.empty:
        # Highlight best map
        best_map = scores_df.iloc[0]
        
        scoring_note = "weighted scoring" if scoring_method == "Weighted (Recommended)" else "equal weights"

        st.markdown(f"""
        <div class="best-map">
        <h3>🥇 Recommended Reference Map: {best_map['map']}</h3>
        <p><strong>Layout Method:</strong> {best_map['method']}</p>
        <p><strong>Overall Score:</strong> {best_map['overall']:.3f} ({scoring_note})</p>
        <p>This map shows the best overall performance across all validation metrics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Full ranking table
        st.markdown("### Complete ranking")
        
        # Determine active weights to filter out zero-weighted metrics
        active_weights = scoring_weights if scoring_weights else {
            'gradient': 0.35, 'prediction': 0.25, 'clustering': 0.25,
            'distance': 0.15, 'procrustes': 0.00, 'habitat': 0.00
        }
        excluded_metrics = {m for m, w in active_weights.items() if w == 0}
        
        display_cols = ['map', 'method', 'overall']
        metric_cols = [c for c in scores_df.columns 
                       if c not in ['map', 'method', 'overall'] and c not in excluded_metrics]
        display_cols.extend(metric_cols)
        
        display_df = scores_df[display_cols].copy()
        
        st.dataframe(
            display_df.style.format({
                c: '{:.3f}' for c in display_df.columns if c not in ['map', 'method']
            }).background_gradient(subset=['overall'], cmap='RdYlGn', vmin=0, vmax=1),
            use_container_width=True,
            hide_index=True
        )
        
        # Performance breakdown
        st.markdown("### Performance breakdown")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        maps = scores_df['map'].tolist()
        y_pos = np.arange(len(maps))
        
        # Colors for each metric
        metric_colors = {
            'procrustes': '#3498db',
            'gradient': '#2ecc71',
            'distance': '#f39c12',
            'prediction': '#e74c3c',
            'habitat': '#9b59b6',
            'clustering': '#34495e'
        }
        
        left = np.zeros(len(maps))
        
        for metric in metric_cols:
            if metric in scores_df.columns:
                values = scores_df[metric].fillna(0).values
                ax.barh(y_pos, values, left=left, 
                       label=metric.capitalize(),
                       color=metric_colors.get(metric, 'gray'),
                       alpha=0.8)
                left += values
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{m[:20]} ({scores_df.iloc[i]['method']})" 
                           for i, m in enumerate(maps)], fontsize=9)
        ax.set_xlabel('Cumulative Score', fontsize=11)
        ax.set_title('Performance Breakdown by Metric', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_figures:
            save_figure(fig, "06_overall_performance_breakdown", figures_folder)
            saved_figures.append("Overall performance breakdown")
        
        st.pyplot(fig)
        plt.close()
        
        # Method comparison if both present
        if 'FR' in methods and 'MDS' in methods:
            st.markdown("### Method comparison")
            
            fr_scores = scores_df[scores_df['method'] == 'FR']['overall']
            mds_scores = scores_df[scores_df['method'] == 'MDS']['overall']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("FR Best Score", f"{fr_scores.max():.3f}")
                st.metric("FR Mean Score", f"{fr_scores.mean():.3f}")
            with col2:
                st.metric("MDS Best Score", f"{mds_scores.max():.3f}")
                st.metric("MDS Mean Score", f"{mds_scores.mean():.3f}")
            
            if best_map['method'] == 'FR':
                st.info("📊 FR method shows best overall performance for this dataset")
            else:
                st.info("📊 MDS method shows best overall performance for this dataset")
    
    # SAVED FIGURES SUMMARY
    if save_figures and saved_figures:
        st.markdown("---")
        st.markdown("### 💾 Saved figures")
        st.success(f"✅ Successfully saved {len(saved_figures)} figures to: `{figures_folder}`")
        
        with st.expander("📋 Saved figures list"):
            for i, fig_name in enumerate(saved_figures, 1):
                st.text(f"{i}. {fig_name}")
        
        # Add button to copy figures to outputs
        if st.button("📤 Copy figures to outputs folder"):
            # Use project directory from session state
            project_dir = Path(st.session_state.get('project_dir', figures_folder.parent))
            outputs_folder = project_dir / "outputs" / "figures"
            outputs_folder.mkdir(parents=True, exist_ok=True)

            import shutil
            copied_count = 0
            for fig_file in figures_folder.glob("*.png"):
                shutil.copy2(fig_file, outputs_folder / fig_file.name)
                copied_count += 1

            st.success(f"✅ Copied {copied_count} figures to: `{outputs_folder}`")
    
    # EXPORT SECTION
    st.markdown("---")
    st.markdown("### 📥 Export results")
    
    # Collect all tables for export
    tables_to_export = {}
    
    if 'procrustes_matrix' in all_results:
        tables_to_export['Procrustes_Similarity'] = all_results['procrustes_matrix']
    
    if 'gradients' in all_results:
        tables_to_export['Environmental_Gradients'] = all_results['gradients']
    
    if 'distance_global' in all_results:
        # Convert distance results to DataFrame
        dist_rows = []
        for name, result in all_results['distance_global'].items():
            dist_rows.append({
                'overlay': name,
                'method': data[name]['method'],
                'correlation': result['correlation'],
                'p_value': result['p_value'],
                'n_plots': result['n_plots'],
                'n_pairs': result['n_pairs']
            })
        if dist_rows:
            tables_to_export['Distance_Preservation'] = pd.DataFrame(dist_rows)
    
    if 'predictions' in all_results:
        tables_to_export['Predictive_Validation'] = all_results['predictions']
    
    if 'habitat' in all_results:
        # Convert habitat ANOVA results to DataFrame
        hab_rows = []
        for name, results in all_results['habitat'].items():
            for indicator, result in results.items():
                hab_rows.append({
                    'overlay': name,
                    'method': data[name]['method'],
                    'indicator': indicator,
                    'F_statistic': result['f_stat'],
                    'p_value': result['p_value'],
                    'n_habitats': result['n_habitats']
                })
        if hab_rows:
            tables_to_export['Habitat_ANOVA'] = pd.DataFrame(hab_rows)
    
    if 'spatial_clustering' in all_results:
        # Convert clustering results to DataFrame
        clust_rows = []
        for name, result in all_results['spatial_clustering'].items():
            clust_rows.append({
                'overlay': name,
                'method': data[name]['method'],
                'mean_within_variance': result['mean_within_variance'],
                'mean_between_distance': result['mean_between_distance'],
                'discrimination_index': result['discrimination_index'],
                'disc_p_value': result.get('disc_p_value', np.nan),
                'n_habitats': result['n_habitats']
            })
        if clust_rows:
            tables_to_export['Habitat_Clustering'] = pd.DataFrame(clust_rows)
        
        # Also export per-habitat details for each map
        for name, result in all_results['spatial_clustering'].items():
            hab_detail_rows = []
            for habitat, stats in result['habitat_stats'].items():
                hab_detail_rows.append({
                    'habitat': str(habitat),
                    'n_plots': stats['n'],
                    'centroid_x': stats['centroid_x'],
                    'centroid_y': stats['centroid_y'],
                    'variance': stats['variance'],
                    'std_dev': stats['std_dev']
                })
            if hab_detail_rows:
                sheet_name = f"HabClust_{name[:20]}"  # Truncate for Excel sheet name limit
                tables_to_export[sheet_name] = pd.DataFrame(hab_detail_rows)
    
    if not scores_df.empty:
        tables_to_export['Overall_Rankings'] = scores_df
    
    # Create Excel file in memory
    if tables_to_export:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            for sheet_name, df in tables_to_export.items():
                # Excel sheet names limited to 31 characters
                safe_sheet_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
        
        buffer.seek(0)
        
        # Single export button
        st.download_button(
            label="📥 Export All Results to Excel",
            data=buffer,
            file_name=f"validation_results_{datetime.date.today()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary"
        )
        
        st.caption(f"**Excel file contains {len(tables_to_export)} sheets:** {', '.join(tables_to_export.keys())}")
    else:
        st.warning("No results to export")


# FOOTER
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 0.9em;'>
    EcoNetMap - Comprehensive Validation | Network-Based Ecological Cartography
</div>
""", unsafe_allow_html=True)
