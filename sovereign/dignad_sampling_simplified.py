"""
Simplified DIGNAD Sampling Strategy for Four-Parameter Model
Focuses on: dY_T (tradable output), dY_N (non-tradable output), 
           dK_priv (private capital), dK_pub (public capital)
"""

import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle

def analyze_parameter_correlations(flood_df, plot=True):
    """
    Analyze correlations between the four DIGNAD parameters
    
    Parameters:
    -----------
    flood_df : pd.DataFrame
        DataFrame with columns [dY_T, dY_N, dK_priv, dK_pub]
    plot : bool
        Whether to create visualization plots
    
    Returns:
    --------
    corr_matrix : pd.DataFrame
        Correlation matrix of parameters
    """
    param_cols = ['dY_T', 'dY_N', 'dK_priv', 'dK_pub']
    
    # Calculate correlations
    corr_matrix = flood_df[param_cols].corr()
    
    if plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Correlation heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   vmin=-1, vmax=1, ax=axes[0,0], square=True,
                   cbar_kws={'label': 'Correlation'})
        axes[0,0].set_title('Parameter Correlations')
        
        # Scatter plots for key relationships
        scatter_pairs = [
            ('dK_priv', 'dK_pub', 'Private vs Public Capital'),
            ('dY_T', 'dY_N', 'Tradable vs Non-tradable Output'),
            ('dK_priv', 'dY_T', 'Private Capital vs Tradable Output'),
            ('dK_pub', 'dY_N', 'Public Capital vs Non-tradable Output'),
            ('dY_T', 'dK_pub', 'Tradable Output vs Public Capital')
        ]
        
        for ax, (x, y, title) in zip(axes.flat[1:], scatter_pairs):
            ax.hexbin(flood_df[x], flood_df[y], gridsize=30, cmap='YlOrRd', alpha=0.8)
            ax.set_xlabel(f'{x} (%)')
            ax.set_ylabel(f'{y} (%)')
            ax.set_title(title)
            
            # Add correlation text
            corr = corr_matrix.loc[x, y]
            ax.text(0.05, 0.95, f'ρ = {corr:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('DIGNAD Parameter Relationships from Flood Simulations', y=1.02)
        plt.tight_layout()
        plt.show()
    
    # Print key correlations
    print("\nKey Parameter Correlations:")
    print("="*50)
    for i, param1 in enumerate(param_cols):
        for param2 in param_cols[i+1:]:
            corr = corr_matrix.loc[param1, param2]
            print(f"{param1} ↔ {param2}: {corr:+.3f}")
    
    return corr_matrix


def create_dignad_parameter_grid(flood_df, n_samples=500, method='combined', 
                                 ensure_extremes=True, random_state=42):
    """
    Create parameter grid for DIGNAD pre-computation from flood simulations
    
    Parameters:
    -----------
    flood_df : pd.DataFrame
        Combined flood simulation results with [dY_T, dY_N, dK_priv, dK_pub]
    n_samples : int
        Target number of parameter combinations
    method : str
        'clustering', 'stratified', 'kde', or 'combined'
    ensure_extremes : bool
        Whether to ensure extreme events are included
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    param_grid : pd.DataFrame
        Parameter grid for DIGNAD simulations
    """
    np.random.seed(random_state)
    param_cols = ['dY_T', 'dY_N', 'dK_priv', 'dK_pub']
    X = flood_df[param_cols].values
    
    print(f"\nCreating DIGNAD parameter grid using '{method}' method...")
    print(f"Target samples: {n_samples}")
    
    if method == 'clustering':
        grid = _create_clustering_grid(X, param_cols, n_samples)
    
    elif method == 'stratified':
        grid = _create_stratified_grid(flood_df, param_cols, n_samples)
    
    elif method == 'kde':
        grid = _create_kde_grid(X, param_cols, n_samples)
    
    elif method == 'combined':
        # Use multiple methods for better coverage
        n_per_method = n_samples // 3
        
        grid1 = _create_clustering_grid(X, param_cols, n_per_method)
        grid2 = _create_stratified_grid(flood_df, param_cols, n_per_method)
        grid3 = _create_kde_grid(X, param_cols, n_samples - 2*n_per_method)
        
        grid = pd.concat([grid1, grid2, grid3], ignore_index=True)
        
        # Remove duplicates
        grid = grid.drop_duplicates(subset=param_cols).reset_index(drop=True)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Add extreme events if requested
    if ensure_extremes:
        grid = _add_extreme_events(grid, flood_df, param_cols)
    
    # Add metadata
    grid['grid_method'] = method
    grid['grid_index'] = range(len(grid))
    
    # Rename columns for DIGNAD compatibility
    grid_renamed = grid.copy()
    grid_renamed.rename(columns={
        'dY_T': 'tradable_impact',
        'dY_N': 'nontradable_impact', 
        'dK_priv': 'private_impact',
        'dK_pub': 'public_impact'
    }, inplace=True)
    
    # Add fixed DIGNAD parameters
    grid_renamed['share_tradable'] = 0.5  # Adjust as needed
    grid_renamed['reconstruction_efficiency'] = 0
    grid_renamed['public_debt_premium'] = 0
    
    print(f"\nFinal parameter grid:")
    print(f"  Grid size: {len(grid_renamed)} unique combinations")
    print(f"  Parameter ranges:")
    for col in param_cols:
        print(f"    {col}: [{grid[col].min():.3f}, {grid[col].max():.3f}]")
    
    return grid_renamed


def _create_clustering_grid(X, param_cols, n_samples):
    """Create grid using k-means clustering"""
    
    # Standardize for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform k-means
    n_clusters = min(n_samples, len(X))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # Get cluster centers
    centers_scaled = kmeans.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)
    
    # Also include some boundary points from each cluster
    grid_points = list(centers)
    
    for i in range(n_clusters):
        cluster_mask = kmeans.labels_ == i
        if cluster_mask.sum() > 1:
            cluster_points = X[cluster_mask]
            # Add point furthest from center
            distances = np.linalg.norm(cluster_points - centers[i], axis=1)
            if len(distances) > 0:
                furthest_idx = np.argmax(distances)
                grid_points.append(cluster_points[furthest_idx])
    
    grid = pd.DataFrame(grid_points[:n_samples], columns=param_cols)
    return grid


def _create_stratified_grid(flood_df, param_cols, n_samples):
    """Create grid using stratified sampling based on severity"""
    
    # Calculate total loss for stratification
    flood_df = flood_df.copy()
    flood_df['total_loss'] = flood_df[param_cols].sum(axis=1)
    
    # Create severity bins
    n_bins = min(20, len(flood_df) // 10)
    flood_df['severity_bin'] = pd.qcut(flood_df['total_loss'], q=n_bins, 
                                       labels=False, duplicates='drop')
    
    # Sample from each bin
    samples_per_bin = n_samples // n_bins
    remainder = n_samples % n_bins
    
    stratified_samples = []
    
    for bin_id in range(n_bins):
        bin_data = flood_df[flood_df['severity_bin'] == bin_id]
        if len(bin_data) > 0:
            # Add extra sample to some bins to reach exact n_samples
            extra = 1 if bin_id < remainder else 0
            sample_size = min(samples_per_bin + extra, len(bin_data))
            
            bin_sample = bin_data[param_cols].sample(n=sample_size, replace=False)
            stratified_samples.append(bin_sample)
    
    grid = pd.concat(stratified_samples, ignore_index=True)
    return grid


def _create_kde_grid(X, param_cols, n_samples):
    """Create grid by sampling from kernel density estimate"""
    
    # Fit KDE to joint distribution
    kde = gaussian_kde(X.T, bw_method='scott')
    
    # Sample from KDE
    samples = kde.resample(n_samples).T
    
    # Ensure samples are within data bounds
    for i in range(samples.shape[1]):
        samples[:, i] = np.clip(samples[:, i], 
                               X[:, i].min() * 0.95,  # Allow slight extrapolation
                               X[:, i].max() * 1.05)
    
    grid = pd.DataFrame(samples, columns=param_cols)
    return grid


def _add_extreme_events(grid, flood_df, param_cols):
    """Add extreme parameter combinations to ensure tail coverage"""
    
    extreme_points = []
    X = flood_df[param_cols].values
    
    # Add maximum for each parameter
    for i, col in enumerate(param_cols):
        max_idx = np.argmax(X[:, i])
        extreme_points.append(X[max_idx])
        
        # Also add 99.5th percentile
        p995_val = np.percentile(X[:, i], 99.5)
        close_idx = np.argmin(np.abs(X[:, i] - p995_val))
        extreme_points.append(X[close_idx])
    
    # Add joint extremes (high total loss)
    total_loss = X.sum(axis=1)
    extreme_indices = np.argsort(total_loss)[-20:]  # Top 20 most severe
    for idx in extreme_indices:
        extreme_points.append(X[idx])
    
    # Add to grid
    extreme_df = pd.DataFrame(extreme_points, columns=param_cols)
    grid = pd.concat([grid, extreme_df], ignore_index=True)
    grid = grid.drop_duplicates(subset=param_cols).reset_index(drop=True)
    
    return grid


def visualize_parameter_grid(param_grid, flood_df, save_path=None):
    """
    Visualize how well the parameter grid covers the flood simulation space
    
    Parameters:
    -----------
    param_grid : pd.DataFrame
        Parameter grid (output from create_dignad_parameter_grid)
    flood_df : pd.DataFrame
        Original flood simulations
    save_path : str, optional
        Path to save the figure
    """
    # Map column names back if needed
    grid_cols = ['tradable_impact', 'nontradable_impact', 'private_impact', 'public_impact']
    flood_cols = ['dY_T', 'dY_N', 'dK_priv', 'dK_pub']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Define projection pairs
    projections = [
        (('dY_T', 'tradable_impact'), ('dY_N', 'nontradable_impact'), 
         'Output Losses: Tradable vs Non-tradable'),
        (('dK_priv', 'private_impact'), ('dK_pub', 'public_impact'),
         'Capital Losses: Private vs Public'),
        (('dY_T', 'tradable_impact'), ('dK_priv', 'private_impact'),
         'Tradable Output vs Private Capital'),
        (('dY_N', 'nontradable_impact'), ('dK_pub', 'public_impact'),
         'Non-tradable Output vs Public Capital'),
        (('dY_T', 'tradable_impact'), ('dK_pub', 'public_impact'),
         'Tradable Output vs Public Capital'),
        (('dY_N', 'nontradable_impact'), ('dK_priv', 'private_impact'),
         'Non-tradable Output vs Private Capital')
    ]
    
    for ax, ((x_flood, x_grid), (y_flood, y_grid), title) in zip(axes.flat, projections):
        # Plot flood simulations as density
        ax.hexbin(flood_df[x_flood], flood_df[y_flood], 
                 gridsize=30, cmap='Blues', alpha=0.5, label='Flood simulations')
        
        # Plot grid points
        ax.scatter(param_grid[x_grid], param_grid[y_grid], 
                  color='red', s=15, alpha=0.7, label='Grid points', zorder=5)
        
        ax.set_xlabel(f'{x_flood} (%)')
        ax.set_ylabel(f'{y_flood} (%)')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Parameter Grid Coverage Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def calculate_grid_coverage(param_grid, flood_df, max_distance_threshold=None):
    """
    Calculate coverage statistics for the parameter grid
    
    Parameters:
    -----------
    param_grid : pd.DataFrame
        Parameter grid for DIGNAD
    flood_df : pd.DataFrame  
        Original flood simulations
    max_distance_threshold : float, optional
        Threshold for "good" coverage
    
    Returns:
    --------
    coverage_stats : dict
        Dictionary with coverage statistics
    """
    # Map columns
    grid_cols = ['tradable_impact', 'nontradable_impact', 'private_impact', 'public_impact']
    flood_cols = ['dY_T', 'dY_N', 'dK_priv', 'dK_pub']
    
    X_flood = flood_df[flood_cols].values
    X_grid = param_grid[grid_cols].values
    
    # For each flood point, find nearest grid point
    min_distances = []
    for flood_point in X_flood:
        distances = np.linalg.norm(X_grid - flood_point, axis=1)
        min_distances.append(np.min(distances))
    
    min_distances = np.array(min_distances)
    
    # Calculate statistics
    coverage_stats = {
        'mean_distance': np.mean(min_distances),
        'median_distance': np.median(min_distances),
        'max_distance': np.max(min_distances),
        'std_distance': np.std(min_distances),
        'p95_distance': np.percentile(min_distances, 95),
        'p99_distance': np.percentile(min_distances, 99)
    }
    
    if max_distance_threshold:
        coverage_stats['pct_well_covered'] = (min_distances <= max_distance_threshold).mean() * 100
    
    print("\nGrid Coverage Statistics:")
    print("="*50)
    for key, value in coverage_stats.items():
        if 'pct' in key:
            print(f"{key}: {value:.1f}%")
        else:
            print(f"{key}: {value:.4f}")
    
    # Plot distance distribution
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(min_distances, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(coverage_stats['mean_distance'], color='red', 
               linestyle='--', label=f"Mean: {coverage_stats['mean_distance']:.3f}")
    plt.axvline(coverage_stats['p95_distance'], color='orange',
               linestyle='--', label=f"95th %ile: {coverage_stats['p95_distance']:.3f}")
    plt.xlabel('Distance to Nearest Grid Point')
    plt.ylabel('Number of Flood Simulations')
    plt.title('Coverage Quality Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(min_distances, vert=True)
    plt.ylabel('Distance to Nearest Grid Point')
    plt.title('Coverage Quality Box Plot')
    plt.grid(alpha=0.3)
    
    plt.suptitle('Parameter Grid Coverage Quality', y=1.02)
    plt.tight_layout()
    plt.show()
    
    return coverage_stats


# Main workflow function
def create_comprehensive_dignad_grid(combined_df, n_samples=750, method='combined'):
    """
    Complete workflow to create DIGNAD parameter grid from flood simulations
    
    Parameters:
    -----------
    combined_df : pd.DataFrame
        combined DataFrame from run_flood_sim_for_macro with columns [dY_T, dY_N, dK_priv, dK_pub]
    n_samples : int
        Number of DIGNAD simulations to pre-compute
    method : str
        Sampling method ('clustering', 'stratified', 'kde', 'combined')
    Returns:
    --------
    param_grid : pd.DataFrame
        Parameter grid ready for DIGNAD simulation
    """
    
    #  Create parameter grid
    print(f"\n Creating parameter grid with {n_samples} points...")
    param_grid = create_dignad_parameter_grid(
        combined_df, 
        n_samples=n_samples,
        method=method,
        ensure_extremes=True
    )
    
    return param_grid


if __name__ == "__main__":
    print("Simplified DIGNAD Sampling Strategy")
    print("Focus on four parameters: dY_T, dY_N, dK_priv, dK_pub")
    print("\nUsage:")
    print("  from dignad_sampling_simplified import create_comprehensive_dignad_grid")
    print("  param_grid, flood_data = create_comprehensive_dignad_grid(")
    print("      baseline_current_df, baseline_adapted_df,")
    print("      future_current_df, future_adapted_df,")
    print("      n_samples=750")
    print("  )")
