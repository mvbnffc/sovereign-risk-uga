import geopandas as gpd
from shapely.geometry import LineString
from pyproj import Geod
import pandas as pd
from pathlib import Path
from rasterio.transform import from_origin
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
from rasterio.mask import mask
import scipy.stats as stats
from scipy.stats import genpareto, kstest

def map_flopros_to_adm(adm, raster_path, out_col="MerL_Riv"):
    '''
    Function for mapping FLOPROS raster data to admin areas.
    Takes the mode of FLOPROS raster values within each admin area polygon.
    Returns a GeoDataFrame with the new column added.
    '''
    with rasterio.open(raster_path) as src:
        gdf = adm.to_crs(src.crs)
        nodata = src.nodata

        def poly_mode(geom):
            a, _ = mask(src, [geom], crop=True, filled=False)
            v = a[0].compressed()
            if nodata is not None:
                v = v[v != nodata]
            if v.size == 0:
                return np.nan
            vals, counts = np.unique(v, return_counts=True)
            return vals[counts.argmax()]

        gdf[out_col] = [
            poly_mode(g) if g and not g.is_empty else np.nan
            for g in gdf.geometry
        ]
        return gdf

def calculate_geod_length(line):
    '''
    Function to caluclate the geodetic length of a LineString
    '''
    geod = Geod(ellps="WGS84") # our data is in WGS84 projection
    length = 0
    if isinstance(line, LineString):
        for i in range(len(line.coords)-1):
            lon1, lat1 = line.coords[i]
            lon2, lat2 = line.coords[i + 1]
            _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
            length += distance

    return length

def calculate_river_length_per_admin(admin, rivers, threshold, urbanisation, urbanisation_threshold):
    '''
    Function that caluclates the length of river in each admin area. The size of the river network
    is determined by an upsteam drainage area threshold (in sq km). Function is written to take
    FLOPROS as input for admin and hydroRIVERS as rivers. Function inputs should be geodataframes.
    Function outputs an admin dataset with an additional column of total river length within the 
    admin area. The function also calculates the length of rivers in highly populated areas using the
    GHSL Degree of Urbanisation Dataset
    ''' 
    # Filter the river lines based on the size threshold
    filtered_rivers = rivers[rivers['UPLAND_SKM'] > threshold] # NOTE: this column name is unique to HydroRIVERS
    # Filter urban areas based on the population density threshold
    urban_areas = urbanisation[urbanisation['DEGURBA_L2'] >= urbanisation_threshold]
    # Intersect filtered rivers with dense urban areas
    urban_rivers = gpd.overlay(filtered_rivers, urban_areas, how='intersection')
    # Intersect river lines with admin areas, splitting them as necessary
    intersected_rivers = gpd.overlay(filtered_rivers, admin, how='intersection')
    intersected_urban_rivers = gpd.overlay(urban_rivers, admin, how='intersection')
    # Calculate the geodetic length of each river segment 
    intersected_rivers['r_lng_m'] = intersected_rivers['geometry'].apply(calculate_geod_length)
    intersected_urban_rivers['u_r_lng_m'] = intersected_urban_rivers['geometry'].apply(calculate_geod_length)
    # Groub by admin area and sum segment lengths
    river_lengths_by_area = intersected_rivers.groupby('OBJECTID')['r_lng_m'].sum().reset_index() # NOTE: this column name is unique to FLOPROS dataset
    urban_river_lengths_by_area = intersected_urban_rivers.groupby('OBJECTID')['u_r_lng_m'].sum().reset_index() # NOTE: this column name is unique to FLOPROS dataset
    # Add the total length to the admin areas DataFrame
    admin = admin.merge(river_lengths_by_area, how='left', left_on='OBJECTID', right_on='OBJECTID') # NOTE: this column name is unique to FLOPROS dataset
    admin = admin.merge(urban_river_lengths_by_area, how='left', left_on='OBJECTID', right_on='OBJECTID') # NOTE: this column name is unique to FLOPROS dataset
    admin['r_lng_km'] = admin['r_lng_m'].fillna(0) / 1000
    admin['u_r_lng_km'] = admin['u_r_lng_m'].fillna(0) / 1000
    admin.drop(columns=['r_lng_m'], inplace=True)
    admin.drop(columns=['u_r_lng_m'], inplace=True)
        
    return admin

def calculate_increased_protection(admin, protection_goal):
    '''
    Function calculates how much additional protection is needed to achieve a protection goal.
    Function requires as input the target protection level and the admin dataset (FLOPROS layer).
    Function ouptus an admin dataset with an additional column with the amount of additional 
    protection needed.
    '''
    # Calculate additional protection needed
    admin['Add_Pr'] = protection_goal - admin['MerL_Riv']
    # Ensure there are no negative values
    admin['Add_Pr'] = admin['Add_Pr'].clip(lower=0)
    # Store the new protection level
    admin['New_Pr_L'] = protection_goal

    return admin

def calculate_increased_protection_costs(admin, unit_cost):
    '''
    This function calculates the cost of additional protection given a unit cost (per km).
    The function calculates the unit cost based on the length of rivers within the admin area.
    The cost is applied using the formula unit_cost * log2(additional_protection) from Boulange et al 2023.
    https://link.springer.com/article/10.1007/s11069-023-06017-7 
    The function outputs an admin dataset with an additional column with cost of increasing protection
    to desired levels. Also calculates cost for only urban rivers.
    '''
    # Calculate additional costs of protection
    admin['Add_Pr_c'] = np.log2(admin['Add_Pr']) * unit_cost * admin['r_lng_km']
    admin['Add_Pr_c_u'] = np.log2(admin['Add_Pr']) * unit_cost * admin['u_r_lng_km']

    return admin


def disaggregate_building_values(admin_areas, df, raster, occupancy_type, column_val):
    '''
    This function disaggregates building values from the GEM exposure database (https://github.com/gem/global_exposure_model)
    at Admin1 level to gridded urban maps maps from GHSL
    '''

    # Initialize an empty array for the output raster values
    output_values = np.zeros_like(raster.read(1), dtype=np.float32)

    # Ensure relevant columns are being treated as numeric
    df['TOTAL_REPL_COST_USD'] = pd.to_numeric(df['TOTAL_REPL_COST_USD'], errors='coerce')
    
    for index, admin_area in admin_areas.iterrows():
        
        # Filter the DataFrame for the current admin area and the specified occupancy type
        filtered_df = df[(df['GID_1'] == admin_area['GID_1']) & (df['OCCUPANCY'] == occupancy_type)]
        
        # Aggregate the total value for the specified occupancy type within the admin area
        total_value = filtered_df['TOTAL_REPL_COST_USD'].sum()
        
        # Proceed if there's a meaningful total value to disaggregate
        if total_value > 0:
            # Create a mask for the admin area
            geom_mask = geometry_mask([admin_area.geometry], invert=True, transform=raster.transform, out_shape=raster.shape)
            
            # Calculate the total area of the admin area covered by buildings in the raster
            total_area = raster.read(1)[geom_mask].sum()
            
            if total_area > 0:  # Prevent division by zero
                # Disaggregate the total value across the raster, weighted by building area
                output_values[geom_mask] += (raster.read(1)[geom_mask] / total_area) * total_value
    
    return output_values


def write_raster(output_path, raster_template, data):
    '''
    Function to write raster datasets
    '''
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=raster_template.height,
        width=raster_template.width,
        count=1,
        dtype=data.dtype,
        crs=raster_template.crs,
        transform=raster_template.transform,
    ) as dst:
        dst.write(data, 1)

def calculate_buidlings_for_dry_proofing(building_area, flood_2, flood_1000):
    '''
    function calculates which buildings are elgible for dry proofing. The criteria is buidlings within the 1000 year flood zone that are not exposed to
    flood depths >1 m in the 2 year flood zone. Function returns building area raster array with buildings elgible for dry proofing.
    '''
    # isolate buidlings within the 1000-year flood zone
    buildings_in_1000_year_flood = np.where((flood_1000 > 0) & (building_area > 0), 1, 0)
    # exclude buidlings in 2-year flood zone with flood depth >1 m
    buildings_for_dry_proofing = np.where((flood_2 <= 100) | (flood_2 == 0), buildings_in_1000_year_flood, 0) # NOTE: GIRI data is in cm

    return buildings_for_dry_proofing

def load_raster(raster_path, save_info=False):
    '''
    Load raster and make sure we get rid of NaNs and negatives.
    '''
    
    raster = rasterio.open(raster_path)
    raster_array = raster.read(1)
    
    # Clean array (remove negatives and nans)
    raster_array[np.isnan(raster_array)] = 0
    raster_array = np.where(raster_array > 0, raster_array, 0)

    if save_info:
        return raster_array, raster.meta
    
    else:
        return raster_array

def calculate_reconstruction_value_exposed(reconstruction_value_grid, flood, depth_threshold=0):
    '''
    function calculates the reconstruction value of buildings within the flood extent and (if specified) a flood depth threshold.
    function retrurns an array of the flood exposed reconstruction value
    '''
    # Mask extent based on depth threshold
    flood_extent = np.where(flood>depth_threshold, flood, 0)
    # Calculate exposed reconstruction value
    value_exposed = np.where(flood_extent>0, reconstruction_value_grid, 0)

    return value_exposed


def union_gdfs(gdf1, gdf2):
    '''
    Function to apply union operation on two GeoDataFrames
    '''
    # Make sure both GeoDataFrames have the same CRS
    gdf2_copy = gdf2.copy().to_crs(gdf1.crs)
    # Perform the union operation
    return gpd.overlay(gdf1, gdf2_copy, how='union')

def sum_rasters(raster_list, output_path):
    """
    Sum multiple rasters (same extent, resolution, and alignment)
    block-by-block and write the result to a new raster.

    raster_list: list of paths to input rasters
    output_path: path for the output raster
    """

    # Open all rasters
    datasets = [rasterio.open(fp) for fp in raster_list]

    # Use metadata from the first raster
    profile = datasets[0].meta.copy()
    profile.update(dtype=rasterio.float32, compress='lzw', nodata=0)

    with rasterio.open(output_path, 'w', **profile) as dst:
        # Iterate over block windows of the first raster
        for ji, window in datasets[0].block_windows(1):

            # Read each raster’s block
            arrays = [ds.read(1, window=window) for ds in datasets]

            # Sum them (ignores nodata=0 correctly)
            summed = np.sum(arrays, axis=0)

            dst.write(summed.astype(rasterio.float32), window=window, indexes=1)

    # Close datasets
    for ds in datasets:
        ds.close()


def raster_sum(raster_path):
    """
    Sum all valid values in raster, return sum.
    """
    with rasterio.open(raster_path) as raster:
        data = raster.read(1)
        data = data[data != raster.nodata]
        return np.nansum(data)



def df_to_raster(sub: pd.DataFrame, out_path: Path, LAT_COL='latitude', LON_COL='longitude', VALUE_COL='adjusted_return_period'):
    """
    Function for converting dataframe subset to raster.
    Covnerts lat/lon/value dataframe to GeoTIFF.

    This function is used to convert the ISIMIP dataframe to individual RP change rasters
    """
    if sub.empty:
        return

    # unique sorted coords
    lats = np.sort(sub[LAT_COL].unique())
    lons = np.sort(sub[LON_COL].unique())

    if len(lats) < 2 or len(lons) < 2:
        print(f"⚠️ Not enough points to build a grid for {out_path.name}")
        return

    lat_res = float(np.min(np.diff(lats)))
    lon_res = float(np.min(np.diff(lons)))

    # pivot to 2D grid (rows = lat, cols = lon)
    grid = sub.pivot(index=LAT_COL, columns=LON_COL, values=VALUE_COL)

    # ensure sorted and north-to-south
    grid = grid.sort_index(ascending=False)

    data = grid.values.astype('float32')

    # affine transform: cell centers → raster origin at top-left
    max_lat = lats.max()
    min_lon = lons.min()
    transform = from_origin(
        min_lon - lon_res / 2.0,
        max_lat + lat_res / 2.0,
        lon_res,
        lat_res,
    )

    height, width = data.shape

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -9999.0,
    }

    # optional: fill NaNs with nodata
    data = np.where(np.isfinite(data), data, profile["nodata"])

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)

import numpy as np
import rasterio

def disaggregate_total_to_raster(total_value, weights_raster_path, output_path, *, nodata_out=0):
    """
    Disaggregate a single total_value over a raster using its cell values as weights.

    out_cell = total_value * (weight_cell / sum_of_weights)

    Parameters
    ----------
    total_value : float
        Total value to distribute (e.g. 500e9).
    weights_raster_path : str or Path
        Path to raster whose values act as weights.
    output_path : str or Path
        Path to write the output raster (GeoTIFF).
    nodata_out : numeric
        Nodata value to set in output (default 0).

    Writes
    ------
    A raster at output_path with same shape/transform/crs as weights raster.
    """

    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(weights_raster_path) as src:
        w = src.read(1).astype("float64")
        profile = src.profile.copy()
        nodata_in = src.nodata

    # Clean weights: mask nodata, NaN, inf, and non-positive values
    valid = np.isfinite(w)
    if nodata_in is not None:
        valid &= (w != nodata_in)
    valid &= (w > 0)

    w_clean = np.where(valid, w, 0.0)
    total_w = w_clean.sum()

    out = np.zeros_like(w_clean, dtype="float32")
    if total_w > 0:
        out = (w_clean / total_w * float(total_value)).astype("float32")

    # Write output
    profile.update(dtype="float32", nodata=nodata_out, count=1, compress="lzw")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out, 1)

def decluster_maxima(data, threshold, min_separation):
    """
    Decluster the data by selecting local maxima that exceed a given threshold,
    ensuring that selected maxima are at least `min_separation` indices apart.

    Parameters
    ----------
    data : np.ndarray
        1D array of data points.
    threshold : float
        The threshold value to identify exceedances.
    min_separation : int
        Minimum number of indices between selected maxima.
    """
    exceed = data > threshold
    declustered = []
    last_peak_idx = -np.inf
    
    i = 0
    n = len(data)
    while i < n:
        if exceed[i]:
            cluster_start = i
            j = i + 1
            while j < n and exceed[j]:
                j += 1
            cluster_end = j
            if cluster_start - last_peak_idx >= min_separation:
                cluster_max = data[cluster_start:cluster_end].max()
                declustered.append(cluster_max)
                last_peak_idx = cluster_start
            i = cluster_end
        else:
            i += 1
    return np.array(declustered)

def gpd_negative_loglikelihood(params, x):
    """
    Negative log-likelihood for Generalized Pareto Distribution (GPD) with location fixed at 0.

    Formula taken from: An Introduction to Statistical Modeling of Extreme Values, Coles (2001), Section 4.3.2

    Parameters
    ----------
    params : array-like, shape (2,)
        GPD parameters: [xi, log_sigma]
    x : array-like
        Excesses over threshold (x > 0)
    """
    xi, log_sigma = params
    sigma = np.exp(log_sigma)
    if sigma <= 0:
        return np.inf

    t = 1.0 + xi * x / sigma
    # If t explodes then return infinity
    if np.any(t <= 0):
        return np.inf
    n = x.size
    nll = n * log_sigma + (1.0/xi + 1.0) * np.log(t).sum()
    return nll

def numerical_hessian(f, theta, x, eps=1e-5):
    """
    Compute the numerical Hessian of function f at point theta using central differences.

    Parameters
    ----------
    f : callable
        Function for which to compute the Hessian. Should take parameters (theta, x).
    theta : array-like, shape (2,)
        Point at which to compute the Hessian.
    x : array-like
        Additional data passed to function f.
    eps : float
        Small perturbation for finite difference approximation.
    """
    theta = np.asarray(theta, dtype=float)
    hessian = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei = np.zeros(2)
            ej = np.zeros(2)
            ei[i] = eps
            ej[j] = eps
            fpp = f(theta + ei + ej, x)
            fpm = f(theta + ei - ej, x)
            fmp = f(theta - ei + ej, x)
            fmm = f(theta - ei - ej, x)
            hessian[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
    return hessian

def gpd_fit_with_covariances(excesses):
    """
    Fit GPD (loc=0) to excesses and return and also provides back the covariance matrix of the
    shape and scale parameters using numerical Hessian around the MLE.

    Parameters
    ----------
    excesses : array-like
        Excesses over threshold (x > 0)

    Returns
    -------
    xi_hat : float
        Estimated shape parameter.
    sigma_hat : float
        Estimated scale parameter.
    cov_theta : np.ndarray, shape (2, 2)
        Covariance matrix of (xi, sigma) estimates.
    """
    excesses = np.asarray(excesses, dtype=float)
    xi_hat, _, sigma_hat = stats.genpareto.fit(excesses, floc=0.0)

    # Next, we compute the Fisher-information–based covariance matrix
    # of the parameter estimates via numerical Hessian.
    log_sigma_hat = np.log(sigma_hat) # This improves numerical stability
    params_hat = np.array([xi_hat, log_sigma_hat])
    H_phi = numerical_hessian(gpd_negative_loglikelihood, params_hat, excesses)

    # Invert to get covariance in phi-space (MLE estimation)
    cov_phi = np.linalg.inv(H_phi)

    # Transform back to (xi, sigma) space using delta method
    sigma = sigma_hat
    J = np.array([ # Jacobian to perform transformation
        [1.0, 0.0],
        [0.0, sigma]
    ])
    cov_theta = J @ cov_phi @ J.T   # 2x2 covariance for (xi, sigma)
    return xi_hat, sigma_hat, cov_theta

def gpd_return_level_and_var_log(delta_excesses, threshold, T, years, cov_xi_sigma):
    """
    Compute the GPD return level z_T and the variance of its logarithm using the delta method.

    Parameters
    ----------
    delta_excesses : array-like
        1D array of excesses (Q - u) after declustering
    threshold : float
        Threshold u
    T : float
        Return period in years (e.g. 100.)
    years : float
        Number of years of data used
    cov_xi_sigma : np.ndarray, shape (2, 2)
        Covariance matrix for (xi, sigma) in the order [xi, sigma]

    Returns
    -------
    zT : float
        Return level for return period T
    var_log_zT : float
        Variance of log(zT) using the delta method
    """
    # Fit GPD similar to before
    xi, _, sigma = stats.genpareto.fit(delta_excesses, floc=0)
    lam = len(delta_excesses) / years   # lambda

    A = lam * T
    if A <= 1:
        raise ValueError("Return period too large. Not enough exceedances in time period.")

    if np.isclose(xi, 0.0): # The gumbell limit
        zT = threshold + sigma * np.log(A)
        dz_dsigma = np.log(A) # derivative wrt sigma
        dz_dxi = 0.0     # derivative wrt xi -> 0
    else:
        B = A**xi   # Standard GPD formula
        zT = threshold + (sigma / xi) * (B - 1.0)
        dz_dsigma = (B - 1.0) / xi # derivative wrt sigma
        dz_dxi = sigma * ((B * np.log(A) * xi - (B - 1.0)) / (xi**2)) # derivative wrt xi

    # Gradient of log zT
    dy_dxi = dz_dxi / zT # derivative of log zT wrt xi
    dy_dsigma = dz_dsigma / zT # derivative of log zT wrt sigma
    grad = np.array([dy_dxi, dy_dsigma]) 

    # Delta-method
    var_log_zT = float(grad @ cov_xi_sigma @ grad)
    return zT, var_log_zT

def get_return_level_and_uncertainty(discharge_data, threshold, return_period, years, min_separation=3):
    """
    Wrapper function to get return level and uncertainty from excesses.

    Parameters
    ----------
    excesses : array-like
        1D array of excesses (Q - u) after declustering
    threshold : float
        Threshold u
    T : float
        Return period in years (e.g. 100.)
    years : float
        Number of years of data used

    Returns
    -------
    zT : float
        Return level for return period T
    var_log_zT : float
        Variance of log(zT) using the delta method
    """
    historical_pot = decluster_maxima(discharge_data, threshold, min_separation) - threshold
    xi_hat, sigma_hat, cov_xi_sigma = gpd_fit_with_covariances(historical_pot)
    zT, var_log_zT = gpd_return_level_and_var_log(
        delta_excesses=historical_pot,
        threshold=threshold,
        T=return_period,
        years=years,
        cov_xi_sigma=cov_xi_sigma
    )
    return zT, np.log(zT), np.sqrt(var_log_zT), xi_hat, sigma_hat


def pot_with_optimal_threshold(
    x,
    candidate_ps=None,
    min_exceedances=50,
    xi_bounds=(-0.2, 0.7),
    alpha_gof=0.05,
    delta_xi=0.05,
    delta_log_q=0.1,
    return_period=100.0,
):
    """
    Automatic POT threshold selection for a single time series, using GPD fits
    with covariance and returning covariance-based uncertainty for Q_T.

    Parameters
    ----------
    x : array-like
        1D array of discharge values (e.g. daily flows).
    candidate_ps : list of float, optional
        Candidate quantile levels for thresholds, e.g. [0.90, 0.94, 0.96, 0.97, 0.98, 0.99].
        If None, a default grid is used.
    min_exceedances : int, optional
        Minimum number of exceedances required for a threshold to be considered.
    xi_bounds : (float, float), optional
        Acceptable range for the GPD shape parameter xi. Thresholds with xi outside
        this range are discarded.
    alpha_gof : float, optional
        Significance level for the KS goodness-of-fit test on the transformed
        exceedances (GPD -> Uniform[0,1]). Thresholds with p < alpha_gof are discarded.
    delta_xi : float, optional
        Maximum allowed change in xi between adjacent candidate thresholds in the
        "stability region".
    delta_log_q : float, optional
        Maximum allowed change in log(Q_T) between adjacent thresholds.
    return_period : float, optional
        Return period T for which Q_T is computed (e.g. 100 for Q100).

    Returns
    -------
    best : dict or None
        Dictionary with fields:
            - 'p'             : chosen quantile level
            - 'u'             : chosen threshold value
            - 'xi'            : GPD shape at u
            - 'sigma'         : GPD scale at u
            - 'cov_xi_sigma'  : 2x2 covariance matrix of (xi, sigma)
            - 'qT'            : Q_T in original units
            - 'log_qT'        : log(Q_T)
            - 'var_qT'        : variance of Q_T (delta method)
            - 'var_log_qT'    : variance of log(Q_T) (delta method)
            - 'n_exc'         : number of exceedances
            - 'ks_pvalue'     : KS test p-value
        Returns None if no acceptable threshold was found.
    diagnostics : pd.DataFrame
        DataFrame with one row per candidate that passed basic feasibility checks,
        including all statistics computed. Useful for debugging and plotting stability.
    """

    x = np.asarray(x)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return None, pd.DataFrame()

    if candidate_ps is None:
        candidate_ps = [0.90, 0.94, 0.96, 0.97, 0.98, 0.99]

    rows = []

    for p in candidate_ps:
        u = np.quantile(x, p)
        z = x[x > u] - u
        n_exc = len(z)
        if n_exc < min_exceedances:
            continue

        try:
            xi, sigma, cov_theta = gpd_fit_with_covariances(z)
        except Exception:
            continue

        if not (xi_bounds[0] < xi < xi_bounds[1]):
            continue
        lambda_exc = n_exc / n
        A = return_period * lambda_exc
        if A <= 0:
            continue

        try:
            if np.abs(xi) < 1e-6:
                qT = u + sigma * np.log(A)
                dq_dsigma = np.log(A)
                dq_dxi = 0.0 
            else:
                B = A**xi
                qT = u + (sigma / xi) * (B - 1.0)
                dq_dsigma = (B - 1.0) / xi
                dq_dxi = (-sigma / (xi**2)) * (B - 1.0) + (sigma / xi) * B * np.log(A)
        except Exception:
            continue

        if not np.isfinite(qT) or qT <= 0:
            continue

        log_qT = np.log(qT)

        # Delta method
        grad = np.array([dq_dxi, dq_dsigma])  # shape (2,)
        var_qT = float(grad @ cov_theta @ grad.T)
        if var_qT < 0:
            var_qT = np.nan
        # Var(log Q_T) ≈ Var(Q_T) / Q_T^2
        var_log_qT = var_qT / (qT**2) if np.isfinite(var_qT) and qT > 0 else np.nan

        # KS
        try:
            uvals = genpareto.cdf(z, c=xi, loc=0.0, scale=sigma)
            _, ks_p = kstest(uvals, "uniform")
        except Exception:
            ks_p = 0.0

        rows.append(
            {
                "p": p,
                "u": u,
                "n_exc": n_exc,
                "xi": xi,
                "sigma": sigma,
                "lambda_exc": lambda_exc,
                "qT": qT,
                "log_qT": log_qT,
                "var_qT": var_qT,
                "var_log_qT": var_log_qT,
                "ks_pvalue": ks_p,
                "cov_xi_sigma": cov_theta,
            }
        )

    diagnostics = pd.DataFrame(rows)
    if diagnostics.empty:
        return None, diagnostics

    # Filter by KS p-value
    diagnostics = diagnostics[diagnostics["ks_pvalue"] >= alpha_gof]
    if diagnostics.empty:
        return None, diagnostics

    # Sort by threshold (quantile p)
    diagnostics = diagnostics.sort_values("p").reset_index(drop=True)

    # ---- Stability-based selection ----
    best_idx = None
    prev_row = None

    for i, row in diagnostics.iterrows():
        if prev_row is None:
            prev_row = row
            best_idx = i
            continue

        d_xi = abs(row["xi"] - prev_row["xi"])
        d_log_q = abs(row["log_qT"] - prev_row["log_qT"])

        if d_xi <= delta_xi and d_log_q <= delta_log_q:
            prev_row = row
        else:
            break

    if best_idx is None:
        return None, diagnostics

    best_row = diagnostics.iloc[best_idx]
    best = {
        "p": float(best_row["p"]),
        "u": float(best_row["u"]),
        "xi": float(best_row["xi"]),
        "sigma": float(best_row["sigma"]),
        "cov_xi_sigma": best_row["cov_xi_sigma"],
        "qT": float(best_row["qT"]),
        "log_qT": float(best_row["log_qT"]),
        "var_qT": float(best_row["var_qT"]),
        "var_log_qT": float(best_row["var_log_qT"]),
        "n_exc": int(best_row["n_exc"]),
        "ks_pvalue": float(best_row["ks_pvalue"]),
    }

    return best, diagnostics


def extract_model_uncertainty(posterior, input_data):
    """
    Docstring for extract_model_uncertainty
    
    :param idata: Description
    """
    beta_m_da       = posterior["beta_m"]
    cell_factor_da  = posterior["cell_factor"]
    model_factor_da = posterior["model_factor"] 

    beta_m_s       = beta_m_da.stack(sample=("chain", "draw"))
    cell_factor_s  = cell_factor_da.stack(sample=("chain", "draw"))
    model_factor_s = model_factor_da.stack(sample=("chain", "draw"))

    beta_m_np       = beta_m_s.values
    cell_factor_np  = cell_factor_s.values
    model_factor_np = model_factor_s.values

    # cell_factor_np
    S_cf, d1_cf, d2_cf = cell_factor_np.shape
    if d1_cf <= 5 and d2_cf > 5:
        # (S, R, I) -> (S, I, R)
        cell_factor_np = np.transpose(cell_factor_np, (0, 2, 1))
        S_cf, I, R_cf = cell_factor_np.shape
    else:
        # (S, I, R)
        S_cf, I, R_cf = cell_factor_np.shape

    # model_factor_np
    S_mf, d1_mf, d2_mf = model_factor_np.shape
    if d1_mf <= 5 and d2_mf > 5:
        model_factor_np = np.transpose(model_factor_np, (0, 2, 1))
        S_mf, M, R_mf = model_factor_np.shape
    else:
        S_mf, M, R_mf = model_factor_np.shape

    S_b, _ = beta_m_np.shape

    S = min(S_cf, S_mf, S_b)
    R = min(R_cf, R_mf)

    beta_m_np       = beta_m_np[:S, :M]
    cell_factor_np  = cell_factor_np[:S, :, :R]
    model_factor_np = model_factor_np[:S, :M, :R]


    global_model_var_scalar = beta_m_np.var(axis=1).mean()
    interaction_s = np.einsum("sik,smk->smi", cell_factor_np, model_factor_np)
    interaction_var_per_sample = interaction_s.var(axis=1)   # (S, I)
    interaction_var_cells = interaction_var_per_sample.mean(axis=0)  # (I,)
    total_model_uncertainty = global_model_var_scalar + interaction_var_cells  # (I,)

    cell_ids = (
        input_data[["cell_id"]]
        .drop_duplicates()
        .sort_values("cell_id")["cell_id"]
        .to_numpy()
    )
    # If lengths mismatch for some reason, fall back to simple range
    if len(cell_ids) != len(total_model_uncertainty):
        cell_ids = np.arange(len(total_model_uncertainty))

    df_model_uncertainty = pd.DataFrame({
        "cell_id": cell_ids,
        "model_var_global": global_model_var_scalar,
        "model_var_spatial": interaction_var_cells,
        "model_var_total": total_model_uncertainty,
    })
    # 1. Get unique cell-level geometry
    cells_geo = (
        input_data[["cell_id", "longitude", "latitude"]]   # <- change to your coord column names
        .drop_duplicates()
        .sort_values("cell_id")
    )

    # 2. Merge with model uncertainty DataFrame
    df_map = cells_geo.merge(df_model_uncertainty, on="cell_id", how="inner")
    return df_map


def compute_multiplicative_factors(H_samples):
    """
    Compute multiplicative lower/upper factors relative to the median level.
    """
    H_med = np.median(H_samples)
    H_low = np.percentile(H_samples, 5)
    H_high = np.percentile(H_samples, 95)

    f_low = H_low / H_med
    f_high = H_high / H_med

    return f_low, f_high, H_med, H_low, H_high

def rating_curve_level_samples(theta_hat, cov_theta, Q_T,
                               rel_Q_unc=0.2, n_samples=5000,
                               random_state=None):
    """
    Draw samples of water level H(Q_T) from:
      - rating-curve parameter uncertainty, and
      - additional relative uncertainty in Q_T (e.g. 20%).

    Parameters
    ----------
    theta_hat : array (4,)
        [alpha_hat, b_hat, h0_hat, sigma_hat]
    cov_theta : (4,4) array
        Covariance matrix corresponding to theta_hat.
    Q_T : float
        Best-estimate discharge for the T-year return period.
    rel_Q_unc : float
        Relative 1-sigma uncertainty in Q_T (e.g. 0.2 for 20%).
    n_samples : int
        Number of Monte Carlo samples.
    random_state : int or None
        Random seed.

    Returns
    -------
    H_samples : array (n_samples,)
        Sampled water levels at Q_T.
        Includes both parameter and Q_T uncertainty.
    """
    rng = np.random.default_rng(random_state)

    # sample parameters [alpha, b, h0, sigma]
    theta_samples = rng.multivariate_normal(theta_hat, cov_theta, size=n_samples)

    alpha_s = theta_samples[:, 0]
    b_s     = theta_samples[:, 1]
    h0_s    = theta_samples[:, 2]

    # base rating-curve levels at fixed Q_T (parameter uncertainty only)
    H_base = h0_s + (Q_T / np.exp(alpha_s)) ** (1.0 / b_s)

    # local derivative dH/dQ at each sample:
    # H - h0 = (Q / e^alpha)^(1/b)
    # dH/dQ = (H - h0) / (b * Q)
    dH_dQ = (H_base - h0_s) / (b_s * Q_T)

    # var(Q_T) = (rel_Q_unc * Q_T)^2
    sigma_Q = rel_Q_unc * Q_T

    # propagate Q-uncertainty: ΔH_Q ~ N(0, (dH/dQ * sigma_Q)^2)
    sigma_H_from_Q = np.abs(dH_dQ) * sigma_Q
    dH_Q = rng.normal(loc=0.0, scale=sigma_H_from_Q)

    # total H samples = base + extra variation from Q-uncertainty
    H_samples = H_base + dH_Q
    return H_samples
