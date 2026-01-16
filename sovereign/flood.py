# Script with functions and classes for flood risk processing

import rasterio
import xarray as xr
from lmoments3 import distr
from scipy.stats import gumbel_r, kstest
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from tqdm import tqdm

def combine_glofas(start, end, dir, area_filter):
    '''
    Function to combine glofas river discharge data into one xarray given a data directory with all the datasets in them
    as well as a start and end year for the desired discharge data. Also loads and clips the accumulating area dataset
    and masks the river discharge data by the specified upstream area threshold (area_filter)
    '''
    all_files = [os.path.join(dir, f"glofas_THA_{year}.grib") for year in range(start, end+1)] # if we do this for other countries will have to adjust filenames
    # Load all datasets into array
    datasets = [xr.open_dataset(file, engine='cfgrib') for file in all_files]
    # Concatenate all datasets along the time dimension
    combined_dataset = xr.concat(datasets, dim='time')
    # Make sure datasets are sorted by time
    combined_dataset = combined_dataset.sortby('time')
    # Load upstream area 
    upstream_area = xr.open_dataset(os.path.join(dir, "uparea_glofas_v4_0.nc"), engine='netcdf4') # might need to update the filename here
    # Get lat-lon limits from glofas data as will use this to clip the upstream area
    lat_limits = [combined_dataset.latitude.values[i] for i in [0, -1]]
    lon_limits = [combined_dataset.longitude.values[i] for i in [0, -1]]
    up_lats = upstream_area.latitude.values.tolist()
    up_lons = upstream_area.longitude.values.tolist()
    # Calculate slice indices
    lat_slice_index = [
    round((i-up_lats[0])/(up_lats[1]-up_lats[0]))
    for i in lat_limits
    ]
    lon_slice_index = [
        round((i-up_lons[0])/(up_lons[1]-up_lons[0]))
        for i in lon_limits
    ]
    # Slice upstream area to chosen glofas region
    red_upstream_area = upstream_area.isel(
        latitude=slice(lat_slice_index[0], lat_slice_index[1]+1),
        longitude=slice(lon_slice_index[0], lon_slice_index[1]+1),
    )
    # There are very minor rounding differences, so we update with the lat/lons from the glofas data
    red_upstream_area = red_upstream_area.assign_coords({
        'latitude': combined_dataset.latitude,
        'longitude': combined_dataset.longitude,
    })
    # Add the upstream area to the main data object and print the updated glofas data object:
    combined_dataset['uparea'] = red_upstream_area['uparea']
    # Mask the river discharge data
    combined_dataset_masked = combined_dataset.where(combined_dataset.uparea>=area_filter*1e6)


    return combined_dataset_masked

def extract_discharge_timeseries(outlets, discharge_data):
    '''
    function to extract discharge timeseries at basin outlet points. Returns a dictionary of timeseries with basin ID as key.
    '''

    # Dictionary to store timeseries data for each basin
    basin_timeseries = {}

    # Loop through basin outlets, storing each in turn
    for index, row in outlets.iterrows():
        basin_id = row['HYBAS_ID_L6']
        lat = row['Latitude']
        lon = row['Longitude']
        point_data = discharge_data.sel(latitude=lat, longitude=lon, method='nearest')
        timeseries = point_data['dis24'].to_series()
        # store in dictionary
        basin_timeseries[basin_id] = timeseries
    
    return basin_timeseries

def fit_gumbel_distribution(basin_timeseries):
    '''
    Calculate extreme value distribution to all the basin timeseries. This function calculates the gumbel distribution and performs 
    the Kolomgorov-Smirnov test to check for the quality of fit. Returns a dictionary that reports each basin's gumbel parameters as 
    well as D and p-value from the Kolmogorov-Smirnov test. 
    '''
    # Initiate dictionaries
    gumbel_params = {}
    fit_quality = {}

    # Loop through basins, calculating annual maxima and fitting Gumbel distribution using L-moments
    for basin_id, timeseries in basin_timeseries.items():
        annual_maxima = timeseries.groupby(timeseries.index.year).max()

        # Fit Gumbel distribution using L-moments
        params = distr.gum.lmom_fit(annual_maxima)

        # Perform the Kolmogorov-Smirnov test (checking quality of fit)
        D, p_value = kstest(annual_maxima, 'gumbel_r', args=(params['loc'], params['scale']))

        gumbel_params[basin_id] = params
        fit_quality[basin_id] = (D, p_value)

    
    return gumbel_params, fit_quality

def calculate_uniform_marginals(basin_timeseries, gumbel_parameters):
    '''
    This function will transform annual maximum values from the discharge timeseries into uniform marginals
    for each river basin using the Cumulative Distribution Function of the fitted Gumbel distribution.
    '''
    # Initialize dictionary for uniform marginals
    uniform_marginals = {}

    for basin_id, timeseries in basin_timeseries.items():
        params = gumbel_parameters[basin_id]
        annual_maxima = timeseries.groupby(timeseries.index.year).max()
        uniform_marginals[basin_id] = gumbel_r.cdf(annual_maxima, loc=params['loc'], scale=params['scale'])
    return uniform_marginals

def vectorized_damage(depth, value, heights, damage_percents):
    '''
    Vectorized damage function
    Apply damage function given a flood depth and exposure value.
    Function also needs as input the damage function heights > damage_percents
    '''
    # Use np.interp for vectorized linear interpolation
    damage_percentage = np.interp(depth, heights, damage_percents)
    return damage_percentage * value

def calculate_risk(flood, building_values, heights, damage_percents):
    '''
    Pass a flood depth array, array of values, and a vulnerability curve
    to calculate risk.
    '''
    exposure = np.where(flood>0, building_values, 0)
    risk = vectorized_damage(flood, exposure, heights, damage_percents)

    return risk

def simple_risk_overlay(flood_path, exposure_path, output_path, damage_function):
    '''
    This function performs a simple risk overlay analysis.
    It takes as input a flood map, an exposure map, and a vulnerability curve.
    It outputs a risk raster
    '''
    # Load the rasters
    flood = rasterio.open(flood_path)
    exposure = rasterio.open(exposure_path)

    # Data info
    profile = flood.meta.copy()
    profile.update(dtype=rasterio.float32, compress='lzw', nodata=0)
    nodata = flood.nodata

    with rasterio.open(output_path, 'w', **profile) as dst:
        i = 0
        for ji, window in flood.block_windows(1):
            i += 1

            affine = rasterio.windows.transform(window, flood.transform)
            height, width = rasterio.windows.shape(window)
            bbox = rasterio.windows.bounds(window, flood.transform)

            profile.update({
                'height': height,
                'width': width,
                'affine': affine
            })

            flood_array = flood.read(1, window=window)
            exposure_array = exposure.read(1, window=window)
            flood_array = np.where(flood_array>0, flood_array, 0) # remove negative values
            risk = calculate_risk(flood_array, exposure_array, damage_function[0], damage_function[1]) # depths index 0 and prp damage index 1

            dst.write(risk.astype(rasterio.float32), window=window, indexes=1)


def flopros_risk_overlay(flood_path, exposure_path, output_path, mask_path, damage_function):
    '''
    This function performs a risk overlay analysis. Before the risk analysis it masks all urban areas in the exposure dataset
    It takes as input a flood map, an exposure map, an urban area mask map, and a vulnerability curve.
    It outputs a risk raster
    '''
    # Load the rasters
    flood = rasterio.open(flood_path)
    exposure = rasterio.open(exposure_path)
    mask = rasterio.open(mask_path)

    # Data info
    profile = flood.meta.copy()
    profile.update(dtype=rasterio.float32, compress='lzw', nodata=0)
    nodata = flood.nodata

    with rasterio.open(output_path, 'w', **profile) as dst:
        i = 0
        for ji, window in flood.block_windows(1):
            i += 1

            affine = rasterio.windows.transform(window, flood.transform)
            height, width = rasterio.windows.shape(window)
            bbox = rasterio.windows.bounds(window, flood.transform)

            profile.update({
                'height': height,
                'width': width,
                'affine': affine
            })

            flood_array = flood.read(1, window=window)
            exposure_array = exposure.read(1, window=window)
            mask_array = mask.read(1, window=window)
            exposure_array = np.where(mask_array==1, 0, exposure_array) # wherever the urban mask equals 1, set to zero in exposure dataset
            flood_array = np.where(flood_array>0, flood_array, 0) # remove negative values
            risk = calculate_risk(flood_array, exposure_array, damage_function[0], damage_function[1]) # depths index 0 and prp damage index 1

            dst.write(risk.astype(rasterio.float32), window=window, indexes=1)


@dataclass
class BasinComponent:
    admin_id: str              # GID_1 or similar
    aeps: np.ndarray            # annual exceedance probabilities
    sector: str                # sector name for flood loss curve
    baseline_losses: np.ndarray # baseline flood losses
    adapted_losses: np.ndarray # adapted flood losses (urban areas masked)
    protection_aep: float       # New_Pr_L or Pr_L
    meta: dict = None

    def baseline_loss_at(self, aep_event: float) -> float:
        return float(np.interp(aep_event, self.aeps, self.baseline_losses))

    def adapted_loss_at(self, aep_event: float) -> float:
        return float(np.interp(aep_event, self.aeps, self.adapted_losses))

    def protected_loss(self, aep_event: float) -> float:
        """Baseline scenario: apply baseline protection only."""
        if aep_event > self.protection_aep:  # protected
            return 0.0
        else:
            return self.baseline_loss_at(aep_event)

    def adapted_loss(self, aep_event:float, adapted_protection_aep: float) -> float:
        """
        Adaptation scenario combining:
        - baseline protection_aep (p_base)
        - stronger adapted_protection_aep (p_adapt)
        """
        p_base = self.protection_aep
        p_adapt = adapted_protection_aep

        # Safety check
        if p_adapt > p_base:
            p_adapt=p_base

        if aep_event > p_base:
            # Baseline protection - no risk
            return 0.0
        elif p_adapt < aep_event < p_base:
            # Above baseline protection but below adapted protection - sample adapted curve
            return self.adapted_loss_at(aep_event)
        else:
            # Above both baseline and adapted protection - sample baseline curve
            return self.baseline_loss_at(aep_event)
        
def build_basin_curves(df: pd.DataFrame):
    basin_dict = {}

    # group by basin, admin, AND sector
    for (basin_id, admin_id, sector), g in df.groupby(["HB_L6", "GID_1", "Sector"]):
        aeps = g["AEP"].to_numpy()
        losses = g["damages"].to_numpy()
        adapted_losses = g["adapted_damages"].to_numpy()

        # Add bankfull 2-year point
        aeps = np.concatenate(([0.5], aeps))
        losses = np.concatenate(([0.0], losses))
        adapted_losses = np.concatenate(([0.0], adapted_losses))

        # sort by AEP ascending for np.interp
        order = np.argsort(aeps)
        aeps = aeps[order]
        losses = losses[order]
        adapted_losses = adapted_losses[order]

        prot = g["Pr_L_AEP"].iloc[0]

        component = BasinComponent(
            admin_id=admin_id,
            sector=sector,
            aeps=aeps,
            baseline_losses=losses,
            adapted_losses=adapted_losses,
            protection_aep=prot,
        )

        basin_dict.setdefault(basin_id, []).append(component)

    return {
        basin_id: BasinLossCurve(basin_id=basin_id, components=components)
        for basin_id, components in basin_dict.items()
    }

@dataclass
class BasinLossCurve:
    basin_id: int
    components: list[BasinComponent]

    def loss_at_event_aep(
        self,
        aep_event: float,
        scenario: str = "baseline",
        adapted_protection_aep: Optional[float] = None,
        sector: Optional[float] = None,
    ) -> float:
        # filter components if sector is specified
        comps = (
            [c for c in self.components if c.sector == sector]
            if sector is not None else
            self.components
        )

        if scenario == "baseline" or adapted_protection_aep is None:
            return sum(c.protected_loss(aep_event) for c in comps)
        elif scenario == "adaptation":
            return sum(c.adapted_loss(aep_event, adapted_protection_aep) for c in comps)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
    
    
def make_uncertainty_curve(values, sds, k=1):
    """Return low and high curves using ±k standard deviations with 0–1 bounds."""
    values = np.array(values)
    sds = np.array(sds)

    low = np.clip(values - k * sds, 0, 1)
    high = np.clip(values + k * sds, 0, 1)

    return low.tolist(), high.tolist()

def risk_data_future_shift(risk_data, future_data, hydro_model, scenario, epoch, stat, degrade_protection=True):
    """
    Function for converting the risk dataframe to reflect future climate shifts
    
    :param risk_data: dataframe with baseline risk data
    :param future_data: datafrane with future climate shift data
    :param hydro_model: hydrological model to filter by ()
    :param scenario: climate scenario to filter by
    :param epoch: future epoch of interest
    :param stat: stat to filter by (e.g. 'mean', 'p10', 'p90')
    :param degrade_protection: whether to degrade protection levels in future (default: True) e.g. 100-year protection becomes 50-year protection if RP changes accordingly 
    """
    # Filter future data
    future_sub = future_data[
        (future_data['hydro'] == hydro_model) &
        (future_data['climate_scenario'] == scenario) &
        (future_data['period'] == epoch) &
        (future_data['stat'] == stat)
    ].copy()

    if future_sub.empty:
        raise ValueError("No future data found for the specified filters.")

    future_sub = future_sub[['HB_L6', 'return_period', 'new_rp_value']] # keep only relevant columns

    # Merge onto baseline risk data
    risk_data_future = risk_data.copy()
    risk_data_future = risk_data_future.merge(future_sub.rename(columns={'return_period': 'RP', 'new_rp_value': 'RP_future'}),
        on=['HB_L6', 'RP'], how='left')
    
    # User RP_future where available, else keep original RP
    rp_eff = np.where(
        risk_data_future['RP_future'].notnull(),
        risk_data_future['RP_future'],
        risk_data_future['RP'])
    
    # Add new AEP column
    risk_data_future['AEP'] = 1.0 / rp_eff

    # Optional: degrade protection with future climate change shifts
    if degrade_protection:
        # Build per-basin RP mapping: baseline_rps → future_rps
        climate_shifts: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for basin_id, grp in future_sub.groupby('HB_L6'):
            grp = grp.sort_values('return_period')
            baseline_rps = grp['return_period'].to_numpy()
            future_rps = grp['new_rp_value'].to_numpy()
            climate_shifts[basin_id] = (baseline_rps, future_rps)

        def shift_protection_aep(row):
            basin_id = row["HB_L6"]
            prot_rp  = row["Pr_L"]  # baseline protection RP

            # No protection or no mapping? keep baseline AEP
            if prot_rp == 0 or basin_id not in climate_shifts:
                return row['Pr_L'], row["Pr_L_AEP"]

            base_rps, fut_rps = climate_shifts[basin_id]

            # Map baseline protection RP → future effective RP (interpolate using the new loss probability curves)
            prot_rp_future = np.interp(
                prot_rp,
                base_rps,
                fut_rps,
                left=fut_rps[0],
                right=fut_rps[-1],
            )

            return prot_rp_future, 1.0 / prot_rp_future
        
        future_Pr_L, future_Pr_L_AEP = zip(*risk_data_future.apply(shift_protection_aep, axis=1)) 
        
        risk_data_future['Pr_L'] = future_Pr_L
        risk_data_future['Pr_L_AEP'] = future_Pr_L_AEP
        
    return risk_data_future

def run_simulation(basin_curves, n_years, adaptation_aep, copula_numbers):
    """
    Function for running the monte carlo flood simulation
    
    :param basin_curves: the basin curves dictionary (baseline or future)
    :param n_years: number of years to simulate
    :param adaptation_aep: the protection level of the adaptation scenario
    :param copula_numbers: dataframe of random numbers from the copula for each basin and year
    """

    # Extract all sectors from the basin curves
    all_sectors = {comp.sector for curve in basin_curves.values() for comp in curve.components}
    # Extract all basin IDs
    basin_ids = list(basin_curves.keys())

    sector_baseline_losses = {s: np.zeros(n_years) for s in all_sectors}
    sector_adapted_losses  = {s: np.zeros(n_years) for s in all_sectors}

    for t in tqdm(range(n_years)):
        sector_year_baseline = {s: 0.0 for s in all_sectors}
        sector_year_adapted  = {s: 0.0 for s in all_sectors}
        random_ns = copula_numbers.loc[t]

        for basin_id in basin_ids:
            basin_str = str(int(basin_id))
            if basin_str not in random_ns:
                continue

            curve = basin_curves[basin_id]
            aep_event = 1 - random_ns[basin_str]

            for s in all_sectors:
                bl = curve.loss_at_event_aep(aep_event, sector=s)
                ad = curve.loss_at_event_aep(
                    aep_event,
                    scenario="adaptation",
                    adapted_protection_aep=adaptation_aep,
                    sector=s,
                )
                sector_year_baseline[s] += bl
                sector_year_adapted[s]  += ad

        for s in all_sectors:
            sector_baseline_losses[s][t] = sector_year_baseline[s]
            sector_adapted_losses[s][t]  = sector_year_adapted[s]

    return sector_baseline_losses, sector_adapted_losses


def extract_sectoral_losses(loss_dict, n_years):
    """
    Function for extracting sectoral losses from the loss dictionary
    :param loss_dict: dictionary of losses per sector
    :param n_years: number of years simulated
    """
    gva_sectors = ['Agriculture', 'Manufacturing', 'Service']
    cap_sectors = ['Public', 'Private']

    # GVA losses (sum of GVA sectors)
    gva_losses = sum(loss_dict[s] for s in gva_sectors)
    # Capital stock damage (sum of capital sectors)
    cap_damage = sum(loss_dict[s] for s in cap_sectors)

    # Store in dataframe and return
    losses_df = pd.DataFrame({
        "year_index": np.arange(n_years),
        'GVA_loss': gva_losses,
        'CAP_dam': cap_damage,
        "AGR_loss": loss_dict['Agriculture'],
        "MAN_loss": loss_dict['Manufacturing'],
        "SER_loss": loss_dict['Service'],
        "PUB_dam": loss_dict['Public'],
        "PRI_dam": loss_dict['Private']
    })

    return losses_df