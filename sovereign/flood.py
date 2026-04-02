# Script with functions and classes for flood risk processing

import rasterio
import xarray as xr
from lmoments3 import distr
from scipy.stats import gumbel_r, kstest
import os
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional
import pandas as pd
from tqdm import tqdm
import warnings

def combine_glofas(start, end, dir, area_filter):
    '''
    Function to combine glofas river discharge data into one xarray given a data directory with all the datasets in them
    as well as a start and end year for the desired discharge data. Also loads and clips the accumulating area dataset
    and masks the river discharge data by the specified upstream area threshold (area_filter)
    '''
    all_files = [os.path.join(dir, f"glofas_UGA_{year}.grib") for year in range(start, end+1)] # if we do this for other countries will have to adjust filenames
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


def masked_vulnerability_overlay(
    flood_path,
    exposure_path,
    output_path,
    mask_path,
    baseline_damage_function,
    adapted_damage_function,
    mask_value=1,
):
    """
    Compute a risk raster using the baseline vulnerability curve outside the
    adaptation mask and the adapted vulnerability curve inside the mask.

    This is intended for raster-explicit vulnerability adaptation tests.
    """
    flood = rasterio.open(flood_path)
    exposure = rasterio.open(exposure_path)
    mask = rasterio.open(mask_path)

    profile = flood.meta.copy()
    profile.update(dtype=rasterio.float32, compress='lzw', nodata=0)

    with rasterio.open(output_path, 'w', **profile) as dst:
        for _, window in flood.block_windows(1):
            flood_array = flood.read(1, window=window)
            exposure_array = exposure.read(1, window=window)
            mask_array = mask.read(1, window=window)

            flood_array = np.where(flood_array > 0, flood_array, 0)

            baseline_risk = calculate_risk(
                flood_array,
                exposure_array,
                baseline_damage_function[0],
                baseline_damage_function[1],
            )
            adapted_risk = calculate_risk(
                flood_array,
                exposure_array,
                adapted_damage_function[0],
                adapted_damage_function[1],
            )

            risk = np.where(mask_array == mask_value, adapted_risk, baseline_risk)
            dst.write(risk.astype(rasterio.float32), window=window, indexes=1)


def build_masked_vulnerability_raster_series(
    flood_map_lookup: dict[int, str],
    flood_dir: str,
    exposure_path: str,
    output_dir: str,
    output_name_template: str,
    mask_path: str,
    baseline_damage_function,
    adapted_damage_function,
    mask_value: int = 1,
) -> dict[int, str]:
    """
    Build a family of adapted risk rasters by applying the adapted vulnerability
    curve only within the mask.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_paths = {}

    for rp, flood_name in flood_map_lookup.items():
        flood_path = os.path.join(flood_dir, flood_name)
        output_path = os.path.join(output_dir, output_name_template.format(rp=rp))
        masked_vulnerability_overlay(
            flood_path=flood_path,
            exposure_path=exposure_path,
            output_path=output_path,
            mask_path=mask_path,
            baseline_damage_function=baseline_damage_function,
            adapted_damage_function=adapted_damage_function,
            mask_value=mask_value,
        )
        output_paths[rp] = output_path

    return output_paths


def combine_raster_series(
    raster_series_list: list[dict[int, str]],
    output_dir: str,
    output_name_template: str,
) -> dict[int, str]:
    """
    Sum multiple raster series, keyed by return period, into one output series.
    """
    from sovereign.utils import sum_rasters

    if not raster_series_list:
        raise ValueError("At least one raster series is required.")

    os.makedirs(output_dir, exist_ok=True)
    rps = sorted(raster_series_list[0].keys())
    output_paths = {}

    for rp in rps:
        raster_list = [series[rp] for series in raster_series_list]
        output_path = os.path.join(output_dir, output_name_template.format(rp=rp))
        sum_rasters(raster_list, output_path)
        output_paths[rp] = output_path

    return output_paths


def aggregate_raster_series_to_basin_risk(
    basin_path: str,
    raster_paths_by_rp: dict[int, str],
    sector_label: str,
    basin_id_col: str = "HYBAS_ID_06",
    admin_id_col: str = "flpr_gid_1",
    name_col: str = "NAME",
    protection_col: str = "MerL_Riv",
) -> pd.DataFrame:
    """
    Aggregate a raster series to basin-level risk rows for one sector.
    """
    import geopandas as gpd
    from rasterstats import zonal_stats

    basin_df = gpd.read_file(basin_path)
    results = []

    for rp, raster_path in raster_paths_by_rp.items():
        with rasterio.open(raster_path) as src:
            raster = src.read(1)
            transform = src.transform

        zs = zonal_stats(
            basin_df,
            raster,
            affine=transform,
            stats="sum",
            geojson_out=True,
        )

        temp_df = pd.DataFrame({
            "FID": [feat["id"] for feat in zs],
            "GID_1": [feat["properties"][admin_id_col] for feat in zs],
            "NAME": [feat["properties"][name_col] for feat in zs],
            "HB_L6": [feat["properties"][basin_id_col] for feat in zs],
            "Pr_L": [feat["properties"][protection_col] for feat in zs],
            "damages": [feat["properties"]["sum"] for feat in zs],
            "RP": rp,
            "Sector": sector_label,
        })
        temp_df["damages"] = temp_df["damages"].fillna(0)
        results.append(temp_df)

    risk_df = pd.concat(results, ignore_index=True)
    risk_df["AEP"] = 1.0 / risk_df["RP"]
    risk_df["Pr_L_AEP"] = np.where(risk_df["Pr_L"] == 0, 0, 1.0 / risk_df["Pr_L"])
    return risk_df


def build_masked_vulnerability_scenario_curves(
    baseline_risk_df: pd.DataFrame,
    basin_path: str,
    flood_map_lookup: dict[int, str],
    flood_dir: str,
    target_exposure_path: str,
    mask_path: str,
    baseline_damage_function,
    adapted_damage_function,
    target_sector_label: str,
    output_dir: str,
    adapted_component_name_template: str,
    combined_sector_name_template: Optional[str] = None,
    baseline_component_paths_by_rp: Optional[dict[int, str]] = None,
    mask_value: int = 1,
) -> tuple[dict[int, "BasinLossCurve"], pd.DataFrame, dict[int, str]]:
    """
    End-to-end reusable workflow for raster-explicit vulnerability adaptation.

    Steps:
    1. Build adapted target-component rasters using the mask
    2. Optionally combine them with baseline component rasters to form sector totals
    3. Aggregate the resulting rasters to basin risk rows for the targeted sector
    4. Overlay those rows onto the full baseline basin risk table
    5. Build vulnerability scenario curves
    """
    adapted_component_paths = build_masked_vulnerability_raster_series(
        flood_map_lookup=flood_map_lookup,
        flood_dir=flood_dir,
        exposure_path=target_exposure_path,
        output_dir=output_dir,
        output_name_template=adapted_component_name_template,
        mask_path=mask_path,
        baseline_damage_function=baseline_damage_function,
        adapted_damage_function=adapted_damage_function,
        mask_value=mask_value,
    )

    final_raster_paths = adapted_component_paths
    if baseline_component_paths_by_rp is not None:
        if combined_sector_name_template is None:
            raise ValueError(
                "combined_sector_name_template is required when combining adapted "
                "component rasters with baseline component rasters."
            )
        final_raster_paths = combine_raster_series(
            raster_series_list=[adapted_component_paths, baseline_component_paths_by_rp],
            output_dir=output_dir,
            output_name_template=combined_sector_name_template,
        )

    adapted_risk_df = aggregate_raster_series_to_basin_risk(
        basin_path=basin_path,
        raster_paths_by_rp=final_raster_paths,
        sector_label=target_sector_label,
    )

    baseline_df = _ensure_risk_columns(baseline_risk_df)
    merge_keys = _standard_risk_merge_keys(baseline_df)
    adapted_subset = adapted_risk_df[merge_keys + ["damages"]].rename(
        columns={"damages": "__adapted_damages"}
    )
    scenario_risk_df = baseline_df.merge(adapted_subset, on=merge_keys, how="left")
    scenario_risk_df["damages"] = np.where(
        scenario_risk_df["__adapted_damages"].notnull(),
        scenario_risk_df["__adapted_damages"],
        scenario_risk_df["damages"],
    )
    scenario_risk_df.drop(columns="__adapted_damages", inplace=True)
    scenario_risk_df["component_type"] = "vulnerability_adapted"
    if "exposure_share" not in scenario_risk_df.columns:
        scenario_risk_df["exposure_share"] = 1.0

    scenario_curves = build_vulnerability_scenario_curves(
        baseline_risk_df=baseline_risk_df,
        adapted_risk_df=adapted_risk_df,
    )
    return scenario_curves, scenario_risk_df, final_raster_paths


@dataclass
class BasinComponent:
    admin_id: str              # GID_1 or similar
    aeps: np.ndarray            # annual exceedance probabilities
    sector: str                # sector name for flood loss curve
    losses: np.ndarray          # scenario-specific flood losses
    protection_aep: float       # protection AEP for this component/scenario
    legacy_adapted_losses: Optional[np.ndarray] = None  # compatibility path for old protection adaptation
    event_aep_min: Optional[float] = None  # active only for events greater than this AEP
    event_aep_max: Optional[float] = None  # active only for events up to and including this AEP
    exposure_share: float = 1.0
    component_type: str = "baseline"
    meta: dict = None

    def loss_at(self, aep_event: float) -> float:
        """Return loss for this component at the sampled event AEP."""
        if self.event_aep_min is not None and aep_event <= self.event_aep_min:
            return 0.0
        if self.event_aep_max is not None and aep_event > self.event_aep_max:
            return 0.0
        if self.protection_aep > 0 and aep_event > self.protection_aep:
            return 0.0
        return float(np.interp(aep_event, self.aeps, self.losses))

    def baseline_loss_at(self, aep_event: float) -> float:
        """Compatibility alias for the baseline curve on this component."""
        return self.loss_at(aep_event)

    def adapted_loss_at(self, aep_event: float) -> float:
        """Compatibility accessor for legacy protection-based adaptation data."""
        if self.legacy_adapted_losses is None:
            return self.loss_at(aep_event)
        return float(np.interp(aep_event, self.aeps, self.legacy_adapted_losses))

    def protected_loss(self, aep_event: float) -> float:
        """Compatibility alias for the scenario-specific protected loss."""
        return self.loss_at(aep_event)

    def adapted_loss(self, aep_event:float, adapted_protection_aep: float) -> float:
        """
        Compatibility path for the legacy protection adaptation workflow.
        """
        if self.legacy_adapted_losses is None:
            return self.loss_at(aep_event)

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
    loss_col = "damages"
    group_cols = ["HB_L6", "GID_1", "Sector"]
    has_component_type = "component_type" in df.columns

    if has_component_type:
        group_cols.append("component_type")

    # group by basin, admin, AND sector
    for group_key, g in df.groupby(group_cols):
        if has_component_type:
            basin_id, admin_id, sector, component_type = group_key
        else:
            basin_id, admin_id, sector = group_key
            component_type = "baseline"

        aeps = g["AEP"].to_numpy()
        losses = g[loss_col].to_numpy()
        adapted_losses = g["adapted_damages"].to_numpy() if "adapted_damages" in g.columns else None

        # Add bankfull 2-year point
        aeps = np.concatenate(([0.5], aeps))
        losses = np.concatenate(([0.0], losses))
        if adapted_losses is not None:
            adapted_losses = np.concatenate(([0.0], adapted_losses))

        # sort by AEP ascending for np.interp
        order = np.argsort(aeps)
        aeps = aeps[order]
        losses = losses[order]
        if adapted_losses is not None:
            adapted_losses = adapted_losses[order]

        prot = g["Pr_L_AEP"].iloc[0]
        event_aep_min = g["event_aep_min"].iloc[0] if "event_aep_min" in g.columns else None
        event_aep_max = g["event_aep_max"].iloc[0] if "event_aep_max" in g.columns else None
        exposure_share = g["exposure_share"].iloc[0] if "exposure_share" in g.columns else 1.0

        component = BasinComponent(
            admin_id=admin_id,
            sector=sector,
            aeps=aeps,
            losses=losses,
            protection_aep=prot,
            legacy_adapted_losses=adapted_losses,
            event_aep_min=event_aep_min if pd.notna(event_aep_min) else None,
            event_aep_max=event_aep_max if pd.notna(event_aep_max) else None,
            exposure_share=exposure_share,
            component_type=component_type,
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
            return sum(c.loss_at(aep_event) for c in comps)
        if scenario == "adaptation":
            return sum(c.adapted_loss(aep_event, adapted_protection_aep) for c in comps)
        raise ValueError(f"Unknown scenario: {scenario}")


def build_protection_scenario_from_baseline(baseline_curves, adapted_protection_aep):
    """
    Build scenario curves from baseline curves using the legacy protection-based
    adaptation semantics.
    """
    scenario_curves = {}

    for basin_id, curve in baseline_curves.items():
        scenario_components = []

        for component in curve.components:
            if component.legacy_adapted_losses is None:
                scenario_components.append(
                    BasinComponent(
                        admin_id=component.admin_id,
                        sector=component.sector,
                        aeps=component.aeps.copy(),
                        losses=component.losses.copy(),
                        protection_aep=component.protection_aep,
                        event_aep_min=component.event_aep_min,
                        event_aep_max=component.event_aep_max,
                        exposure_share=component.exposure_share,
                        component_type=component.component_type,
                        meta=component.meta.copy() if isinstance(component.meta, dict) else component.meta,
                    )
                )
                continue

            p_base = component.protection_aep
            p_adapt = adapted_protection_aep
            if p_adapt > p_base:
                p_adapt = p_base

            scenario_components.append(
                BasinComponent(
                    admin_id=component.admin_id,
                    sector=component.sector,
                    aeps=component.aeps.copy(),
                    losses=component.losses.copy(),
                    protection_aep=p_base,
                    event_aep_max=p_adapt,
                    exposure_share=component.exposure_share,
                    component_type="legacy_baseline_window",
                    meta=component.meta.copy() if isinstance(component.meta, dict) else component.meta,
                )
            )
            scenario_components.append(
                BasinComponent(
                    admin_id=component.admin_id,
                    sector=component.sector,
                    aeps=component.aeps.copy(),
                    losses=component.legacy_adapted_losses.copy(),
                    protection_aep=p_base,
                    event_aep_min=p_adapt,
                    event_aep_max=p_base,
                    exposure_share=component.exposure_share,
                    component_type="legacy_adapted_window",
                    meta=component.meta.copy() if isinstance(component.meta, dict) else component.meta,
                )
            )

        scenario_curves[basin_id] = BasinLossCurve(basin_id=basin_id, components=scenario_components)

    return scenario_curves


def _ensure_risk_columns(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the core probability/protection columns exist before curve building.
    """
    df = risk_df.copy()

    if "AEP" not in df.columns:
        if "RP" not in df.columns:
            raise ValueError("Risk dataframe must contain either `AEP` or `RP`.")
        df["AEP"] = 1.0 / df["RP"]

    if "Pr_L_AEP" not in df.columns:
        if "Pr_L" not in df.columns:
            raise ValueError("Risk dataframe must contain either `Pr_L_AEP` or `Pr_L`.")
        df["Pr_L_AEP"] = np.where(df["Pr_L"] == 0, 0, 1.0 / df["Pr_L"])

    return df


def _standard_risk_merge_keys(df: pd.DataFrame) -> list[str]:
    merge_keys = ["HB_L6", "GID_1", "Sector"]
    if "RP" in df.columns:
        merge_keys.append("RP")
    elif "AEP" in df.columns:
        merge_keys.append("AEP")
    else:
        raise ValueError("Risk dataframe must contain either `RP` or `AEP` for alignment.")
    return merge_keys


def _copy_curve_metadata(source_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Pr_L", "Pr_L_AEP", "AEP", "RP"]:
        if col in source_df.columns and col not in target_df.columns:
            target_df[col] = source_df[col]
    return target_df


def build_protection_scenario_curves(
    baseline_risk_df: pd.DataFrame,
    adapted_risk_df: pd.DataFrame,
    adapted_protection_aep: float,
) -> dict[int, BasinLossCurve]:
    """
    Build protection scenario curves using raster-derived baseline and adapted
    basin loss tables.

    This preserves the current geospatial protection logic:
    - baseline losses below the adapted protection threshold
    - adapted losses between the adapted and baseline protection thresholds
    - zero losses above the baseline protection threshold
    """
    scenario_df = build_protection_scenario_risk_data(
        baseline_risk_df=baseline_risk_df,
        adapted_risk_df=adapted_risk_df,
        adapted_protection_aep=adapted_protection_aep,
    )
    return build_basin_curves(scenario_df)


def build_protection_scenario_risk_data(
    baseline_risk_df: pd.DataFrame,
    adapted_risk_df: pd.DataFrame,
    adapted_protection_aep: float,
) -> pd.DataFrame:
    """
    Build a scenario basin-risk dataframe for raster-derived protection
    adaptation.

    The output dataframe is intended for saving to CSV and later reconstructing
    with `build_basin_curves(...)` in the generic scenario analysis notebooks.
    """
    baseline_df = _ensure_risk_columns(baseline_risk_df)
    adapted_df = _ensure_risk_columns(adapted_risk_df)
    merge_keys = _standard_risk_merge_keys(baseline_df)

    adapted_loss_col = "__scenario_adapted_damages"
    adapted_loss_cols = merge_keys + ["damages"]
    merged = baseline_df.merge(
        adapted_df[adapted_loss_cols].rename(columns={"damages": adapted_loss_col}),
        on=merge_keys,
        how="left",
    )
    merged[adapted_loss_col] = merged[adapted_loss_col].fillna(merged["damages"])

    switch_aep = np.where(
        adapted_protection_aep > merged["Pr_L_AEP"],
        merged["Pr_L_AEP"],
        adapted_protection_aep,
    )

    baseline_window = merged.copy()
    baseline_window["component_type"] = "protection_baseline_window"
    baseline_window["event_aep_min"] = pd.NA
    baseline_window["event_aep_max"] = switch_aep

    adapted_window = merged.copy()
    adapted_window["damages"] = adapted_window[adapted_loss_col]
    adapted_window["component_type"] = "protection_adapted_window"
    adapted_window["event_aep_min"] = switch_aep
    adapted_window["event_aep_max"] = adapted_window["Pr_L_AEP"]

    scenario_df = pd.concat([baseline_window, adapted_window], ignore_index=True)
    scenario_df.drop(columns=[adapted_loss_col], inplace=True, errors="ignore")
    if "exposure_share" not in scenario_df.columns:
        scenario_df["exposure_share"] = 1.0
    return scenario_df


def build_vulnerability_scenario_curves(
    baseline_risk_df: pd.DataFrame,
    adapted_risk_df: pd.DataFrame,
) -> dict[int, BasinLossCurve]:
    """
    Build scenario curves for raster-derived vulnerability adaptation.

    `adapted_risk_df` is expected to come from raster recomputation where cells
    flagged as adapted use the modified vulnerability curve.
    """
    baseline_df = _ensure_risk_columns(baseline_risk_df)
    adapted_df = _ensure_risk_columns(adapted_risk_df)

    merge_keys = _standard_risk_merge_keys(baseline_df)
    required_cols = merge_keys + ["damages"]
    missing = [c for c in required_cols if c not in adapted_df.columns]
    if missing:
        raise ValueError(f"Adapted vulnerability risk dataframe is missing columns: {missing}")

    adapted_subset = adapted_df[required_cols].rename(columns={"damages": "__adapted_damages"})
    scenario_df = baseline_df.merge(adapted_subset, on=merge_keys, how="left")
    scenario_df["damages"] = np.where(
        scenario_df["__adapted_damages"].notnull(),
        scenario_df["__adapted_damages"],
        scenario_df["damages"],
    )
    scenario_df.drop(columns="__adapted_damages", inplace=True)
    scenario_df["component_type"] = "vulnerability_adapted"
    if "exposure_share" not in scenario_df.columns:
        scenario_df["exposure_share"] = 1.0
    return build_basin_curves(scenario_df)


def _normalize_frequency_shift_table(frequency_shift_df: pd.DataFrame) -> pd.DataFrame:
    shift_df = frequency_shift_df.copy()

    if {"RP", "RP_future"}.issubset(shift_df.columns):
        shift_df["AEP"] = 1.0 / shift_df["RP"]
        shift_df["AEP_future"] = 1.0 / shift_df["RP_future"]
        return shift_df

    if {"AEP", "AEP_future"}.issubset(shift_df.columns):
        shift_df["RP"] = 1.0 / shift_df["AEP"]
        shift_df["RP_future"] = 1.0 / shift_df["AEP_future"]
        return shift_df

    raise ValueError(
        "Frequency shift dataframe must contain either (`RP`, `RP_future`) "
        "or (`AEP`, `AEP_future`)."
    )


def build_uniform_frequency_shift_table(
    return_periods,
    shift_factor: float,
) -> pd.DataFrame:
    """
    Build a simple RP-to-RP shift table using one multiplicative factor.

    Example:
    - shift_factor > 1 increases effective return periods and reduces frequency
    - shift_factor < 1 decreases effective return periods and increases frequency
    """
    rp = np.array(sorted(set(return_periods)), dtype=float)
    if np.any(rp <= 0):
        raise ValueError("Return periods must be positive.")
    if shift_factor <= 0:
        raise ValueError("shift_factor must be positive.")

    return pd.DataFrame({
        "RP": rp,
        "RP_future": rp * shift_factor,
    })


def apply_basin_frequency_shift(
    risk_df: pd.DataFrame,
    frequency_shift_df: pd.DataFrame,
    basin_ids: Optional[list[int]] = None,
    degrade_protection: bool = False,
) -> pd.DataFrame:
    """
    Apply a basin-level frequency adjustment by remapping the AEP/RP axis of the
    basin loss curves.
    """
    baseline_df = _ensure_risk_columns(risk_df)
    shift_df = _normalize_frequency_shift_table(frequency_shift_df)

    if basin_ids is not None:
        basin_id_set = set(basin_ids)
        baseline_df = baseline_df.copy()
        shift_df = shift_df.copy()
        baseline_df["_apply_shift"] = baseline_df["HB_L6"].isin(basin_id_set)
        if "HB_L6" not in shift_df.columns:
            shift_df = pd.concat(
                [shift_df.assign(HB_L6=basin_id) for basin_id in sorted(basin_id_set)],
                ignore_index=True,
            )
    else:
        baseline_df = baseline_df.copy()
        baseline_df["_apply_shift"] = True

    merge_keys = ["RP"]
    if "HB_L6" in shift_df.columns:
        merge_keys = ["HB_L6", "RP"]

    shifted = baseline_df.merge(
        shift_df[merge_keys + ["RP_future", "AEP_future"]],
        on=merge_keys,
        how="left",
    )

    shifted["AEP"] = np.where(
        shifted["_apply_shift"] & shifted["AEP_future"].notnull(),
        shifted["AEP_future"],
        shifted["AEP"],
    )
    shifted["RP"] = np.where(
        shifted["_apply_shift"] & shifted["RP_future"].notnull(),
        shifted["RP_future"],
        shifted["RP"],
    )

    if degrade_protection:
        shift_map = shift_df[merge_keys + ["RP_future"]].copy()

        def shift_row_protection(row):
            if not row["_apply_shift"] or row["Pr_L"] == 0:
                return row["Pr_L"], row["Pr_L_AEP"]

            if "HB_L6" in shift_map.columns:
                basin_shift = shift_map[shift_map["HB_L6"] == row["HB_L6"]].sort_values("RP")
            else:
                basin_shift = shift_map.sort_values("RP")

            if basin_shift.empty:
                return row["Pr_L"], row["Pr_L_AEP"]

            prot_rp_future = np.interp(
                row["Pr_L"],
                basin_shift["RP"].to_numpy(),
                basin_shift["RP_future"].to_numpy(),
                left=basin_shift["RP_future"].iloc[0],
                right=basin_shift["RP_future"].iloc[-1],
            )
            return prot_rp_future, 1.0 / prot_rp_future

        future_Pr_L, future_Pr_L_AEP = zip(*shifted.apply(shift_row_protection, axis=1))
        shifted["Pr_L"] = future_Pr_L
        shifted["Pr_L_AEP"] = future_Pr_L_AEP

    shifted["component_type"] = "frequency_shifted"
    shifted.drop(columns=[c for c in ["RP_future", "AEP_future", "_apply_shift"] if c in shifted.columns], inplace=True)
    return shifted


def build_frequency_scenario_curves(
    baseline_risk_df: pd.DataFrame,
    frequency_shift_df: pd.DataFrame,
    basin_ids: Optional[list[int]] = None,
    degrade_protection: bool = False,
) -> dict[int, BasinLossCurve]:
    """
    Build basin curves for a basin-level frequency adaptation scenario.
    """
    shifted_risk_df = apply_basin_frequency_shift(
        baseline_risk_df,
        frequency_shift_df,
        basin_ids=basin_ids,
        degrade_protection=degrade_protection,
    )
    return build_basin_curves(shifted_risk_df)


@dataclass
class AdaptationSpec:
    """
    Specification for building an adaptation scenario from a baseline risk table.

    The current workflow expects basin-level coverage fractions for partial-area
    interventions. Raster-to-basin aggregation can be handled upstream and the
    resulting fractions passed in through `coverage`.
    """
    name: str
    mode: str  # protection, vulnerability, frequency
    value: Optional[float] = None
    target_sectors: Optional[list[str]] = None
    coverage: Optional[pd.DataFrame] = None
    lookup_table: Optional[pd.DataFrame] = None
    raster_path: Optional[str] = None
    value_unit: Optional[str] = None
    meta: Optional[dict[str, Any]] = None


def _clone_component_frame(df: pd.DataFrame) -> pd.DataFrame:
    scenario_df = df.copy()
    if "component_type" not in scenario_df.columns:
        scenario_df["component_type"] = "baseline"
    if "exposure_share" not in scenario_df.columns:
        scenario_df["exposure_share"] = 1.0
    if "scenario_name" not in scenario_df.columns:
        scenario_df["scenario_name"] = "baseline"
    if "adaptation_name" not in scenario_df.columns:
        scenario_df["adaptation_name"] = pd.NA
    if "event_aep_min" not in scenario_df.columns:
        scenario_df["event_aep_min"] = pd.NA
    if "event_aep_max" not in scenario_df.columns:
        scenario_df["event_aep_max"] = pd.NA
    return scenario_df


def build_baseline_risk_components(risk_df: pd.DataFrame, scenario_name: str = "baseline") -> pd.DataFrame:
    """
    Normalize the baseline risk dataframe into a component-based table that can
    be modified by scenario builders.
    """
    component_df = _clone_component_frame(risk_df)
    component_df["scenario_name"] = scenario_name
    component_df["component_type"] = "baseline"
    component_df["adaptation_name"] = pd.NA
    return component_df


def _coverage_to_dict(coverage: Optional[pd.DataFrame]) -> dict[Any, float]:
    if coverage is None:
        return {}
    required = {"HB_L6", "coverage_fraction"}
    missing = required.difference(coverage.columns)
    if missing:
        raise ValueError(f"Coverage table is missing required columns: {sorted(missing)}")

    cleaned = coverage[["HB_L6", "coverage_fraction"]].copy()
    cleaned["coverage_fraction"] = cleaned["coverage_fraction"].clip(lower=0.0, upper=1.0)
    return dict(zip(cleaned["HB_L6"], cleaned["coverage_fraction"]))


def _split_rows_by_coverage(
    df: pd.DataFrame,
    spec: AdaptationSpec,
    target_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage_map = _coverage_to_dict(spec.coverage)
    targeted_sectors = set(spec.target_sectors or [])
    base = _clone_component_frame(df)
    base["_coverage_fraction"] = base["HB_L6"].map(coverage_map).fillna(0.0)

    if target_only and targeted_sectors:
        targeted_mask = base["Sector"].isin(targeted_sectors)
    elif target_only:
        targeted_mask = pd.Series(True, index=base.index)
    else:
        targeted_mask = pd.Series(True, index=base.index)

    unaffected = base.copy()
    unaffected["exposure_share"] = unaffected["exposure_share"] * np.where(
        targeted_mask,
        1.0 - unaffected["_coverage_fraction"],
        1.0,
    )
    unaffected["damages"] = unaffected["damages"] * np.where(
        targeted_mask,
        1.0 - unaffected["_coverage_fraction"],
        1.0,
    )
    if "adapted_damages" in unaffected.columns:
        unaffected["adapted_damages"] = unaffected["adapted_damages"] * np.where(
            targeted_mask,
            1.0 - unaffected["_coverage_fraction"],
            1.0,
        )
    unaffected["component_type"] = np.where(
        targeted_mask & (unaffected["_coverage_fraction"] > 0),
        "unaffected_share",
        unaffected["component_type"],
    )

    affected = base.loc[targeted_mask & (base["_coverage_fraction"] > 0)].copy()
    affected["exposure_share"] = affected["exposure_share"] * affected["_coverage_fraction"]
    affected["damages"] = affected["damages"] * affected["_coverage_fraction"]
    if "adapted_damages" in affected.columns:
        affected["adapted_damages"] = affected["adapted_damages"] * affected["_coverage_fraction"]
    affected["scenario_name"] = spec.name
    affected["adaptation_name"] = spec.name

    unaffected = unaffected.loc[unaffected["exposure_share"] > 0].copy()
    unaffected.drop(columns="_coverage_fraction", inplace=True)
    affected.drop(columns="_coverage_fraction", inplace=True)
    return unaffected, affected


def apply_vulnerability_adaptation(risk_df: pd.DataFrame, spec: AdaptationSpec) -> pd.DataFrame:
    """
    Apply a vulnerability reduction to the targeted sectors within the covered
    basin share. This is currently implemented as a basin-share approximation.
    """
    warnings.warn(
        "apply_vulnerability_adaptation uses a basin-coverage approximation. "
        "Prefer raster-derived adapted risk tables with build_vulnerability_scenario_curves.",
        UserWarning,
        stacklevel=2,
    )
    if spec.value is None:
        raise ValueError("Vulnerability adaptation requires `value` as a fractional reduction.")

    unaffected, affected = _split_rows_by_coverage(risk_df, spec, target_only=True)
    reduction_fraction = float(spec.value)
    affected["damages"] = affected["damages"] * (1.0 - reduction_fraction)
    affected["component_type"] = "vulnerability_reduced_share"

    return pd.concat([unaffected, affected], ignore_index=True)


def _build_frequency_lookup(lookup_table: pd.DataFrame) -> pd.DataFrame:
    if {"RP", "RP_future"}.issubset(lookup_table.columns):
        freq_df = lookup_table[["RP", "RP_future"]].copy()
        freq_df["AEP_future"] = 1.0 / freq_df["RP_future"]
        return freq_df
    if {"AEP", "AEP_future"}.issubset(lookup_table.columns):
        freq_df = lookup_table[["AEP", "AEP_future"]].copy()
        freq_df["RP"] = 1.0 / freq_df["AEP"]
        freq_df["RP_future"] = 1.0 / freq_df["AEP_future"]
        return freq_df[["RP", "RP_future", "AEP_future"]]
    raise ValueError("Frequency lookup table must contain either (RP, RP_future) or (AEP, AEP_future).")


def apply_frequency_adaptation(risk_df: pd.DataFrame, spec: AdaptationSpec) -> pd.DataFrame:
    """
    Apply a frequency shift to the targeted basin share by remapping the AEP/RP
    axis while keeping the loss magnitude for that share unchanged.
    """
    warnings.warn(
        "apply_frequency_adaptation uses a basin-coverage approximation. "
        "Prefer apply_basin_frequency_shift/build_frequency_scenario_curves for the main workflow.",
        UserWarning,
        stacklevel=2,
    )
    if spec.lookup_table is None:
        raise ValueError("Frequency adaptation requires `lookup_table`.")

    unaffected, affected = _split_rows_by_coverage(risk_df, spec, target_only=True)
    freq_lookup = _build_frequency_lookup(spec.lookup_table)

    merge_keys = ["RP"]
    if "HB_L6" in freq_lookup.columns and "HB_L6" in affected.columns:
        merge_keys = ["HB_L6", "RP"]

    affected = affected.merge(freq_lookup, on=merge_keys, how="left")
    affected["AEP"] = np.where(
        affected["AEP_future"].notnull(),
        affected["AEP_future"],
        affected["AEP"],
    )
    if "RP_future" in affected.columns:
        affected["RP"] = np.where(
            affected["RP_future"].notnull(),
            affected["RP_future"],
            affected["RP"],
        )
    affected["component_type"] = "frequency_shifted_share"
    affected.drop(columns=[c for c in ["RP_future", "AEP_future"] if c in affected.columns], inplace=True)

    return pd.concat([unaffected, affected], ignore_index=True)


def apply_protection_adaptation(risk_df: pd.DataFrame, spec: AdaptationSpec) -> pd.DataFrame:
    """
    Apply protection adaptation directly on the risk dataframe by splitting the
    covered share into event windows.

    This mirrors the legacy protection semantics:
    - baseline loss for events rarer than the adapted threshold
    - adapted loss for events between the adapted and baseline protection levels
    - zero loss for events more frequent than the baseline protection level
    """
    warnings.warn(
        "apply_protection_adaptation uses a coverage-based approximation. "
        "Prefer raster-derived adapted risk tables with build_protection_scenario_curves.",
        UserWarning,
        stacklevel=2,
    )
    if spec.value is None:
        raise ValueError("Protection adaptation requires `value` as the adapted protection AEP.")
    if "adapted_damages" not in risk_df.columns:
        raise ValueError("Protection adaptation requires an `adapted_damages` column in the risk dataframe.")

    unaffected, affected = _split_rows_by_coverage(risk_df, spec, target_only=True)
    adapted_protection_aep = float(spec.value)

    if affected.empty:
        return unaffected

    baseline_window = affected.copy()
    baseline_window["component_type"] = "protection_baseline_window"
    baseline_window["event_aep_max"] = adapted_protection_aep
    baseline_window["Pr_L_AEP"] = baseline_window["Pr_L_AEP"]

    adapted_window = affected.copy()
    adapted_window["damages"] = adapted_window["adapted_damages"]
    adapted_window["component_type"] = "protection_adapted_window"
    adapted_window["event_aep_min"] = adapted_protection_aep
    adapted_window["event_aep_max"] = adapted_window["Pr_L_AEP"]

    baseline_window = baseline_window.loc[baseline_window["damages"] > 0].copy()
    adapted_window = adapted_window.loc[adapted_window["damages"] > 0].copy()

    return pd.concat([unaffected, baseline_window, adapted_window], ignore_index=True)


def build_scenario_risk_data(
    baseline_risk_df: pd.DataFrame,
    adaptation_specs: list[AdaptationSpec],
    scenario_name: str = "scenario",
) -> pd.DataFrame:
    """
    Build a scenario-ready risk dataframe from the baseline data and a sequence
    of adaptation specifications.
    """
    scenario_df = build_baseline_risk_components(baseline_risk_df, scenario_name=scenario_name)

    for spec in adaptation_specs:
        if spec.mode == "vulnerability":
            scenario_df = apply_vulnerability_adaptation(scenario_df, spec)
        elif spec.mode == "frequency":
            scenario_df = apply_frequency_adaptation(scenario_df, spec)
        elif spec.mode == "protection":
            scenario_df = apply_protection_adaptation(scenario_df, spec)
        else:
            raise ValueError(f"Unknown adaptation mode: {spec.mode}")

        scenario_df["scenario_name"] = scenario_name

    return scenario_df


def build_scenario_curves(
    baseline_risk_df: pd.DataFrame,
    adaptation_specs: list[AdaptationSpec],
    scenario_name: str = "scenario",
) -> dict[int, BasinLossCurve]:
    """
    Convenience wrapper to go from a baseline risk dataframe plus adaptation
    specs directly to basin loss curves.
    """
    scenario_risk_df = build_scenario_risk_data(
        baseline_risk_df,
        adaptation_specs,
        scenario_name=scenario_name,
    )
    return build_basin_curves(scenario_risk_df)
    
    
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

def _extract_all_sectors(*curve_sets):
    all_sectors = set()
    for basin_curves in curve_sets:
        all_sectors.update(
            comp.sector
            for curve in basin_curves.values()
            for comp in curve.components
        )
    return all_sectors


def _run_simulation_pair(baseline_curves, scenario_curves, n_years, copula_numbers):
    """
    Internal implementation for running paired baseline/scenario simulations.
    """
    all_sectors = _extract_all_sectors(baseline_curves, scenario_curves)
    basin_ids = sorted(set(baseline_curves.keys()) | set(scenario_curves.keys()))

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

            aep_event = 1 - random_ns[basin_str]
            baseline_curve = baseline_curves.get(basin_id)
            scenario_curve = scenario_curves.get(basin_id)

            for s in all_sectors:
                bl = baseline_curve.loss_at_event_aep(aep_event, sector=s) if baseline_curve is not None else 0.0
                ad = scenario_curve.loss_at_event_aep(aep_event, sector=s) if scenario_curve is not None else 0.0
                sector_year_baseline[s] += bl
                sector_year_adapted[s]  += ad

        for s in all_sectors:
            sector_baseline_losses[s][t] = sector_year_baseline[s]
            sector_adapted_losses[s][t]  = sector_year_adapted[s]

    return sector_baseline_losses, sector_adapted_losses


def run_simulation(*args):
    """
    Run the Monte Carlo flood simulation.

    Preferred signature:
        run_simulation(baseline_curves, scenario_curves, n_years, copula_numbers)

    Legacy compatibility signature:
        run_simulation(basin_curves, n_years, adaptation_aep, copula_numbers)
    """
    if len(args) != 4:
        raise TypeError(
            "run_simulation expects either "
            "(baseline_curves, scenario_curves, n_years, copula_numbers) or "
            "(basin_curves, n_years, adaptation_aep, copula_numbers)."
        )

    first, second, third, fourth = args

    if isinstance(second, dict):
        baseline_curves = first
        scenario_curves = second
        n_years = third
        copula_numbers = fourth
        return _run_simulation_pair(baseline_curves, scenario_curves, n_years, copula_numbers)

    warnings.warn(
        "The legacy run_simulation(basin_curves, n_years, adaptation_aep, copula_numbers) "
        "signature is deprecated. Prefer passing baseline_curves and scenario_curves directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    basin_curves = first
    n_years = second
    adaptation_aep = third
    copula_numbers = fourth
    scenario_curves = build_protection_scenario_from_baseline(basin_curves, adaptation_aep)
    return _run_simulation_pair(basin_curves, scenario_curves, n_years, copula_numbers)


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
