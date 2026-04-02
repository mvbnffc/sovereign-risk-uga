# Adaptation Output Convention

## Goal

Keep baseline preparation and adaptation scenario generation separate, while
making the handoff to the simulation notebooks consistent.

The common handoff artifact should be a **basin risk CSV** with the same schema
as the baseline risk tables:

- `HB_L6`
- `GID_1`
- `Sector`
- `RP`
- `damages`
- `Pr_L`
- `AEP`
- `Pr_L_AEP`

## Recommended output layout

### Baseline

Keep the current baseline outputs:

- `outputs/flood/risk/maps/...`
- `outputs/flood/risk/basins/risk_basins_m-<model>.csv`

### Adaptation scenarios

Store adaptation outputs under:

- `outputs/flood/adaptation/<mode>/<scenario_name>/maps/...`
- `outputs/flood/adaptation/<mode>/<scenario_name>/basins/<scenario_file>.csv`

Where:

- `<mode>` is one of:
  - `protection`
  - `vulnerability`
  - `frequency`
- `<scenario_name>` is a short descriptive identifier for the case

## Example paths

### Vulnerability

- `outputs/flood/adaptation/vulnerability/pubinf_urban_mask_50pct/maps/WRI_100_pub_inf_cap_damages.tif`
- `outputs/flood/adaptation/vulnerability/pubinf_urban_mask_50pct/maps/WRI_100_pub_cap_damages.tif`
- `outputs/flood/adaptation/vulnerability/pubinf_urban_mask_50pct/basins/risk_basins_m-wri.csv`

### Protection

- `outputs/flood/adaptation/protection/urban_mask_aep001/maps/...`
- `outputs/flood/adaptation/protection/urban_mask_aep001/basins/risk_basins_m-wri.csv`

### Frequency

- `outputs/flood/adaptation/frequency/basin_shift_150pct/basins/risk_basins_m-wri.csv`

Frequency adaptation does not necessarily need map outputs. In that case only
the `basins` folder is required.

## Notebook roles

### Prep notebooks

Prep notebooks should:

1. build scenario-specific maps if needed
2. aggregate or transform them into a basin risk CSV
3. save that CSV under the scenario folder

### Simulation notebooks

Simulation notebooks should:

1. load baseline basin CSV
2. load scenario basin CSV
3. build `baseline_curves`
4. build `scenario_curves`
5. run the paired-curve simulation

Simulation notebooks should not contain raster risk-generation logic.

## Naming recommendation

Use scenario names that encode the essential adaptation assumption:

- `pubinf_urban_mask_50pct`
- `urban_protection_aep001`
- `nbs_shift_150pct`

This keeps scenario directories readable and makes it easy to compare cases.

## Recommended notebook pattern

### Prep

- `0.5_adaptation_vulnerability_prep.ipynb`
- `0.6_adaptation_frequency_prep.ipynb`
- `0.7_adaptation_protection_prep.ipynb`

### Analysis

- `3_national_flood_simulation_scenario.ipynb`
- `4_macro_simulation_scenario.ipynb`

These scenario analysis notebooks should be generic: they should read a baseline
CSV plus a scenario CSV and run the same downstream analysis regardless of how
the scenario was generated.
