# Adaptation Integration Plan

## Goal

Extend the flood risk workflow so adaptation is no longer limited to a single
"design protection" pathway. We want one framework that can support:

1. Protection upgrades
2. Vulnerability-curve reductions for a targeted sector and area
3. Flood-frequency adjustments for a targeted area
4. Future combinations of the above in one scenario

The recommended direction is to move adaptation out of the Monte Carlo logic and
into a scenario-building step upstream. Monte Carlo should sample two curve
sets:

- baseline curves
- scenario curves

This keeps the simulation engine simple and makes each adaptation option a
different way of constructing the scenario curves.

## Current Architecture

The current implementation in [flood.py](../sovereign/flood.py) works like this:

1. A risk dataframe is prepared with basin, sector, AEP, loss, and protection
   information.
2. `build_basin_curves(df)` creates `BasinComponent` objects with:
   - `baseline_losses`
   - `adapted_losses`
   - `protection_aep`
3. `BasinLossCurve.loss_at_event_aep(...)` switches between baseline and
   adaptation logic.
4. `run_simulation(...)` and `run_flood_sim_for_macro(...)` sample baseline and
   adapted losses by passing `scenario="baseline"` or
   `scenario="adaptation"`.

This works for the current "raise protection level" approach, but it couples
adaptation assumptions to the Monte Carlo loop and assumes only one adapted
state per component.

## Target Architecture

### Core principle

Represent adaptation as scenario-specific curve construction rather than
scenario-specific sampling logic.

The Monte Carlo engine should receive:

- `baseline_curves`
- `scenario_curves`

and use the same event draws on both.

### Revised object model

Replace the current "baseline + adapted in one component" pattern with one
component per scenario state.

Proposed replacement:

```python
@dataclass
class BasinComponent:
    admin_id: str
    sector: str
    aeps: np.ndarray
    losses: np.ndarray
    protection_aep: float
    exposure_share: float = 1.0
    component_type: str = "baseline"   # baseline, protected_area, vuln_reduced_area, freq_shifted_area
    meta: dict | None = None

    def loss_at(self, aep_event: float) -> float:
        if self.protection_aep > 0 and aep_event > self.protection_aep:
            return 0.0
        return float(np.interp(aep_event, self.aeps, self.losses))
```

`BasinLossCurve` then becomes a plain sum across its components:

```python
@dataclass
class BasinLossCurve:
    basin_id: int
    components: list[BasinComponent]

    def loss_at_event_aep(self, aep_event: float, sector: str | None = None) -> float:
        comps = self.components if sector is None else [c for c in self.components if c.sector == sector]
        return sum(c.loss_at(aep_event) for c in comps)
```

This means:

- baseline and adaptation are separate curve dictionaries
- each scenario can contain any mix of component types
- Monte Carlo does not need to know why a scenario differs

## Why this is the right fit

This design matches the current codebase well because:

1. `BasinLossCurve` already aggregates multiple components per basin-sector.
2. Climate change is already handled as an upstream curve transformation via
   `risk_data_future_shift(...)`.
3. Partial-area adaptation can be represented by splitting one basin-sector into
   multiple components and summing them during simulation.

## Adaptation Modes

## 1. Protection adaptation

### Current behavior

Protection adaptation currently uses:

- baseline losses below baseline protection
- adapted losses between baseline and upgraded protection
- zero loss above the baseline protection threshold

### Recommended future behavior

Build a scenario curve set where the protected area is already represented in
the scenario component losses and/or protection threshold.

Implementation options:

- **Simple option:** keep one loss curve and change only `protection_aep`
- **Preferred option:** keep separate scenario losses for the protected share if
  the adaptation also changes exposed footprint or local hazard response

This lets protection fit the same scenario-building pattern as the other modes.

## 2. Vulnerability-curve adjustment

### User requirement

For a specified raster area, apply a vulnerability reduction for one sector. For
example, reduce infrastructure vulnerability by 30%.

### Conceptual effect

This does not change event frequency. It changes the damage severity conditional
on hazard intensity.

### Best integration point

The best place to implement this is before basin aggregation, at the raster risk
overlay stage. Vulnerability is a pixel-level property, not a basin-level
property.

### Recommended workflow

1. Take the adaptation raster and align it to the flood/exposure grid.
2. For the target sector only, split the exposure into:
   - inside adaptation area
   - outside adaptation area
3. Recompute damages inside the adaptation area using a modified vulnerability
   curve.
4. Keep baseline damages outside the adaptation area.
5. Aggregate both into basin-sector AEP loss curves.

### How to represent this downstream

For each affected basin-sector, create two scenario components:

- unaffected component
- vulnerability-reduced component

Each gets its own loss array and metadata.

### Curve transformation rule

If the reduction is applied as a scalar on damage ratio:

```python
adjusted_damage_percents = baseline_damage_percents * (1 - reduction_fraction)
```

For a 30% reduction:

```python
reduction_fraction = 0.30
```

If the sector has sector-specific vulnerability curves, only that sector's curve
should be modified.

### Important note

If the adaptation area covers only part of a basin, a basin-level scalar loss
reduction is only an approximation. The component split approach is preferred
because it preserves the unaffected share separately.

## 3. Flood-frequency adjustment

### User requirement

For a specified raster area, adjust flood frequency to reflect the impacts of
Nature Based Solutions.

### Conceptual effect

This does not primarily change damage conditional on flood depth. It changes the
probability structure of events or the mapping from return period to loss.

### Best integration point

This aligns closely with the existing `risk_data_future_shift(...)` logic, which
changes the AEP/RP coordinates of the loss curves.

### Recommended workflow

1. Identify the adapted spatial share using the raster mask.
2. For the adapted share only, remap return periods or AEPs using a supplied
   frequency adjustment curve/table.
3. Keep the unaffected share on the baseline AEP axis.
4. Aggregate both shares into basin-sector scenario components.

### Representation downstream

For each affected basin-sector, create:

- unaffected component with baseline `aeps`
- frequency-shifted component with shifted `aeps`

Loss magnitudes may stay the same while the AEP positions change.

### Key modeling point

Because `loss_at_event_aep(...)` interpolates losses against event AEP, changing
the component's `aeps` is sufficient. The Monte Carlo event sampler does not
need to be rewritten.

### Relationship to climate change

Climate change currently applies a basin-wide frequency shift. Nature Based
Solutions can use the same mechanism, but only for the adapted spatial share.

That suggests a shared transformation utility with a different spatial mask and
different shift table.

## Partial-area interventions

This is the main design issue for both new adaptation types.

### Recommended rule

Do not collapse partial-area interventions into one modified basin curve unless
you explicitly want a coarse approximation.

Instead, split each affected basin-sector into components:

- baseline/unaffected share
- adapted share

This is already compatible with the current `BasinLossCurve` pattern and avoids
losing information on spatial coverage.

### Minimum metadata to carry

Each component should keep:

- `sector`
- `admin_id`
- `protection_aep`
- `component_type`
- `exposure_share`
- `meta`

Suggested `meta` fields:

```python
{
    "adaptation_name": "nbs_floodplain_restoration",
    "adaptation_mode": "frequency",
    "raster_coverage_fraction": 0.34,
    "source": "user_input",
}
```

## Data Model Changes

## Input specification

Add a scenario specification object that can describe one or more adaptation
actions.

```python
@dataclass
class AdaptationSpec:
    name: str
    mode: str                   # protection, vulnerability, frequency
    raster_path: str
    target_sectors: list[str] | None = None
    value: float | None = None
    value_unit: str | None = None
    lookup_table: pd.DataFrame | None = None
    meta: dict | None = None
```

Examples:

- vulnerability reduction:
  - `mode="vulnerability"`
  - `target_sectors=["Infrastructure"]`
  - `value=0.30`
  - `value_unit="fraction_reduction"`

- frequency adjustment:
  - `mode="frequency"`
  - `lookup_table=<rp remapping table>`
  - `value_unit="rp_shift_table"`

## Risk dataframe

The current risk dataframe already includes:

- `HB_L6`
- `GID_1`
- `Sector`
- `RP`
- `AEP`
- `damages`
- `adapted_damages`
- `Pr_L`
- `Pr_L_AEP`

Recommended future shape:

- remove reliance on one hard-coded `adapted_damages` column
- allow scenario construction from a normalized table with:

```python
[
    "HB_L6",
    "GID_1",
    "Sector",
    "RP",
    "AEP",
    "damages",
    "Pr_L",
    "Pr_L_AEP",
    "component_type",
    "scenario_name",
    "exposure_share",
]
```

Optional:

- `component_id`
- `parent_component_id`
- `adaptation_name`

This makes multiple adaptation components possible without adding a new column
for every adaptation style.

## Function-Level Design

## Keep

- `risk_data_future_shift(...)` as the template for frequency-axis
  transformations
- `extract_sectoral_losses(...)`

## Refactor

### `build_basin_curves(df)`

Change it to build one curve set from any scenario-ready dataframe.

Proposed signature:

```python
def build_basin_curves(
    df: pd.DataFrame,
    loss_col: str = "damages",
    group_cols: list[str] | None = None,
) -> dict[int, BasinLossCurve]:
```

Recommended grouping:

```python
["HB_L6", "GID_1", "Sector", "component_type"]
```

### `run_simulation(...)`

Current signature:

```python
run_simulation(basin_curves, n_years, adaptation_aep, copula_numbers)
```

Recommended signature:

```python
def run_simulation(
    baseline_curves: dict[int, BasinLossCurve],
    scenario_curves: dict[int, BasinLossCurve],
    n_years: int,
    copula_numbers: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
```

Behavior:

- sample baseline losses from `baseline_curves`
- sample scenario losses from `scenario_curves`
- use identical event draws for both

This makes the simulation generic across all adaptation types.

### `run_flood_sim_for_macro(...)`

Apply the same refactor:

```python
def run_flood_sim_for_macro(
    baseline_curves,
    scenario_curves,
    num_sims,
    copula_random_ns,
    agr_GVA,
    man_GVA,
    ser_GVA,
    tradable_shares,
    thai_gdp,
):
```

## Add

### Scenario construction utilities

```python
def build_baseline_risk_components(risk_df: pd.DataFrame) -> pd.DataFrame:
    ...

def apply_protection_adaptation(
    risk_df: pd.DataFrame,
    spec: AdaptationSpec,
) -> pd.DataFrame:
    ...

def apply_vulnerability_adaptation(
    risk_df: pd.DataFrame,
    spec: AdaptationSpec,
    vulnerability_curves: dict,
) -> pd.DataFrame:
    ...

def apply_frequency_adaptation(
    risk_df: pd.DataFrame,
    spec: AdaptationSpec,
    frequency_shift_table: pd.DataFrame,
) -> pd.DataFrame:
    ...

def build_scenario_risk_data(
    baseline_risk_df: pd.DataFrame,
    adaptation_specs: list[AdaptationSpec],
    vulnerability_curves: dict | None = None,
    frequency_shift_tables: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    ...
```

### Optional raster utility layer

If the raster operations are repeated, add helpers like:

```python
def load_adaptation_mask(raster_path: str, template_profile: dict) -> np.ndarray:
    ...

def split_exposure_by_mask(exposure_array: np.ndarray, mask_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...
```

## Implementation Strategy By Adaptation Type

## Vulnerability reduction

### Preferred implementation

Implement at the raster loss-generation stage.

Reason:

- vulnerability reduction is depth-damage logic
- applying it after basin aggregation loses fidelity
- raster masking is already conceptually aligned with the current overlay
  workflow

### Workflow

1. Read baseline flood and exposure rasters
2. Read adaptation mask raster
3. For target sector:
   - compute baseline loss outside mask
   - compute reduced-vulnerability loss inside mask
4. Zonal sum by basin
5. Build scenario components

### Fallback approximation

If a first implementation must be quicker, compute a basin-specific coverage
fraction from the raster and scale sector losses:

```python
scenario_damage = baseline_damage * (
    (1 - covered_fraction) + covered_fraction * (1 - reduction_fraction)
)
```

This is acceptable as a temporary shortcut, but it should be clearly labeled as
an approximation.

## Frequency adjustment

### Preferred implementation

Apply an AEP/RP remapping to the adapted spatial share.

### Workflow

1. Estimate covered share per basin-sector
2. Split baseline curve into:
   - unaffected share
   - adapted share
3. For the adapted share, remap AEPs using an NBS shift relationship
4. Build scenario components and simulate normally

### Remapping interface

Use either:

- RP-to-RP lookup table
- AEP-to-AEP lookup table
- functional transform

Example helper:

```python
def remap_aep(aeps: np.ndarray, shift_table: pd.DataFrame) -> np.ndarray:
    ...
```

## Combining adaptations

The long-term design should support multiple adaptation actions in one scenario.

Recommended rule:

- scenario builder applies adaptation specs sequentially
- each spec only modifies the targeted share/components
- outputs remain normalized as scenario components

Example:

1. Start from baseline risk components
2. Apply vulnerability reduction in urban infrastructure cells
3. Apply frequency adjustment in upstream NBS catchments
4. Apply protection upgrade in flood-defense polygons
5. Build final `scenario_curves`

## Backward Compatibility

To avoid breaking notebooks immediately:

1. Keep a compatibility wrapper for the old API for one transition period
2. Internally translate old protection-only adaptation into the new scenario
   builder

Example:

```python
def run_simulation_legacy(basin_curves, n_years, adaptation_aep, copula_numbers):
    scenario_curves = build_protection_scenario_from_baseline(
        basin_curves,
        adaptation_aep,
    )
    return run_simulation(basin_curves, scenario_curves, n_years, copula_numbers)
```

This reduces notebook churn while the new scenario framework is adopted.

## Validation Plan

Validation should be explicit because each mode changes a different part of the
loss-generation chain.

## Unit tests

### Basin curve behavior

- interpolation returns expected values
- protection threshold zeroes out losses above threshold
- component sums equal expected total basin loss

### Vulnerability adaptation

- target sector changes, non-target sectors do not
- zero mask leaves losses unchanged
- full mask with 30% reduction scales damage as expected

### Frequency adaptation

- zero mask leaves AEPs unchanged
- full mask remaps AEPs exactly as in the shift table
- unchanged losses with shifted AEPs produce expected differences at sampled
  events

### Simulation

- identical copula draws are used for baseline and scenario
- baseline output equals scenario output when no adaptations are applied

## Integration tests

- protection-only scenario reproduces current results within tolerance
- vulnerability-only scenario reduces only the selected sector
- frequency-only scenario shifts losses in the expected tail of the loss
  distribution
- combined scenario produces additive or interaction-consistent behavior

## Rollout Plan

## Phase 1: Refactor curve model

1. Simplify `BasinComponent` to one scenario state per component
2. Refactor `build_basin_curves(...)`
3. Refactor `run_simulation(...)`
4. Refactor `run_flood_sim_for_macro(...)`
5. Add compatibility wrapper for protection-only runs

## Phase 2: Vulnerability adaptation

1. Define `AdaptationSpec`
2. Add mask loading and sector targeting
3. Add vulnerability-curve adjustment pipeline
4. Generate scenario risk dataframe and curves
5. Validate against simple controlled examples

## Phase 3: Frequency adaptation

1. Generalize the climate shift utility into a reusable frequency remapper
2. Add spatially targeted share splitting
3. Create NBS frequency scenario builder
4. Validate against controlled RP/AEP remapping examples

## Phase 4: Multi-adaptation scenarios

1. Support sequential application of multiple specs
2. Add metadata tracking
3. Add notebook examples for mixed scenarios

## Recommended first implementation choice

If we want the highest value with controlled complexity:

1. Refactor the simulation API to `baseline_curves` + `scenario_curves`
2. Implement vulnerability adjustment first
3. Reuse the climate-shift pattern for frequency adjustment second

Reason:

- the API refactor is the foundation for everything else
- vulnerability adaptation is the most distinct new capability
- frequency adjustment has a strong existing analogue in the climate-shift code

## Open Decisions

These need confirmation before coding the full workflow:

1. What is the canonical sector label for the "infrastructure" case in the risk
   dataframe: `Public`, `Private`, or another sector?
2. Will vulnerability reduction always be a scalar multiplier on the damage
   ratio, or do you want to support alternative curve shapes?
3. For frequency adaptation, what is the source format for the NBS shift:
   RP-to-RP table, AEP-to-AEP table, or another hazard-layer workflow?
4. Do you want the first release to use:
   - full raster recomputation
   - or basin-level coverage fractions as an approximation

## Summary Recommendation

The best long-term design is:

- baseline and adaptation represented as separate curve dictionaries
- adaptation built upstream through scenario transformers
- partial-area interventions represented as split basin components
- vulnerability handled at the raster-damage stage
- frequency handled as an AEP/RP remapping on the adapted share

This will let the model support protection, vulnerability, frequency, and mixed
adaptation scenarios without continuing to add special cases to Monte Carlo.

## Revised Architecture Note

After initial prototyping, the recommended implementation has been refined.

### What should stay

The Phase 1 refactor toward:

- `baseline_curves`
- `scenario_curves`

is still the right target for Monte Carlo.

The simulation layer should remain generic and should only compare two curve
sets using identical event draws.

### What should change

The earlier basin-coverage approach should not be treated as the main workflow
for adaptation implementation.

It is acceptable as a coarse approximation, but it is not the preferred
representation for this project because the current flood modelling chain is
already geospatially explicit.

The better design is to unify adaptation at the **curve interface**, not at the
**coverage-fraction interface**.

That means all adaptation modes should ultimately produce:

- baseline basin curves
- adapted basin curves

but they do **not** need to get there in the same way.

## Preferred adaptation design by mode

## 1. Protection adaptation

### Recommendation

Keep the existing raster-explicit logic.

This is already a strong implementation because:

- the adaptation footprint is defined spatially
- protected cells are masked at raster level
- adapted damages are aggregated to basin curves
- Monte Carlo selects between baseline and adapted curves according to event
  magnitude and protection level

### Implementation pattern

1. Start from the baseline raster loss workflow
2. Apply the protection or urban mask raster
3. Recompute adapted raster losses for the protected area
4. Aggregate both baseline and adapted losses to basin-level loss probability
   curves
5. Build:
   - `baseline_curves`
   - `protection_scenario_curves`
6. Run Monte Carlo on both

### Why this is preferred

It preserves geospatial targeting directly and avoids replacing a physically
meaningful raster operation with a basin-average approximation.

## 2. Vulnerability-curve adjustment

### Recommendation

Implement this geospatially explicitly as well.

### Concept

For grid cells flagged as adapted, use a modified vulnerability curve for the
target sector. Then recompute raster damages and aggregate to basin curves.

### Implementation pattern

1. Load the adaptation raster that flags adapted cells
2. For the target sector exposure raster:
   - use baseline vulnerability outside adapted cells
   - use reduced vulnerability inside adapted cells
3. Recompute raster damages for each return period
4. Aggregate to basin loss curves
5. Build:
   - `baseline_curves`
   - `vulnerability_scenario_curves`
6. Run Monte Carlo on both

### Why this is preferred

Vulnerability reduction is fundamentally a depth-damage change at the grid-cell
level. Implementing it at basin level would discard the spatial meaning of the
intervention.

## 3. Flood-frequency adjustment

### Recommendation

Implement this at basin scale, not through coverage fractions, unless you later
decide the effect is sub-basin and spatially heterogeneous.

### Concept

This adaptation does not primarily change damage conditional on flood depth. It
changes the frequency mapping of basin losses.

### Implementation pattern

1. Define which basins are affected
2. Define a frequency adjustment relationship:
   - RP to RP
   - or AEP to AEP
   - or a scalar/function that can be converted to one of those
3. For the affected basins, remap the AEP/RP axis of the basin loss curves
4. Build:
   - `baseline_curves`
   - `frequency_scenario_curves`
5. Run Monte Carlo on both

### Why this is preferred

You stated that this should apply to all risk within the basin. In that case a
basin-level frequency remapping is the cleanest and most interpretable design.

## Revised unifying architecture

The correct common layer is the **scenario curve layer**.

### Upstream scenario generators

Different adaptation modes should have different upstream generators:

- `build_protection_scenario_curves(...)`
- `build_vulnerability_scenario_curves(...)`
- `build_frequency_scenario_curves(...)`

### Common Monte Carlo interface

All of them should feed the same downstream API:

```python
run_simulation(
    baseline_curves,
    scenario_curves,
    n_years,
    copula_numbers,
)
```

and similarly for macro:

```python
run_flood_sim_for_macro(
    baseline_curves,
    scenario_curves,
    ...
)
```

## Suggested function design

## Protection

```python
def build_protection_scenario_curves(
    baseline_risk_df: pd.DataFrame,
    adapted_risk_df: pd.DataFrame,
    adapted_protection_aep: float,
) -> dict[int, BasinLossCurve]:
    ...
```

Where:

- `baseline_risk_df` contains baseline basin losses
- `adapted_risk_df` contains raster-derived adapted basin losses
- the function builds the correct event-window logic for protection switching

## Vulnerability

```python
def build_vulnerability_scenario_rasters(...):
    ...

def aggregate_vulnerability_scenario_to_basins(...):
    ...

def build_vulnerability_scenario_curves(
    baseline_risk_df: pd.DataFrame,
    adapted_risk_df: pd.DataFrame,
) -> dict[int, BasinLossCurve]:
    ...
```

The key point is that `adapted_risk_df` should come from raster recomputation
using the modified vulnerability curve.

## Frequency

```python
def apply_basin_frequency_shift(
    risk_df: pd.DataFrame,
    frequency_shift_table: pd.DataFrame,
    basin_ids: list[int] | None = None,
) -> pd.DataFrame:
    ...

def build_frequency_scenario_curves(
    baseline_risk_df: pd.DataFrame,
    frequency_shift_table: pd.DataFrame,
    basin_ids: list[int] | None = None,
) -> dict[int, BasinLossCurve]:
    ...
```

## Practical implication for current code

The recently added coverage-fraction utilities in `sovereign/flood.py` should
be treated as prototype helpers, not as the final modelling path.

Recommended use going forward:

- keep the paired-curve Monte Carlo refactor
- keep backward compatibility wrappers temporarily
- redesign the scenario builders around:
  - raster-explicit protection
  - raster-explicit vulnerability
  - basin-level frequency shift

## Recommended next implementation order

1. Preserve the paired-curve simulation API
2. Reintroduce protection as a raster-derived scenario-curve builder
3. Add vulnerability as a raster-derived scenario-curve builder
4. Add frequency as a basin-level curve-axis modifier
5. Update notebooks only after those scenario builders are stable

## Final recommendation

Use a **mixed upstream / unified downstream** architecture:

- mixed upstream because the physical meaning of each adaptation type differs
- unified downstream because all scenarios should enter Monte Carlo as basin
  curve dictionaries

That will preserve the geospatial realism of the current workflow while still
giving you a clean and extensible adaptation framework.
