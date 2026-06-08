# Projections of Earth's Hottest Surface Temperatures in CMIP6

This repository contains analysis code, figures, and supporting material for a study of Earth's hottest near-surface land temperatures in observations and CMIP6 climate model simulations. The project focuses on the upper tail of land heat extremes, including where the hottest land temperatures occur, how well models reproduce historical extreme-heat patterns, and how record-like hottest-land events change under future warming.

The analysis combines Berkeley Earth observations with CMIP6 historical, SSP2-4.5, and abrupt-4xCO2 simulations. Historical model fidelity is evaluated using the spatial pattern of the 99th percentile of daily maximum temperature over land, while future changes are assessed using tropical land warming, annual hottest-day metrics, hotspot-frequency maps, and fixed-threshold exceedance probabilities.

## Project overview

Extreme land heat affects human health, infrastructure, agriculture, ecosystems, and energy demand. Rather than focusing only on mean temperature change, this project asks how the most extreme land temperatures respond as the climate warms.

The central questions are whether CMIP6 models reproduce the observed geography of extreme heat, whether the hottest land temperatures warm faster than the tropical land mean, and whether the locations of record-setting heat remain concentrated in historically hot subtropical dry regions.

## Data sources

This project uses:

- Berkeley Earth daily temperature observations for historical evaluation.
- CMIP6 historical simulations for model-observation comparison.
- CMIP6 SSP2-4.5 simulations for transient 21st-century projections.
- CMIP6 abrupt-4xCO2 simulations for idealized equilibrium-like warming responses.

The main historical evaluation period is 1980-2014. SSP2-4.5 projections are analyzed through 2100, with emphasis on changes relative to a 2005-2014 baseline. The analysis includes 21 CMIP6 models for the primary hottest-day metric, with sample size varying by diagnostic depending on available variables and experiments.

## Key temperature metrics

The analysis uses three annual tropical land indices over 45°S-45°N land:

### `T_TL`

Tropical land-mean temperature, calculated from annual mean near-surface air temperature.

### `T_DM`

Daily-mean temperature on the hottest day of each year, calculated from the annual maximum of daily mean near-surface air temperature.

### `T_DX`

Annual maximum daily-maximum temperature, calculated from daily maximum near-surface air temperature. This is the main hottest-day metric used to track Earth's most extreme land heat.

## Historical model evaluation

Historical fidelity is assessed by comparing the modeled and observed spatial pattern of the 99th percentile of daily maximum temperature, `p99(Tmax)`, over land.

Model skill is summarized with:

- Mean bias
- Centered root-mean-square error
- Spatial R²

The model-observation comparison shows that many CMIP6 models capture the broad geography of historical extreme heat, especially the concentration of high-temperature extremes in subtropical arid and semi-arid regions. However, some models show substantial warm or cold biases and can misplace the hottest regions.

## Hotspot-frequency analysis

For each year, the analysis identifies the single land grid cell with the highest annual `T_DX`. Repeating this across years produces hotspot-frequency maps showing how often different regions host the annual hottest land temperature.

The historical and future hotspot maps show strong spatial persistence. The most frequent hotspots occur in subtropical dry regions, especially the Middle East, North Africa, the Arabian Peninsula, South Asia, and Australia. In future SSP2-4.5 simulations, these hotspots generally intensify and can broaden, but the dominant record-setting regions remain geographically concentrated.

## Warming of tropical land extremes

Under SSP2-4.5, tropical land-mean temperature increases steadily through the 21st century. The hottest-day metrics warm faster than the tropical land mean.

Across the model ensemble:

- `T_TL` warms more slowly than the hottest-day metrics.
- `T_DM` and `T_DX` warm approximately 30-40% faster than `T_TL`.
- This amplified warming appears in both transient SSP2-4.5 trends and late-period abrupt-4xCO2 responses.

This indicates that upper-tail land heat intensifies faster than the background tropical land climate.

## Hottest-land temperature benchmark

The analysis defines an annual hottest-land temperature, `T*(y)`, as the maximum land `T_DX` in each year. A fixed historical benchmark is then defined as:

`T*_hist = max(T*(y)) over 1980-2014`

Future years are evaluated by whether annual `T*(y)` exceeds this historical benchmark.

This fixed-threshold approach provides a record-like measure of how often the climate system produces annual land temperatures hotter than the late-20th-century maximum.

## Exceedance likelihood

The probability of exceeding the historical hottest-land benchmark increases with tropical land warming. A pooled ridge-regularized logistic regression relates exceedance likelihood to tropical land-mean warming.

The multi-model mean fit indicates that exceedance odds rise by approximately 35% per 1°C of tropical land warming. Individual models vary, but most show increasing likelihood as tropical land warms.

## Main findings

1. CMIP6 models generally reproduce the broad observed spatial pattern of historical extreme heat, though biases and regional placement errors remain.

2. The hottest tropical land metrics, `T_DM` and `T_DX`, warm faster than tropical land-mean temperature in both transient and abrupt-4xCO2 experiments.

3. Annual hottest-land events remain concentrated in subtropical arid and semi-arid regions, especially the Middle East, North Africa, South Asia, and Australia.

4. The late-20th-century hottest-land benchmark is exceeded more often as tropical land warms.

5. Mean warming alone does not fully determine where the most dangerous heat occurs or how often record-like extremes happen.

## Repository contents

```text
.
├── data/                  # Processed or intermediate datasets
├── notebooks/             # Analysis notebooks
├── scripts/               # Reproducible analysis scripts
├── figures/               # Manuscript and repository figures
├── README.md              # Project overview
└── environment.yml        # Computational environment, if available
