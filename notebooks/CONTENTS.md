# Project Contents — CMIP6 Extreme Heat (TXx)

This repository organizes the notebooks and scripts used to preprocess data, compute metrics, and generate figures for the manuscript on extreme land surface temperatures.

## Notebooks

### `Process_Data_Anthony.ipynb`
- Preprocesses model data for **historical**, **SSP2‑4.5**, and **abrupt‑4×CO₂** experiments.
- Handles concatenation across decades, regridding to a common grid, land masking, and optional saving of intermediate NetCDFs.
- **Inputs:** Raw CMIP6 tasmax files (historical, ssp245, abrupt‑4xCO2); Berkeley/ERA5 reference grids.
- **Outputs:** Cleaned, optionally regridded NetCDFs ready for analysis.

### `Plot_Max_Temp_Data.ipynb`
- Generates **time‑series** plots of maximum‑temperature statistics (e.g., TXx) across models.
- Includes multi‑model means, 5‑year smoothing, and experiment overlays for manuscript “temporal evolution” figures.
- **Outputs:** PNGs/SVGs for the paper and slides.

### `spatial_record_occurence.ipynb`
- Identifies **where extremes occur** via record‑breaking event detection.
- Produces **maps of record counts / incidence** and late‑century hotspots.
- **Outputs:** Static maps and model‑by‑model comparison panels.

### `spatial_metrics_models.ipynb`
- Calculates **spatial skill metrics** vs observations (Berkeley Earth / ERA5):
  - RMSE, Mean Bias, Spatial \(R^2\)
- Includes regridding routines to align each model to the obs grid.
- **Outputs:** Table of metrics and rank summaries; figure panels for the manuscript.

### `Heatwave_Record_Pattern.ipynb`
- Analyzes **patterns of heatwave occurrence**, including:
  - Trends and ratios of record‑breaking events through time.
  - Pattern consistency across the ensemble.
- **Outputs:** Regional hot‑spot maps and time‑evolution plots.

---

## New/Updated Notebooks (added here for clarity)

### `SpatialMetrics.ipynb`  *(updated)*
- Streamlined pipeline for RMSE / Bias / spatial \(R^2\) with optional land‑only masking and consistent grid handling.
- Produces the **model‑metrics table** and CSV for downstream use in the paper.

### `Temperature_Trends.ipynb`  *(updated)*
- Computes warming trends for **T\_L**, **T\_DM**, and **T\_DX**; generates boxplots and experiment‑wise comparisons.
- Reproduces Fig. “Temperature Change / Warming Trends / Equilibrium Response”.

### `add_S95_full_colab.ipynb`
- Percentile‑based analysis to compute ΔT\_x and scaling \(S_x = \Delta T_x / \Delta T\) across the distribution, with reproducible Colab setup.
- Generates tropical land vs ocean scaling curves used in the manuscript.

---

## Suggested Run Order
1) `Process_Data_Anthony.ipynb` → produce harmonized, regridded NetCDFs  
2) `SpatialMetrics.ipynb` and/or `spatial_metrics_models.ipynb` → compute model skill  
3) `Temperature_Trends.ipynb` and `Plot_Max_Temp_Data.ipynb` → time‑series + trend figures  
4) `spatial_record_occurence.ipynb` & `Heatwave_Record_Pattern.ipynb` → records / hotspots  
5) `add_S95_full_colab.ipynb` → percentile scaling results

## Paths & Configuration
- **Data roots:** Set the input directories at the top of each notebook (historical, ssp245, abrupt‑4xCO2, and observations).  
- **Output directories:** Figures and intermediate NetCDFs default to a `./outputs/` folder; change as needed.  
- **Regridding:** All regridding is linear/conservative as specified per notebook; ensure the target grid matches obs.  
- **Land mask:** Default is land‑only; toggle in each notebook if ocean is needed.

## Reproducing Figures (examples)
- **Incidence vs warming (Poisson GLM):** `Plot_Max_Temp_Data.ipynb`  
- **Three‑by‑two TXx maps:** `spatial_record_occurence.ipynb`  
- **Tropical scaling vs percentile:** `add_S95_full_colab.ipynb`  
- **Model metrics table:** `SpatialMetrics.ipynb`

---

**Tip:** Keep environment info in `environment.yml` or `requirements.txt` for portability (xarray, dask, netCDF4, xesmf, numpy, pandas, matplotlib, cartopy, statsmodels).

