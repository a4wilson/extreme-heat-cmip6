# Project Contents — CMIP6 Extreme Heat (TXx)

This repository organizes the notebooks and scripts used to preprocess data, compute metrics, and generate figures for the manuscript on extreme land surface temperatures.

## Notebooks

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
1) `SpatialMetrics.ipynb`  → produce harmonized, regridded NetCDFs and compute model skill  
2) `Temperature_Trends.ipynb` → time‑series + trend figures + records / hotspots
3) `add_S95_full_colab.ipynb` → percentile scaling results + frequency

## Paths & Configuration
- **Data roots:** Set the input directories at the top of each notebook (historical, ssp245, abrupt‑4xCO2, and observations).  
- **Output directories:** Figures and intermediate NetCDFs default to a `./outputs/` folder; change as needed.  
- **Regridding:** All regridding is linear/conservative as specified per notebook; ensure the target grid matches obs.  
- **Land mask:** Default is land‑only; toggle in each notebook if ocean is needed.

## Reproducing Figures (examples)  
- **Where do hottest heatwaves occur:** `SpatialMetrics.ipynb`
- **How well do CMIP6 models capture the magnitude, spatial distribution, and evolution of these extremes under both gradual (SSP2-4.5) and abrupt (e.g., abrupt 4×CO2 forcing:** `Temperature_Trends.ipynb` 
- **How frequently do these extreme temperatures recur, and does their frequency scale linearly or accelerate with global warming:** `add_S95_full_colab.ipynb`  
- **Model metrics table:** `SpatialMetrics.ipynb`

---

**Tip:** Keep environment info in `environment.yml` or `requirements.txt` for portability (xarray, dask, netCDF4, xesmf, numpy, pandas, matplotlib, cartopy, statsmodels).

