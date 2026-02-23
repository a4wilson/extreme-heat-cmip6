# Notebooks

This folder contains the Jupyter notebooks used for analysis and figure generation
in the GRL extreme-heat study. The notebooks largely mirror the `scripts/` pipeline,
but are kept for transparency, interactive diagnostics, and figure polishing.

---

## Contents

### `SpatialMetrics.ipynb`  *(currently: `SpatialMetrics (2).ipynb`)*
**Purpose:** Historical model evaluation and spatial diagnostics for extreme land heat.

**Key tasks**
- Loads Berkeley Earth daily tasmax and CMIP6 historical tasmax.
- Computes the historical extreme field (e.g., **p99(tasmax)** over **1980–2014**).
- Regrids model fields to the Berkeley Earth grid (bilinear) before comparing.
- Computes spatial skill metrics used in Table 1 / SI:
  - **Bias**
  - **Centered RMSE (cRMSE)**
  - **Spatial R²**
- Produces “stamp” diagnostic panels (model p99 and bias maps).

**Figures / tables generated (examples)**
- `cmip6_model_metrics_1980_2014_p99_table_shaded_vs_MEDIAN_with_MEAN_and_MEDIAN_rows.png` *(Table 1-style)*
- `figS_p99_and_bias_stamp.png`, `figS_p99_bias_stamp.png` *(SI stamp panels)*
- `figS_model_skillscore_ranked.png` *(composite skill ranking; if used)*
- Additional p99 diagnostic plots (global/tropics versions)

**Related scripts**
- Use this notebook to validate or reproduce outputs from your spatial-metrics scripts
  (e.g., regridding + Bias/cRMSE/R² utilities and table-generation code).

---

### `Temperature_Trends.ipynb` *(currently: `Temperature_Trends (1).ipynb`)*
**Purpose:** Transient vs equilibrium warming comparisons for tropical land indices.

**Key tasks**
- Loads preprocessed time series for:
  - **T_TL** (tropical land mean)
  - **T_DM** (daily-mean on the hottest day)
  - **T_DX** (annual max tasmax)
- Generates multi-model mean time series under **SSP2-4.5** (anomalies relative to a baseline).
- Computes per-model transient trends (e.g., **2015–2100**) and summarizes spread.
- Computes equilibrium warming in **abrupt-4×CO₂** relative to **piControl** (subset availability).

**Figures generated**
- `fig2_threepanel_SSPfull_equilsubset_C_yTicksFixed.png` *(Fig 2-style three-panel)*
- `figS_per_model_TL_TDM_TDX_spaghetti_and_trends_ALLOWED.png` *(SI per-model trajectories + trends)*

**Related scripts**
- Mirrors your trend / equilibrium processing scripts (time series assembly + regression fits).

---

### `add_S95_full_colab.ipynb` *(currently: `add_S95_full_colab (1).ipynb`)*
**Purpose:** Byrne-style tail-amplification diagnostics + record-like exceedance analysis.

**Key tasks**
- Computes **percentile-dependent warming / scaling** (“Byrne-style” metrics), including
  land vs ocean comparisons in the tropics (if enabled in the workflow).
- Produces record-like exceedance curves using a **fixed historical benchmark**:
  - Defines **T\*_hist = max(T\*(y)) over 1980–2014**
  - Computes annual exceedance indicator **I(y)**
  - Fits pooled (ridge-regularized) logistic scaling vs warming predictor

**Figures generated (examples)**
- `Fig2_like_tas_tropics.png` *(percentile/tail scaling diagnostic)*
- `fig_recordprob_fixedThreshold_twoPanel_GRLclean.png` *(Fig 3-style exceedance scaling)*
- `fig_recordprob_fixedThreshold_twoPanel_fig2style.png` *(alternate style/layout)*

**Related scripts**
- Mirrors your exceedance/GLM scripts (benchmark definition, pooled logit, bootstrapping).

---

## Notes / recommended workflow

- **Prefer `scripts/` for batch preprocessing** (especially CMIP6 downloads, calendar fixes,
  regridding, and annual aggregation). Use notebooks mainly for:
  - diagnostics,
  - figure recreation,
  - sensitivity checks,
  - final plot styling.
- If you rename notebooks for cleanliness, recommended final names:
  - `SpatialMetrics.ipynb`
  - `Temperature_Trends.ipynb`
  - `Byrne_Metrics_and_RecordLike_Exceedance.ipynb`

---

## Outputs

Notebooks typically write to `figures/` (and occasionally `tables/`), depending on your local paths.
If a notebook saves intermediate processed files, those should go to a dedicated `processed/`
directory to keep the repo reproducible and tidy.
