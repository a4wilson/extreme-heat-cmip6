# Projections of Earth’s Hottest Land Temperatures in CMIP6

This repository contains scripts and outputs used to evaluate and project **Earth’s hottest near-surface air temperatures over land** using **CMIP6** models and **Berkeley Earth** observations. The analysis focuses on (i) how well models reproduce the **spatial pattern of extreme land heat** in the historical period, (ii) how **tropical land extremes amplify** relative to mean warming under **SSP2-4.5** and **abrupt-4×CO₂**, (iii) where the **annual global land maximum** occurs (hotspot persistence), and (iv) how often future climate exceeds a **fixed historical “record-like”** hottest-land benchmark.

## Scientific goals

1. **Historical fidelity:** Evaluate CMIP6 ability to simulate the observed spatial pattern of extreme land heat using Berkeley Earth.
2. **Tail amplification:** Quantify whether hottest-day indices over tropical land warm faster than tropical land-mean temperature under SSP2-4.5 and abrupt-4×CO₂.
3. **Hotspot geography:** Diagnose where the **annual global land maximum of tasmax** occurs and how persistent those regions are historically and in projections.
4. **Record-like exceedance:** Define a fixed historical benchmark for the **annual hottest-land temperature** and estimate how exceedance likelihood scales with tropical land warming using a pooled logistic model.

## Core metrics and indices

### Historical evaluation metric
- **p99(tasmax)**: climatological 99th percentile of daily maximum near-surface air temperature over land (1980–2014), computed on each model grid and regridded to Berkeley Earth for pattern comparisons.

### Tropical land indices (45°S–45°N land)
- **T_TL**: tropical land-mean near-surface temperature (annual mean of tas).
- **T_DM**: daily-mean temperature on the hottest day each year (annual max of daily tas).
- **T_DX**: annual maximum daily-maximum temperature (annual max of daily tasmax).

### Hottest-land index
- **T\*(y)**: annual hottest-land temperature, defined as the annual maximum of the daily land maximum of tasmax over 45°S–45°N land.

## What’s in the main figures

### Figure 1 — Observed and modeled spatial patterns of extreme land heat + hotspot frequency
- **Left column:** Maps of **p99(tasmax)** (1980–2014) for Berkeley Earth and example CMIP6 models (e.g., a high-skill and a biased model).
- **Right column:** **Hotspot-frequency maps**: for each year, the single land grid cell attaining the **global land maximum** of annual tasmax is identified; counts across years show where record-setting heat concentrates historically and under SSP2-4.5 (2015–2100).

### Figure 2 — Tail amplification of tropical land extremes
Multi-model comparisons for SSP2-4.5 (2015–2100) and abrupt-4×CO₂:
- **(a) Time series:** anomalies relative to a reference baseline (e.g., 2005–2014), shown for **T_TL**, **T_DM**, and **T_DX**.
- **(b) Transient trends:** per-model warming trends (°C/decade) over 2015–2100.
- **(c) Equilibrium warming:** abrupt-4×CO₂ response relative to piControl (late-time mean difference).

### Figure 3 — Fixed-threshold “record-like” exceedance of the hottest-land benchmark
- **(a) Model spread** in the annual hottest-land temperature **T\*** across time windows.
- **(b) Exceedance probability** for surpassing a fixed historical benchmark  
  **T\*_hist = max(T\*(y)) over 1980–2014**,  
  modeled as a function of tropical land warming using a pooled ridge-regularized logistic regression.
  The scaling is summarized as an **odds ratio per +1°C** of tropical land warming.

## Supplementary figures and tables (typical contents)

- **Table S1:** Model inventory and experiment availability (historical, SSP2-4.5, abrupt-4×CO₂, piControl; plus index availability).
- **Table S2:** Historical skill metrics for p99(tasmax): **Bias**, **cRMSE**, and **spatial R²**.
- **Fig S1:** Per-model trajectories/trends for T_TL, T_DM, and T_DX under SSP2-4.5.
- **Fig S2–S3:** Hotspot-frequency maps for each model (historical and SSP2-4.5).
- **Fig S4:** “Stamp” panels of p99(tasmax) and bias maps across the ensemble.
- **Fig S5:** Composite model-skill summary/ranking (if used).
- **Fig S6:** Percentile-dependent warming / amplification (e.g., S_x = ΔT^x / ΔT), optionally separated by land vs ocean.

## Workflow overview

1. **Preprocess daily fields**
   - Load tasmax (and tas where needed), harmonize calendars/time axis, apply land masks and latitude bands.
   - Regrid model fields to the Berkeley Earth grid for spatial pattern metrics.

2. **Historical evaluation**
   - Compute p99(tasmax) over 1980–2014.
   - Compute spatial **Bias**, **cRMSE**, and **R²** vs Berkeley Earth.

3. **Tropical land indices**
   - Compute T_TL, T_DM, and T_DX annually.
   - Compare SSP2-4.5 trends (2015–2100) and abrupt-4×CO₂ equilibrium responses.

4. **Hotspot-frequency maps**
   - For each year, identify the grid cell with the global land maximum of annual tasmax.
   - Count frequency across years for historical and SSP2-4.5.

5. **Record-like exceedance**
   - Define T\*_hist from 1980–2014 and annual exceedance indicator I(y).
   - Fit pooled ridge-logit scaling vs tropical land warming; summarize as odds ratio per °C.

## Reproducibility

### Environment
**Conda**
```bash
conda env create -f environment.yml
conda activate <env-name>
