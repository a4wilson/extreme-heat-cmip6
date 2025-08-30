# Notebooks

This folder contains the Jupyter notebooks used for analysis and figure 
generation in the GRL extreme heat study.

---

## Contents

### `Process_Data_Anthony.ipynb`
- Preprocesses model data for both **historical/SSP245** and 
**abrupt-4xCO₂** experiments.
- Handles concatenation, regridding, and optional saving of intermediate 
NetCDFs.

### `Plot_Max_Temp_Data.ipynb`
- Generates **time series plots** of maximum temperature statistics across 
models.
- Produces figures for manuscript sections on temporal evolution.

### `spatial_record_occurence.ipynb`
- Identifies **where extremes occur spatially** (record-breaking events).
- Useful for understanding geographic distribution of rare heat extremes.

### `spatial_metrics_models.ipynb`
- Calculates **spatial performance metrics** across models:
  - RMSE
  - Bias
  - R²
- Includes regridding routines to align models with observations.

### `Heatwave_Record_Pattern.ipynb`
- Analyzes **patterns of heatwave occurrence**.
- Computes **trends** and **ratios** of record-breaking events.
- Generates figures for pattern consistency across models.

---

## Notes
- Notebooks are intended for **transparency and figure reproduction**.
- Helper code (e.g., metrics functions, regridding) is also available in 
[`../scripts/spatial_metrics_models.py`](../scripts/spatial_metrics_models.py).
- For heavy preprocessing, it is recommended to use the scripts first, 
then run notebooks.

---


