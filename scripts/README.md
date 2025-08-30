# Scripts

This folder contains helper scripts used in the GRL extreme heat study.

---

## `spatial_metrics_models.py`

Main script for calculating **spatial metrics** and preprocessing climate 
datasets.  
It was originally developed in a notebook but consolidated here for easier 
reuse.

### 📌 Tasks covered
- **Data preprocessing**
  - Adding climatology back to anomaly datasets  
  - Regridding model data to a common observational grid  
  - Subsetting to tropical land (30°S–30°N)  

- **Metric calculations**
  - Bias (model – observations)  
  - RMSE (root mean square error, bias-adjusted)  
  - R² (pattern correlation)  

- **File handling**
  - Loading and concatenating NetCDF files  
  - Saving processed outputs and summary metrics  

---


