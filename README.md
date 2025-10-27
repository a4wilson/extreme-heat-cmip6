# Disproportionate Warming of Extreme Land Surface Temperatures in CMIP6 
Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17009316.svg)](https://doi.org/10.5281/zenodo.17009316)

---

## 📖 Overview
This repository contains the code and analysis supporting the article:

> **Wilson, A., Lutsko, N., & Miller, A. (2025). Disproportionate Warming 
of Extreme Land Surface Temperatures in CMIP6 Models. _Geophysical 
Research Letters._**  (prep)

### Description
We assess how extreme land surface temperatures evolve through the 21st century using output from 24 CMIP6 models. Our primary metric is the annual maximum of daily maximum temperature (TDX), evaluated against Berkeley Earth and ERA5 for spatial fidelity, mean bias, RMSE, and record-setting behavior. Models consistently show faster warming of hot extremes than of mean land temperatures. In the tropics (20°S–20°N), ΔTx increases with percentile, with the multi-model scaling factor Sx =ΔTx/ΔT, Sx = ΔTx/ΔT rising from ~1.0 at the median to ~1.25–1.30 at the 99th percentile; oceans show weaker amplification (~1.05–1.12 at the 99th). A Poisson GLM fitted to global land‐day exceedances indicates ~1.58× more hot-day incidence per +1 °C of global warming. For a 40 °C absolute threshold (example: GFDL-ESM4), the fraction of land experiencing ≥1 exceedance per year rises from ~8–9% historically to ~13–14% by 2100, with mean exceedance days increasing from ~2.8 yr⁻¹ (1980–2014) to ~4.7 yr⁻¹ (2040–2059) and ~6.1 yr⁻¹ (2081–2100). Spatial skill varies across models: the multi-model mean achieves RMSE ≈ 2.63 °C, near-zero mean bias, and spatial R2, R2 ≈ 0.84, while individual models span wide performance ranges. Regions with recurring late-century records are concentrated in South Asia, East Africa, the Middle East, and parts of Australia, though the intensity and footprint are model-dependent. Overall, both the magnitude of extreme-heat amplification and the realism of regional patterns differ substantially across models.

### Plain Language Summary
Dangerously hot days increase much faster than average warming on land. Looking across 24 climate models, the hottest daytime temperatures each year warm ~25–30% faster than the average in the tropics, while ocean hot days grow more slowly. A simple statistical model shows that every extra 1 °C of global warming leads to about 1.6 times more hot-day occurrences. For a concrete threshold (40 °C), the share of global land with at least one such day each year climbs from under 10% in recent decades to roughly one in seven land areas by late century, and the average number of 40 °C days per year roughly doubles from historical levels to late century. Late-century record-setting heat concentrates in South Asia, East Africa, the Middle East, and (in several models) Australia. While many models reproduce where extremes occur reasonably well on average, some do much better than others; this model spread matters for planning. The key message is that extreme heat intensifies faster than mean warming and is unevenly distributed, so understanding where models place the strongest extremes is essential for risk reduction and adaptation.

---

## 📂 Repository structure

├── notebooks/ # Jupyter notebooks for reproducing figures and analysis
├── scripts/ # Helper scripts for preprocessing and regridding
├── figures/ # Figures included in the GRL article
├── environment.yml # Conda environment file with dependencies
├── LICENSE
└── README.md

Data

This study uses:
CMIP6 daily maximum temperature 24 global mean (global mean), (tas) from 21 models, 17 models (tasmax)
→ Earth System Grid Federation (ESGF)
Berkeley Earth observational dataset
→ Berkeley Earth Data
ERA5 reanalysis
→ Copernicus Climate Data Store

⚠️ Due to file sizes, raw data are not hosted in this repository.
Scripts and instructions are provided to reproduce results after 
downloading from the above sources.
