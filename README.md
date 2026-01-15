# Projections of Earth’s Hottest Surface Temperatures in CMIP6 


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17009316.svg)](https://doi.org/10.5281/zenodo.17009316)

---

## 📖 Overview
This repository contains the code and analysis supporting the article:

> **Wilson, A., Lutsko, N., & Miller, A. (2026). Projections of Earth’s Hottest Surface Temperatures in
CMIP6._Geophysical 
Research Letters._**  (prep)

## Project description

Extreme land surface temperatures pose growing risks to human health, infrastructure, and ecosystems. This repository analyzes CMIP6 model simulations (historical, SSP2-4.5, and abrupt 4×CO₂) to understand how the hottest land temperatures evolve with warming and where the most extreme heat occurs. We compare temperature-change metrics that weight the warm tail more strongly to mean-temperature measures, showing that hot-tail indices amplify relative to the mean state in both transient and equilibrium responses. To connect warming with impacts, we also quantify the global frequency of extremely hot conditions by counting worldwide land-days exceeding 40°C each year and fitting a Poisson generalized linear model (GLM) to estimate how these extremes scale with global-mean warming. Finally, we map and track the location of the most extreme land temperatures through time to identify persistent and emerging hotspot regions and to assess why extreme outcomes cannot be inferred from mean warming alone.

## Plain Language Summary

Dangerously hot days are becoming more common as the planet warms, especially over land. This project uses simulations from many global climate models to study the most extreme heat, focusing on (1) how the hottest land temperatures change as the world warms and (2) where on Earth the most extreme heat tends to occur. We analyze a moderate future-emissions pathway (SSP2-4.5) and an idealized experiment where carbon dioxide is abruptly quadrupled, which helps separate short-term from long-term responses.

Across models, the hottest conditions generally increase faster than average temperatures, meaning extreme heat intensifies more than the background climate. To summarize how often very hot conditions occur worldwide, we count how many land areas exceed 40°C each year and fit a statistical model to relate that count to global warming. The multi-model relationship indicates a large increase in the frequency of ≥40°C conditions with each additional degree of warming. We also track the geographic hotspots where the most extreme land temperatures occur, highlighting regions that repeatedly dominate the hottest end of the global distribution. Importantly, models that show similar average warming can still produce very different extreme-heat patterns and hotspot behavior. This is why planning for heat risk requires analyzing extremes directly, not just changes in global mean temperature.


---

## 📂 Repository structure

├── notebooks/
# Jupyter notebooks for reproducing figures and analysis
├── scripts/ 
# Helper scripts for preprocessing and regridding
├── figures/ 
# Figures included in the GRL article
├── environment.yml 
# Conda environment file with dependencies
├── LICENSE
└── README.md

Data

This study uses:
CMIP6 daily maximum temperature 21 global mean (global mean), (tas) from 21 models, 17 models (tasmax)
→ Earth System Grid Federation (ESGF)
Berkeley Earth observational dataset
→ Berkeley Earth Data
ERA5 reanalysis
→ Copernicus Climate Data Store

⚠️ Due to file sizes, raw data are not hosted in this repository.
Scripts and instructions are provided to reproduce results after 
downloading from the above sources.
