# Disproportionate Warming of Extreme Land Surface Temperatures in CMIP6 
Models

[![DOI](https://doi.org/10.5281/zenodo.17009316)]

---

## 📖 Overview
This repository contains the code and analysis supporting the article:

> **Wilson, A., Lutsko, N., & Miller, A. (2025). Disproportionate Warming 
of Extreme Land Surface Temperatures in CMIP6 Models. _Geophysical 
Research Letters._**  (prep)

### Description
This study evaluates how extreme land surface temperatures are projected 
to change in the 21st century using output from 23 CMIP6 models. We 
compare annual maximum daily maximum temperature (TXx / TDX) against 
Berkeley Earth observations and ERA5 reanalysis, examining spatial 
fidelity, bias, RMSE, and record-setting behavior. We show that extremes 
are warming disproportionately faster than mean tropical land 
temperatures, with strong inter-model spread in both magnitude and spatial 
skill.

### Plain Language Summary
As the planet warms, dangerously hot days are becoming more frequent. This 
study looks at how the hottest daytime land temperatures each year are 
projected to change in the future. Using global climate models, we compare 
moderate greenhouse gas scenarios and abrupt CO₂ increases. We find that 
extreme heat rises much faster than average warming — in the tropics, the 
hottest days warm about **60% faster** than the mean. Some models simulate 
regional patterns of extreme heat better than others. By the late 21st 
century, record-setting hot days become increasingly concentrated in 
**South Asia, East Africa, and the Middle East**. However, knowing only 
how much a region warms is not enough to predict where the most dangerous 
extremes will occur. Understanding these patterns is critical for helping 
societies prepare for future climate risks.

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
CMIP6 daily maximum temperature (tasmax) from 23 models
→ Earth System Grid Federation (ESGF)
Berkeley Earth observational dataset
→ Berkeley Earth Data
ERA5 reanalysis
→ Copernicus Climate Data Store

⚠️ Due to file sizes, raw data are not hosted in this repository.
Scripts and instructions are provided to reproduce results after 
downloading from the above sources.
