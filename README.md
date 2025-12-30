# Projections of Earth’s Hottest Surface Temperatures in CMIP6 


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17009316.svg)](https://doi.org/10.5281/zenodo.17009316)

---

## 📖 Overview
This repository contains the code and analysis supporting the article:

> **Wilson, A., Lutsko, N., & Miller, A. (2025). Projections of Earth’s Hottest Surface Temperatures in
CMIP6._Geophysical 
Research Letters._**  (prep)

### Description
Extreme land surface temperatures threaten health, infrastructure, and
ecosystems. Using the annual maximum of daily maximum temperature ($T_{DX}$)
from 23 CMIP6 models, we examine how hot extremes evolve under SSP2-4.5 and
abrupt 4$\times$CO$_2$ forcing, and how well models reproduce observed
patterns. In both transient and equilibrium simulations, indices that emphasise
hotter conditions---especially $T_{DX}$---warm more strongly than tropical-mean
land temperature, indicating intensification of extremes relative to the mean
state.

Comparing historical $T_{DX}$ to Berkeley Earth over land, we find substantial
inter-model spread in spatial bias, RMSE, and pattern correlation, with the
largest errors in tropical and semi-arid regions. We also track the location
and recurrence of the hottest land grid cell each year. Record-setting heat
becomes increasingly concentrated over South Asia, East Africa, and the Middle
East, while models with similar mean warming show diverse record frequencies.
This underscores the need to evaluate extremes explicitly, rather than relying
on mean warming alone.

### Plain Language Summary
Dangerously hot days are becoming more common as the planet warms, especially
over land. This study examines how the hottest daytime temperature each year
(the single hottest day at a given location, denoted $T_{DX}$) changes in
simulations from 23 global climate models. We analyse a moderate emissions
pathway (SSP2-4.5) and an experiment where carbon dioxide is abruptly
quadrupled. Across these simulations, the most intense hot days over tropical
land warm faster than the average land temperature and faster than related
daily-mean measures. In other words, the extremes strengthen more than the
background climate.

When we compare the models to observations from the Berkeley Earth dataset,
some reproduce the observed pattern of extreme heat reasonably well, while
others show large warm biases and weaker spatial agreement, particularly in
tropical and dry regions. We also track where the hottest land temperature on
Earth occurs each year. Over time, record-setting heat becomes increasingly
concentrated in regions such as South Asia, East Africa, and the Middle East.
Models with similar overall warming, however, can still produce very different
patterns and frequencies of record-breaking heat. This means that mean warming
alone is not enough to identify where the most dangerous extremes will occur,
which is critical for planning adaptation and protecting vulnerable
communities.

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
CMIP6 daily maximum temperature 24 global mean (global mean), (tas) from 21 models, 17 models (tasmax)
→ Earth System Grid Federation (ESGF)
Berkeley Earth observational dataset
→ Berkeley Earth Data
ERA5 reanalysis
→ Copernicus Climate Data Store

⚠️ Due to file sizes, raw data are not hosted in this repository.
Scripts and instructions are provided to reproduce results after 
downloading from the above sources.
