# Projections of Earth’s Hottest Land Temperatures in CMIP6

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17009316.svg)](https://doi.org/10.5281/zenodo.17009316)

---

## 📖 Overview
This repository contains code and analysis supporting the GRL manuscript:

> **Wilson, A., Lutsko, N., & Miller, A. (2026). Projections of Earth’s Hottest Land Temperatures in CMIP6. _Geophysical Research Letters_ (in prep).**

The project evaluates how well CMIP6 models reproduce the **observed spatial pattern of extreme land heat** and how the **hottest land temperatures** change under warming in both **transient (SSP2-4.5)** and **equilibrium (abrupt-4×CO₂)** experiments.

---

## 🔥 Project description

Extreme land heat threatens health, infrastructure, and ecosystems worldwide. Here we use CMIP6 simulations (historical, SSP2-4.5, abrupt-4×CO₂) together with **Berkeley Earth** observations to study:

1. **Historical model fidelity for extremes**  
   We evaluate each model’s ability to reproduce the **climatological 99th percentile of daily maximum temperature** over land, using **Bias**, **centered RMSE (cRMSE)**, and **spatial R²** computed on the Berkeley Earth grid.

2. **Tail amplification over tropical land**  
   We compare mean and hottest-day warming using three annual indices over tropical land (45°S–45°N):
   - **T_TL**: tropical land-mean temperature (annual mean of `tas`)
   - **T_DM**: daily-mean temperature on the hottest day each year (annual max of daily `tas`)
   - **T_DX**: annual maximum daily-maximum temperature (annual max of daily `tasmax`)

   Across models, **T_DM** and **T_DX** typically warm faster than **T_TL** in SSP2-4.5 trends and abrupt-4×CO₂ equilibrium responses.

3. **Geography of record-setting heat (hotspot persistence)**  
   Each year, we identify the single land grid cell attaining the **global land maximum** of annual `tasmax` and map its **hotspot frequency**. This highlights the regions that most often dominate the warm tail and whether hotspots intensify in place or broaden/shift.

4. **Record-like exceedance likelihood under warming**  
   We define the annual hottest-land temperature **T\*(y)** and a fixed historical benchmark
   **T\*_hist = max(T\*(y)) over 1980–2014**. We then quantify how exceedance likelihood scales with tropical land warming using a pooled, ridge-regularized logistic regression, summarized as an **odds ratio per +1°C**.

---

## 🧾 Plain Language Summary

Dangerously hot days are becoming more common as the planet warms—especially over land. This project uses many CMIP6 climate models to study the most extreme land heat, focusing on (1) how the hottest land temperatures change as the climate warms and (2) where on Earth these hottest conditions tend to occur. We analyze a moderate future-emissions pathway (SSP2-4.5) and an idealized experiment where carbon dioxide is abruptly quadrupled, which helps separate short-term from long-term responses.

Across models, the hottest-day temperatures over tropical land increase faster than tropical-average land temperature. We also test how well models reproduce the observed pattern of extreme land heat using Berkeley Earth observations. Finally, we track where the single hottest land grid cell occurs each year and show that record-setting heat is strongly concentrated in subtropical arid and semi-arid regions. These results show that average warming alone cannot tell us where the most dangerous heat will occur or how often record-like extremes will happen there.

---

## 🖼️ Figures and tables

- **Table 1:** Historical model skill for extreme land heat (Bias, cRMSE, spatial R²) comparing **p99(tasmax)** with Berkeley Earth (1980–2014).
- **Figure 1:** Spatial pattern of **p99(tasmax)** and **hotspot-frequency maps** for the annual global land maximum.
- **Figure 2:** Transient and equilibrium warming for **T_TL**, **T_DM**, and **T_DX** (SSP2-4.5 + abrupt-4×CO₂).
- **Figure 3:** Fixed-threshold **record-like exceedance** of **T\*_hist** and its scaling with tropical land warming (ridge-logit pooled fit).

Supporting figures include per-model “stamp” plots of p99/bias, hotspot-frequency maps across the ensemble, and (optionally) percentile-dependent warming diagnostics.

---

## 📂 Repository structure

