# Extreme hot days vs global warming (CMIP6)

This repository contains scripts and outputs used to quantify how the **frequency of extreme hot days** (e.g., **global land-days ≥ 40 °C per year**) scales with **global-mean warming** in CMIP6 models, and how that scaling relates to different temperature-change metrics (transient vs equilibrium warming; land vs ocean amplification; percentile-dependent warming).

## Scientific goals

1. **Track warming over time** under SSP2-4.5 using multiple global temperature-change metrics (Fig. 1a).
2. Compare **transient warming rates** (K/decade) across temperature metrics (Fig. 1b).
3. Compare **equilibrium warming** under **abrupt 4×CO₂** across temperature metrics (Fig. 1c).
4. Compute the **model spread** in the frequency of **≥40 °C land-days** for historical and future periods (Fig. 2a).
5. Fit a statistical relationship between **≥40 °C land-day frequency** and **global-mean annual warming ΔT**, summarized with a **Poisson GLM** and multi-model spread (Fig. 2b; Sup. Fig. GLM; Sup. Fig. per-model fits).
6. Diagnose how warming depends on the **temperature percentile** and differs over **land vs ocean in the tropics (20°S–20°N)**, and quantify the resulting **percentile-dependent amplification** (Fig. 3).

> **Note on temperature metrics:** The repository uses three temperature-change definitions (e.g., `T_TL`, `T_DM`, `T_DX`). These correspond to the temperature indices used in the figures and scripts. See `docs/metrics.md` (or the relevant script header) for precise definitions and spatial masking.

---


---

## What’s in the figures

### Figure 1 — Temperature-change metrics (SSP2-4.5 + abrupt 4×CO₂)
- **(a) Temperature change over time:** multi-model mean warming evolution under SSP2-4.5 for three temperature-change metrics.
- **(b) Transient warming trends:** distribution of warming rates (K/decade) across models for each metric.
- **(c) Equilibrium warming (abrupt 4×CO₂):** distribution of the long-term warming response (K) for each metric.

### Figure 2 — Extreme hot-day frequency and its scaling with warming
- **(a) Model spread of ≥40 °C land-days:** distribution of **global land-days ≥40 °C per year** across historical and future time windows (e.g., 1980–2014, 2040–2059, 2081–2100).
- **(b) Frequency vs warming:** relationship between annual extreme-day counts and **global-mean annual warming ΔT**, shown as a multi-model mean fit and model spread.

### Figure 3 — Where the most extreme temperatures occur (spatial hotspots)

Figure 3 maps the **locations of the most extreme temperatures** and highlights the regions that dominate the warm tail of the land temperature distribution. Rather than focusing on global-mean relationships, this figure answers: **where on Earth do the hottest extremes occur, and where do they concentrate as climate warms?**

**What is plotted**
- Spatial maps of extreme-temperature characteristics (e.g., annual maximum daily maximum temperature, TXx, or the upper-percentile daily maximum such as the 99th percentile).
- Hotspot regions are identified by the highest values of the warm-tail metric over land.

**How to interpret**
- Persistent hotspots tend to appear over **subtropical deserts and arid/semi-arid regions** where clear skies and low soil moisture favor very high surface temperatures.
- Regions with strong land–atmosphere coupling (soil moisture limitations) can show especially high extremes because reduced evaporative cooling allows more energy to go into sensible heating.
- If the figure includes a warming or future period comparison, it can show whether hotspots:
  - **intensify in place** (same locations, higher extremes), and/or
  - **expand/shift** (new areas joining the hottest tail).

---

## Supplementary figures & tables

### Supplementary Figure S1 — GLM summary: extreme hot days vs warming (binned medians + Poisson fit)
This figure summarizes the relationship between **global land-days above a hot threshold** (e.g., ≥40 °C per year) and **global-mean annual warming ΔT** using:
- Year-by-year points (individual model-years),
- Binned medians (with 10–90% spread across models), and
- A **Poisson GLM** that estimates the multiplicative increase in extreme-day frequency per +1 °C warming (reported in the legend).

**Purpose:** provides a compact, model-aggregated estimate of scaling and its uncertainty/spread across models.

---

### Supplementary Figure S2 — Per-model fits: model-by-model scaling of extreme hot days with ΔT
This multi-panel figure shows the same relationship as S1 but **separately for each CMIP6 model**, typically including:
- Scatter of annual extreme-day counts vs ΔT,
- A fitted curve (often the GLM mean),
- Binned summaries or uncertainty bars.

**Purpose:** highlights inter-model differences in curvature/slope and identifies models that are outliers (e.g., steeper scaling or nonlinearity at high warming).

---

### Supplementary Figure S3 — Percentile-dependent warming (land vs ocean) in the tropics (20°S–20°N)
This figure diagnoses how warming depends on the **temperature percentile** and differs over **land vs ocean**, typically shown as:
- (a) ΔTˣ across percentiles of daily temperature (multi-model mean ± IQR),
- (b) a scaling factor Sₓ = ΔTˣ / ΔT indicating whether the warm tail warms faster than the mean.

**Purpose:** provides a mechanistic/statistical explanation for why extremes can increase faster than global-mean warming, especially over land.

---

### Supplementary Table S1 — Model inventory and experiment availability
A table listing CMIP6 models and whether each model includes:
- Historical
- SSP2-4.5
- Abrupt 4×CO₂
- (Optional) “core ensemble” membership

**Purpose:** documents the exact model set used in each analysis component and ensures reproducibility.

---

### Supplementary Table S2 — Additional preprocessing/selection details (if included)
If present, this table can summarize key methodological choices, such as:
- Reference period used for ΔT,
- Land/ocean masking and latitude bands,
- Regridding target grid,
- Threshold definition (≥40 °C) and how “land-days” are counted,
- Any filtering of duplicate downloads / duplicate ensemble members.

**Purpose:** makes technical setup easy to audit and replicate.


## Model inventory

The CMIP6 models used here include those with combinations of:
- **Historical**
- **SSP2-4.5**
- **Abrupt 4×CO₂**
- (Optionally) a “core ensemble” designation

See `tables/` for the model list and availability by experiment (example shown in the supplement model-inventory table).

---

## Workflow overview

Typical pipeline:

1. **Preprocess** daily temperature fields  
   - harmonize calendars/time axis (if needed)  
   - regrid to a common grid (if needed)  
   - apply land/ocean masks and latitude bands (e.g., tropics 20°S–20°N)  
   - compute annual metrics and percentiles as required

2. **Compute warming indices**  
   - annual global-mean ΔT (baseline relative to a reference period)  
   - transient trends (K/decade) over a defined window  
   - equilibrium response for abrupt 4×CO₂ (e.g., late vs early or relative to baseline)

3. **Compute extremes**  
   - count global land-days ≥40 °C per year (or other thresholds)  
   - aggregate across models/ensembles and time windows (e.g., 1980–2014, 2040–2059, 2081–2100)

4. **Fit scaling models**  
   - Poisson GLM (or other count model) linking extreme-day frequency to ΔT  
   - summarize scaling as multiplicative change per +1 °C (e.g., ×1.5–×1.6 per °C)

5. **Make figures & tables**  
   - main figures in `figures/main/`  
   - supplement in `figures/supplement/` and `tables/`

---

## Reproducibility

### Environment
Use one of the following approaches:

**Conda**
```bash
conda env create -f environment.yml
conda activate <env-name>
