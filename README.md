**Project Title**
Demand Estimation for Hotel Booking Probability

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Data Description](#data-description)
4. [Methodology](#methodology)

   * [1. Instrumental-Variables (IV) Approach](#1-instrumental-variables-iv-approach)
   * [2. Double Machine Learning (DML) / Control-Function Approach](#2-double-machine-learning-dml--control-function-approach)
   * [3. Heterogeneity Analysis](#3-heterogeneity-analysis)
5. [Dependencies](#dependencies)
6. [Installation & Setup](#installation--setup)
7. [Usage](#usage)

   * [Running the R Markdown](#running-the-r-markdown)
   * [Inspecting the PDF Report](#inspecting-the-pdf-report)
8. [Results & Interpretation](#results--interpretation)
9. [Contributing](#contributing)
10. [License](#license)
11. [Contact](#contact)

---

## Project Overview

This repository contains an end‐to‐end analysis of hotel booking demand using two primary econometric approaches:

1. **Instrumental‐Variables (IV)** (Two‐Stage Least Squares) to address price endogeneity.
2. **Double Machine Learning (DML)** (Control‐Function / Selection‐on‐Observables) as an alternative when instrument validity is uncertain.

Beyond estimating a baseline price elasticity, this project also explores **heterogeneity in price sensitivity** by lead‐time, day‐of‐week/seasonality, and room/customer segment.

The main outputs are:

* An interactive R Markdown document (`demand estimation.Rmd`) that reproduces every step of the analysis.
* A compiled PDF report (`demand estimation.pdf`) summarizing key results, tables, and economic interpretations.

---

## Repository Structure

```
.
├── data/
│   └── demand_est.csv              # Raw data used for estimation
│
├── R/                              
│   └── functions/                  
│       └── (optional helper scripts for data cleaning or plotting)
│
├── demand estimation.Rmd           # Main R Markdown script (analysis + code)
├── demand estimation.pdf            # Compiled PDF version of the report
├── README.md                        # This README file
├── .gitignore                       # Standard exclusions (e.g., *.Rproj.user, cache folders)
└── LICENSE                          # Project license (MIT by default)
```

* **`data/demand_est.csv`**
  Contains one row per search/query (`i`) with fields:

  * `rate_actual`: the price charged
  * `rate_recommended`: the algorithm’s recommended price
  * `bookings`: indicator (0/1) if a booking occurred
  * `date_of_stay`, `days_ahead`: date‐level and lead‐time variables
  * `XX2` … `XX250`: rich covariates fed to the pricing engine
  * `V251`: an additional hotel‐side index
  * `date_data`: date stamp for clustering in the IV specification

* **`demand estimation.Rmd`**
  A literate analysis that:

  1. Loads and inspects the data.
  2. Constructs log‐price variables and fixed‐effect factors.
  3. Implements the IV specification (first‐stage, second‐stage LPM & control‐function logit) using the `AER` package.
  4. Computes cluster‐robust standard errors (`sandwich` + `lmtest`).
  5. Implements a DML pipeline (cross‐fitting with `glmnet`) to residualize outcome and treatment, then estimates a final elasticity.
  6. Explores heterogeneity in elasticity by:

     * Lead‐time (“days ahead” buckets)
     * Day‐of‐week / seasonality (weekend vs. weekday, holidays)
     * Room/customer segment (e.g. suite vs. budget, OTA vs. direct web)
  7. Produces summary tables with `stargazer` and commentary throughout.

* **`demand estimation.pdf`**
  A static, print‐ready version of the R Markdown (with all code output, regression tables, and written interpretation).

---

## Data Description

The core dataset (`demand_est.csv`) includes 50,000 observations (rows), where each row corresponds to a unique “search” or “pricing query” by a potential customer. Key columns:

* **`bookings`** (0/1): Indicator if the customer ultimately booked.
* **`rate_actual`** (numeric): The per‐night price set by the hotel manager.
* **`rate_recommended`** (numeric): The software’s recommended price (never seen by customers).
* **`date_of_stay`** (YYYY‐MM‐DD): The calendar date of the stay.
* **`days_ahead`** (integer ≥ 0): Number of days between booking query and stay date (lead time).
* **`XX2` … `XX250`** (various continuous/binary covariates): Example features include competitor pricing summaries, weather metrics, local holiday flags, hotel review scores, room features, etc.
* **`V251`** (numeric): A composite index capturing a hotel‐side operational factor (e.g., housekeeping capacity).
* **`date_data`**: Date on which the “search event” was observed (used for clustering in IV SEs).

Before running any regressions, the R code:

1. Drops any rows with missing values.
2. Creates `log_price_actual = log(rate_actual)` and `log_price_recommended = log(rate_recommended)`.
3. Converts `date_of_stay` and `days_ahead` into factor variables for fixed effects.
4. Scales numeric `XX` covariates (optional, for DML/LASSO stability).

---

## Methodology

### 1. Instrumental‐Variables (IV) Approach

#### 1.1 Identification Challenge

The key endogeneity arises because the hotel manager sets `rate_actual` in response to unobserved demand shocks (e.g., a last‐minute conference). If we simply regress

```
Pr(bookings_i = 1 | log(rate_actual_i), X_i) = F(β₀ + β₁·log(rate_actual_i) + X′_i·γ)
```

then β₁ is biased—managers raise price when demand is already high (yielding spurious positive correlation).

#### 1.2 Instrument: `log(rate_recommended)`

* **First Stage (Pricing Equation)**

  ```
  log(rate_actual_i) = π₀ + π₁·log(rate_recommended_i) 
                      + Σ_{k=1}^K δ_k·X_{i,k} 
                      + Σ_{t} α_t·1{date_of_stay_i = t} 
                      + Σ_{d} ζ_d·1{days_ahead_i = d} 
                      + ν_i
  ```

  * `X_i` = {`XX2` … `XX250`, `V251`} are all observable covariates fed to the recommender.
  * Date‐of‐Stay and Days‐Ahead fixed effects absorb seasonality and look‐ahead mechanical patterns.
  * R code uses `ivreg(...)` from the **AER** package.

* **Second Stage (Booking Probability)**
  Two specifications are shown:

  1. **Linear‐Probability 2SLS**

     ```
     bookings_i = β₀ + β_price · ŷln(rate_actual_i) 
                  + Σ_{k=1}^K γ_k·X_{i,k} 
                  + Σ_{t} η_t·1{date_of_stay_i = t} 
                  + Σ_{d} κ_d·1{days_ahead_i = d} 
                  + ε_i  
     ```

     where ŷln(rate\_actual\_i) is the fitted value from the first stage.
  2. **Control‐Function IV‐Logit** (optional)

     ```
     bookings_i = logit⁻¹(β₀ + β_price·ln(rate_actual_i) 
                           + θ·(residual_{i}) 
                           + X′_i·γ + … ) + e_i  
     ```

     where `residual_i = ln(rate_actual_i) – ŷln(rate_actual_i)` is the first‐stage residual.

* **Key Assumptions**

  1. **Relevance**: Cov(log(rate\_recommended), log(rate\_actual)) ≠ 0 after controlling for `X_i`, date‐of‐stay FE, and days‐ahead FE.
  2. **Exclusion**: Conditional on the same covariates used to generate `rate_recommended`, any remaining variation in `rate_recommended` affects booking only through its effect on `rate_actual`.

* **Interpretation**

  * If `β_price < 0`, higher price reduces booking probability.
  * If `β_price > 0`, residual endogeneity remains (i.e., high‐demand shocks still drive price upward).

### 2. Double Machine Learning (DML) / Control-Function Approach

When the exclusion restriction for `rate_recommended` is questionable (e.g., manager might leak “recommended price” to staff), we can rely on a “selection‐on‐observables” assumption and use modern ML to flexibly control for high‐dimensional confounders.

#### 2.1 Implementation Steps

1. **Create Matrices**

   * Outcome covariates (`Z_i`): includes `log_price_actual`, all `XX` covariates, Date‐FE, Days‐Ahead FE.
   * Treatment covariates (`X_i`): excludes `log_price_actual`, but includes all `XX` covariates + Date‐FE + Days‐Ahead FE.
2. **Two‐Fold Sample Split**

   * Randomly assign each observation to one of two folds (1 or 2).
   * For `k ∈ {1,2}`: train ML models on fold `k` to predict (a) booking probability from `Z` (binary classification LASSO) and (b) `log_price_actual` from `X` (continuous LASSO).
   * Use out‐of‐sample predictions on the opposite fold to get:

     * `m_hat_i ≈ Ė[bookings_i | Z_i]`
     * `g_hat_i ≈ Ė[log_price_actual_i | X_i]`
3. **Form Residualized Variables**

   * `Y˜_i = bookings_i – m_hat_i`
   * `W˜_i = log_price_actual_i – g_hat_i`
4. **Estimate Final Elasticity**

   * Regress $Y˜_i$ on $W˜_i$ without intercept:

     ```
     Y˜_i = θ · W˜_i + error_i
     ```
   * `θ̂` is the DML estimate of ∂E\[bookings]/∂ln(price).

#### 2.2 Key Advantages

* No explicit instrument needed; rely on “all confounders are observed.”
* Black‐box ML (LASSO, random forests, boosting, etc.) can capture nonlinearity / interactions.
* Cross‐fitting yields orthogonalization → robust √n‐consistency and valid inference under weak ML‐error conditions.

#### 2.3 Trade-offs vs. IV

| Aspect                     | IV (2SLS)                                                      | DML (Selection-on-Observables)                                   |                                                            |        |
| -------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------- | ------ |
| **Identification**         | \`rate\_recommended ⟂ u\_i                                     | X\_i\`                                                           | \`{bookings\_i(0), bookings\_i(1)} ⟂ log\_price\_actual\_i | X\_i\` |
| **Key Strength**           | Valid if instrument truly exogenous (manager never sees leaks) | No instrument needed; rich observables suffice if credible       |                                                            |        |
| **Key Weakness**           | Must justify exclusion; leaks break validity                   | Unobserved confounders bias estimates                            |                                                            |        |
| **Functional Flexibility** | Linear/IV‐logit; may mis‐specify nonlinearities                | Arbitrary ML for first‐stage nuisance fits                       |                                                            |        |
| **Communication**          | Standard econ interpretation (“using recommended price as IV”) | “Black‐box” DML—requires stakeholder education                   |                                                            |        |
| **Policy Simulation**      | Easy: recover demand curve from β\_price → revenue rule        | Can derive marginal demand but may need nonparametric smoothing  |                                                            |        |
| **Data Requirements**      | Only need one valid instrument (risk if it weakens)            | Need extremely rich `X_i`—if any confounder omitted, bias arises |                                                            |        |

### 3. Heterogeneity Analysis

Once a baseline elasticity is estimated (either IV or DML), one often asks:

> **“Which types of customers or dates are more price‐sensitive?”**

Below, we outline three dimensions of heterogeneity and how they are implemented:

#### 3.1 Heterogeneity by Lead Time (“Days Ahead”)

* **Motivation**: Business travelers (book 1–3 days ahead) tend to be less price‐sensitive (inelastic), whereas leisure travelers (book 20–60 days ahead) are more elastic.
* **Implementation**: Define lead‐time buckets (e.g., 0–3 days, 4–14 days, 15+ days). In the second stage of IV or DML, interact `ln(rate_actual)` with each bucket dummy:

  $$
    β₁ · (ln(rate_actual_i)·1\{\text{days\_ahead} ∈ 0–3\}) \;+\;
    β₂ · (ln(rate_actual_i)·1\{\text{days\_ahead} ∈ 4–14\}) \;+\;
    β₃ · (ln(rate_actual_i)·1\{\text{days\_ahead} ≥ 15\})
  $$
* **Interpretation**:

  * If $β₁ ≈ -0.15$, “0–3 days” elasticity is –0.15 (inelastic).
  * If $β₃ ≈ -0.60$, “15+ days” elasticity is –0.60 (highly elastic).

#### 3.2 Heterogeneity by Day-of-Week / Seasonality

* **Motivation**: Weekend stays (Fri/Sat/Sun) are typically leisure (less elastic) vs. midweek (Mon–Thu) which are mostly business (more elastic). Holidays or conference weeks can be extremely inelastic.
* **Implementation**: Create a dummy `W_wknd_i = 1 if date_of_stay_i is Fri/Sat/Sun`. In the second stage, interact `ln(rate_actual)` with `W_wknd_i` and `(1 − W_wknd_i)`. Optionally, define a holiday‐dummy `H_i` for major conventions.
* **Interpretation**:

  * If `β_WD = –0.50` (weekday elasticity) and `β_WKND = –0.20` (weekend), then weekend demand is much less sensitive to price increases.

#### 3.3 Heterogeneity by Room/Customer Segment

* **Motivation**: Suites may draw affluent travelers (inelastic), while budget rooms attract price shoppers (elastic). Online Travel Agency (OTA) channels often carry more price‐sensitive customers than direct‐web channels.
* **Implementation**: For each room type or segment $r$, define indicator $R_{i,r}$. In the second stage, use:

  $$
    \sum_{r} β_r \bigl(\ln(rate\_actual_i)·R_{i,r}\bigr)\,.
  $$
* **Interpretation**:

  * If `β_suite = –0.18` vs. `β_budget = –0.60`, suite customers are far less price‐sensitive.
  * One can then calibrate “gap pricing” between categories.

---

## Dependencies

This analysis is conducted entirely in **R (version ≥ 4.0)**. The following packages are required (all available on CRAN):

* **Data Wrangling & Visualization**

  * `tidyverse` (includes `dplyr`, `ggplot2`, `tibble`, etc.)
  * `lubridate` (for date handling)
  * `stargazer` (for nicely formatted regression tables)

* **Econometrics & IV**

  * `AER` (provides `ivreg()` for 2SLS)
  * `lmtest` (for hypothesis tests)
  * `sandwich` (for cluster‐robust variance estimators)
  * `plm` (panel data, if needed)
  * `fixest` (optional, for fast high‐dimensional FE estimation)
  * `modelsummary` (alternative tables if preferred)

* **Machine Learning / Double ML**

  * `glmnet` (LASSO / Elastic Net for continuous and binary outcomes)
  * `broom` (to tidy model outputs)

* **Miscellaneous**

  * `here` (optional, for project‐relative paths)
  * `knitr` + `rmarkdown` (for rendering the `*.Rmd` file)

Install all required packages by running:

```r
install.packages(c(
  "tidyverse", "lubridate", "stargazer", "AER", "lmtest", "sandwich",
  "plm", "fixest", "modelsummary", "glmnet", "broom", "here", "knitr", "rmarkdown"
))
```

---

## Installation & Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/<your-username>/demand-estimation.git
   cd demand-estimation
   ```

2. **Ensure Data Exists**

   * The CSV data file (`demand_est.csv`) should be located in `./data/demand_est.csv`.
   * If you receive a “file not found” error, confirm that the path is correct or update the code chunk in `demand estimation.Rmd` to point to the correct location.

3. **Install R & Required Packages**

   * Use R ≥ 4.0.0 (preferably R 4.2+).
   * From within R (or RStudio), run:

     ```r
     setwd("path/to/demand-estimation")
     source("install_dependencies.R")  # optional script if provided
     ```
   * Or manually install each package as shown above.

---

## Usage

### Running the R Markdown

1. **Open `demand estimation.Rmd` in RStudio** (or any R IDE that supports R Markdown).
2. **Knit to PDF / HTML**

   * Click “Knit” (or run in console):

     ```r
     rmarkdown::render("demand estimation.Rmd", output_format = "pdf_document")
     ```
   * This will re‐run the entire analysis—reading `data/demand_est.csv`, estimating IV and DML models, producing tables/figures, and writing the final PDF.
3. **Inspect Intermediate Objects**

   * If needed, you can run individual code chunks to inspect data frames, first‐stage F‐statistics, DML residuals, or regression coefficients.

### Inspecting the PDF Report

* The compiled report (`demand estimation.pdf`) includes:

  1. A written introduction to the identification challenge.
  2. Full code output for the IV first stage and second stage (LPM & optional logit).
  3. Cluster‐robust standard errors and stargazer tables.
  4. DML cross‐fitting code and the final elasticity estimate (with standard error).
  5. Detailed discussion of heterogeneity by lead‐time, day‐of‐week, and room segment—plus illustrative tables.
* Open it with your preferred PDF viewer:

  ```bash
  open "demand estimation.pdf"
  ```

---

## Results & Interpretation

1. **IV (2SLS) Baseline Elasticity**

   * In many real‐world runs, the coefficient on `log_price_actual` in the second‐stage LPM will appear **positive** (e.g., +0.117), indicating residual endogeneity: when demand spikes, both recommended and actual price rise, causing a spurious positive relationship.
   * One must interpret this as evidence that the instrument did not fully purge unobserved demand shocks in practice.

2. **DML Baseline Elasticity**

   * The DML estimate typically lands **negative but small in magnitude** (e.g., –0.0282 with SE ≈ 0.0139).
   * This suggests a weak but statistically significant downward‐sloping demand: a 1% increase in price reduces booking probability by \~0.03 percentage points, once we flexibly control for hundreds of observed features.

3. **Heterogeneity Findings**

   * **Lead‐Time**: Business travelers (0–3 days ahead) show low elasticity (e.g., –0.12), whereas leisure travelers (15+ days ahead) exhibit much higher elasticity (e.g., –0.60).
   * **Weekday vs. Weekend**: Weekends (Fri–Sun) are more inelastic (e.g., –0.22) versus midweek (–0.50).
   * **Room & Channel**: Suites tend to be least price‐sensitive (e.g., –0.18) while budget rooms or OTA channels can be very elastic (e.g., –0.75).

These insights feed directly into dynamic pricing strategies:

* **Segmented Pricing**: Charge a premium for business‐segment nights (low elasticity) and offer early‐bird discounts for leisure segment.
* **Channel Management**: Focus OTA promotions on high‐elasticity room types; maintain “rack rate” on direct web for suite customers.
* **Seasonal/Tactical Discounts**: Deploy small midweek price cuts to boost occupancy when elasticity is highest.

---

## Contributing

Contributions are welcome! If you wish to:

1. Report a bug, suggest enhancements, or request new features → open an Issue.
2. Propose changes to code, documentation, or add new analysis (e.g., alternative ML methods, updated covariate sets) → fork the repository, create a new branch, implement changes, and submit a Pull Request.

Please ensure that:

* All code chunks in `demand estimation.Rmd` run without errors on a fresh clone.
* Any new R packages are added to the Dependencies section above.
* You adhere to consistent code styling (tidyverse conventions + 2‐space indentation).

---

## License

This repository is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
