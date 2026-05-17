## 📂 Files

### 1. Data Acquisition & Cleaning
* **`scraper.py`**: Web scraper that pulls historical year-end ATP Singles Rankings (2000-2025).
* **`clean_earnings.py`**: Parses and cleans raw ATP prize money files, separating singles, doubles, and YTD metrics.
* **`prepare_data.py`**: Merges the cleaned ATP rankings with the prize money data using fuzzy string matching to handle name variations, producing a master analytical dataset.

### 2. Quantitative Modeling
* **`rank_singles_mapping.py`**: Establishes the relationship between actual ATP rank and singles earnings across historical years.
* **`aging_curve_analysis.py`**: The core statistical engine. It calculates a "Delta Method" aging curve, applies age-varying survivorship bias corrections (so players dropping off tour don't artificially inflate the curve), tests Lognormal vs. Power-Law distributions (using the Vuong test), and normalizes historical data for prize money inflation.
* **`earnings_dist_2025.py`**: Detailed snapshot analysis of the 2025 earnings distribution, showcasing inequality (Lorenz curve) and rank-tier breakdowns.

### 3. Simulation & Pricing Engine
* **`simulation_pricing.py`**: The final actuarial pricing model. Runs a 10,000-iteration Monte Carlo simulation for a portfolio of players, factoring in calibrated lognormal distributions, bust probabilities, aging curves, and inflation to determine the exact purchase price required to hit a target IRR (e.g., 15%).

### 4. Visualization & Outputs
* **`plot_results.py`**: Generates publication-ready visualizations of the data.
* **`aging_curve_best_estimate.json`**: The calibrated, smoothed aging curve multipliers.
* **`*.png`**: The generated charts (`aging_curves.png`, `distribution_fit.png`, `earnings_dist_2025.png`, `inflation_trends.png`).

---

## 🚀 How to Run the Pipeline

The pipeline is designed to be run sequentially from raw data extraction to final portfolio pricing:

1. **Get Raw Data**: Run `scraper.py` and `clean_earnings.py` to get the base rankings and money files.
2. **Merge**: Run `python prepare_data.py` to create the master mapped file (`filtered_top_players_earnings.xlsx`).
3. **Map Ranks**: Run `python rank_singles_mapping.py` to aggregate the ranks.
4. **Calibrate Models**: Run `python aging_curve_analysis.py` to compute the aging curve multipliers, fit the distributions, and calculate the prize pool inflation rate.
5. **Generate Visuals**: Run `python earnings_dist_2025.py` and `python plot_results.py` to output the charts.
6. **Price the Portfolio**: Run `python simulation_pricing.py` to perform the Monte Carlo simulation and output the final target pricing.

---

## 📈 Key Findings & Calibrations
* **Earnings Distribution**: While the extreme right tail exhibits Power-Law behavior (Gini index > 0.8), the conditional mean is heavily lognormal. Calibrated parameters: $\mu = 10.32$, $\sigma = 1.76$.
* **Aging Curve**: Earnings potential peaks precisely at **age 26**. The model corrects for survivorship bias to ensure the post-peak decline is modeled accurately (clamped monotonically).
* **Inflation**: The overall prize pool has experienced an annualized inflation rate of **~7.8%** since 2015.
* **Portfolio Pricing**: Assuming a baseline 5% bust rate for ranked 18-year-olds and a target IRR of 15%, the fair market price to purchase 1% of a player's future tournament earnings is approximately **$10,500**.
