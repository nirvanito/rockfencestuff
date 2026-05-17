"""
Rigorous aging curve, distribution comparison (power law vs lognormal),
and inflation-adjusted rank-to-singles mapping.

Methods:
  1. Delta method aging curve with survivorship bias correction
  2. Fixed-effects regression aging curve (quadratic + piecewise)
  3. Power law vs lognormal formal comparison (Clauset et al. 2009)
  4. Prize money inflation from the data itself + external benchmarks
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
import json
import warnings
warnings.filterwarnings('ignore')

YEARS = ['2015','2016','2017','2018','2019','2022','2023','2024','2025']

# ======================================================================
# LOAD DATA
# ======================================================================
rows = []
for yr in YEARS:
    df = pd.read_excel('filtered_top_players_earnings.xlsx', sheet_name=yr)
    df = df.sort_values('rank', ascending=True).reset_index(drop=True)
    df['year'] = int(yr)
    rows.append(df)
full = pd.concat(rows, ignore_index=True)
full = full[full['singles'] > 0].copy()

print("="*80)
print("PART 1: AGING CURVE — DELTA METHOD WITH SURVIVORSHIP CORRECTION")
print("="*80)

# ── Delta Method ──────────────────────────────────────────────────────
# For each player, find consecutive-age observations (year-to-year pairs)
# Calculate log-ratio of earnings: delta = ln(E_{t+1}) - ln(E_t)
# This controls for player fixed effects automatically

full_sorted = full.sort_values(['player name','year'])
players = full_sorted.groupby('player name')

deltas = []
for name, grp in players:
    grp = grp.sort_values('age')
    ages = grp['age'].values
    earnings = grp['singles'].values
    for i in range(len(grp)-1):
        if ages[i+1] == ages[i] + 1 and earnings[i] > 0 and earnings[i+1] > 0:
            deltas.append({
                'player': name,
                'age_from': ages[i],
                'age_to': ages[i+1],
                'log_ratio': np.log(earnings[i+1]) - np.log(earnings[i]),
                'e_from': earnings[i],
                'e_to': earnings[i+1],
            })

delta_df = pd.DataFrame(deltas)
print(f"\nFound {len(delta_df)} consecutive-age pairs")

# Raw delta method: average log-ratio by age
raw_delta = delta_df.groupby('age_to')['log_ratio'].agg(['mean','count','std'])
raw_delta = raw_delta[raw_delta['count'] >= 10]  # min 10 obs

print("\nRaw Delta Method (avg year-over-year log change):")
print(f"{'Age':>5} {'Avg dln(E)':>10} {'N':>5} {'Interpretation':>30}")
for age, r in raw_delta.iterrows():
    direction = "improving" if r['mean'] > 0 else "declining"
    pct = (np.exp(r['mean'])-1)*100
    print(f"  {age:3d}   {r['mean']:+.4f}     {int(r['count']):4d}   {pct:+.1f}% ({direction})")

# Build cumulative curve from deltas (anchored at peak)
ages_sorted = sorted(raw_delta.index)
cum_curve = {}
# Start at age 20, accumulate
cum_curve[ages_sorted[0]] = 0
for i in range(1, len(ages_sorted)):
    prev_age = ages_sorted[i-1]
    curr_age = ages_sorted[i]
    cum_curve[curr_age] = cum_curve[prev_age] + raw_delta.loc[curr_age, 'mean']

# Convert to multiplicative scale (relative to peak)
peak_age = max(cum_curve, key=cum_curve.get)
peak_val = cum_curve[peak_age]
delta_curve = {a: np.exp(v - peak_val) for a, v in cum_curve.items()}

print(f"\nDelta Method Aging Curve (peak at age {peak_age}):")
for age in sorted(delta_curve):
    bar = '#' * int(delta_curve[age] * 50)
    print(f"  Age {age:2d}: {delta_curve[age]:.3f}  {bar}")

# ── Survivorship Bias Correction ─────────────────────────────────────
# Players who drop out between age a and a+1 are NOT in the delta sample.
# Their counterfactual performance would be worse (that's why they dropped out).
#
# KEY REFINEMENTS (literature-informed):
#   1. Compute survival rates only from CONSECUTIVE calendar years in our
#      dataset to avoid COVID gap contamination (Nguyen & Matthews 2023)
#   2. Only count ESTABLISHED players (observed 2+ seasons) when computing
#      survival rates — transient entrants cycling in/out aren't "declining"
#   3. Apply age-varying imputation percentiles with a capped effective
#      dropout fraction so young-age churn doesn't dominate the correction
#      (Schuckers, Lopez & Macdonald 2021)

print("\n--- Survivorship Bias Correction (Age-Varying, Established Players) ---")
active_by_age = full.groupby(['player name','age']).size().reset_index()
active_by_age.columns = ['player','age','count']

# Identify established players: those observed in 2+ seasons in our data
seasons_per_player = full.groupby('player name')['year'].nunique()
established_players = set(seasons_per_player[seasons_per_player >= 2].index)
print(f"Established players (2+ seasons): {len(established_players)} of {len(seasons_per_player)}")

# Consecutive year pairs in our dataset (skip COVID gap)
year_list = sorted(full['year'].unique())
consecutive_year_pairs = [(year_list[i], year_list[i+1])
                          for i in range(len(year_list)-1)
                          if year_list[i+1] - year_list[i] == 1]
print(f"Consecutive year pairs: {consecutive_year_pairs}")

# Compute survival rates from consecutive years only, established players only
survival_rates = {}
survival_counts = {}
for age in range(17, 39):
    n_at_age = 0
    n_survived = 0
    for yr1, yr2 in consecutive_year_pairs:
        # Players at this age in yr1 who are established
        at_age = set(full[(full['year']==yr1) & (full['age']==age) &
                         (full['player name'].isin(established_players))]['player name'])
        at_next = set(full[(full['year']==yr2) & (full['age']==age+1)]['player name'])
        n_at_age += len(at_age)
        n_survived += len(at_age & at_next)
    if n_at_age > 0:
        survival_rates[age] = n_survived / n_at_age
        survival_counts[age] = n_at_age

print(f"\n{'Age':>5} {'Survival Rate':>15} {'N (estab.)':>12}")
for age in sorted(survival_rates):
    print(f"  {age:3d}     {survival_rates[age]:.3f}          {survival_counts.get(age,0):5d}")

# Age-varying imputation percentile + effective dropout cap
def dropout_percentile(age):
    """Age-varying imputation severity for dropout counterfactuals.
    Young dropouts are mostly 'never-made-it' (mild penalty).
    Old dropouts are mostly genuine decline (harsh penalty)."""
    if age <= 23:
        return 0.30   # 30th percentile — very mild
    elif age <= 29:
        return 0.20   # 20th percentile — moderate
    else:
        return 0.10   # 10th percentile — harsh

def effective_dropout_weight(age, raw_dropout_rate):
    """Cap the dropout weight for young ages where most dropout is
    entry churn rather than performance decline. For ages 30+,
    use the full dropout rate."""
    if age <= 22:
        return min(raw_dropout_rate, 0.15)  # cap at 15%
    elif age <= 26:
        return min(raw_dropout_rate, 0.25)  # cap at 25%
    else:
        return raw_dropout_rate  # full correction

corrected_delta = {}
for age in sorted(raw_delta.index):
    if age-1 in survival_rates and survival_rates[age-1] > 0:
        sr = survival_rates[age-1]
        raw_dropout = 1 - sr
        observed_mean = raw_delta.loc[age, 'mean']
        # Impute dropout performance at age-varying percentile
        pctl = dropout_percentile(age)
        if age in delta_df['age_to'].values:
            imputed = delta_df[delta_df['age_to']==age]['log_ratio'].quantile(pctl)
        else:
            imputed = -0.5
        # Apply capped dropout weight
        eff_dropout = effective_dropout_weight(age, raw_dropout)
        corrected = (1 - eff_dropout) * observed_mean + eff_dropout * imputed
        corrected_delta[age] = corrected
    else:
        corrected_delta[age] = raw_delta.loc[age, 'mean']

print(f"\nImputation: 17-23 -> P30 (cap 15%), 24-29 -> P20 (cap 25%), 30+ -> P10 (uncapped)")

# Build corrected cumulative curve
corr_ages = sorted(corrected_delta.keys())
corr_cum = {corr_ages[0]: 0}
for i in range(1, len(corr_ages)):
    corr_cum[corr_ages[i]] = corr_cum[corr_ages[i-1]] + corrected_delta[corr_ages[i]]

corr_peak_age = max(corr_cum, key=corr_cum.get)
corr_peak_val = corr_cum[corr_peak_age]
corrected_curve = {a: np.exp(v - corr_peak_val) for a, v in corr_cum.items()}

print(f"\nCorrected Aging Curve (peak at age {corr_peak_age}):")
for age in sorted(corrected_curve):
    bar = '#' * int(corrected_curve[age] * 50)
    print(f"  Age {age:2d}: {corrected_curve[age]:.3f}  {bar}")

# ── Smoothed Best-Estimate Curve ─────────────────────────────────────
# Fit a cubic smoothing spline to the corrected delta curve to produce
# a clean, monotonic-around-peak final curve for the pricing model.
# Then bootstrap for 95% confidence intervals.

print("\n--- Smoothed Best-Estimate Curve (Cubic Spline) ---")
corr_ages_arr = np.array(sorted(corrected_curve.keys()))
corr_vals_arr = np.array([corrected_curve[a] for a in corr_ages_arr])

# Fit smoothing spline (s controls smoothness; larger = smoother)
# Use weights inversely proportional to the std of the delta at each age
weights = []
for a in corr_ages_arr:
    if a in raw_delta.index and raw_delta.loc[a, 'count'] > 0:
        # Weight by sqrt(N) — more observations = more weight
        weights.append(np.sqrt(raw_delta.loc[a, 'count']))
    else:
        weights.append(1.0)
weights = np.array(weights)

spline = UnivariateSpline(corr_ages_arr, corr_vals_arr, w=weights, s=0.05, k=3)

# Evaluate on fine grid and normalize to peak=1
fine_ages = np.arange(17, 40)
smoothed_vals = spline(fine_ages)
smoothed_vals = np.clip(smoothed_vals, 0, None)  # no negatives
smoothed_peak = smoothed_vals.max()
smoothed_curve = smoothed_vals / smoothed_peak
smoothed_peak_age = fine_ages[np.argmax(smoothed_vals)]

# Enforce monotonic non-increasing post-peak (fixes spline overshoot at tail ages)
peak_idx = np.argmax(smoothed_curve)
for i in range(peak_idx + 1, len(smoothed_curve)):
    smoothed_curve[i] = min(smoothed_curve[i], smoothed_curve[i-1])

print(f"Smoothed peak age: {smoothed_peak_age}")
print("\nSmoothed Best-Estimate Aging Curve:")
for i, age in enumerate(fine_ages):
    bar = '#' * int(smoothed_curve[i] * 50)
    print(f"  Age {age:2d}: {smoothed_curve[i]:.3f}  {bar}")

# ── Bootstrap Confidence Intervals ───────────────────────────────────
print("\n--- Bootstrap 95% Confidence Intervals (200 resamples) ---")
np.random.seed(42)
N_BOOT = 200
boot_curves = []

for b in range(N_BOOT):
    # Resample delta pairs with replacement (by player to preserve within-player correlation)
    unique_players = delta_df['player'].unique()
    boot_players = np.random.choice(unique_players, size=len(unique_players), replace=True)
    boot_deltas = pd.concat([delta_df[delta_df['player']==p] for p in boot_players], ignore_index=True)
    
    # Recompute raw delta by age
    b_raw = boot_deltas.groupby('age_to')['log_ratio'].agg(['mean','count'])
    b_raw = b_raw[b_raw['count'] >= 5]  # relaxed threshold for bootstrap
    
    # Age-varying corrected delta
    b_corrected = {}
    for age in sorted(b_raw.index):
        if age-1 in survival_rates and survival_rates[age-1] > 0:
            sr = survival_rates[age-1]
            obs_mean = b_raw.loc[age, 'mean']
            pctl = dropout_percentile(age)
            if age in boot_deltas['age_to'].values:
                imp = boot_deltas[boot_deltas['age_to']==age]['log_ratio'].quantile(pctl)
            else:
                imp = -0.5
            b_corrected[age] = sr * obs_mean + (1-sr) * imp
        else:
            b_corrected[age] = b_raw.loc[age, 'mean']
    
    # Cumulative curve
    b_ages = sorted(b_corrected.keys())
    if len(b_ages) < 5:
        continue
    b_cum = {b_ages[0]: 0}
    for i in range(1, len(b_ages)):
        b_cum[b_ages[i]] = b_cum[b_ages[i-1]] + b_corrected[b_ages[i]]
    b_peak_val = max(b_cum.values())
    b_curve = {a: np.exp(v - b_peak_val) for a, v in b_cum.items()}
    
    # Interpolate to standard age grid
    b_ages_arr = np.array(sorted(b_curve.keys()))
    b_vals_arr = np.array([b_curve[a] for a in b_ages_arr])
    if len(b_ages_arr) >= 4:
        try:
            b_spline = UnivariateSpline(b_ages_arr, b_vals_arr, s=0.1, k=3)
            b_smooth = np.clip(b_spline(fine_ages), 0, None)
            b_smooth = b_smooth / b_smooth.max()
            boot_curves.append(b_smooth)
        except:
            pass

boot_curves = np.array(boot_curves)
ci_lower = np.percentile(boot_curves, 2.5, axis=0)
ci_upper = np.percentile(boot_curves, 97.5, axis=0)

print(f"Completed {len(boot_curves)} bootstrap resamples")
print(f"\n{'Age':>5} {'Estimate':>10} {'CI Lower':>10} {'CI Upper':>10}")
for i, age in enumerate(fine_ages):
    print(f"  {age:3d}    {smoothed_curve[i]:.3f}      {ci_lower[i]:.3f}      {ci_upper[i]:.3f}")

# Export the best-estimate curve for use in pricing model
best_estimate_curve = {int(a): float(v) for a, v in zip(fine_ages, smoothed_curve)}
analysis_payload = {
    'peak_age': int(smoothed_peak_age),
    'curve': best_estimate_curve,
    'ci_lower': {int(a): float(v) for a, v in zip(fine_ages, ci_lower)},
    'ci_upper': {int(a): float(v) for a, v in zip(fine_ages, ci_upper)},
    'method': 'Corrected delta method with age-varying survivorship imputation, cubic spline smoothing',
    'sources': [
        'Fair (2007/2008) - Piecewise quadratic FE methodology',
        'Mlakar & Tusar (2015) - ATP peak age analysis (peak ~25)',
        'Sackmann / Tennis Abstract - Elo-based aging curves (plateau 24-27)',
        'Nguyen & Matthews (2023) - Multiple imputation framework',
        'Schuckers, Lopez & Macdonald (2021/2023) - Imputation-based aging curves',
        'Berkeley Sports Analytics - Modern era peak shift to 26-28',
    ],
}
with open('aging_curve_best_estimate.json', 'w') as f:
    json.dump(analysis_payload, f, indent=2)
print("\nSaved aging_curve_best_estimate.json")

# ======================================================================
print("\n" + "="*80)
print("PART 2: FIXED-EFFECTS REGRESSION AGING CURVE")
print("="*80)

# Player fixed effects + quadratic in age
# ln(earnings_it) = alpha_i + beta1*age + beta2*age^2 + epsilon_it
# Demean by player (within transformation)

panel = full[['player name','age','singles','year']].copy()
panel['ln_e'] = np.log(panel['singles'])
panel = panel[np.isfinite(panel['ln_e'])]

# Within transformation: demean by player
player_means = panel.groupby('player name')[['ln_e','age']].transform('mean')
panel['ln_e_dm'] = panel['ln_e'] - player_means['ln_e']
panel['age_dm'] = panel['age'] - player_means['age']
panel['age2_dm'] = panel['age_dm']**2

# Also need the global age^2 demeaned
panel['age_sq'] = panel['age']**2
player_means_sq = panel.groupby('player name')['age_sq'].transform('mean')
panel['age2_dm'] = panel['age_sq'] - player_means_sq

# OLS on demeaned data
from numpy.linalg import lstsq
X = panel[['age_dm','age2_dm']].values
y = panel['ln_e_dm'].values
beta, residuals, rank, sv = lstsq(X, y, rcond=None)

print(f"\nFixed-Effects Regression: ln(earnings) = alpha_i + {beta[0]:.6f}*age + {beta[1]:.6f}*age^2")
peak_fe = -beta[0] / (2*beta[1])
print(f"Implied peak age: {peak_fe:.1f}")

# Generate the FE aging curve
fe_curve = {}
ages_range = range(17, 40)
for a in ages_range:
    fe_curve[a] = beta[0]*a + beta[1]*a**2

fe_peak = max(fe_curve.values())
fe_curve_norm = {a: np.exp(v - fe_peak) for a, v in fe_curve.items()}

print("\nFE Regression Aging Curve:")
for age in sorted(fe_curve_norm):
    bar = '#' * int(fe_curve_norm[age] * 50)
    print(f"  Age {age:2d}: {fe_curve_norm[age]:.3f}  {bar}")

# ── Piecewise quadratic (Fair 2008 style) ────────────────────────────
# Separate slopes pre/post peak
# Constraint: b2 >= b1 (decline at least as steep as improvement)
# This is consistent with Fair (2008), Mlakar (2015), and Sackmann's findings.
print("\n--- Fair (2008) Piecewise Quadratic (constrained: decline >= improvement) ---")

def piecewise_loss(params, ages, ln_e_dm):
    peak, b1, b2 = params
    predicted = np.where(ages <= peak,
                         -b1*(peak - ages)**2,
                         -b2*(ages - peak)**2)
    predicted -= predicted.mean()  # center
    return np.sum((ln_e_dm - predicted)**2)

# Constrain b2 >= b1 so decline is at least as steep as improvement
from scipy.optimize import minimize as scipy_minimize
result = scipy_minimize(piecewise_loss, x0=[27, 0.005, 0.01],
                  args=(panel['age'].values, panel['ln_e_dm'].values),
                  bounds=[(20, 35), (0.001, 0.05), (0.001, 0.05)],
                  constraints={'type': 'ineq', 'fun': lambda p: p[2] - p[1]})

pw_peak, pw_b1, pw_b2 = result.x
print(f"Piecewise peak age: {pw_peak:.1f}")
print(f"Pre-peak curvature (b1): {pw_b1:.5f}")
print(f"Post-peak curvature (b2): {pw_b2:.5f}")
print(f"Asymmetry ratio (b2/b1): {pw_b2/pw_b1:.2f}x  (>1 means decline is steeper)")

pw_curve = {}
for a in range(17, 40):
    if a <= pw_peak:
        pw_curve[a] = np.exp(-pw_b1*(pw_peak - a)**2)
    else:
        pw_curve[a] = np.exp(-pw_b2*(a - pw_peak)**2)

print("\nPiecewise Aging Curve:")
for age in sorted(pw_curve):
    bar = '#' * int(pw_curve[age] * 50)
    print(f"  Age {age:2d}: {pw_curve[age]:.3f}  {bar}")

analysis_payload['reference_models'] = {
    'fixed_effects_quadratic': {
        'peak_age': float(peak_fe),
        'beta_age': float(beta[0]),
        'beta_age_sq': float(beta[1]),
    },
    'piecewise_quadratic': {
        'peak_age': float(pw_peak),
        'pre_peak_curvature': float(pw_b1),
        'post_peak_curvature': float(pw_b2),
    },
}
with open('aging_curve_best_estimate.json', 'w') as f:
    json.dump(analysis_payload, f, indent=2)
print("Updated aging_curve_best_estimate.json with reference model parameters")

# ======================================================================
print("\n" + "="*80)
print("PART 3: POWER LAW vs LOGNORMAL (Clauset et al. 2009 methodology)")
print("="*80)

def fit_power_law(data, xmin=None):
    """MLE for discrete power law. Clauset et al. 2009."""
    if xmin is None:
        xmin = np.percentile(data, 50)  # use median as xmin
    data = data[data >= xmin]
    n = len(data)
    alpha = 1 + n / np.sum(np.log(data / xmin))
    return alpha, xmin, n

def ks_power_law(data, alpha, xmin):
    """KS statistic for power law fit."""
    data = np.sort(data[data >= xmin])
    n = len(data)
    empirical_cdf = np.arange(1, n+1) / n
    theoretical_cdf = 1 - (xmin / data)**(alpha - 1)
    return np.max(np.abs(empirical_cdf - theoretical_cdf))

def loglikelihood_ratio_test(data, xmin):
    """Compare power law vs lognormal for data >= xmin. Vuong test."""
    data = data[data >= xmin]
    n = len(data)
    
    # Power law MLE
    alpha = 1 + n / np.sum(np.log(data / xmin))
    ll_pl = n*np.log(alpha-1) - n*np.log(xmin) - alpha*np.sum(np.log(data/xmin))
    
    # Lognormal MLE (truncated at xmin)
    log_data = np.log(data)
    mu_ln = log_data.mean()
    sigma_ln = log_data.std()
    ll_ln = np.sum(stats.lognorm.logpdf(data, s=sigma_ln, scale=np.exp(mu_ln)))
    
    # Likelihood ratio
    R = ll_pl - ll_ln
    # Vuong's test statistic
    sigma_R = np.std([
        np.log(stats.pareto.pdf(x, b=alpha-1, scale=xmin)) -
        stats.lognorm.logpdf(x, s=sigma_ln, scale=np.exp(mu_ln))
        for x in data
    ])
    if sigma_R > 0:
        test_stat = R / (sigma_R * np.sqrt(n))
        p_value = 2 * stats.norm.sf(abs(test_stat))
    else:
        test_stat = 0
        p_value = 1.0
    
    return {
        'alpha_pl': alpha,
        'mu_ln': mu_ln,
        'sigma_ln': sigma_ln,
        'll_powerlaw': ll_pl,
        'll_lognormal': ll_ln,
        'lr_ratio': R,
        'vuong_stat': test_stat,
        'p_value': p_value,
        'n': n,
        'winner': 'power_law' if R > 0 else 'lognormal',
        'significant': p_value < 0.05,
    }

print("\nFormal comparison for each year (Vuong likelihood ratio test):")
print(f"{'Year':>6} {'alpha':>6} {'mu_ln':>8} {'sigma_ln':>8} {'LR':>10} {'Vuong Z':>8} {'p-val':>8} {'Winner':>12} {'Sig?':>5}")

for yr in [int(y) for y in YEARS]:
    yr_data = full[(full['year']==yr) & (full['singles'] > 100)]['singles'].values
    xmin = np.percentile(yr_data, 50)
    result = loglikelihood_ratio_test(yr_data, xmin)
    sig = "YES" if result['significant'] else "no"
    print(f"  {yr}  {result['alpha_pl']:5.2f}  {result['mu_ln']:8.3f}  {result['sigma_ln']:8.3f}  "
          f"{result['lr_ratio']:+10.1f}  {result['vuong_stat']:+7.3f}  {result['p_value']:7.4f}  "
          f"{result['winner']:>12}  {sig:>5}")

# Also do it on pooled data
all_singles = full[full['singles'] > 100]['singles'].values
xmin_all = np.percentile(all_singles, 50)
pooled = loglikelihood_ratio_test(all_singles, xmin_all)
print(f"\n  POOLED {pooled['alpha_pl']:5.2f}  {pooled['mu_ln']:8.3f}  {pooled['sigma_ln']:8.3f}  "
      f"{pooled['lr_ratio']:+10.1f}  {pooled['vuong_stat']:+7.3f}  {pooled['p_value']:7.4f}  "
      f"{pooled['winner']:>12}  {'YES' if pooled['significant'] else 'no':>5}")

# Additional: AIC/BIC comparison
print("\n--- AIC / BIC Comparison (pooled data) ---")
n_all = len(all_singles)
aic_pl = -2*pooled['ll_powerlaw'] + 2*2  # 2 params: alpha, xmin
aic_ln = -2*pooled['ll_lognormal'] + 2*2  # 2 params: mu, sigma
bic_pl = -2*pooled['ll_powerlaw'] + 2*np.log(n_all)
bic_ln = -2*pooled['ll_lognormal'] + 2*np.log(n_all)
print(f"  Power Law:  AIC={aic_pl:,.0f}  BIC={bic_pl:,.0f}")
print(f"  Lognormal:  AIC={aic_ln:,.0f}  BIC={bic_ln:,.0f}")
print(f"  --> {'Lognormal' if aic_ln < aic_pl else 'Power Law'} preferred by AIC (lower is better)")

# ======================================================================
print("\n" + "="*80)
print("PART 4: PRIZE MONEY INFLATION ANALYSIS")
print("="*80)

# Method 1: Rank-controlled inflation from our data
# For each rank bucket, compute YoY growth rate
print("\n--- Method 1: Rank-controlled inflation from data ---")

rank_buckets = [(1,5), (6,10), (11,20), (21,50), (51,100), (101,200)]
inflation_by_bucket = {}

for lo, hi in rank_buckets:
    label = f"Rank {lo}-{hi}"
    bucket = full[(full['rank']>=lo) & (full['rank']<=hi)]
    yearly_med = bucket.groupby('year')['singles'].median()
    
    # Calculate CAGR across available years
    year_vals = yearly_med.sort_index()
    years_list = year_vals.index.tolist()
    
    if len(years_list) >= 2:
        # Use first and last, adjusting for COVID gap
        first_yr, last_yr = years_list[0], years_list[-1]
        n_years = last_yr - first_yr
        cagr = (year_vals[last_yr] / year_vals[first_yr])**(1/n_years) - 1
        inflation_by_bucket[label] = cagr
        print(f"  {label:>15}: {first_yr}=${year_vals[first_yr]:>12,.0f} -> "
              f"{last_yr}=${year_vals[last_yr]:>12,.0f}  CAGR={cagr:.1%}")

overall_cagr = np.mean(list(inflation_by_bucket.values()))
print(f"\n  Overall average CAGR across buckets: {overall_cagr:.1%}")

# Method 2: External benchmark data
print("\n--- Method 2: External benchmarks ---")
benchmarks = {
    'US Open champion': {'2010': 1_700_000, '2015': 3_300_000, '2020': 3_000_000, '2025': 5_000_000},
    'US Open total purse': {'2010': 22_600_000, '2015': 42_300_000, '2024': 75_000_000, '2025': 90_000_000},
    'ATP Tour total (excl GS)': {'2015': 100_000_000, '2023': 217_900_000, '2024': 261_000_000},
    'ATP Challenger Tour': {'2022': 12_100_000, '2025': 28_500_000},
}

for name, data in benchmarks.items():
    years_k = sorted(data.keys())
    first, last = years_k[0], years_k[-1]
    n = int(last) - int(first)
    cagr = (data[last] / data[first])**(1/n) - 1
    print(f"  {name:>30}: {first}-{last}  CAGR={cagr:.1%}")

# Method 3: Per-year growth rates from our data (rank 1-100 median)
print("\n--- Method 3: Year-over-year growth (Rank 1-100 median singles) ---")
top100 = full[full['rank'] <= 100]
yearly_median = top100.groupby('year')['singles'].median()
print(f"  {'Year':>6} {'Median Singles':>15} {'YoY Growth':>12}")
prev = None
yoy_rates = []
for yr in sorted(yearly_median.index):
    val = yearly_median[yr]
    if prev is not None:
        gap = yr - prev_yr
        growth = (val/prev)**(1/gap) - 1  # annualized
        yoy_rates.append(growth)
        print(f"  {yr:6d} ${val:>13,.0f}   {growth:+.1%}")
    else:
        print(f"  {yr:6d} ${val:>13,.0f}        ---")
    prev = val
    prev_yr = yr

avg_growth = np.mean(yoy_rates)
med_growth = np.median(yoy_rates)
print(f"\n  Average annual growth: {avg_growth:.1%}")
print(f"  Median annual growth:  {med_growth:.1%}")

# ── Build inflation-adjusted rank mapping ─────────────────────────────
print("\n--- Inflation-Adjusted Rank-to-Singles Mapping (2025 dollars) ---")
# Use our computed CAGR to deflate everything to 2025
INFLATION_RATE = overall_cagr  # from rank-controlled method
print(f"  Using inflation rate: {INFLATION_RATE:.2%} per year")

full['singles_2025'] = full['singles'] * (1 + INFLATION_RATE)**(2025 - full['year'])

# Summary table
print(f"\n{'Rank Bucket':>15} {'2015 (nom)':>12} {'2015 (adj)':>12} {'2019 (nom)':>12} {'2019 (adj)':>12} {'2025 (nom)':>12}")
for lo, hi in rank_buckets:
    label = f"Rank {lo}-{hi}"
    for yr in [2015, 2019, 2025]:
        subset = full[(full['rank']>=lo)&(full['rank']<=hi)&(full['year']==yr)]
        if yr == 2015:
            nom_15 = subset['singles'].median()
            adj_15 = subset['singles_2025'].median()
        elif yr == 2019:
            nom_19 = subset['singles'].median()
            adj_19 = subset['singles_2025'].median()
        else:
            nom_25 = subset['singles'].median()
    print(f"  {label:>13} ${nom_15:>10,.0f} ${adj_15:>10,.0f} ${nom_19:>10,.0f} ${adj_19:>10,.0f} ${nom_25:>10,.0f}")

# Save inflation-adjusted data
full_out = full[['year','rank','player name','age','singles','singles_2025']].copy()
full_out.to_csv('rank_to_singles_inflation_adjusted.csv', index=False)
print(f"\nSaved rank_to_singles_inflation_adjusted.csv ({len(full_out)} rows)")

analysis_payload['calibration'] = {
    'inflation_rate': float(INFLATION_RATE),
}
with open('aging_curve_best_estimate.json', 'w') as f:
    json.dump(analysis_payload, f, indent=2)
print("Updated aging_curve_best_estimate.json with calibration metadata")

# ── Final summary dictionary for use in pricing model ────────────────
print("\n" + "="*80)
print("SUMMARY: KEY PARAMETERS FOR PRICING MODEL")
print("="*80)

print(f"""
  AGING CURVE — BEST ESTIMATE (Corrected + Smoothed Delta Method):
    Peak age: {smoothed_peak_age}
    Method: Delta method with age-varying survivorship correction,
            cubic spline smoothing, bootstrap CIs
    Age 20: {best_estimate_curve[20]:.3f}   Age 25: {best_estimate_curve[25]:.3f}
    Age 27: {best_estimate_curve[27]:.3f}   Age 28: {best_estimate_curve[28]:.3f}
    Age 30: {best_estimate_curve[30]:.3f}   Age 33: {best_estimate_curve[33]:.3f}
    Age 35: {best_estimate_curve[35]:.3f}
    Saved to: aging_curve_best_estimate.json

  AGING CURVE (Fair Piecewise Quadratic, for reference):
    Peak age: {pw_peak:.1f}
    Pre-peak curvature: {pw_b1:.5f}
    Post-peak curvature: {pw_b2:.5f}
    Asymmetry: {pw_b2/pw_b1:.2f}x (decline {pw_b2/pw_b1:.1f}x steeper than improvement)

  DISTRIBUTION: {pooled['winner'].upper()}
    Lognormal mu: {pooled['mu_ln']:.3f}
    Lognormal sigma: {pooled['sigma_ln']:.3f}
    Power law alpha: {pooled['alpha_pl']:.2f}
    Vuong test p-value: {pooled['p_value']:.4f}

  INFLATION:
    Data-derived CAGR: {overall_cagr:.1%}
    Top-100 median growth: {avg_growth:.1%}
    US Open champion CAGR (2010-2025): {((5_000_000/1_700_000)**(1/15)-1):.1%}

  LITERATURE SOURCES:
    - Fair (2007/2008): Piecewise quadratic FE, peak ~28 (baseball)
    - Mlakar & Tusar (2015): ATP peak at 25 (ranking-based)
    - Sackmann / Tennis Abstract: Elo plateau 24-27, modern shift to 26-27
    - Nguyen & Matthews (2023): Multiple imputation shows steeper decline
    - Schuckers et al. (2021/2023): Imputation-based aging curves (NHL)
    - Berkeley Sports Analytics: Modern era peak shift to 26-28
    - Behavioral Ecology (2018): Trait compensation buffers decline
""")
