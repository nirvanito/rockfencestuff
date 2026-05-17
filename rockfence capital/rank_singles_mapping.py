"""
Generate ATP rank -> singles earnings mapping on an annual basis.
Reads from filtered_top_players_earnings.xlsx (which has per-year sheets).
Outputs:
  1. rank_to_singles_by_year.csv  — full mapping
  2. Summary statistics & aging curve analysis
"""
import pandas as pd
import numpy as np

YEARS = ['2015', '2016', '2017', '2018', '2019', '2022', '2023', '2024', '2025']

# ── 1. Build rank-to-singles mapping per year ──────────────────────────
rows = []
for yr in YEARS:
    df = pd.read_excel('filtered_top_players_earnings.xlsx', sheet_name=yr)
    # Preserve ATP rank from the prepared workbook.
    df = df.sort_values('rank', ascending=True).reset_index(drop=True)
    df['year'] = int(yr)
    rows.append(df[['year', 'rank', 'player name', 'age', 'singles']])

full = pd.concat(rows, ignore_index=True)
full.to_csv('rank_to_singles_by_year.csv', index=False)
print(f"Saved rank_to_singles_by_year.csv — {len(full)} rows\n")

# ── 2. Summary: median/mean singles by rank bucket per year ───────────
print("="*80)
print("SINGLES EARNINGS BY RANK BUCKET (ANNUAL)")
print("="*80)

rank_buckets = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 200), (201, 500), (501, 1000)]

for lo, hi in rank_buckets:
    label = f"Rank {lo}" if lo == hi else f"Rank {lo}-{hi}"
    subset = full[(full['rank'] >= lo) & (full['rank'] <= hi)]
    pivot = subset.groupby('year')['singles'].agg(['mean', 'median', 'count'])
    print(f"\n{label}:")
    for yr in [int(y) for y in YEARS]:
        if yr in pivot.index:
            r = pivot.loc[yr]
            print(f"  {yr}: mean=${r['mean']:>12,.0f}  median=${r['median']:>12,.0f}  (n={int(r['count'])})")

# ── 3. Aging curve: average singles by age ────────────────────────────
print("\n" + "="*80)
print("AGING CURVE: AVERAGE SINGLES EARNINGS BY AGE")
print("="*80)
age_curve = full.groupby('age')['singles'].agg(['mean', 'median', 'count', 'std'])
age_curve = age_curve.sort_index()
for age, r in age_curve.iterrows():
    print(f"  Age {age:2d}: mean=${r['mean']:>10,.0f}  median=${r['median']:>8,.0f}  n={int(r['count']):>4d}  std=${r['std']:>10,.0f}")

# ── 4. Player trajectory analysis (aging curve by cohort) ─────────────
print("\n" + "="*80)
print("PLAYER CAREER TRAJECTORIES (players with 4+ years of data)")
print("="*80)
player_years = full.groupby('player name')['year'].count()
multi_year = player_years[player_years >= 4].index
trajectories = full[full['player name'].isin(multi_year)].copy()
trajectories = trajectories.sort_values(['player name', 'age'])

# Normalize: each player's earnings relative to their own peak
def normalize_career(group):
    peak = group['singles'].max()
    group['pct_of_peak'] = group['singles'] / peak if peak > 0 else 0
    return group

trajectories = trajectories.groupby('player name').apply(normalize_career, include_groups=False).reset_index(drop=True)

# Average normalized career curve by age
norm_curve = trajectories.groupby('age')['pct_of_peak'].agg(['mean', 'median', 'count'])
norm_curve = norm_curve[norm_curve['count'] >= 5]  # at least 5 observations
print("\nNormalized earnings as % of peak, by age:")
for age, r in norm_curve.iterrows():
    bar = '#' * int(r['mean'] * 50)
    print(f"  Age {age:2d}: {r['mean']:.1%} of peak  (n={int(r['count']):>3d})  {bar}")

# ── 5. Distribution analysis: fit lognormal to conditional earnings ───
print("\n" + "="*80)
print("LOGNORMAL FIT TO SINGLES EARNINGS (conditional on >0)")
print("="*80)
for yr in [int(y) for y in YEARS]:
    yr_data = full[(full['year'] == yr) & (full['singles'] > 0)]
    log_earnings = np.log(yr_data['singles'])
    mu, sigma = log_earnings.mean(), log_earnings.std()
    total = len(full[full['year'] == yr])
    nonzero = len(yr_data)
    bust_pct = (total - nonzero) / total * 100
    print(f"  {yr}: mu={mu:.3f}, sigma={sigma:.3f}, "
          f"implied median=${np.exp(mu):>10,.0f}, "
          f"bust rate={bust_pct:.1f}% (n_total={total}, n_earning={nonzero})")

# ── 6. Peak age analysis ─────────────────────────────────────────────
print("\n" + "="*80)
print("PEAK EARNINGS AGE DISTRIBUTION")
print("="*80)
peak_ages = full.loc[full.groupby('player name')['singles'].idxmax()]['age']
print(peak_ages.describe().to_string())
print(f"\nPeak age distribution:")
print(peak_ages.value_counts().sort_index().to_string())
