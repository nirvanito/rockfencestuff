import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load 2025 data
df = pd.read_excel('filtered_top_players_earnings.xlsx', sheet_name='2025')
df = df.sort_values('singles', ascending=False).reset_index(drop=True)
df['rank'] = range(1, len(df)+1)

# Filter to players who actually earned singles prize money
singles = df[df['singles'] > 0]['singles'].values
log_singles = np.log(singles)

print(f"Total players in 2025 sheet: {len(df)}")
print(f"Players with singles earnings > 0: {len(singles)}")
print(f"\n--- 2025 Singles Earnings Summary ---")
print(f"  Max   : ${singles.max():>12,.0f}  ({df.iloc[0]['player name']})")
print(f"  P99   : ${np.percentile(singles, 99):>12,.0f}")
print(f"  P95   : ${np.percentile(singles, 95):>12,.0f}")
print(f"  P90   : ${np.percentile(singles, 90):>12,.0f}")
print(f"  P75   : ${np.percentile(singles, 75):>12,.0f}")
print(f"  Median: ${np.median(singles):>12,.0f}")
print(f"  P25   : ${np.percentile(singles, 25):>12,.0f}")
print(f"  Mean  : ${np.mean(singles):>12,.0f}")
print(f"  Std   : ${np.std(singles):>12,.0f}")
print(f"  Min   : ${singles.min():>12,.0f}")
print(f"\n--- Rank Tier Breakdown ---")
tiers = [(1,5,'Top 5'), (6,10,'Top 6-10'), (11,20,'Top 11-20'),
         (21,50,'Top 21-50'), (51,100,'Top 51-100'),
         (101,200,'Top 101-200'), (201,500,'Top 201-500'), (501,1000,'Top 501-1000')]
for lo, hi, label in tiers:
    subset = df[(df['rank']>=lo) & (df['rank']<=hi)]['singles']
    subset = subset[subset > 0]
    if len(subset):
        print(f"  {label:<18}: mean=${subset.mean():>10,.0f}  median=${subset.median():>10,.0f}  n={len(subset)}")

# Fit distributions
mu_ln, sigma_ln = log_singles.mean(), log_singles.std()
print(f"\n--- Lognormal Fit ---")
print(f"  mu (log): {mu_ln:.4f}   =>  median = ${np.exp(mu_ln):,.0f}")
print(f"  sigma (log): {sigma_ln:.4f}")

xmin = np.median(singles)
tail = singles[singles >= xmin]
alpha = 1 + len(tail) / np.sum(np.log(tail / xmin))
print(f"\n--- Power Law Fit (tail, xmin=median=${xmin:,.0f}) ---")
print(f"  alpha: {alpha:.4f}")

# Build the figure
fig = plt.figure(figsize=(16, 10), facecolor='#0f0f1a')
gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
ax_main  = fig.add_subplot(gs[0, :])   # log-scale histogram
ax_bar   = fig.add_subplot(gs[1, 0])   # rank tier bar chart
ax_lorenz = fig.add_subplot(gs[1, 1])  # Lorenz curve

ACCENT    = '#7c3aed'
ACCENT2   = '#06b6d4'
ACCENT3   = '#f59e0b'
BG        = '#0f0f1a'
PANEL_BG  = '#1a1a2e'
TEXT      = '#e2e8f0'
GRID      = '#2d2d4e'

def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.7)

# Log-scale histogram + fitted distributions
log10_singles = np.log10(singles)
bins = np.linspace(log10_singles.min(), log10_singles.max(), 40)

n_vals, bin_edges, patches = ax_main.hist(
    log10_singles, bins=bins, density=True,
    color=ACCENT, alpha=0.55, edgecolor='#3b1fa3', linewidth=0.4,
    label='Empirical data (2025, n={})'.format(len(singles))
)

# Color gradient on bars
import matplotlib.cm as cm
cmap = cm.get_cmap('cool')
for i, patch in enumerate(patches):
    patch.set_facecolor(cmap(i / len(patches)))
    patch.set_alpha(0.75)

# Lognormal curve on log10 axis
x_vals = np.linspace(singles.min(), singles.max(), 2000)
pdf_ln = stats.lognorm.pdf(x_vals, s=sigma_ln, scale=np.exp(mu_ln))
pdf_log10_ln = pdf_ln * x_vals * np.log(10)
ax_main.plot(np.log10(x_vals), pdf_log10_ln,
             color=ACCENT2, lw=2.5, label=f'Lognormal fit  (μ={mu_ln:.2f}, σ={sigma_ln:.2f})')

# Power-law curve on log10 axis (tail only)
x_pl = np.linspace(xmin, singles.max(), 2000)
pdf_pl = ((alpha-1)/xmin) * (x_pl/xmin)**(-alpha) * 0.5
pdf_log10_pl = pdf_pl * x_pl * np.log(10)
ax_main.plot(np.log10(x_pl), pdf_log10_pl,
             color=ACCENT3, lw=2, linestyle='--',
             label=f'Power-law tail  (α={alpha:.2f}, x_min=median)')

ax_main.axvline(np.log10(np.median(singles)), color='#ef4444', lw=1.5, linestyle=':',
                label=f'Median  ${np.median(singles):,.0f}')
ax_main.axvline(np.log10(np.mean(singles)), color='#22c55e', lw=1.5, linestyle=':',
                label=f'Mean  ${np.mean(singles):,.0f}')

# X-axis: show dollar amounts
tick_vals = [4, 4.5, 5, 5.5, 6, 6.5, 7]
ax_main.set_xticks(tick_vals)
ax_main.set_xticklabels(
    [f'${10**v:,.0f}' if v < 6 else f'${10**v/1e6:.1f}M' for v in tick_vals],
    rotation=30, ha='right'
)
ax_main.set_xlabel('Singles Earnings (log scale)', fontsize=11, labelpad=6)
ax_main.set_ylabel('Density', fontsize=11)
ax_main.set_title('2025 ATP Singles Earnings Distribution', fontsize=14, fontweight='bold', pad=12)
leg = ax_main.legend(fontsize=9, facecolor='#1a1a2e', edgecolor=GRID, labelcolor=TEXT)
style_ax(ax_main)

# Rank-tier bar chart
tier_labels = ['Top 5','6-10','11-20','21-50','51-100','101-200','201-500','501+']
tier_ranges = [(1,5),(6,10),(11,20),(21,50),(51,100),(101,200),(201,500),(501,2000)]
medians, means, counts = [], [], []
for lo, hi in tier_ranges:
    s = df[(df['rank']>=lo) & (df['rank']<=hi)]['singles']
    s = s[s > 0]
    medians.append(s.median() if len(s) else 0)
    means.append(s.mean() if len(s) else 0)
    counts.append(len(s))

x = np.arange(len(tier_labels))
w = 0.38
bars1 = ax_bar.bar(x - w/2, [m/1e3 for m in medians], w,
                   color=ACCENT2, alpha=0.85, label='Median', zorder=3)
bars2 = ax_bar.bar(x + w/2, [m/1e3 for m in means], w,
                   color=ACCENT3, alpha=0.85, label='Mean', zorder=3)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(tier_labels, fontsize=8.5)
ax_bar.set_xlabel('Rank Tier', fontsize=10)
ax_bar.set_ylabel('Earnings ($K)', fontsize=10)
ax_bar.set_title('Median vs Mean by Rank Tier (2025)', fontsize=11, fontweight='bold')
ax_bar.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'${v:.0f}K' if v < 1000 else f'${v/1000:.1f}M'))
ax_bar.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT)
style_ax(ax_bar)
ax_bar.grid(axis='x', alpha=0)

# Add value labels on bars
for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax_bar.text(bar.get_x()+bar.get_width()/2, h*1.02,
                    f'${h:.0f}K' if h < 1000 else f'${h/1000:.1f}M',
                    ha='center', va='bottom', fontsize=7, color=TEXT)

# Lorenz curve
sorted_s = np.sort(singles)
n = len(sorted_s)
cumulative_share = np.cumsum(sorted_s) / sorted_s.sum()
pop_share = np.arange(1, n+1) / n

gini = 1 - 2 * np.trapz(cumulative_share, pop_share)

ax_lorenz.fill_between(pop_share, cumulative_share, pop_share,
                        color=ACCENT, alpha=0.35, label=f'Inequality gap (Gini={gini:.3f})')
ax_lorenz.plot(pop_share, cumulative_share, color=ACCENT2, lw=2.5, label='Lorenz curve')
ax_lorenz.plot([0,1],[0,1], color='#94a3b8', lw=1.2, linestyle='--', label='Perfect equality')

# Mark key percentiles
for pct, label_str in [(0.5, '50%'), (0.8, '80%'), (0.99, '99%')]:
    idx = int(pct * n)
    ax_lorenz.annotate(
        f'Bottom {int(pct*100)}%\nearns {cumulative_share[idx-1]:.1%}',
        xy=(pop_share[idx-1], cumulative_share[idx-1]),
        xytext=(pop_share[idx-1]-0.18, cumulative_share[idx-1]+0.06),
        fontsize=7.5, color=TEXT,
        arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=0.8)
    )

ax_lorenz.set_xlabel('Cumulative share of players', fontsize=10)
ax_lorenz.set_ylabel('Cumulative share of earnings', fontsize=10)
ax_lorenz.set_title('Lorenz Curve (Earnings Inequality 2025)', fontsize=11, fontweight='bold')
ax_lorenz.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax_lorenz.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax_lorenz.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT)
style_ax(ax_lorenz)

# ── Title ─────────────────────────────────────────────────────────────────
fig.text(0.5, 0.98, 'ATP 2025 Singles Earnings Distribution',
         ha='center', va='top', fontsize=15, fontweight='bold', color=TEXT)

plt.savefig('earnings_dist_2025.png', dpi=180, bbox_inches='tight', facecolor=BG)
print("\nSaved earnings_dist_2025.png")


