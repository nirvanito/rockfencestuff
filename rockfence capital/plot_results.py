import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from matplotlib.ticker import FuncFormatter
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('ggplot')
sns.set_palette("husl")

def build_reference_models(best):
    ages = np.arange(17, 40)
    ref_models = best.get('reference_models', {})

    if ref_models.get('piecewise_quadratic') and ref_models.get('fixed_effects_quadratic'):
        pw = ref_models['piecewise_quadratic']
        fe = ref_models['fixed_effects_quadratic']

        fair_curve = np.where(
            ages <= pw['peak_age'],
            np.exp(-pw['pre_peak_curvature'] * (pw['peak_age'] - ages) ** 2),
            np.exp(-pw['post_peak_curvature'] * (ages - pw['peak_age']) ** 2),
        )
        fe_curve_raw = fe['beta_age'] * ages + fe['beta_age_sq'] * ages**2
        fe_peak_val = (
            fe['beta_age'] * fe['peak_age'] + fe['beta_age_sq'] * fe['peak_age']**2
        )
        fe_curve = np.exp(fe_curve_raw - fe_peak_val)

        return {
            'ages': ages,
            'piecewise_curve': fair_curve,
            'piecewise_peak': float(pw['peak_age']),
            'fe_curve': fe_curve,
            'fe_peak': float(fe['peak_age']),
        }

    full = pd.read_csv('rank_to_singles_by_year.csv')
    full = full[full['singles'] > 0].copy()

    panel = full[['player name', 'age', 'singles', 'year']].copy()
    panel['ln_e'] = np.log(panel['singles'])
    panel = panel[np.isfinite(panel['ln_e'])]

    player_means = panel.groupby('player name')[['ln_e', 'age']].transform('mean')
    panel['ln_e_dm'] = panel['ln_e'] - player_means['ln_e']
    panel['age_dm'] = panel['age'] - player_means['age']
    panel['age_sq'] = panel['age']**2
    player_means_sq = panel.groupby('player name')['age_sq'].transform('mean')
    panel['age2_dm'] = panel['age_sq'] - player_means_sq

    X = panel[['age_dm', 'age2_dm']].values
    y = panel['ln_e_dm'].values
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    fe_peak = -beta[0] / (2 * beta[1])
    fe_curve_raw = beta[0] * ages + beta[1] * ages**2
    fe_peak_val = beta[0] * fe_peak + beta[1] * fe_peak**2
    fe_curve = np.exp(fe_curve_raw - fe_peak_val)

    def piecewise_loss(params, ages_arr, ln_e_dm_arr):
        peak, b1, b2 = params
        predicted = np.where(
            ages_arr <= peak,
            -b1 * (peak - ages_arr) ** 2,
            -b2 * (ages_arr - peak) ** 2,
        )
        predicted -= predicted.mean()
        return np.sum((ln_e_dm_arr - predicted) ** 2)

    result = minimize(
        piecewise_loss,
        x0=[27, 0.005, 0.01],
        args=(panel['age'].values, panel['ln_e_dm'].values),
        bounds=[(20, 35), (0.001, 0.05), (0.001, 0.05)],
        constraints={'type': 'ineq', 'fun': lambda p: p[2] - p[1]},
    )
    pw_peak, pw_b1, pw_b2 = result.x
    fair_curve = np.where(
        ages <= pw_peak,
        np.exp(-pw_b1 * (pw_peak - ages) ** 2),
        np.exp(-pw_b2 * (ages - pw_peak) ** 2),
    )

    return {
        'ages': ages,
        'piecewise_curve': fair_curve,
        'piecewise_peak': float(pw_peak),
        'fe_curve': fe_curve,
        'fe_peak': float(fe_peak),
    }

# 1. AGING CURVE PLOT
def plot_aging_curves():
    # Load best-estimate curve from JSON
    with open('aging_curve_best_estimate.json') as f:
        best = json.load(f)
    
    ages_best = np.array(sorted(best['curve'].keys(), key=int), dtype=int)
    vals_best = np.array([best['curve'][str(a)] for a in ages_best])
    ci_lo = np.array([best['ci_lower'][str(a)] for a in ages_best])
    ci_hi = np.array([best['ci_upper'][str(a)] for a in ages_best])
    peak_age = best['peak_age']

    refs = build_reference_models(best)
    ages = refs['ages']
    fair_curve = refs['piecewise_curve']
    pw_peak = refs['piecewise_peak']
    fe_curve = refs['fe_curve']
    fe_peak = refs['fe_peak']

    plt.figure(figsize=(12, 7))
    
    # Bootstrap CI band
    plt.fill_between(ages_best, ci_lo, ci_hi, alpha=0.15, color='#2563eb',
                     label='95% Bootstrap CI')
    
    # Best estimate (primary)
    plt.plot(ages_best, vals_best,
             label=f'Best Estimate (Corrected Delta, peak {peak_age})',
             linewidth=3, color='#2563eb', zorder=5)
    
    # Fair piecewise (reference)
    plt.plot(ages, fair_curve,
             label=f'Fair Piecewise Quadratic (peak {pw_peak:.1f})',
             linewidth=2, linestyle='--', color='#f97316', alpha=0.8)
    
    # FE regression (reference)
    plt.plot(ages, fe_curve,
             label=f'FE Regression Symmetric (peak {fe_peak:.0f})',
             linewidth=1.5, linestyle=':', color='#64748b', alpha=0.6)
    
    # Literature reference lines
    plt.axvline(x=25, color='#10b981', linestyle=':', alpha=0.4,
                label='Mlakar (2015) peak: 25')
    plt.axvspan(25, 27, alpha=0.06, color='#10b981',
                label='Sackmann Elo plateau: 25-27')
    
    # Peak marker
    plt.scatter([peak_age], [1.0], color='#2563eb', s=100, zorder=6,
                edgecolors='white', linewidth=2)
    
    plt.title('Tennis Earnings Aging Curves\n'
              '(Corrected for Survivorship Bias)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Earnings as Fraction of Peak', fontsize=12)
    plt.legend(loc='upper left', fontsize=9, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.02, 1.15)
    plt.xlim(17, 39)
    
    # Add annotation for key values
    plt.annotate(f'Peak: age {peak_age}',
                 xy=(peak_age, 1.0), xytext=(peak_age+2, 1.08),
                 fontsize=10, fontweight='bold', color='#2563eb',
                 arrowprops=dict(arrowstyle='->', color='#2563eb', lw=1.5))
    
    plt.tight_layout()
    plt.savefig('aging_curves.png', dpi=300)
    plt.close()
    print(f"  Peak age: {peak_age}")
    print(f"  Key values: age 25={vals_best[ages_best==25][0]:.3f}, "
          f"age 28={vals_best[ages_best==28][0]:.3f}, "
          f"age 30={vals_best[ages_best==30][0]:.3f}, "
          f"age 33={vals_best[ages_best==33][0]:.3f}")

# 2. DISTRIBUTION PLOT (2025 Data)
def plot_distribution():
    # Load data
    df = pd.read_csv('rank_to_singles_by_year.csv')
    df_2025 = df[(df['year'] == 2025) & (df['singles'] > 100)].copy()
    earnings = df_2025['singles'].values
    
    plt.figure(figsize=(12, 6))
    
    # Histogram of Log Earnings
    log_earnings = np.log10(earnings)
    sns.histplot(log_earnings, bins=30, stat='density', color='lightgray', label='Empirical Data (2025)')
    
    # Fit Lognormal
    mu_ln = np.mean(np.log(earnings))
    sigma_ln = np.std(np.log(earnings))
    x_ln = np.linspace(min(earnings), max(earnings), 1000)
    pdf_ln = stats.lognorm.pdf(x_ln, s=sigma_ln, scale=np.exp(mu_ln))
    # Convert PDF to log10 scale for plotting over log10 histogram
    # f_log10(x) = f(x) * x * ln(10)
    pdf_log10_ln = pdf_ln * x_ln * np.log(10)
    plt.plot(np.log10(x_ln), pdf_log10_ln, label=f'Lognormal Fit', color='green', linewidth=2)
    
    # Power Law tail (using median as xmin)
    xmin = np.median(earnings)
    tail_data = earnings[earnings >= xmin]
    alpha = 1 + len(tail_data) / np.sum(np.log(tail_data / xmin))
    
    x_pl = np.linspace(xmin, max(earnings), 1000)
    # Power law PDF: f(x) = ((alpha-1)/xmin) * (x/xmin)^(-alpha)
    pdf_pl = ((alpha-1)/xmin) * (x_pl/xmin)**(-alpha)
    # Scale by the probability of being in the tail (0.5 since xmin is median)
    pdf_pl = pdf_pl * 0.5
    pdf_log10_pl = pdf_pl * x_pl * np.log(10)
    plt.plot(np.log10(x_pl), pdf_log10_pl, label=f'Power Law Fit (Tail)', color='red', linewidth=2, linestyle='--')
    
    plt.axvline(x=np.log10(xmin), color='black', linestyle=':', label=f'Power Law $x_{{min}}$ (Median: ${xmin:,.0f})')
    
    plt.title('Earnings Distribution: Lognormal vs. Power Law Tail (2025)', fontsize=14)
    plt.xlabel('Log10(Singles Earnings $)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    
    # Format x-axis ticks to show actual dollar amounts
    locs, _ = plt.xticks()
    new_labels = [f"${10**loc:,.0f}" if loc >= 0 else "" for loc in locs]
    plt.xticks(locs, new_labels, rotation=45)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('distribution_fit.png', dpi=300)
    plt.close()

# 3. INFLATION PLOT
def plot_inflation():
    df = pd.read_csv('rank_to_singles_by_year.csv')
    
    rank_buckets = [(1,5), (6,10), (11,20), (21,50), (51,100), (101,200)]
    
    plt.figure(figsize=(12, 7))
    
    for lo, hi in rank_buckets:
        bucket = df[(df['rank']>=lo) & (df['rank']<=hi)]
        yearly_med = bucket.groupby('year')['singles'].median()
        
        # Plot
        plt.plot(yearly_med.index, yearly_med.values, marker='o', linewidth=2, label=f'Rank {lo}-{hi}')
    
    plt.yscale('log')
    plt.title('Median Singles Earnings by Rank Bucket (2015-2025)', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Median Earnings (Log Scale)', fontsize=12)
    
    # Format y-axis ticks as dollars
    def currency(x, pos):
        if x >= 1e6:
            return f'${x*1e-6:.1f}M'
        return f'${x:,.0f}'
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency))
    
    plt.legend(title="Rank Bucket", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, which="both", ls="--")
    plt.tight_layout()
    plt.savefig('inflation_trends.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    print("Generating aging_curves.png...")
    plot_aging_curves()
    print("Generating distribution_fit.png...")
    plot_distribution()
    print("Generating inflation_trends.png...")
    plot_inflation()
    print("Done!")
