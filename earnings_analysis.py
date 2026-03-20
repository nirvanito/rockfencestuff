
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# [CHANGE THIS STUFF TO ADJUST DISTRUBUTION]


CSV_PATH = "rank_winnings_adjusted.csv"

#Distribution settings
DIST_LOWER_BOUND = 100000      # Lower bound
DIST_UPPER_BOUND = 1500000     # Upper bound
DIST_BIN_COUNT   = None       # Number of bins(set a specific num to override, None to autoscale)
DIST_BIN_WIDTH   = 100000     # Target bin width when autoscaling
DIST_FIELD       = "YTD"     #dont change this


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
   
    df.columns = df.columns.str.strip()
    return df


def print_summary(df: pd.DataFrame):
    valid = df.dropna(subset=[DIST_FIELD])
    field_vals = valid[DIST_FIELD]

    print(f"  ATP 2025 {DIST_FIELD.upper()} SUMMARY")
    print("=" * 55)
    print(f"  Players with data          : {len(valid)}")
    print(f"  Mean {DIST_FIELD}                 : ${field_vals.mean():,.0f}")
    print(f"  Median {DIST_FIELD}               : ${field_vals.median():,.0f}")
    print(f"  Std dev              : ${field_vals.std():,.0f}")
    print(f"  Min                        : ${field_vals.min():,.0f}")
    print(f"  Max                        : ${field_vals.max():,.0f}  ({valid.loc[field_vals.idxmax(), 'Name']})")

def plot_distribution(df: pd.DataFrame):
    """Histogram of earnings."""
    valid = df.dropna(subset=[DIST_FIELD])
    values = valid[DIST_FIELD]
    values = values[(values >= DIST_LOWER_BOUND) & (values <= DIST_UPPER_BOUND)]

    fig, ax = plt.subplots(figsize=(12, 6))

    if DIST_BIN_COUNT is not None:
        num_bins = DIST_BIN_COUNT
    else:
        num_bins = max(10, int((DIST_UPPER_BOUND - DIST_LOWER_BOUND) / DIST_BIN_WIDTH))

    counts, bin_edges, patches = ax.hist(
        values,
        bins=num_bins,
        range=(DIST_LOWER_BOUND, DIST_UPPER_BOUND),
        edgecolor="white",
        linewidth=0.6,
    )

    # Color 
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=len(patches))
    for i, patch in enumerate(patches):
        patch.set_facecolor(cmap(norm(i)))
        patch.set_alpha(0.85)

    # power law curve
    import warnings
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    valid_bins = (counts > 0) & (bin_centers > 0)
    if np.sum(valid_bins) > 1:
        log_x = np.log(bin_centers[valid_bins])
        log_y = np.log(counts[valid_bins])
        
        # Fit log(y) = m * log(x) + c -> y = e^c * x^m
        m, c = np.polyfit(log_x, log_y, 1)
        
        # points for curve
        x_min = max(bin_edges[0], 1)
        x_max = bin_edges[-1]
        x_curve = np.linspace(x_min, x_max, 200)
        k = np.exp(c)
        y_curve = k * (x_curve ** m)
        
        # Format k so its readable
        base, exp = f"{k:.2e}".split('e')
        exp = int(exp)  # remove leading zeros
        label_str = rf"Power Law Fit ($y = {base} \times 10^{{{exp}}} \cdot x^{{{m:.2f}}}$)"
        
        ax.plot(x_curve, y_curve, color="deeppink", linewidth=2.5, label=label_str)

    ax.set_xlabel(f"{DIST_FIELD} ($)", fontsize=11)
    ax.set_ylabel("Number of Players", fontsize=11)
    ax.set_title(
        f"2025 ATP {DIST_FIELD} Distribution\n"
        f"Range: ${DIST_LOWER_BOUND:,.0f} – ${DIST_UPPER_BOUND:,.0f}  |  Bins: {num_bins}",
        fontsize=13,
        fontweight="bold",
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x / 1e6:.1f}M" if x >= 1e6 else f"${x / 1e3:.0f}K"))

    # right y axis for probability
    ax2 = ax.twinx()
    ax_ylim = ax.get_ylim()
    ax2.set_ylim([y / len(values) for y in ax_ylim])
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.set_ylabel("Percentage of Players", fontsize=11)

    # median and mean lines [comment this stuff out if you think it clutters the graph]
    med = values.median()
    mn = values.mean()
    ax.axvline(med, color="orange", linestyle="--", linewidth=1.5, label=f"Median: ${med:,.0f}")
    ax.axvline(mn, color="red", linestyle="--", linewidth=1.5, label=f"Mean: ${mn:,.0f}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    df = load_data(CSV_PATH)

    print_summary(df)

    print(f">>> Showing {DIST_FIELD} Distribution...")
    plot_distribution(df)
