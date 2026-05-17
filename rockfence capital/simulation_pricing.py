import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

DEFAULT_BUST_RATE = 0.05
DEFAULT_INFLATION_RATE = 0.078
TARGET_IRR = 0.15

def build_simulation_engine():
    print("="*80)
    print("PORTFOLIO SIMULATION & PRICING ENGINE")
    print("="*80)
    
    # 1. Load Data
    full = pd.read_csv('rank_to_singles_by_year.csv')
    with open('aging_curve_best_estimate.json', 'r') as f:
        aging = json.load(f)
        
    curve = {int(k): float(v) for k, v in aging['curve'].items()}
    ages = sorted(curve.keys())
    
    # 2. Calibrate Parameters
    # We use the pooled historical data for lognormal parameters
    earnings = full[full['singles'] > 0]['singles'].values
    log_earnings = np.log(earnings)
    mu_ln = np.mean(log_earnings)
    sigma_ln = np.std(log_earnings)
    
    # Bust probability is still a portfolio assumption rather than a direct
    # estimate from the prepared dataset, which drops unmatched players.
    bust_rate = DEFAULT_BUST_RATE

    # Prefer the inflation rate exported by the analysis step when available.
    inflation_rate = float(
        aging.get('calibration', {}).get('inflation_rate', DEFAULT_INFLATION_RATE)
    )
    
    print(f"Calibrated Parameters:")
    print(f"  Lognormal Mu: {mu_ln:.4f}")
    print(f"  Lognormal Sigma: {sigma_ln:.4f}")
    print(f"  Bust Rate Assumption: {bust_rate:.1%}")
    print(f"  Annual Inflation: {inflation_rate:.1%}\n")
    
    # 3. Simulation Parameters
    num_players_per_portfolio = 50
    num_portfolios = 10000
    target_irr = TARGET_IRR
    discount_rate = target_irr
    purchase_pct = 0.01 # Buying 1% of future earnings
    
    # 4. Run Simulation
    print(f"Simulating {num_portfolios} portfolios of {num_players_per_portfolio} players each...")
    
    # Assuming target is 18 years old
    start_age = 18
    max_age = max(ages)
    
    t_arr = np.array([a - start_age for a in range(start_age, max_age + 1)])
    discount_factors = (1 + discount_rate) ** (-t_arr)
    inflation_factors = (1 + inflation_rate) ** t_arr
    curve_vals = np.array([curve.get(a, 0.0) for a in range(start_age, max_age + 1)])
    
    expected_observed_earnings = np.exp(mu_ln + (sigma_ln**2) / 2.0)
    
    # Adjust to find peak
    mean_curve = np.mean([curve[a] for a in range(20, 32)])
    implied_peak_expected = expected_observed_earnings / mean_curve
    
    print(f"  Expected Peak Annual Earnings (Conditional): ${implied_peak_expected:,.0f}")
    
    # Simulate portfolios
    np.random.seed(42)
    portfolio_npvs = []
    
    for _ in range(num_portfolios):
        # Draw peaks
        peaks = np.random.lognormal(mean=mu_ln, sigma=sigma_ln, size=num_players_per_portfolio)
        
        # Adjust peaks based on implied peak multiplier
        peaks = peaks / mean_curve
        
        # Apply bust rate
        is_bust = np.random.rand(num_players_per_portfolio) < bust_rate
        peaks[is_bust] = 0.0
        
        # Calculate NPV for each player in portfolio
        # matrix mult: (players, 1) * (1, ages)
        cashflows = np.outer(peaks, curve_vals * inflation_factors)
        
        # 1% of cashflows
        cashflows = cashflows * purchase_pct
        
        # NPV for each player
        npvs = np.sum(cashflows * discount_factors, axis=1)
        
        # Total portfolio NPV
        portfolio_npvs.append(np.sum(npvs))
        
    portfolio_npvs = np.array(portfolio_npvs)
    
    fair_price_per_portfolio = np.mean(portfolio_npvs)
    fair_price_per_player = fair_price_per_portfolio / num_players_per_portfolio
    
    print("\n" + "="*80)
    print(f"PRICING OUTPUT (Target IRR = {target_irr:.1%})")
    print("="*80)
    print(f"Fair Price per 1% of an 18-year-old: ${fair_price_per_player:,.0f}")
    print(f"Fair Price for 1% of a {num_players_per_portfolio}-player portfolio: ${fair_price_per_portfolio:,.0f}")
    print(f"\nDistribution of Portfolio NPVs (at {discount_rate:.1%} discount):")
    print(f"  5th Percentile : ${np.percentile(portfolio_npvs, 5):,.0f}")
    print(f"  25th Percentile: ${np.percentile(portfolio_npvs, 25):,.0f}")
    print(f"  Median         : ${np.percentile(portfolio_npvs, 50):,.0f}")
    print(f"  75th Percentile: ${np.percentile(portfolio_npvs, 75):,.0f}")
    print(f"  95th Percentile: ${np.percentile(portfolio_npvs, 95):,.0f}")

if __name__ == '__main__':
    build_simulation_engine()
