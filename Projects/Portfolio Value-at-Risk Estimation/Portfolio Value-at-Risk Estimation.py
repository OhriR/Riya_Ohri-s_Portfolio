import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import yfinance as yf

# Fetch historical data for 3 assets using yfinance
symbols = ['AAPL', 'MSFT', 'GOOGL']  # Example symbols for Apple, Microsoft, and Google
start_date = '2000-01-01'
end_date = '2024-01-01'

data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
daily_returns = data.pct_change().dropna()

# Portfolio weights (modify these based on your portfolio composition)
weights = np.array([0.3, 0.3, 0.4])  # Example weights for a 3-asset portfolio

# Calculate portfolio returns
daily_portfolio_returns = daily_returns.dot(weights)

# Set VaR parameters
confidence_level = 0.95  # Confidence level for VaR
time_horizon = 1  # Time horizon for VaR estimation (1 day)

# 1. Historical Simulation VaR
var_historical = -np.percentile(daily_portfolio_returns, (1 - confidence_level) * 100)

# 2. Parametric (Variance-Covariance) VaR
mean_return = np.mean(daily_portfolio_returns)
std_deviation = np.std(daily_portfolio_returns)
var_parametric = - (mean_return + stats.norm.ppf(1 - confidence_level) * std_deviation)

# 3. Monte Carlo Simulation VaR
num_simulations = 10000
simulated_returns = np.random.normal(mean_return, std_deviation, num_simulations)
var_monte_carlo = -np.percentile(simulated_returns, (1 - confidence_level) * 100)

# Display results
print(f"Historical Simulation VaR: {var_historical:.2f}")
print(f"Parametric (Variance-Covariance) VaR: {var_parametric:.2f}")
print(f"Monte Carlo Simulation VaR: {var_monte_carlo:.2f}")

# Backtesting the VaR Models
def backtest_var(var_values, portfolio_returns):
    """
    Backtest the VaR model by comparing actual returns to the VaR estimates.
    """
    violations = portfolio_returns < -var_values
    num_violations = np.sum(violations)
    total_days = len(portfolio_returns)
    var_failure_rate = num_violations / total_days
    print(f"Number of violations: {num_violations}")
    print(f"Failure rate: {var_failure_rate:.2%}")

    return violations

# Backtest each model
print("\nBacktesting Historical VaR:")
historical_violations = backtest_var(var_historical, daily_portfolio_returns)

print("\nBacktesting Parametric VaR:")
parametric_violations = backtest_var(var_parametric, daily_portfolio_returns)

print("\nBacktesting Monte Carlo VaR:")
monte_carlo_violations = backtest_var(var_monte_carlo, daily_portfolio_returns)

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(daily_returns.index, daily_portfolio_returns, label='Daily Portfolio Returns', color='blue')
plt.axhline(y=-var_historical, color='red', linestyle='--', label='Historical VaR')
plt.axhline(y=-var_parametric, color='green', linestyle='--', label='Parametric VaR')
plt.axhline(y=-var_monte_carlo, color='orange', linestyle='--', label='Monte Carlo VaR')
plt.xlabel('Date')
plt.ylabel('Return')
plt.title('Value-at-Risk (VaR) Estimates')
plt.legend()
plt.show()