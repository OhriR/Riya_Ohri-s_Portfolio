import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from scipy.stats import norm
import matplotlib.pyplot as plt

# Define the ticker symbol for Ford
ticker_symbol = 'F'

# Create a Ticker object for Ford
ford = yf.Ticker(ticker_symbol)

# Get historical stock prices for the past year
start_date = datetime.datetime.now() - datetime.timedelta(days=365)
historical_prices = ford.history(start=start_date)

# Display the first few rows of the historical data
print(historical_prices.head())

# Outstanding shares
shares_outstanding = ford.info['sharesOutstanding']
print(f"Shares Outstanding: {shares_outstanding}")

# Get the latest closing price
latest_price = historical_prices['Close'].iloc[-1]

# Calculate Market Value of Equity (E)
market_value_of_equity = 48438.5863 * 1e6  # Market value in millions converted to actual value
print(f"Market Value of Equity (E): ${market_value_of_equity:.2f}")

# Calculate daily returns
historical_prices['Daily Return'] = historical_prices['Close'].pct_change()

# Calculate standard deviation of daily returns (volatility of equity)
equity_volatility = historical_prices['Daily Return'].std() * np.sqrt(252)
print(f"Equity Volatility (σ_E): {equity_volatility:.2%}")

# Assume a 5% risk-free rate
risk_free_rate = 0.0482  # 1 year treasury from yield curve
print(f"Risk-Free Rate (r): {risk_free_rate:.2%}")

# Define debt value (K) - Total liabilities (use long-term debt + current liabilities)
debt_value = (50150 + 100957) * 1e6  # Debt value in millions converted to actual value
print(f"Debt Value (K): ${debt_value:.2f}")

# Time to maturity (T) - Assume 1 year
T = 1

# Initial guesses for Asset Value (A) and Asset Volatility (σ_A)
A = 273310 * 1e6  # Initial asset value estimate in millions converted to actual value
sigma_A = equity_volatility  # Initial asset volatility estimate

# Convergence settings
tolerance = 1e-6
max_iterations = 1000
iteration = 0

# Iterative process to estimate Asset Value (A) and Asset Volatility (σ_A)
while iteration < max_iterations:
    d1 = (np.log(A / debt_value) + (risk_free_rate + 0.5 * sigma_A ** 2) * T) / (sigma_A * np.sqrt(T))
    d2 = d1 - sigma_A * np.sqrt(T)

    # Calculate estimated equity value using Black-Scholes formula
    E_estimate = A * norm.cdf(d1) - debt_value * np.exp(-risk_free_rate * T) * norm.cdf(d2)

    # Calculate estimated equity volatility
    sigma_E_estimate = sigma_A * A / E_estimate * norm.cdf(d1)

    # Update asset value and volatility using the equity estimate
    A_new = market_value_of_equity + debt_value * norm.cdf(d2) * np.exp(-risk_free_rate * T)
    sigma_A_new = equity_volatility / (A / E_estimate * norm.cdf(d1))

    # Check for convergence
    if abs(A_new - A) < tolerance and abs(sigma_A_new - sigma_A) < tolerance:
        break

    # Update estimates for next iteration
    A = A_new
    sigma_A = sigma_A_new
    iteration += 1

# Print the estimated Asset Value (A) and Asset Volatility (σ_A)
print(f"\nEstimated Asset Value (A): ${A:.2f}")
print(f"Estimated Asset Volatility (σ_A): {sigma_A:.2%}")

# Sensitivity analysis for different debt values
debt_values = np.linspace(debt_value * 0.8, debt_value * 1.2, 5)
pd_values = []
for K in debt_values:
    d2 = (np.log(A / K) + (risk_free_rate - 0.5 * sigma_A ** 2) * T) / (sigma_A * np.sqrt(T))
    pd = 1 - norm.cdf(d2)
    pd_values.append(pd)
    print(f"Debt Value: ${K:.2f}, Probability of Default (PD): {pd:.2%}")

plt.figure(figsize=(10, 6))
plt.plot(debt_values, pd_values, marker='o')
plt.title('Sensitivity Analysis: PD vs. Debt Value')
plt.xlabel('Debt Value ($)')
plt.ylabel('Probability of Default (PD)')
plt.grid(True)
plt.show()

# Sensitivity analysis for different asset volatilities
volatility_values = np.linspace(sigma_A * 0.8, sigma_A * 1.2, 5)
pd_values_volatility = []
for sigma in volatility_values:
    d2 = (np.log(A / debt_value) + (risk_free_rate - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    pd = 1 - norm.cdf(d2)
    pd_values_volatility.append(pd)
    print(f"Asset Volatility: {sigma:.2%}, Probability of Default (PD): {pd:.2%}")

plt.figure(figsize=(10, 6))
plt.plot(volatility_values, pd_values_volatility, marker='o')
plt.title('Sensitivity Analysis: PD vs. Asset Volatility')
plt.xlabel('Asset Volatility (σ_A)')
plt.ylabel('Probability of Default (PD)')
plt.grid(True)
plt.show()

# Sensitivity analysis for different risk-free rates
risk_free_rates = np.linspace(risk_free_rate * 0.8, risk_free_rate * 1.2, 5)
pd_values_risk_free_rate = []
for r in risk_free_rates:
    d2 = (np.log(A / debt_value) + (r - 0.5 * sigma_A ** 2) * T) / (sigma_A * np.sqrt(T))
    pd = 1 - norm.cdf(d2)
    pd_values_risk_free_rate.append(pd)
    print(f"Risk-Free Rate: {r:.2%}, Probability of Default (PD): {pd:.2%}")

plt.figure(figsize=(10, 6))
plt.plot(risk_free_rates, pd_values_risk_free_rate, marker='o')
plt.title('Sensitivity Analysis: PD vs. Risk-Free Rate')
plt.xlabel('Risk-Free Rate (r)')
plt.ylabel('Probability of Default (PD)')
plt.grid(True)
plt.show()

# Example default rate table (based on Moody's data)
# Example default rate table (based on Moody's data)
default_rates = {
    'AAA': 0.01,
    'AA': 0.05,
    'A': 0.10,
    'BBB': 0.30,
    'BB': 1.50,
    'B': 6.00,
    'CCC': 15.00
}

# My calculated PD results
pd_results = [
    {'Debt Value': 120885600000.00, 'PD': 0.00},
    {'Debt Value': 135996300000.00, 'PD': 0.00},
    {'Debt Value': 151107000000.00, 'PD': 0.16},
    {'Debt Value': 166217700000.00, 'PD': 2.51},
    {'Debt Value': 181328400000.00, 'PD': 14.56},
    {'Asset Volatility': 7.72, 'PD': 0.01},
    {'Asset Volatility': 8.68, 'PD': 0.05},
    {'Asset Volatility': 9.64, 'PD': 0.16},
    {'Asset Volatility': 10.61, 'PD': 0.38},
    {'Asset Volatility': 11.57, 'PD': 0.74},
    {'Risk-Free Rate': 3.86, 'PD': 0.22},
    {'Risk-Free Rate': 4.34, 'PD': 0.19},
    {'Risk-Free Rate': 4.82, 'PD': 0.16},
    {'Risk-Free Rate': 5.30, 'PD': 0.14},
    {'Risk-Free Rate': 5.78, 'PD': 0.12}
]

# Compare each calculated PD with default rate table to determine closest rating
for result in pd_results:
    pd_value = result['PD']
    closest_rating = min(default_rates, key=lambda x: abs(default_rates[x] - pd_value))
    print(f"The closest credit rating equivalent to the calculated PD ({pd_value:.2f}%) is: {closest_rating}")