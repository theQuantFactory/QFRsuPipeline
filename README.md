# QFPyToolbox

[![CI](https://github.com/QFPyToolbox/QFPyToolbox/actions/workflows/ci.yml/badge.svg)](https://github.com/QFPyToolbox/QFPyToolbox/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Pythonic quantitative finance toolbox — a Python equivalent of Julia-based quant toolboxes, built on NumPy with optional SciPy, matplotlib, and pandas support.

---

## Features

- **Risk & Return Statistics** — Sharpe, Sortino, Calmar, VaR, CVaR, alpha, beta, drawdown, and more
- **Time Series Analysis** — Rolling windows, EWM, cumulative returns, drawdown series, autocorrelation
- **Portfolio Optimization** — Min-variance, max-Sharpe, efficient frontier, risk parity (requires scipy)
- **Financial Calculations** — Bond pricing/duration/convexity, Black-Scholes options, NPV, IRR, time value of money
- **Pure NumPy core** — Only `numpy>=1.24` required; scipy/matplotlib/pandas are optional

---

## Installation

```bash
# Core (NumPy only)
pip install qfpytoolbox

# With portfolio optimization and options pricing
pip install "qfpytoolbox[scipy]"

# With plotting support
pip install "qfpytoolbox[plot]"

# With pandas integration
pip install "qfpytoolbox[pandas]"

# Everything
pip install "qfpytoolbox[all]"

# Development
pip install "qfpytoolbox[dev,all]"
```

---

## Quick Start

```python
import numpy as np
from qfpytoolbox import (
    mean_return, volatility, sharpe_ratio, max_drawdown,
    cumulative_returns, rolling_sharpe, value_at_risk,
)

# Generate sample daily returns
rng = np.random.default_rng(42)
returns = rng.normal(0.0005, 0.01, 252)

# Risk/return statistics
print(f"Mean return (ann.): {mean_return(returns, annualize=True):.2%}")
print(f"Volatility (ann.):  {volatility(returns, annualize=True):.2%}")
print(f"Sharpe ratio (ann.): {sharpe_ratio(returns, annualize=True):.2f}")
print(f"Max drawdown:        {max_drawdown(returns):.2%}")
print(f"95% VaR (daily):     {value_at_risk(returns, confidence=0.95):.2%}")

# Rolling Sharpe
roll_sharpe = rolling_sharpe(returns, window=20, annualize=True)

# Cumulative returns (length = len(returns) + 1)
cum_ret = cumulative_returns(returns)
print(f"Total return: {cum_ret[-1] - 1:.2%}")
```

### Portfolio Optimization (requires scipy)

```python
import numpy as np
from qfpytoolbox import (
    covariance_matrix, min_variance_portfolio,
    max_sharpe_portfolio, efficient_frontier,
)

# Sample multi-asset returns (n_obs, n_assets)
rng = np.random.default_rng(0)
returns_matrix = rng.normal(0.001, 0.01, (252, 4))
expected_returns = np.array([0.10, 0.12, 0.08, 0.15])
cov = covariance_matrix(returns_matrix, annualize=True)

# Minimum variance portfolio
mv = min_variance_portfolio(expected_returns, cov)
print(f"Min-var weights: {mv['weights'].round(3)}")
print(f"Min-var vol:     {mv['volatility']:.2%}")

# Maximum Sharpe portfolio
ms = max_sharpe_portfolio(expected_returns, cov, risk_free_rate=0.02)
print(f"Max-Sharpe:      {ms['sharpe']:.2f}")

# Efficient frontier
ef = efficient_frontier(expected_returns, cov, n_points=30)
```

### Bond Pricing

```python
from qfpytoolbox import bond_price, bond_duration, bond_convexity

# 6% semi-annual coupon bond, 5-year maturity, 5% YTM
price = bond_price(face_value=1000, coupon_rate=0.06, yield_to_maturity=0.05, periods=10, frequency=2)
dur   = bond_duration(1000, 0.06, 0.05, 10, frequency=2)
conv  = bond_convexity(1000, 0.06, 0.05, 10, frequency=2)
print(f"Price: {price:.2f}, Duration: {dur:.2f}yr, Convexity: {conv:.2f}")
```

### Black-Scholes Options (requires scipy)

```python
from qfpytoolbox import black_scholes_call, black_scholes_put, black_scholes_greeks

call = black_scholes_call(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
put  = black_scholes_put(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
greeks = black_scholes_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
print(f"Call: {call:.4f}, Put: {put:.4f}")
print(f"Delta(call)={greeks['delta_call']:.4f}, Gamma={greeks['gamma']:.4f}, Vega={greeks['vega']:.4f}")
```

---

## Module Overview

| Module | Description | scipy required? |
|---|---|---|
| `qfpytoolbox.stats` | Risk/return statistics: Sharpe, VaR, beta, alpha, drawdown, … | No |
| `qfpytoolbox.timeseries` | Rolling windows, EWM, cumulative returns, log/simple returns | No |
| `qfpytoolbox.optim` | Min-variance, max-Sharpe, efficient frontier, risk parity | Yes |
| `qfpytoolbox.finance` | Bonds, Black-Scholes, NPV, IRR, compounding | Yes (for B-S) |

---

## Python Version Support

| Python | Status |
|---|---|
| 3.9 | ✅ Supported |
| 3.10 | ✅ Supported |
| 3.11 | ✅ Supported |
| 3.12 | ✅ Supported |
| 3.13 | ✅ Supported |

---

## Documentation

- **[Julia → Python Mapping](docs/MAPPING.md)** — Concept and function mapping from Julia toolbox equivalents

---

## Development

```bash
git clone https://github.com/QFPyToolbox/QFPyToolbox.git
cd QFPyToolbox
pip install -e ".[dev,all]"

# Run tests
pytest tests/ -v --cov=qfpytoolbox

# Lint
ruff check src/ tests/
ruff format src/ tests/
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
