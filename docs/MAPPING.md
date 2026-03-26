# Julia → Python Concept Mapping

This document maps concepts from a Julia quantitative finance toolbox to their Python equivalents in QFPyToolbox.

## Module Mapping

| Julia Module | QFPyToolbox Module | Description |
|---|---|---|
| `Stats` | `qfpytoolbox.stats` | Risk/return statistics |
| `TimeSeries` | `qfpytoolbox.timeseries` | Rolling & EWM calculations, return series utilities |
| `PortfolioOpt` | `qfpytoolbox.optim` | Portfolio optimization |
| `FixedIncome` / `Derivatives` | `qfpytoolbox.finance` | Bonds, options, discounting |

## Function Mapping

### Stats

| Julia | Python | Notes |
|---|---|---|
| `mean_return(r)` | `stats.mean_return(r)` | `annualize=True` to annualize |
| `volatility(r)` | `stats.volatility(r)` | Uses `ddof=1` |
| `skewness(r)` | `stats.skewness(r)` | Sample (bias-corrected) |
| `kurtosis(r) - 3` | `stats.excess_kurtosis(r)` | Excess (Fisher) convention |
| `sharpe(r, rf)` | `stats.sharpe_ratio(r, risk_free_rate=rf)` | |
| `sortino(r, rf)` | `stats.sortino_ratio(r, risk_free_rate=rf)` | Downside std |
| `max_drawdown(r)` | `stats.max_drawdown(r)` | Returns negative value |
| `calmar(r)` | `stats.calmar_ratio(r)` | |
| `var(r, 0.95)` | `stats.value_at_risk(r, confidence=0.95)` | Returns negative |
| `cvar(r, 0.95)` | `stats.conditional_var(r, confidence=0.95)` | Expected Shortfall |
| `information_ratio(r, b)` | `stats.information_ratio(r, b)` | |
| `tracking_error(r, b)` | `stats.tracking_error(r, b)` | |
| `beta(r, m)` | `stats.beta(r, m)` | CAPM beta |
| `alpha(r, m, rf)` | `stats.alpha(r, m, risk_free_rate=rf)` | Jensen's alpha |
| `cor(R)` | `stats.correlation_matrix(R)` | `R` is (n_obs, n_assets) |
| `cov(R)` | `stats.covariance_matrix(R)` | |

### Time Series

| Julia | Python | Notes |
|---|---|---|
| `rolling_mean(r, w)` | `timeseries.rolling_mean(r, window=w)` | NaN-padded prefix |
| `rolling_std(r, w)` | `timeseries.rolling_std(r, window=w)` | |
| `rolling_sharpe(r, w)` | `timeseries.rolling_sharpe(r, window=w)` | |
| `ewm(r; span=s)` | `timeseries.ewm_mean(r, span=s)` | Also supports halflife, alpha |
| `cum_returns(r)` | `timeseries.cumulative_returns(r)` | Length = len(r)+1, starts at 1.0 |
| `log_returns(p)` | `timeseries.log_returns(p)` | |
| `simple_returns(p)` | `timeseries.simple_returns(p)` | |
| `drawdown(r)` | `timeseries.drawdown_series(r)` | Array of per-period drawdowns |
| `autocor(r; lag=1)` | `timeseries.autocorrelation(r, lag=1)` | |

### Portfolio Optimization

| Julia | Python | Notes |
|---|---|---|
| `min_var_portfolio(μ, Σ)` | `optim.min_variance_portfolio(μ, Σ)` | Requires scipy |
| `max_sharpe_portfolio(μ, Σ, rf)` | `optim.max_sharpe_portfolio(μ, Σ, rf)` | Requires scipy |
| `efficient_frontier(μ, Σ)` | `optim.efficient_frontier(μ, Σ)` | Returns dict with arrays |
| `risk_parity(Σ)` | `optim.risk_parity_portfolio(Σ)` | Equal risk contribution |
| `equal_weight(n)` | `optim.equal_weight_portfolio(n)` | No scipy needed |

### Finance

| Julia | Python | Notes |
|---|---|---|
| `compound_interest(P, r, t)` | `finance.compound_interest(P, r, t)` | |
| `pv(FV, r, t)` | `finance.present_value(FV, r, t)` | |
| `fv(PV, r, t)` | `finance.future_value(PV, r, t)` | |
| `npv(cf, r)` | `finance.npv(cf, r)` | `cf[0]` is initial outflow |
| `irr(cf)` | `finance.irr(cf)` | Polynomial root finding |
| `bond_price(F, c, y, T)` | `finance.bond_price(F, c, y, T)` | `frequency=2` default |
| `duration(F, c, y, T)` | `finance.bond_duration(F, c, y, T)` | Macaulay, in years |
| `convexity(F, c, y, T)` | `finance.bond_convexity(F, c, y, T)` | |
| `bs_call(S, K, T, r, σ)` | `finance.black_scholes_call(S, K, T, r, σ)` | Requires scipy |
| `bs_put(S, K, T, r, σ)` | `finance.black_scholes_put(S, K, T, r, σ)` | |
| `bs_greeks(S, K, T, r, σ)` | `finance.black_scholes_greeks(S, K, T, r, σ)` | Returns dict |

---

## Code Translation Examples

### Computing Sharpe Ratio

**Julia:**
```julia
using MyJuliaToolbox
r = rand(252) .* 0.02 .- 0.005
sharpe = sharpe_ratio(r; risk_free=0.0, annualize=true)
```

**Python:**
```python
import numpy as np
from qfpytoolbox import sharpe_ratio

rng = np.random.default_rng(42)
r = rng.normal(0.001, 0.01, 252)
sharpe = sharpe_ratio(r, risk_free_rate=0.0, annualize=True, periods_per_year=252)
```

### Portfolio Optimization

**Julia:**
```julia
using MyJuliaToolbox
μ = [0.10, 0.12, 0.08, 0.15]
Σ = cov_matrix(returns_matrix)
result = min_var_portfolio(μ, Σ)
println(result.weights)
```

**Python:**
```python
import numpy as np
from qfpytoolbox import covariance_matrix, min_variance_portfolio

μ = np.array([0.10, 0.12, 0.08, 0.15])
Σ = covariance_matrix(returns_matrix, annualize=True)
result = min_variance_portfolio(μ, Σ)
print(result["weights"])
```

### Rolling Analysis

**Julia:**
```julia
using MyJuliaToolbox
prices = [100.0, 102.0, 101.5, 104.0, 103.0]
r = simple_returns(prices)
roll_sharpe = rolling_sharpe(r; window=20, annualize=true)
```

**Python:**
```python
import numpy as np
from qfpytoolbox import simple_returns, rolling_sharpe

prices = np.array([100.0, 102.0, 101.5, 104.0, 103.0])
r = simple_returns(prices)
roll_sharpe = rolling_sharpe(r, window=20, annualize=True, periods_per_year=252)
```

### Black-Scholes Pricing

**Julia:**
```julia
using MyJuliaToolbox
call_price = bs_call(100.0, 100.0, 1.0, 0.05, 0.2)
greeks = bs_greeks(100.0, 100.0, 1.0, 0.05, 0.2)
```

**Python:**
```python
from qfpytoolbox import black_scholes_call, black_scholes_greeks

call_price = black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.2)
greeks = black_scholes_greeks(100.0, 100.0, 1.0, 0.05, 0.2)
print(f"Delta (call): {greeks['delta_call']:.4f}")
```

---

## Design Differences

### Return Conventions
- Both toolboxes represent returns as simple arithmetic returns (not log returns) by default.
- `log_returns()` / `simple_returns()` are provided for price-to-return conversion.
- `max_drawdown()` and `value_at_risk()` return **negative** values in QFPyToolbox to preserve sign conventions.

### Array Shape
- Julia often uses column vectors; Python/NumPy prefers 1D arrays for return series.
- The 2D `returns_matrix` is shaped `(n_obs, n_assets)` in QFPyToolbox (rows = time, columns = assets), consistent with NumPy convention.

### Optional Dependencies
- Julia packages declare dependencies at the module level; in QFPyToolbox, `scipy` is optional and only required for `optim` optimization functions and `black_scholes_*` functions. A helpful `ImportError` is raised if scipy is missing.
- Install scipy support: `pip install qfpytoolbox[scipy]`

### Cumulative Returns
- `cumulative_returns(r)` returns an array of **length n+1** (starting at 1.0), unlike Julia which may return length n. The extra element represents the starting value before any returns.

### EWM Parameters
- Julia's `ewma` typically uses `λ` (decay factor). QFPyToolbox uses `span`, `halflife`, or `alpha` (smoothing factor), consistent with pandas conventions.
