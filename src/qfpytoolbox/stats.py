from __future__ import annotations

import numpy as np

from qfpytoolbox._internal._utils import validate_2d_returns, validate_returns


def mean_return(returns: object, annualize: bool = False, periods_per_year: float = 252) -> float:
    """Compute the arithmetic mean return."""
    r = validate_returns(returns)
    m = float(np.mean(r))
    if annualize:
        m = m * periods_per_year
    return m


def volatility(returns: object, annualize: bool = False, periods_per_year: float = 252) -> float:
    """Compute the standard deviation of returns (ddof=1)."""
    r = validate_returns(returns)
    vol = float(np.std(r, ddof=1))
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def skewness(returns: object) -> float:
    """Compute the sample skewness of returns."""
    r = validate_returns(returns)
    n = len(r)
    if n < 3:
        raise ValueError("skewness requires at least 3 observations")
    m = np.mean(r)
    s = np.std(r, ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((r - m) / s) ** 3) * n * (n - 1) / (n - 2) / 1.0)


def excess_kurtosis(returns: object) -> float:
    """Compute the sample excess kurtosis (Fisher definition, normal=0)."""
    r = validate_returns(returns)
    n = len(r)
    if n < 4:
        raise ValueError("excess_kurtosis requires at least 4 observations")
    m = np.mean(r)
    s = np.std(r, ddof=1)
    if s == 0:
        return 0.0
    # Sample excess kurtosis (bias-corrected)
    kurt = float(np.mean(((r - m) / s) ** 4))
    return kurt - 3.0


def sharpe_ratio(
    returns: object,
    risk_free_rate: float = 0.0,
    annualize: bool = False,
    periods_per_year: float = 252,
) -> float:
    """Compute the Sharpe ratio."""
    r = validate_returns(returns)
    excess = r - risk_free_rate / (periods_per_year if annualize else 1.0)
    m = float(np.mean(excess))
    s = float(np.std(excess, ddof=1))
    if s < 1e-14:
        return float("nan")
    ratio = m / s
    if annualize:
        ratio = ratio * np.sqrt(periods_per_year)
    return float(ratio)


def sortino_ratio(
    returns: object,
    risk_free_rate: float = 0.0,
    annualize: bool = False,
    periods_per_year: float = 252,
) -> float:
    """Compute the Sortino ratio using downside deviation."""
    r = validate_returns(returns)
    threshold = risk_free_rate / (periods_per_year if annualize else 1.0)
    excess = r - threshold
    downside = np.minimum(excess, 0.0)
    downside_std = float(np.sqrt(np.mean(downside**2)))
    if downside_std == 0:
        return float("nan")
    m = float(np.mean(excess))
    ratio = m / downside_std
    if annualize:
        ratio = ratio * np.sqrt(periods_per_year)
    return float(ratio)


def max_drawdown(returns: object) -> float:
    """Compute the maximum drawdown (returns a negative value, e.g. -0.3 for 30% drawdown)."""
    r = validate_returns(returns)
    cumulative = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(cumulative)
    dd = (cumulative - peak) / peak
    return float(np.min(dd))


def calmar_ratio(returns: object, periods_per_year: float = 252) -> float:
    """Compute the Calmar ratio (annualized return / abs(max drawdown))."""
    r = validate_returns(returns)
    ann_return = mean_return(r, annualize=True, periods_per_year=periods_per_year)
    mdd = max_drawdown(r)
    if mdd == 0:
        return float("nan")
    return float(ann_return / abs(mdd))


def value_at_risk(returns: object, confidence: float = 0.95, method: str = "historical") -> float:
    """Compute Value at Risk (negative value, e.g. -0.02 means 2% loss at given confidence)."""
    r = validate_returns(returns)
    if method == "historical":
        return float(np.percentile(r, (1 - confidence) * 100))
    raise ValueError(f"Unknown VaR method '{method}'. Supported: 'historical'")


def conditional_var(returns: object, confidence: float = 0.95, method: str = "historical") -> float:
    """Compute Conditional Value at Risk / Expected Shortfall (negative value)."""
    r = validate_returns(returns)
    if method == "historical":
        var = value_at_risk(r, confidence=confidence, method="historical")
        return float(np.mean(r[r <= var]))
    raise ValueError(f"Unknown CVaR method '{method}'. Supported: 'historical'")


def information_ratio(
    returns: object,
    benchmark_returns: object,
    annualize: bool = False,
    periods_per_year: float = 252,
) -> float:
    """Compute the Information Ratio (active return / tracking error)."""
    r = validate_returns(returns)
    b = validate_returns(benchmark_returns)
    if len(r) != len(b):
        raise ValueError("returns and benchmark_returns must have the same length")
    active = r - b
    m = float(np.mean(active))
    s = float(np.std(active, ddof=1))
    if s == 0:
        return float("nan")
    ratio = m / s
    if annualize:
        ratio = ratio * np.sqrt(periods_per_year)
    return float(ratio)


def tracking_error(
    returns: object,
    benchmark_returns: object,
    annualize: bool = False,
    periods_per_year: float = 252,
) -> float:
    """Compute the tracking error (std dev of active returns)."""
    r = validate_returns(returns)
    b = validate_returns(benchmark_returns)
    if len(r) != len(b):
        raise ValueError("returns and benchmark_returns must have the same length")
    active = r - b
    te = float(np.std(active, ddof=1))
    if annualize:
        te = te * np.sqrt(periods_per_year)
    return te


def beta(returns: object, market_returns: object) -> float:
    """Compute the CAPM beta of returns relative to market returns."""
    r = validate_returns(returns)
    m = validate_returns(market_returns)
    if len(r) != len(m):
        raise ValueError("returns and market_returns must have the same length")
    cov = np.cov(r, m, ddof=1)
    market_var = cov[1, 1]
    if market_var == 0:
        return float("nan")
    return float(cov[0, 1] / market_var)


def alpha(
    returns: object,
    market_returns: object,
    risk_free_rate: float = 0.0,
    annualize: bool = False,
    periods_per_year: float = 252,
) -> float:
    """Compute the CAPM alpha (Jensen's alpha)."""
    r = validate_returns(returns)
    m = validate_returns(market_returns)
    b = beta(r, m)
    rf_per_period = risk_free_rate / periods_per_year if annualize else risk_free_rate
    port_excess = float(np.mean(r)) - rf_per_period
    mkt_excess = float(np.mean(m)) - rf_per_period
    a = port_excess - b * mkt_excess
    if annualize:
        a = a * periods_per_year
    return float(a)


def correlation_matrix(returns_matrix: object) -> np.ndarray:
    """Compute the correlation matrix from a 2D returns matrix (n_obs, n_assets)."""
    arr = validate_2d_returns(returns_matrix)
    return np.corrcoef(arr.T)


def covariance_matrix(
    returns_matrix: object,
    annualize: bool = False,
    periods_per_year: float = 252,
) -> np.ndarray:
    """Compute the covariance matrix from a 2D returns matrix (n_obs, n_assets)."""
    arr = validate_2d_returns(returns_matrix)
    cov = np.cov(arr.T, ddof=1)
    if annualize:
        cov = cov * periods_per_year
    return cov
