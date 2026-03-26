from __future__ import annotations

import numpy as np

from qfpytoolbox._internal._utils import _ewm_alpha, validate_returns


def rolling_mean(data: object, window: int) -> np.ndarray:
    """Compute the rolling mean with NaN-padding for the initial window."""
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.mean(arr[i - window + 1 : i + 1])
    return result


def rolling_std(data: object, window: int, ddof: int = 1) -> np.ndarray:
    """Compute the rolling standard deviation with NaN-padding."""
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    result = np.full(n, np.nan)
    if window <= ddof:
        return result
    for i in range(window - 1, n):
        result[i] = np.std(arr[i - window + 1 : i + 1], ddof=ddof)
    return result


def rolling_sharpe(
    returns: object,
    window: int,
    risk_free_rate: float = 0.0,
    annualize: bool = False,
    periods_per_year: float = 252,
) -> np.ndarray:
    """Compute the rolling Sharpe ratio."""
    arr = validate_returns(returns)
    n = len(arr)
    result = np.full(n, np.nan)
    rf = risk_free_rate / (periods_per_year if annualize else 1.0)
    for i in range(window - 1, n):
        window_r = arr[i - window + 1 : i + 1]
        excess = window_r - rf
        m = np.mean(excess)
        s = np.std(excess, ddof=1)
        if s == 0:
            result[i] = np.nan
        else:
            ratio = m / s
            result[i] = ratio * np.sqrt(periods_per_year) if annualize else ratio
    return result


def rolling_max_drawdown(returns: object, window: int) -> np.ndarray:
    """Compute the rolling maximum drawdown."""
    arr = validate_returns(returns)
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        window_r = arr[i - window + 1 : i + 1]
        cumulative = np.cumprod(1.0 + window_r)
        peak = np.maximum.accumulate(cumulative)
        dd = (cumulative - peak) / peak
        result[i] = float(np.min(dd))
    return result


def rolling_correlation(x: object, y: object, window: int) -> np.ndarray:
    """Compute the rolling correlation between two series."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) != len(ya):
        raise ValueError("x and y must have the same length")
    n = len(xa)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        xw = xa[i - window + 1 : i + 1]
        yw = ya[i - window + 1 : i + 1]
        if np.std(xw) == 0 or np.std(yw) == 0:
            result[i] = np.nan
        else:
            result[i] = float(np.corrcoef(xw, yw)[0, 1])
    return result


def rolling_beta(returns: object, market_returns: object, window: int) -> np.ndarray:
    """Compute the rolling CAPM beta."""
    ra = validate_returns(returns)
    ma = validate_returns(market_returns)
    if len(ra) != len(ma):
        raise ValueError("returns and market_returns must have the same length")
    n = len(ra)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        rw = ra[i - window + 1 : i + 1]
        mw = ma[i - window + 1 : i + 1]
        cov = np.cov(rw, mw, ddof=1)
        market_var = cov[1, 1]
        if market_var == 0:
            result[i] = np.nan
        else:
            result[i] = cov[0, 1] / market_var
    return result


def ewm_mean(
    data: object,
    span: float | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
) -> np.ndarray:
    """Compute the exponentially weighted moving average."""
    arr = np.asarray(data, dtype=float)
    a = _ewm_alpha(span=span, halflife=halflife, alpha=alpha)
    n = len(arr)
    result = np.empty(n)
    result[0] = arr[0]
    for i in range(1, n):
        result[i] = a * arr[i] + (1 - a) * result[i - 1]
    return result


def ewm_std(
    data: object,
    span: float | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
    ddof: int = 0,
) -> np.ndarray:
    """Compute the exponentially weighted moving standard deviation."""
    arr = np.asarray(data, dtype=float)
    a = _ewm_alpha(span=span, halflife=halflife, alpha=alpha)
    n = len(arr)
    result = np.full(n, np.nan)
    mean = arr[0]
    var = 0.0
    result[0] = 0.0
    for i in range(1, n):
        prev_mean = mean
        mean = a * arr[i] + (1 - a) * mean
        var = (1 - a) * (var + a * (arr[i] - prev_mean) ** 2)
        if ddof == 0:
            result[i] = np.sqrt(var)
        else:
            # Bias correction analogous to pandas' EWM ddof=1:
            # effective_weight_sum² / (effective_weight_sum² - sum_of_squared_weights)
            # where effective_weight_sum = 1-(1-α)^(t+1) and
            # sum_of_squared_weights = α²*(1-(1-α)^(2t)) using the geometric series.
            weight_sum = 1 - (1 - a) ** (i + 1)
            correction = weight_sum**2 / (weight_sum**2 - a**2 * (1 - (1 - a) ** (2 * i)))
            result[i] = np.sqrt(var * correction) if correction > 0 else np.sqrt(var)
    return result


def cumulative_returns(returns: object) -> np.ndarray:
    """Compute cumulative returns, starting at 1.0. Length is len(returns)+1."""
    r = validate_returns(returns)
    cum = np.empty(len(r) + 1)
    cum[0] = 1.0
    cum[1:] = np.cumprod(1.0 + r)
    return cum


def log_returns(prices: object) -> np.ndarray:
    """Compute log returns from a price series."""
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or len(p) < 2:
        raise ValueError("prices must be a 1D array with at least 2 elements")
    return np.log(p[1:] / p[:-1])


def simple_returns(prices: object) -> np.ndarray:
    """Compute simple (arithmetic) returns from a price series."""
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or len(p) < 2:
        raise ValueError("prices must be a 1D array with at least 2 elements")
    return (p[1:] - p[:-1]) / p[:-1]


def annualize_returns(cumulative_return: float, periods: float, periods_per_year: float = 252) -> float:
    """Annualize a cumulative return over a given number of periods."""
    if periods <= 0:
        raise ValueError("periods must be > 0")
    years = periods / periods_per_year
    return float((1.0 + cumulative_return) ** (1.0 / years) - 1.0)


def drawdown_series(returns: object) -> np.ndarray:
    """Compute the drawdown at each point in time (negative or zero)."""
    r = validate_returns(returns)
    cumulative = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(cumulative)
    return (cumulative - peak) / peak


def autocorrelation(data: object, lag: int = 1) -> float:
    """Compute the autocorrelation at the specified lag."""
    arr = np.asarray(data, dtype=float)
    if lag < 1:
        raise ValueError("lag must be >= 1")
    if len(arr) <= lag:
        raise ValueError("data length must be greater than lag")
    x = arr[:-lag]
    y = arr[lag:]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
