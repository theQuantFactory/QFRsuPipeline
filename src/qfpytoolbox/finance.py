from __future__ import annotations

import numpy as np


def compound_interest(
    principal: float,
    rate: float,
    periods: float,
    compounding_frequency: float = 1,
) -> float:
    """Compute the future value with compound interest.

    Args:
        principal: Initial investment amount.
        rate: Annual interest rate (e.g. 0.05 for 5%).
        periods: Number of years.
        compounding_frequency: Times compounded per year (default 1 = annual).

    Returns:
        Future value after compounding.
    """
    return float(principal * (1 + rate / compounding_frequency) ** (compounding_frequency * periods))


def present_value(future_value: float, rate: float, periods: float) -> float:
    """Compute the present value of a future cash flow.

    Args:
        future_value: Amount received in the future.
        rate: Discount rate per period.
        periods: Number of periods.

    Returns:
        Present value.
    """
    return float(future_value / (1 + rate) ** periods)


def future_value(present_value_amount: float, rate: float, periods: float) -> float:
    """Compute the future value of a present cash flow.

    Args:
        present_value_amount: Current amount.
        rate: Growth rate per period.
        periods: Number of periods.

    Returns:
        Future value.
    """
    return float(present_value_amount * (1 + rate) ** periods)


def npv(cash_flows: object, discount_rate: float) -> float:
    """Compute Net Present Value.

    Args:
        cash_flows: Array of cash flows; cash_flows[0] is typically the initial investment (negative).
        discount_rate: Discount rate per period.

    Returns:
        Net present value.
    """
    cf = np.asarray(cash_flows, dtype=float)
    periods = np.arange(len(cf))
    discount_factors = (1.0 + discount_rate) ** periods
    return float(np.sum(cf / discount_factors))


def irr(cash_flows: object, guess: float = 0.1) -> float:
    """Compute the Internal Rate of Return using numpy polynomial root finding.

    Args:
        cash_flows: Array of cash flows; cash_flows[0] is typically negative (investment).
        guess: Used to select the most economically meaningful root when the polynomial
            has multiple real roots > -1 (picks the root closest to this value).

    Returns:
        Internal rate of return as a decimal.
    """
    cf = np.asarray(cash_flows, dtype=float)
    if len(cf) < 2:
        raise ValueError("cash_flows must have at least 2 elements")
    # NPV = sum(cf[t] / (1+r)^t) = 0
    # Multiply through by (1+r)^(n-1) to get a polynomial in (1+r)
    # cf[0]*(1+r)^(n-1) + cf[1]*(1+r)^(n-2) + ... + cf[n-1] = 0
    # numpy.roots expects coefficients from highest to lowest power: cf[0], cf[1], ...
    roots = np.roots(cf)
    # Filter for real positive roots that give r > -1
    real_roots = roots[np.isreal(roots)].real
    valid = real_roots[real_roots > -1]
    if len(valid) == 0:
        return float("nan")
    # Pick root closest to guess
    r_values = valid - 1.0
    best = r_values[np.argmin(np.abs(r_values - guess))]
    return float(best)


def bond_price(
    face_value: float,
    coupon_rate: float,
    yield_to_maturity: float,
    periods: int,
    frequency: int = 2,
) -> float:
    """Compute the price of a coupon bond.

    Args:
        face_value: Par/face value of the bond.
        coupon_rate: Annual coupon rate (e.g. 0.05 for 5%).
        yield_to_maturity: Annual yield to maturity.
        periods: Total number of coupon periods.
        frequency: Coupon payments per year (default 2 = semi-annual).

    Returns:
        Bond price.
    """
    coupon = face_value * coupon_rate / frequency
    ytm_period = yield_to_maturity / frequency
    t = np.arange(1, periods + 1)
    coupon_pv = np.sum(coupon / (1 + ytm_period) ** t)
    face_pv = face_value / (1 + ytm_period) ** periods
    return float(coupon_pv + face_pv)


def bond_duration(
    face_value: float,
    coupon_rate: float,
    yield_to_maturity: float,
    periods: int,
    frequency: int = 2,
) -> float:
    """Compute the Macaulay duration of a coupon bond (in years).

    Args:
        face_value: Par/face value of the bond.
        coupon_rate: Annual coupon rate.
        yield_to_maturity: Annual yield to maturity.
        periods: Total number of coupon periods.
        frequency: Coupon payments per year (default 2 = semi-annual).

    Returns:
        Macaulay duration in years.
    """
    coupon = face_value * coupon_rate / frequency
    ytm_period = yield_to_maturity / frequency
    t = np.arange(1, periods + 1)
    cash_flows = np.full(periods, coupon, dtype=float)
    cash_flows[-1] += face_value
    discount_factors = (1 + ytm_period) ** t
    pv_cf = cash_flows / discount_factors
    price = np.sum(pv_cf)
    # Duration in periods; convert to years
    duration_periods = float(np.sum(t * pv_cf) / price)
    return duration_periods / frequency


def bond_convexity(
    face_value: float,
    coupon_rate: float,
    yield_to_maturity: float,
    periods: int,
    frequency: int = 2,
) -> float:
    """Compute the convexity of a coupon bond.

    Args:
        face_value: Par/face value of the bond.
        coupon_rate: Annual coupon rate.
        yield_to_maturity: Annual yield to maturity.
        periods: Total number of coupon periods.
        frequency: Coupon payments per year (default 2 = semi-annual).

    Returns:
        Convexity (annualized).
    """
    coupon = face_value * coupon_rate / frequency
    ytm_period = yield_to_maturity / frequency
    t = np.arange(1, periods + 1)
    cash_flows = np.full(periods, coupon, dtype=float)
    cash_flows[-1] += face_value
    discount_factors = (1 + ytm_period) ** t
    pv_cf = cash_flows / discount_factors
    price = np.sum(pv_cf)
    # Convexity in periods^2; annualize by dividing by frequency^2
    convexity_periods = float(np.sum(t * (t + 1) * pv_cf) / (price * (1 + ytm_period) ** 2))
    return convexity_periods / frequency**2


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF using scipy if available, else math.erf."""
    try:
        from scipy.stats import norm  # noqa: PLC0415

        return norm.cdf(x)
    except ImportError:
        pass
    # Fallback: use math.erf for scalar, vectorize for arrays
    import math  # noqa: PLC0415

    scalar = np.ndim(x) == 0
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    result = np.array([0.5 * (1 + math.erf(v / math.sqrt(2))) for v in x_arr.ravel()])
    result = result.reshape(x_arr.shape)
    return result[0] if scalar else result


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal PDF."""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def _bs_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """Compute Black-Scholes d1 and d2."""
    if T <= 0:
        raise ValueError("Time to expiry T must be > 0")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be > 0")
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(d1), float(d2)


def _require_scipy_for_bs() -> None:
    try:
        from scipy.stats import norm  # noqa: F401, PLC0415
    except ImportError as e:
        raise ImportError(
            "scipy is required for Black-Scholes pricing. Install it with: pip install qfpytoolbox[scipy]"
        ) from e


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute the Black-Scholes price of a European call option.

    Args:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free rate (continuously compounded).
        sigma: Volatility of the underlying.

    Returns:
        Call option price.
    """
    _require_scipy_for_bs()
    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    cdf = _norm_cdf
    return float(S * cdf(d1) - K * np.exp(-r * T) * cdf(d2))


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute the Black-Scholes price of a European put option.

    Args:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free rate (continuously compounded).
        sigma: Volatility of the underlying.

    Returns:
        Put option price.
    """
    _require_scipy_for_bs()
    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    cdf = _norm_cdf
    return float(K * np.exp(-r * T) * cdf(-d2) - S * cdf(-d1))


def black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float) -> dict[str, float]:
    """Compute Black-Scholes Greeks for European options.

    Args:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free rate (continuously compounded).
        sigma: Volatility of the underlying.

    Returns:
        Dict with keys: delta_call, delta_put, gamma, theta_call, theta_put, vega, rho_call, rho_put.
    """
    _require_scipy_for_bs()
    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    cdf = _norm_cdf
    pdf = _norm_pdf

    sqrt_T = np.sqrt(T)
    exp_rT = np.exp(-r * T)

    delta_call = float(cdf(d1))
    delta_put = float(delta_call - 1.0)
    gamma = float(pdf(d1) / (S * sigma * sqrt_T))
    theta_call = float((-S * pdf(d1) * sigma / (2 * sqrt_T) - r * K * exp_rT * cdf(d2)) / 365.0)
    theta_put = float((-S * pdf(d1) * sigma / (2 * sqrt_T) + r * K * exp_rT * cdf(-d2)) / 365.0)
    vega = float(S * pdf(d1) * sqrt_T / 100.0)
    rho_call = float(K * T * exp_rT * cdf(d2) / 100.0)
    rho_put = float(-K * T * exp_rT * cdf(-d2) / 100.0)

    return {
        "delta_call": delta_call,
        "delta_put": delta_put,
        "gamma": gamma,
        "theta_call": theta_call,
        "theta_put": theta_put,
        "vega": vega,
        "rho_call": rho_call,
        "rho_put": rho_put,
    }
