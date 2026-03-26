from __future__ import annotations

import numpy as np


def portfolio_return(weights: object, expected_returns: object) -> float:
    """Compute the expected portfolio return."""
    w = np.asarray(weights, dtype=float)
    er = np.asarray(expected_returns, dtype=float)
    return float(w @ er)


def portfolio_volatility(weights: object, cov_matrix: object) -> float:
    """Compute the portfolio volatility (standard deviation)."""
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    var = float(w @ cov @ w)
    return float(np.sqrt(max(var, 0.0)))


def portfolio_sharpe(
    weights: object,
    expected_returns: object,
    cov_matrix: object,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute the portfolio Sharpe ratio."""
    ret = portfolio_return(weights, expected_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    if vol == 0:
        return float("nan")
    return float((ret - risk_free_rate) / vol)


def _try_import_scipy() -> object:
    try:
        import scipy.optimize as opt  # noqa: PLC0415

        return opt
    except ImportError as e:
        raise ImportError(
            "scipy is required for portfolio optimization. Install it with: pip install qfpytoolbox[scipy]"
        ) from e


def min_variance_portfolio(
    expected_returns: object,
    cov_matrix: object,
    allow_short: bool = False,
) -> dict[str, object]:
    """Find the minimum variance portfolio.

    Uses scipy.optimize if available. Returns dict with weights, expected_return, volatility.
    """
    opt = _try_import_scipy()
    er = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    n = len(er)

    def objective(w: np.ndarray) -> float:
        return portfolio_volatility(w, cov)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if allow_short:
        bounds = None
    else:
        bounds = [(0.0, 1.0)] * n

    w0 = np.ones(n) / n
    result = opt.minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x
    return {
        "weights": weights,
        "expected_return": portfolio_return(weights, er),
        "volatility": portfolio_volatility(weights, cov),
    }


def max_sharpe_portfolio(
    expected_returns: object,
    cov_matrix: object,
    risk_free_rate: float = 0.0,
    allow_short: bool = False,
) -> dict[str, object]:
    """Find the maximum Sharpe ratio portfolio.

    Returns dict with weights, expected_return, volatility, sharpe.
    """
    opt = _try_import_scipy()
    er = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    n = len(er)

    def neg_sharpe(w: np.ndarray) -> float:
        return -portfolio_sharpe(w, er, cov, risk_free_rate)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = None if allow_short else [(0.0, 1.0)] * n
    w0 = np.ones(n) / n
    result = opt.minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x
    ret = portfolio_return(weights, er)
    vol = portfolio_volatility(weights, cov)
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else float("nan")
    return {
        "weights": weights,
        "expected_return": ret,
        "volatility": vol,
        "sharpe": sharpe,
    }


def efficient_frontier(
    expected_returns: object,
    cov_matrix: object,
    n_points: int = 50,
    allow_short: bool = False,
) -> dict[str, object]:
    """Compute the efficient frontier.

    Returns dict with keys 'returns', 'volatilities', 'weights'.
    """
    opt = _try_import_scipy()
    er = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    n = len(er)

    min_ret = float(np.min(er))
    max_ret = float(np.max(er))
    target_returns = np.linspace(min_ret, max_ret, n_points)

    all_weights = []
    all_vols = []
    all_rets = []

    bounds = None if allow_short else [(0.0, 1.0)] * n
    w0 = np.ones(n) / n

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: portfolio_return(w, er) - t},
        ]
        result = opt.minimize(
            lambda w: portfolio_volatility(w, cov),
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if result.success:
            all_weights.append(result.x)
            all_vols.append(portfolio_volatility(result.x, cov))
            all_rets.append(portfolio_return(result.x, er))
        else:
            # Skip infeasible targets
            continue

    return {
        "returns": np.array(all_rets),
        "volatilities": np.array(all_vols),
        "weights": np.array(all_weights),
    }


def risk_parity_portfolio(cov_matrix: object) -> dict[str, object]:
    """Find the risk parity portfolio (equal risk contribution).

    Returns dict with keys 'weights', 'volatility'.
    """
    opt = _try_import_scipy()
    cov = np.asarray(cov_matrix, dtype=float)
    n = cov.shape[0]

    def risk_budget_objective(w: np.ndarray) -> float:
        vol = portfolio_volatility(w, cov)
        if vol == 0:
            return 0.0
        marginal_contrib = cov @ w / vol
        risk_contrib = w * marginal_contrib
        target = vol / n
        return float(np.sum((risk_contrib - target) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, 1.0)] * n
    w0 = np.ones(n) / n
    result = opt.minimize(risk_budget_objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x
    return {
        "weights": weights,
        "volatility": portfolio_volatility(weights, cov),
    }


def equal_weight_portfolio(n_assets: int) -> dict[str, object]:
    """Construct an equal-weight portfolio.

    Returns dict with key 'weights'.
    """
    if n_assets < 1:
        raise ValueError("n_assets must be >= 1")
    return {"weights": np.ones(n_assets) / n_assets}
