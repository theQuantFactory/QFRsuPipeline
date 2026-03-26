from __future__ import annotations

import numpy as np


def validate_returns(returns: object) -> np.ndarray:
    """Validate and convert input to a 1D numpy float array."""
    arr = np.asarray(returns, dtype=float)
    if arr.ndim == 0:
        raise ValueError("returns must be array-like with at least one element")
    if arr.ndim != 1:
        raise ValueError(f"returns must be 1D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("returns must not be empty")
    return arr


def validate_2d_returns(returns_matrix: object) -> np.ndarray:
    """Validate and convert input to a 2D numpy float array (n_obs, n_assets)."""
    arr = np.asarray(returns_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"returns_matrix must be 2D (n_obs, n_assets), got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError("returns_matrix must have at least 2 observations")
    if arr.shape[1] < 1:
        raise ValueError("returns_matrix must have at least 1 asset")
    return arr


def _alpha_from_span(span: float) -> float:
    """Convert EWM span to smoothing factor alpha."""
    if span < 1:
        raise ValueError(f"span must be >= 1, got {span}")
    return 2.0 / (span + 1.0)


def _alpha_from_halflife(halflife: float) -> float:
    """Convert EWM half-life to smoothing factor alpha."""
    if halflife <= 0:
        raise ValueError(f"halflife must be > 0, got {halflife}")
    return 1.0 - np.exp(-np.log(2.0) / halflife)


def _ewm_alpha(
    span: float | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
) -> float:
    """Resolve EWM smoothing factor from span, halflife, or alpha."""
    provided = sum(x is not None for x in (span, halflife, alpha))
    if provided != 1:
        raise ValueError("Exactly one of span, halflife, or alpha must be provided")
    if span is not None:
        return _alpha_from_span(span)
    if halflife is not None:
        return _alpha_from_halflife(halflife)
    assert alpha is not None
    if not (0 < alpha <= 1):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    return float(alpha)
