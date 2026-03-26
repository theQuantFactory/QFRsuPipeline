from __future__ import annotations

import numpy as np
import pytest

from qfpytoolbox.optim import (
    equal_weight_portfolio,
    portfolio_return,
    portfolio_sharpe,
    portfolio_volatility,
)

# These tests require scipy
scipy_available = True
try:
    import scipy  # noqa: F401
except ImportError:
    scipy_available = False

pytestmark = pytest.mark.skipif(not scipy_available, reason="scipy not installed")

from qfpytoolbox.optim import (  # noqa: E402
    efficient_frontier,
    max_sharpe_portfolio,
    min_variance_portfolio,
    risk_parity_portfolio,
)

# Test data
RNG = np.random.default_rng(7)
N = 4
EXPECTED_RETURNS = np.array([0.10, 0.12, 0.08, 0.15])  # annual
RAW = RNG.normal(0, 1, (252, N))
# Build a reasonable cov matrix
COV = np.cov(RAW.T) * 0.01  # scale to realistic values


class TestPortfolioReturnVol:
    def test_equal_weight(self) -> None:
        w = np.ones(N) / N
        ret = portfolio_return(w, EXPECTED_RETURNS)
        np.testing.assert_allclose(ret, np.mean(EXPECTED_RETURNS), rtol=1e-10)

    def test_vol_non_negative(self) -> None:
        w = np.ones(N) / N
        vol = portfolio_volatility(w, COV)
        assert vol >= 0

    def test_single_asset(self) -> None:
        w = np.array([1.0])
        er = np.array([0.1])
        cov = np.array([[0.04]])
        np.testing.assert_allclose(portfolio_return(w, er), 0.1)
        np.testing.assert_allclose(portfolio_volatility(w, cov), 0.2)

    def test_sharpe_positive(self) -> None:
        w = np.ones(N) / N
        sr = portfolio_sharpe(w, EXPECTED_RETURNS, COV, risk_free_rate=0.02)
        assert sr > 0

    def test_sharpe_zero_vol_is_nan(self) -> None:
        w = np.array([1.0])
        er = np.array([0.1])
        cov = np.array([[0.0]])
        assert np.isnan(portfolio_sharpe(w, er, cov))


class TestEqualWeight:
    def test_weights_sum_to_one(self) -> None:
        result = equal_weight_portfolio(4)
        np.testing.assert_allclose(np.sum(result["weights"]), 1.0)

    def test_weights_equal(self) -> None:
        result = equal_weight_portfolio(4)
        np.testing.assert_allclose(result["weights"], np.ones(4) / 4)

    def test_single_asset(self) -> None:
        result = equal_weight_portfolio(1)
        np.testing.assert_allclose(result["weights"], [1.0])

    def test_invalid_n(self) -> None:
        with pytest.raises(ValueError):
            equal_weight_portfolio(0)


class TestMinVariance:
    def test_weights_sum_to_one(self) -> None:
        result = min_variance_portfolio(EXPECTED_RETURNS, COV)
        np.testing.assert_allclose(np.sum(result["weights"]), 1.0, atol=1e-6)

    def test_weights_non_negative(self) -> None:
        result = min_variance_portfolio(EXPECTED_RETURNS, COV)
        assert np.all(result["weights"] >= -1e-6)

    def test_returns_dict_keys(self) -> None:
        result = min_variance_portfolio(EXPECTED_RETURNS, COV)
        assert "weights" in result
        assert "expected_return" in result
        assert "volatility" in result

    def test_lower_vol_than_equal_weight(self) -> None:
        result = min_variance_portfolio(EXPECTED_RETURNS, COV)
        ew_vol = portfolio_volatility(np.ones(N) / N, COV)
        # Min variance should not exceed equal weight vol (or be very close)
        assert result["volatility"] <= ew_vol + 1e-6

    def test_allow_short(self) -> None:
        result = min_variance_portfolio(EXPECTED_RETURNS, COV, allow_short=True)
        np.testing.assert_allclose(np.sum(result["weights"]), 1.0, atol=1e-6)


class TestMaxSharpe:
    def test_weights_sum_to_one(self) -> None:
        result = max_sharpe_portfolio(EXPECTED_RETURNS, COV, risk_free_rate=0.02)
        np.testing.assert_allclose(np.sum(result["weights"]), 1.0, atol=1e-6)

    def test_keys(self) -> None:
        result = max_sharpe_portfolio(EXPECTED_RETURNS, COV)
        assert set(result.keys()) == {"weights", "expected_return", "volatility", "sharpe"}

    def test_sharpe_better_than_equal_weight(self) -> None:
        result = max_sharpe_portfolio(EXPECTED_RETURNS, COV, risk_free_rate=0.02)
        ew_sharpe = portfolio_sharpe(np.ones(N) / N, EXPECTED_RETURNS, COV, risk_free_rate=0.02)
        assert result["sharpe"] >= ew_sharpe - 1e-4


class TestEfficientFrontier:
    def test_keys(self) -> None:
        result = efficient_frontier(EXPECTED_RETURNS, COV, n_points=10)
        assert "returns" in result
        assert "volatilities" in result
        assert "weights" in result

    def test_non_empty(self) -> None:
        result = efficient_frontier(EXPECTED_RETURNS, COV, n_points=10)
        assert len(result["returns"]) > 0

    def test_volatilities_positive(self) -> None:
        result = efficient_frontier(EXPECTED_RETURNS, COV, n_points=10)
        assert np.all(result["volatilities"] >= 0)


class TestRiskParity:
    def test_weights_sum_to_one(self) -> None:
        result = risk_parity_portfolio(COV)
        np.testing.assert_allclose(np.sum(result["weights"]), 1.0, atol=1e-5)

    def test_weights_positive(self) -> None:
        result = risk_parity_portfolio(COV)
        assert np.all(result["weights"] > 0)

    def test_keys(self) -> None:
        result = risk_parity_portfolio(COV)
        assert "weights" in result
        assert "volatility" in result

    def test_equal_vol_equal_weights(self) -> None:
        # For identity covariance matrix, risk parity should give equal weights
        cov = np.eye(3) * 0.01
        result = risk_parity_portfolio(cov)
        np.testing.assert_allclose(result["weights"], np.ones(3) / 3, atol=1e-4)


class TestSciPyNotAvailableError:
    """Test that helpful ImportError is raised when scipy is not available."""

    def test_import_error_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys  # noqa: PLC0415

        # Temporarily hide scipy
        original = sys.modules.get("scipy.optimize")
        sys.modules["scipy.optimize"] = None  # type: ignore[assignment]
        try:
            from qfpytoolbox import optim  # noqa: PLC0415

            with pytest.raises(ImportError, match="scipy"):
                optim._try_import_scipy()
        finally:
            if original is None:
                sys.modules.pop("scipy.optimize", None)
            else:
                sys.modules["scipy.optimize"] = original
