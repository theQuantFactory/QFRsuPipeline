from __future__ import annotations

import numpy as np
import pytest

from qfpytoolbox.stats import (
    alpha,
    beta,
    calmar_ratio,
    conditional_var,
    correlation_matrix,
    covariance_matrix,
    excess_kurtosis,
    information_ratio,
    max_drawdown,
    mean_return,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    tracking_error,
    value_at_risk,
    volatility,
)

# Reproducible test data
RNG = np.random.default_rng(42)
RETURNS = RNG.normal(0.001, 0.01, 252)
MARKET = RNG.normal(0.001, 0.012, 252)
BENCHMARK = RNG.normal(0.0008, 0.009, 252)


class TestMeanReturn:
    def test_basic(self) -> None:
        r = np.array([0.01, 0.02, -0.01, 0.03])
        np.testing.assert_allclose(mean_return(r), np.mean(r))

    def test_annualize(self) -> None:
        r = np.array([0.001] * 252)
        np.testing.assert_allclose(mean_return(r, annualize=True, periods_per_year=252), 0.252)

    def test_list_input(self) -> None:
        r = [0.01, 0.02, -0.01]
        assert isinstance(mean_return(r), float)

    def test_invalid_empty(self) -> None:
        with pytest.raises(ValueError):
            mean_return([])

    def test_invalid_2d(self) -> None:
        with pytest.raises(ValueError):
            mean_return([[0.01, 0.02], [0.03, 0.04]])


class TestVolatility:
    def test_basic(self) -> None:
        r = np.array([0.01, 0.02, -0.01, 0.03, 0.005])
        np.testing.assert_allclose(volatility(r), np.std(r, ddof=1))

    def test_annualize(self) -> None:
        r = np.ones(252) * 0.001
        vol = volatility(r, annualize=True, periods_per_year=252)
        # std of constant is 0
        np.testing.assert_allclose(vol, 0.0, atol=1e-12)

    def test_annualize_scaling(self) -> None:
        r = np.array([0.01, -0.01, 0.02, -0.02, 0.005])
        daily_vol = volatility(r, annualize=False)
        ann_vol = volatility(r, annualize=True, periods_per_year=252)
        np.testing.assert_allclose(ann_vol, daily_vol * np.sqrt(252))


class TestSkewness:
    def test_symmetric(self) -> None:
        r = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
        # Symmetric distribution should have near-zero skewness
        assert abs(skewness(r)) < 1e-10

    def test_positive_skew(self) -> None:
        r = np.array([0.01, 0.01, 0.01, 0.01, 0.1])
        assert skewness(r) > 0

    def test_too_few_obs(self) -> None:
        with pytest.raises(ValueError):
            skewness([0.01, 0.02])


class TestExcessKurtosis:
    def test_normal_approx(self) -> None:
        rng = np.random.default_rng(0)
        r = rng.normal(0, 1, 10000)
        # Should be close to 0 for normal distribution
        assert abs(excess_kurtosis(r)) < 0.2

    def test_too_few_obs(self) -> None:
        with pytest.raises(ValueError):
            excess_kurtosis([0.01, 0.02, 0.03])


class TestSharpeRatio:
    def test_positive(self) -> None:
        r = np.array([0.01, 0.02, 0.015, 0.018, 0.012])
        sr = sharpe_ratio(r)
        assert sr > 0

    def test_zero_vol_is_nan(self) -> None:
        r = np.ones(10) * 0.01
        assert np.isnan(sharpe_ratio(r))

    def test_annualized(self) -> None:
        r = RETURNS
        daily_sr = sharpe_ratio(r, annualize=False)
        ann_sr = sharpe_ratio(r, annualize=True, periods_per_year=252)
        np.testing.assert_allclose(ann_sr, daily_sr * np.sqrt(252), rtol=1e-10)

    def test_with_rf(self) -> None:
        r = np.array([0.01, 0.02, 0.015])
        sr_no_rf = sharpe_ratio(r, risk_free_rate=0.0)
        sr_with_rf = sharpe_ratio(r, risk_free_rate=0.005)
        assert sr_with_rf < sr_no_rf


class TestSortinoRatio:
    def test_positive(self) -> None:
        r = np.array([0.01, 0.02, -0.005, 0.015, 0.012])
        assert sortino_ratio(r) > 0

    def test_no_downside(self) -> None:
        r = np.array([0.01, 0.02, 0.03])
        assert np.isnan(sortino_ratio(r))

    def test_annualized(self) -> None:
        r = RETURNS
        daily = sortino_ratio(r, annualize=False)
        ann = sortino_ratio(r, annualize=True, periods_per_year=252)
        np.testing.assert_allclose(ann, daily * np.sqrt(252), rtol=1e-10)


class TestMaxDrawdown:
    def test_no_drawdown(self) -> None:
        r = np.array([0.01, 0.02, 0.03])
        np.testing.assert_allclose(max_drawdown(r), 0.0)

    def test_known_drawdown(self) -> None:
        # Price goes 100 -> 110 -> 80 -> 90: drawdown = (80-110)/110 ≈ -0.2727
        prices = np.array([100.0, 110.0, 80.0, 90.0])
        r = (prices[1:] - prices[:-1]) / prices[:-1]
        mdd = max_drawdown(r)
        np.testing.assert_allclose(mdd, (80 - 110) / 110, rtol=1e-10)

    def test_negative_value(self) -> None:
        r = np.array([0.1, -0.2, 0.1, -0.15])
        assert max_drawdown(r) < 0


class TestCalmarRatio:
    def test_positive(self) -> None:
        r = RETURNS
        cr = calmar_ratio(r)
        assert np.isfinite(cr)

    def test_zero_drawdown_is_nan(self) -> None:
        r = np.array([0.01, 0.02, 0.03])
        assert np.isnan(calmar_ratio(r))


class TestVaR:
    def test_historical_var(self) -> None:
        r = np.arange(-10, 10) / 100.0  # -0.10 to 0.09
        var = value_at_risk(r, confidence=0.95)
        assert var <= 0

    def test_var_is_negative(self) -> None:
        var = value_at_risk(RETURNS, confidence=0.95)
        assert var < 0

    def test_unknown_method(self) -> None:
        with pytest.raises(ValueError):
            value_at_risk(RETURNS, method="gaussian")


class TestCVaR:
    def test_cvar_leq_var(self) -> None:
        cvar = conditional_var(RETURNS, confidence=0.95)
        var = value_at_risk(RETURNS, confidence=0.95)
        assert cvar <= var

    def test_cvar_negative(self) -> None:
        assert conditional_var(RETURNS) < 0


class TestInformationRatio:
    def test_basic(self) -> None:
        ir = information_ratio(RETURNS, BENCHMARK)
        assert np.isfinite(ir)

    def test_same_as_benchmark(self) -> None:
        r = np.array([0.01, 0.02, 0.03])
        assert np.isnan(information_ratio(r, r))

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            information_ratio(RETURNS[:10], RETURNS[:20])


class TestTrackingError:
    def test_positive(self) -> None:
        te = tracking_error(RETURNS, BENCHMARK)
        assert te > 0

    def test_zero_for_same(self) -> None:
        r = np.array([0.01, 0.02, 0.03])
        np.testing.assert_allclose(tracking_error(r, r), 0.0)

    def test_annualized(self) -> None:
        te_daily = tracking_error(RETURNS, BENCHMARK, annualize=False)
        te_ann = tracking_error(RETURNS, BENCHMARK, annualize=True, periods_per_year=252)
        np.testing.assert_allclose(te_ann, te_daily * np.sqrt(252))


class TestBeta:
    def test_beta_equals_one_for_identical(self) -> None:
        r = RETURNS
        np.testing.assert_allclose(beta(r, r), 1.0, rtol=1e-10)

    def test_beta_sign(self) -> None:
        b = beta(RETURNS, MARKET)
        assert np.isfinite(b)

    def test_zero_market_variance_is_nan(self) -> None:
        r = np.array([0.01, 0.02, 0.03])
        m = np.ones(3) * 0.01
        assert np.isnan(beta(r, m))


class TestAlpha:
    def test_alpha_finite(self) -> None:
        a = alpha(RETURNS, MARKET)
        assert np.isfinite(a)

    def test_alpha_zero_for_market(self) -> None:
        # Market vs itself with rf=0 should give alpha ≈ 0
        a = alpha(MARKET, MARKET, risk_free_rate=0.0)
        np.testing.assert_allclose(a, 0.0, atol=1e-12)


class TestCorrelationMatrix:
    def test_shape(self) -> None:
        mat = RNG.normal(0, 1, (100, 3))
        corr = correlation_matrix(mat)
        assert corr.shape == (3, 3)

    def test_diagonal_ones(self) -> None:
        mat = RNG.normal(0, 1, (100, 4))
        corr = correlation_matrix(mat)
        np.testing.assert_allclose(np.diag(corr), np.ones(4), atol=1e-10)

    def test_symmetric(self) -> None:
        mat = RNG.normal(0, 1, (100, 3))
        corr = correlation_matrix(mat)
        np.testing.assert_allclose(corr, corr.T)


class TestCovarianceMatrix:
    def test_shape(self) -> None:
        mat = RNG.normal(0, 1, (100, 3))
        cov = covariance_matrix(mat)
        assert cov.shape == (3, 3)

    def test_positive_semidefinite(self) -> None:
        mat = RNG.normal(0, 1, (100, 4))
        cov = covariance_matrix(mat)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)

    def test_annualized(self) -> None:
        mat = RNG.normal(0, 1, (100, 2))
        cov_daily = covariance_matrix(mat, annualize=False)
        cov_ann = covariance_matrix(mat, annualize=True, periods_per_year=252)
        np.testing.assert_allclose(cov_ann, cov_daily * 252)
