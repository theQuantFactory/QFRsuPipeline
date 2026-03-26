from __future__ import annotations

import numpy as np
import pytest

from qfpytoolbox.timeseries import (
    annualize_returns,
    autocorrelation,
    cumulative_returns,
    drawdown_series,
    ewm_mean,
    ewm_std,
    log_returns,
    rolling_beta,
    rolling_correlation,
    rolling_max_drawdown,
    rolling_mean,
    rolling_sharpe,
    rolling_std,
    simple_returns,
)

RNG = np.random.default_rng(0)
RETURNS = RNG.normal(0.001, 0.01, 100)
MARKET = RNG.normal(0.001, 0.012, 100)
PRICES = np.cumprod(1 + RETURNS) * 100


class TestRollingMean:
    def test_nan_prefix(self) -> None:
        r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_mean(r, window=3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert not np.isnan(result[2])

    def test_values(self) -> None:
        r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_mean(r, window=3)
        np.testing.assert_allclose(result[2], 2.0)
        np.testing.assert_allclose(result[4], 4.0)

    def test_length_preserved(self) -> None:
        r = np.ones(50)
        assert len(rolling_mean(r, window=5)) == 50

    def test_window_one(self) -> None:
        r = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(rolling_mean(r, window=1), r)


class TestRollingStd:
    def test_nan_prefix(self) -> None:
        r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_std(r, window=3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])

    def test_values(self) -> None:
        r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_std(r, window=3)
        np.testing.assert_allclose(result[2], np.std([1, 2, 3], ddof=1))

    def test_length_preserved(self) -> None:
        assert len(rolling_std(RETURNS, window=10)) == len(RETURNS)


class TestRollingSharpe:
    def test_nan_prefix(self) -> None:
        result = rolling_sharpe(RETURNS, window=20)
        assert np.all(np.isnan(result[:19]))

    def test_length_preserved(self) -> None:
        assert len(rolling_sharpe(RETURNS, window=20)) == len(RETURNS)

    def test_finite_values(self) -> None:
        result = rolling_sharpe(RETURNS, window=20)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0


class TestRollingMaxDrawdown:
    def test_nan_prefix(self) -> None:
        result = rolling_max_drawdown(RETURNS, window=20)
        assert np.all(np.isnan(result[:19]))

    def test_non_positive(self) -> None:
        result = rolling_max_drawdown(RETURNS, window=20)
        valid = result[~np.isnan(result)]
        assert np.all(valid <= 0)

    def test_length_preserved(self) -> None:
        assert len(rolling_max_drawdown(RETURNS, window=10)) == len(RETURNS)


class TestRollingCorrelation:
    def test_self_correlation_is_one(self) -> None:
        result = rolling_correlation(RETURNS, RETURNS, window=20)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, np.ones(len(valid)), atol=1e-10)

    def test_length_preserved(self) -> None:
        assert len(rolling_correlation(RETURNS, MARKET, window=20)) == len(RETURNS)

    def test_mismatch_length(self) -> None:
        with pytest.raises(ValueError):
            rolling_correlation(RETURNS[:10], RETURNS[:20], window=5)


class TestRollingBeta:
    def test_self_beta_is_one(self) -> None:
        result = rolling_beta(RETURNS, RETURNS, window=20)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, np.ones(len(valid)), atol=1e-10)

    def test_length_preserved(self) -> None:
        assert len(rolling_beta(RETURNS, MARKET, window=20)) == len(RETURNS)


class TestEwmMean:
    def test_span(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ewm_mean(data, span=2)
        assert len(result) == len(data)
        np.testing.assert_allclose(result[0], 1.0)

    def test_halflife(self) -> None:
        data = np.ones(10)
        result = ewm_mean(data, halflife=5)
        np.testing.assert_allclose(result, np.ones(10))

    def test_alpha(self) -> None:
        data = np.array([1.0, 0.0])
        result = ewm_mean(data, alpha=0.5)
        np.testing.assert_allclose(result[0], 1.0)
        np.testing.assert_allclose(result[1], 0.5)

    def test_mutual_exclusion(self) -> None:
        with pytest.raises(ValueError):
            ewm_mean(np.ones(5), span=2, halflife=3)

    def test_at_least_one_required(self) -> None:
        with pytest.raises(ValueError, match="Exactly one of span, halflife, or alpha"):
            ewm_mean(np.ones(5))


class TestEwmStd:
    def test_constant_series_zero_std(self) -> None:
        data = np.ones(20)
        result = ewm_std(data, span=5)
        np.testing.assert_allclose(result[1:], 0.0, atol=1e-15)

    def test_length_preserved(self) -> None:
        assert len(ewm_std(RETURNS, span=10)) == len(RETURNS)


class TestCumulativeReturns:
    def test_starts_at_one(self) -> None:
        r = np.array([0.1, -0.05, 0.08])
        cum = cumulative_returns(r)
        np.testing.assert_allclose(cum[0], 1.0)

    def test_length(self) -> None:
        r = np.array([0.1, -0.05, 0.08])
        assert len(cumulative_returns(r)) == len(r) + 1

    def test_known_values(self) -> None:
        r = np.array([0.1, 0.1])
        cum = cumulative_returns(r)
        np.testing.assert_allclose(cum[1], 1.1)
        np.testing.assert_allclose(cum[2], 1.21)


class TestLogReturns:
    def test_known_values(self) -> None:
        prices = np.array([100.0, 110.0, 99.0])
        lr = log_returns(prices)
        np.testing.assert_allclose(lr[0], np.log(110 / 100))
        np.testing.assert_allclose(lr[1], np.log(99 / 110))

    def test_length(self) -> None:
        prices = np.arange(1, 11, dtype=float)
        assert len(log_returns(prices)) == 9

    def test_invalid_input(self) -> None:
        with pytest.raises(ValueError):
            log_returns([100.0])  # only one element


class TestSimpleReturns:
    def test_known_values(self) -> None:
        prices = np.array([100.0, 110.0, 99.0])
        sr = simple_returns(prices)
        np.testing.assert_allclose(sr[0], 0.1)
        np.testing.assert_allclose(sr[1], (99 - 110) / 110)

    def test_length(self) -> None:
        prices = np.arange(1, 11, dtype=float)
        assert len(simple_returns(prices)) == 9

    def test_roundtrip_with_cumulative(self) -> None:
        prices = PRICES
        sr = simple_returns(prices)
        cum = cumulative_returns(sr)
        np.testing.assert_allclose(cum[-1], prices[-1] / prices[0], rtol=1e-10)


class TestAnnualizeReturns:
    def test_one_year(self) -> None:
        total_return = 0.1
        result = annualize_returns(total_return, periods=252, periods_per_year=252)
        np.testing.assert_allclose(result, 0.1, rtol=1e-10)

    def test_two_years(self) -> None:
        total_return = (1.1**2) - 1  # 10% per year for 2 years
        result = annualize_returns(total_return, periods=504, periods_per_year=252)
        np.testing.assert_allclose(result, 0.1, rtol=1e-8)

    def test_invalid_periods(self) -> None:
        with pytest.raises(ValueError):
            annualize_returns(0.1, periods=0)


class TestDrawdownSeries:
    def test_non_positive(self) -> None:
        dd = drawdown_series(RETURNS)
        assert np.all(dd <= 1e-10)

    def test_monotone_up_no_drawdown(self) -> None:
        r = np.array([0.01, 0.01, 0.01, 0.01])
        dd = drawdown_series(r)
        np.testing.assert_allclose(dd, 0.0, atol=1e-10)

    def test_known_drawdown(self) -> None:
        prices = np.array([100.0, 120.0, 90.0, 110.0])
        r = (prices[1:] - prices[:-1]) / prices[:-1]
        dd = drawdown_series(r)
        np.testing.assert_allclose(dd[1], (90 - 120) / 120, rtol=1e-10)


class TestAutocorrelation:
    def test_self_ac_lag1_iid(self) -> None:
        # IID series should have near-zero autocorrelation
        rng = np.random.default_rng(99)
        data = rng.normal(0, 1, 5000)
        ac = autocorrelation(data, lag=1)
        assert abs(ac) < 0.05

    def test_ar1_positive(self) -> None:
        # AR(1) series with positive coefficient
        rng = np.random.default_rng(1)
        n = 1000
        data = np.zeros(n)
        for i in range(1, n):
            data[i] = 0.7 * data[i - 1] + rng.normal(0, 1)
        ac = autocorrelation(data, lag=1)
        assert ac > 0.5

    def test_invalid_lag(self) -> None:
        with pytest.raises(ValueError):
            autocorrelation(np.ones(10), lag=0)

    def test_lag_too_large(self) -> None:
        with pytest.raises(ValueError):
            autocorrelation(np.ones(5), lag=5)
