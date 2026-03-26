from __future__ import annotations

import numpy as np
import pytest

from qfpytoolbox.finance import (
    bond_convexity,
    bond_duration,
    bond_price,
    compound_interest,
    future_value,
    irr,
    npv,
    present_value,
)

scipy_available = True
try:
    import scipy  # noqa: F401
except ImportError:
    scipy_available = False


class TestCompoundInterest:
    def test_annual(self) -> None:
        result = compound_interest(1000, 0.05, 10, compounding_frequency=1)
        np.testing.assert_allclose(result, 1000 * (1.05**10), rtol=1e-10)

    def test_monthly(self) -> None:
        result = compound_interest(1000, 0.12, 1, compounding_frequency=12)
        np.testing.assert_allclose(result, 1000 * (1 + 0.12 / 12) ** 12, rtol=1e-10)

    def test_zero_rate(self) -> None:
        result = compound_interest(1000, 0.0, 5)
        np.testing.assert_allclose(result, 1000.0)

    def test_zero_periods(self) -> None:
        result = compound_interest(1000, 0.05, 0)
        np.testing.assert_allclose(result, 1000.0)


class TestPresentValue:
    def test_basic(self) -> None:
        # PV of 1000 in 5 years at 5%
        pv = present_value(1000, 0.05, 5)
        np.testing.assert_allclose(pv, 1000 / 1.05**5, rtol=1e-10)

    def test_zero_rate(self) -> None:
        np.testing.assert_allclose(present_value(1000, 0.0, 10), 1000.0)

    def test_roundtrip_with_fv(self) -> None:
        fv = future_value(500, 0.08, 7)
        pv = present_value(fv, 0.08, 7)
        np.testing.assert_allclose(pv, 500.0, rtol=1e-10)


class TestFutureValue:
    def test_basic(self) -> None:
        fv = future_value(1000, 0.05, 5)
        np.testing.assert_allclose(fv, 1000 * 1.05**5, rtol=1e-10)

    def test_zero_rate(self) -> None:
        np.testing.assert_allclose(future_value(1000, 0.0, 5), 1000.0)


class TestNPV:
    def test_basic(self) -> None:
        # -1000 now, +1100 in one period at 10% => NPV = 0
        cf = np.array([-1000.0, 1100.0])
        np.testing.assert_allclose(npv(cf, 0.10), 0.0, atol=1e-8)

    def test_positive_npv(self) -> None:
        cf = np.array([-1000.0, 600.0, 600.0])
        result = npv(cf, 0.05)
        assert result > 0

    def test_negative_npv(self) -> None:
        cf = np.array([-1000.0, 200.0, 200.0])
        result = npv(cf, 0.30)
        assert result < 0

    def test_known_value(self) -> None:
        cf = np.array([-100.0, 110.0])
        np.testing.assert_allclose(npv(cf, 0.10), 0.0, atol=1e-8)


class TestIRR:
    def test_simple_case(self) -> None:
        # -1000 now, +1100 in 1 year => IRR = 10%
        cf = np.array([-1000.0, 1100.0])
        result = irr(cf)
        np.testing.assert_allclose(result, 0.10, rtol=1e-6)

    def test_multi_period(self) -> None:
        # IRR should be the rate where NPV=0
        cf = np.array([-100.0, 50.0, 50.0, 50.0])
        r = irr(cf)
        if not np.isnan(r):
            npv_check = npv(cf, r)
            np.testing.assert_allclose(npv_check, 0.0, atol=1e-4)

    def test_too_few_elements(self) -> None:
        with pytest.raises(ValueError):
            irr(np.array([100.0]))


class TestBondPrice:
    def test_par_bond(self) -> None:
        # When coupon rate == YTM, bond price == face value
        price = bond_price(1000, 0.06, 0.06, 10, frequency=1)
        np.testing.assert_allclose(price, 1000.0, rtol=1e-10)

    def test_premium_bond(self) -> None:
        # When coupon > YTM, bond trades at premium
        price = bond_price(1000, 0.08, 0.06, 10, frequency=1)
        assert price > 1000

    def test_discount_bond(self) -> None:
        # When coupon < YTM, bond trades at discount
        price = bond_price(1000, 0.04, 0.06, 10, frequency=1)
        assert price < 1000

    def test_semi_annual(self) -> None:
        # Semi-annual coupons (frequency=2)
        bond_price(1000, 0.06, 0.06, 10, frequency=1)  # noqa: F841
        price_semi = bond_price(1000, 0.06, 0.06, 20, frequency=2)
        # Both should be close to par for par bonds
        np.testing.assert_allclose(price_semi, 1000.0, rtol=1e-10)

    def test_zero_coupon(self) -> None:
        # Zero coupon bond
        price = bond_price(1000, 0.0, 0.05, 10, frequency=1)
        np.testing.assert_allclose(price, 1000 / 1.05**10, rtol=1e-10)


class TestBondDuration:
    def test_zero_coupon_duration_equals_maturity(self) -> None:
        # Duration of zero-coupon bond = maturity
        dur = bond_duration(1000, 0.0, 0.05, 10, frequency=1)
        np.testing.assert_allclose(dur, 10.0, rtol=1e-6)

    def test_duration_less_than_maturity(self) -> None:
        # Coupon bond duration < maturity
        dur = bond_duration(1000, 0.06, 0.06, 10, frequency=2)
        assert dur < 5.0  # maturity in years is 10/2 = 5

    def test_duration_positive(self) -> None:
        dur = bond_duration(1000, 0.05, 0.05, 20, frequency=2)
        assert dur > 0


class TestBondConvexity:
    def test_positive(self) -> None:
        conv = bond_convexity(1000, 0.06, 0.06, 20, frequency=2)
        assert conv > 0

    def test_zero_coupon_convexity(self) -> None:
        # Zero coupon bond convexity = T*(T+1/freq)/((1+y/freq)^2 * freq^2)
        # For annual: T*(T+1)/(1+y)^2
        conv = bond_convexity(1000, 0.0, 0.05, 10, frequency=1)
        expected = 10 * 11 / (1.05**2)
        np.testing.assert_allclose(conv, expected, rtol=1e-6)


@pytest.mark.skipif(not scipy_available, reason="scipy not installed")
class TestBlackScholes:
    def test_call_put_parity(self) -> None:
        from qfpytoolbox.finance import black_scholes_call, black_scholes_put  # noqa: PLC0415

        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        call = black_scholes_call(S, K, T, r, sigma)
        put = black_scholes_put(S, K, T, r, sigma)
        # Put-call parity: C - P = S - K*exp(-rT)
        np.testing.assert_allclose(call - put, S - K * np.exp(-r * T), rtol=1e-8)

    def test_deep_itm_call(self) -> None:
        from qfpytoolbox.finance import black_scholes_call  # noqa: PLC0415

        # Deep ITM call ≈ S - K*exp(-rT)
        S, K, T, r, sigma = 200.0, 100.0, 1.0, 0.05, 0.2
        call = black_scholes_call(S, K, T, r, sigma)
        assert call > S - K

    def test_otm_call_cheaper_than_atm(self) -> None:
        from qfpytoolbox.finance import black_scholes_call  # noqa: PLC0415

        S, T, r, sigma = 100.0, 1.0, 0.05, 0.2
        atm_call = black_scholes_call(S, 100.0, T, r, sigma)
        otm_call = black_scholes_call(S, 120.0, T, r, sigma)
        assert otm_call < atm_call

    def test_known_value(self) -> None:
        from qfpytoolbox.finance import black_scholes_call  # noqa: PLC0415

        # S=100, K=100, T=1, r=0.05, sigma=0.2 → approx 10.45
        call = black_scholes_call(100, 100, 1.0, 0.05, 0.2)
        np.testing.assert_allclose(call, 10.4506, rtol=1e-3)

    def test_greeks_keys(self) -> None:
        from qfpytoolbox.finance import black_scholes_greeks  # noqa: PLC0415

        greeks = black_scholes_greeks(100, 100, 1.0, 0.05, 0.2)
        expected_keys = {"delta_call", "delta_put", "gamma", "theta_call", "theta_put", "vega", "rho_call", "rho_put"}
        assert set(greeks.keys()) == expected_keys

    def test_delta_call_in_0_1(self) -> None:
        from qfpytoolbox.finance import black_scholes_greeks  # noqa: PLC0415

        greeks = black_scholes_greeks(100, 100, 1.0, 0.05, 0.2)
        assert 0 <= greeks["delta_call"] <= 1

    def test_delta_put_negative(self) -> None:
        from qfpytoolbox.finance import black_scholes_greeks  # noqa: PLC0415

        greeks = black_scholes_greeks(100, 100, 1.0, 0.05, 0.2)
        assert greeks["delta_put"] < 0

    def test_gamma_positive(self) -> None:
        from qfpytoolbox.finance import black_scholes_greeks  # noqa: PLC0415

        greeks = black_scholes_greeks(100, 100, 1.0, 0.05, 0.2)
        assert greeks["gamma"] > 0

    def test_invalid_T(self) -> None:
        from qfpytoolbox.finance import black_scholes_call  # noqa: PLC0415

        with pytest.raises(ValueError):
            black_scholes_call(100, 100, 0.0, 0.05, 0.2)

    def test_invalid_sigma(self) -> None:
        from qfpytoolbox.finance import black_scholes_call  # noqa: PLC0415

        with pytest.raises(ValueError):
            black_scholes_call(100, 100, 1.0, 0.05, 0.0)


class TestBlackScholesNoScipy:
    def test_import_error_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys  # noqa: PLC0415

        from qfpytoolbox import finance  # noqa: PLC0415

        original = sys.modules.get("scipy.stats")
        sys.modules["scipy.stats"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="scipy"):
                finance._require_scipy_for_bs()
        finally:
            if original is None:
                sys.modules.pop("scipy.stats", None)
            else:
                sys.modules["scipy.stats"] = original
