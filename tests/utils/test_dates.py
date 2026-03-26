"""Tests for date utilities — mirrors test/utils/dates-tests.jl."""

from __future__ import annotations

from datetime import date

import pytest

from qfpytoolbox.utils.dates import (
    add_days_convention,
    date2int,
    days_between,
    int2date,
    yearfrac,
)

# ---------------------------------------------------------------------------
# date2int / int2date
# ---------------------------------------------------------------------------


class TestDate2Int:
    def test_basic(self):
        assert date2int(date(2025, 12, 31)) == 20251231

    def test_round_trip(self):
        d = date(2020, 1, 5)
        assert int2date(date2int(d)) == d

    def test_leading_zero_month(self):
        assert date2int(date(2024, 1, 9)) == 20240109


class TestInt2Date:
    def test_basic(self):
        assert int2date(20251231) == date(2025, 12, 31)

    def test_jan_1(self):
        assert int2date(20250101) == date(2025, 1, 1)


# ---------------------------------------------------------------------------
# days_between
# ---------------------------------------------------------------------------


class TestDaysBetween:
    def test_positive(self):
        assert days_between(20250101, 20250103) == 2

    def test_negative(self):
        assert days_between(20250103, 20250101) == -2

    def test_same(self):
        assert days_between(20250101, 20250101) == 0

    def test_cross_year(self):
        assert days_between(20241231, 20250101) == 1


# ---------------------------------------------------------------------------
# yearfrac
# ---------------------------------------------------------------------------


class TestYearFrac:
    def test_actual365(self):
        yf = yearfrac(20250101, 20251231, basis="actual365")
        assert abs(yf - 364 / 365) < 1e-10

    def test_actual360(self):
        yf = yearfrac(20250101, 20250701, basis="actual360")
        assert abs(yf - 181 / 360) < 1e-10

    def test_actual365_25(self):
        yf = yearfrac(20240101, 20250101, basis="actual365_25")
        assert abs(yf - 366 / 365.25) < 1e-10

    def test_actual_gregorian(self):
        yf = yearfrac(20250101, 20260101, basis="actual")
        assert abs(yf - 365 / 365.2425) < 1e-10

    def test_thirty360_us(self):
        # 30/360 US: from 2020-01-31 to 2020-03-31
        yf = yearfrac(20200131, 20200331, basis="thirty360")
        assert abs(yf - 60 / 360) < 1e-10

    def test_thirty360_eu(self):
        yf = yearfrac(20200131, 20200331, basis="thirty360_eu")
        assert abs(yf - 60 / 360) < 1e-10

    def test_unknown_basis_raises(self):
        with pytest.raises(ValueError, match="Unknown basis"):
            yearfrac(20250101, 20250201, basis="invalid")

    def test_negative_yearfrac(self):
        yf = yearfrac(20250201, 20250101, basis="actual365")
        assert yf < 0


# ---------------------------------------------------------------------------
# add_days_convention
# ---------------------------------------------------------------------------


class TestAddDaysConvention:
    def test_following_no_holiday(self):
        # Mon 2025-01-06 + 0 days = Mon → business
        result = add_days_convention(date(2025, 1, 6), 0, convention="following")
        assert result == date(2025, 1, 6)

    def test_following_lands_on_saturday(self):
        # 2025-01-03 (Fri) + 1 = Sat 2025-01-04 → following → Mon 2025-01-06
        result = add_days_convention(date(2025, 1, 3), 1, convention="following")
        assert result == date(2025, 1, 6)

    def test_previous_lands_on_sunday(self):
        # 2025-01-04 (Sat) + 1 = Sun 2025-01-05 → previous → Fri 2025-01-03
        result = add_days_convention(date(2025, 1, 4), 1, convention="previous")
        assert result == date(2025, 1, 3)

    def test_modified_following_stays_same_month(self):
        # 2025-01-28 (Tue) + 2 = Thu 2025-01-30 → business
        result = add_days_convention(date(2025, 1, 28), 2, convention="modified_following")
        assert result == date(2025, 1, 30)

    def test_modified_following_crosses_month_goes_back(self):
        # 2026-01-30 (Fri) + 1 = Sat 2026-01-31 → following = Mon 2026-02-02 (crosses to Feb)
        # → modified_following goes back to Fri 2026-01-30
        result = add_days_convention(date(2026, 1, 30), 1, convention="modified_following")
        assert result.month == 1
        assert result.year == 2026

    def test_holiday_skipped(self):
        # 2025-01-01 is a holiday; +0 following → next business day
        result = add_days_convention(
            date(2025, 1, 1),
            0,
            convention="following",
            holidays=[date(2025, 1, 1)],
        )
        assert result > date(2025, 1, 1)
        assert result.weekday() not in (5, 6)

    def test_invalid_convention_raises(self):
        with pytest.raises(ValueError, match="Unsupported convention"):
            add_days_convention(date(2025, 1, 6), 1, convention="bogus")
