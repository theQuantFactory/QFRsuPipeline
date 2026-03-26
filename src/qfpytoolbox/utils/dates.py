"""Date utilities — mirrors ``MyJuliaToolbox.jl/src/utils/dates.jl``.

All ``yyyymmdd`` integers use the convention: ``year * 10000 + month * 100 + day``.
"""

from __future__ import annotations

from datetime import date, timedelta

__all__ = [
    "date2int",
    "int2date",
    "days_between",
    "yearfrac",
    "add_days_convention",
]


# ---------------------------------------------------------------------------
# date2int / int2date
# ---------------------------------------------------------------------------


def date2int(d: date) -> int:
    """Convert a :class:`datetime.date` to an integer ``yyyymmdd``.

    Examples
    --------
    >>> date2int(date(2025, 12, 31))
    20251231
    """
    return d.year * 10_000 + d.month * 100 + d.day


def int2date(i: int) -> date:
    """Convert an integer ``yyyymmdd`` to a :class:`datetime.date`.

    Examples
    --------
    >>> int2date(20251231)
    datetime.date(2025, 12, 31)
    """
    y, rem = divmod(i, 10_000)
    m, day = divmod(rem, 100)
    return date(y, m, day)


# ---------------------------------------------------------------------------
# days_between
# ---------------------------------------------------------------------------


def days_between(i1: int, i2: int) -> int:
    """Return the signed number of calendar days from ``i1`` to ``i2``.

    Parameters
    ----------
    i1, i2:
        Dates as ``yyyymmdd`` integers.

    Examples
    --------
    >>> days_between(20250101, 20250103)
    2
    >>> days_between(20250103, 20250101)
    -2
    """
    return (int2date(i2) - int2date(i1)).days


# ---------------------------------------------------------------------------
# yearfrac
# ---------------------------------------------------------------------------


def yearfrac(
    i1: int,
    i2: int,
    *,
    basis: str = "actual365",
) -> float:
    """Compute the year fraction between two ``yyyymmdd`` integer dates.

    Parameters
    ----------
    i1, i2:
        Start and end dates as ``yyyymmdd`` integers.
    basis:
        Day-count convention.  Supported values:

        - ``"actual365"`` (default) — days / 365
        - ``"actual365_25"`` — days / 365.25
        - ``"actual360"`` — days / 360
        - ``"actual"`` — days / 365.2425 (Gregorian mean)
        - ``"thirty360"`` / ``"30/360"`` / ``"30_360"`` — US NASD 30/360
        - ``"thirty360_eu"`` / ``"30E/360"`` — European 30/360
        - ``"thirty360_isda"`` — ISDA 30/360 (approximated as US)

    Returns
    -------
    float
        Signed year fraction (positive if ``i2 > i1``).
    """
    d1 = int2date(i1)
    d2 = int2date(i2)
    days = (d2 - d1).days

    if basis == "actual365":
        return days / 365
    if basis == "actual365_25":
        return days / 365.25
    if basis == "actual360":
        return days / 360
    if basis == "actual":
        return days / 365.2425
    if basis in ("thirty360", "30/360", "30_360"):
        return _days_30_360_us(d1, d2) / 360
    if basis in ("thirty360_eu", "30E/360"):
        return _days_30_360_eu(d1, d2) / 360
    if basis == "thirty360_isda":
        return _days_30_360_us(d1, d2) / 360  # ISDA approximated as US
    raise ValueError(f"Unknown basis: {basis!r}")


def _days_30_360_us(d1: date, d2: date) -> int:
    """30/360 US (NASD) convention."""
    y1, m1, day1 = d1.year, d1.month, d1.day
    y2, m2, day2 = d2.year, d2.month, d2.day
    d1p = 30 if day1 == 31 else day1
    d2p = 30 if (day2 == 31 and d1p == 30) else min(day2, 30)
    return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2p - d1p)


def _days_30_360_eu(d1: date, d2: date) -> int:
    """30E/360 European convention — any 31st day becomes 30."""
    y1, m1, day1 = d1.year, d1.month, d1.day
    y2, m2, day2 = d2.year, d2.month, d2.day
    return 360 * (y2 - y1) + 30 * (m2 - m1) + (min(day2, 30) - min(day1, 30))


# ---------------------------------------------------------------------------
# add_days_convention
# ---------------------------------------------------------------------------


def _to_date(x: date | int | str) -> date:
    if isinstance(x, date):
        return x
    if isinstance(x, int):
        return int2date(x)
    if isinstance(x, str):
        try:
            return date.fromisoformat(x)
        except Exception as exc:
            raise ValueError(f"Cannot parse string to date: {x!r}") from exc
    raise TypeError(f"Cannot convert {type(x).__name__!r} to date")


def _is_business_day(
    d: date,
    weekends: list[int],
    holidays: frozenset[date],
) -> bool:
    """Return ``True`` if ``d`` is a business day."""
    return d.weekday() not in weekends and d not in holidays


def _adjust_following(d: date, weekends: list[int], holidays: frozenset[date]) -> date:
    while not _is_business_day(d, weekends, holidays):
        d += timedelta(days=1)
    return d


def _adjust_previous(d: date, weekends: list[int], holidays: frozenset[date]) -> date:
    while not _is_business_day(d, weekends, holidays):
        d -= timedelta(days=1)
    return d


def add_days_convention(
    d: date,
    delta: int,
    *,
    convention: str = "following",
    weekends: list[int] | None = None,
    holidays: list | set[date] | None = None,
) -> date:
    """Add ``delta`` calendar days to ``d`` and adjust for business-day convention.

    Parameters
    ----------
    d:
        Start date.
    delta:
        Number of calendar days to add (may be negative).
    convention:
        Business-day adjustment rule:

        - ``"following"`` — next business day
        - ``"modified_following"`` — following, unless in next month → previous
        - ``"previous"`` / ``"preceding"`` — previous business day
        - ``"modified_previous"`` / ``"modified_preceding"`` — previous, unless
          in prior month → following

    weekends:
        Weekday integers (Python convention: 0=Mon … 6=Sun) treated as
        non-business days.  Defaults to ``[5, 6]`` (Saturday, Sunday).
    holidays:
        Collection of :class:`datetime.date` objects (or ``yyyymmdd`` integers
        or ISO strings) to treat as non-business days.  Also accepts a
        ``pandas.DataFrame`` with a ``ref_dt`` column of business days.

    Returns
    -------
    :class:`datetime.date`
    """
    if weekends is None:
        weekends = [5, 6]  # Saturday=5, Sunday=6 in Python

    # Normalise holidays to frozenset[date]
    hol_set: frozenset[date]
    if holidays is None or (isinstance(holidays, (list, set)) and len(holidays) == 0):
        hol_set = frozenset()
    else:
        # Check for pandas DataFrame with ref_dt column (business-day calendar)
        try:
            import pandas as pd  # noqa: PLC0415

            if isinstance(holidays, pd.DataFrame):
                if "ref_dt" not in holidays.columns:
                    raise ValueError("calendar DataFrame must contain a 'ref_dt' column of dates or yyyymmdd integers")
                # ref_dt lists *business* days — invert: everything not in it is non-business
                # We handle this via a special flag below
                biz_days: frozenset[date] = frozenset(_to_date(v) for v in holidays["ref_dt"])
                unadjusted = d + timedelta(days=delta)
                return _adjust_with_biz_calendar(unadjusted, weekends, biz_days, convention, unadjusted)
        except ImportError:
            pass

        try:
            hol_set = frozenset(_to_date(v) for v in holidays)
        except Exception as exc:
            raise TypeError(
                f"Unsupported holidays type: {type(holidays).__name__!r}. "
                "Expected list/set of date/int/str, or pandas DataFrame with 'ref_dt'."
            ) from exc

    unadjusted = d + timedelta(days=delta)

    if convention == "following":
        return _adjust_following(unadjusted, weekends, hol_set)
    if convention == "modified_following":
        adj = _adjust_following(unadjusted, weekends, hol_set)
        if adj.month != unadjusted.month:
            return _adjust_previous(unadjusted, weekends, hol_set)
        return adj
    if convention in ("previous", "preceding"):
        return _adjust_previous(unadjusted, weekends, hol_set)
    if convention in ("modified_previous", "modified_preceding"):
        adj = _adjust_previous(unadjusted, weekends, hol_set)
        if adj.month != unadjusted.month:
            return _adjust_following(unadjusted, weekends, hol_set)
        return adj
    raise ValueError(
        f"Unsupported convention: {convention!r}. "
        "Supported: 'following', 'modified_following', 'previous', 'modified_previous'"
    )


def _adjust_with_biz_calendar(
    d: date,
    weekends: list[int],
    biz_days: frozenset[date],
    convention: str,
    unadjusted: date,
) -> date:
    """Adjust ``d`` using a set of known business days."""

    def is_biz(dd: date) -> bool:
        return dd.weekday() not in weekends and dd in biz_days

    def fwd(dd: date) -> date:
        while not is_biz(dd):
            dd += timedelta(days=1)
        return dd

    def bwd(dd: date) -> date:
        while not is_biz(dd):
            dd -= timedelta(days=1)
        return dd

    if convention == "following":
        return fwd(d)
    if convention == "modified_following":
        adj = fwd(d)
        return bwd(d) if adj.month != unadjusted.month else adj
    if convention in ("previous", "preceding"):
        return bwd(d)
    if convention in ("modified_previous", "modified_preceding"):
        adj = bwd(d)
        return fwd(d) if adj.month != unadjusted.month else adj
    raise ValueError(f"Unsupported convention: {convention!r}")
