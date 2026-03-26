"""DataFrame comparison — mirrors ``MyJuliaToolbox.jl/src/utils/dataframe_compare.jl``.

Compare two DataFrames by key column(s), report differences, and control
numeric tolerance per column.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DataFrameComparisonResult",
    "compare_dataframes",
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class DataFrameComparisonResult:
    """Returned by :func:`compare_dataframes`.

    Attributes
    ----------
    title:
        Label passed via ``title=`` (empty string when omitted).
    equal:
        ``True`` when no differences were found.
    compared_rows:
        Rows present in **both** DataFrames (matched by key).
    target_columns:
        Ordered list of columns that were compared.
    total_checks:
        Total cell-level checks performed.
    passed_checks:
        Cells considered equal.
    failed_checks:
        Cells that differed.  Always equals ``len(differences)``.
    left_only_rows:
        Rows present in ``left`` but absent in ``right``.
    right_only_rows:
        Rows present in ``right`` but absent in ``left``.
    missing_mismatches:
        Cells where exactly one side is ``NaN`` / ``None``.
    type_mismatches:
        Cells where ``type(left) != type(right)`` when ``check_type=True``.
    numeric_mismatches:
        Numeric cells that differ beyond the tolerance.
    value_mismatches:
        Non-numeric cells that are not equal.
    differences:
        ``pandas.DataFrame`` with columns ``issue``, ``column``, ``key``,
        ``left_value``, ``right_value``, ``detail`` — one row per failed check.
    """

    title: str
    equal: bool
    compared_rows: int
    target_columns: list[str]
    total_checks: int
    passed_checks: int
    failed_checks: int
    left_only_rows: int
    right_only_rows: int
    missing_mismatches: int
    type_mismatches: int
    numeric_mismatches: int
    value_mismatches: int
    differences: Any  # pandas.DataFrame


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_dataframes(
    left: Any,
    right: Any,
    left_key: str | list[str],
    right_key: str | list[str] | None = None,
    target_columns: list[str] | None = None,
    *,
    precision: float | dict[str, float] = 0.0,
    check_type: bool = False,
    title: str = "",
    max_differences: int | None = None,
    logger: Any = None,
    color: bool = True,
) -> DataFrameComparisonResult:
    """Compare two DataFrames and return a structured result.

    Parameters
    ----------
    left, right:
        ``pandas.DataFrame`` objects to compare.
    left_key:
        Column(s) in ``left`` used as the join key.
    right_key:
        Column(s) in ``right`` used as the join key.  Defaults to ``left_key``.
    target_columns:
        Columns to compare.  When ``None`` all columns present in both DataFrames
        (excluding keys) are used.
    precision:
        Numeric tolerance — either a single float applied to all numeric columns
        or a ``dict`` mapping column name → tolerance.  Default ``0.0``.
    check_type:
        When ``True`` a value mismatch is also reported if the Python types differ.
    title:
        Label added to log output (does not affect the result).
    max_differences:
        Stop collecting after this many differences.
    logger:
        Optional :class:`~qfpytoolbox.io.logger.AsyncLogger` to emit the report to.
    color:
        Whether to use ANSI colours in the summary (default ``True``).
    """
    import pandas as pd  # noqa: PLC0415

    # -- normalise keys -------------------------------------------------------
    lkeys = _norm_keys(left_key)
    rkeys = _norm_keys(right_key) if right_key is not None else lkeys

    if len(lkeys) != len(rkeys):
        raise ValueError("left_key and right_key must have the same number of columns")

    # -- validate columns exist -----------------------------------------------
    _ensure_cols(left, lkeys, "left")
    _ensure_cols(right, rkeys, "right")

    # -- resolve target columns -----------------------------------------------
    if target_columns is None:
        right_cols = set(right.columns)
        excl = set(lkeys) | set(rkeys)
        targets = [c for c in left.columns if c in right_cols and c not in excl]
        if not targets:
            raise ValueError("Could not infer target columns. Provide target_columns explicitly.")
    else:
        targets = list(target_columns)
        _ensure_cols(left, targets, "left")
        _ensure_cols(right, targets, "right")

    # -- build occurrence index -----------------------------------------------
    left2 = left.copy()
    right2 = right.copy()
    left2["__occ__"] = _occurrence_index(left2, lkeys)
    right2["__occ__"] = _occurrence_index(right2, rkeys)

    # -- align right keys to left key names -----------------------------------
    rename = {rk: lk for lk, rk in zip(lkeys, rkeys) if lk != rk}
    if rename:
        right2 = right2.rename(columns=rename)
    all_rkeys_as_lkeys = lkeys  # after rename

    # -- merge ----------------------------------------------------------------
    merge_keys = lkeys + ["__occ__"]
    merged = pd.merge(
        left2[lkeys + ["__occ__"] + targets].rename(columns={c: f"{c}__left" for c in targets}),
        right2[all_rkeys_as_lkeys + ["__occ__"] + targets].rename(columns={c: f"{c}__right" for c in targets}),
        on=merge_keys,
        how="outer",
        indicator=True,
    )

    left_only_mask = merged["_merge"] == "left_only"
    right_only_mask = merged["_merge"] == "right_only"
    both_mask = merged["_merge"] == "both"

    left_only_rows = int(left_only_mask.sum())
    right_only_rows = int(right_only_mask.sum())
    compared_rows = int(both_mask.sum())

    # -- collect differences --------------------------------------------------
    diff_rows: list[dict[str, Any]] = []
    total_checks = left_only_rows + right_only_rows
    passed_checks = 0
    missing_mm = 0
    type_mm = 0
    numeric_mm = 0
    value_mm = 0

    # left-only / right-only rows
    for _, row in merged[left_only_mask].iterrows():
        key_txt = _key_text(row, lkeys)
        diff_rows.append({"issue": "left_only", "column": "", "key": key_txt, "left_value": "", "right_value": "", "detail": ""})
        if max_differences is not None and len(diff_rows) >= max_differences:
            break
    for _, row in merged[right_only_mask].iterrows():
        key_txt = _key_text(row, lkeys)
        diff_rows.append({"issue": "right_only", "column": "", "key": key_txt, "left_value": "", "right_value": "", "detail": ""})
        if max_differences is not None and len(diff_rows) >= max_differences:
            break

    # cell-level checks for matched rows
    matched = merged[both_mask]
    total_checks += len(matched) * len(targets)

    for col in targets:
        tol = _precision_for(precision, col)
        lcol = f"{col}__left"
        rcol = f"{col}__right"
        for _, row in matched.iterrows():
            lv = row[lcol]
            rv = row[rcol]
            key_txt = _key_text(row, lkeys)

            lv_is_na = _is_na(lv)
            rv_is_na = _is_na(rv)

            if lv_is_na and rv_is_na:
                passed_checks += 1
                continue

            if lv_is_na != rv_is_na:
                missing_mm += 1
                diff_rows.append({
                    "issue": "missing_mismatch",
                    "column": col,
                    "key": key_txt,
                    "left_value": _vt(lv),
                    "right_value": _vt(rv),
                    "detail": "",
                })
                if max_differences is not None and len(diff_rows) >= max_differences:
                    break
                continue

            # type check
            if check_type and type(lv) is not type(rv):
                abs_diff: str | None = None
                if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
                    abs_diff = str(abs(float(lv) - float(rv)))
                type_mm += 1
                diff_rows.append({
                    "issue": "type_mismatch",
                    "column": col,
                    "key": key_txt,
                    "left_value": _vt(lv),
                    "right_value": _vt(rv),
                    "detail": abs_diff or "",
                })
                if max_differences is not None and len(diff_rows) >= max_differences:
                    break
                continue

            # numeric comparison
            if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
                if _nums_equal(lv, rv, tol):
                    passed_checks += 1
                else:
                    numeric_mm += 1
                    diff_rows.append({
                        "issue": "numeric_mismatch",
                        "column": col,
                        "key": key_txt,
                        "left_value": _vt(lv),
                        "right_value": _vt(rv),
                        "detail": str(abs(float(lv) - float(rv))),
                    })
                    if max_differences is not None and len(diff_rows) >= max_differences:
                        break
                continue

            # value comparison
            if lv == rv or (lv is rv):
                passed_checks += 1
            else:
                value_mm += 1
                diff_rows.append({
                    "issue": "value_mismatch",
                    "column": col,
                    "key": key_txt,
                    "left_value": _vt(lv),
                    "right_value": _vt(rv),
                    "detail": "",
                })
                if max_differences is not None and len(diff_rows) >= max_differences:
                    break

    failed_checks = len(diff_rows)
    # recalculate passed_checks as (total - failed - not-counted-above)
    passed_checks = total_checks - failed_checks

    diff_df = pd.DataFrame(diff_rows, columns=["issue", "column", "key", "left_value", "right_value", "detail"])

    result = DataFrameComparisonResult(
        title=title,
        equal=failed_checks == 0,
        compared_rows=compared_rows,
        target_columns=targets,
        total_checks=total_checks,
        passed_checks=max(0, passed_checks),
        failed_checks=failed_checks,
        left_only_rows=left_only_rows,
        right_only_rows=right_only_rows,
        missing_mismatches=missing_mm,
        type_mismatches=type_mm,
        numeric_mismatches=numeric_mm,
        value_mismatches=value_mm,
        differences=diff_df,
    )

    if logger is not None:
        _emit_report(logger, result, color=color)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm_keys(key: str | list[str]) -> list[str]:
    if isinstance(key, str):
        return [key]
    keys = list(key)
    if not keys:
        raise ValueError("key cannot be empty")
    return keys


def _ensure_cols(df: Any, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} columns not found: {missing!r}")


def _occurrence_index(df: Any, keys: list[str]) -> list[int]:
    """Return a 1-based occurrence counter for duplicate keys."""
    counts: dict[tuple, int] = {}
    result = []
    for _, row in df[keys].iterrows():
        k = tuple(row)
        counts[k] = counts.get(k, 0) + 1
        result.append(counts[k])
    return result


def _precision_for(precision: float | dict[str, float], col: str) -> float:
    if isinstance(precision, dict):
        val = precision.get(col, 0.0)
        if not isinstance(val, (int, float)):
            raise TypeError(f"precision for column {col!r} must be numeric")
        if val < 0:
            raise ValueError(f"precision for column {col!r} must be >= 0")
        return float(val)
    if isinstance(precision, (int, float)):
        if precision < 0:
            raise ValueError("precision must be >= 0")
        return float(precision)
    raise TypeError("precision must be a numeric value or dict")


def _is_na(v: Any) -> bool:
    if v is None:
        return True
    try:
        return math.isnan(float(v)) if isinstance(v, (int, float)) else False
    except (TypeError, ValueError):
        return False


def _nums_equal(a: Any, b: Any, tol: float) -> bool:
    fa, fb = float(a), float(b)
    if math.isnan(fa) and math.isnan(fb):
        return True
    if math.isnan(fa) or math.isnan(fb):
        return False
    if not math.isfinite(fa) or not math.isfinite(fb):
        return fa == fb
    return abs(fa - fb) <= tol


def _vt(v: Any) -> str:
    if _is_na(v):
        return "missing"
    return repr(v)


def _key_text(row: Any, keys: list[str]) -> str:
    parts = [f"{k}={row[k]!r}" for k in keys]
    return ", ".join(parts)


def _emit_report(logger: Any, result: DataFrameComparisonResult, *, color: bool) -> None:
    from qfpytoolbox.io.logger import log_info, log_warn  # noqa: PLC0415

    GREEN = "\033[32m" if color else ""
    RED = "\033[31m" if color else ""
    RESET = "\033[0m" if color else ""

    prefix = f"[{result.title}] " if result.title.strip() else ""
    status = f"{GREEN}passed{RESET}" if result.equal else f"{RED}failed{RESET}"
    msg = (
        f"{prefix}compare_dataframes {status}: "
        f"compared_rows={result.compared_rows}, "
        f"failed_checks={result.failed_checks}/{result.total_checks}"
    )
    if result.equal:
        log_info(logger, msg)
    else:
        log_warn(logger, msg)
