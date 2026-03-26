"""Tests for DataFrame comparison utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from qfpytoolbox.utils.dataframe_compare import DataFrameComparisonResult, compare_dataframes


def _make_left():
    return pd.DataFrame({"id": [1, 2, 3], "amount": [10.0, 20.0, 30.0], "code": ["A", "B", "C"]})


def _make_right():
    return pd.DataFrame({"id": [1, 2, 4], "amount": [10.0, 20.5, 40.0], "code": ["A", "X", "D"]})


class TestCompareDataframes:
    def test_equal_dataframes(self):
        df = _make_left()
        result = compare_dataframes(df.copy(), df.copy(), "id", target_columns=["amount", "code"])
        assert result.equal
        assert result.compared_rows == 3
        assert result.failed_checks == 0
        assert len(result.differences) == 0

    def test_numeric_mismatch(self):
        left = pd.DataFrame({"id": [1], "val": [1.0]})
        right = pd.DataFrame({"id": [1], "val": [1.5]})
        result = compare_dataframes(left, right, "id", target_columns=["val"])
        assert not result.equal
        assert result.numeric_mismatches == 1

    def test_numeric_within_tolerance(self):
        left = pd.DataFrame({"id": [1], "val": [1.0]})
        right = pd.DataFrame({"id": [1], "val": [1.05]})
        result = compare_dataframes(left, right, "id", target_columns=["val"], precision=0.1)
        assert result.equal

    def test_value_mismatch(self):
        left = pd.DataFrame({"id": [1], "code": ["A"]})
        right = pd.DataFrame({"id": [1], "code": ["B"]})
        result = compare_dataframes(left, right, "id", target_columns=["code"])
        assert not result.equal
        assert result.value_mismatches == 1

    def test_left_only_row(self):
        left = _make_left()
        right = _make_right()
        result = compare_dataframes(left, right, "id", target_columns=["amount", "code"])
        assert result.left_only_rows == 1  # id=3
        assert result.right_only_rows == 1  # id=4

    def test_missing_mismatch(self):
        left = pd.DataFrame({"id": [1], "val": [float("nan")]})
        right = pd.DataFrame({"id": [1], "val": [1.0]})
        result = compare_dataframes(left, right, "id", target_columns=["val"])
        assert result.missing_mismatches == 1

    def test_different_key_names(self):
        left = pd.DataFrame({"id": [1, 2], "val": [10.0, 20.0]})
        right = pd.DataFrame({"user_id": [1, 2], "val": [10.0, 20.0]})
        result = compare_dataframes(left, right, "id", "user_id", target_columns=["val"])
        assert result.equal
        assert result.compared_rows == 2

    def test_max_differences(self):
        left = pd.DataFrame({"id": range(10), "val": [float(i) for i in range(10)]})
        right = pd.DataFrame({"id": range(10), "val": [float(i) + 1 for i in range(10)]})
        result = compare_dataframes(left, right, "id", target_columns=["val"], max_differences=3)
        assert len(result.differences) <= 3

    def test_check_type(self):
        left = pd.DataFrame({"id": [1], "val": [1]})
        right = pd.DataFrame({"id": [1], "val": [1.0]})
        result = compare_dataframes(left, right, "id", target_columns=["val"], check_type=True)
        # int vs float are different types
        assert result.type_mismatches >= 1

    def test_title_stored(self):
        df = _make_left()
        result = compare_dataframes(df.copy(), df.copy(), "id", target_columns=["amount"], title="reconcile")
        assert result.title == "reconcile"

    def test_infer_target_columns(self):
        left = pd.DataFrame({"id": [1], "a": [1.0], "b": ["x"]})
        right = pd.DataFrame({"id": [1], "a": [1.0], "b": ["x"]})
        result = compare_dataframes(left, right, "id")  # target_columns=None → auto
        assert "a" in result.target_columns
        assert "b" in result.target_columns

    def test_missing_key_raises(self):
        df = _make_left()
        with pytest.raises(ValueError, match="left columns not found"):
            compare_dataframes(df, df.copy(), "nonexistent", target_columns=["amount"])

    def test_precision_per_column_dict(self):
        left = pd.DataFrame({"id": [1], "a": [1.0], "b": [10.0]})
        right = pd.DataFrame({"id": [1], "a": [1.09], "b": [10.5]})
        result = compare_dataframes(
            left, right, "id", target_columns=["a", "b"], precision={"a": 0.1, "b": 0.1}
        )
        # a within 0.1, b not within 0.1
        assert result.numeric_mismatches >= 1
        assert not result.equal

    def test_result_is_dataclass(self):
        df = _make_left()
        result = compare_dataframes(df.copy(), df.copy(), "id", target_columns=["amount"])
        assert isinstance(result, DataFrameComparisonResult)
        assert isinstance(result.differences, pd.DataFrame)
