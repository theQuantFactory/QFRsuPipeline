"""Public API boundary tests for qfpytoolbox package."""

from __future__ import annotations

import qfpytoolbox


def test_toolbox_exposes_core_primitives() -> None:
    assert hasattr(qfpytoolbox, "read_dataframe")
    assert hasattr(qfpytoolbox, "write_dataframe")
    assert hasattr(qfpytoolbox, "iDataSet")
    assert hasattr(qfpytoolbox, "iParameters")


def test_toolbox_does_not_export_rsu_app_logic() -> None:
    # RSU/dashboard pipeline logic must live in application repositories
    # (e.g., QFRsuDashboard), not in the generic toolbox package.
    forbidden_symbols = {
        "run_rsu_pipeline",
        "run_csv_etl",
        "build_master_events",
        "build_delta_frame",
        "build_churn_timeline",
        "build_score_timeseries",
        "build_dashboard_cache",
        "load_dashboard_cache",
    }
    for symbol in forbidden_symbols:
        assert not hasattr(qfpytoolbox, symbol), f"Unexpected public symbol: {symbol}"
