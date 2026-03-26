from __future__ import annotations

import pandas as pd

from qfpytoolbox.rsu import (
    build_churn_timeline,
    build_master_events,
    build_near_threshold_timeseries,
)


def _sample_frames() -> dict[str, pd.DataFrame]:
    menage = pd.DataFrame({"menage_ano": [1, 2], "region": ["R1", "R2"]})
    scores = pd.DataFrame(
        {
            "menage_ano": [1, 1, 2, 2],
            "date_calcul": ["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"],
            "score_final": [9.40, 9.20, 9.80, 9.60],
        }
    )
    programmes = pd.DataFrame({"menage_ano": [1, 2], "programme": ["AMOT", "ASD"]})
    return {"menage": menage, "scores": scores, "programmes": programmes}


def test_master_events_has_threshold_fields():
    f = _sample_frames()
    master = build_master_events(f["menage"], f["scores"], f["programmes"])
    assert "dist_threshold" in master.columns
    assert "near_0.10" in master.columns
    assert "eligible" in master.columns


def test_churn_timeline_present():
    f = _sample_frames()
    master = build_master_events(f["menage"], f["scores"], f["programmes"])
    churn = build_churn_timeline(master)
    assert not churn.empty
    assert {"churn_rate", "acquisition_rate", "net_rate"}.issubset(churn.columns)


def test_near_threshold_timeseries_present():
    f = _sample_frames()
    master = build_master_events(f["menage"], f["scores"], f["programmes"])
    near = build_near_threshold_timeseries(master)
    assert not near.empty
    assert {"n_near_010", "n_near_025", "n_near_050"}.issubset(near.columns)
