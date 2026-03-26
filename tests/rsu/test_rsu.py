from __future__ import annotations

import pandas as pd

from qfpytoolbox.rsu import (
    build_churn_timeline,
    build_master_events,
    build_near_threshold_timeseries,
    run_csv_etl,
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


def test_run_csv_etl_dashboard_profile_omits_heavy_frames(tmp_path):
    menage_path = tmp_path / "menage.csv"
    scores_path = tmp_path / "score.csv"
    amot_path = tmp_path / "amot.csv"
    asd_path = tmp_path / "asd.csv"

    pd.DataFrame(
        {
            "menage_ano": [1, 2],
            "region": ["R1", "R2"],
            "milieu": ["Urbain", "Rural"],
            "genre_cm": ["Homme", "Femme"],
        }
    ).to_csv(menage_path, index=False)

    pd.DataFrame(
        {
            "menage_ano": [1, 1, 2, 2],
            "score_id_ano": [10, 11, 20, 21],
            "type_demande": ["Inscription", "mise à jour du dossier", "Inscription", "mise à jour du dossier"],
            "score_corrige": [None, 9.2, None, 9.6],
            "score_calcule": [9.4, 9.3, 9.8, 9.7],
            "date_calcul": ["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"],
        }
    ).to_csv(scores_path, index=False)

    pd.DataFrame({"menage_ano": [1]}).to_csv(amot_path, index=False)
    pd.DataFrame({"menage_ano": [2]}).to_csv(asd_path, index=False)

    frames = run_csv_etl(
        menage_path=menage_path,
        scores_path=scores_path,
        programme_paths={"AMOT": amot_path, "ASD": asd_path},
        save_snapshots=False,
        snapshot_profile="dashboard",
    )

    assert "master_events" in frames
    assert "raw_scores" not in frames
    assert "reentry_detail" not in frames
    assert "pivot_wide" not in frames
    assert "monthly_beneficiaire_flows" not in frames


def test_run_csv_etl_full_profile_keeps_heavy_frames(tmp_path):
    menage_path = tmp_path / "menage.csv"
    scores_path = tmp_path / "score.csv"
    amot_path = tmp_path / "amot.csv"
    asd_path = tmp_path / "asd.csv"

    pd.DataFrame(
        {
            "menage_ano": [1, 2],
            "region": ["R1", "R2"],
            "milieu": ["Urbain", "Rural"],
            "genre_cm": ["Homme", "Femme"],
        }
    ).to_csv(menage_path, index=False)

    pd.DataFrame(
        {
            "menage_ano": [1, 1, 2, 2],
            "score_id_ano": [10, 11, 20, 21],
            "type_demande": ["Inscription", "mise à jour du dossier", "Inscription", "mise à jour du dossier"],
            "score_corrige": [None, 9.2, None, 9.6],
            "score_calcule": [9.4, 9.3, 9.8, 9.7],
            "date_calcul": ["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"],
        }
    ).to_csv(scores_path, index=False)

    pd.DataFrame({"menage_ano": [1]}).to_csv(amot_path, index=False)
    pd.DataFrame({"menage_ano": [2]}).to_csv(asd_path, index=False)

    frames = run_csv_etl(
        menage_path=menage_path,
        scores_path=scores_path,
        programme_paths={"AMOT": amot_path, "ASD": asd_path},
        save_snapshots=False,
        snapshot_profile="full",
    )

    assert "master_events" in frames
    assert "raw_scores" in frames
    assert "reentry_detail" in frames
    assert "pivot_wide" in frames
