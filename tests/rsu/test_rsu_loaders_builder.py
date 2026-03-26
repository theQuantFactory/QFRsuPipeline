from __future__ import annotations

import pandas as pd

from qfpytoolbox.rsu_builder import (
    build_master_events,
    build_monthly_eligibility_flows,
    build_reentry_analysis,
)
from qfpytoolbox.rsu_loaders import load_scores


def test_load_scores_creates_score_final(tmp_path):
    p = tmp_path / "scores.csv"
    pd.DataFrame(
        {
            "menage_ano": [1, 1],
            "score_id_ano": [10, 11],
            "type_demande": ["Inscription", "mise à jour du dossier"],
            "score_corrige": [None, 9.5],
            "score_calcule": [10.0, 9.6],
            "date_calcul": ["2024-01-01", "2024-02-01"],
        }
    ).to_csv(p, index=False)
    df = load_scores(p, chunk_size=10)
    assert "score_final" in df.columns
    assert len(df) == 2


def test_build_monthly_eligibility_flows_not_empty():
    menage = pd.DataFrame({"menage_ano": [1]})
    scores = pd.DataFrame(
        {
            "menage_ano": [1, 1],
            "score_final": [9.5, 10.2],
            "date_calcul": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        }
    )
    programmes = pd.DataFrame({"menage_ano": [1], "programme": ["AMOT"]})
    master = build_master_events(menage, scores, programmes)
    out = build_monthly_eligibility_flows(master)
    assert not out.empty
    assert "n_transitions" in out.columns


def test_build_reentry_analysis_is_per_program_even_same_date():
    master = pd.DataFrame(
        {
            "menage_ano": [1, 1, 1, 1, 1, 1, 1],
            "programme": ["AMOT", "AMOT", "AMOT", "AMOT", "AMOT", "ASD", "ASD"],
            "date_calcul": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-04-01",
                    "2024-05-01",
                    "2024-03-01",  # same date as AMOT event
                    "2024-04-01",
                ]
            ),
            # AMOT: True -> False -> True -> False -> True => 2 re-entries
            # ASD:  False -> True => 1 re-entry
            "eligible": [True, False, True, False, True, False, True],
        }
    )

    detail, summary = build_reentry_analysis(master)

    amot = detail[(detail["menage_ano"] == 1) & (detail["programme"] == "AMOT")].iloc[0]
    asd = detail[(detail["menage_ano"] == 1) & (detail["programme"] == "ASD")].iloc[0]
    assert int(amot["n_reentries"]) == 2
    assert int(asd["n_reentries"]) == 1
    assert bool(amot["ever_lost"]) is True
    assert bool(asd["ever_lost"]) is True

    assert not summary.empty
    assert {"programme", "n_reentries", "n_menages", "pct_menages"}.issubset(summary.columns)
