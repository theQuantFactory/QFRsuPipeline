from __future__ import annotations

import pandas as pd

from qfpytoolbox.rsu_builder import build_master_events, build_monthly_eligibility_flows
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
