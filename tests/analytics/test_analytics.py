"""Tests for analytics pipeline helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from qfpytoolbox.analytics import (
    build_dashboard_cache,
    load_dashboard_cache,
    load_frames,
    load_parquet_frames,
    run_calculations,
    save_frames_as_parquet,
)


@pytest.fixture
def sample_sources(tmp_path):
    events = pd.DataFrame(
        {
            "menage_ano": [1, 1, 2],
            "score_final": [10.0, 9.5, 8.0],
            "programme": ["AMOT", "AMOT", "ASD"],
        }
    )
    menage = pd.DataFrame({"menage_ano": [1, 2], "region": ["R1", "R2"]})
    events_path = tmp_path / "events.csv"
    menage_path = tmp_path / "menage.csv"
    events.to_csv(events_path, index=False)
    menage.to_csv(menage_path, index=False)
    return {"events": str(events_path), "menage": str(menage_path)}


def test_load_and_save_parquet(sample_sources, tmp_path):
    frames = load_frames(sample_sources)
    assert sorted(frames.keys()) == ["events", "menage"]
    assert len(frames["events"]) == 3

    written = save_frames_as_parquet(frames, tmp_path / "snapshots")
    assert set(written.keys()) == {"events", "menage"}

    loaded = load_parquet_frames(tmp_path / "snapshots")
    pd.testing.assert_frame_equal(loaded["events"], frames["events"])
    pd.testing.assert_frame_equal(loaded["menage"], frames["menage"])


def test_run_calculations(sample_sources):
    frames = load_frames(sample_sources)

    def score_stats(frames_map):
        return (
            frames_map["events"]
            .groupby("programme", as_index=False)["score_final"]
            .mean()
            .rename(columns={"score_final": "score_mean"})
        )

    out = run_calculations(frames, {"score_stats": score_stats})
    assert "score_stats" in out
    assert set(out["score_stats"]["programme"]) == {"AMOT", "ASD"}


def test_build_cache_with_callable_aggs(sample_sources, tmp_path):
    frames = load_frames(sample_sources)

    cache = build_dashboard_cache(
        frames=frames,
        aggregations={
            "n_events": lambda f: len(f["events"]),
            "score_by_prog": lambda f: f["events"].groupby("programme", as_index=False)["score_final"].mean(),
        },
        output_path=tmp_path / "dashboard_cache.pkl",
    )
    assert cache["n_events"] == 3
    assert "score_by_prog" in cache

    loaded = load_dashboard_cache(tmp_path / "dashboard_cache.pkl")
    assert loaded["n_events"] == 3


def test_build_cache_with_sql_agg(sample_sources, tmp_path):
    duckdb = pytest.importorskip("duckdb")
    _ = duckdb  # silence lint in environments where it is available

    frames = load_frames(sample_sources)
    save_frames_as_parquet(frames, tmp_path / "snapshots")

    cache = build_dashboard_cache(
        parquet_dir=tmp_path / "snapshots",
        aggregations={
            "events_count": "SELECT COUNT(*) AS n FROM read_parquet(getvariable('parquet_dir') || '/events.parquet')"
        },
    )
    assert int(cache["events_count"]["n"].iloc[0]) == 3
