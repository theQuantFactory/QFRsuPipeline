"""RSU analytical DataFrame builders."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROGRAMME_THRESHOLDS = {"AMOT": 9.3264284, "ASD": 9.743001}

__all__ = [
    "build_master_events",
    "build_menage_timeline",
    "build_delta_frame",
    "build_programme_frame",
    "build_beneficiaire_enriched_events",
    "build_pivot_wide",
    "build_volatility_summary",
    "build_eligibility_churn",
    "build_menage_trajectory",
    "build_score_timeseries",
    "build_monthly_eligibility_flows",
    "build_monthly_beneficiaire_flows",
    "build_churn_timeline",
    "build_reentry_analysis",
    "build_near_threshold_timeseries",
]


def build_master_events(df_menage: pd.DataFrame, df_scores: pd.DataFrame, df_programmes: pd.DataFrame) -> pd.DataFrame:
    df = df_scores.merge(df_menage, on="menage_ano", how="left", suffixes=("", "_menage"))
    if not df_programmes.empty:
        df = df.merge(df_programmes, on="menage_ano", how="left")
        df["programme"] = df["programme"].fillna("NON CLASSIFIE")
    else:
        df["programme"] = "NON CLASSIFIE"
    df["programme"] = df["programme"].astype("string").str.upper()
    df["threshold"] = df["programme"].map(PROGRAMME_THRESHOLDS)
    df["dist_threshold"] = df["score_final"] - df["threshold"]
    df["eligible"] = df["dist_threshold"] < 0
    for b in [0.10, 0.25, 0.50]:
        df[f"near_{b:.2f}"] = df["dist_threshold"].abs() <= b
    sort_cols = [c for c in ["menage_ano", "programme", "date_calcul"] if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df


def build_menage_timeline(df_master: pd.DataFrame, score_col: str = "score_final") -> pd.DataFrame:
    grp_cols = [c for c in ["menage_ano", "programme"] if c in df_master.columns]
    g = df_master.sort_values("date_calcul").groupby(grp_cols, observed=True)
    agg = g.agg(
        n_events=(score_col, "count"),
        score_latest=(score_col, "last"),
        score_first=(score_col, "first"),
        score_best=(score_col, "min"),
        score_worst=(score_col, "max"),
        date_first=("date_calcul", "min"),
        date_last=("date_calcul", "max"),
    ).reset_index()
    if "eligible" in df_master.columns:
        elig = g["eligible"].agg(ever_eligible="any", currently_eligible="last", first_eligible="first").reset_index()
        agg = agg.merge(elig, on=grp_cols, how="left")
    agg["days_active"] = (agg["date_last"] - agg["date_first"]).dt.days
    return agg


def build_delta_frame(df_master: pd.DataFrame, score_col: str = "score_final", group_cols: Any = None) -> pd.DataFrame:
    _ = group_cols
    sort_keys = ["menage_ano", "date_calcul"] + [c for c in ["programme", "score_id_ano"] if c in df_master.columns]
    df = df_master.sort_values(sort_keys).drop_duplicates(["menage_ano", "date_calcul"], keep="first").copy()
    df = df.sort_values(["menage_ano", "date_calcul"]).reset_index(drop=True)
    df["score_avant"] = df.groupby("menage_ano", observed=True)[score_col].shift(1)
    df["score_apres"] = df[score_col]
    df["date_avant"] = df.groupby("menage_ano", observed=True)["date_calcul"].shift(1)
    df["date_apres"] = df["date_calcul"]
    if "eligible" in df.columns:
        df["side_avant"] = df.groupby("menage_ano", observed=True)["eligible"].shift(1).map({True: "eligible", False: "excluded"})
        df["side_apres"] = df["eligible"].map({True: "eligible", False: "excluded"})
    df = df.dropna(subset=["score_avant"]).copy()
    df["delta_ISE"] = df["score_apres"] - df["score_avant"]
    df["abs_delta"] = df["delta_ISE"].abs()
    df["days_between"] = (df["date_apres"] - df["date_avant"]).dt.days
    df["delta_distribue"] = np.where(df["days_between"] > 90, df["delta_ISE"] / df["days_between"], np.nan)
    if {"side_avant", "side_apres"}.issubset(df.columns):
        conds = [
            (df["side_avant"] == "excluded") & (df["side_apres"] == "eligible"),
            (df["side_avant"] == "eligible") & (df["side_apres"] == "excluded"),
        ]
        df["status_change"] = np.select(conds, ["gained", "lost"], default="stable")
    return df.reset_index(drop=True)


def build_programme_frame(df_master: pd.DataFrame, programme: str) -> pd.DataFrame:
    if "programme" not in df_master.columns:
        raise ValueError("df_master has no programme column")
    return df_master[df_master["programme"] == programme.upper()].copy()


def build_beneficiaire_enriched_events(df_master: pd.DataFrame, df_beneficiaire: pd.DataFrame) -> pd.DataFrame:
    if df_beneficiaire.empty:
        return df_master.copy()
    b = df_beneficiaire.copy()
    m = df_master.copy()
    if "date_insert" in b.columns:
        b["date_insert"] = pd.to_datetime(b["date_insert"], errors="coerce")
    if "date_calcul" in m.columns:
        m["date_calcul"] = pd.to_datetime(m["date_calcul"], errors="coerce")
    out = m.merge(b, on="menage_ano", how="left", suffixes=("", "_beneficiaire"))
    if {"date_insert", "date_calcul"}.issubset(out.columns):
        out = out[(out["date_insert"].isna()) | (out["date_insert"] <= out["date_calcul"])].copy()
    rename_map = {"partner_id": "beneficiaire_partner_id", "motif": "beneficiaire_motif", "actif": "beneficiaire_actif", "date_insert": "beneficiaire_date_insert"}
    return out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})


def build_pivot_wide(df_master: pd.DataFrame, score_col: str = "score_final", pivot_col: str = "type_demande", agg_func: str = "last") -> pd.DataFrame:
    group_cols = [c for c in ["menage_ano", "programme"] if c in df_master.columns]
    if pivot_col not in df_master.columns:
        return pd.DataFrame()
    df_sorted = df_master.sort_values(group_cols + ["date_calcul"])
    p = df_sorted.groupby(group_cols + [pivot_col], observed=True)[score_col].agg(agg_func).unstack(pivot_col).reset_index()
    p.columns.name = None
    event_cols = [c for c in p.columns if c not in group_cols]
    return p.rename(columns={c: f"score_{c}" for c in event_cols})


def build_volatility_summary(df_delta: pd.DataFrame, group_cols: list[str], min_events: int = 3) -> pd.DataFrame:
    avail = [c for c in group_cols if c in df_delta.columns]
    if not avail:
        raise ValueError(f"None of {group_cols} found")
    g = df_delta.groupby(avail, observed=True)["delta_ISE"]
    tbl = g.agg(n="count", mean_delta="mean", sigma_delta="std", p25=lambda x: x.quantile(0.25), median="median", p75=lambda x: x.quantile(0.75), p90_abs=lambda x: x.abs().quantile(0.9)).reset_index()
    return tbl[tbl["n"] >= min_events].reset_index(drop=True)


def build_eligibility_churn(df_delta: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    req = {"status_change", "side_avant", "side_apres", "menage_ano"}
    missing = req - set(df_delta.columns)
    if missing:
        raise ValueError(f"df_delta missing required columns: {missing}")
    avail = [c for c in group_cols if c in df_delta.columns]
    if not avail:
        raise ValueError(f"None of {group_cols} found")

    def _stats(grp: pd.DataFrame) -> pd.Series:
        n_entrees = int((grp["status_change"] == "gained").sum())
        n_sorties = int((grp["status_change"] == "lost").sum())
        n_elig_av = int((grp["side_avant"] == "eligible").sum())
        n_elig_ap = int((grp["side_apres"] == "eligible").sum())
        stock = (n_elig_av + n_elig_ap) / 2.0
        churn = (n_entrees + n_sorties) / stock if stock > 0 else np.nan
        return pd.Series({"n_transitions": len(grp), "n_entrees": n_entrees, "n_sorties": n_sorties, "stock_moyen": stock, "churn_eligibilite": churn})

    return df_delta.groupby(avail, observed=True).apply(_stats).reset_index()


def build_menage_trajectory(df_master: pd.DataFrame, score_col: str = "score_final") -> pd.DataFrame:
    df = df_master.sort_values(["menage_ano", "date_calcul"])
    base = df.groupby("menage_ano", observed=True).agg(n_events=(score_col, "count"), score_actuel=(score_col, "last"), score_min=(score_col, "min"), score_max=(score_col, "max"), date_premier=("date_calcul", "min"), date_dernier=("date_calcul", "max")).reset_index()
    if "eligible" in df.columns:
        elig = df.groupby("menage_ano", observed=True)["eligible"].agg(eligible_inscription="first", eligible_actuel="last").reset_index()
        base = base.merge(elig, on="menage_ano", how="left")
    base["jours_dans_systeme"] = (base["date_dernier"] - base["date_premier"]).dt.days
    return base


def build_score_timeseries(df_master: pd.DataFrame, score_col: str = "score_final") -> dict[str, pd.DataFrame]:
    req = {"menage_ano", "date_calcul", score_col}
    if not req.issubset(df_master.columns):
        return {"daily_stats": pd.DataFrame(), "daily_menages": pd.DataFrame()}
    df = df_master[["menage_ano", "date_calcul", score_col]].copy()
    df["date_calcul"] = pd.to_datetime(df["date_calcul"], errors="coerce")
    df["week"] = df["date_calcul"].dt.to_period("W").dt.start_time
    out = df.groupby("week", observed=True)[score_col].agg(score_mean="mean", score_median="median", score_std="std", score_min="min", score_max="max").reset_index().rename(columns={"week": "date_calcul"})
    n = df.groupby("week", observed=True)["menage_ano"].nunique().rename("n_menages").reset_index().rename(columns={"week": "date_calcul"})
    out = out.merge(n, on="date_calcul", how="left")
    return {"daily_stats": out.sort_values("date_calcul").reset_index(drop=True), "daily_menages": pd.DataFrame()}


def build_monthly_eligibility_flows(df_master: pd.DataFrame) -> pd.DataFrame:
    if not {"menage_ano", "date_calcul", "eligible"}.issubset(df_master.columns):
        return pd.DataFrame()
    df = df_master[["menage_ano", "date_calcul", "eligible"]].copy()
    df["date_calcul"] = pd.to_datetime(df["date_calcul"], errors="coerce")
    df["year_month"] = df["date_calcul"].dt.to_period("M")
    m = df.sort_values("date_calcul").groupby(["year_month", "menage_ano"], observed=True)["eligible"].last().reset_index()
    m["prev_eligible"] = m.groupby("menage_ano", observed=True)["eligible"].shift(1)
    m["became_eligible"] = (m["eligible"] == True) & (m["prev_eligible"] == False)
    m["became_ineligible"] = (m["eligible"] == False) & (m["prev_eligible"] == True)
    counts = m.groupby(["year_month", "eligible"], observed=True).size().unstack(fill_value=0)
    if True not in counts.columns:
        counts[True] = 0
    if False not in counts.columns:
        counts[False] = 0
    trans = m.groupby("year_month", observed=True)[["became_eligible", "became_ineligible"]].sum()
    out = pd.DataFrame({"date": counts.index.to_timestamp(), "unique_menages_eligible": counts[True].astype(int).values, "unique_menages_not_eligible": counts[False].astype(int).values, "menages_became_eligible": trans["became_eligible"].reindex(counts.index, fill_value=0).astype(int).values, "menages_became_ineligible": trans["became_ineligible"].reindex(counts.index, fill_value=0).astype(int).values})
    out["n_transitions"] = out["menages_became_eligible"] + out["menages_became_ineligible"]
    return out.sort_values("date").reset_index(drop=True)


def build_monthly_beneficiaire_flows(df_beneficiaire: pd.DataFrame) -> pd.DataFrame:
    req = {"date_insert", "actif", "partner_id", "menage_ano"}
    if df_beneficiaire.empty or not req.issubset(df_beneficiaire.columns):
        return pd.DataFrame()
    df = df_beneficiaire[list(req)].copy()
    df["date_insert"] = pd.to_datetime(df["date_insert"], errors="coerce")
    df["year_month"] = df["date_insert"].dt.to_period("M")
    df["actif"] = df["actif"].astype("boolean")
    m = df.sort_values("date_insert").groupby(["year_month", "partner_id", "menage_ano"], observed=True)["actif"].last().reset_index()
    rows: list[dict[str, Any]] = []
    for prog, sub in m.groupby("partner_id", observed=True):
        cur_elig: set[Any] = set()
        cur_not: set[Any] = set()
        for month in sorted(sub["year_month"].unique()):
            sm = sub[sub["year_month"] == month]
            became_e = 0
            became_i = 0
            for hid, actif in zip(sm["menage_ano"], sm["actif"]):
                was = hid in cur_elig
                now = bool(actif)
                if (not was) and now:
                    cur_not.discard(hid)
                    cur_elig.add(hid)
                    became_e += 1
                elif was and (not now):
                    cur_elig.discard(hid)
                    cur_not.add(hid)
                    became_i += 1
                elif (not was) and (not now):
                    cur_not.add(hid)
            rows.append({"date": month.to_timestamp(), "programme": str(prog).upper(), "unique_menages_eligible": len(cur_elig), "unique_menages_not_eligible": len(cur_not), "menages_became_eligible": became_e, "menages_became_ineligible": became_i})
    return pd.DataFrame(rows).sort_values(["programme", "date"]).reset_index(drop=True)


def build_churn_timeline(df_master: pd.DataFrame) -> pd.DataFrame:
    if not {"eligible", "programme", "menage_ano", "date_calcul"}.issubset(df_master.columns):
        return pd.DataFrame()
    df = df_master[["menage_ano", "programme", "date_calcul", "eligible"]].copy()
    df["date_calcul"] = pd.to_datetime(df["date_calcul"], errors="coerce")
    df["year_month"] = df["date_calcul"].dt.to_period("M")
    monthly = (
        df.sort_values(["menage_ano", "programme", "date_calcul"])
        .groupby(["programme", "year_month", "menage_ano"], observed=True)["eligible"]
        .last()
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for prog, sub in monthly.groupby("programme", observed=True):
        prev: dict[Any, bool] = {}
        for month in sorted(sub["year_month"].unique()):
            sm = sub[sub["year_month"] == month]
            now = dict(zip(sm["menage_ano"], sm["eligible"]))
            pool_start = sum(1 for v in prev.values() if v)
            entries = 0
            exits = 0
            for hid, is_elig in now.items():
                prev_val = prev.get(hid)
                if prev_val is None:
                    pass
                elif (not prev_val) and bool(is_elig):
                    entries += 1
                elif bool(prev_val) and (not bool(is_elig)):
                    exits += 1
                prev[hid] = bool(is_elig)
            pool_end = sum(1 for v in prev.values() if v)
            denom = pool_start if pool_start > 0 else np.nan
            rows.append({"date": month.to_timestamp(), "programme": prog, "pool_start": pool_start, "pool_end": pool_end, "entries": entries, "exits": exits, "churn_rate": exits / denom, "acquisition_rate": entries / denom, "net_rate": (entries - exits) / denom})
    return pd.DataFrame(rows).sort_values(["programme", "date"]).reset_index(drop=True)


def build_reentry_analysis(df_master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not {"eligible", "programme", "menage_ano", "date_calcul"}.issubset(df_master.columns):
        return pd.DataFrame(), pd.DataFrame()
    df = df_master[["menage_ano", "date_calcul", "eligible", "programme"]].copy()
    df["date_calcul"] = pd.to_datetime(df["date_calcul"], errors="coerce")
    df = df.sort_values(["menage_ano", "programme", "date_calcul"])
    rows = []
    for (hid, prog), grp in df.groupby(["menage_ano", "programme"], observed=True):
        statuses = grp["eligible"].astype(bool).tolist()
        reentries = 0
        lost_seen = False
        for s in statuses:
            if not s:
                lost_seen = True
            elif lost_seen:
                reentries += 1
                lost_seen = False
        rows.append({"menage_ano": hid, "programme": prog, "n_reentries": reentries, "ever_lost": any(not s for s in statuses)})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = detail.groupby(["programme", "n_reentries"], observed=True)["menage_ano"].nunique().rename("n_menages").reset_index()
    total = detail.groupby("programme", observed=True)["menage_ano"].nunique().rename("total")
    summary = summary.merge(total, on="programme", how="left")
    summary["pct_menages"] = (summary["n_menages"] / summary["total"] * 100).round(2)
    summary["n_reentries"] = summary["n_reentries"].astype(str)
    return detail, summary.drop(columns=["total"])


def build_near_threshold_timeseries(df_master: pd.DataFrame, score_col: str = "score_final", bands: tuple = (0.10, 0.25, 0.50)) -> pd.DataFrame:
    req = {"menage_ano", "date_calcul", score_col, "programme", "dist_threshold"}
    if not req.issubset(df_master.columns):
        return pd.DataFrame()
    df = df_master[["menage_ano", "date_calcul", score_col, "programme", "dist_threshold"]].copy()
    df["date_calcul"] = pd.to_datetime(df["date_calcul"], errors="coerce")
    df["week"] = df["date_calcul"].dt.to_period("W").dt.start_time
    df = df.sort_values(["week", "menage_ano", "programme", "date_calcul"]).drop_duplicates(["week", "menage_ano", "programme"], keep="last")
    rows: list[dict[str, Any]] = []
    for (week, prog), grp in df.groupby(["week", "programme"], observed=True):
        row: dict[str, Any] = {"week": week, "programme": prog, "n_total": int(grp["menage_ano"].nunique())}
        for b in bands:
            suffix = f"{int(b*100):03d}"
            d = grp["dist_threshold"]
            s = grp[score_col]
            mask_net = d.abs() <= b
            mask_neg = (d >= -b) & (d < 0)
            mask_pos = (d > 0) & (d <= b)
            row[f"n_near_{suffix}"] = int(mask_net.sum())
            row[f"mean_score_{suffix}"] = float(s[mask_net].mean()) if mask_net.any() else np.nan
            row[f"median_score_{suffix}"] = float(s[mask_net].median()) if mask_net.any() else np.nan
            row[f"n_near_{suffix}_neg"] = int(mask_neg.sum())
            row[f"n_near_{suffix}_pos"] = int(mask_pos.sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["week", "programme"]).reset_index(drop=True)

