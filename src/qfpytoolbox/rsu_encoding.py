"""Encoding detection and mojibake repair helpers for RSU flat files."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

MOJIBAKE_MAP: dict[str, str] = {
    "Rabat-SalÃ©-KÃ©nitra": "Rabat-Salé-Kénitra",
    "Tanger-TÃ©touan-Al HoceÃ¯ma": "Tanger-Tétouan-Al Hoceïma",
    "FÃ¨s-MeknÃ¨s": "Fès-Meknès",
    "BÃ©ni Mellal-KhÃ©nifra": "Béni Mellal-Khénifra",
    "DrÃ¢a-Tafilalet": "Drâa-Tafilalet",
    "LaÃ¢youne-Sakia El Hamra": "Laâyoune-Sakia El Hamra",
    "MariÃ©(e)": "Marié(e)",
    "CÃ©libataire": "Célibataire",
    "DivorÃ©(e)": "Divorcé(e)",
    "SÃ©parÃ©(e)": "Séparé(e)",
    "FÃ©minin": "Féminin",
}


def detect_encoding(path: str | Path, n_bytes: int = 50000) -> str:
    """Detect input encoding with chardet if available."""
    p = Path(path)
    try:
        import chardet  # noqa: PLC0415

        raw = p.read_bytes()[:n_bytes]
        d = chardet.detect(raw)
        return d.get("encoding") or "latin-1"
    except ImportError:
        return "latin-1"


def _recover_mojibake(text: str) -> str:
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def repair_dataframe(df: pd.DataFrame, *, columns: list[str] | None = None, verbose: bool = False) -> pd.DataFrame:
    """Repair mojibake corruption in object columns."""
    out = df.copy()
    target_cols = columns
    if target_cols is None:
        target_cols = [c for c in out.columns if out[c].dtype == "object"]

    for col in target_cols:
        s = out[col]
        if s.dtype != "object":
            continue
        before = s.astype(str).str.contains("Ã|â€|Â", na=False, regex=True).sum()
        fixed = s.astype(str).map(_recover_mojibake).replace(MOJIBAKE_MAP, regex=False)
        out[col] = fixed.where(s.notna(), other=None)
        if verbose:
            after = out[col].astype(str).str.contains("Ã|â€|Â", na=False, regex=True).sum()
            log.info("repair %-30s %d -> %d", col, int(before), int(after))
    return out
