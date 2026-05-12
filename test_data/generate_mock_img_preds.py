#!/usr/bin/env python3
"""Build mock img_pred JSON matching run_simple_mm_model.py structure.

Uses the same b_cpr / m_cpr / GA_days / pregnancy_end as ``test_data/test.csv``
(the local table used when developing with ``configs/test_xgb.yaml``).

Train/test split of IDs: 80/20, random_state=42 (same idea as run_simple_model).

Usage (from repo root)::

    python test_data/generate_mock_img_preds.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "test.csv"
OUT_TRAIN = HERE / "img_pred_train_mock.json"
OUT_TEST = HERE / "img_pred_test_mock.json"


def _ga_value(raw: str) -> int | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _mock_imgs(b_cpr: str) -> list[dict]:
    """1–3 mock studies per patient; pred is deterministic but varies by id."""
    h = sum(ord(c) for c in b_cpr) % 3 + 1
    out: list[dict] = []
    for i in range(h):
        pred = round(0.05 + 0.11 * ((i + h * 7) % 8), 4)
        out.append({"study_date": f"2009-{1 + i:02d}-15", "pred": pred})
    return out


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_cpr: dict[str, dict[str, str]] = {}
    for r in rows:
        by_cpr[r["b_cpr"]] = r

    unique_bcpr = list(dict.fromkeys(r["b_cpr"] for r in rows))
    n = len(unique_bcpr)
    rng = np.random.RandomState(42)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_test = int(np.ceil(n * 0.2))
    n_train = n - n_test
    train_ids = [unique_bcpr[int(i)] for i in indices[:n_train]]
    test_ids = [unique_bcpr[int(i)] for i in indices[n_train:]]

    def build_dict(id_list: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for b_cpr in id_list:
            r = by_cpr[b_cpr]
            out[b_cpr] = {
                "CPR_MOTHER": r.get("m_cpr") or None,
                "GA": _ga_value(r.get("GA_days", "")),
                "BIRTHDAY": (r.get("pregnancy_end") or "").strip() or None,
                "imgs": _mock_imgs(b_cpr),
            }
        return out

    train_obj = build_dict(train_ids)
    test_obj = build_dict(test_ids)

    OUT_TRAIN.write_text(json.dumps(train_obj, indent=2), encoding="utf-8")
    OUT_TEST.write_text(json.dumps(test_obj, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_TRAIN} ({len(train_obj)} patients)")
    print(f"Wrote {OUT_TEST} ({len(test_obj)} patients)")
    print(f"Source IDs: {CSV_PATH}")


if __name__ == "__main__":
    main()
