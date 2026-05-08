"""
Re-score saved CatBoost C3 model on c3_holdout, save holdout_predictions.parquet
для будущего блендинга. Не переобучает модель — только predict_proba.

Запускать ПОСЛЕ окончания тяжёлых тренировок (предсказание тоже ест CPU).
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features"
ART = ROOT / "artifacts" / "baseline_cat"
LOG_FILE = ART / "rescore_holdout.log"

CAT_COLS = [
    "delivery_service", "platform_id", "city", "category_name",
    "microcat_name", "buyer_gender", "dominant_payment_method",
]
ID_COLS = ["deliveryorder_id", "item_id", "order_create_date"]
TARGET = "is_return"

_lines: list[str] = []
def log(m: str = "") -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {m}" if m else ""
    _lines.append(line)
    LOG_FILE.write_text("\n".join(_lines), encoding="utf-8")


def main() -> None:
    t0 = time.time()
    log("Re-score CatBoost C3 holdout")
    model = CatBoostClassifier()
    model.load_model(str(ART / "catboost.cbm"))
    log(f"  loaded {ART/'catboost.cbm'}")

    df = pl.read_parquet(FEAT / "c3_holdout.parquet").to_pandas()
    for c in CAT_COLS:
        df[c] = df[c].astype("string").fillna("__missing__")
    log(f"  holdout: {len(df):,} rows × {df.shape[1]} cols")

    feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET]]
    X = df[feature_cols]
    y = df[TARGET].astype(np.int8).values

    p = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, p)
    log(f"  AUC re-check: {auc:.5f}")

    out = pl.from_pandas(pd.DataFrame({
        "deliveryorder_id": df["deliveryorder_id"].astype(str).to_numpy(),
        "item_id":          df["item_id"].astype(str).to_numpy(),
        "order_create_date": pd.to_datetime(df["order_create_date"]).dt.date.to_numpy(),
        "prob_return":      p,
        "is_return":        y.astype(bool),
    }))
    out.write_parquet(ART / "holdout_predictions.parquet", compression="zstd")
    log(f"  → {ART / 'holdout_predictions.parquet'}")
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        main()
    except BaseException as exc:
        import traceback
        crash = ART / "rescore_crash.log"
        crash.write_text(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", encoding="utf-8")
        raise
