"""
Шаг E8: Blending LGBM + CatBoost.

Загружает holdout-prob и test-prob от двух моделей, перебирает alpha
в [0..1] на holdout, выбирает лучшую по F1, пишет финальный сабмит.

Usage:
  python scripts/20_blend.py --lgbm artifacts/baseline_c4 --cat artifacts/baseline_cat_c4
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, confusion_matrix,
    precision_score, recall_score,
)

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "blend"
ART.mkdir(parents=True, exist_ok=True)
LOG_FILE = ART / "run.log"

_lines: list[str] = []
def log(m: str = "") -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {m}" if m else ""
    _lines.append(line)
    LOG_FILE.write_text("\n".join(_lines), encoding="utf-8")


def load_pred(folder: Path, kind: str) -> pl.DataFrame:
    """kind = 'holdout_predictions' or 'predictions'"""
    p = folder / f"{kind}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run baseline with holdout_predictions saving enabled")
    return pl.read_parquet(p)


def best_f1(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float, float]:
    auc = roc_auc_score(y, p)
    precs, recs, thrs = precision_recall_curve(y, p)
    f1s = 2 * precs * recs / np.clip(precs + recs, 1e-12, None)
    bidx = int(np.argmax(f1s[:-1]))
    return auc, float(thrs[bidx]), float(f1s[bidx]), float(np.mean(p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lgbm", required=True, help="folder with LGBM predictions")
    ap.add_argument("--cat", required=True, help="folder with CatBoost predictions")
    ap.add_argument("--name", default="blend", help="output subfolder name")
    args = ap.parse_args()

    t0 = time.time()
    lgbm_dir = Path(args.lgbm)
    cat_dir = Path(args.cat)
    out_dir = ART / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    global LOG_FILE
    LOG_FILE = out_dir / "run.log"
    _lines.clear()

    log(f"Blend: lgbm={lgbm_dir.name}, cat={cat_dir.name}")

    log("=" * 60)
    log("Step 1/3: load holdout predictions")
    h_l = load_pred(lgbm_dir, "holdout_predictions").to_pandas()
    h_c = load_pred(cat_dir, "holdout_predictions").to_pandas()

    keys = ["deliveryorder_id", "item_id", "order_create_date"]
    h = h_l[keys + ["prob_return", "is_return"]].rename(columns={"prob_return": "p_lgbm"})
    h = h.merge(
        h_c[keys + ["prob_return"]].rename(columns={"prob_return": "p_cat"}),
        on=keys, how="inner"
    )
    log(f"  holdout joined: {len(h):,} rows (lgbm={len(h_l):,}, cat={len(h_c):,})")
    y = h["is_return"].astype(np.int8).values

    log("=" * 60)
    log("Step 2/3: search alpha on holdout")
    a_lgbm = best_f1(y, h["p_lgbm"].values)
    a_cat = best_f1(y, h["p_cat"].values)
    log(f"  LGBM alone:    AUC={a_lgbm[0]:.5f}  F1={a_lgbm[2]:.5f} @ thr {a_lgbm[1]:.4f}")
    log(f"  CatBoost alone: AUC={a_cat[0]:.5f}  F1={a_cat[2]:.5f} @ thr {a_cat[1]:.4f}")

    best = (-1.0, -1.0, -1.0, -1.0)  # alpha, auc, f1, thr
    for alpha in np.arange(0.0, 1.01, 0.05):
        p = alpha * h["p_lgbm"].values + (1 - alpha) * h["p_cat"].values
        auc, thr, f1, _ = best_f1(y, p)
        log(f"  alpha={alpha:.2f}  AUC={auc:.5f}  F1={f1:.5f} @ thr {thr:.4f}")
        if f1 > best[2]:
            best = (alpha, auc, f1, thr)

    a_best, auc_best, f1_best, thr_best = best
    log(f"  → best alpha={a_best:.2f}  AUC={auc_best:.5f}  F1={f1_best:.5f} @ thr {thr_best:.4f}")

    p_blend = a_best * h["p_lgbm"].values + (1 - a_best) * h["p_cat"].values
    pred = (p_blend >= thr_best).astype(np.int8)
    p_at = precision_score(y, pred)
    r_at = recall_score(y, pred)
    cm = confusion_matrix(y, pred)
    log(f"  precision: {p_at:.5f}, recall: {r_at:.5f}")
    log(f"  CM: TN={cm[0,0]:,} FP={cm[0,1]:,} FN={cm[1,0]:,} TP={cm[1,1]:,}")

    log("=" * 60)
    log("Step 3/3: blend test predictions")
    t_l = load_pred(lgbm_dir, "predictions").to_pandas()
    t_c = load_pred(cat_dir, "predictions").to_pandas()
    t = t_l[keys + ["prob_return"]].rename(columns={"prob_return": "p_lgbm"})
    t = t.merge(
        t_c[keys + ["prob_return"]].rename(columns={"prob_return": "p_cat"}),
        on=keys, how="inner"
    )
    log(f"  test joined: {len(t):,} rows (lgbm={len(t_l):,}, cat={len(t_c):,})")

    p_test = a_best * t["p_lgbm"].values + (1 - a_best) * t["p_cat"].values
    pred_test = (p_test >= thr_best).astype(np.int8)
    log(f"  test pos: {pred_test.sum():,} ({pred_test.mean():.4%})")

    out_pred = pl.from_pandas(pd.DataFrame({
        "deliveryorder_id": t["deliveryorder_id"].astype(str).to_numpy(),
        "item_id":          t["item_id"].astype(str).to_numpy(),
        "order_create_date": pd.to_datetime(t["order_create_date"]).dt.date.to_numpy(),
        "prob_return":      p_test,
        "is_return":        pred_test.astype(bool),
    }))
    out_pred.write_parquet(out_dir / "predictions.parquet", compression="zstd")
    log(f"  → {out_dir / 'predictions.parquet'}")

    metrics = (
        f"alpha:     {a_best:.2f} (LGBM weight)\n"
        f"ROC-AUC:   {auc_best:.5f}\n"
        f"best F1:   {f1_best:.5f} @ threshold {thr_best:.4f}\n"
        f"precision: {p_at:.5f}\n"
        f"recall:    {r_at:.5f}\n"
        f"vs LGBM alone: AUC {a_lgbm[0]:.5f} F1 {a_lgbm[2]:.5f}\n"
        f"vs CatB alone: AUC {a_cat[0]:.5f} F1 {a_cat[2]:.5f}\n"
        f"test_pos_rate: {pred_test.mean():.4%}\n"
    )
    (out_dir / "metrics.txt").write_text(metrics, encoding="utf-8")
    log(f"  → {out_dir / 'metrics.txt'}")

    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        main()
    except BaseException as exc:
        import traceback
        crash = ART / "crash.log"
        crash.write_text(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", encoding="utf-8")
        raise
