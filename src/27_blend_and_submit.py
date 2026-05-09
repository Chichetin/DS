"""
Шаг F6: Blending LGBM full + CatBoost-22, генерация финального сабмита.

Логика:
1. Загружаем holdout-prob от обеих моделей (на c4_holdout_full).
2. Перебираем alpha ∈ [0, 1] шагом 0.05, ищем лучший по F1.
3. Берём порог по F1 на блендженом holdout.
4. Применяем (alpha, threshold) к test-prob → бинаризация → CSV сабмит.

Формат сабмита (по ТЗ Avito):
- columns: item_id, deliveryorder_id, order_create_date, is_return
- is_return: true / false
- 4,030,995 строк (все пары test'а)

Артефакты в artifacts/submission/.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, confusion_matrix,
    precision_score, recall_score,
)

from _log import Logger

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "submission"
ART.mkdir(parents=True, exist_ok=True)

# Логгер пересоздаётся в main() после парсинга --name (run.log в out_dir).
# До этого момента используем default-файл, чтобы любой сбой при парсинге аргов
# тоже попал в лог.
log = Logger(ART / "_default.log")


def best_f1(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    auc = roc_auc_score(y, p)
    precs, recs, thrs = precision_recall_curve(y, p)
    f1s = 2 * precs * recs / np.clip(precs + recs, 1e-12, None)
    bidx = int(np.argmax(f1s[:-1]))
    return auc, float(thrs[bidx]), float(f1s[bidx])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lgbm", required=True, help="папка с LGBM predictions (holdout_predictions.parquet + predictions.parquet)")
    ap.add_argument("--cat", required=True, help="папка с CatBoost predictions")
    ap.add_argument("--name", default="final", help="название сабмита (имя поддиректории)")
    ap.add_argument("--team", default="ИИ в массы", help="название команды для сабмита")
    args = ap.parse_args()

    out_dir = ART / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Переключаемся на «настоящий» лог в out_dir/run.log.
    global log
    log = Logger(out_dir / "run.log")

    log(f"Final submission build. team={args.team!r}")
    log(f"  lgbm dir: {args.lgbm}")
    log(f"  cat dir:  {args.cat}")

    # 1. Load holdout predictions ----------------------------------------------
    log.step("Step 1/4: load holdout predictions")
    h_l = pl.read_parquet(Path(args.lgbm) / "holdout_predictions.parquet").to_pandas()
    h_c = pl.read_parquet(Path(args.cat) / "holdout_predictions.parquet").to_pandas()
    log(f"  lgbm holdout: {len(h_l):,} rows")
    log(f"  cat  holdout: {len(h_c):,} rows")

    keys = ["deliveryorder_id", "item_id", "order_create_date"]
    h = h_l[keys + ["prob_return", "is_return"]].rename(columns={"prob_return": "p_lgbm"})
    h = h.merge(
        h_c[keys + ["prob_return"]].rename(columns={"prob_return": "p_cat"}),
        on=keys, how="inner",
    )
    log(f"  joined: {len(h):,} rows")
    if len(h) < 0.99 * min(len(h_l), len(h_c)):
        log(f"  ⚠️ join потерял заметную долю строк — сверь holdout-файлы у моделей")

    y = h["is_return"].astype(np.int8).values

    # 2. Search alpha ----------------------------------------------------------
    log.step("Step 2/4: alpha sweep (по F1 на holdout)")
    auc_l, thr_l, f1_l = best_f1(y, h["p_lgbm"].values)
    auc_c, thr_c, f1_c = best_f1(y, h["p_cat"].values)
    log(f"  LGBM  alone:    AUC={auc_l:.5f}  F1={f1_l:.5f} @ thr {thr_l:.4f}")
    log(f"  CatBoost alone: AUC={auc_c:.5f}  F1={f1_c:.5f} @ thr {thr_c:.4f}")

    best_alpha, best_auc, best_f1_, best_thr = 0.0, -1.0, -1.0, -1.0
    for alpha in np.arange(0.0, 1.01, 0.05):
        p = alpha * h["p_lgbm"].values + (1 - alpha) * h["p_cat"].values
        auc, thr, f1 = best_f1(y, p)
        marker = ""
        if f1 > best_f1_:
            best_alpha, best_auc, best_f1_, best_thr = float(alpha), auc, f1, thr
            marker = "  ← best F1"
        log(f"  alpha={alpha:.2f}  AUC={auc:.5f}  F1={f1:.5f} @ thr {thr:.4f}{marker}")

    log(f"  → best alpha={best_alpha:.2f}  AUC={best_auc:.5f}  F1={best_f1_:.5f} @ thr {best_thr:.4f}")

    p_blend = best_alpha * h["p_lgbm"].values + (1 - best_alpha) * h["p_cat"].values
    pred = (p_blend >= best_thr).astype(np.int8)
    p_at = precision_score(y, pred)
    r_at = recall_score(y, pred)
    cm = confusion_matrix(y, pred)
    log(f"  precision: {p_at:.5f}, recall: {r_at:.5f}")
    log(f"  CM: TN={cm[0,0]:,} FP={cm[0,1]:,} FN={cm[1,0]:,} TP={cm[1,1]:,}")

    # 3. Apply to test ---------------------------------------------------------
    log.step("Step 3/4: blend test predictions")
    t_l = pl.read_parquet(Path(args.lgbm) / "predictions.parquet").to_pandas()
    t_c = pl.read_parquet(Path(args.cat) / "predictions.parquet").to_pandas()
    log(f"  lgbm test: {len(t_l):,} rows")
    log(f"  cat  test: {len(t_c):,} rows")

    t = t_l[keys + ["prob_return"]].rename(columns={"prob_return": "p_lgbm"})
    t = t.merge(
        t_c[keys + ["prob_return"]].rename(columns={"prob_return": "p_cat"}),
        on=keys, how="inner",
    )
    log(f"  joined test: {len(t):,} rows")

    p_test = best_alpha * t["p_lgbm"].values + (1 - best_alpha) * t["p_cat"].values
    pred_test = (p_test >= best_thr).astype(bool)
    pos_rate = float(pred_test.mean())
    log(f"  test pos: {pred_test.sum():,} ({pos_rate:.4%})")

    # 4. Sanity + write submission ---------------------------------------------
    log.step("Step 4/4: sanity + write submission CSV")

    # Sanity: target rate в холдауте
    holdout_rate = float(y.mean())
    log(f"  holdout target rate:  {holdout_rate:.4%}")
    log(f"  test predicted rate:  {pos_rate:.4%}")
    if abs(pos_rate - holdout_rate) > 0.01:
        log(f"  ⚠️ test_rate откланяется от holdout_rate >1pp — проверь threshold/калибровку")

    # Sanity: NaN
    if np.isnan(p_test).any():
        log(f"  ⚠️ NaN в предсказаниях: {np.isnan(p_test).sum()}")

    # Sanity: уникальные пары
    n_unique = len(set(zip(t["deliveryorder_id"], t["item_id"])))
    log(f"  unique (deliveryorder_id, item_id) pairs: {n_unique:,} (всего строк {len(t):,})")

    submit = pd.DataFrame({
        "item_id":           t["item_id"].astype(str).values,
        "deliveryorder_id":  t["deliveryorder_id"].astype(str).values,
        "order_create_date": pd.to_datetime(t["order_create_date"]).dt.strftime("%Y-%m-%d").values,
        "is_return":         pred_test,
    })
    csv_path = out_dir / "submission.csv"
    submit.to_csv(csv_path, index=False, encoding="utf-8")
    log(f"  → {csv_path}  ({csv_path.stat().st_size/1024**2:.1f} MB)")

    # Также сохраним вероятности (на случай если понадобится откалибровать порог)
    probs_path = out_dir / "submission_probs.parquet"
    pl.from_pandas(pd.DataFrame({
        "item_id":           t["item_id"].astype(str).values,
        "deliveryorder_id":  t["deliveryorder_id"].astype(str).values,
        "order_create_date": pd.to_datetime(t["order_create_date"]).dt.date.to_numpy(),
        "p_lgbm":            t["p_lgbm"].values.astype(np.float32),
        "p_cat":             t["p_cat"].values.astype(np.float32),
        "p_blend":           p_test.astype(np.float32),
        "is_return":         pred_test,
    })).write_parquet(probs_path, compression="zstd")
    log(f"  → {probs_path}")

    metrics = (
        f"team:           {args.team}\n"
        f"alpha (LGBM):   {best_alpha:.2f}\n"
        f"threshold:      {best_thr:.4f}\n"
        f"holdout AUC:    {best_auc:.5f}\n"
        f"holdout F1:     {best_f1_:.5f}\n"
        f"holdout prec:   {p_at:.5f}\n"
        f"holdout recall: {r_at:.5f}\n"
        f"vs LGBM alone:  AUC {auc_l:.5f}  F1 {f1_l:.5f}\n"
        f"vs Cat alone:   AUC {auc_c:.5f}  F1 {f1_c:.5f}\n"
        f"test pos rate:  {pos_rate:.4%}\n"
        f"target rate:    {holdout_rate:.4%}\n"
        f"submission rows: {len(submit):,}\n"
    )
    (out_dir / "metrics.txt").write_text(metrics, encoding="utf-8")
    log(f"  → {out_dir/'metrics.txt'}")

    log.done()
    # stdout-summary специально оставлен — этот скрипт почти всегда запускается
    # в foreground, чтобы оператор сразу увидел итоговые цифры перед сдачей.
    print(f"\n=== SUBMISSION READY ===")
    print(f"  alpha={best_alpha:.2f}, thr={best_thr:.4f}")
    print(f"  holdout AUC={best_auc:.5f}, F1={best_f1_:.5f}")
    print(f"  CSV: {csv_path}")


if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        main()
    except BaseException as exc:
        log.crash(exc, ART / "submission_crash.log")
        raise
