"""
Шаг E4: CatBoost на C3-фичах. Сравнение с лучшим LGBM (A_baseline).

CatBoost нативно работает с категориальными (cat_features=...), сам справляется
с NaN в числовых, и часто даёт лучше результат на high-card категориальных
без отдельного target encoding.

Конфиг (стартовый, без тюнинга):
  iterations=2000, learning_rate=0.03, depth=6 (≈ num_leaves=64)
  auto_class_weights='Balanced'
  od_type='Iter', od_wait=30 (early stopping)

Артефакты в artifacts/baseline_cat/.
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, confusion_matrix,
    precision_score, recall_score,
)

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features"
ART = ROOT / "artifacts" / "baseline_cat"
ART.mkdir(parents=True, exist_ok=True)
LOG_FILE = ART / "run.log"

CAT_COLS = [
    "delivery_service",
    "platform_id",
    "city",
    "category_name",
    "microcat_name",
    "buyer_gender",
    "dominant_payment_method",
]
ID_COLS = ["deliveryorder_id", "item_id", "order_create_date"]
TARGET = "is_return"

PARAMS = dict(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    eval_metric="AUC",
    auto_class_weights="Balanced",
    od_type="Iter",
    od_wait=30,
    random_seed=42,
    verbose=100,
    thread_count=-1,
)

_lines: list[str] = []
def log(m: str = "") -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {m}" if m else ""
    _lines.append(line)
    LOG_FILE.write_text("\n".join(_lines), encoding="utf-8")


def load_pandas(name: str, has_target: bool) -> pd.DataFrame:
    df = pl.read_parquet(FEAT / f"c3_{name}.parquet").to_pandas()
    # CatBoost требует категориальные как string (не Float / не Categorical)
    for c in CAT_COLS:
        df[c] = df[c].astype("string").fillna("__missing__")
    log(f"  {name}: {len(df):,} rows × {df.shape[1]} cols, target_in_cols={has_target and TARGET in df.columns}")
    return df


def main() -> None:
    t0 = time.time()
    import catboost
    log(f"CatBoost baseline. catboost={catboost.__version__}, polars={pl.__version__}")

    log("=" * 60)
    log("Step 1/5: load")
    train = load_pandas("train", True)
    holdout = load_pandas("holdout", True)
    test = load_pandas("test", False)

    feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
    log(f"  feature_cols ({len(feature_cols)}): {feature_cols}")
    X_train, y_train = train[feature_cols], train[TARGET].astype(np.int8).values
    X_hold, y_hold = holdout[feature_cols], holdout[TARGET].astype(np.int8).values
    X_test = test[feature_cols]

    log("=" * 60)
    log("Step 2/5: fit CatBoost")
    log(f"  params: {PARAMS}")
    model = CatBoostClassifier(**PARAMS, cat_features=CAT_COLS)
    model.fit(
        X_train, y_train,
        eval_set=(X_hold, y_hold),
        use_best_model=True,
    )
    best_iter = int(model.get_best_iteration() or PARAMS["iterations"])
    log(f"  best_iteration: {best_iter}")
    log(f"  best score: {model.get_best_score()}")

    log("=" * 60)
    log("Step 3/5: holdout metrics")
    p_hold = model.predict_proba(X_hold)[:, 1]
    auc = roc_auc_score(y_hold, p_hold)
    precs, recs, thrs = precision_recall_curve(y_hold, p_hold)
    f1s = 2 * precs * recs / np.clip(precs + recs, 1e-12, None)
    bidx = int(np.argmax(f1s[:-1]))
    bthr = float(thrs[bidx])
    bf1 = float(f1s[bidx])
    pred = (p_hold >= bthr).astype(np.int8)
    p_at = precision_score(y_hold, pred)
    r_at = recall_score(y_hold, pred)
    cm = confusion_matrix(y_hold, pred)

    log(f"  ROC-AUC:   {auc:.5f}")
    log(f"  best F1:   {bf1:.5f}  @ threshold {bthr:.4f}")
    log(f"  precision: {p_at:.5f}")
    log(f"  recall:    {r_at:.5f}")
    log(f"  CM (rows=true, cols=pred): TN={cm[0,0]:,} FP={cm[0,1]:,} | FN={cm[1,0]:,} TP={cm[1,1]:,}")

    base_rate = float(y_hold.mean())
    log(f"  baseline (predict majority): F1=0, target_rate={base_rate:.4%}")

    # Feature importance — CatBoost: PredictionValuesChange (default)
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    imp.to_csv(ART / "importance.csv", index=False, encoding="utf-8")
    log("  top-15 by importance:")
    for _, r in imp.head(15).iterrows():
        log(f"    {r['feature']:35s} importance={r['importance']:.4f}")

    log("=" * 60)
    log("Step 4/5: test predictions")
    p_test = model.predict_proba(X_test)[:, 1]
    pred_test = (p_test >= bthr).astype(np.int8)
    log(f"  test prob: min={p_test.min():.4f}, max={p_test.max():.4f}, mean={p_test.mean():.4f}")
    log(f"  test predictions positive: {pred_test.sum():,} ({pred_test.mean():.4%})")

    out_pred = pl.from_pandas(pd.DataFrame({
        "deliveryorder_id": test["deliveryorder_id"].astype(str).to_numpy(),
        "item_id":          test["item_id"].astype(str).to_numpy(),
        "order_create_date": pd.to_datetime(test["order_create_date"]).dt.date.to_numpy(),
        "prob_return":      p_test,
        "is_return":        pred_test.astype(bool),
    }))
    out_pred.write_parquet(ART / "predictions.parquet", compression="zstd")
    log(f"  → {ART / 'predictions.parquet'}")

    log("=" * 60)
    log("Step 5/5: save model & metrics")
    model.save_model(str(ART / "catboost.cbm"))

    metrics_txt = (
        f"ROC-AUC:   {auc:.5f}\n"
        f"best F1:   {bf1:.5f} @ threshold {bthr:.4f}\n"
        f"precision: {p_at:.5f}\n"
        f"recall:    {r_at:.5f}\n"
        f"confusion matrix [[TN, FP],[FN, TP]]:\n{cm}\n"
        f"best_iteration: {best_iter}\n"
        f"target_rate_holdout: {base_rate:.4%}\n"
        f"test_pos_rate: {pred_test.mean():.4%}\n"
    )
    (ART / "metrics.txt").write_text(metrics_txt, encoding="utf-8")
    log(f"  → {ART / 'metrics.txt'}")

    log("=" * 60)
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
