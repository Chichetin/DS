"""
Шаг F5: CatBoost на 22%-сэмпле, eval на ПОЛНОМ holdout.

Train:   c4_train_22.parquet (~7.3M строк, 22% buyers)
Holdout: c4_holdout_full.parquet (~3.7M строк, ВСЕ buyers за 09-21..09-30)
Test:    c4_test.parquet (4.03M строк, 10-01..10-10)

Параметры — те же, что у текущего чемпиона CatBoost C4 (depth=6, iter=2000),
но на 1.7x больших данных.

Артефакты в artifacts/catboost_22/.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, confusion_matrix,
    precision_score, recall_score,
)

from _log import Logger

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features"
ART = ROOT / "artifacts" / "catboost_22"
ART.mkdir(parents=True, exist_ok=True)

CAT_COLS = [
    "delivery_service", "platform_id", "city", "category_name",
    "microcat_name", "buyer_gender", "dominant_payment_method",
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

TRAIN_PATH = FEAT / "c4_train_22.parquet"
HOLDOUT_PATH = FEAT / "c4_holdout_full.parquet"
TEST_PATH = FEAT / "c4_test.parquet"

log = Logger(ART / "run.log")


def load_pandas(path: Path, name: str) -> pd.DataFrame:
    """Float32 даункаст + кат-колонки строками (требование CatBoost)."""
    import gc
    df_pl = pl.read_parquet(path)
    skip = set(CAT_COLS) | set(ID_COLS) | {TARGET}
    casts: list[pl.Expr] = []
    for col, dt in zip(df_pl.columns, df_pl.dtypes):
        if col in skip:
            continue
        if dt == pl.Float64:
            casts.append(pl.col(col).cast(pl.Float32).alias(col))
        elif dt == pl.Int64:
            casts.append(pl.col(col).cast(pl.Int32).alias(col))
    if casts:
        df_pl = df_pl.with_columns(casts)
    df = df_pl.to_pandas()
    del df_pl
    gc.collect()
    for c in CAT_COLS:
        df[c] = df[c].astype("string").fillna("__missing__")
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    log(f"  {name}: {len(df):,} rows × {df.shape[1]} cols, {mem_mb:.0f} MB")
    return df


def main() -> None:
    import catboost
    log(f"CatBoost on 22% sample. catboost={catboost.__version__}, polars={pl.__version__}")
    log(f"  train: {TRAIN_PATH}")
    log(f"  holdout (full): {HOLDOUT_PATH}")
    log(f"  test: {TEST_PATH}")

    log.step("Step 1/5: load")
    train = load_pandas(TRAIN_PATH, "train")
    holdout = load_pandas(HOLDOUT_PATH, "holdout")
    test = load_pandas(TEST_PATH, "test")

    feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
    log(f"  feature_cols ({len(feature_cols)}): {feature_cols}")
    X_train, y_train = train[feature_cols], train[TARGET].astype(np.int8).values
    X_hold, y_hold = holdout[feature_cols], holdout[TARGET].astype(np.int8).values
    X_test = test[feature_cols]

    log.step("Step 2/5: fit CatBoost")
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

    log.step("Step 3/5: holdout metrics")
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
    log(f"  baseline (predict majority): F1=0, target_rate={float(y_hold.mean()):.4%}")

    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    imp.to_csv(ART / "importance.csv", index=False, encoding="utf-8")
    log("  top-20 by importance:")
    for _, r in imp.head(20).iterrows():
        log(f"    {r['feature']:35s} importance={r['importance']:.4f}")

    log.step("Step 4/5: test predictions")
    p_test = model.predict_proba(X_test)[:, 1]
    pred_test = (p_test >= bthr).astype(np.int8)
    log(f"  test prob: min={p_test.min():.4f}, max={p_test.max():.4f}, mean={p_test.mean():.4f}")
    log(f"  test predictions positive: {pred_test.sum():,} ({pred_test.mean():.4%})")

    out_hold = pl.from_pandas(pd.DataFrame({
        "deliveryorder_id": holdout["deliveryorder_id"].astype(str).to_numpy(),
        "item_id":          holdout["item_id"].astype(str).to_numpy(),
        "order_create_date": pd.to_datetime(holdout["order_create_date"]).dt.date.to_numpy(),
        "prob_return":      p_hold,
        "is_return":        y_hold.astype(bool),
    }))
    out_hold.write_parquet(ART / "holdout_predictions.parquet", compression="zstd")
    log(f"  → {ART / 'holdout_predictions.parquet'}")

    out_pred = pl.from_pandas(pd.DataFrame({
        "deliveryorder_id": test["deliveryorder_id"].astype(str).to_numpy(),
        "item_id":          test["item_id"].astype(str).to_numpy(),
        "order_create_date": pd.to_datetime(test["order_create_date"]).dt.date.to_numpy(),
        "prob_return":      p_test,
        "is_return":        pred_test.astype(bool),
    }))
    out_pred.write_parquet(ART / "predictions.parquet", compression="zstd")
    log(f"  → {ART / 'predictions.parquet'}")

    log.step("Step 5/5: save model & metrics")
    model.save_model(str(ART / "catboost.cbm"))

    metrics_txt = (
        f"ROC-AUC:   {auc:.5f}\n"
        f"best F1:   {bf1:.5f} @ threshold {bthr:.4f}\n"
        f"precision: {p_at:.5f}\n"
        f"recall:    {r_at:.5f}\n"
        f"confusion matrix [[TN, FP],[FN, TP]]:\n{cm}\n"
        f"best_iteration: {best_iter}\n"
        f"target_rate_holdout: {float(y_hold.mean()):.4%}\n"
        f"test_pos_rate: {pred_test.mean():.4%}\n"
        f"params: {PARAMS}\n"
    )
    (ART / "metrics.txt").write_text(metrics_txt, encoding="utf-8")
    log(f"  → {ART / 'metrics.txt'}")

    log.done()


if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        main()
    except BaseException as exc:
        log.crash(exc, ART / "crash.log")
        raise
