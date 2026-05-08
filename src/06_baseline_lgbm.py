"""
Шаг D1: LGBM бейзлайн на C1-фичах.

Что делаем:
1. Читаем c1_{train, holdout, test}.parquet.
2. Синхронизируем pd.Categorical уровни между тремя сетами (concat → fit, потом transform каждого).
3. Тренируем LGBMClassifier с early stopping по AUC на holdout.
4. Метрики на holdout: ROC-AUC, F1@best_threshold, precision, recall, confusion matrix.
5. Feature importance, predictions на test.

Артефакты в artifacts/baseline_c1/:
  - lgbm.txt          — booster
  - metrics.txt       — все метрики
  - importance.csv    — feature importance (gain + split)
  - predictions.parquet — predictions на test (deliveryorder_id, item_id, order_create_date, prob_return)
  - run.log
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, confusion_matrix,
    f1_score, precision_score, recall_score,
)

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features"
ART = ROOT / "artifacts" / "baseline_c1"
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
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    is_unbalance=True,
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

_lines: list[str] = []
def log(m: str = "") -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {m}" if m else ""
    _lines.append(line)
    LOG_FILE.write_text("\n".join(_lines), encoding="utf-8")


def load_pandas(name: str, has_target: bool) -> pd.DataFrame:
    df = pl.read_parquet(FEAT / f"c1_{name}.parquet").to_pandas()
    # platform_id из Float64 — для category лучше как str (NaN сохранится)
    df["platform_id"] = df["platform_id"].astype("string")
    log(f"  {name}: {len(df):,} rows × {df.shape[1]} cols, target_in_cols={has_target and TARGET in df.columns}")
    return df


def sync_categoricals(dfs: list[pd.DataFrame]) -> None:
    """Inplace: для каждой CAT_COLS — общие категории между всеми DataFrame'ами."""
    for col in CAT_COLS:
        all_vals = pd.concat([d[col] for d in dfs], ignore_index=True)
        cats = pd.Index(all_vals.dropna().unique()).sort_values()
        log(f"    {col}: {len(cats)} unique values")
        for d in dfs:
            d[col] = pd.Categorical(d[col], categories=cats)


def main() -> None:
    t0 = time.time()
    log(f"LGBM baseline. lightgbm={lgb.__version__}, polars={pl.__version__}")

    # 1. Загрузка
    log("=" * 60)
    log("Step 1/5: load")
    train = load_pandas("train", has_target=True)
    holdout = load_pandas("holdout", has_target=True)
    test = load_pandas("test", has_target=False)

    # 2. Sync категориальных уровней
    log("=" * 60)
    log("Step 2/5: sync categorical levels")
    sync_categoricals([train, holdout, test])

    # 3. Подготовка X, y
    feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
    log(f"  feature_cols ({len(feature_cols)}): {feature_cols}")
    X_train, y_train = train[feature_cols], train[TARGET].astype(np.int8).values
    X_hold, y_hold = holdout[feature_cols], holdout[TARGET].astype(np.int8).values
    X_test = test[feature_cols]

    # 4. Train
    log("=" * 60)
    log("Step 3/5: fit LGBM")
    log(f"  params: {PARAMS}")
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_hold, y_hold)],
        eval_metric="auc",
        categorical_feature=CAT_COLS,
        callbacks=[
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(50),
        ],
    )
    best_iter = model.best_iteration_
    log(f"  best_iteration: {best_iter}")
    log(f"  best score: {model.best_score_}")

    # 5. Метрики
    log("=" * 60)
    log("Step 4/5: holdout metrics")
    p_hold = model.predict_proba(X_hold, num_iteration=best_iter)[:, 1]
    auc = roc_auc_score(y_hold, p_hold)

    precisions, recalls, thrs = precision_recall_curve(y_hold, p_hold)
    f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-12, None)
    best_idx = int(np.argmax(f1s[:-1]))  # последний — special, без threshold
    best_thr = float(thrs[best_idx])
    best_f1 = float(f1s[best_idx])
    pred_hold = (p_hold >= best_thr).astype(np.int8)

    p_at_thr = precision_score(y_hold, pred_hold)
    r_at_thr = recall_score(y_hold, pred_hold)
    cm = confusion_matrix(y_hold, pred_hold)

    log(f"  ROC-AUC:   {auc:.5f}")
    log(f"  best F1:   {best_f1:.5f}  @ threshold {best_thr:.4f}")
    log(f"  precision: {p_at_thr:.5f}")
    log(f"  recall:    {r_at_thr:.5f}")
    log(f"  CM (rows=true, cols=pred): TN={cm[0,0]:,} FP={cm[0,1]:,} | FN={cm[1,0]:,} TP={cm[1,1]:,}")

    # Бэйзлайн «всегда False» для сравнения
    base_rate = float(y_hold.mean())
    log(f"  baseline (predict majority): F1=0 (вся positive class missed), target_rate={base_rate:.4%}")

    # Feature importance
    imp = pd.DataFrame({
        "feature": feature_cols,
        "gain":  model.booster_.feature_importance(importance_type="gain"),
        "split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    imp.to_csv(ART / "importance.csv", index=False, encoding="utf-8")
    log("  top-10 by gain:")
    for _, r in imp.head(10).iterrows():
        log(f"    {r['feature']:30s} gain={r['gain']:.0f}  split={r['split']}")

    # 6. Test predictions
    log("=" * 60)
    log("Step 5/5: test predictions")
    p_test = model.predict_proba(X_test, num_iteration=best_iter)[:, 1]
    pred_test = (p_test >= best_thr).astype(np.int8)
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

    # Сохранить booster
    model.booster_.save_model(str(ART / "lgbm.txt"))
    log(f"  → {ART / 'lgbm.txt'}")

    # Сводка метрик
    metrics_txt = (
        f"ROC-AUC:   {auc:.5f}\n"
        f"best F1:   {best_f1:.5f} @ threshold {best_thr:.4f}\n"
        f"precision: {p_at_thr:.5f}\n"
        f"recall:    {r_at_thr:.5f}\n"
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
    try:
        main()
    except BaseException as exc:
        import traceback
        crash = ART / "crash.log"
        crash.write_text(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", encoding="utf-8")
        raise
