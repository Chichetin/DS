"""
Шаг E3: тюнинг LGBM на C3-фичах. Последовательно прогоняем 4 конфигурации,
сравниваем ROC-AUC и F1 на holdout.

Конфигурации:
  A) baseline          — текущие параметры (10/13_baseline)
  B) deeper+slower     — num_leaves=127, lr=0.03, min_data=500, n_est=2000
  C) regularized       — B + reg_alpha=0.5, reg_lambda=0.5
  D) scale_pos_weight  — B, но без is_unbalance, с scale_pos_weight=20

Артефакты в artifacts/tuning/:
  run.log
  results.csv  — сводная таблица
  importance_<cfg>.csv для каждого
  predictions_<cfg>.parquet для каждого (test)
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
    precision_score, recall_score,
)

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features"
ART = ROOT / "artifacts" / "tuning"
ART.mkdir(parents=True, exist_ok=True)
LOG_FILE = ART / "run.log"

CAT_COLS = [
    "delivery_service", "platform_id", "city", "category_name",
    "microcat_name", "buyer_gender", "dominant_payment_method",
]
ID_COLS = ["deliveryorder_id", "item_id", "order_create_date"]
TARGET = "is_return"

BASE = dict(
    objective="binary", metric="auc",
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
    random_state=42, n_jobs=-1, verbose=-1,
)

CONFIGS = [
    ("A_baseline", {
        **BASE, "learning_rate": 0.05, "num_leaves": 63,
        "min_data_in_leaf": 200, "n_estimators": 1000,
        "is_unbalance": True,
    }),
    ("B_deeper", {
        **BASE, "learning_rate": 0.03, "num_leaves": 127,
        "min_data_in_leaf": 500, "n_estimators": 2000,
        "is_unbalance": True,
    }),
    ("C_regularized", {
        **BASE, "learning_rate": 0.03, "num_leaves": 127,
        "min_data_in_leaf": 500, "n_estimators": 2000,
        "is_unbalance": True,
        "reg_alpha": 0.5, "reg_lambda": 0.5,
    }),
    ("D_scale_pos_weight", {
        **BASE, "learning_rate": 0.03, "num_leaves": 127,
        "min_data_in_leaf": 500, "n_estimators": 2000,
        "scale_pos_weight": 20.0,
    }),
]

_lines: list[str] = []
def log(m: str = "") -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {m}" if m else ""
    _lines.append(line)
    LOG_FILE.write_text("\n".join(_lines), encoding="utf-8")


def load_pandas(name: str, has_target: bool) -> pd.DataFrame:
    df = pl.read_parquet(FEAT / f"c3_{name}.parquet").to_pandas()
    df["platform_id"] = df["platform_id"].astype("string")
    log(f"  {name}: {len(df):,} rows × {df.shape[1]} cols")
    return df


def sync_categoricals(dfs: list[pd.DataFrame]) -> None:
    for col in CAT_COLS:
        all_vals = pd.concat([d[col] for d in dfs], ignore_index=True)
        cats = pd.Index(all_vals.dropna().unique()).sort_values()
        for d in dfs:
            d[col] = pd.Categorical(d[col], categories=cats)


def run_one(cfg_name: str, params: dict, X_train, y_train, X_hold, y_hold,
            X_test, test_ids: pd.DataFrame) -> dict:
    t0 = time.time()
    log("=" * 60)
    log(f"Config: {cfg_name}")
    log(f"  params: {params}")

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_hold, y_hold)], eval_metric="auc",
        categorical_feature=CAT_COLS,
        callbacks=[
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    best_iter = int(model.best_iteration_ or model.n_estimators)

    p_hold = model.predict_proba(X_hold, num_iteration=best_iter)[:, 1]
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
    elapsed = time.time() - t0

    log(f"  best_iter:  {best_iter}")
    log(f"  ROC-AUC:    {auc:.5f}")
    log(f"  best F1:    {bf1:.5f} @ thr {bthr:.4f}")
    log(f"  precision:  {p_at:.5f}")
    log(f"  recall:     {r_at:.5f}")
    log(f"  CM: TN={cm[0,0]:,} FP={cm[0,1]:,} | FN={cm[1,0]:,} TP={cm[1,1]:,}")
    log(f"  elapsed:    {elapsed:.1f}s")

    # Feature importance
    feature_cols = list(X_train.columns)
    imp = pd.DataFrame({
        "feature": feature_cols,
        "gain":  model.booster_.feature_importance(importance_type="gain"),
        "split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    imp.to_csv(ART / f"importance_{cfg_name}.csv", index=False, encoding="utf-8")
    log("  top-5 by gain:")
    for _, r in imp.head(5).iterrows():
        log(f"    {r['feature']:32s} gain={r['gain']:.0f}")

    # Test predictions
    p_test = model.predict_proba(X_test, num_iteration=best_iter)[:, 1]
    pred_test = (p_test >= bthr).astype(np.int8)
    log(f"  test prob: mean={p_test.mean():.4f}, pos_rate={pred_test.mean():.4%}")

    out_pred = pl.from_pandas(pd.DataFrame({
        "deliveryorder_id": test_ids["deliveryorder_id"].astype(str).to_numpy(),
        "item_id":          test_ids["item_id"].astype(str).to_numpy(),
        "order_create_date": pd.to_datetime(test_ids["order_create_date"]).dt.date.to_numpy(),
        "prob_return":      p_test,
        "is_return":        pred_test.astype(bool),
    }))
    out_pred.write_parquet(ART / f"predictions_{cfg_name}.parquet", compression="zstd")

    return {
        "cfg": cfg_name, "best_iter": best_iter,
        "auc": auc, "f1": bf1, "thr": bthr,
        "precision": p_at, "recall": r_at,
        "test_pos_rate": float(pred_test.mean()),
        "elapsed_s": elapsed,
    }


def main() -> None:
    t0 = time.time()
    log(f"LGBM tuning. lightgbm={lgb.__version__}, polars={pl.__version__}")

    log("=" * 60)
    log("Load c3_*")
    train = load_pandas("train", True)
    holdout = load_pandas("holdout", True)
    test = load_pandas("test", False)
    sync_categoricals([train, holdout, test])

    feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
    log(f"  feature_cols: {len(feature_cols)}")
    X_train, y_train = train[feature_cols], train[TARGET].astype(np.int8).values
    X_hold, y_hold = holdout[feature_cols], holdout[TARGET].astype(np.int8).values
    X_test = test[feature_cols]
    test_ids = test[["deliveryorder_id", "item_id", "order_create_date"]]

    results = []
    for cfg_name, params in CONFIGS:
        r = run_one(cfg_name, params, X_train, y_train, X_hold, y_hold, X_test, test_ids)
        results.append(r)
        # Сохраняем сводку после каждого прогона на случай краша
        pd.DataFrame(results).to_csv(ART / "results.csv", index=False, encoding="utf-8")

    log("=" * 60)
    log("SUMMARY")
    df = pd.DataFrame(results).sort_values("auc", ascending=False)
    for _, r in df.iterrows():
        log(f"  {r['cfg']:22s} AUC={r['auc']:.5f}  F1={r['f1']:.5f}  iter={int(r['best_iter']):4d}  "
            f"pos_rate={r['test_pos_rate']:.2%}  elapsed={r['elapsed_s']:.0f}s")
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
