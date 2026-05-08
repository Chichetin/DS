"""
Шаг F4: LGBM на ПОЛНОМ train (~30M строк).

Memory-safe loader:
- polars читает parquet, кастит Float64 → Float32 и Int64 → Int32 ДО to_pandas()
- категориальные колонки: pd.Categorical с общими уровнями (sync через polars)
- Holdout всегда `c4_holdout_full.parquet` (полный, ~3.7M строк, для честной метрики)
- После fit освобождаем train DataFrame перед загрузкой test

Ожидаемая память: ~13GB peak. Если OOM — fallback на c4_train_70 (надо отдельно построить).

Артефакты в artifacts/lgbm_full/.
"""
from __future__ import annotations
import gc
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
ART = ROOT / "artifacts" / "lgbm_full"
ART.mkdir(parents=True, exist_ok=True)
LOG_FILE = ART / "run.log"

CAT_COLS = [
    "delivery_service", "platform_id", "city", "category_name",
    "microcat_name", "buyer_gender", "dominant_payment_method",
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
    n_estimators=2000,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

TRAIN_PATH = FEAT / "c4_train_full.parquet"
HOLDOUT_PATH = FEAT / "c4_holdout_full.parquet"
TEST_PATH = FEAT / "c4_test.parquet"

_lines: list[str] = []
def log(m: str = "") -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {m}" if m else ""
    _lines.append(line)
    LOG_FILE.write_text("\n".join(_lines), encoding="utf-8")


def downcast_lazy(lf: pl.LazyFrame, skip: set[str]) -> pl.LazyFrame:
    """Float64→Float32, Int64→Int32 (где возможно). Lazy: каст применяется в стриминге."""
    schema = lf.collect_schema()
    casts: list[pl.Expr] = []
    for col, dt in zip(schema.names(), schema.dtypes()):
        if col in skip:
            continue
        if dt == pl.Float64:
            casts.append(pl.col(col).cast(pl.Float32).alias(col))
        elif dt == pl.Int64:
            casts.append(pl.col(col).cast(pl.Int32).alias(col))
    if casts:
        lf = lf.with_columns(casts)
    return lf


def precompute_cat_mappings(paths: list[Path]) -> dict[str, list]:
    """Через polars lazy: уникальные значения для каждой кат-колонки во всех файлах.
    Очень дешёво по памяти — читаем только нужный столбец."""
    mappings: dict[str, list] = {}
    for col in CAT_COLS:
        unions = [pl.scan_parquet(p).select(pl.col(col).cast(pl.Utf8, strict=False)) for p in paths]
        unique_vals = (
            pl.concat(unions)
            .unique()
            .drop_nulls()
            .sort(col)
            .collect(engine="streaming")
            .to_series()
            .to_list()
        )
        log(f"    {col}: {len(unique_vals)} unique values")
        mappings[col] = unique_vals
    return mappings


def load_compact(path: Path, name: str, cat_mappings: dict[str, list]) -> pd.DataFrame:
    """Стриминг через pl.scan_parquet с даункастом в lazy-плане + кодирование cats
    в polars (через replace_strict). После collect конвертим в pandas: numerics
    уже Float32, cats — int32 коды → pd.Categorical с готовыми code'ами."""
    t = time.time()
    skip = set(CAT_COLS) | set(ID_COLS) | {TARGET}

    lf = pl.scan_parquet(path)
    lf = downcast_lazy(lf, skip=skip)

    # Кодируем cat колонки в polars: string → int (индекс в mappings)
    cat_casts: list[pl.Expr] = []
    for col, cats in cat_mappings.items():
        cat_casts.append(
            pl.col(col).cast(pl.Utf8, strict=False)
            .replace_strict(
                cats,
                list(range(len(cats))),
                default=-1,
                return_dtype=pl.Int32,
            ).alias(col)
        )
    if cat_casts:
        lf = lf.with_columns(cat_casts)

    df_pl = lf.collect(engine="streaming")
    log(f"  {name}: collected polars {len(df_pl):,} rows × {df_pl.width} cols in {time.time()-t:.1f}s, memo={df_pl.estimated_size('mb'):.0f} MB")

    # Сначала вытаскиваем числовые/ID колонки в pandas (Float32/Int32/строки),
    # затем восстанавливаем cat как pd.Categorical из int-кодов.
    cat_codes: dict[str, np.ndarray] = {}
    for col in cat_mappings:
        cat_codes[col] = df_pl[col].to_numpy()
    df_pl = df_pl.drop(list(cat_mappings))

    # Конвертим колонка-за-колонкой через numpy, минуя pl.DataFrame.to_pandas().
    # Под капотом to_pandas() строит pyarrow.Table и зовёт table_to_blocks,
    # которая консолидирует одинаковые dtype'ы в один 2D-массив на блок
    # (~6 ГБ для 30M × 50 Float32) и падает в OOM на 16 ГБ RAM.
    data: dict[str, np.ndarray] = {}
    for col in list(df_pl.columns):
        data[col] = df_pl[col].to_numpy()
    del df_pl
    gc.collect()
    df = pd.DataFrame(data, copy=False)
    del data
    gc.collect()

    for col, codes in cat_codes.items():
        cats = cat_mappings[col]
        # codes -1 → NaN. pandas Categorical .from_codes требует валидные индексы [0, len(cats))
        codes_safe = codes.astype(np.int32, copy=False)
        codes_safe[codes_safe < 0] = -1  # ensure
        df[col] = pd.Categorical.from_codes(codes_safe, categories=cats)
    del cat_codes
    gc.collect()

    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    log(f"  {name}: pandas {len(df):,} rows × {df.shape[1]} cols, {mem_mb:.0f} MB in {time.time()-t:.1f}s")
    return df


def main() -> None:
    t0 = time.time()
    log(f"LGBM FULL. lightgbm={lgb.__version__}, polars={pl.__version__}")
    log(f"  train: {TRAIN_PATH}")
    log(f"  holdout: {HOLDOUT_PATH}")
    log(f"  test: {TEST_PATH}")

    log("=" * 60)
    log("Step 1/5: precompute cat mappings (lazy scan через polars)")
    cat_mappings = precompute_cat_mappings([TRAIN_PATH, HOLDOUT_PATH, TEST_PATH])

    log("=" * 60)
    log("Step 2/5: load (Float32 downcast + cat encoding)")
    train = load_compact(TRAIN_PATH, "train", cat_mappings)
    holdout = load_compact(HOLDOUT_PATH, "holdout", cat_mappings)
    # test загружаем ПОСЛЕ тренировки, чтобы не держать в памяти зря

    feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
    log(f"  feature_cols ({len(feature_cols)}): {feature_cols}")
    X_train, y_train = train[feature_cols], train[TARGET].astype(np.int8).values
    X_hold, y_hold = holdout[feature_cols], holdout[TARGET].astype(np.int8).values

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
            lgb.early_stopping(40, verbose=False),
            lgb.log_evaluation(50),
        ],
    )
    best_iter = model.best_iteration_
    log(f"  best_iteration: {best_iter}")
    log(f"  best score: {model.best_score_}")

    # Освобождаем train (~6 GB), он больше не нужен
    del X_train, y_train, train
    gc.collect()
    log("  freed train memory")

    log("=" * 60)
    log("Step 4/5: holdout metrics")
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

    log(f"  ROC-AUC:   {auc:.5f}")
    log(f"  best F1:   {bf1:.5f}  @ threshold {bthr:.4f}")
    log(f"  precision: {p_at:.5f}")
    log(f"  recall:    {r_at:.5f}")
    log(f"  CM (rows=true, cols=pred): TN={cm[0,0]:,} FP={cm[0,1]:,} | FN={cm[1,0]:,} TP={cm[1,1]:,}")
    log(f"  baseline (predict majority): F1=0, target_rate={float(y_hold.mean()):.4%}")

    imp = pd.DataFrame({
        "feature": feature_cols,
        "gain":  model.booster_.feature_importance(importance_type="gain"),
        "split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    imp.to_csv(ART / "importance.csv", index=False, encoding="utf-8")
    log("  top-20 by gain:")
    for _, r in imp.head(20).iterrows():
        log(f"    {r['feature']:35s} gain={r['gain']:.0f}  split={r['split']}")

    log("=" * 60)
    log("Step 5/5: test predictions (load test now)")
    test = load_compact(TEST_PATH, "test", cat_mappings)
    test_ids = test[ID_COLS].copy()
    X_test = test[feature_cols]
    p_test = model.predict_proba(X_test, num_iteration=best_iter)[:, 1]
    pred_test = (p_test >= bthr).astype(np.int8)
    log(f"  test prob: min={p_test.min():.4f}, max={p_test.max():.4f}, mean={p_test.mean():.4f}")
    log(f"  test predictions positive: {pred_test.sum():,} ({pred_test.mean():.4%})")

    # Save holdout & test predictions для блендинга
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
        "deliveryorder_id": test_ids["deliveryorder_id"].astype(str).to_numpy(),
        "item_id":          test_ids["item_id"].astype(str).to_numpy(),
        "order_create_date": pd.to_datetime(test_ids["order_create_date"]).dt.date.to_numpy(),
        "prob_return":      p_test,
        "is_return":        pred_test.astype(bool),
    }))
    out_pred.write_parquet(ART / "predictions.parquet", compression="zstd")
    log(f"  → {ART / 'predictions.parquet'}")

    model.booster_.save_model(str(ART / "lgbm.txt"))
    log(f"  → {ART / 'lgbm.txt'}")

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
