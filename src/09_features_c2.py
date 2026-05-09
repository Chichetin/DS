"""
Шаг C2/часть 2: C1 + history → C2.

Логика:
1. Загружаем c1_{train,holdout,test}.parquet (без buyer_id/seller_id).
2. Подтягиваем buyer_id/seller_id из orders_train_sample / orders_test
   через join по (deliveryorder_id, item_id).
3. Для train+holdout: join с hist_{buyer,seller,item}_daily по (entity, order_create_date).
4. Для test: join с hist_{buyer,seller,item}_snapshot по entity.
5. Считаем производные фичи: past_return_rate (returns/orders), past_avg_price,
   days_since_last_order.
6. Сохраняем c2_{train,holdout,test}.parquet.

Стратегия: всё через polars LazyFrame + sink_parquet (стримим, не валим память).
"""
from __future__ import annotations
import os
from pathlib import Path

import polars as pl

from _log import Logger

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
FEAT = ROOT / "data" / "features"
ART = ROOT / "artifacts" / "features"
ART.mkdir(parents=True, exist_ok=True)

SUFFIX = os.environ.get("SUFFIX", "")
SAMPLE_PATH = os.environ.get("SAMPLE_PATH", "orders_train_sample.parquet")
TARGETS = set(os.environ.get("TARGETS", "train,holdout,test").split(","))

log = Logger(ART / f"c2{SUFFIX}.log")


ENTITIES = ["buyer", "seller", "item"]


def add_history_features(
    c1_lf: pl.LazyFrame,
    orders_src: Path,
    is_test: bool,
) -> pl.LazyFrame:
    """Подмешивает buyer_id/seller_id из orders_src и history фичи 3 сущностей."""
    # 1. buyer_id / seller_id из исходного orders parquet
    ids = (
        pl.scan_parquet(orders_src)
        .select("deliveryorder_id", "item_id", "buyer_id", "seller_id")
    )
    lf = c1_lf.join(ids, on=["deliveryorder_id", "item_id"], how="left")

    # 2. join с history-агрегатами для каждой сущности
    for short in ENTITIES:
        key_col = f"{short}_id" if short != "item" else "item_id"
        if is_test:
            hist = pl.scan_parquet(FEAT / f"hist_{short}_snapshot.parquet")
            lf = lf.join(hist, on=key_col, how="left")
        else:
            hist = pl.scan_parquet(FEAT / f"hist_{short}_daily.parquet")
            lf = lf.join(
                hist,
                on=[key_col, "order_create_date"],
                how="left",
            )

    # 3. Производные: rate, avg_price, days_since_last_order, 30d/7d rates
    derived: list[pl.Expr] = []
    for short in ENTITIES:
        n_o = pl.col(f"{short}_past_orders").fill_null(0)
        n_r = pl.col(f"{short}_past_returns").fill_null(0)
        s_p = pl.col(f"{short}_past_price_sum").fill_null(0.0)
        last_d = pl.col(f"{short}_last_order_date")
        n_o_30 = pl.col(f"{short}_past_30d_orders").fill_null(0)
        n_r_30 = pl.col(f"{short}_past_30d_returns").fill_null(0)
        n_o_7 = pl.col(f"{short}_past_7d_orders").fill_null(0)
        n_r_7 = pl.col(f"{short}_past_7d_returns").fill_null(0)

        derived += [
            n_o.cast(pl.UInt32).alias(f"{short}_past_orders"),
            n_r.cast(pl.UInt32).alias(f"{short}_past_returns"),
            (
                pl.when(n_o > 0).then(n_r.cast(pl.Float64) / n_o.cast(pl.Float64))
                .otherwise(None)
            ).alias(f"{short}_past_return_rate"),
            (
                pl.when(n_o > 0).then(s_p / n_o.cast(pl.Float64))
                .otherwise(None)
            ).alias(f"{short}_past_avg_price"),
            (
                pl.when(last_d.is_not_null())
                .then((pl.col("order_create_date") - last_d).dt.total_days())
                .otherwise(None)
                .cast(pl.Int32)
            ).alias(f"{short}_days_since_last_order"),
            n_o_30.cast(pl.UInt32).alias(f"{short}_past_30d_orders"),
            (
                pl.when(n_o_30 > 0).then(n_r_30.cast(pl.Float64) / n_o_30.cast(pl.Float64))
                .otherwise(None)
            ).alias(f"{short}_past_30d_return_rate"),
            n_o_7.cast(pl.UInt32).alias(f"{short}_past_7d_orders"),
            (
                pl.when(n_o_7 > 0).then(n_r_7.cast(pl.Float64) / n_o_7.cast(pl.Float64))
                .otherwise(None)
            ).alias(f"{short}_past_7d_return_rate"),
        ]

    # Удаляем join-only/сырые поля
    drop_cols = ["buyer_id", "seller_id"] + [
        f"{short}_past_price_sum" for short in ENTITIES
    ] + [
        f"{short}_last_order_date" for short in ENTITIES
    ] + [
        f"{short}_past_30d_returns" for short in ENTITIES
    ] + [
        f"{short}_past_7d_returns" for short in ENTITIES
    ]

    lf = lf.with_columns(derived).drop(drop_cols)
    return lf


def main() -> None:
    log(f"C2 features start. polars={pl.__version__}")

    sample_full = CLEAN / SAMPLE_PATH
    targets_all = [
        ("train",   FEAT / f"c1_train{SUFFIX}.parquet",   sample_full,                   False),
        ("holdout", FEAT / f"c1_holdout{SUFFIX}.parquet", sample_full,                   False),
        ("test",    FEAT / "c1_test.parquet",             CLEAN / "orders_test.parquet", True),
    ]

    for name, c1_src, orders_src, is_test in targets_all:
        if name not in TARGETS:
            continue
        log.step(f"Build c2_{name}{SUFFIX} from {c1_src.name}")
        c1_lf = pl.scan_parquet(c1_src)
        c2_lf = add_history_features(c1_lf, orders_src, is_test)
        # test без суффикса (одинаковый для всех)
        out_suffix = "" if is_test else SUFFIX
        out = FEAT / f"c2_{name}{out_suffix}.parquet"
        log(f"  sink → {out.name}")
        c2_lf.sink_parquet(out, compression="zstd")
        n = pl.scan_parquet(out).select(pl.len()).collect().item()
        log(f"  c2_{name}{out_suffix}: {n:,} rows")

    # Schema dump
    if "train" in TARGETS:
        log.step(f"Final schema (c2_train{SUFFIX}):")
        schema = pl.scan_parquet(FEAT / f"c2_train{SUFFIX}.parquet").collect_schema()
        for n, dt in zip(schema.names(), schema.dtypes()):
            log(f"  {n}: {dt}")

    log.done()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        main()
    except BaseException as exc:
        log.crash(exc, ART / "c2_crash.log")
        raise
