"""
Шаг C1: базовые фичи без history (order + item + user + payment).

Вход:
- data/clean/orders_train_sample.parquet (4.97M)
- data/clean/orders_test.parquet (4.03M)
- data/clean/items.parquet (20.6M)
- data/clean/users.parquet (11.7M)
- data/clean/payments_agg.parquet (29.6M)

Выход (data/features/):
- c1_train.parquet     — orders_train_sample, order_create_date < 2025-09-21
- c1_holdout.parquet   — orders_train_sample, order_create_date in 09-21..09-30
- c1_test.parquet      — orders_test, без is_return

Стратегия: один pipeline-builder, применяется к sample и test одинаково (фичи симметричны).
"""
from __future__ import annotations
import os
from datetime import date
from pathlib import Path

import polars as pl

from _log import Logger

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "data" / "features"
ART = ROOT / "artifacts" / "features"
OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)

# Env vars (backward compat: default = original behavior)
SUFFIX = os.environ.get("SUFFIX", "")
SAMPLE_PATH = os.environ.get("SAMPLE_PATH", "orders_train_sample.parquet")
TARGETS = set(os.environ.get("TARGETS", "train,holdout,test").split(","))

HOLDOUT_START = date(2025, 9, 21)
HOLDOUT_END = date(2025, 9, 30)

log = Logger(ART / f"c1{SUFFIX}.log")


def load_lookups() -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Items / users(buyer) / users(seller) / payments_agg, prepared for join."""
    items = (
        pl.scan_parquet(CLEAN / "items.parquet")
        .select(
            "item_id",
            "category_name",
            "microcat_name",
            pl.col("starttime").cast(pl.Date).alias("item_starttime"),
            pl.col("close_date").cast(pl.Date).alias("item_close_date"),
        )
    )
    buyer_feats = (
        pl.scan_parquet(CLEAN / "users.parquet")
        .select(
            pl.col("user_id").alias("buyer_id"),
            pl.col("gender").alias("buyer_gender"),
            pl.col("iscompany").alias("buyer_iscompany"),
            pl.col("isblocked").alias("buyer_isblocked"),
            pl.col("tenure_days").alias("buyer_tenure_days"),
            pl.col("is_seller").alias("buyer_is_seller"),
        )
    )
    seller_feats = (
        pl.scan_parquet(CLEAN / "users.parquet")
        .select(
            pl.col("user_id").alias("seller_id"),
            pl.col("iscompany").alias("seller_iscompany"),
            pl.col("isblocked").alias("seller_isblocked"),
            pl.col("tenure_days").alias("seller_tenure_days"),
        )
    )
    pay = (
        pl.scan_parquet(CLEAN / "payments_agg.parquet")
        .select(
            "deliveryorder_id",
            "n_pay",
            pl.col("sum_amount").alias("pay_sum_amount"),
            "dominant_payment_method",
        )
    )
    return items, buyer_feats, seller_feats, pay


def build_features(orders_lf: pl.LazyFrame, has_target: bool) -> pl.LazyFrame:
    items, buyer_feats, seller_feats, pay = load_lookups()

    lf = (
        orders_lf
        .join(items, on="item_id", how="left")
        .join(buyer_feats, on="buyer_id", how="left")
        .join(seller_feats, on="seller_id", how="left")
        .join(pay, on="deliveryorder_id", how="left")
    )

    # Убраны лики: is_cancelled, days_to_accept, has_buyer_terminal,
    # has_seller_terminal, days_first_pay — все эти поля заполняются ПОСЛЕ
    # order_create_date (отмена / приём / попадание в ПВЗ / refund-платёж).
    out_cols: list[pl.Expr] = [
        # IDs (для сабмита и трейсинга)
        pl.col("deliveryorder_id"),
        pl.col("item_id"),
        pl.col("order_create_date"),
        # --- Order-level numeric ---
        pl.col("order_create_date").dt.weekday().alias("order_dow"),
        pl.col("order_create_date").dt.day().alias("order_dom"),
        (pl.col("order_create_date").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
        pl.col("order_price"),
        (pl.col("order_price") + 1.0).log().alias("log_order_price"),
        # --- Order-level categorical ---
        pl.col("delivery_service"),
        pl.col("platform_id"),
        pl.col("city"),
        # --- Item-level ---
        pl.col("category_name"),
        pl.col("microcat_name"),
        (pl.col("order_create_date") - pl.col("item_starttime")).dt.total_days().alias("item_lifetime_at_order"),
        (
            pl.col("item_close_date").is_null()
            | (pl.col("item_close_date") > pl.col("order_create_date"))
        ).cast(pl.Int8).alias("is_active_at_order"),
        # --- Buyer ---
        pl.col("buyer_gender"),
        pl.col("buyer_iscompany").cast(pl.Int8).alias("buyer_iscompany"),
        pl.col("buyer_isblocked").cast(pl.Int8).alias("buyer_isblocked"),
        pl.col("buyer_tenure_days"),
        pl.col("buyer_is_seller").cast(pl.Int8).alias("buyer_is_seller"),
        # --- Seller ---
        pl.col("seller_iscompany").cast(pl.Int8).alias("seller_iscompany"),
        pl.col("seller_isblocked").cast(pl.Int8).alias("seller_isblocked"),
        pl.col("seller_tenure_days"),
        # --- Payment (small leak ~0.02% via late_pay — keeping aggregates) ---
        pl.col("n_pay"),
        pl.col("pay_sum_amount"),
        pl.col("dominant_payment_method"),
        (pl.col("pay_sum_amount") / pl.col("order_price")).alias("pay_to_price_ratio"),
    ]

    if has_target:
        out_cols.append(pl.col("is_return").cast(pl.Int8).alias("is_return"))

    return lf.select(out_cols)


def main() -> None:
    log(f"C1 features start. polars={pl.__version__}, SUFFIX={SUFFIX!r}, SAMPLE_PATH={SAMPLE_PATH}, TARGETS={TARGETS}")

    # 1. Sample → train + holdout (по env var)
    if {"train", "holdout"} & TARGETS:
        log.step(f"Step 1/2: {SAMPLE_PATH} → train + holdout (time-based split)")
        sample_feats = build_features(
            pl.scan_parquet(CLEAN / SAMPLE_PATH),
            has_target=True,
        )

        if "train" in TARGETS:
            train_lf = sample_feats.filter(pl.col("order_create_date") < HOLDOUT_START)
            out_train = OUT / f"c1_train{SUFFIX}.parquet"
            log(f"  sink → {out_train.name}")
            train_lf.sink_parquet(out_train, compression="zstd")
            n_train = pl.scan_parquet(out_train).select(pl.len()).collect().item()
            train_rate = pl.scan_parquet(out_train).select(pl.col("is_return").mean()).collect().item()
            log(f"  train:   {n_train:>10,} rows, target rate {train_rate:.4%}")

        if "holdout" in TARGETS:
            holdout_lf = sample_feats.filter(
                (pl.col("order_create_date") >= HOLDOUT_START)
                & (pl.col("order_create_date") <= HOLDOUT_END)
            )
            out_holdout = OUT / f"c1_holdout{SUFFIX}.parquet"
            log(f"  sink → {out_holdout.name}")
            holdout_lf.sink_parquet(out_holdout, compression="zstd")
            n_holdout = pl.scan_parquet(out_holdout).select(pl.len()).collect().item()
            holdout_rate = pl.scan_parquet(out_holdout).select(pl.col("is_return").mean()).collect().item()
            log(f"  holdout: {n_holdout:>10,} rows, target rate {holdout_rate:.4%}")

    # 2. Test
    if "test" in TARGETS:
        log.step("Step 2/2: orders_test → c1_test")
        test_feats = build_features(
            pl.scan_parquet(CLEAN / "orders_test.parquet"),
            has_target=False,
        )
        out_test = OUT / "c1_test.parquet"  # test без суффикса (одинаковый для всех)
        log(f"  sink → {out_test.name}")
        test_feats.sink_parquet(out_test, compression="zstd")
        n_test = pl.scan_parquet(out_test).select(pl.len()).collect().item()
        log(f"  test:    {n_test:>10,} rows")

    # 3. Schema dump для отчёта (только если строили train)
    if "train" in TARGETS:
        log.step("Final schema (c1_train):")
        schema = pl.scan_parquet(OUT / f"c1_train{SUFFIX}.parquet").collect_schema()
        for name, dt in zip(schema.names(), schema.dtypes()):
            log(f"  {name}: {dt}")

    log.done()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        log.crash(exc, ART / "c1_crash.log")
        raise
