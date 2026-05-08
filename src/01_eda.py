"""
EDA на сырых CSV через polars в streaming-режиме (out-of-core).
Считает: строки, диапазоны дат, доли пропусков, распределение target,
проверка уникальности (deliveryorder_id, item_id).
Все принты дублируются в artifacts/eda/run.log.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "artifacts" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT / "run.log"
_log_fh = LOG_PATH.open("w", encoding="utf-8")


def log(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    _log_fh.write(msg + "\n")
    _log_fh.flush()
    try:
        print(msg, **kwargs)
    except UnicodeEncodeError:
        # консоль cp1251 не вывозит кириллицу — игнорируем, файл всё равно записан
        pass


# чтобы DataFrame.print тоже летел в лог — переопределим сравнительно мягко
def log_df(df):
    s = str(df)
    _log_fh.write(s + "\n")
    _log_fh.flush()
    try:
        print(s)
    except UnicodeEncodeError:
        pass


FILES = {
    "orders": DATA / "orders.csv",
    "items": DATA / "items.csv",
    "users": DATA / "users.csv",
    "payments": DATA / "payments.csv",
}


def scan(name: str) -> pl.LazyFrame:
    lf = pl.scan_csv(
        FILES[name],
        infer_schema_length=10_000,
        ignore_errors=True,
    )
    if "" in lf.collect_schema().names():
        lf = lf.drop("")
    return lf


def section(title: str) -> None:
    log(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def stats_table(name: str) -> dict:
    section(f"{name.upper()}")
    lf = scan(name)
    schema = lf.collect_schema()
    log("schema:")
    for col, dtype in schema.items():
        log(f"  {col:25s} {dtype}")

    t0 = time.time()
    n_rows = lf.select(pl.len()).collect(engine="streaming").item()
    log(f"\nrows: {n_rows:,}  ({time.time() - t0:.1f}s)")

    null_counts = (
        lf.select([pl.col(c).null_count().alias(c) for c in schema.names()])
        .collect(engine="streaming")
        .row(0, named=True)
    )
    log("nulls:")
    for c, v in null_counts.items():
        pct = 100.0 * v / n_rows if n_rows else 0
        log(f"  {c:25s} {v:>15,}  ({pct:5.2f}%)")

    return {"name": name, "rows": n_rows, "schema": {c: str(t) for c, t in schema.items()}, "nulls": null_counts}


def main():
    summary = {}
    for name in FILES:
        summary[name] = stats_table(name)

    # --- ORDERS: даты с парсингом + target distribution
    section("ORDERS — даты (строгий парсинг, чтобы отсечь сдвинутые строки)")
    orders = scan("orders")
    # Парсим даты строго: непарсящиеся → NULL. Это позволит отсечь broken rows
    orders_clean = orders.with_columns(
        pl.col("order_create_date").str.strptime(pl.Date, format="%Y-%m-%d", strict=False).alias("d_create"),
        pl.col("order_accept_date").str.to_datetime(strict=False).alias("d_accept"),
        pl.col("cancel_date").str.to_datetime(strict=False).alias("d_cancel"),
    )

    bad_rows = (
        orders_clean.select(
            pl.len().alias("total"),
            pl.col("d_create").is_null().sum().alias("create_unparseable"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    log(f"\nstrict parse stats: {bad_rows}")

    date_stats = (
        orders_clean.select(
            pl.col("d_create").min().alias("min_create"),
            pl.col("d_create").max().alias("max_create"),
            pl.col("d_accept").min().alias("min_accept"),
            pl.col("d_accept").max().alias("max_accept"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    log("date ranges (после парсинга):")
    for k, v in date_stats.items():
        log(f"  {k:15s} {v}")

    # is_return: считаем как строку, потом приводим
    target_dist = (
        orders.group_by("is_return")
        .agg(pl.len().alias("n"))
        .sort("is_return")
        .collect(engine="streaming")
    )
    log("\nis_return distribution:")
    log_df(target_dist)

    # test slice (1-10 Oct 2025) по order_create_date
    test_slice = (
        orders_clean.filter(
            pl.col("d_create").is_between(
                pl.date(2025, 10, 1), pl.date(2025, 10, 10), closed="both"
            )
        )
        .select(
            pl.len().alias("rows_in_test"),
            pl.col("is_return").is_null().sum().alias("is_return_null_in_test"),
            pl.col("deliveryorder_id").n_unique().alias("uniq_orders"),
            pl.col("item_id").n_unique().alias("uniq_items"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    log(f"\ntest 01.10.25–10.10.25: {test_slice}")

    # --- ORDERS: уникальность ключей
    section("ORDERS — уникальность (deliveryorder_id, item_id)")
    dup = (
        orders.group_by(["deliveryorder_id", "item_id"])
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
        .select(
            pl.len().alias("n_dup_keys"),
            pl.col("n").sum().alias("rows_in_dups"),
            pl.col("n").max().alias("max_dup"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    log(f"  duplicate (deliveryorder_id, item_id): {dup}")

    items_per_order = (
        orders.group_by("deliveryorder_id")
        .agg(pl.col("item_id").n_unique().alias("n_items"))
        .select(
            pl.col("n_items").mean().alias("mean"),
            pl.col("n_items").median().alias("median"),
            pl.col("n_items").max().alias("max"),
            (pl.col("n_items") > 1).sum().alias("orders_with_multi_items"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    log(f"  items per order: {items_per_order}")

    # --- PAYMENTS
    section("PAYMENTS — оплат на заказ")
    payments = scan("payments")
    pay_per_order = (
        payments.group_by("deliveryorder_id")
        .agg(pl.len().alias("n_pay"), pl.col("amount").sum().alias("sum_amount"))
        .select(
            pl.col("n_pay").mean().alias("mean_n_pay"),
            pl.col("n_pay").max().alias("max_n_pay"),
            (pl.col("n_pay") > 1).sum().alias("orders_with_multi_pay"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    log(f"  payments per order: {pay_per_order}")

    pmethod = (
        payments.group_by("payment_method")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .collect(engine="streaming")
    )
    log("\npayment_method:")
    log_df(pmethod)

    # --- ORDERS категориальные
    section("ORDERS — категориальные распределения")
    for col in ["delivery_service", "platform_id", "is_pod"]:
        d = (
            orders.group_by(col).agg(pl.len().alias("n")).sort("n", descending=True)
            .collect(engine="streaming").head(15)
        )
        log(f"\n{col}:")
        log_df(d)

    # --- ITEMS top categories
    section("ITEMS — топ категорий")
    items = scan("items")
    log_df(
        items.group_by("category_name")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .collect(engine="streaming")
        .head(20)
    )

    # save summary
    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    log(f"\nsaved: {OUT / 'summary.json'}")
    log("DONE")


if __name__ == "__main__":
    try:
        main()
    finally:
        _log_fh.close()
