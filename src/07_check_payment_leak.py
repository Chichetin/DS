"""
Диагностика: содержит ли payments.parquet refund-проводки?

Проверки:
1. Распределение payment_method
2. Распределение знака amount (negative = refund?)
3. Коррелирует ли amount<0 с is_return через join к orders_train
4. Сколько платежей произошло ПОСЛЕ order_create_date (утечка в агрегатах)?
"""
from __future__ import annotations
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
ART = ROOT / "artifacts" / "features"
ART.mkdir(parents=True, exist_ok=True)
LOG = ART / "leak_check.log"

_lines: list[str] = []
def log(m: str = "") -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {m}" if m else ""
    _lines.append(line)
    LOG.write_text("\n".join(_lines), encoding="utf-8")


def main() -> None:
    t0 = time.time()
    log(f"Payment leak check. polars={pl.__version__}")

    # 1. payment_method распределение
    log("=" * 60)
    log("1) payment_method distribution (full payments.parquet)")
    pm = (
        pl.scan_parquet(CLEAN / "payments.parquet")
        .group_by("payment_method")
        .agg(pl.len().alias("n"), pl.col("amount").sum().alias("sum_amt"),
             pl.col("amount").min().alias("min_amt"), pl.col("amount").max().alias("max_amt"))
        .sort("n", descending=True)
        .collect()
    )
    for row in pm.iter_rows(named=True):
        log(f"  {row['payment_method']:<20} n={row['n']:>12,} sum={row['sum_amt']:>16,.0f} "
            f"min={row['min_amt']:>10,.0f} max={row['max_amt']:>14,.0f}")

    # 2. знак amount
    log("=" * 60)
    log("2) Знак amount (negative = refund?)")
    sign = (
        pl.scan_parquet(CLEAN / "payments.parquet")
        .with_columns(
            pl.when(pl.col("amount") < 0).then(pl.lit("neg"))
            .when(pl.col("amount") == 0).then(pl.lit("zero"))
            .otherwise(pl.lit("pos")).alias("sign")
        )
        .group_by("sign")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .collect()
    )
    for row in sign.iter_rows(named=True):
        log(f"  {row['sign']:<5} n={row['n']:>12,}")

    # 3. payments после order_create_date — утечка в агрегатах
    log("=" * 60)
    log("3) Сколько платежей произошло ПОСЛЕ order_create_date (sample на train)")
    pay = pl.scan_parquet(CLEAN / "payments.parquet").select(
        "deliveryorder_id",
        pl.col("created_txtime").cast(pl.Date).alias("pay_date"),
        "amount",
    )
    orders = pl.scan_parquet(CLEAN / "orders_train.parquet").select(
        "deliveryorder_id", "order_create_date", "is_return"
    )
    joined = pay.join(orders, on="deliveryorder_id", how="inner")
    rel = (
        joined
        .with_columns(
            pl.when(pl.col("pay_date") < pl.col("order_create_date")).then(pl.lit("before"))
            .when(pl.col("pay_date") == pl.col("order_create_date")).then(pl.lit("same_day"))
            .otherwise(pl.lit("after")).alias("rel")
        )
        .group_by("rel")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .collect()
    )
    for row in rel.iter_rows(named=True):
        log(f"  {row['rel']:<10} n={row['n']:>12,}")

    # 4. Корреляция: late_payments с is_return
    log("=" * 60)
    log("4) Доля поздних платежей (pay_date > order_create_date+7d) vs is_return")
    cor = (
        joined
        .with_columns(
            ((pl.col("pay_date") - pl.col("order_create_date")).dt.total_days() > 7)
            .cast(pl.Int8).alias("is_late_pay")
        )
        .group_by("deliveryorder_id")
        .agg(
            pl.col("is_late_pay").max().alias("has_late_pay"),
            pl.col("is_return").first().alias("is_return"),
        )
        .group_by("has_late_pay", "is_return")
        .agg(pl.len().alias("n"))
        .sort("has_late_pay", "is_return")
        .collect()
    )
    for row in cor.iter_rows(named=True):
        log(f"  has_late_pay={row['has_late_pay']} is_return={row['is_return']} n={row['n']:>12,}")

    log("=" * 60)
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        main()
    except BaseException as exc:
        import traceback
        crash = ART / "leak_check_crash.log"
        crash.write_text(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", encoding="utf-8")
        raise
