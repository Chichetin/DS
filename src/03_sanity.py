"""Sanity check для clean parquet'ов: schema, dtypes, row counts, ожидаемые границы."""
from __future__ import annotations
from datetime import date
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "artifacts" / "preprocess" / "sanity.txt"

lines: list[str] = []
def w(s: str = "") -> None:
    lines.append(s)


def section(name: str) -> None:
    w("")
    w("=" * 70)
    w(name)
    w("=" * 70)


def check_orders(path: Path, expect_is_return: bool, lo: date, hi: date) -> None:
    section(path.name)
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    w(f"columns: {list(schema.names())}")
    w(f"dtypes:  {[str(t) for t in schema.dtypes()]}")
    w(f"rows:    {lf.select(pl.len()).collect().item():,}")

    # date bounds
    bounds = lf.select(
        pl.col("order_create_date").min().alias("min"),
        pl.col("order_create_date").max().alias("max"),
        pl.col("order_create_date").is_null().sum().alias("nulls"),
    ).collect().to_dicts()[0]
    w(f"order_create_date: min={bounds['min']}, max={bounds['max']}, nulls={bounds['nulls']:,}")
    assert bounds["min"] is not None and bounds["max"] is not None
    assert bounds["min"] >= lo, f"date below expected: {bounds['min']} < {lo}"
    assert bounds["max"] <= hi, f"date above expected: {bounds['max']} > {hi}"
    assert bounds["nulls"] == 0, "broken rows leaked"

    # is_return presence
    has_ret = "is_return" in schema.names()
    w(f"is_return present: {has_ret}")
    assert has_ret == expect_is_return, "is_return column mismatch"
    if expect_is_return:
        ret = lf.select(
            pl.col("is_return").sum().alias("true"),
            (~pl.col("is_return")).sum().alias("false"),
            pl.col("is_return").is_null().sum().alias("null"),
        ).collect().to_dicts()[0]
        total = ret["true"] + ret["false"] + ret["null"]
        rate = ret["true"] / (ret["true"] + ret["false"]) if (ret["true"] + ret["false"]) else 0.0
        w(f"is_return: true={ret['true']:,}, false={ret['false']:,}, null={ret['null']:,}, rate={rate:.4%}")

    # dedup sanity: пары (deliveryorder_id, item_id) теперь должны быть уникальны
    dups = lf.group_by(["deliveryorder_id", "item_id"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1).select(pl.len()).collect().item()
    w(f"duplicate (deliveryorder_id,item_id) pairs: {dups:,}")


def check_payments(path: Path) -> None:
    section(path.name)
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    w(f"columns: {list(schema.names())}")
    w(f"dtypes:  {[str(t) for t in schema.dtypes()]}")
    w(f"rows:    {lf.select(pl.len()).collect().item():,}")

    nulls = lf.select(pl.col("deliveryorder_id").is_null().sum()).collect().item()
    w(f"deliveryorder_id NULL: {nulls:,}")
    assert nulls == 0, "NULL deliveryorder_id leaked"

    # СБП не должен оставаться
    pm = lf.group_by("payment_method").agg(pl.len().alias("n")).sort("n", descending=True).collect()
    w("payment_method counts:")
    for r in pm.iter_rows(named=True):
        w(f"  {r['payment_method']}: {r['n']:,}")
    assert "СБП" not in pm["payment_method"].to_list(), "СБП still present"

    # created_txtime parsed
    bounds = lf.select(
        pl.col("created_txtime").min().alias("min"),
        pl.col("created_txtime").max().alias("max"),
    ).collect().to_dicts()[0]
    w(f"created_txtime: min={bounds['min']}, max={bounds['max']}")


def check_payments_agg(path: Path) -> None:
    section(path.name)
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    w(f"columns: {list(schema.names())}")
    w(f"dtypes:  {[str(t) for t in schema.dtypes()]}")
    w(f"rows:    {lf.select(pl.len()).collect().item():,}")
    stats = lf.select(
        pl.col("n_pay").min().alias("npay_min"),
        pl.col("n_pay").max().alias("npay_max"),
        pl.col("n_pay").mean().alias("npay_mean"),
        pl.col("sum_amount").min().alias("amt_min"),
        pl.col("sum_amount").max().alias("amt_max"),
        pl.col("dominant_payment_method").is_null().sum().alias("dom_null"),
    ).collect().to_dicts()[0]
    w(f"n_pay: min={stats['npay_min']}, max={stats['npay_max']}, mean={stats['npay_mean']:.3f}")
    w(f"sum_amount: min={stats['amt_min']:.2f}, max={stats['amt_max']:.2f}")
    w(f"dominant_payment_method NULL: {stats['dom_null']:,}")


def check_items(path: Path) -> None:
    section(path.name)
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    w(f"columns: {list(schema.names())}")
    w(f"dtypes:  {[str(t) for t in schema.dtypes()]}")
    w(f"rows:    {lf.select(pl.len()).collect().item():,}")
    stats = lf.select(
        pl.col("starttime").min().alias("st_min"),
        pl.col("starttime").max().alias("st_max"),
        pl.col("close_date").is_null().sum().alias("close_null"),
        pl.col("is_active").sum().alias("active"),
        pl.col("lifetime_days").min().alias("life_min"),
        pl.col("lifetime_days").max().alias("life_max"),
        pl.col("lifetime_days").is_null().sum().alias("life_null"),
    ).collect().to_dicts()[0]
    w(f"starttime: {stats['st_min']} … {stats['st_max']}")
    w(f"close_date NULL = is_active count: close_null={stats['close_null']:,}, is_active={stats['active']:,}")
    w(f"lifetime_days: min={stats['life_min']}, max={stats['life_max']}, null={stats['life_null']:,}")
    assert stats["close_null"] == stats["active"], "is_active != close_date.is_null()"


def check_users(path: Path) -> None:
    section(path.name)
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    w(f"columns: {list(schema.names())}")
    w(f"dtypes:  {[str(t) for t in schema.dtypes()]}")
    w(f"rows:    {lf.select(pl.len()).collect().item():,}")
    stats = lf.select(
        pl.col("registrationtime").min().alias("reg_min"),
        pl.col("registrationtime").max().alias("reg_max"),
        pl.col("firstlistingdate").is_null().sum().alias("fl_null"),
        pl.col("is_seller").sum().alias("sellers"),
        pl.col("tenure_days").min().alias("ten_min"),
        pl.col("tenure_days").max().alias("ten_max"),
    ).collect().to_dicts()[0]
    w(f"registrationtime: {stats['reg_min']} … {stats['reg_max']}")
    w(f"is_seller=true: {stats['sellers']:,}, firstlistingdate NULL: {stats['fl_null']:,}")
    w(f"tenure_days: min={stats['ten_min']}, max={stats['ten_max']}")


def main() -> None:
    check_orders(CLEAN / "orders_train.parquet", expect_is_return=True,
                 lo=date(2025, 7, 1), hi=date(2025, 9, 30))
    check_orders(CLEAN / "orders_test.parquet", expect_is_return=False,
                 lo=date(2025, 10, 1), hi=date(2025, 10, 10))
    check_payments(CLEAN / "payments.parquet")
    check_payments_agg(CLEAN / "payments_agg.parquet")
    check_items(CLEAN / "items.parquet")
    check_users(CLEAN / "users.parquet")
    w("")
    w("=" * 70)
    w("ALL CHECKS PASSED")
    OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        import traceback
        lines.append(f"\nFAILED: {type(exc).__name__}: {exc}")
        lines.append(traceback.format_exc())
        OUT.write_text("\n".join(lines), encoding="utf-8")
        raise
