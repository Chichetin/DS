"""
Препроцессинг 4 CSV в parquet (data/clean/).

Шаги:
- orders: drop broken rows + .unique() + parse 4 dates + drop is_pod / unnamed index → split train/test
- payments: drop NULL deliveryorder_id, склейка SBP+СБП, parse created_txtime → payments.parquet + payments_agg.parquet
- items: parse starttime/close_date, is_active, lifetime_days → items.parquet
- users: parse registrationtime/firstlistingdate, tenure_days, is_seller → users.parquet

Все pipeline'ы через polars streaming (`sink_parquet`/`engine="streaming"`) для 16GB RAM.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CLEAN = ROOT / "data" / "clean"
ART = ROOT / "artifacts" / "preprocess"
CLEAN.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)
LOG_FILE = ART / "run.log"

TEST_START = date(2025, 10, 1)
TEST_END = date(2025, 10, 10)

_log_lines: list[str] = []


def log(msg: str = "") -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else ""
    _log_lines.append(line)
    # Append-flush сразу — чтобы видеть прогресс в реальном времени
    LOG_FILE.write_text("\n".join(_log_lines), encoding="utf-8")


def flush_log() -> None:
    LOG_FILE.write_text("\n".join(_log_lines), encoding="utf-8")


def step_orders() -> None:
    """Через DuckDB: dedup + parse dates + split. Polars streaming OOMит на 39M×15 unique()."""
    import duckdb

    log("=" * 60)
    log("STEP 1/4: orders (via DuckDB)")
    src = DATA / "orders.csv"
    out_train = CLEAN / "orders_train.parquet"
    out_test = CLEAN / "orders_test.parquet"
    work_db = ART / "work.duckdb"
    tmp_dir = ART / "duckdb_tmp"
    tmp_dir.mkdir(exist_ok=True)

    if work_db.exists():
        work_db.unlink()

    con = duckdb.connect(str(work_db))
    con.execute("PRAGMA memory_limit='8GB'")
    con.execute(f"PRAGMA temp_directory='{tmp_dir.as_posix()}'")
    con.execute("PRAGMA threads=4")
    log("  duckdb connected, mem_limit=8GB")

    log("  reading CSV → orders_clean (DISTINCT, parse dates, drop broken)")
    con.execute(
        f"""
        CREATE TABLE orders_clean AS
        SELECT DISTINCT
            deliveryorder_id,
            CAST(order_create_date AS DATE) AS order_create_date,
            TRY_CAST(order_accept_date AS DATE) AS order_accept_date,
            item_id,
            buyer_terminal,
            seller_terminal,
            delivery_service,
            buyer_id,
            seller_id,
            order_price,
            city,
            platform_id,
            TRY_CAST(cancel_date AS TIMESTAMP) AS cancel_date,
            is_return
        FROM read_csv(
            '{src.as_posix()}',
            ignore_errors=true,
            header=true,
            null_padding=true
        )
        WHERE TRY_CAST(order_create_date AS DATE) IS NOT NULL
        """
    )
    n_clean = con.execute("SELECT COUNT(*) FROM orders_clean").fetchone()[0]
    log(f"  orders_clean: {n_clean:,} rows (after dedup + filter broken)")

    log(f"  COPY → {out_train.name}")
    con.execute(
        f"""
        COPY (
            SELECT * FROM orders_clean
            WHERE order_create_date < DATE '2025-10-01'
        ) TO '{out_train.as_posix()}' (FORMAT PARQUET, COMPRESSION 'zstd')
        """
    )
    log(f"  COPY → {out_test.name}")
    con.execute(
        f"""
        COPY (
            SELECT * EXCLUDE (is_return) FROM orders_clean
            WHERE order_create_date >= DATE '2025-10-01'
              AND order_create_date <= DATE '2025-10-10'
        ) TO '{out_test.as_posix()}' (FORMAT PARQUET, COMPRESSION 'zstd')
        """
    )
    con.close()

    n_train = pl.scan_parquet(out_train).select(pl.len()).collect().item()
    n_test = pl.scan_parquet(out_test).select(pl.len()).collect().item()
    log(f"  orders_train: {n_train:,} rows")
    log(f"  orders_test:  {n_test:,} rows")

    # Cleanup duckdb workfile
    work_db.unlink(missing_ok=True)
    for f in tmp_dir.glob("*"):
        f.unlink(missing_ok=True)
    tmp_dir.rmdir()
    log("  duckdb workfile removed")


def step_payments() -> None:
    log("=" * 60)
    log("STEP 2/4: payments")
    src = DATA / "payments.csv"
    out_full = CLEAN / "payments.parquet"
    out_agg = CLEAN / "payments_agg.parquet"

    base = (
        pl.scan_csv(src, ignore_errors=True)
        .drop("")  # безымянный индекс
        .filter(pl.col("deliveryorder_id").is_not_null())
        .with_columns(
            pl.col("created_txtime").str.to_datetime(strict=False),
            # Склейка SBP/СБП → SBP
            pl.when(pl.col("payment_method") == "СБП")
            .then(pl.lit("SBP"))
            .otherwise(pl.col("payment_method"))
            .alias("payment_method"),
        )
    )

    log(f"  sink → {out_full.name}")
    base.sink_parquet(out_full, compression="zstd")

    # Агрегаты на заказ — читаем уже из parquet (быстрее) и стримим обратно
    agg = (
        pl.scan_parquet(out_full)
        .group_by("deliveryorder_id")
        .agg(
            pl.len().alias("n_pay"),
            pl.col("amount").sum().alias("sum_amount"),
            pl.col("created_txtime").min().alias("min_txtime"),
            pl.col("created_txtime").max().alias("max_txtime"),
            # доминирующий метод оплаты — mode (берём первый при ничьей)
            pl.col("payment_method").mode().first().alias("dominant_payment_method"),
        )
    )
    log(f"  sink → {out_agg.name}")
    agg.sink_parquet(out_agg, compression="zstd")

    n_full = pl.scan_parquet(out_full).select(pl.len()).collect().item()
    n_agg = pl.scan_parquet(out_agg).select(pl.len()).collect().item()
    log(f"  payments:     {n_full:,} rows")
    log(f"  payments_agg: {n_agg:,} unique deliveryorder_id")


def step_items() -> None:
    log("=" * 60)
    log("STEP 3/4: items")
    src = DATA / "items.csv"
    out = CLEAN / "items.parquet"

    base = (
        pl.scan_csv(src, ignore_errors=True)
        .drop("")
        .with_columns(
            pl.col("starttime").str.to_datetime(strict=False),
            pl.col("close_date").str.to_datetime(strict=False),
        )
        .with_columns(
            pl.col("close_date").is_null().alias("is_active"),
            (
                (pl.col("close_date") - pl.col("starttime")).dt.total_days()
            ).alias("lifetime_days"),
        )
    )

    log(f"  sink → {out.name}")
    base.sink_parquet(out, compression="zstd")
    n = pl.scan_parquet(out).select(pl.len()).collect().item()
    log(f"  items: {n:,} rows")


def step_users() -> None:
    log("=" * 60)
    log("STEP 4/4: users")
    src = DATA / "users.csv"
    out = CLEAN / "users.parquet"

    base = (
        pl.scan_csv(src, ignore_errors=True)
        .drop("")
        .with_columns(
            pl.col("registrationtime").str.to_datetime(strict=False),
            pl.col("firstlistingdate").str.to_datetime(strict=False),
        )
        .with_columns(
            pl.col("firstlistingdate").is_not_null().alias("is_seller"),
            # tenure от регистрации до конца test-периода
            (
                pl.lit(TEST_END).cast(pl.Datetime("us"))
                - pl.col("registrationtime")
            ).dt.total_days().alias("tenure_days"),
        )
    )

    log(f"  sink → {out.name}")
    base.sink_parquet(out, compression="zstd")
    n = pl.scan_parquet(out).select(pl.len()).collect().item()
    log(f"  users: {n:,} rows")


def main() -> None:
    t0 = time.time()
    log(f"Preprocess start. polars={pl.__version__}")
    log(f"Output: {CLEAN}")
    try:
        step_orders()
        step_payments()
        step_items()
        step_users()
        log("=" * 60)
        log(f"DONE in {time.time() - t0:.1f}s")
    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        flush_log()
        raise
    finally:
        flush_log()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        main()
    except BaseException as exc:  # noqa: BLE001
        import traceback
        crash = ART / "crash.log"
        crash.write_text(
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        sys.exit(1)
