"""
Шаг F2'. Альтернатива 09_features_c2.py для full-режима через DuckDB.

Polars sink_parquet с 4 joins на 30M×70M hist OOMит (segfault на 16GB).
DuckDB же стабилен — splills temp на диск, memory_limit гарантирует не выходить за лимит.

Логика идентична 09_features_c2.py:
  c1_train_full ⋈ orders (buyer_id, seller_id by deliveryorder_id+item_id)
                ⋈ hist_buyer_daily  ON (buyer_id, order_create_date)
                ⋈ hist_seller_daily ON (seller_id, order_create_date)
                ⋈ hist_item_daily   ON (item_id, order_create_date)
  → производные: rate, days_since_last, 30d/7d rates
  → drop: buyer_id, seller_id, past_price_sum, last_order_date, *_30d_returns, *_7d_returns

Артефакты: c2_train_full.parquet, c2_holdout_full.parquet
"""
from __future__ import annotations
import os
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
FEAT = ROOT / "data" / "features"
ART = ROOT / "artifacts" / "features"
LOG = ART / "c2_full_duckdb.log"

TARGETS = set(os.environ.get("TARGETS", "train,holdout").split(","))

_lines: list[str] = []
def log(m: str = "") -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {m}" if m else ""
    _lines.append(line)
    LOG.write_text("\n".join(_lines), encoding="utf-8")


def build(con: duckdb.DuckDBPyConnection, name: str) -> None:
    src = (FEAT / f"c1_{name}_full.parquet").as_posix()
    out = (FEAT / f"c2_{name}_full.parquet").as_posix()
    orders = (CLEAN / "orders_train.parquet").as_posix()
    hist_b = (FEAT / "hist_buyer_daily.parquet").as_posix()
    hist_s = (FEAT / "hist_seller_daily.parquet").as_posix()
    hist_i = (FEAT / "hist_item_daily.parquet").as_posix()

    log("=" * 60)
    log(f"Build c2_{name}_full ← {src}")

    sql = f"""
    COPY (
        WITH base AS (
            SELECT
                c1.*,
                o.buyer_id,
                o.seller_id
            FROM read_parquet('{src}') c1
            LEFT JOIN (
                SELECT deliveryorder_id, item_id, buyer_id, seller_id
                FROM read_parquet('{orders}')
            ) o
            USING (deliveryorder_id, item_id)
        ),
        with_buyer AS (
            SELECT b.*,
                   hb.buyer_past_orders,
                   hb.buyer_past_returns,
                   hb.buyer_past_price_sum,
                   hb.buyer_last_order_date,
                   hb.buyer_past_30d_orders,
                   hb.buyer_past_30d_returns,
                   hb.buyer_past_7d_orders,
                   hb.buyer_past_7d_returns
            FROM base b
            LEFT JOIN read_parquet('{hist_b}') hb
              ON b.buyer_id = hb.buyer_id
             AND b.order_create_date = hb.order_create_date
        ),
        with_seller AS (
            SELECT wb.*,
                   hs.seller_past_orders,
                   hs.seller_past_returns,
                   hs.seller_past_price_sum,
                   hs.seller_last_order_date,
                   hs.seller_past_30d_orders,
                   hs.seller_past_30d_returns,
                   hs.seller_past_7d_orders,
                   hs.seller_past_7d_returns
            FROM with_buyer wb
            LEFT JOIN read_parquet('{hist_s}') hs
              ON wb.seller_id = hs.seller_id
             AND wb.order_create_date = hs.order_create_date
        ),
        with_item AS (
            SELECT ws.*,
                   hi.item_past_orders,
                   hi.item_past_returns,
                   hi.item_past_price_sum,
                   hi.item_last_order_date,
                   hi.item_past_30d_orders,
                   hi.item_past_30d_returns,
                   hi.item_past_7d_orders,
                   hi.item_past_7d_returns
            FROM with_seller ws
            LEFT JOIN read_parquet('{hist_i}') hi
              ON ws.item_id = hi.item_id
             AND ws.order_create_date = hi.order_create_date
        )
        SELECT
            deliveryorder_id, item_id, order_create_date,
            order_dow, order_dom, is_weekend,
            order_price, log_order_price,
            delivery_service, platform_id, city,
            category_name, microcat_name,
            item_lifetime_at_order, is_active_at_order,
            buyer_gender, buyer_iscompany, buyer_isblocked,
            buyer_tenure_days, buyer_is_seller,
            seller_iscompany, seller_isblocked, seller_tenure_days,
            n_pay, pay_sum_amount, dominant_payment_method, pay_to_price_ratio,
            is_return,

            -- buyer
            CAST(COALESCE(buyer_past_orders,  0) AS UINTEGER) AS buyer_past_orders,
            CAST(COALESCE(buyer_past_returns, 0) AS UINTEGER) AS buyer_past_returns,
            CAST(COALESCE(buyer_past_30d_orders, 0) AS UINTEGER) AS buyer_past_30d_orders,
            CAST(COALESCE(buyer_past_7d_orders,  0) AS UINTEGER) AS buyer_past_7d_orders,
            CASE WHEN COALESCE(buyer_past_orders, 0) > 0
                 THEN buyer_past_returns::DOUBLE / buyer_past_orders
                 ELSE NULL END AS buyer_past_return_rate,
            CASE WHEN COALESCE(buyer_past_orders, 0) > 0
                 THEN buyer_past_price_sum / buyer_past_orders
                 ELSE NULL END AS buyer_past_avg_price,
            CASE WHEN buyer_last_order_date IS NOT NULL
                 THEN CAST(DATE_DIFF('day', buyer_last_order_date, order_create_date) AS INTEGER)
                 ELSE NULL END AS buyer_days_since_last_order,
            CASE WHEN COALESCE(buyer_past_30d_orders, 0) > 0
                 THEN buyer_past_30d_returns::DOUBLE / buyer_past_30d_orders
                 ELSE NULL END AS buyer_past_30d_return_rate,
            CASE WHEN COALESCE(buyer_past_7d_orders, 0) > 0
                 THEN buyer_past_7d_returns::DOUBLE / buyer_past_7d_orders
                 ELSE NULL END AS buyer_past_7d_return_rate,

            -- seller
            CAST(COALESCE(seller_past_orders,  0) AS UINTEGER) AS seller_past_orders,
            CAST(COALESCE(seller_past_returns, 0) AS UINTEGER) AS seller_past_returns,
            CAST(COALESCE(seller_past_30d_orders, 0) AS UINTEGER) AS seller_past_30d_orders,
            CAST(COALESCE(seller_past_7d_orders,  0) AS UINTEGER) AS seller_past_7d_orders,
            CASE WHEN COALESCE(seller_past_orders, 0) > 0
                 THEN seller_past_returns::DOUBLE / seller_past_orders
                 ELSE NULL END AS seller_past_return_rate,
            CASE WHEN COALESCE(seller_past_orders, 0) > 0
                 THEN seller_past_price_sum / seller_past_orders
                 ELSE NULL END AS seller_past_avg_price,
            CASE WHEN seller_last_order_date IS NOT NULL
                 THEN CAST(DATE_DIFF('day', seller_last_order_date, order_create_date) AS INTEGER)
                 ELSE NULL END AS seller_days_since_last_order,
            CASE WHEN COALESCE(seller_past_30d_orders, 0) > 0
                 THEN seller_past_30d_returns::DOUBLE / seller_past_30d_orders
                 ELSE NULL END AS seller_past_30d_return_rate,
            CASE WHEN COALESCE(seller_past_7d_orders, 0) > 0
                 THEN seller_past_7d_returns::DOUBLE / seller_past_7d_orders
                 ELSE NULL END AS seller_past_7d_return_rate,

            -- item
            CAST(COALESCE(item_past_orders,  0) AS UINTEGER) AS item_past_orders,
            CAST(COALESCE(item_past_returns, 0) AS UINTEGER) AS item_past_returns,
            CAST(COALESCE(item_past_30d_orders, 0) AS UINTEGER) AS item_past_30d_orders,
            CAST(COALESCE(item_past_7d_orders,  0) AS UINTEGER) AS item_past_7d_orders,
            CASE WHEN COALESCE(item_past_orders, 0) > 0
                 THEN item_past_returns::DOUBLE / item_past_orders
                 ELSE NULL END AS item_past_return_rate,
            CASE WHEN COALESCE(item_past_orders, 0) > 0
                 THEN item_past_price_sum / item_past_orders
                 ELSE NULL END AS item_past_avg_price,
            CASE WHEN item_last_order_date IS NOT NULL
                 THEN CAST(DATE_DIFF('day', item_last_order_date, order_create_date) AS INTEGER)
                 ELSE NULL END AS item_days_since_last_order,
            CASE WHEN COALESCE(item_past_30d_orders, 0) > 0
                 THEN item_past_30d_returns::DOUBLE / item_past_30d_orders
                 ELSE NULL END AS item_past_30d_return_rate,
            CASE WHEN COALESCE(item_past_7d_orders, 0) > 0
                 THEN item_past_7d_returns::DOUBLE / item_past_7d_orders
                 ELSE NULL END AS item_past_7d_return_rate
        FROM with_item
    ) TO '{out}' (FORMAT PARQUET, COMPRESSION 'zstd', ROW_GROUP_SIZE 100000)
    """
    con.execute(sql)
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out}')").fetchone()[0]
    log(f"  c2_{name}_full: {n:,} rows")


def main() -> None:
    t0 = time.time()
    log(f"C2 FULL via DuckDB. duckdb={duckdb.__version__}")
    log(f"  TARGETS={TARGETS}")

    tmp_dir = ART / "duckdb_tmp_c2_full"
    tmp_dir.mkdir(exist_ok=True)
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA memory_limit='10GB'")
    con.execute(f"PRAGMA temp_directory='{tmp_dir.as_posix()}'")
    con.execute("PRAGMA threads=4")
    log("  duckdb connected, mem_limit=10GB, threads=4")

    if "train" in TARGETS:
        build(con, "train")
    if "holdout" in TARGETS:
        build(con, "holdout")

    con.close()
    for f in tmp_dir.glob("*"):
        f.unlink(missing_ok=True)
    tmp_dir.rmdir()

    log("=" * 60)
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        main()
    except BaseException as exc:
        import traceback
        crash = ART / "c2_full_duckdb_crash.log"
        crash.write_text(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", encoding="utf-8")
        raise
