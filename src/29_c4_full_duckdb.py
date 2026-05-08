"""
Шаг F3'. Альтернатива 17_features_c4.py для full-режима через DuckDB.

Join'ы 30M × 21M te_seller_daily в polars могут OOM. DuckDB стабильнее.

Логика идентична 17_features_c4.py:
  c3_*_full ⋈ orders_train (для seller_id by deliveryorder_id+item_id)
            ⋈ te_seller_daily ON (seller_id, order_create_date)
            ⋈ microcat_price_stats ON microcat_name
  → производные: dynamics_diff, price_to_microcat_median, is_first_sale
  → drop: seller_id, microcat_mean_price, microcat_price_count

Артефакты: c4_train_full.parquet, c4_holdout_full.parquet
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
LOG = ART / "c4_full_duckdb.log"

GLOBAL_MEAN = 0.0456
TARGETS = set(os.environ.get("TARGETS", "train,holdout").split(","))

_lines: list[str] = []
def log(m: str = "") -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {m}" if m else ""
    _lines.append(line)
    LOG.write_text("\n".join(_lines), encoding="utf-8")


def build(con: duckdb.DuckDBPyConnection, name: str) -> None:
    src = (FEAT / f"c3_{name}_full.parquet").as_posix()
    out = (FEAT / f"c4_{name}_full.parquet").as_posix()
    orders = (CLEAN / "orders_train.parquet").as_posix()
    te_seller = (FEAT / "te_seller_daily.parquet").as_posix()
    price_stats = (FEAT / "microcat_price_stats.parquet").as_posix()

    log("=" * 60)
    log(f"Build c4_{name}_full ← {src}")

    sql = f"""
    COPY (
        WITH base AS (
            SELECT c3.*, o.seller_id
            FROM read_parquet('{src}') c3
            LEFT JOIN (
                SELECT deliveryorder_id, item_id, seller_id
                FROM read_parquet('{orders}')
            ) o
            USING (deliveryorder_id, item_id)
        ),
        with_seller_te AS (
            SELECT b.*,
                   COALESCE(ts.seller_te, {GLOBAL_MEAN}) AS seller_te,
                   CAST(COALESCE(ts.seller_te_count, 0) AS UINTEGER) AS seller_te_count
            FROM base b
            LEFT JOIN read_parquet('{te_seller}') ts
              ON b.seller_id = ts.seller_id
             AND b.order_create_date = ts.order_create_date
        ),
        with_price AS (
            SELECT wt.*,
                   ps.microcat_median_price
            FROM with_seller_te wt
            LEFT JOIN (
                SELECT microcat_name, microcat_median_price
                FROM read_parquet('{price_stats}')
            ) ps
            USING (microcat_name)
        )
        SELECT
            -- ID + базовые поля c1
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

            -- c2 history фичи
            buyer_past_orders, buyer_past_returns,
            buyer_past_30d_orders, buyer_past_7d_orders,
            buyer_past_return_rate, buyer_past_avg_price,
            buyer_days_since_last_order,
            buyer_past_30d_return_rate, buyer_past_7d_return_rate,
            seller_past_orders, seller_past_returns,
            seller_past_30d_orders, seller_past_7d_orders,
            seller_past_return_rate, seller_past_avg_price,
            seller_days_since_last_order,
            seller_past_30d_return_rate, seller_past_7d_return_rate,
            item_past_orders, item_past_returns,
            item_past_30d_orders, item_past_7d_orders,
            item_past_return_rate, item_past_avg_price,
            item_days_since_last_order,
            item_past_30d_return_rate, item_past_7d_return_rate,

            -- c3 TE фичи (microcat, city)
            microcat_te, microcat_te_count,
            city_te, city_te_count,

            -- c4: новые
            seller_te, seller_te_count,
            microcat_median_price,
            (buyer_past_30d_return_rate - buyer_past_return_rate) AS buyer_dynamics_30d_diff,
            (buyer_past_7d_return_rate  - buyer_past_return_rate) AS buyer_dynamics_7d_diff,
            (seller_past_30d_return_rate - seller_past_return_rate) AS seller_dynamics_30d_diff,
            (seller_past_7d_return_rate  - seller_past_return_rate) AS seller_dynamics_7d_diff,
            CASE WHEN microcat_median_price > 0
                 THEN order_price / microcat_median_price
                 ELSE NULL END AS price_to_microcat_median,
            CASE WHEN microcat_median_price > 0
                 THEN LN(order_price + 1.0) - LN(microcat_median_price + 1.0)
                 ELSE NULL END AS log_price_to_microcat_median,
            CAST(CASE WHEN item_past_orders = 0 THEN 1 ELSE 0 END AS TINYINT) AS is_first_sale
        FROM with_price
    ) TO '{out}' (FORMAT PARQUET, COMPRESSION 'zstd', ROW_GROUP_SIZE 100000)
    """
    con.execute(sql)
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out}')").fetchone()[0]
    log(f"  c4_{name}_full: {n:,} rows")


def main() -> None:
    t0 = time.time()
    log(f"C4 FULL via DuckDB. duckdb={duckdb.__version__}")
    log(f"  TARGETS={TARGETS}")

    tmp_dir = ART / "duckdb_tmp_c4_full"
    tmp_dir.mkdir(exist_ok=True)
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA memory_limit='10GB'")
    con.execute(f"PRAGMA temp_directory='{tmp_dir.as_posix()}'")
    con.execute("PRAGMA threads=4")
    log("  duckdb connected, mem_limit=10GB")

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
        crash = ART / "c4_full_duckdb_crash.log"
        crash.write_text(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", encoding="utf-8")
        raise
