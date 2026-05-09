"""
Шаг C2/часть 1: rolling-агрегаты по buyer_id / seller_id / item_id (без time-leak).

Логика:
  Для каждой сущности E и даты D:
    past_orders        = COUNT по [-inf, D-1]
    past_returns       = SUM(is_return) по [-inf, D-1]
    past_price_sum     = SUM(order_price) по [-inf, D-1]
    last_order_date    = MAX(order_create_date) по [-inf, D-1]
    past_30d_orders    = COUNT по [D-30, D-1]
    past_30d_returns   = SUM(is_return) по [D-30, D-1]
    past_7d_orders     = COUNT по [D-7, D-1]
    past_7d_returns    = SUM(is_return) по [D-7, D-1]

Делается через DuckDB: groupby(E, date) → SUM() OVER (PARTITION BY E ORDER BY date) c
обычным cum-окном для all-time и RANGE-окном с интервалом для 7d/30d.

Источник: ПОЛНЫЙ data/clean/orders_train.parquet (33M, не сэмпл).

Test snapshot: для каждой сущности — счётчики по всему train + счётчики по последним
30/7 дням train (т.е. [2025-09-01, 2025-09-30] и [2025-09-24, 2025-09-30]). Это
"что было известно на старте test-периода".
"""
from __future__ import annotations
from pathlib import Path

import duckdb

from _log import Logger

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "data" / "features"
ART = ROOT / "artifacts" / "features"
OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)

log = Logger(ART / "history.log")


ENTITIES = [
    ("buyer_id", "buyer"),
    ("seller_id", "seller"),
    ("item_id", "item"),
]


def build_for_entity(con: duckdb.DuckDBPyConnection, key_col: str, short: str) -> None:
    """Строит daily и snapshot parquet для одной сущности."""
    log.step(f"Entity: {key_col}")

    daily_out = OUT / f"hist_{short}_daily.parquet"
    snap_out = OUT / f"hist_{short}_snapshot.parquet"

    src = (CLEAN / "orders_train.parquet").as_posix()

    # Daily aggregates → cumulative + rolling 30d/7d → past = cum - today
    log(f"  build daily → {daily_out.name}")
    con.execute(
        f"""
        COPY (
            WITH daily AS (
                SELECT {key_col} AS k,
                       order_create_date AS d,
                       COUNT(*) AS d_orders,
                       SUM(CAST(is_return AS INT)) AS d_returns,
                       SUM(order_price) AS d_price_sum
                FROM read_parquet('{src}')
                WHERE {key_col} IS NOT NULL
                GROUP BY k, d
            ),
            cum AS (
                SELECT k, d, d_orders, d_returns, d_price_sum,
                       SUM(d_orders)    OVER w_all AS cum_orders,
                       SUM(d_returns)   OVER w_all AS cum_returns,
                       SUM(d_price_sum) OVER w_all AS cum_price_sum,
                       LAG(d)           OVER w_all AS prev_date,
                       COALESCE(SUM(d_orders)  OVER w_30d, 0) AS w30_orders,
                       COALESCE(SUM(d_returns) OVER w_30d, 0) AS w30_returns,
                       COALESCE(SUM(d_orders)  OVER w_7d,  0) AS w7_orders,
                       COALESCE(SUM(d_returns) OVER w_7d,  0) AS w7_returns
                FROM daily
                WINDOW
                  w_all AS (PARTITION BY k ORDER BY d),
                  w_30d AS (PARTITION BY k ORDER BY d
                            RANGE BETWEEN INTERVAL 30 DAYS PRECEDING
                                      AND INTERVAL 1 DAY PRECEDING),
                  w_7d  AS (PARTITION BY k ORDER BY d
                            RANGE BETWEEN INTERVAL 7 DAYS PRECEDING
                                      AND INTERVAL 1 DAY PRECEDING)
            )
            SELECT k AS {key_col},
                   d AS order_create_date,
                   (cum_orders    - d_orders)    AS {short}_past_orders,
                   (cum_returns   - d_returns)   AS {short}_past_returns,
                   (cum_price_sum - d_price_sum) AS {short}_past_price_sum,
                   prev_date AS {short}_last_order_date,
                   w30_orders   AS {short}_past_30d_orders,
                   w30_returns  AS {short}_past_30d_returns,
                   w7_orders    AS {short}_past_7d_orders,
                   w7_returns   AS {short}_past_7d_returns
            FROM cum
        ) TO '{daily_out.as_posix()}' (FORMAT PARQUET, COMPRESSION 'zstd')
        """
    )
    n_daily = con.execute(f"SELECT COUNT(*) FROM read_parquet('{daily_out.as_posix()}')").fetchone()[0]
    log(f"  daily rows: {n_daily:,}")

    # Snapshot: для каждой сущности — счётчики на конец train + 30d/7d
    # (test начинается 2025-10-01, окно 30d = [2025-09-01, 2025-09-30],
    #  окно 7d = [2025-09-24, 2025-09-30])
    log(f"  build snapshot → {snap_out.name}")
    con.execute(
        f"""
        COPY (
            SELECT {key_col} AS {key_col},
                   COUNT(*) AS {short}_past_orders,
                   SUM(CAST(is_return AS INT)) AS {short}_past_returns,
                   SUM(order_price) AS {short}_past_price_sum,
                   MAX(order_create_date) AS {short}_last_order_date,
                   SUM(CASE WHEN order_create_date >= DATE '2025-09-01' THEN 1 ELSE 0 END)
                       AS {short}_past_30d_orders,
                   SUM(CASE WHEN order_create_date >= DATE '2025-09-01'
                            THEN CAST(is_return AS INT) ELSE 0 END)
                       AS {short}_past_30d_returns,
                   SUM(CASE WHEN order_create_date >= DATE '2025-09-24' THEN 1 ELSE 0 END)
                       AS {short}_past_7d_orders,
                   SUM(CASE WHEN order_create_date >= DATE '2025-09-24'
                            THEN CAST(is_return AS INT) ELSE 0 END)
                       AS {short}_past_7d_returns
            FROM read_parquet('{src}')
            WHERE {key_col} IS NOT NULL
            GROUP BY {key_col}
        ) TO '{snap_out.as_posix()}' (FORMAT PARQUET, COMPRESSION 'zstd')
        """
    )
    n_snap = con.execute(f"SELECT COUNT(*) FROM read_parquet('{snap_out.as_posix()}')").fetchone()[0]
    log(f"  snapshot rows: {n_snap:,}")


def main() -> None:
    log(f"History aggregates start. duckdb={duckdb.__version__}")

    tmp_dir = ART / "duckdb_tmp_history"
    tmp_dir.mkdir(exist_ok=True)
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA memory_limit='10GB'")
    con.execute(f"PRAGMA temp_directory='{tmp_dir.as_posix()}'")
    con.execute("PRAGMA threads=4")
    log("  duckdb connected, mem_limit=10GB")

    for key_col, short in ENTITIES:
        build_for_entity(con, key_col, short)

    con.close()
    # cleanup tmp
    for f in tmp_dir.glob("*"):
        f.unlink(missing_ok=True)
    tmp_dir.rmdir()

    log.done()


if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        main()
    except BaseException as exc:
        log.crash(exc, ART / "history_crash.log")
        raise
