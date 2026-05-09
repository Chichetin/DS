"""
Шаг C4/часть 1: target encoding для seller_id (high-cardinality, без time-leak).

Зеркало 11_target_encoding.py: bayesian smoothing с GLOBAL_MEAN=0.0456, ALPHA=100.
Источник — ПОЛНЫЙ orders_train.parquet (33M).

Артефакты в data/features/:
  te_seller_daily.parquet     — (seller_id, order_create_date) → seller_te, te_count
  te_seller_snapshot.parquet  — (seller_id) → seller_te, count на конец train
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

GLOBAL_MEAN = 0.0456
ALPHA = 100.0

log = Logger(ART / "te_seller.log")


def main() -> None:
    log(f"TE seller start. duckdb={duckdb.__version__}")
    log(f"  GLOBAL_MEAN={GLOBAL_MEAN}, ALPHA={ALPHA}")

    tmp_dir = ART / "duckdb_tmp_te_seller"
    tmp_dir.mkdir(exist_ok=True)
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA memory_limit='10GB'")
    con.execute(f"PRAGMA temp_directory='{tmp_dir.as_posix()}'")
    con.execute("PRAGMA threads=4")
    log("  duckdb connected, mem_limit=10GB")

    orders = (CLEAN / "orders_train.parquet").as_posix()
    daily_out = OUT / "te_seller_daily.parquet"
    snap_out = OUT / "te_seller_snapshot.parquet"

    log.step(f"build daily → {daily_out.name}")
    con.execute(
        f"""
        COPY (
            WITH src AS (
                SELECT seller_id AS k,
                       order_create_date AS d,
                       is_return
                FROM read_parquet('{orders}')
                WHERE seller_id IS NOT NULL
            ),
            daily AS (
                SELECT k, d,
                       COUNT(*) AS d_orders,
                       SUM(CAST(is_return AS INT)) AS d_returns
                FROM src
                GROUP BY k, d
            ),
            cum AS (
                SELECT k, d, d_orders, d_returns,
                       SUM(d_orders)  OVER w AS cum_orders,
                       SUM(d_returns) OVER w AS cum_returns
                FROM daily
                WINDOW w AS (PARTITION BY k ORDER BY d)
            )
            SELECT k AS seller_id,
                   d AS order_create_date,
                   ((cum_returns - d_returns) + {ALPHA} * {GLOBAL_MEAN})
                       / ((cum_orders - d_orders) + {ALPHA})
                       AS seller_te,
                   (cum_orders - d_orders) AS seller_te_count
            FROM cum
        ) TO '{daily_out.as_posix()}' (FORMAT PARQUET, COMPRESSION 'zstd')
        """
    )
    n_daily = con.execute(f"SELECT COUNT(*) FROM read_parquet('{daily_out.as_posix()}')").fetchone()[0]
    log(f"  daily rows: {n_daily:,}")

    log.step(f"build snapshot → {snap_out.name}")
    con.execute(
        f"""
        COPY (
            SELECT seller_id,
                   (SUM(CAST(is_return AS INT)) + {ALPHA} * {GLOBAL_MEAN})
                       / (COUNT(*) + {ALPHA})
                       AS seller_te,
                   COUNT(*) AS seller_te_count
            FROM read_parquet('{orders}')
            WHERE seller_id IS NOT NULL
            GROUP BY seller_id
        ) TO '{snap_out.as_posix()}' (FORMAT PARQUET, COMPRESSION 'zstd')
        """
    )
    n_snap = con.execute(f"SELECT COUNT(*) FROM read_parquet('{snap_out.as_posix()}')").fetchone()[0]
    log(f"  snapshot rows: {n_snap:,}")

    con.close()
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
        log.crash(exc, ART / "te_seller_crash.log")
        raise
