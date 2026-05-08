"""
Шаг C3/часть 1: target encoding для microcat_name и city (без time-leak).

Логика (как у history): для каждой категории K и даты D:
    past_orders  = COUNT по [-inf, D-1]
    past_returns = SUM(is_return) по [-inf, D-1]
    te           = (past_returns + ALPHA * GLOBAL_MEAN) / (past_orders + ALPHA)

Где ALPHA — параметр Bayesian smoothing (тянет редкие категории к глобальному).
GLOBAL_MEAN ≈ 0.0456 (target rate в full train, см. project_eda_findings.md).

Источники:
  microcat_name → orders_train ⋈ items (по item_id)
  city          → orders_train (есть напрямую)

Артефакты в data/features/:
  te_microcat_daily.parquet     — (microcat_name, order_create_date) → microcat_te, te_count
  te_microcat_snapshot.parquet  — (microcat_name) → te + count на конец train
  te_city_daily.parquet
  te_city_snapshot.parquet
"""
from __future__ import annotations
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "data" / "features"
ART = ROOT / "artifacts" / "features"
OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)
LOG = ART / "te.log"

GLOBAL_MEAN = 0.0456
ALPHA = 100.0

_lines: list[str] = []
def log(m: str = "") -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {m}" if m else ""
    _lines.append(line)
    LOG.write_text("\n".join(_lines), encoding="utf-8")


def build(con: duckdb.DuckDBPyConnection,
          short: str,
          source_sql: str,
          key_col: str) -> None:
    """source_sql даёт (k, d, is_return) на полном train. Строит TE daily + snapshot."""
    log("=" * 60)
    log(f"TE: {short} (key_col={key_col})")

    daily_out = OUT / f"te_{short}_daily.parquet"
    snap_out = OUT / f"te_{short}_snapshot.parquet"

    # Daily aggregates → cumulative shifted → smoothed TE
    log(f"  build daily → {daily_out.name}")
    con.execute(
        f"""
        COPY (
            WITH src AS ({source_sql}),
            daily AS (
                SELECT k, d,
                       COUNT(*) AS d_orders,
                       SUM(CAST(is_return AS INT)) AS d_returns
                FROM src
                WHERE k IS NOT NULL
                GROUP BY k, d
            ),
            cum AS (
                SELECT k, d, d_orders, d_returns,
                       SUM(d_orders)  OVER w AS cum_orders,
                       SUM(d_returns) OVER w AS cum_returns
                FROM daily
                WINDOW w AS (PARTITION BY k ORDER BY d)
            )
            SELECT k AS {key_col},
                   d AS order_create_date,
                   ((cum_returns - d_returns) + {ALPHA} * {GLOBAL_MEAN})
                       / ((cum_orders - d_orders) + {ALPHA})
                       AS {short}_te,
                   (cum_orders - d_orders) AS {short}_te_count
            FROM cum
        ) TO '{daily_out.as_posix()}' (FORMAT PARQUET, COMPRESSION 'zstd')
        """
    )
    n_daily = con.execute(f"SELECT COUNT(*) FROM read_parquet('{daily_out.as_posix()}')").fetchone()[0]
    log(f"  daily rows: {n_daily:,}")

    # Snapshot: TE на конец train (для test)
    log(f"  build snapshot → {snap_out.name}")
    con.execute(
        f"""
        COPY (
            WITH src AS ({source_sql})
            SELECT k AS {key_col},
                   (SUM(CAST(is_return AS INT)) + {ALPHA} * {GLOBAL_MEAN})
                       / (COUNT(*) + {ALPHA})
                       AS {short}_te,
                   COUNT(*) AS {short}_te_count
            FROM src
            WHERE k IS NOT NULL
            GROUP BY k
        ) TO '{snap_out.as_posix()}' (FORMAT PARQUET, COMPRESSION 'zstd')
        """
    )
    n_snap = con.execute(f"SELECT COUNT(*) FROM read_parquet('{snap_out.as_posix()}')").fetchone()[0]
    log(f"  snapshot rows: {n_snap:,}")


def main() -> None:
    t0 = time.time()
    log(f"Target encoding start. duckdb={duckdb.__version__}")
    log(f"  GLOBAL_MEAN={GLOBAL_MEAN}, ALPHA={ALPHA}")

    tmp_dir = ART / "duckdb_tmp_te"
    tmp_dir.mkdir(exist_ok=True)
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA memory_limit='10GB'")
    con.execute(f"PRAGMA temp_directory='{tmp_dir.as_posix()}'")
    con.execute("PRAGMA threads=4")
    log("  duckdb connected, mem_limit=10GB")

    orders = (CLEAN / "orders_train.parquet").as_posix()
    items = (CLEAN / "items.parquet").as_posix()

    # 1. microcat_name (через join с items)
    src_microcat = f"""
        SELECT i.microcat_name AS k,
               o.order_create_date AS d,
               o.is_return
        FROM read_parquet('{orders}') o
        JOIN read_parquet('{items}') i USING (item_id)
    """
    build(con, "microcat", src_microcat, "microcat_name")

    # 2. city (есть в orders напрямую)
    src_city = f"""
        SELECT city AS k,
               order_create_date AS d,
               is_return
        FROM read_parquet('{orders}')
    """
    build(con, "city", src_city, "city")

    con.close()
    for f in tmp_dir.glob("*"):
        f.unlink(missing_ok=True)
    tmp_dir.rmdir()

    log("=" * 60)
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        main()
    except BaseException as exc:
        import traceback
        crash = ART / "te_crash.log"
        crash.write_text(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", encoding="utf-8")
        raise
