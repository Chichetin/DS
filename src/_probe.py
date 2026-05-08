"""Quick probe: schema, columns, и проверка что .drop('') работает."""
from __future__ import annotations
import os, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "artifacts" / "preprocess" / "probe.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

lines: list[str] = []
def log(m: str = "") -> None:
    lines.append(m)

try:
    import polars as pl
    log(f"polars version: {pl.__version__}")

    for name in ["orders", "items", "users", "payments"]:
        path = ROOT / "data" / f"{name}.csv"
        log(f"\n=== {name}.csv ===")
        try:
            schema = pl.scan_csv(path, ignore_errors=True).collect_schema()
            log(f"columns: {list(schema.names())}")
            log(f"dtypes:  {[str(t) for t in schema.dtypes()]}")
        except Exception as e:
            log(f"SCHEMA ERR: {type(e).__name__}: {e}")

    # quick dry-run: можно ли сделать .drop('') на orders
    log("\n=== dry-run: orders drop unnamed ===")
    try:
        cols = list(
            pl.scan_csv(ROOT / "data" / "orders.csv", ignore_errors=True)
            .drop("")
            .collect_schema()
            .names()
        )
        log(f"after drop(''): {cols}")
    except Exception as e:
        log(f"DROP ERR: {type(e).__name__}: {e}")

except Exception as exc:
    log(f"\nFATAL: {type(exc).__name__}: {exc}")
    log(traceback.format_exc())
finally:
    LOG.write_text("\n".join(lines), encoding="utf-8")
