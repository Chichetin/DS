"""
Шаг C4: C3 + seller TE + microcat price stats + dynamics + derived → C4.

Шаги:
1. DuckDB: считаем microcat_price_stats.parquet (microcat → median, mean, count)
   из ПОЛНОГО orders_train.parquet (snapshot, без time-leak guard — медиана по
   популяции стабильна, тонкая утечка приемлема).
2. Polars: для каждого split (train/holdout/test):
   - load c3
   - join seller_id из orders_src по (deliveryorder_id, item_id)
   - join seller TE (daily для train/holdout, snapshot для test)
   - join microcat price stats (snapshot для всех)
   - производные:
     * buyer/seller dynamics_30d_diff = past_30d_return_rate - past_return_rate
     * buyer/seller dynamics_7d_diff  = past_7d_return_rate  - past_return_rate
     * price_to_microcat_median = order_price / microcat_median_price
     * log_price_to_microcat_median
     * is_first_sale = (item_past_orders == 0)
   - drop seller_id и сырые стат-колонки
   - sink → c4_*.parquet
"""
from __future__ import annotations
import os
from pathlib import Path

import duckdb
import polars as pl

from _log import Logger

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
FEAT = ROOT / "data" / "features"
ART = ROOT / "artifacts" / "features"
ART.mkdir(parents=True, exist_ok=True)

SUFFIX = os.environ.get("SUFFIX", "")
SAMPLE_PATH = os.environ.get("SAMPLE_PATH", "orders_train_sample.parquet")
TARGETS = set(os.environ.get("TARGETS", "train,holdout,test").split(","))

GLOBAL_MEAN = 0.0456

log = Logger(ART / f"c4{SUFFIX}.log")


def build_microcat_price_stats() -> Path:
    """Snapshot median/mean order_price per microcat_name (через items)."""
    out = FEAT / "microcat_price_stats.parquet"
    log.step(f"Build microcat_price_stats → {out.name}")

    tmp_dir = ART / "duckdb_tmp_microcat_price"
    tmp_dir.mkdir(exist_ok=True)
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA memory_limit='10GB'")
    con.execute(f"PRAGMA temp_directory='{tmp_dir.as_posix()}'")
    con.execute("PRAGMA threads=4")

    orders = (CLEAN / "orders_train.parquet").as_posix()
    items = (CLEAN / "items.parquet").as_posix()

    con.execute(
        f"""
        COPY (
            SELECT i.microcat_name AS microcat_name,
                   MEDIAN(o.order_price) AS microcat_median_price,
                   AVG(o.order_price)    AS microcat_mean_price,
                   COUNT(*)              AS microcat_price_count
            FROM read_parquet('{orders}') o
            JOIN read_parquet('{items}') i USING (item_id)
            WHERE o.order_price IS NOT NULL AND i.microcat_name IS NOT NULL
            GROUP BY i.microcat_name
        ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION 'zstd')
        """
    )
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    log(f"  microcats: {n:,}")
    con.close()
    for f in tmp_dir.glob("*"):
        f.unlink(missing_ok=True)
    tmp_dir.rmdir()
    return out


def add_c4_features(c3_lf: pl.LazyFrame, orders_src: Path, is_test: bool) -> pl.LazyFrame:
    # 1. Re-attach seller_id для join'а с TE
    ids = pl.scan_parquet(orders_src).select("deliveryorder_id", "item_id", "seller_id")
    lf = c3_lf.join(ids, on=["deliveryorder_id", "item_id"], how="left")

    # 2. seller TE
    if is_test:
        te = pl.scan_parquet(FEAT / "te_seller_snapshot.parquet")
        lf = lf.join(te, on="seller_id", how="left")
    else:
        te = pl.scan_parquet(FEAT / "te_seller_daily.parquet")
        lf = lf.join(te, on=["seller_id", "order_create_date"], how="left")

    # 3. microcat price stats (snapshot для всех)
    price_stats = pl.scan_parquet(FEAT / "microcat_price_stats.parquet")
    lf = lf.join(price_stats, on="microcat_name", how="left")

    # 4. Derived
    derived: list[pl.Expr] = [
        pl.col("seller_te").fill_null(GLOBAL_MEAN).alias("seller_te"),
        pl.col("seller_te_count").fill_null(0).cast(pl.UInt32).alias("seller_te_count"),

        # Dynamics: отклонение свежего поведения от исторического. Null если нет
        # all-time истории ИЛИ нет 30d/7d истории (rate=null) — pl.Float64 NaN
        # выглядит как 0 для LGBM, оставим Null чтобы LGBM его трактовал отдельно.
        (
            pl.col("buyer_past_30d_return_rate") - pl.col("buyer_past_return_rate")
        ).alias("buyer_dynamics_30d_diff"),
        (
            pl.col("buyer_past_7d_return_rate") - pl.col("buyer_past_return_rate")
        ).alias("buyer_dynamics_7d_diff"),
        (
            pl.col("seller_past_30d_return_rate") - pl.col("seller_past_return_rate")
        ).alias("seller_dynamics_30d_diff"),
        (
            pl.col("seller_past_7d_return_rate") - pl.col("seller_past_return_rate")
        ).alias("seller_dynamics_7d_diff"),

        # Price ratio к микрокатегорной медиане
        (
            pl.when(pl.col("microcat_median_price") > 0)
            .then(pl.col("order_price") / pl.col("microcat_median_price"))
            .otherwise(None)
        ).alias("price_to_microcat_median"),
        (
            pl.when(pl.col("microcat_median_price") > 0)
            .then(
                (pl.col("order_price") + 1.0).log()
                - (pl.col("microcat_median_price") + 1.0).log()
            )
            .otherwise(None)
        ).alias("log_price_to_microcat_median"),

        # Is first sale: первый заказ для item_id (новинка)
        (pl.col("item_past_orders") == 0).cast(pl.Int8).alias("is_first_sale"),
    ]

    drop_cols = ["seller_id", "microcat_mean_price", "microcat_price_count"]
    return lf.with_columns(derived).drop(drop_cols)


def main() -> None:
    log(f"C4 features start. polars={pl.__version__}, duckdb={duckdb.__version__}")

    if not (FEAT / "microcat_price_stats.parquet").exists():
        build_microcat_price_stats()
    else:
        log("microcat_price_stats.parquet уже существует — пропускаем")

    sample_full = CLEAN / SAMPLE_PATH
    targets_all = [
        ("train",   FEAT / f"c3_train{SUFFIX}.parquet",   sample_full,                   False),
        ("holdout", FEAT / f"c3_holdout{SUFFIX}.parquet", sample_full,                   False),
        ("test",    FEAT / "c3_test.parquet",             CLEAN / "orders_test.parquet", True),
    ]

    for name, c3_src, orders_src, is_test in targets_all:
        if name not in TARGETS:
            continue
        out_suffix = "" if is_test else SUFFIX
        log.step(f"Build c4_{name}{out_suffix} from {c3_src.name}")
        c3_lf = pl.scan_parquet(c3_src)
        c4_lf = add_c4_features(c3_lf, orders_src, is_test)
        out = FEAT / f"c4_{name}{out_suffix}.parquet"
        log(f"  sink → {out.name}")
        c4_lf.sink_parquet(out, compression="zstd")
        n = pl.scan_parquet(out).select(pl.len()).collect().item()
        log(f"  c4_{name}{out_suffix}: {n:,} rows")

    if "train" in TARGETS:
        log.step(f"Final schema (c4_train{SUFFIX}) — new columns vs c3:")
        schema = pl.scan_parquet(FEAT / f"c4_train{SUFFIX}.parquet").collect_schema()
        new_cols = [
            "seller_te", "seller_te_count",
            "buyer_dynamics_30d_diff", "buyer_dynamics_7d_diff",
            "seller_dynamics_30d_diff", "seller_dynamics_7d_diff",
            "microcat_median_price",
            "price_to_microcat_median", "log_price_to_microcat_median",
            "is_first_sale",
        ]
        for n_, dt in zip(schema.names(), schema.dtypes()):
            if n_ in new_cols:
                log(f"  {n_}: {dt}")

    log.done()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        main()
    except BaseException as exc:
        log.crash(exc, ART / "c4_crash.log")
        raise
