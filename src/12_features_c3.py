"""
Шаг C3/часть 2: C2 + target encoding → C3.

Логика:
1. Загружаем c2_{train,holdout,test}.parquet.
2. Для train+holdout: join с te_{microcat,city}_daily по (key, order_create_date).
3. Для test: join с te_{microcat,city}_snapshot по key.
4. Заполняем NULL (новые категории не виданы до даты) глобальным средним.
5. Сохраняем c3_{train,holdout,test}.parquet.
"""
from __future__ import annotations
import os
from pathlib import Path

import polars as pl

from _log import Logger

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features"
ART = ROOT / "artifacts" / "features"
ART.mkdir(parents=True, exist_ok=True)

SUFFIX = os.environ.get("SUFFIX", "")
TARGETS = set(os.environ.get("TARGETS", "train,holdout,test").split(","))

GLOBAL_MEAN = 0.0456

log = Logger(ART / f"c3{SUFFIX}.log")


TE_KEYS = [
    ("microcat", "microcat_name"),
    ("city", "city"),
]


def add_te(c2_lf: pl.LazyFrame, is_test: bool) -> pl.LazyFrame:
    lf = c2_lf
    for short, key_col in TE_KEYS:
        if is_test:
            te = pl.scan_parquet(FEAT / f"te_{short}_snapshot.parquet")
            lf = lf.join(te, on=key_col, how="left")
        else:
            te = pl.scan_parquet(FEAT / f"te_{short}_daily.parquet")
            lf = lf.join(te, on=[key_col, "order_create_date"], how="left")

    # Заполняем null'ы глобальным средним для новых категорий
    fills: list[pl.Expr] = []
    for short, _ in TE_KEYS:
        fills += [
            pl.col(f"{short}_te").fill_null(GLOBAL_MEAN).alias(f"{short}_te"),
            pl.col(f"{short}_te_count").fill_null(0).cast(pl.UInt32).alias(f"{short}_te_count"),
        ]
    return lf.with_columns(fills)


def main() -> None:
    log(f"C3 features start. polars={pl.__version__}")

    targets_all = [
        ("train",   FEAT / f"c2_train{SUFFIX}.parquet",   False),
        ("holdout", FEAT / f"c2_holdout{SUFFIX}.parquet", False),
        ("test",    FEAT / "c2_test.parquet",             True),
    ]

    for name, c2_src, is_test in targets_all:
        if name not in TARGETS:
            continue
        out_suffix = "" if is_test else SUFFIX
        log.step(f"Build c3_{name}{out_suffix} from {c2_src.name}")
        c2_lf = pl.scan_parquet(c2_src)
        c3_lf = add_te(c2_lf, is_test)
        out = FEAT / f"c3_{name}{out_suffix}.parquet"
        log(f"  sink → {out.name}")
        c3_lf.sink_parquet(out, compression="zstd")
        n = pl.scan_parquet(out).select(pl.len()).collect().item()
        log(f"  c3_{name}{out_suffix}: {n:,} rows")

    if "train" in TARGETS:
        log.step(f"Final schema (c3_train{SUFFIX}) — new TE columns:")
        schema = pl.scan_parquet(FEAT / f"c3_train{SUFFIX}.parquet").collect_schema()
        for n, dt in zip(schema.names(), schema.dtypes()):
            if "_te" in n:
                log(f"  {n}: {dt}")

    log.done()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        main()
    except BaseException as exc:
        log.crash(exc, ART / "c3_crash.log")
        raise
