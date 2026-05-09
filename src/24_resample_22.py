"""
Шаг F1: создаём 22%-сэмпл по buyer_id для тренировки CatBoost.

Зачем 22%: CatBoost C4 (depth=6) на 4.36M ел ~8GB RAM. На ~7.3M строк (22%
buyers ≈ 1.5x от 15%) ожидаем ~12GB peak — впритык в 16GB. Больше — риск OOM.

Вход:  data/clean/orders_train.parquet
Выход: data/clean/orders_train_sample22.parquet
"""
from __future__ import annotations
from pathlib import Path

import polars as pl

from _log import Logger

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
ART = ROOT / "artifacts" / "preprocess"
ART.mkdir(parents=True, exist_ok=True)

SAMPLE_FRAC = 0.22
SEED = 42

log = Logger(ART / "sample22.log")


def main() -> None:
    src = CLEAN / "orders_train.parquet"
    out = CLEAN / "orders_train_sample22.parquet"
    log(f"Sample-22 start. frac={SAMPLE_FRAC}, seed={SEED}")
    log(f"src: {src}")

    log.step("Step 1/3: collect unique buyer_id from full train")
    buyers = (
        pl.scan_parquet(src)
        .select(pl.col("buyer_id"))
        .drop_nulls()
        .unique()
        .collect(engine="streaming")
    )
    n_buyers_total = buyers.height
    log(f"  unique buyer_id (full train): {n_buyers_total:,}")

    log.step("Step 2/3: sample buyers")
    sampled_buyers = buyers.sample(fraction=SAMPLE_FRAC, seed=SEED, with_replacement=False)
    n_buyers_sample = sampled_buyers.height
    log(f"  sampled buyers: {n_buyers_sample:,} ({n_buyers_sample/n_buyers_total:.2%})")

    log.step("Step 3/3: semi-join → orders_train_sample22.parquet")
    (
        pl.scan_parquet(src)
        .join(sampled_buyers.lazy(), on="buyer_id", how="semi")
        .sink_parquet(out, compression="zstd")
    )

    n_rows = pl.scan_parquet(out).select(pl.len()).collect().item()
    n_returns = pl.scan_parquet(out).select(pl.col("is_return").sum()).collect().item()
    rate = n_returns / n_rows
    log(f"  rows:           {n_rows:>12,}")
    log(f"  returns rate:   {rate:.4%}")
    log(f"  output: {out}")
    log.done()


if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        main()
    except BaseException as exc:
        log.crash(exc, ART / "sample22_crash.log")
        raise
