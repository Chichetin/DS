"""
Шаг B: репрезентативный сэмпл orders_train.

Стратегия: случайный sample 15% уникальных buyer_id (seed=42) → semi-join orders_train.
Не разбиваем историю байера → можно строить time-leak-safe фичи на сэмпле.
Простой random на buyer_id (без явной стратификации): на 33M заказов и ~12M
buyer'ов target rate должен сохраниться, проверим в логах.
"""
from __future__ import annotations
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
ART = ROOT / "artifacts" / "preprocess"
LOG = ART / "sample.log"

SAMPLE_FRAC = 0.15
SEED = 42

_lines: list[str] = []
def log(m: str = "") -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {m}" if m else ""
    _lines.append(line)
    LOG.write_text("\n".join(_lines), encoding="utf-8")


def main() -> None:
    src = CLEAN / "orders_train.parquet"
    out = CLEAN / "orders_train_sample.parquet"
    t0 = time.time()
    log(f"Sample start. frac={SAMPLE_FRAC}, seed={SEED}")
    log(f"src: {src}")

    # Шаг 1. Собираем уникальные buyer_id (drop NULL — после дедупа их быть не должно, но на всякий случай)
    buyers = (
        pl.scan_parquet(src)
        .select(pl.col("buyer_id"))
        .drop_nulls()
        .unique()
        .collect(engine="streaming")
    )
    n_buyers_total = buyers.height
    log(f"  unique buyer_id (full train): {n_buyers_total:,}")

    # Шаг 2. Random sample SAMPLE_FRAC из buyer'ов
    sampled_buyers = buyers.sample(fraction=SAMPLE_FRAC, seed=SEED, with_replacement=False)
    n_buyers_sample = sampled_buyers.height
    log(f"  sampled buyers: {n_buyers_sample:,} ({n_buyers_sample/n_buyers_total:.2%})")

    # Шаг 3. semi-join: оставляем только заказы выбранных байеров
    log("  semi-join → orders_train_sample.parquet")
    (
        pl.scan_parquet(src)
        .join(sampled_buyers.lazy(), on="buyer_id", how="semi")
        .sink_parquet(out, compression="zstd")
    )

    # Шаг 4. Статистика по результату — сравниваем с полным train
    sample_stats = (
        pl.scan_parquet(out)
        .select(
            pl.len().alias("rows"),
            pl.col("buyer_id").n_unique().alias("buyers"),
            pl.col("is_return").sum().alias("returns_true"),
            (~pl.col("is_return")).sum().alias("returns_false"),
            pl.col("order_create_date").min().alias("date_min"),
            pl.col("order_create_date").max().alias("date_max"),
        )
        .collect()
        .to_dicts()[0]
    )
    full_stats = (
        pl.scan_parquet(src)
        .select(
            pl.len().alias("rows"),
            pl.col("is_return").sum().alias("returns_true"),
            (~pl.col("is_return")).sum().alias("returns_false"),
        )
        .collect()
        .to_dicts()[0]
    )

    s_rate = sample_stats["returns_true"] / (sample_stats["returns_true"] + sample_stats["returns_false"])
    f_rate = full_stats["returns_true"] / (full_stats["returns_true"] + full_stats["returns_false"])

    log("")
    log("=== SAMPLE STATS ===")
    log(f"  rows:           {sample_stats['rows']:>12,}  ({sample_stats['rows']/full_stats['rows']:.2%} of train)")
    log(f"  unique buyers:  {sample_stats['buyers']:>12,}")
    log(f"  date range:     {sample_stats['date_min']} … {sample_stats['date_max']}")
    log(f"  is_return=True: {sample_stats['returns_true']:>12,}")
    log(f"  is_return rate: {s_rate:.4%}  (full train: {f_rate:.4%}, diff: {abs(s_rate-f_rate)*100:.3f}pp)")
    log("")
    log(f"DONE in {time.time()-t0:.1f}s. Output: {out}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        import traceback
        crash = ART / "sample_crash.log"
        crash.write_text(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", encoding="utf-8")
        raise
