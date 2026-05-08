"""
Найти и показать реальные примеры дублирующихся пар (deliveryorder_id, item_id) в orders.csv.
Цель: понять, это полные дубли строк или разные транзакции (например, несколько единиц товара).
Вывод пишем в artifacts/eda/duplicates_examples.txt.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
ORDERS_CSV = ROOT / "data" / "orders.csv"
OUT_DIR = ROOT / "artifacts" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "duplicates_examples.txt"

lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    lines.append(msg)


def main() -> None:
    log(f"orders.csv: {ORDERS_CSV}")
    log("=" * 80)

    # Шаг 1. Найти counts по парам (deliveryorder_id, item_id), фильтровать count>=2.
    log("Шаг 1. Считаем повторы пар (deliveryorder_id, item_id)...")
    dup_counts = (
        pl.scan_csv(ORDERS_CSV, ignore_errors=True)
        .group_by(["deliveryorder_id", "item_id"])
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= 2)
        .collect(engine="streaming")
    )
    log(f"  Пар с >=2 повторами: {dup_counts.height:,}")
    log(f"  Распределение n: {dup_counts['n'].describe()}")
    log("")

    # Шаг 2. Выбрать примеры: n=2 (самые частые), n=3, n=10, n=max
    n_max = int(dup_counts["n"].max())
    log(f"  Max n = {n_max}")

    sample_pairs: list[tuple[int, pl.DataFrame]] = []
    for target_n in sorted({2, 3, 5, 10, n_max}):
        candidates = dup_counts.filter(pl.col("n") == target_n).head(2)
        if candidates.height == 0:
            # ближайшее n
            candidates = dup_counts.filter(pl.col("n") >= target_n).sort("n").head(2)
        sample_pairs.append((target_n, candidates))

    # Шаг 3. Для каждой выбранной пары — вытащить полные строки из orders.csv
    log("Шаг 2. Вытаскиваем полные строки для примеров...")
    log("")
    for target_n, pairs in sample_pairs:
        if pairs.height == 0:
            continue
        log("=" * 80)
        log(f"### Примеры пар с n={target_n} повторов")
        log("=" * 80)
        for row in pairs.iter_rows(named=True):
            do_id = row["deliveryorder_id"]
            it_id = row["item_id"]
            actual_n = row["n"]
            log(f"\n--- (deliveryorder_id={do_id}, item_id={it_id}), n={actual_n} ---")
            full_rows = (
                pl.scan_csv(ORDERS_CSV, ignore_errors=True)
                .filter(
                    (pl.col("deliveryorder_id") == do_id)
                    & (pl.col("item_id") == it_id)
                )
                .collect(engine="streaming")
            )
            with pl.Config(
                tbl_rows=50,
                tbl_cols=20,
                fmt_str_lengths=80,
                tbl_width_chars=200,
            ):
                log(str(full_rows))

    log("")
    log("=" * 80)
    log("ГОТОВО")

    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nОтчёт сохранён: {OUT_FILE}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    main()
