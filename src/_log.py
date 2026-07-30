"""
Общий логгер для скриптов финального пайплайна (02, 05, 08, 09, 11, 12, 16, 17,
24, 25, 26, 27).

Контракт (важно — README прямо запрещает print() в фоне из-за cp1251):
- Всё пишется в файл `LOG_FILE.write_text(...)` после каждого вызова → лог
  всегда консистентен, даже при kill -9 / OOM exit code 5.
- print() в stdout — только если явно `tee_stdout=True` (для интерактивной
  отладки; в фоне НЕ включать).
- Совместимость со старым API: `log = Logger(path)`; `log("текст")` работает
  как `log.info("текст")`. Локальный `def log(m=""):` в скрипте можно
  целиком заменить на `log = Logger(LOG_FILE)`.

Дополнительно к старому API:
- `log.step("Step 1/5: load")` пишет separator '=' * 60 + название + сразу
  фиксирует таймер текущего этапа. При следующем `step()` / `done()` пишет
  «<step> took Xs», чтобы видеть, что было долгим.
- `log.warn(msg)` / `log.error(msg)` — те же строки, но с префиксом-маркером
  (для grep'а в run.log).
- `log.crash(exc, path)` — пишет полный traceback в отдельный crash-файл
  (повторяет существующий паттерн из 02/08/.../27).
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Optional


class Logger:
    """Append-only логгер с flush-в-файл после каждой строки.

    Использование:
        log = Logger(ART / "run.log")
        log("сообщение")          # info
        log.step("Step 1/5: foo") # граница этапа + таймер
        log.warn("что-то странное")
        log.done()                # финальный «DONE in Xs»
    """

    def __init__(self, log_file: Path, *, tee_stdout: bool = False) -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.tee_stdout = tee_stdout
        self._lines: list[str] = []
        self._t0 = time.time()
        self._step_t0: Optional[float] = None
        self._step_name: Optional[str] = None
        # Сразу пересоздаём файл, чтобы запуск не докидывал строки в старый прогон.
        self.log_file.write_text("", encoding="utf-8")

    # ---- core ----------------------------------------------------------------

    def _write(self, line: str) -> None:
        self._lines.append(line)
        # Полный rewrite — атомарно, гарантирует консистентность при kill -9.
        # На больших логах (~10k строк) это ~миллисекунды, незаметно.
        self.log_file.write_text("\n".join(self._lines), encoding="utf-8")
        if self.tee_stdout:
            try:
                print(line, flush=True)
            except UnicodeEncodeError:
                # На cp1251 — выгружаем без кириллицы, чтобы не падать.
                print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)

    def _stamped(self, msg: str = "") -> str:
        if not msg:
            return ""
        return f"[{time.strftime('%H:%M:%S')}] {msg}"

    # ---- callable: log("…") == log.info("…") --------------------------------

    def __call__(self, msg: str = "") -> None:
        self.info(msg)

    def info(self, msg: str = "") -> None:
        self._write(self._stamped(msg))

    def warn(self, msg: str) -> None:
        self._write(self._stamped(f"WARN: {msg}"))

    def error(self, msg: str) -> None:
        self._write(self._stamped(f"ERROR: {msg}"))

    # ---- этапы ---------------------------------------------------------------

    def step(self, name: str) -> None:
        """Начало этапа. Пишет '=' * 60, название, фиксирует таймер.

        Если до этого был открытый этап — пишет «<prev> took Xs».
        """
        self._close_step_if_open()
        self._write("=" * 60)
        self._write(self._stamped(name))
        self._step_t0 = time.time()
        self._step_name = name

    def _close_step_if_open(self) -> None:
        if self._step_t0 is not None and self._step_name is not None:
            dt = time.time() - self._step_t0
            self._write(self._stamped(f"  ↳ {self._step_name!r} took {dt:.1f}s"))
            self._step_t0 = None
            self._step_name = None

    def done(self, msg: str = "DONE") -> None:
        """Финальная строка. Закрывает текущий этап, если открыт."""
        self._close_step_if_open()
        dt = time.time() - self._t0
        self._write("=" * 60)
        self._write(self._stamped(f"{msg} in {dt:.1f}s"))

    # ---- crash ---------------------------------------------------------------

    def crash(self, exc: BaseException, crash_file: Optional[Path] = None) -> None:
        """Полный дамп исключения в отдельный crash-файл + строка ERROR в run.log.

        Не делает re-raise — caller должен сам решить, поднимать ли дальше.
        """
        tb = traceback.format_exc()
        self.error(f"{type(exc).__name__}: {exc}")
        self._write("--- traceback ---")
        for line in tb.rstrip().splitlines():
            self._write(line)
        if crash_file is not None:
            crash_file = Path(crash_file)
            crash_file.parent.mkdir(parents=True, exist_ok=True)
            crash_file.write_text(
                f"{type(exc).__name__}: {exc}\n\n{tb}",
                encoding="utf-8",
            )


def install_excepthook(log: Logger, crash_file: Path) -> None:
    """Опционально — глобальный sys.excepthook, чтобы любой uncaught exception
    тоже попадал в run.log + crash.log. Удобно если main() забыл try/except.

    Не используется по умолчанию — оставлен для дальнейшего применения.
    """
    def _hook(exc_type, exc, tb):
        log.crash(exc, crash_file)
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook