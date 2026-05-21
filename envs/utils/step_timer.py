"""Lightweight step-level profiler for RoboTwin ``take_action``.

Default-disabled. When enabled via ``task._step_timer = StepTimer(enabled=True)``,
records per-phase wall-clock times and prints summaries every *log_interval* calls.
"""

from __future__ import annotations

import time

import numpy as np


class StepTimer:
    def __init__(self, enabled: bool = True, log_interval: int = 50) -> None:
        self.enabled = enabled
        self.log_interval = log_interval
        self._call_count = 0
        self._t0: float | None = None
        self._current_lap: str | None = None
        self._lap_start: float | None = None
        self._stats: dict[str, list[float]] = {}

    # ---- public API ----

    def start(self) -> None:
        """Begin timing a new ``take_action`` call."""
        if not self.enabled:
            return
        self._t0 = time.perf_counter()
        self._current_lap = None
        self._lap_start = None

    def lap(self, label: str) -> None:
        """Close the previous lap and start a new one named *label*."""
        if not self.enabled:
            return
        now = time.perf_counter()
        if self._current_lap is not None and self._lap_start is not None:
            elapsed = now - self._lap_start
            self._stats.setdefault(self._current_lap, []).append(elapsed)
        self._current_lap = label
        self._lap_start = now

    def stop(self) -> None:
        """Close the final lap, increment call count, print summary if due."""
        if not self.enabled:
            return
        now = time.perf_counter()
        if self._current_lap is not None and self._lap_start is not None:
            elapsed = now - self._lap_start
            self._stats.setdefault(self._current_lap, []).append(elapsed)
        self._call_count += 1
        if self._call_count % self.log_interval == 0:
            self.summary()

    def summary(self) -> None:
        """Print per-phase statistics for all recorded calls."""
        if not self._stats:
            return
        all_values = [v for values in self._stats.values() for v in values]
        if not all_values:
            return
        grand_total = sum(all_values)
        lines = [f"  {self._call_count} calls — {grand_total:.2f}s total"]
        for label, values in self._stats.items():
            a = np.array(values)
            total = a.sum()
            pct = total / grand_total * 100
            lines.append(
                f"  {label:30s}  mean={a.mean()*1000:7.1f}ms  "
                f"min={a.min()*1000:7.1f}ms  max={a.max()*1000:7.1f}ms  "
                f"total={total:6.2f}s ({pct:5.1f}%)"
            )
        print("\n".join(lines))

    def reset(self) -> None:
        """Clear accumulated statistics."""
        self._call_count = 0
        self._stats.clear()
