import contextlib
import json
import platform
import threading
import time
from pathlib import Path

import psutil

_PROCESS = psutil.Process()

def _tree_rss_bytes(proc: psutil.Process) -> int:
    total = 0
    for p in [proc, *proc.children(recursive=True)]:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _named_rss_bytes(names: list[str]) -> int:
    total = 0
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            name = p.info["name"] or ""
            if any(n.lower() in name.lower() for n in names):
                total += p.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def rss_mb(extra_process_names: list[str] | None = None) -> float:
    """Current process (+ children, + any named processes) RSS in MB."""
    total = _tree_rss_bytes(_PROCESS)
    if extra_process_names:
        total += _named_rss_bytes(extra_process_names)
    return total / (1024 * 1024)


class Report:
    """Accumulates per-stage latency + RAM readings and prints/saves them.

    Each stage is stored as a flat row: stage name, how long it took, RAM
    (MB) when the stage started, and the highest RAM (MB) seen *at any point
    during* the stage (sampled in the background every 50ms, not just at the
    start/end). The peak matters more than the end reading: things like
    memory-mapped model weights can spike RAM while loading and then look
    "cheap" again once the OS pages them back out, which a start/end-only
    reading would miss entirely.
    """

    def __init__(self, name: str, out_dir: str = "benchmarks") -> None:
        self.name = name
        self.out_dir = Path(out_dir)
        self.stages: list[dict] = []

    @contextlib.contextmanager
    def stage(
        self,
        label: str,
        extra_process_names: list[str] | None = None,
        poll_seconds: float = 0.05,
    ):
        mb_start = rss_mb(extra_process_names)
        peak = [mb_start]
        stop = threading.Event()

        def sample() -> None:
            while not stop.wait(poll_seconds):
                peak[0] = max(peak[0], rss_mb(extra_process_names))

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        start = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - start
            stop.set()
            sampler.join()
            peak[0] = max(peak[0], rss_mb(extra_process_names))
            self.stages.append(
                {
                    "stage": label,
                    "seconds": round(seconds, 2),
                    "mb_start": round(mb_start, 1),
                    "mb_peak": round(peak[0], 1),
                }
            )

    def _row(self, s: dict) -> str:
        return f"{s['stage']:<30}{s['seconds']:>10}s{s['mb_start']:>10} MB start ->{s['mb_peak']:>8} MB peak"

    def print_table(self) -> None:
        print(f"\n=== {self.name} ===")
        for s in self.stages:
            print(self._row(s))
        if self.stages:
            total_seconds = round(sum(s["seconds"] for s in self.stages), 2)
            peak_mb = max(s["mb_peak"] for s in self.stages)
            print(f"\ntotal time: {total_seconds}s | peak RAM: {peak_mb} MB")

    def save(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.name}.json"
        total_seconds = round(sum(s["seconds"] for s in self.stages), 2)
        peak_mb = max((s["mb_peak"] for s in self.stages), default=0)
        path.write_text(
            json.dumps(
                {
                    "name": self.name,
                    "platform": platform.platform(),
                    "total_seconds": total_seconds,
                    "peak_mb": peak_mb,
                    "stages": self.stages,
                },
                indent=2,
            )
        )
        return path
