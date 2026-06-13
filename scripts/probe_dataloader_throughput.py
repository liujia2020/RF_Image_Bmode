from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rf_cached_dataset import DEFAULT_CACHE_ROOT, RFCachedDataset


def rss_mb(pid: int) -> float:
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return 0.0
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    return 0.0


def total_rss_mb(pids: list[int]) -> float:
    return sum(rss_mb(pid) for pid in pids)


def worker_pids(iterator: object) -> list[int]:
    workers = getattr(iterator, "_workers", None)
    if not workers:
        return []
    return [worker.pid for worker in workers if worker.pid is not None]


def print_diagnostics() -> None:
    print(f"torch.__version__={torch.__version__}", flush=True)
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("torch.cuda.get_device_name(0)=NA", flush=True)


def probe_workers(cache_root: Path, num_workers: int, batch_size: int, max_batches: int) -> None:
    dataset = RFCachedDataset(cache_root=cache_root, split="train")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    iterator = iter(loader)
    parent_pid = os.getpid()
    peak_rss = total_rss_mb([parent_pid])
    measured_batches = 0
    measured_samples = 0
    start = None
    elapsed = 0.0

    try:
        for batch_idx in range(max_batches):
            batch = next(iterator)
            pids = [parent_pid] + worker_pids(iterator)
            peak_rss = max(peak_rss, total_rss_mb(pids))

            if batch_idx == 1:
                start = perf_counter()

            if batch_idx >= 1:
                measured_batches += 1
                measured_samples += int(batch["input_rf"].shape[0])
                elapsed = perf_counter() - start

            del batch
    finally:
        if hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()

    samples_per_sec = measured_samples / elapsed if elapsed > 0 else 0.0
    est_epoch_sec = len(dataset) / samples_per_sec if samples_per_sec > 0 else float("inf")
    est_epoch_min = est_epoch_sec / 60.0
    print(
        f"num_workers={num_workers}  samples/s={samples_per_sec:.1f}  "
        f"est_epoch={est_epoch_sec:.0f}s (about {est_epoch_min:.1f} min)  "
        f"peak_rss={peak_rss:.0f}MB",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe RF cached DataLoader throughput from local cache.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=24)
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 4, 8])
    args = parser.parse_args()

    print_diagnostics()
    for num_workers in args.workers:
        probe_workers(args.cache_root, num_workers, args.batch_size, args.max_batches)


if __name__ == "__main__":
    main()
