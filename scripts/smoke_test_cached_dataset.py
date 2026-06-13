from __future__ import annotations

import argparse
import ctypes
import gc
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rf_cached_dataset import DEFAULT_CACHE_ROOT, RFCachedDataset


def trim_memory() -> None:
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass


def memory_status_mb() -> tuple[float | None, float | None]:
    status = Path("/proc/self/status")
    if not status.exists():
        return None, None

    rss_kb = None
    hwm_kb = None
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            rss_kb = float(line.split()[1])
        elif line.startswith("VmHWM:"):
            hwm_kb = float(line.split()[1])
    rss_mb = rss_kb / 1024.0 if rss_kb is not None else None
    hwm_mb = hwm_kb / 1024.0 if hwm_kb is not None else None
    return rss_mb, hwm_mb


def format_memory() -> str:
    rss_mb, hwm_mb = memory_status_mb()
    if rss_mb is None or hwm_mb is None:
        return "rss_mb=NA hwm_mb=NA"
    return f"rss_mb={rss_mb:.1f} hwm_mb={hwm_mb:.1f}"


def tensor_stats(tensor: torch.Tensor, chunk_size: int = 1_000_000) -> dict[str, object]:
    flat = tensor.reshape(-1)
    min_value = float("inf")
    max_value = float("-inf")
    sum_value = 0.0
    count = 0
    nan_count = 0
    inf_count = 0

    for start in range(0, flat.numel(), chunk_size):
        chunk = flat[start : start + chunk_size]
        nan_count += int(torch.isnan(chunk).sum().item())
        inf_count += int(torch.isinf(chunk).sum().item())
        min_value = min(min_value, float(chunk.min().item()))
        max_value = max(max_value, float(chunk.max().item()))
        sum_value += float(chunk.sum(dtype=torch.float64).item())
        count += int(chunk.numel())

    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": min_value,
        "max": max_value,
        "mean": sum_value / count,
        "nan": nan_count,
        "inf": inf_count,
    }


def print_stats(name: str, tensor: torch.Tensor) -> dict[str, object]:
    stats = tensor_stats(tensor)
    print(
        f"    {name}: shape={stats['shape']} dtype={stats['dtype']} "
        f"min={stats['min']:.6g} max={stats['max']:.6g} mean={stats['mean']:.6g} "
        f"nan={stats['nan']} inf={stats['inf']}",
        flush=True,
    )
    return stats


def sample_indices(dataset: RFCachedDataset) -> list[int]:
    indices = set(range(min(3, len(dataset))))
    for category in ("carotid", "muscle", "phantom"):
        matches = np.flatnonzero(dataset.categories == category)
        if matches.size == 0:
            raise RuntimeError(f"No sample found for category={category!r} split={dataset.split!r}")
        indices.add(int(matches[0]))
    return sorted(indices)


def smoke_split(cache_root: Path, split: str) -> None:
    print(f"\n=== split={split} cache_root={cache_root} ===", flush=True)
    dataset = RFCachedDataset(cache_root=cache_root, split=split)
    print(f"len={len(dataset)} meta_keys={list(dataset.meta_keys)}", flush=True)
    values, counts = np.unique(dataset.categories, return_counts=True)
    print("category_counts=" + str({str(v): int(c) for v, c in zip(values, counts)}), flush=True)
    print(
        "memmap_shapes="
        f"input_rf={dataset.input_rf_shape} "
        f"input_bmode={dataset.input_bmode_shape} "
        f"label_bmode={dataset.label_bmode_shape} "
        f"scale={dataset.scale_shape}",
        flush=True,
    )
    print("before_samples " + format_memory(), flush=True)

    for index in sample_indices(dataset):
        sample = dataset[index]
        print(f"  index={index} category={sample['category']} scale={float(sample['scale']):.6g}", flush=True)
        input_rf_stats = print_stats("input_rf", sample["input_rf"])
        input_bmode_stats = print_stats("input_bmode", sample["input_bmode"])
        label_bmode_stats = print_stats("label_bmode", sample["label_bmode"])

        input_rf_mean = float(input_rf_stats["mean"])
        if abs(input_rf_mean) > 3.0:
            raise AssertionError(f"input_rf mean out of expected +/-3 range: index={index} mean={input_rf_mean}")
        for key, stats in (("input_bmode", input_bmode_stats), ("label_bmode", label_bmode_stats)):
            mn = float(stats["min"])
            mx = float(stats["max"])
            if mn < -0.01 or mx > 1.01:
                raise AssertionError(f"{key} out of [0,1] tolerance: index={index} min={mn} max={mx}")

        del sample
        gc.collect()
        trim_memory()
        print("  after_sample " + format_memory(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the RF dual-input memmap cache reader.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    args = parser.parse_args()

    for split in args.splits:
        smoke_split(args.cache_root, split)

    print("\nALL_SMOKE_PASS", flush=True)


if __name__ == "__main__":
    main()
