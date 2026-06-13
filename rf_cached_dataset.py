from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_CACHE_ROOT = Path("/mnt/g/rf_training_cache/dual_input_bmode")
VALID_SPLITS = {"train", "val", "test"}


class RFCachedDataset(Dataset):
    """Memmap-backed dual-input cache reader.

    The large arrays are never loaded in bulk. Each __getitem__ call slices one
    sample from each memmap and converts that sample to float32 tensors.
    """

    def __init__(self, cache_root: str | Path = DEFAULT_CACHE_ROOT, split: str = "train"):
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")

        self.cache_root = Path(cache_root)
        self.split = split
        self.split_dir = self.cache_root / split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Cache split directory not found: {self.split_dir}")

        self.paths = {
            "input_rf": self.split_dir / "input.dat",
            "input_bmode": self.split_dir / "baseline_bmode.dat",
            "label_bmode": self.split_dir / "label_bmode.dat",
            "scale": self.split_dir / "scale.dat",
            "meta": self.split_dir / "meta.npz",
        }
        missing = [str(path) for path in self.paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing cache files:\n" + "\n".join(missing))

        with np.load(self.paths["meta"], allow_pickle=True) as meta:
            self.meta_keys = tuple(meta.keys())
            required = {
                "input_rf_shape",
                "input_bmode_shape",
                "label_bmode_shape",
                "scale_shape",
                "input_rf_dtype",
                "input_bmode_dtype",
                "label_bmode_dtype",
                "scale_dtype",
                "category",
            }
            missing_keys = sorted(required.difference(self.meta_keys))
            if missing_keys:
                raise KeyError(f"Missing keys in {self.paths['meta']}: {missing_keys}")

            self.input_rf_shape = tuple(int(x) for x in meta["input_rf_shape"])
            self.input_bmode_shape = tuple(int(x) for x in meta["input_bmode_shape"])
            self.label_bmode_shape = tuple(int(x) for x in meta["label_bmode_shape"])
            self.scale_shape = tuple(int(x) for x in meta["scale_shape"])
            self.input_rf_dtype = np.dtype(str(meta["input_rf_dtype"].item()))
            self.input_bmode_dtype = np.dtype(str(meta["input_bmode_dtype"].item()))
            self.label_bmode_dtype = np.dtype(str(meta["label_bmode_dtype"].item()))
            self.scale_dtype = np.dtype(str(meta["scale_dtype"].item()))
            self.categories = np.asarray(meta["category"])

        self._validate_shapes_and_sizes()

    def __len__(self) -> int:
        return self.input_rf_shape[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        return {
            "input_rf": self._read_sample("input_rf", index),
            "input_bmode": self._read_sample("input_bmode", index),
            "label_bmode": self._read_sample("label_bmode", index),
            "scale": self._read_scale(index),
            "category": str(self.categories[index]),
        }

    def _validate_shapes_and_sizes(self) -> None:
        n = self.input_rf_shape[0]
        expected_shapes = {
            "input_rf": (n, 1536, 64, 32, 32),
            "input_bmode": (n, 1, 64, 32, 32),
            "label_bmode": (n, 1, 64, 32, 32),
            "scale": (n,),
        }
        actual_shapes = {
            "input_rf": self.input_rf_shape,
            "input_bmode": self.input_bmode_shape,
            "label_bmode": self.label_bmode_shape,
            "scale": self.scale_shape,
        }
        for name, expected in expected_shapes.items():
            actual = actual_shapes[name]
            if actual != expected:
                raise ValueError(f"{name} shape mismatch: expected {expected}, got {actual}")

        expected_dtypes = {
            "input_rf": np.dtype("float16"),
            "input_bmode": np.dtype("float16"),
            "label_bmode": np.dtype("float16"),
            "scale": np.dtype("float32"),
        }
        actual_dtypes = {
            "input_rf": self.input_rf_dtype,
            "input_bmode": self.input_bmode_dtype,
            "label_bmode": self.label_bmode_dtype,
            "scale": self.scale_dtype,
        }
        for name, expected in expected_dtypes.items():
            actual = actual_dtypes[name]
            if actual != expected:
                raise ValueError(f"{name} dtype mismatch: expected {expected}, got {actual}")

        if self.categories.shape != (n,):
            raise ValueError(f"category shape mismatch: expected {(n,)}, got {self.categories.shape}")

        size_checks = {
            "input_rf": (self.input_rf_shape, self.input_rf_dtype),
            "input_bmode": (self.input_bmode_shape, self.input_bmode_dtype),
            "label_bmode": (self.label_bmode_shape, self.label_bmode_dtype),
            "scale": (self.scale_shape, self.scale_dtype),
        }
        for name, (shape, dtype) in size_checks.items():
            expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            actual_bytes = self.paths[name].stat().st_size
            if actual_bytes != expected_bytes:
                raise ValueError(
                    f"{self.paths[name]} size mismatch: expected {expected_bytes} bytes, "
                    f"got {actual_bytes} bytes"
                )

    @staticmethod
    def _to_float32_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(array, dtype=np.float32))

    def _read_sample(self, name: str, index: int) -> torch.Tensor:
        specs = {
            "input_rf": (self.input_rf_shape, self.input_rf_dtype),
            "input_bmode": (self.input_bmode_shape, self.input_bmode_dtype),
            "label_bmode": (self.label_bmode_shape, self.label_bmode_dtype),
        }
        shape, dtype = specs[name]
        mm = np.memmap(self.paths[name], dtype=dtype, mode="r", shape=shape)
        try:
            return self._to_float32_tensor(mm[index])
        finally:
            del mm

    def _read_scale(self, index: int) -> torch.Tensor:
        mm = np.memmap(self.paths["scale"], dtype=self.scale_dtype, mode="r", shape=self.scale_shape)
        try:
            return torch.tensor(float(mm[index]), dtype=torch.float32)
        finally:
            del mm
