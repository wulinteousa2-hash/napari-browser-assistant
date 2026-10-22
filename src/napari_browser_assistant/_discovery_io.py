"""Bounded, read-only region access for Discover; coordinates always use level 0."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

from ._library import TIFF_SUFFIXES
from ._reader import (
    is_ome_zarr, read_workspace, storage_path, workspace_manifest, zarr_arrays,
)

MAX_DECODE_BYTES = 128 * 1024 * 1024
MAX_PATCH_PIXELS = 4 * 1024 * 1024


def fingerprint(path):
    """Metadata revision, not a claim of hashing all pixels/chunks."""
    path = Path(path)
    paths = [path]
    if path.is_dir():
        paths += [path / name for name in (".zattrs", ".zgroup", "zarr.json")]
    manifest = workspace_manifest(path)
    if manifest:
        document = read_workspace(manifest)
        for record in document["layers"]:
            storage = record.get("storage", {})
            if record.get("layer_type") == "Image" and storage.get("kind") == "zarr":
                root = storage_path(manifest, storage)
                paths += [root] + [root / name for name in (".zattrs", ".zgroup", "zarr.json")]
                base = root / (storage.get("array_path") or "")
                for key in storage.get("datasets") or [""]:
                    paths += [base / key / marker for marker in (".zarray", "zarr.json")]
    elif path.is_dir():
        # Metadata only: do not traverse any Zarr chunk tree.
        for marker in (path / ".zattrs", path / "zarr.json"):
            if marker.is_file():
                from ._reader import read_json, ngff_metadata
                attrs = read_json(marker)
                attrs = attrs.get("attributes", attrs)
                for multi in ngff_metadata(attrs):
                    for item in multi.get("datasets", []):
                        from ._reader import child_path
                        array = child_path(path, item["path"])
                        paths += [array / name for name in (".zarray", "zarr.json")]
    records = []
    for item in paths:
        try:
            stat = item.stat()
            records.append((str(item.relative_to(path.parent)), stat.st_size, stat.st_mtime_ns))
        except OSError:
            records.append((str(item), "missing"))
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


class RegionSource:
    """Metadata and lazy arrays; opening does not decode an entire image."""
    def __init__(self, path, image_index=0):
        self.path = Path(path).resolve()
        self.image_index = int(image_index)
        self._tif = self._pil = None
        self._stores = []
        self._loaded = {}
        self.arrays = None
        self.pixel_scale = None
        self.units = None
        self.rgb = False
        try:
            manifest = workspace_manifest(self.path)
            if manifest or is_ome_zarr(self.path):
                storage = record = None
                target = self.path
                if manifest:
                    images = [r for r in read_workspace(manifest)["layers"]
                              if r.get("layer_type") == "Image"]
                    if not 0 <= self.image_index < len(images):
                        raise ValueError("Workspace Image layer index is unavailable")
                    record = images[self.image_index]
                    storage = record.get("storage") or {}
                    if storage.get("kind") != "zarr":
                        raise ValueError("Discover currently needs Zarr-backed workspace images")
                    target = storage_path(manifest, storage)
                    self.rgb = bool(record.get("rgb", False))
                self.arrays, axes, metadata = zarr_arrays(target, storage)
                self.shapes = [tuple(a.shape) for a in self.arrays]
                self.dtype = str(self.arrays[0].dtype)
                self.axes = [str(a).upper() for a in axes] or self._fallback_axes(self.shapes[0])
                from ._reader import _transforms
                self.pixel_scale = (record or {}).get("scale") or _transforms(metadata, len(self.axes))["scale"]
                self.units = [a.get("unit") if isinstance(a, dict) else None
                              for a in metadata.get("axes", [])]
            elif self.path.suffix.lower() in TIFF_SUFFIXES:
                if self.image_index:
                    raise ValueError("TIFF discovery currently uses its first series")
                self._tif = tifffile.TiffFile(self.path)
                self._levels = list(self._tif.series[0].levels)
                self.shapes = [tuple(level.shape) for level in self._levels]
                self.axes = list(self._levels[0].axes.upper())
                self.dtype = str(self._levels[0].dtype)
                self.rgb = "S" in self.axes and self.shapes[0][self.axes.index("S")] in (3, 4)
            else:
                if self.image_index:
                    raise ValueError("Only workspace images support an Image layer index")
                self._pil = Image.open(self.path)
                if self._pil.mode not in ("L", "I", "F", "I;16", "I;16B", "1"):
                    self.rgb = True
                self.shapes = [(self._pil.height, self._pil.width) + ((3,) if self.rgb else ())]
                self.axes = ["Y", "X"] + (["S"] if self.rgb else [])
                self.dtype = "uint8" if self.rgb else str(np.dtype(
                    "uint16" if "16" in self._pil.mode else
                    "int32" if self._pil.mode == "I" else
                    "float32" if self._pil.mode == "F" else "uint8"))
            self.axes = [f"{name}{index}" if name not in ("Y", "X", "S") and self.axes.count(name) > 1 else name
                         for index, name in enumerate(self.axes)]
            if self.axes.count("Y") != 1 or self.axes.count("X") != 1 or len(self.axes) != len(self.shapes[0]):
                raise ValueError("Discover requires identifiable Y/X axes")
            if self.rgb and self.axes[-1] != "S":
                raise ValueError("RGB Discover currently requires a final sample axis")
            self.y_axis, self.x_axis = self.axes.index("Y"), self.axes.index("X")
            self.revision = fingerprint(self.path)
        except Exception:
            self.close()
            raise

    def _fallback_axes(self, shape):
        tail = ["Y", "X", "S"] if self.rgb else ["Y", "X"]
        return [f"D{i}" for i in range(len(shape) - len(tail))] + tail

    def describe(self):
        return {
            "shape": list(self.shapes[0]), "levels": [list(s) for s in self.shapes],
            "axes": self.axes, "dtype": self.dtype, "rgb": self.rgb,
            "pixel_scale": self.pixel_scale, "units": self.units,
            "plane": [None if a in ("Y", "X") or (a == "S" and self.rgb) else n // 2
                      for a, n in zip(self.axes, self.shapes[0])],
            "source_revision": self.revision, "revision_method": "metadata-stat-v1",
        }

    def _array(self, level):
        if self.arrays is not None:
            return self.arrays[level]
        if level in self._loaded:
            return self._loaded[level]
        if self._tif is not None:
            series = self._levels[level]
            # Uncompressed contiguous TIFFs may expose a whole image as one
            # enormous strip. Map their native pixels read-only and copy only
            # the bounded ROI, rather than decoding that strip through Zarr.
            try:
                array = tifffile.memmap(self.path, series=0, level=level, mode="r")
            except ValueError:
                array = None  # Compressed/tiled/noncontiguous TIFF: bounded decode below.
            if array is not None:
                self._loaded[level] = array
                return array
            try:
                import zarr
                store = series.aszarr()
                self._stores.append(store)
                array = zarr.open(store, mode="r")
            except ImportError:
                if math.prod(series.shape) * series.dtype.itemsize > MAX_DECODE_BYTES:
                    raise ValueError("Large TIFF region access requires zarr; use the discover extra or OME-Zarr")
                array = series.asarray()
        else:
            if math.prod(self.shapes[0]) * np.dtype(self.dtype).itemsize > MAX_DECODE_BYTES:
                raise ValueError("Image exceeds the bounded decode budget; convert it to tiled OME-Zarr")
            source = self._pil.convert("RGB") if self.rgb else self._pil
            array = np.asarray(source)
        self._loaded[level] = array
        return array

    def read(self, bounds, plane=None, level=0):
        if not 0 <= int(level) < len(self.shapes):
            raise ValueError(f"Requested pyramid level {level} is unavailable")
        level = int(level)
        shape = self.shapes[level]
        base = self.shapes[0]
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all():
            raise ValueError("Region bounds must be [[y_start,y_stop],[x_start,x_stop]]")
        if any(lo < 0 or hi <= lo or hi > base[axis]
               for (lo, hi), axis in zip(bounds, (self.y_axis, self.x_axis))):
            raise ValueError("Region bounds lie outside the source image")
        plane = list(plane if plane is not None else self.describe()["plane"])
        if len(plane) != len(base):
            raise ValueError("Saved plane dimensions do not match the source")
        selection = []
        for axis, (name, n) in enumerate(zip(self.axes, shape)):
            if axis in (self.y_axis, self.x_axis):
                interval = bounds[0 if axis == self.y_axis else 1]
                ratio = n / base[axis]
                lo = max(0, math.floor(interval[0] * ratio))
                hi = min(n, math.ceil(interval[1] * ratio))
                selection.append(slice(lo, hi))
            elif name == "S" and self.rgb:
                selection.append(slice(None))
            else:
                value = plane[axis]
                if not isinstance(value, (int, np.integer)) or not 0 <= value < base[axis]:
                    raise ValueError("Invalid nonspatial plane coordinate")
                selection.append(min(n - 1, int(value * n / base[axis])))
        pixels = math.prod((sel.stop - sel.start) if sel.stop is not None else n
                           for sel, n in zip(selection, shape) if isinstance(sel, slice))
        if pixels > MAX_PATCH_PIXELS:
            raise ValueError("ROI exceeds the 4-million-sample limit; choose a coarser pyramid level")
        array = self._array(level)
        if hasattr(array, "chunks"):
            chunks = array.chunks
            chunk_bytes = math.prod(chunks) * np.dtype(array.dtype).itemsize
            touched = math.prod(
                (math.ceil((sel.stop if sel.stop is not None else n) / c) - (sel.start or 0) // c)
                if isinstance(sel, slice) else 1
                for sel, n, c in zip(selection, shape, chunks))
            if chunk_bytes * touched > MAX_DECODE_BYTES:
                raise ValueError("Region exceeds the 128 MiB chunk decode budget; choose a coarser level")
        result = np.asarray(array[tuple(selection)])
        kept = [a for a, sel in zip(self.axes, selection) if isinstance(sel, slice)]
        order = [kept.index("Y"), kept.index("X")] + ([kept.index("S")] if self.rgb else [])
        return result.transpose(order).copy()

    def close(self):
        for array in self._loaded.values():
            if isinstance(array, np.memmap):
                array._mmap.close()
        self._loaded.clear()
        for store in self._stores:
            if hasattr(store, "close"):
                store.close()
        if self._tif is not None:
            self._tif.close()
        if self._pil is not None:
            self._pil.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
