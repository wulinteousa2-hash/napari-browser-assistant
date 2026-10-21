"""Read-only, lazy OME-Zarr and Label Assistant image I/O.

JSON is a dataset manifest, never pixel data. Only the recognized workspace schema
is accepted. Saved writable storage modes are deliberately ignored.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WORKSPACE_FORMAT = "napari-label-assistant-workspace"
WORKSPACE_SUFFIX = ".label-assistant.json"
DATA_SUFFIX = "_label_assistant_data"


def read_json(path, limit=64 * 1024 * 1024):
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"JSON metadata exceeds {limit // 1024 // 1024} MiB: {path}")
    return json.loads(data)


def read_workspace(path):
    path = Path(path)
    document = read_json(path)
    if not isinstance(document, dict) or document.get("format") != WORKSPACE_FORMAT:
        raise ValueError("Not a Label Assistant workspace")
    if document.get("version") != 1:
        raise ValueError(f"Unsupported Label Assistant workspace version: {document.get('version')}")
    layers = document.get("layers")
    if not isinstance(layers, list) or not all(isinstance(layer, dict) for layer in layers):
        raise ValueError("Workspace layers must be a list of records")
    if any(not isinstance(layer.get("storage"), dict) for layer in layers):
        raise ValueError("Workspace layer storage must be an object")
    return document


def workspace_manifest(path):
    """Resolve a JSON manifest or its conventionally named companion directory."""
    path = Path(path)
    if path.name.endswith(DATA_SUFFIX):
        path = path.with_name(path.name[:-len(DATA_SUFFIX)] + WORKSPACE_SUFFIX)
    if path.suffix.lower() != ".json" or not path.is_file():
        return None
    try:
        read_workspace(path)
    except (OSError, ValueError, TypeError):
        return None
    return path


def storage_path(manifest, storage):
    value = storage.get("path")
    if not isinstance(value, str) or not value or "://" in value:
        raise ValueError("Workspace storage must reference a local path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(manifest).parent / path


def child_path(root, key):
    """Internal array keys must remain inside their Zarr store."""
    if not isinstance(key, str):
        raise ValueError("Zarr array paths must be strings")
    root = Path(root).resolve()
    result = (root / key).resolve()
    try:
        result.relative_to(root)
    except ValueError as error:
        raise ValueError("Invalid OME-Zarr dataset path") from error
    return result


def ngff_metadata(attributes):
    # NGFF 0.4 (Zarr v2) and NGFF 0.5 (Zarr v3) metadata locations.
    metadata = attributes.get("ome", attributes)
    return metadata.get("multiscales", []) if hasattr(metadata, "get") else []


def is_ome_zarr(path):
    path = Path(path)
    try:
        if (path / "zarr.json").is_file():
            document = read_json(path / "zarr.json", 8 * 1024 * 1024)
            attrs = document.get("attributes", {})
        else:
            attrs = read_json(path / ".zattrs", 8 * 1024 * 1024)
        return bool(ngff_metadata(attrs))
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def require_zarr():
    try:
        import zarr
    except ImportError as error:
        raise ValueError("Install napari-browser-assistant[ome-zarr] for dataset I/O") from error
    return zarr


def zarr_arrays(path, storage=None):
    """Return array handles, axes, and NGFF metadata without reading pixels."""
    zarr = require_zarr()
    root = Path(path)
    group = None
    metadata = {}
    try:
        group = zarr.open_group(str(root), mode="r")
        multiscales = ngff_metadata(group.attrs)
        if multiscales:
            metadata = multiscales[0]
    except (ValueError, KeyError, FileNotFoundError):
        if storage is None:
            raise
    axes = [axis["name"] if isinstance(axis, dict) else axis
            for axis in metadata.get("axes", [])]
    if storage is not None:
        base = child_path(root, storage.get("array_path") or "")
        if storage.get("multiscale"):
            keys = storage.get("datasets")
            if not isinstance(keys, list) or not keys:
                raise ValueError("Multiscale workspace image has no datasets")
            arrays = [zarr.open_array(str(child_path(base, key)), mode="r") for key in keys]
        else:
            arrays = [zarr.open_array(str(base), mode="r")]
    else:
        if not metadata:
            raise ValueError("No OME multiscales metadata")
        datasets = metadata.get("datasets", [])
        if not datasets:
            raise ValueError("OME-Zarr has no image datasets")
        arrays = [zarr.open_array(str(child_path(root, dataset["path"])), mode="r")
                  for dataset in datasets]
    if not arrays or any(array.ndim != arrays[0].ndim for array in arrays):
        raise ValueError("Pyramid levels have inconsistent dimensions")
    if any(any(b > a for a, b in zip(previous.shape, current.shape))
           for previous, current in zip(arrays, arrays[1:])):
        raise ValueError("Pyramid levels must be ordered from largest to smallest")
    return arrays, axes, metadata


def primary_image(path):
    manifest = workspace_manifest(path)
    if manifest is None:
        raise ValueError("Not a supported Label Assistant workspace")
    document = read_workspace(manifest)
    images = [record for record in document["layers"] if record.get("layer_type") == "Image"]
    if not images:
        raise ValueError("Workspace has no Image layer to preview/open")
    return manifest, images[0]


def _transforms(metadata, ndim):
    scale, translate = np.ones(ndim), np.zeros(ndim)
    datasets = metadata.get("datasets", [])
    transforms = (datasets[0].get("coordinateTransformations", []) if datasets else [])
    transforms = transforms + metadata.get("coordinateTransformations", [])
    for transform in transforms:
        kind = transform.get("type")
        if kind == "identity":
            continue
        if kind not in ("scale", "translation"):
            raise ValueError(f"Unsupported OME coordinate transformation: {kind}")
        values = np.asarray(transform.get(kind), dtype=float)
        if values.shape != (ndim,) or not np.isfinite(values).all():
            raise ValueError("Invalid OME coordinate transformation dimensions")
        if kind == "scale":
            scale *= values
            translate *= values
        else:
            translate += values
    return {"scale": scale.tolist(), "translate": translate.tolist()}


def image_layer(path, storage=None, record=None, layer_type="image"):
    arrays, axes, metadata = zarr_arrays(path, storage)
    record = record or {}
    options = _transforms(metadata, arrays[0].ndim)
    for key in ("name", "visible", "opacity", "blending", "scale", "translate"):
        if key in record:
            options[key] = record[key]
    options.setdefault("name", Path(path).name)
    options["multiscale"] = len(arrays) > 1
    if layer_type == "image":
        for key in ("contrast_limits", "colormap", "gamma", "rgb"):
            if key in record:
                options[key] = record[key]
        options.setdefault("rgb", False)
        if "contrast_limits" not in options and arrays[0].dtype.kind in "ui":
            info = np.iinfo(arrays[0].dtype)
            options["contrast_limits"] = [int(info.min), int(info.max)]
    options["metadata"] = {"browser_dataset": str(path), "axes": axes,
                           "storage_mode": "read-only"}
    return (arrays if len(arrays) > 1 else arrays[0]), options, layer_type


def read_dataset(path, *, include_annotations=False):
    path = Path(path)
    manifest = workspace_manifest(path)
    if manifest is None:
        return [image_layer(path)]
    document = read_workspace(manifest)
    result = []
    for record in document["layers"]:
        kind = str(record.get("layer_type", "")).lower()
        if kind != "image" and not include_annotations:
            continue
        storage = record.get("storage") or {}
        if storage.get("kind") == "zarr" and kind in ("image", "labels"):
            result.append(image_layer(storage_path(manifest, storage), storage, record, kind))
        elif storage.get("kind") == "inline" and kind in ("shapes", "points"):
            common = ("name", "visible", "opacity", "blending", "scale", "translate", "features")
            specific = (("shape_type", "edge_width", "edge_color", "face_color") if kind == "shapes"
                        else ("size", "symbol", "face_color", "border_color"))
            options = {key: record[key] for key in common + specific
                       if key in record and record[key] is not None}
            result.append((record.get("data", []), options, kind))
        else:
            raise ValueError(f"Unsupported workspace layer storage: {record.get('name', kind)} ({storage.get('kind')})")
    if not any(layer[2] == "image" for layer in result):
        raise ValueError("Workspace has no Image layer")
    return result


def napari_get_reader(path):
    paths = [path] if isinstance(path, (str, Path)) else path
    if not paths:
        return None
    if not all(workspace_manifest(item) is not None or is_ome_zarr(item) for item in paths):
        return None
    return read_paths


def read_paths(path):
    paths = [path] if isinstance(path, (str, Path)) else path
    return [layer for item in paths for layer in read_dataset(item)]


def workspace_data_folder(manifest):
    """Only suppress the conventional companion folder, never arbitrary ancestors."""
    manifest = Path(manifest)
    if not manifest.name.endswith(WORKSPACE_SUFFIX):
        return None
    candidate = manifest.with_name(manifest.name[:-len(WORKSPACE_SUFFIX)] + DATA_SUFFIX)
    document = read_workspace(manifest)
    for record in document["layers"]:
        storage = record.get("storage") or {}
        if storage.get("kind") == "zarr":
            target = storage_path(manifest, storage)
            if candidate.resolve() in target.resolve().parents:
                return candidate
    return None
