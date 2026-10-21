"""Mask classification, conservative source pairing, and categorical previews."""
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image


def image_stem(path):
    stem = Path(path).stem
    # OME compound extensions are part of the format, not the specimen name.
    if stem.casefold().endswith(".ome"):
        stem = stem[:-4]
    return stem


def mask_source_stem(path):
    if Path(path).suffix.lower() == '.json':
        return None
    stem = image_stem(path)
    for suffix in ("_mask", "_m"):
        if stem.casefold().endswith(suffix) and len(stem) > len(suffix):
            return stem[:-len(suffix)]
    return None


def pair_masks(entries, annotations):
    """Suggest only unique same-directory matches; manual roles take precedence."""
    roles = {}
    for entry in entries:
        annotation = annotations.get(str(entry.path))
        roles[str(entry.path)] = (annotation[0] if annotation and annotation[2] == "manual"
                                 else "mask" if mask_source_stem(entry.path) else "image")
    sources = defaultdict(list)
    for entry in entries:
        if roles[str(entry.path)] == "image":
            sources[(str(entry.path.parent), image_stem(entry.path).casefold())].append(entry.path)
    result = []
    for entry in entries:
        path = str(entry.path)
        if roles[path] != "mask":
            result.append(replace(entry, role="image", source_path="", pairing=""))
            continue
        annotation = annotations.get(path)
        if annotation and annotation[2] == "manual":
            source = annotation[1]
            pairing = "Confirmed pair" if source else "Unpaired"
        else:
            stem = mask_source_stem(entry.path)
            candidates = sources.get((str(entry.path.parent), (stem or "").casefold()), [])
            source = str(candidates[0]) if len(candidates) == 1 else ""
            pairing = ("Suggested pair" if source else
                       "Ambiguous source" if candidates else "Missing source")
        result.append(replace(entry, role="mask", source_path=source, pairing=pairing))
    return result


def label_preview(data):
    """Color label IDs without contrast normalization or interpolating new IDs."""
    data = np.asarray(data)
    if data.ndim != 2 or data.dtype.kind not in "biu":
        raise ValueError("Mask preview requires a 2D integer/bool label plane; RGB and floating-point masks need conversion")
    if data.dtype.kind == "i" and np.any(data < 0):
        raise ValueError("Mask label IDs must be nonnegative")
    labels = data.astype(np.uint64, copy=False)
    # Stable color per ID (including sparse/large IDs); never allocate max(ID) colors.
    hashed = labels * np.uint64(11400714819323198485)
    rgba = np.empty(data.shape + (4,), dtype=np.uint8)
    for channel, shift in enumerate((0, 21, 42)):
        rgba[..., channel] = 64 + ((hashed >> np.uint64(shift)) % 192).astype(np.uint8)
    rgba[..., 3] = np.where(labels == 0, 0, 255)
    rgba[labels == 0, :3] = 0
    return Image.fromarray(rgba)


def layer_shape(layer):
    data = layer.data
    if getattr(layer, "multiscale", False):
        data = data[0]
    shape = tuple(data.shape)
    return shape[:-1] if getattr(layer, "rgb", False) else shape


def pair_shape_warning(source_layers, mask_layers):
    """Check shapes exposed by readers without materializing their arrays."""
    sources = [layer_shape(layer) for layer in source_layers if hasattr(layer, "data")]
    masks = [layer_shape(layer) for layer in mask_layers if hasattr(layer, "data")]
    if sources and masks and any(mask not in sources for mask in masks):
        return "Shape mismatch or extra source axes: verify image/mask alignment before training"
    return ""
