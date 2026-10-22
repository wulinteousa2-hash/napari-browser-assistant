"""Background preparation, exact cosine retrieval, and versioned dataset export."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import threading

import numpy as np
from PIL import Image
import tifffile

from ._discovery_io import RegionSource
from ._encoders import Encoder, gray_image, unit_vector
from ._study import atomic_json, now, uid

class Cancelled(Exception):
    pass


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled("Operation cancelled; saved studies and reviews are unchanged")


class VectorCache:
    """Rebuildable vectors, isolated by study and encoder/run provenance."""
    def __init__(self, root, study_id):
        self.root = Path(root) / hashlib.sha256(study_id.encode()).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "vectors.sqlite3"
        with sqlite3.connect(self.path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS vectors (run_id TEXT, region_id TEXT, vector BLOB, PRIMARY KEY(run_id,region_id))")

    def write_run(self, run_id, vectors):
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM vectors WHERE run_id=?", (run_id,))
            conn.executemany("INSERT INTO vectors VALUES (?,?,?)",
                             [(run_id, key, np.asarray(value, dtype="<f4").tobytes()) for key, value in vectors.items()])

    def load(self, run_id):
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute("SELECT region_id,vector FROM vectors WHERE run_id=?", (run_id,)).fetchall()
        return {key: np.frombuffer(blob, dtype="<f4").copy() for key, blob in rows}

    def thumbnail(self, region_id, revision):
        return self.root / (hashlib.sha256((region_id + revision).encode()).hexdigest() + ".png")

    def save_thumbnail(self, region, data):
        target = self.thumbnail(region["id"], region["source_revision"])
        if not target.exists():
            image = Image.fromarray((gray_image(data) * 255).astype("uint8"))
            descriptor, temporary = tempfile.mkstemp(dir=self.root, suffix=".png")
            os.close(descriptor)
            try:
                image.save(temporary)
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
        return target


def _grid(image, level, tile_size, maximum):
    if not 0 <= level < len(image["levels"]):
        raise ValueError(f"Requested level {level} is unavailable")
    y, x = image["axes"].index("Y"), image["axes"].index("X")
    height, width = image["levels"][level][y], image["levels"][level][x]
    rows, columns = math.ceil(height / tile_size), math.ceil(width / tile_size)
    total = rows * columns
    indices = np.unique(np.linspace(0, total - 1, min(total, maximum), dtype=int))
    sy, sx = image["shape"][y] / height, image["shape"][x] / width
    for index in indices:
        row, column = divmod(int(index), columns)
        yield [[math.floor(row * tile_size * sy), min(image["shape"][y], math.ceil((row + 1) * tile_size * sy))],
               [math.floor(column * tile_size * sx), min(image["shape"][x], math.ceil((column + 1) * tile_size * sx))]]


def prepare(study, cache_root, config, cancel=None, progress=None):
    """Return a new study snapshot; the GUI adopts it only after successful work."""
    study = study.clone()
    images = study.document["images"]
    if not images:
        raise ValueError("Add Library images to this study first")
    maximum = int(config.get("max_regions", 256))
    tile = int(config.get("tile_size", 256))
    level = int(config.get("level", 0))
    if not 1 <= maximum <= 10000 or not 16 <= tile <= 2048 or len(images) > maximum:
        raise ValueError("Choose 1–10,000 regions, tile size 16–2048, and at least one region per image")
    encoder = Encoder(config.get("encoder", "texture"), config.get("weights", ""), config.get("device", "cpu"))
    cache = VectorCache(cache_root, study.document["id"])
    run_id, vectors, errors, ids = uid(), {}, [], []
    total_candidates = 0
    for number, image in enumerate(images):
        check_cancel(cancel)
        try:
            study.assert_current(image)
            budget = maximum // len(images) + (1 if number < maximum % len(images) else 0)
            y, x = image["axes"].index("Y"), image["axes"].index("X")
            shape = image["levels"][level]
            total_candidates += math.ceil(shape[y] / tile) * math.ceil(shape[x] / tile)
            with RegionSource(study.resolve(image["path"]), image["image_index"]) as source:
                for bounds in _grid(image, level, tile, budget):
                    check_cancel(cancel)
                    region = study.add_region(image, bounds)
                    data = source.read(bounds, region["plane"], level)
                    vectors[region["id"]] = encoder.encode(data)
                    ids.append(region["id"])
                    cache.save_thumbnail(region, data)
                    if progress:
                        progress(len(ids), maximum, f"Prepared {len(ids)} regions · {Path(image['path']).name}")
                study.assert_current(image)
        except (Cancelled, ImportError):
            raise
        except Exception as error:
            # Do not keep partial entries for an unreadable/stale image.
            for region_id in list(ids):
                if study.row("regions", region_id)["image_id"] == image["id"]:
                    vectors.pop(region_id, None)
                    ids.remove(region_id)
            errors.append(f"{Path(image['path']).name}: {error}")
    check_cancel(cancel)
    if not ids:
        raise ValueError("No regions could be prepared. " + "; ".join(errors[:3]))
    dimensions = {value.size for value in vectors.values()}
    if len(dimensions) != 1:
        raise ValueError("Encoder returned inconsistent embedding dimensions")
    saved_config = dict(config)
    if saved_config.get("weights"):
        saved_config["weights"] = os.path.relpath(Path(saved_config["weights"]).expanduser().absolute(), study.path.parent)
    run = {"id": run_id, "created": now(), "encoder": encoder.metadata, "encoder_key": encoder.key,
           "config": saved_config, "region_ids": ids, "dimension": next(iter(dimensions)),
           "sampling": "uniform grid indices per image; bounded representative plane",
           "total_grid_regions": total_candidates, "prepared_regions": len(ids), "errors": errors}
    study.document["runs"].append(run)
    cache.write_run(run_id, vectors)
    return study, run


def search(study, cache_root, run_id, query_region_id, limit=30, cancel=None):
    study = study.clone()
    run = study.row("runs", run_id)
    cache = VectorCache(cache_root, study.document["id"])
    vectors = cache.load(run_id)
    if any(key not in vectors for key in run["region_ids"]):
        raise ValueError("Embedding cache is missing; prepare a new run. Saved searches/reviews remain available.")
    query = study.row("regions", query_region_id)
    image = study.row("images", query["image_id"])
    study.assert_current(image, query["source_revision"])
    config = dict(run["config"])
    weights = str(study.resolve(config["weights"])) if config.get("weights") else ""
    encoder = Encoder(config.get("encoder", "texture"), weights, config.get("device", "cpu"))
    if encoder.key != run["encoder_key"]:
        raise ValueError("Encoder/checkpoint or preprocessing changed; prepare a new run")
    with RegionSource(study.resolve(image["path"]), image["image_index"]) as source:
        pixels = source.read(query["bounds"], query["plane"], int(config.get("level", 0)))
    query_vector = encoder.encode(pixels)
    cache.save_thumbnail(query, pixels)
    study.assert_current(image, query["source_revision"])
    results, stale = [], []
    checked = {}
    for identifier in run["region_ids"]:
        check_cancel(cancel)
        candidate = study.row("regions", identifier)
        source = study.row("images", candidate["image_id"])
        key = (source["id"], candidate["source_revision"])
        if key not in checked:
            try:
                study.assert_current(source, candidate["source_revision"])
                checked[key] = True
            except (OSError, ValueError):
                checked[key] = False
                stale.append(source["path"])
        if not checked[key] or identifier == query_region_id:
            continue
        # Exclude overlapping same-plane regions from trivial self-matches.
        if candidate["image_id"] == query["image_id"] and candidate["plane"] == query["plane"]:
            a, b = np.asarray(candidate["bounds"]), np.asarray(query["bounds"])
            if np.all(np.minimum(a[:, 1], b[:, 1]) > np.maximum(a[:, 0], b[:, 0])):
                continue
        score = float(np.dot(unit_vector(vectors[identifier]), query_vector))
        results.append({"region_id": identifier, "score": float(np.clip(score, -1, 1))})
    results.sort(key=lambda result: (-result["score"], result["region_id"]))
    results = results[:max(1, min(500, int(limit)))]
    for rank, result in enumerate(results, 1):
        result["rank"] = rank
    session = {"id": uid(), "created": now(), "run_id": run_id,
               "query_region": query_region_id, "results": results, "excluded_sources": sorted(set(stale)),
               "metric": "cosine", "score_meaning": "appearance similarity; not probability or biological identity"}
    study.document["sessions"].append(session)
    return study, session


def export_dataset(study, destination, task="self-supervised", cancel=None, progress=None, level=0):
    """Export native-valued TIFF crops and a trainer-neutral, immutable manifest."""
    study = study.clone()
    if task not in ("self-supervised", "classification", "similarity-pairs"):
        raise ValueError("Unsupported dataset task")
    selected = set(study.document["selections"])
    if not selected:
        raise ValueError("Explicitly include regions in the dataset first")
    annotations = {a["region_id"]: a for a in study.document["annotations"]}
    classes = {c["id"]: c["name"] for c in study.document["classes"]}
    pairs = []
    reviews = {(r["session_id"], r["region_id"]): r["decision"] for r in study.document["reviews"]}
    if task == "similarity-pairs":
        for session in study.document["sessions"]:
            for result in session["results"]:
                key = (session["id"], result["region_id"])
                if result["region_id"] in selected and reviews.get(key) in ("relevant", "irrelevant"):
                    pairs.append({"query": session["query_region"], "candidate": result["region_id"],
                                  "relevant": reviews[key] == "relevant", "session_id": session["id"]})
        if not pairs:
            raise ValueError("Pair export needs selected results with relevant/irrelevant reviews")
        selected.update(pair["query"] for pair in pairs)
    if task == "classification" and any(key not in annotations or annotations[key]["status"] != "confirmed"
                                        for key in selected):
        raise ValueError("Classification export requires a confirmed biological class for every selected region")
    for key in selected:
        region = study.row("regions", key)
        study.assert_current(study.row("images", region["image_id"]), region["source_revision"])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".browser-export-", dir=destination))
    dataset_id = uid()
    final = destination / ("dataset-" + dataset_id[:12])
    samples, opened = [], {}
    label_ids = {key: index for index, key in enumerate(sorted(classes))}
    try:
        (staging / "crops").mkdir()
        with ExitStack() as stack:
            for index, identifier in enumerate(sorted(selected)):
                check_cancel(cancel)
                region = study.row("regions", identifier)
                image = study.row("images", region["image_id"])
                if image["id"] not in opened:
                    opened[image["id"]] = stack.enter_context(RegionSource(study.resolve(image["path"]), image["image_index"]))
                # Native values at an explicit source pyramid level, never display thumbnails.
                source = opened[image["id"]]
                crop = source.read(region["bounds"], region["plane"], level)
                y, x = image["axes"].index("Y"), image["axes"].index("X")
                pixel_scale = image.get("pixel_scale")
                crop_scale = ([pixel_scale[axis] * image["shape"][axis] / source.shapes[level][axis]
                               for axis in (y, x)] if pixel_scale else None)
                relative = "crops/" + identifier + ".tif"
                tifffile.imwrite(staging / relative, crop,
                                 photometric="rgb" if crop.ndim == 3 else "minisblack")
                annotation = annotations.get(identifier)
                group = image["group_id"]
                samples.append({
                    "region_id": identifier, "crop": relative,
                    "source_path": os.path.relpath(study.resolve(image["path"]), final),
                    "image_id": image["id"], "image_index": image["image_index"],
                    "bounds": region["bounds"], "plane": region["plane"],
                    "source_revision": region["source_revision"], "group_id": group,
                    "split": study.document["groups"].get(group, "unassigned"),
                    "annotation": annotation,
                    "class_index": label_ids[annotation["class_id"]] if annotation else None,
                    "axes": "YXS" if crop.ndim == 3 else "YX", "dtype": str(crop.dtype),
                    "crop_level": int(level), "crop_shape": list(crop.shape),
                    "crop_pixel_scale_yx": crop_scale, "source_units": image.get("units"),
                })
                if progress:
                    progress(index + 1, len(selected), f"Exported {index + 1}/{len(selected)} native-valued crops")
            for identifier in selected:
                region = study.row("regions", identifier)
                study.assert_current(study.row("images", region["image_id"]), region["source_revision"])
        by_id = {sample["region_id"]: sample for sample in samples}
        for pair in pairs:
            splits = {by_id[pair[key]]["split"] for key in ("query", "candidate")}
            if len(splits) > 1:
                raise ValueError("A reviewed pair spans different dataset splits; assign compatible specimen splits first")
            pair["split"] = next(iter(splits))
        manifest = {"format": "napari-browser-training-dataset", "version": 1, "id": dataset_id,
                    "created": now(), "study_id": study.document["id"], "task": task,
                    "study_path": os.path.relpath(study.path, final),
                    "run_reference_base": "study_directory",
                    "label_scope": "region-category; no pixel masks generated",
                    "classes": {str(label_ids[key]): classes[key] for key in label_ids},
                    "class_ids": {str(label_ids[key]): key for key in label_ids},
                    "samples": samples, "pairs": pairs,
                    "runs": study.document["runs"], "sessions": study.document["sessions"],
                    "split_policy": "explicit specimen/group assignments; unassigned requires assignment before training"}
        atomic_json(staging / "dataset.json", manifest)
        check_cancel(cancel)
        if final.exists():
            raise FileExistsError(final)
        staging.rename(final)
        return final / "dataset.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
