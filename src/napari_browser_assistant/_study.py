"""Portable study records, atomic autosave, and explicit review/annotation decisions."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile
import uuid

import numpy as np

from ._discovery_io import RegionSource, fingerprint
from ._reader import read_json

FORMAT = "napari-browser-analysis"


def now():
    return datetime.now(timezone.utc).isoformat()


def uid():
    return uuid.uuid4().hex


def atomic_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".browser-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None


class Study:
    def __init__(self, path, document):
        self.path = Path(path).resolve()
        self.document = document
        self._disk_digest = _digest(self.path)
        self.validate()
        self._lookups = {}

    @classmethod
    def create(cls, path, name="Untitled study"):
        if Path(path).exists():
            raise FileExistsError("Study already exists; open it instead")
        study = cls(path, {"format": FORMAT, "version": 1, "id": uid(), "name": name,
                          "created": now(), "updated": now(), "images": [], "regions": [],
                          "runs": [], "sessions": [], "classes": [], "annotations": [],
                          "reviews": [], "selections": [], "groups": {}, "datasets": []})
        study.save()
        return study

    @classmethod
    def load(cls, path):
        return cls(path, read_json(path))

    def validate(self):
        doc = self.document
        if not isinstance(doc, dict) or doc.get("format") != FORMAT or doc.get("version") != 1:
            raise ValueError("Not a supported Browser Assistant study (version 1)")
        valid_id = lambda value: isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value) is not None
        if not valid_id(doc.get("id")):
            raise ValueError("Study ID is missing")
        for key in ("images", "regions", "runs", "sessions", "classes", "annotations",
                    "reviews", "selections", "datasets"):
            if not isinstance(doc.get(key), list):
                raise ValueError(f"Study field {key} must be a list")
        if not isinstance(doc.get("groups"), dict):
            raise ValueError("Study groups must be an object")
        for key in ("images", "regions", "runs", "sessions", "classes"):
            ids = [row.get("id") for row in doc[key] if isinstance(row, dict)]
            if len(ids) != len(doc[key]) or any(not valid_id(x) for x in ids) or len(set(ids)) != len(ids):
                raise ValueError(f"Invalid or duplicate IDs in {key}")
        images = {row["id"] for row in doc["images"]}
        records = doc.get("ocr_records", [])
        if not isinstance(records, list):
            raise ValueError("Study OCR records must be a list")
        record_ids = set()
        for record in records:
            if not isinstance(record, dict) or not valid_id(record.get("id")) or record["id"] in record_ids:
                raise ValueError("Invalid or duplicate OCR record ID")
            record_ids.add(record["id"])
            if record.get("image_id") not in images or record.get("review_status") not in ("unverified", "reviewed"):
                raise ValueError("Invalid OCR image or review status")
            for key in ("original_text", "corrected_text", "source_revision", "engine", "engine_version", "extracted_at"):
                if not isinstance(record.get(key), str):
                    raise ValueError(f"Invalid OCR {key}")
            if record.get("scope") not in ("footer", "full") or not isinstance(record.get("settings"), dict):
                raise ValueError("Invalid OCR settings")
            for key in ("fields", "original_fields"):
                if not isinstance(record.get(key), dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in record[key].items()):
                    raise ValueError("Invalid OCR fields")
            if not isinstance(record.get("words"), list):
                raise ValueError("Invalid OCR words")
            for word in record["words"]:
                if not isinstance(word, dict) or not isinstance(word.get("text"), str):
                    raise ValueError("Invalid OCR word")
                box = np.asarray(word.get("bbox"), dtype=float)
                confidence = word.get("confidence")
                if (box.shape != (4,) or not np.isfinite(box).all() or np.any(box < 0)
                        or not isinstance(confidence, (int, float)) or not np.isfinite(confidence)):
                    raise ValueError("Invalid OCR word coordinates/confidence")
        regions = {row["id"] for row in doc["regions"]}
        runs = {row["id"] for row in doc["runs"]}
        sessions = {row["id"] for row in doc["sessions"]}
        classes = {row["id"] for row in doc["classes"]}
        for image in doc["images"]:
            if not isinstance(image.get("path"), str) or not isinstance(image.get("shape"), list):
                raise ValueError("Invalid study image record")
        for region in doc["regions"]:
            if region.get("image_id") not in images:
                raise ValueError("Region references an unknown image")
            bounds = np.asarray(region.get("bounds"), dtype=float)
            if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
                raise ValueError("Invalid saved region bounds")
        for session in doc["sessions"]:
            if session.get("query_region") not in regions or session.get("run_id") not in runs:
                raise ValueError("Search references an unknown region/run")
            for result in session.get("results", []):
                if result.get("region_id") not in regions or not np.isfinite(result.get("score", np.nan)):
                    raise ValueError("Invalid saved search result")
        for review in doc["reviews"]:
            if review.get("session_id") not in sessions or review.get("region_id") not in regions:
                raise ValueError("Review references an unknown session/region")
            if review.get("decision") not in ("relevant", "irrelevant", "uncertain"):
                raise ValueError("Invalid relevance review")
        for annotation in doc["annotations"]:
            if annotation.get("region_id") not in regions or annotation.get("class_id") not in classes:
                raise ValueError("Invalid biological annotation")
            if annotation.get("status") not in ("proposed", "confirmed"):
                raise ValueError("Invalid annotation review status")
        if any(region_id not in regions for region_id in doc["selections"]):
            raise ValueError("Dataset selection references an unknown region")
        if any(value not in ("unassigned", "train", "validation", "test") for value in doc["groups"].values()):
            raise ValueError("Invalid group split")

    def clone(self):
        other = Study(self.path, deepcopy(self.document))
        other._disk_digest = self._disk_digest
        return other

    def save(self):
        self.validate()
        if _digest(self.path) != self._disk_digest:
            raise RuntimeError("Study changed on disk; reopen it before saving to avoid overwriting reviews")
        self.document["updated"] = now()
        atomic_json(self.path, self.document)
        self._disk_digest = _digest(self.path)

    def save_as(self, path):
        path = Path(path).resolve()
        if path == self.path:
            self.save()
            return self
        if path.exists():
            raise FileExistsError("Choose a new filename for Save As")
        document = deepcopy(self.document)
        for image in document["images"]:
            image["path"] = os.path.relpath(self.resolve(image["path"]), path.parent)
        for run in document["runs"]:
            weights = run.get("config", {}).get("weights")
            if weights:
                run["config"]["weights"] = os.path.relpath(self.resolve(weights), path.parent)
        for dataset in document["datasets"]:
            dataset["manifest"] = os.path.relpath(self.resolve(dataset["manifest"]), path.parent)
        other = Study(path, document)
        other.save()
        return other

    def resolve(self, path):
        value = Path(path)
        return value if value.is_absolute() else Path(os.path.abspath(self.path.parent / value))

    def row(self, collection, identifier):
        rows = self.document[collection]
        lookup = self._lookups.get(collection)
        if lookup is None or len(lookup) != len(rows):
            lookup = {item["id"]: item for item in rows}
            self._lookups[collection] = lookup
        try:
            return lookup[identifier]
        except KeyError as error:
            raise ValueError(f"Unknown {collection} record: {identifier}") from error

    def add_image(self, path, image_index=0):
        path = Path(path).resolve()
        existing = next((image for image in self.document["images"]
                         if self.resolve(image["path"]) == path and image["image_index"] == image_index), None)
        with RegionSource(path, image_index) as source:
            description = source.describe()
        if existing:
            existing.update(description)
            return existing
        image = dict(description, id=uid(), path=os.path.relpath(path, self.path.parent),
                     image_index=int(image_index), group_id="")
        image["group_id"] = image["id"]
        self.document["groups"][image["group_id"]] = "unassigned"
        self.document["images"].append(image)
        return image

    def assert_current(self, image, revision=None):
        if fingerprint(self.resolve(image["path"])) != (revision or image["source_revision"]):
            raise ValueError("Source metadata changed; re-add the image and prepare a new run. Saved reviews are retained.")

    def add_region(self, image, bounds, plane=None):
        plane = list(plane if plane is not None else image["plane"])
        bounds = [[int(a), int(b)] for a, b in bounds]
        key = json.dumps([image["id"], image["source_revision"], bounds, plane])
        identifier = hashlib.sha256(key.encode()).hexdigest()[:32]
        lookup = self._lookups.get("regions")
        if lookup is None or len(lookup) != len(self.document["regions"]):
            lookup = {r["id"]: r for r in self.document["regions"]}
            self._lookups["regions"] = lookup
        if identifier in lookup:
            return lookup[identifier]
        region = {"id": identifier, "image_id": image["id"], "bounds": bounds,
                  "plane": plane, "coordinate_space": "level-0-pixels",
                  "source_revision": image["source_revision"]}
        self.document["regions"].append(region)
        lookup[identifier] = region
        return region

    def review(self, session_id, region_id, decision):
        session = self.row("sessions", session_id)
        if not any(result["region_id"] == region_id for result in session["results"]):
            raise ValueError("Region is not a result of this search")
        if decision not in ("relevant", "irrelevant", "uncertain"):
            raise ValueError("Unknown review decision")
        self.document["reviews"] = [r for r in self.document["reviews"]
                                    if (r["session_id"], r["region_id"]) != (session_id, region_id)]
        self.document["reviews"].append({"session_id": session_id, "region_id": region_id,
                                         "decision": decision, "origin": "human", "updated": now()})

    def set_class(self, region_id, name, status="proposed"):
        self.row("regions", region_id)
        name = name.strip()
        if not name or status not in ("proposed", "confirmed"):
            raise ValueError("Enter a class name and a valid review status")
        category = next((c for c in self.document["classes"] if c["name"].casefold() == name.casefold()), None)
        if category is None:
            category = {"id": uid(), "name": name}
            self.document["classes"].append(category)
        self.document["annotations"] = [a for a in self.document["annotations"] if a["region_id"] != region_id]
        self.document["annotations"].append({"region_id": region_id, "class_id": category["id"],
                                             "status": status, "origin": "human",
                                             "meaning": "region-category", "updated": now()})

    def select(self, region_id, included=True):
        self.row("regions", region_id)
        selected = set(self.document["selections"])
        (selected.add if included else selected.discard)(region_id)
        self.document["selections"] = sorted(selected)

    def set_group(self, image_id, group, split):
        if split not in ("unassigned", "train", "validation", "test") or not group.strip():
            raise ValueError("Enter a specimen/group and a valid split")
        image = self.row("images", image_id)
        image["group_id"] = group.strip()
        # Every image in a specimen shares this group split.
        self.document["groups"][group.strip()] = split
