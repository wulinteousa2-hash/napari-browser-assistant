"""Discover lives beside Library and uses its sources and the same napari canvas."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import os
import threading

import numpy as np
from qtpy.QtCore import QObject, QRunnable, QThreadPool, QTimer, QSize, Qt, Signal, Slot
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QProgressBar, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from ._discovery import Cancelled, VectorCache, check_cancel, export_dataset, prepare, search
from ._discovery_io import RegionSource
from ._study import Study, now, uid

ROI_ROLE = "browser_discover_role"


class WorkSignals(QObject):
    done = Signal(object, object)
    progress = Signal(int, int, str)


class Work(QRunnable):
    def __init__(self, function):
        super().__init__()
        self.signals = WorkSignals()
        self.function = function

    @Slot()
    def run(self):
        try:
            result = self.function(self.signals.progress.emit)
        except Exception as error:
            self.signals.done.emit(None, error)
        else:
            self.signals.done.emit(result, None)


def transforms(layer):
    return {"scale": tuple(layer.scale), "translate": tuple(layer.translate),
            "rotate": deepcopy(layer.rotate), "shear": deepcopy(layer.shear),
            "affine": np.array(layer.affine.affine_matrix, copy=True)}


def layer_reference(layer):
    reference = getattr(layer, "metadata", {}).get("browser_source")
    if reference:
        return reference["path"], int(reference.get("image_index", 0))
    path = getattr(getattr(layer, "source", None), "path", None)
    if path:
        return str(path), 0
    raise ValueError("Open a file-backed image from Library before drawing a query")


def region_from_vertices(study, path, image_index, vertices, viewer_transform=None):
    image = study.add_image(path, image_index)
    if viewer_transform is not None:
        image["viewer_transform"] = viewer_transform
    axes = [a for a in image["axes"] if not (a == "S" and image["rgb"])]
    points = np.asarray(vertices, dtype=float)
    if points.shape != (4, len(axes)) or not np.isfinite(points).all():
        raise ValueError("Select one rectangular ROI matching the image dimensions")
    y, x = axes.index("Y"), axes.index("X")
    bounds = []
    for axis, shape_axis in ((y, image["axes"].index("Y")), (x, image["axes"].index("X"))):
        values = points[:, axis]
        lo, hi = values.min(), values.max()
        if not np.all(np.isclose(values, lo) | np.isclose(values, hi)):
            raise ValueError("Use an axis-aligned rectangle in image coordinates")
        start = max(0, math.floor(float(lo)))
        stop = min(image["shape"][shape_axis], math.ceil(float(hi)))
        if stop <= start:
            raise ValueError("Query rectangle lies outside the image")
        bounds.append([start, stop])
    plane = list(image["plane"])
    for axis, name in enumerate(axes):
        if name not in ("Y", "X"):
            if np.ptp(points[:, axis]) > 1e-4:
                raise ValueError("Query ROI must lie on one nonspatial plane")
            value = int(round(float(points[0, axis])))
            target_axis = image["axes"].index(name)
            if not 0 <= value < image["shape"][target_axis]:
                raise ValueError("Query plane lies outside the image")
            plane[target_axis] = value
    return study.add_region(image, bounds, plane)


class DiscoverWidget(QWidget):
    study_changed = Signal()
    operation_changed = Signal(bool)
    _work_progress = Signal(int, int, str)

    def __init__(self, browser):
        super().__init__(browser)
        self.browser = browser
        self.study = None
        self.cache_root = browser.cache_root / "discover"
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self._work = None
        self._cancel = threading.Event()
        self._query_image = self._query_shapes = None
        self._checkpoint_kind = None
        self._build_ui()
        self.encoder.currentIndexChanged.connect(self._encoder_selected)
        self._encoder_selected()
        self.weights.editingFinished.connect(self._remember_checkpoint)
        QTimer.singleShot(0, self.restore_last_study)

    @property
    def viewer(self):
        return self.browser.viewer

    def button(self, text, function):
        button = QPushButton(text)
        button.clicked.connect(lambda checked=False: self._guard(function))
        return button

    def _guard(self, function):
        try:
            function()
        except Exception as error:
            self.status.setText(str(error))

    def _build_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel("Discover · saved image studies")
        title.setStyleSheet("font-size: 18px; font-weight: bold")
        layout.addWidget(title)
        toolbar = QHBoxLayout()
        self.new_button = self.button("New study", self.new_study)
        self.open_button = self.button("Open study…", self.open_study_dialog)
        self.save_as_button = self.button("Save As…", self.save_as_dialog)
        for button in (self.new_button, self.open_button, self.save_as_button):
            toolbar.addWidget(button)
        layout.addLayout(toolbar)
        self.study_label = QLabel("Create a study, then add images from the Library view.")
        self.study_label.setWordWrap(True)
        self.study_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.study_label)

        self.add_button = self.button("Add Library images to study", self.add_library_images)
        self.add_button.setToolTip("Adds image entries in the current Library filter, using the saved index; excludes masks")
        layout.addWidget(self.add_button)
        form = QFormLayout()
        self.encoder = QComboBox()
        self.encoder.addItem("Texture baseline — no learned biology", "texture")
        self.encoder.addItem("ResNet18 — local checkpoint", "resnet18")
        self.encoder.addItem("EM-DINO — local checkpoint + omniem", "emdino")
        form.addRow("Encoder", self.encoder)
        weight_row = QHBoxLayout()
        self.weights = QLineEdit()
        self.weights.setPlaceholderText("Local .safetensors / .pth / EM-DINO .pt")
        self.weight_button = self.button("Choose…", self.choose_weights)
        weight_row.addWidget(self.weights, 1)
        weight_row.addWidget(self.weight_button)
        form.addRow("Checkpoint", weight_row)
        self.device = QComboBox()
        self.device.addItems(["cpu", "cuda"])
        form.addRow("Device", self.device)
        sampling = QHBoxLayout()
        self.tile_size = QSpinBox()
        self.tile_size.setRange(16, 2048)
        self.tile_size.setValue(256)
        self.tile_size.setSuffix(" px")
        self.level = QSpinBox()
        self.level.setRange(0, 12)
        self.maximum = QSpinBox()
        self.maximum.setRange(1, 10000)
        self.maximum.setValue(256)
        sampling.addWidget(QLabel("Tile"))
        sampling.addWidget(self.tile_size)
        sampling.addWidget(QLabel("Pyramid level"))
        sampling.addWidget(self.level)
        sampling.addWidget(QLabel("Max regions"))
        sampling.addWidget(self.maximum)
        form.addRow("Sampling", sampling)
        layout.addLayout(form)
        self.scale_hint = QLabel("Tiles use the chosen level's pixels; other axes use a saved representative plane. Check physical scale and acquisition batch when comparing images.")
        self.scale_hint.setWordWrap(True)
        layout.addWidget(self.scale_hint)
        prepare_row = QHBoxLayout()
        self.prepare_button = self.button("Prepare study embeddings", self.prepare_study)
        self.runs = QComboBox()
        self.runs.currentIndexChanged.connect(self._run_selected)
        prepare_row.addWidget(self.prepare_button)
        prepare_row.addWidget(self.runs, 1)
        layout.addLayout(prepare_row)
        query_row = QHBoxLayout()
        self.draw_button = self.button("Draw query region", self.draw_query)
        self.find_button = self.button("Find similar regions", self.find_similar)
        query_row.addWidget(self.draw_button)
        query_row.addWidget(self.find_button)
        self.show_query_button = self.button("Show saved query", self.show_saved_query)
        query_row.addWidget(self.show_query_button)
        layout.addLayout(query_row)
        self.query_label = QLabel("Draw one rectangle on an open image in napari. Select it, then find similar regions.")
        self.query_label.setWordWrap(True)
        layout.addWidget(self.query_label)
        history = QHBoxLayout()
        history.addWidget(QLabel("Saved search"))
        self.sessions = QComboBox()
        self.sessions.currentIndexChanged.connect(self.refresh_results)
        history.addWidget(self.sessions, 1)
        layout.addLayout(history)
        self.results = QListWidget()
        self.results.setViewMode(QListWidget.IconMode)
        self.results.setResizeMode(QListWidget.Adjust)
        self.results.setMovement(QListWidget.Static)
        self.results.setIconSize(QSize(112, 112))
        self.results.setGridSize(QSize(185, 180))
        self.results.setUniformItemSizes(True)
        self.results.setWordWrap(True)
        self.results.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.results.itemClicked.connect(lambda item: self._guard(lambda: self.open_result(item.data(Qt.UserRole))))
        layout.addWidget(self.results, 1)
        review_row = QHBoxLayout()
        self.review_buttons = []
        for text, decision in (("Relevant", "relevant"), ("Not relevant", "irrelevant"), ("Uncertain", "uncertain")):
            button = self.button(text, lambda d=decision: self.review_selected(d))
            review_row.addWidget(button)
            self.review_buttons.append(button)
        self.include_button = self.button("Include in dataset", lambda: self.include_selected(True))
        self.exclude_button = self.button("Remove from dataset", lambda: self.include_selected(False))
        review_row.addWidget(self.include_button)
        review_row.addWidget(self.exclude_button)
        layout.addLayout(review_row)
        self.dataset_toggle = QPushButton("Dataset classes, specimen splits, and export ▸")
        self.dataset_toggle.setCheckable(True)
        self.dataset_panel = QWidget()
        dataset_layout = QVBoxLayout(self.dataset_panel)
        dataset_layout.setContentsMargins(0, 0, 0, 0)
        self.dataset_panel.hide()
        self.dataset_toggle.toggled.connect(self.dataset_panel.setVisible)
        layout.addWidget(self.dataset_toggle)
        layout.addWidget(self.dataset_panel)
        biology = QHBoxLayout()
        self.class_name = QLineEdit()
        self.class_name.setPlaceholderText("Optional biological class, e.g. mitochondrion")
        self.class_status = QComboBox()
        self.class_status.addItem("Proposed", "proposed")
        self.class_status.addItem("Confirmed by reviewer", "confirmed")
        self.class_button = self.button("Assign class", self.annotate_selected)
        biology.addWidget(self.class_name, 1)
        biology.addWidget(self.class_status)
        biology.addWidget(self.class_button)
        dataset_layout.addLayout(biology)

        groups = QHBoxLayout()
        self.images = QComboBox()
        self.images.currentIndexChanged.connect(self._show_group)
        self.group_name = QLineEdit()
        self.group_name.setPlaceholderText("Specimen/group")
        self.group_name.setToolTip("Default group is this source image. Enter the same animal/specimen ID for related images to share one split.")
        self.split = QComboBox()
        self.split.addItems(["unassigned", "train", "validation", "test"])
        self.group_button = self.button("Set specimen split", self.assign_group)
        for widget in (self.images, self.group_name, self.split, self.group_button):
            groups.addWidget(widget)
        dataset_layout.addLayout(groups)
        export_row = QHBoxLayout()
        self.export_task = QComboBox()
        self.export_task.addItem("Self-supervised crops", "self-supervised")
        self.export_task.addItem("Confirmed region classification", "classification")
        self.export_task.addItem("Reviewed similarity pairs", "similarity-pairs")
        self.export_level = QSpinBox()
        self.export_level.setRange(0, 12)
        self.export_level.setToolTip("Native-valued crops at this source pyramid level; saved coordinates remain level 0")
        self.export_button = self.button("Export selected dataset…", self.export_dialog)
        self.selection_label = QLabel("0 regions selected for dataset")
        export_row.addWidget(self.export_task)
        export_row.addWidget(QLabel("Crop level"))
        export_row.addWidget(self.export_level)
        export_row.addWidget(self.export_button)
        dataset_layout.addWidget(self.selection_label)
        dataset_layout.addLayout(export_row)
        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.cancel_button = self.button("Cancel", self.cancel_work)
        self.cancel_button.setEnabled(False)
        progress_row.addWidget(self.progress, 1)
        progress_row.addWidget(self.cancel_button)
        layout.addLayout(progress_row)
        self.status = QLabel("Results are appearance matches, not biological ground truth. Reviews and classes are saved separately.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        for combo in (self.encoder, self.runs, self.sessions, self.images, self.export_task):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
        self._mutable = [self.dataset_toggle, self.new_button, self.open_button, self.save_as_button, self.add_button,
                         self.encoder, self.weights, self.weight_button, self.device, self.tile_size,
                         self.level, self.maximum, self.prepare_button, self.runs, self.draw_button,
                         self.find_button, self.show_query_button, self.sessions, self.include_button, self.exclude_button,
                         self.class_button, self.class_name, self.class_status, self.images,
                         self.group_name, self.split, self.group_button, self.export_task,
                         self.export_button, self.export_level, self.results] + self.review_buttons

    def _remember_checkpoint(self):
        if self._checkpoint_kind and self._checkpoint_kind != "texture":
            self.browser.settings.setValue("discover_weights_" + self._checkpoint_kind,
                                           self.weights.text().strip())

    def _encoder_selected(self, *_):
        kind = self.encoder.currentData()
        learned = kind != "texture"
        self.weights.setEnabled(learned)
        self.weight_button.setEnabled(learned)
        self.device.setEnabled(learned)
        if kind == self._checkpoint_kind:
            return
        self._remember_checkpoint()
        self._checkpoint_kind = kind
        checkpoint = str(self.browser.settings.value("discover_weights_" + kind, "")) if learned else ""
        if not checkpoint and kind == "emdino":
            candidate = self.browser.data_root / "models" / "emdino" / "backbone_emdino_v1.pt"
            if candidate.is_file():
                checkpoint = str(candidate)
        if not checkpoint and kind == "resnet18":
            cache = Path.home() / ".cache/huggingface/hub/models--smp-hub--resnet18.imagenet"
            candidates = sorted(cache.glob("snapshots/*/model.safetensors"))
            if candidates:
                checkpoint = str(candidates[-1])
        self.weights.setText(checkpoint)

    def cancel_work(self, *_):
        self._cancel.set()

    def restore_last_study(self):
        if self.study is not None:
            return
        saved = str(self.browser.settings.value("discover_last_study", ""))
        if saved:
            try:
                self.open_study(saved)
            except Exception as error:
                self.status.setText(f"Last study could not be restored: {error}")

    def _run_selected(self, *_):
        if self.study is not None and self.runs.currentData():
            run = self.study.row("runs", self.runs.currentData())
            self.export_level.setValue(int(run["config"].get("level", 0)))

    def show_saved_query(self):
        self._require_study()
        identifier = self.sessions.currentData()
        if not identifier:
            raise ValueError("Select a saved search first")
        self.open_result(self.study.row("sessions", identifier)["query_region"])

    def _require_study(self):
        if self.study is None:
            raise ValueError("Create or open a study first")
        if self._work is not None:
            raise ValueError("Wait for the current operation or cancel it")

    def new_study(self, path=None):
        if self._work is not None:
            raise ValueError("Cancel the current operation before changing studies")
        if path is None:
            path = self.browser.data_root / "studies" / uid() / "study.browser-analysis.json"
        self.study = Study.create(path, "Study " + now()[:19].replace("T", " "))
        self._refresh_study()
        self.status.setText("Study created. Add Library images; decisions will autosave to the path above.")

    def open_study_dialog(self):
        start = str(self.study.path.parent if self.study is not None else self.browser.data_root / "studies")
        path, _ = QFileDialog.getOpenFileName(self, "Open Discover study", start, "Browser study (*.json)")
        if path:
            self.open_study(path)

    def open_study(self, path):
        if self._work is not None:
            raise ValueError("Cancel the current operation before changing studies")
        self.study = Study.load(path)
        self._query_image = self._query_shapes = None
        self._refresh_study()
        self.status.setText("Study restored, including saved searches, reviews, classes, and selections.")

    def save_as_dialog(self):
        self._require_study()
        path, _ = QFileDialog.getSaveFileName(self, "Save study as a new file", str(self.study.path), "Browser study (*.json)")
        if path:
            self.study = self.study.save_as(path)
            self._refresh_study()

    def choose_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose local encoder checkpoint", self.weights.text(),
                                            "Model weights (*.pt *.pth *.safetensors);;All files (*)")
        if path:
            self.weights.setText(path)
            self._remember_checkpoint()

    def _refresh_study(self):
        if self.study is None:
            return
        doc = self.study.document
        self.study_label.setText(f"{doc['name']} · {len(doc['images'])} images · autosaved to:\n{self.study.path}")
        self.browser.settings.setValue("discover_last_study", str(self.study.path))
        self.runs.blockSignals(True)
        selected_run = self.runs.currentData()
        self.runs.clear()
        for run in doc["runs"]:
            self.runs.addItem(f"{run['encoder']['kind']} · {run['prepared_regions']} regions · {run['id'][:6]}", run["id"])
        self.runs.setCurrentIndex(self.runs.findData(selected_run) if selected_run and self.runs.findData(selected_run) >= 0 else self.runs.count() - 1)
        self.runs.blockSignals(False)
        self._run_selected()
        self.sessions.blockSignals(True)
        selected_session = self.sessions.currentData()
        self.sessions.clear()
        for index, session in enumerate(doc["sessions"], 1):
            self.sessions.addItem(f"Search {index} · {len(session['results'])} matches · {session['created'][:19]}", session["id"])
        self.sessions.setCurrentIndex(self.sessions.findData(selected_session) if selected_session and self.sessions.findData(selected_session) >= 0 else self.sessions.count() - 1)
        self.sessions.blockSignals(False)
        image_id = self.images.currentData()
        self.images.blockSignals(True)
        self.images.clear()
        for image in doc["images"]:
            self.images.addItem(f"{Path(image['path']).name} [{image['image_index']}]", image["id"])
        self.images.setCurrentIndex(max(0, self.images.findData(image_id)))
        self.images.blockSignals(False)
        self._show_group()
        self.refresh_results()
        self.study_changed.emit()

    def _edit(self, function):
        self._require_study()
        edited = self.study.clone()
        function(edited)
        edited.save()
        self.study = edited
        self._refresh_study()

    def _start(self, kind, function):
        self._require_study()
        self._cancel = threading.Event()
        self._job_kind = kind
        self._work = Work(function)
        self._work.signals.done.connect(self._done)
        self._work.signals.progress.connect(self._progress)
        for widget in self._mutable:
            widget.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.status.setText(f"{kind}…")
        self.operation_changed.emit(True)
        self.pool.start(self._work)

    @Slot(int, int, str)
    def _progress(self, value, total, text):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(value)
        self.status.setText(text)
        self._work_progress.emit(value, total, text)

    @Slot(object, object)
    def _done(self, result, error):
        kind = self._job_kind
        self._work = None
        for widget in self._mutable:
            widget.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._encoder_selected()
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        if error:
            self.status.setText(str(error))
            self.operation_changed.emit(False)
            return
        try:
            if kind == "Export":
                edited = self.study.clone()
                edited.document["datasets"].append({"id": uid(), "created": now(),
                                                    "manifest": os.path.relpath(result, edited.path.parent)})
                edited.save()
                self.study = edited
                self._refresh_study()
                self.status.setText(f"Dataset exported: {result}")
            else:
                edited, detail = result
                edited.save()
                self.study = edited
                self._refresh_study()
                if kind == "OCR":
                    self.status.setText(f"OCR: {detail['completed']} extracted, {detail['skipped']} cached; "
                                        + ("cancelled; completed results saved. " if detail["cancelled"] else "")
                                        + "\n".join(detail["errors"]))
                elif kind == "Search":
                    self.sessions.setCurrentIndex(self.sessions.findData(detail["id"]))
                    self.status.setText(f"Saved {len(detail['results'])} appearance matches. Review relevance and biological classes independently.")
                elif kind == "Prepare":
                    self.runs.setCurrentIndex(self.runs.findData(detail["id"]))
                    errors = "; ".join(detail["errors"][:3])
                    self.status.setText(f"Prepared {detail['prepared_regions']} of {detail['total_grid_regions']} possible grid regions. " + errors)
                else:
                    self.status.setText("Images added. " + "; ".join(detail[:3]))
        except Exception as failure:
            self.status.setText(f"Could not autosave study: {failure}")
            if kind == "OCR":
                self._unsaved_ocr_result = result  # Retain completed output if disk conflict prevents saving.
        finally:
            self.operation_changed.emit(False)

    def add_library_images(self):
        self._require_study()
        paths = []
        for row in range(self.browser.proxy.rowCount()):
            path = self.browser.proxy.index(row, 0).data(Qt.UserRole + 1)
            if self.browser._entries[path].role == "image":
                paths.append(path)
        if not paths:
            raise ValueError("The current Library filter contains no image entries")
        snapshot = self.study.clone()
        def work(progress):
            errors = []
            for index, path in enumerate(paths):
                check_cancel(self._cancel)
                try:
                    snapshot.add_image(path)
                except Exception as error:
                    errors.append(f"{Path(path).name}: {error}")
                progress(index + 1, len(paths), f"Adding image metadata {index + 1}/{len(paths)}")
            if not snapshot.document["images"]:
                raise ValueError("No supported image sources were added. " + "; ".join(errors[:3]))
            return snapshot, errors
        self._start("Add images", work)

    def config(self):
        self._remember_checkpoint()
        return {"encoder": self.encoder.currentData(), "weights": self.weights.text().strip() if self.encoder.currentData() != "texture" else "",
                "device": self.device.currentText(), "tile_size": self.tile_size.value(),
                "level": self.level.value(), "max_regions": self.maximum.value()}

    def prepare_study(self):
        self._require_study()
        snapshot, config = self.study.clone(), self.config()
        self._start("Prepare", lambda progress: prepare(snapshot, self.cache_root, config, self._cancel, progress))

    def draw_query(self):
        self._require_study()
        layer = getattr(self.viewer.layers.selection, "active", None)
        if layer is None or getattr(layer, "_type_string", "") != "image":
            raise ValueError("Select an open Image layer in napari first")
        layer_reference(layer)
        shapes = next((item for item in self.viewer.layers
                       if getattr(item, "metadata", {}).get(ROI_ROLE) == "query"), None)
        if shapes is not None:
            self.viewer.layers.remove(shapes)
        self._query_image = layer
        self._query_shapes = self.viewer.add_shapes(
            ndim=layer.ndim, name="Discover query", edge_color="yellow",
            face_color="transparent", metadata={ROI_ROLE: "query"}, **transforms(layer))
        self._query_shapes.mode = "add_rectangle"
        self.viewer.layers.selection.active = self._query_shapes
        self.query_label.setText(f"Query image: {layer.name}. Draw one rectangle, then click Find similar regions.")

    def capture_query(self):
        shapes, image = self._query_shapes, self._query_image
        if shapes is None or image is None or shapes not in self.viewer.layers or image not in self.viewer.layers:
            raise ValueError("Click Draw query region on an Image layer first")
        selected = set(shapes.selected_data)
        if not selected and len(shapes.data) == 1:
            selected = {0}
        if len(selected) != 1:
            raise ValueError("Select exactly one query rectangle")
        index = next(iter(selected))
        if shapes.shape_type[index] != "rectangle":
            raise ValueError("Query must be a rectangle")
        vertices = np.asarray([image.world_to_data(shapes.data_to_world(point)) for point in shapes.data[index]])
        transform = {key: np.asarray(value).tolist() for key, value in transforms(image).items()}
        return (*layer_reference(image), vertices, transform)

    def find_similar(self):
        self._require_study()
        run_id = self.runs.currentData()
        if not run_id:
            raise ValueError("Prepare study embeddings first")
        path, image_index, vertices, transform = self.capture_query()
        snapshot = self.study.clone()
        def work(progress):
            region = region_from_vertices(snapshot, path, image_index, vertices, transform)
            progress(0, 1, "Embedding the selected ROI and searching saved vectors")
            return search(snapshot, self.cache_root, run_id, region["id"], cancel=self._cancel)
        self._start("Search", work)

    def refresh_results(self, *_):
        selected = {item.data(Qt.UserRole) for item in self.results.selectedItems()}
        self.results.clear()
        if self.study is None:
            return
        doc = self.study.document
        self.selection_label.setText(f"{len(doc['selections'])} regions selected for dataset")
        self.dataset_toggle.setText(f"Dataset ({len(doc['selections'])} regions): classes, specimen splits, and export ▸")
        session_id = self.sessions.currentData()
        if not session_id:
            return
        session = self.study.row("sessions", session_id)
        reviews = {(r["session_id"], r["region_id"]): r["decision"] for r in doc["reviews"]}
        annotations = {a["region_id"]: a for a in doc["annotations"]}
        classes = {c["id"]: c["name"] for c in doc["classes"]}
        cache = VectorCache(self.cache_root, doc["id"])
        for result in session["results"]:
            region = self.study.row("regions", result["region_id"])
            image = self.study.row("images", region["image_id"])
            review = reviews.get((session_id, region["id"]), "unreviewed")
            prefix = "[Dataset] " if region["id"] in doc["selections"] else ""
            label = f"{prefix}{Path(image['path']).name}\n{result['score']:.3f} · {review}"
            annotation = annotations.get(region["id"])
            if annotation:
                label += "\n" + classes[annotation["class_id"]] + " (" + annotation["status"] + ")"
            if region["id"] in doc["selections"]:
                label += "\nIncluded in dataset"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, region["id"])
            item.setToolTip(f"{self.study.resolve(image['path'])}\nYX bounds: {region['bounds']}\nPlane: {region['plane']}\nCosine appearance similarity, not biological confidence")
            target = cache.thumbnail(region["id"], region["source_revision"])
            if target.exists():
                item.setIcon(QIcon(str(target)))
            self.results.addItem(item)
            item.setSelected(region["id"] in selected)

    def selected_regions(self):
        identifiers = [item.data(Qt.UserRole) for item in self.results.selectedItems()]
        if not identifiers:
            raise ValueError("Select result thumbnails first")
        return identifiers

    def review_selected(self, decision):
        identifiers, session = self.selected_regions(), self.sessions.currentData()
        self._edit(lambda study: [study.review(session, identifier, decision) for identifier in identifiers])

    def include_selected(self, included):
        identifiers = self.selected_regions()
        self._edit(lambda study: [study.select(identifier, included) for identifier in identifiers])
        self.status.setText(f"Saved dataset selection: {len(self.study.document['selections'])} regions included.")

    def annotate_selected(self):
        identifiers = self.selected_regions()
        name, status = self.class_name.text(), self.class_status.currentData()
        self._edit(lambda study: [study.set_class(identifier, name, status) for identifier in identifiers])

    def _show_group(self, *_):
        if self.study is None or not self.images.currentData():
            return
        image = self.study.row("images", self.images.currentData())
        self.group_name.setText(image["group_id"])
        self.split.setCurrentText(self.study.document["groups"].get(image["group_id"], "unassigned"))

    def assign_group(self):
        image_id, group, split = self.images.currentData(), self.group_name.text(), self.split.currentText()
        self._edit(lambda study: study.set_group(image_id, group, split))

    def export_dialog(self):
        self._require_study()
        destination = QFileDialog.getExistingDirectory(self, "Choose a parent folder for a new dataset version")
        if destination:
            self.start_export(destination)

    def start_export(self, destination):
        self._require_study()
        snapshot, task = self.study.clone(), self.export_task.currentData()
        level = self.export_level.value()
        self._start("Export", lambda progress: export_dataset(snapshot, destination, task, self._cancel, progress, level))

    def open_result(self, region_id):
        self._require_study()
        region = self.study.row("regions", region_id)
        image = self.study.row("images", region["image_id"])
        self.study.assert_current(image, region["source_revision"])
        path = str(self.study.resolve(image["path"]))
        layer = None
        for candidate in self.viewer.layers:
            if getattr(candidate, "_type_string", "") == "image":
                try:
                    if layer_reference(candidate) == (path, image["image_index"]):
                        layer = candidate
                        break
                except ValueError:
                    pass
        if layer is None:
            opened = self.browser._open_paths([path])
            layers = (opened or {}).get((path, "image"), [])
            if image["image_index"] >= len(layers):
                raise ValueError("Source image could not be opened; see Library status")
            layer = layers[image["image_index"]]
            for key, value in image.get("viewer_transform", {}).items():
                setattr(layer, key, value)
        ndim = layer.ndim
        axes = [axis for axis in image["axes"] if not (axis == "S" and image["rgb"])]
        point = [0 if value is None else value for axis, value in zip(image["axes"], region["plane"])
                 if not (axis == "S" and image["rgb"])]
        y, x = axes.index("Y"), axes.index("X")
        (y0, y1), (x0, x1) = region["bounds"]
        corners = []
        for row, column in ((y0, x0), (y0, x1), (y1, x1), (y1, x0)):
            vertex = list(point)
            vertex[y], vertex[x] = row, column
            corners.append(vertex)
        outline = next((item for item in self.viewer.layers
                        if getattr(item, "metadata", {}).get(ROI_ROLE) == "result"), None)
        if outline is not None:
            self.viewer.layers.remove(outline)
        self.viewer.add_shapes([corners], ndim=ndim, shape_type="rectangle",
                               name="Discover result", edge_color="cyan", face_color="transparent",
                               metadata={ROI_ROLE: "result", "region_id": region_id}, **transforms(layer))
        center = np.asarray(corners).mean(axis=0)
        world = tuple(layer.data_to_world(center))
        self.viewer.dims.set_point(tuple(range(self.viewer.dims.ndim - ndim, self.viewer.dims.ndim)), world)
        self.viewer.camera.center = (0,) * max(0, 3 - ndim) + world[-3:]
        world_corners = np.asarray([layer.data_to_world(vertex) for vertex in corners])
        span = float(np.ptp(world_corners, axis=0).max())
        if span > 0:
            self.viewer.camera.zoom = 400 / span
        self.viewer.layers.selection.active = layer
        self.status.setText(f"Showing saved region {region_id[:8]} in {Path(path).name}")
