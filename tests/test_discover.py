import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from napari_browser_assistant._discovery import (
    Cancelled, VectorCache, export_dataset, prepare, search,
)
from napari_browser_assistant._discovery_io import RegionSource
from napari_browser_assistant._encoders import Encoder
from napari_browser_assistant._study import Study
from napari_browser_assistant._discover_widget import region_from_vertices
from test_browser import app, make_widget, settle
from test_reader import make_store, make_workspace


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    yy, xx = np.indices((128, 128))
    pattern = (((xx % 32 - 16) ** 2 + (yy % 32 - 16) ** 2) < 60).astype("uint16") * 4097
    for name, array in (("reference.tif", pattern), ("matching.tif", pattern),
                        ("different.tif", np.random.default_rng(17).integers(0, 65535, pattern.shape, dtype="uint16"))):
        tifffile.imwrite(root / name, array, photometric="minisblack")
    study = Study.create(tmp_path / "study.browser-analysis.json", "EM study")
    for path in sorted(root.glob("*.tif")):
        study.add_image(path)
    study.save()
    return study, tmp_path / "cache", root, pattern


def config(**kwargs):
    return dict(encoder="texture", weights="", device="cpu", tile_size=32,
                level=0, max_regions=48, **kwargs)


def complete_search(study, cache, root):
    study, run = prepare(study, cache, config())
    image = next(image for image in study.document["images"] if study.resolve(image["path"]).name == "reference.tif")
    query = study.add_region(image, [[0, 32], [0, 32]])
    study, session = search(study, cache, run["id"], query["id"])
    return study, run, session


def test_region_source_tiff_plane_and_native_values(tmp_path):
    data = np.arange(3 * 32 * 48, dtype="uint16").reshape(3, 32, 48)
    path = tmp_path / "volume.tif"
    tifffile.imwrite(path, data, photometric="minisblack", metadata={"axes": "ZYX"})
    before = path.read_bytes()
    with RegionSource(path) as source:
        description = source.describe()
        assert description["axes"] == ["Z", "Y", "X"]
        crop = source.read([[4, 12], [7, 19]], plane=[2, None, None])
        assert np.array_equal(crop, data[2, 4:12, 7:19])
    assert path.read_bytes() == before


@pytest.mark.parametrize("version", [2, 3])
def test_pyramid_coordinates_and_workspace_sources(tmp_path, version):
    path = tmp_path / "image.ome.zarr"
    group = make_store(path, version=version)
    with RegionSource(path) as source:
        assert source.describe()["shape"] == [32, 48]
        assert source.describe()["pixel_scale"] == [2.0, 3.0]
        crop = source.read([[8, 24], [12, 36]], level=1)
        assert np.array_equal(crop, group["s1"][2:6, 3:9])
    manifest, _ = make_workspace(tmp_path)
    with RegionSource(manifest) as source:
        assert source.describe()["shape"] == [32, 48]
        assert source.describe()["pixel_scale"] == [1, 1]


def test_huge_zarr_chunk_is_rejected_before_pixel_access(tmp_path, monkeypatch):
    import zarr
    path = tmp_path / "huge.ome.zarr"
    group = zarr.open_group(str(path), mode="w")
    array = group.create_array("0", shape=(100000, 100000), chunks=(16384, 16384), dtype="uint8")
    group.attrs["multiscales"] = [{"axes": ["y", "x"], "datasets": [{"path": "0"}]}]
    with RegionSource(path) as source:
        def forbidden(*args, **kwargs):
            raise AssertionError("Oversized chunk decoded")
        monkeypatch.setattr(type(array), "__getitem__", forbidden)
        with pytest.raises(ValueError, match="decode budget"):
            source.read([[0, 32], [0, 32]])


def test_preparation_retrieval_provenance_and_self_match_exclusion(experiment):
    study, cache, root, _ = experiment
    prepared, run, session = complete_search(study, cache, root)
    assert study.document["runs"] == []  # Worker operates on a snapshot.
    assert run["dimension"] > 1 and run["encoder"]["learned"] is False
    assert run["prepared_regions"] == 48
    assert session["results"]
    assert session["results"][0]["score"] == pytest.approx(1, abs=1e-5)
    for result in session["results"]:
        region = prepared.row("regions", result["region_id"])
        assert region["coordinate_space"] == "level-0-pixels"
        assert region["id"] != session["query_region"]
        assert -1 <= result["score"] <= 1
    prepared.save()
    restored = Study.load(prepared.path)
    assert restored.document["sessions"] == prepared.document["sessions"]


def test_query_search_reuses_cached_vectors(experiment, monkeypatch):
    study, cache, root, _ = experiment
    study, run, session = complete_search(study, cache, root)
    calls = []
    original = Encoder.encode
    def count(self, data):
        calls.append(1)
        return original(self, data)
    monkeypatch.setattr(Encoder, "encode", count)
    _, new_session = search(study, cache, run["id"], session["query_region"])
    assert len(calls) == 1  # Encode the query, never rescan or re-encode the indexed tiles.
    assert new_session["results"] == session["results"]


def test_reviews_classes_selections_and_save_as_survive_cache_loss(experiment):
    study, cache, root, _ = experiment
    study, run, session = complete_search(study, cache, root)
    identifier = session["results"][0]["region_id"]
    study.review(session["id"], identifier, "relevant")
    study.set_class(identifier, "mitochondrion", "proposed")
    study.select(identifier)
    study.save()
    copy = study.save_as(study.path.parent / "other" / "copy.browser-analysis.json")
    assert copy.resolve(copy.document["images"][0]["path"]).exists()
    cache_file = VectorCache(cache, study.document["id"]).path
    cache_file.unlink()
    restored = Study.load(copy.path)
    assert restored.document["reviews"][0]["decision"] == "relevant"
    assert restored.document["annotations"][0]["status"] == "proposed"
    assert identifier in restored.document["selections"]
    with pytest.raises(ValueError, match="cache is missing"):
        search(restored, cache, run["id"], session["query_region"])


def test_changed_source_preserves_reviews_and_blocks_training_export(experiment):
    study, cache, root, _ = experiment
    study, _, session = complete_search(study, cache, root)
    identifier = session["results"][0]["region_id"]
    region = study.row("regions", identifier)
    image = study.row("images", region["image_id"])
    study.review(session["id"], identifier, "relevant")
    study.select(identifier)
    import os
    path = study.resolve(image["path"])
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
    with pytest.raises(ValueError, match="Source metadata changed"):
        export_dataset(study, root / "exports")
    assert study.document["reviews"][0]["decision"] == "relevant"


def test_export_native_crops_confirmed_classes_and_group_splits(experiment):
    study, cache, root, pattern = experiment
    study, _, session = complete_search(study, cache, root)
    identifier = session["results"][0]["region_id"]
    region = study.row("regions", identifier)
    image = study.row("images", region["image_id"])
    study.select(identifier)
    study.set_class(identifier, "mitochondrion", "proposed")
    with pytest.raises(ValueError, match="confirmed"):
        export_dataset(study, root / "exports", "classification")
    study.set_class(identifier, "mitochondrion", "confirmed")
    study.set_group(image["id"], "animal-A", "train")
    manifest_path = export_dataset(study, root / "exports", "classification")
    manifest = json.loads(manifest_path.read_text())
    sample = manifest["samples"][0]
    crop = tifffile.imread(manifest_path.parent / sample["crop"])
    (y0, y1), (x0, x1) = region["bounds"]
    original = tifffile.imread(study.resolve(image["path"]))
    assert np.array_equal(crop, original[y0:y1, x0:x1])
    assert crop.dtype == np.uint16
    assert sample["group_id"] == "animal-A" and sample["split"] == "train"
    assert manifest["classes"][str(sample["class_index"])] == "mitochondrion"
    second = export_dataset(study, root / "exports", "classification")
    assert second != manifest_path and manifest_path.exists()


def test_pair_export_and_split_leakage_check(experiment):
    study, cache, root, _ = experiment
    study, _, session = complete_search(study, cache, root)
    query = study.row("regions", session["query_region"])
    result = next(result for result in session["results"]
                  if study.row("regions", result["region_id"])["image_id"] != query["image_id"])
    identifier = result["region_id"]
    study.select(identifier)
    study.review(session["id"], identifier, "relevant")
    candidate = study.row("regions", identifier)
    study.set_group(query["image_id"], "animal-A", "train")
    study.set_group(candidate["image_id"], "animal-B", "validation")
    destination = root / "exports"
    with pytest.raises(ValueError, match="different dataset splits"):
        export_dataset(study, destination, "similarity-pairs")
    assert not list(destination.iterdir())
    study.set_group(candidate["image_id"], "animal-B", "train")
    manifest_path = export_dataset(study, destination, "similarity-pairs")
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["samples"]) == 2
    assert manifest["pairs"][0]["relevant"] is True
    assert manifest["pairs"][0]["split"] == "train"


def test_cancelled_preparation_and_export_are_transactional(experiment):
    study, cache, root, _ = experiment
    cancel = threading.Event()
    def progress(*args):
        cancel.set()
    before = study.path.read_bytes()
    with pytest.raises(Cancelled):
        prepare(study, cache, config(), cancel, progress)
    assert study.path.read_bytes() == before
    cancel.clear()
    study, _, session = complete_search(study, cache, root)
    study.select(session["results"][0]["region_id"])
    destination = root / "cancelled-export"
    with pytest.raises(Cancelled):
        export_dataset(study, destination, cancel=cancel, progress=progress)
    assert not list(destination.iterdir())


def test_external_study_edit_conflict_and_invalid_reference(experiment):
    study, _, _, _ = experiment
    other = Study.load(study.path)
    study.document["name"] = "First writer"
    study.save()
    other.document["name"] = "Second writer"
    with pytest.raises(RuntimeError, match="changed on disk"):
        other.save()
    invalid = json.loads(study.path.read_text())
    invalid["selections"] = ["missing-region"]
    with pytest.raises(ValueError, match="unknown region"):
        Study(study.path, invalid)


def test_invalid_roi_and_encoder_outputs(experiment):
    study, _, root, _ = experiment
    path = root / "reference.tif"
    with pytest.raises(ValueError, match="axis-aligned"):
        region_from_vertices(study, path, 0, [[0, 10], [10, 20], [20, 10], [10, 0]])
    region = region_from_vertices(study, path, 0, [[3, 5], [3, 20], [19, 20], [19, 5]])
    assert region["bounds"] == [[3, 19], [5, 20]]
    with pytest.raises(ValueError, match="finite"):
        Encoder().encode(np.full((32, 32), np.nan))


def wait_discover(app, discover):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        app.processEvents()
        if discover._work is None:
            return
        time.sleep(.01)
    raise AssertionError(discover.status.text())


def test_discover_gui_complete_workflow_with_real_napari(app, tmp_path, monkeypatch):
    from napari.components import ViewerModel
    from npe2 import PluginManager
    from qtpy.QtCore import Qt
    manager = PluginManager()
    monkeypatch.setattr(PluginManager, "instance", classmethod(lambda cls: manager))
    manager.register("napari")
    root = tmp_path / "images"
    root.mkdir()
    data = np.random.default_rng(3).integers(0, 65535, (96, 96), dtype="uint16")
    tifffile.imwrite(root / "source.tif", data, photometric="minisblack")
    widget, _, _ = make_widget(app, tmp_path / "ui", root)
    widget.viewer = ViewerModel()
    layer = widget.viewer.add_image(data, name="Source", scale=(2, 3), translate=(10, 20),
                                    metadata={"browser_source": {"path": str(root / "source.tif"), "image_index": 0}})
    discover = widget.discover
    widget.main_tabs.setCurrentWidget(discover)
    discover.new_study(tmp_path / "study.browser-analysis.json")
    discover.add_library_images()
    wait_discover(app, discover)
    assert len(discover.study.document["images"]) == 1, discover.status.text()
    discover.tile_size.setValue(32)
    discover.maximum.setValue(9)
    main_thread = threading.get_ident()
    threads = []
    original = RegionSource.read
    def read(*args, **kwargs):
        threads.append(threading.get_ident())
        return original(*args, **kwargs)
    monkeypatch.setattr(RegionSource, "read", read)
    discover.prepare_study()
    assert not discover.new_button.isEnabled()
    wait_discover(app, discover)
    assert len(discover.study.document["runs"]) == 1, discover.status.text()
    widget.viewer.layers.selection.active = layer
    discover.draw_query()
    discover._query_shapes.add([[0, 0], [0, 32], [32, 32], [32, 0]], shape_type="rectangle")
    discover.find_similar()
    wait_discover(app, discover)
    assert discover.results.count() == 8, discover.status.text()
    assert threads and all(thread != main_thread for thread in threads)
    item = discover.results.item(0)
    item.setSelected(True)
    identifier = item.data(Qt.UserRole)
    discover.review_selected("relevant")
    discover.class_name.setText("example structure")
    discover.class_status.setCurrentIndex(1)
    discover.annotate_selected()
    discover.include_selected(True)
    restored = Study.load(discover.study.path)
    assert restored.document["reviews"][0]["decision"] == "relevant"
    assert restored.document["annotations"][0]["status"] == "confirmed"
    assert identifier in restored.document["selections"]
    discover.open_result(identifier)
    outline = next(item for item in widget.viewer.layers
                   if item.metadata.get("browser_discover_role") == "result")
    assert outline.metadata["region_id"] == identifier
    assert tuple(outline.scale) == (2, 3)
    discover.export_task.setCurrentIndex(1)
    discover.start_export(tmp_path / "exports")
    wait_discover(app, discover)
    assert discover.study.document["datasets"], discover.status.text()
    assert discover.study.resolve(discover.study.document["datasets"][0]["manifest"]).is_file()
    discover.open_study(discover.study.path)
    assert discover.results.count() == 8
    assert discover.results.item(0).icon().isNull() is False
    widget.viewer.layers.clear()
    discover.show_saved_query()
    reopened = next(layer for layer in widget.viewer.layers if layer._type_string == "image")
    assert tuple(reopened.scale) == (2, 3)
    assert tuple(reopened.translate) == (10, 20)
    widget.viewer.layers.clear()
    widget.close()


def test_pyramid_export_uses_source_values_and_records_scale(tmp_path):
    path = tmp_path / "image.ome.zarr"
    group = make_store(path)
    study = Study.create(tmp_path / "study.browser-analysis.json")
    image = study.add_image(path)
    region = study.add_region(image, [[8, 24], [12, 36]])
    study.select(region["id"])
    destination = export_dataset(study, tmp_path / "exports", level=1)
    document = json.loads(destination.read_text())
    sample = document["samples"][0]
    assert sample["crop_level"] == 1
    assert sample["crop_pixel_scale_yx"] == [8, 12]
    assert np.array_equal(tifffile.imread(destination.parent / sample["crop"]), group["s1"][2:6, 3:9])


def test_em_dino_adapter_contract_without_unavailable_checkpoint(tmp_path, monkeypatch):
    import sys
    import importlib.metadata
    torch = pytest.importorskip("torch")
    calls = {}
    class FakeEncoder:
        @classmethod
        def load(cls, arch, path):
            calls["arch"], calls["path"] = arch, path
            return cls()
        def eval(self):
            return self
        def to(self, device):
            return self
        def run(self, pixels, **kwargs):
            calls["shape"], calls["kwargs"] = tuple(pixels.shape), kwargs
            return {"cls": torch.ones(1, 1, 8)}
    checkpoint = tmp_path / "contract-only.pt"
    checkpoint.write_bytes(b"not-model-weights")
    monkeypatch.setitem(sys.modules, "omniem", SimpleNamespace(EMEncoder=FakeEncoder))
    original = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "contract-test" if name == "omniem" else original(name))
    encoder = Encoder("emdino", checkpoint, "cpu")
    result = encoder.encode(np.arange(32*32).reshape(32, 32))
    assert calls["arch"] == "emdinov1"
    assert calls["shape"] == (224, 224)
    assert calls["kwargs"] == {"axes": "yx", "return_cls": True, "return_patch": False}
    assert result.shape == (8,) and np.linalg.norm(result) == pytest.approx(1)


def test_checkpoint_symlink_retains_format_across_prepare_search_and_save_as(experiment, tmp_path, monkeypatch):
    import napari_browser_assistant._discovery as module
    study, cache, root, _ = experiment
    blob = tmp_path / "checkpoint-blob-without-extension"
    blob.write_bytes(b"fixture weights")
    checkpoint = tmp_path / "encoder.safetensors"
    checkpoint.symlink_to(blob)
    baseline = Encoder()
    seen = []
    class LocalEncoder:
        def __init__(self, kind, weights, device):
            seen.append(str(weights))
            assert str(weights).endswith(".safetensors")
            self.key = "fixture-checkpoint-key"
            self.metadata = {"kind": "fixture-resnet18", "learned": True}
        def encode(self, data):
            return baseline.encode(data)
    monkeypatch.setattr(module, "Encoder", LocalEncoder)
    options = config()
    options.update(encoder="resnet18", weights=str(checkpoint))
    study, run = prepare(study, cache, options)
    image = next(i for i in study.document["images"] if study.resolve(i["path"]).name == "reference.tif")
    query = study.add_region(image, [[0, 32], [0, 32]])
    study, _ = search(study, cache, run["id"], query["id"])
    study.save()
    copy = study.save_as(tmp_path / "moved" / "study.browser-analysis.json")
    assert copy.resolve(copy.document["runs"][0]["config"]["weights"]).suffix == ".safetensors"
    assert len(seen) == 2


def test_last_study_restores_automatically_in_new_browser(app, tmp_path):
    from napari_browser_assistant._widget import BrowserAssistantWidget
    from qtpy.QtCore import QSettings
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    viewer = SimpleNamespace(layers=SimpleNamespace(selection=SimpleNamespace(active=None)))
    first = BrowserAssistantWidget(viewer, cache_root=tmp_path / "cache", settings=settings)
    first.discover.new_study(tmp_path / "saved.browser-analysis.json")
    identifier = first.discover.study.document["id"]
    app.processEvents()
    first.close()
    second = BrowserAssistantWidget(viewer, cache_root=tmp_path / "cache", settings=settings)
    app.processEvents()
    assert second.discover.study.document["id"] == identifier
    second.close()


def test_gui_cancel_signals_the_current_worker(app, tmp_path, monkeypatch):
    import napari_browser_assistant._discover_widget as module
    from napari_browser_assistant._widget import BrowserAssistantWidget
    from qtpy.QtCore import QSettings
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    viewer = SimpleNamespace(layers=SimpleNamespace(selection=SimpleNamespace(active=None)))
    widget = BrowserAssistantWidget(viewer, cache_root=tmp_path / "cache", settings=settings)
    discover = widget.discover
    discover.new_study(tmp_path / "study.browser-analysis.json")
    before = discover.study.path.read_bytes()
    started = threading.Event()
    def controlled(study, cache, config, cancel, progress):
        started.set()
        if not cancel.wait(3):
            raise AssertionError("GUI Cancel did not reach the active worker")
        raise Cancelled("Operation cancelled")
    monkeypatch.setattr(module, "prepare", controlled)
    discover.prepare_study()
    assert started.wait(3)
    discover.cancel_button.click()
    wait_discover(app, discover)
    assert discover._cancel.is_set()
    assert "Operation cancelled" in discover.status.text()
    assert discover.study.path.read_bytes() == before
    widget.close()


def test_loaded_region_ids_cannot_escape_export_directory(experiment):
    study, _, _, _ = experiment
    image = study.document["images"][0]
    document = json.loads(study.path.read_text())
    document["regions"] = [{"id": "../outside", "image_id": image["id"],
                            "bounds": [[0, 16], [0, 16]], "plane": [None, None],
                            "source_revision": image["source_revision"]}]
    document["selections"] = ["../outside"]
    with pytest.raises(ValueError, match="Invalid or duplicate IDs"):
        Study(study.path, document)


def test_encoder_switch_uses_separate_checkpoints(app, tmp_path):
    root = tmp_path / 'images'
    root.mkdir()
    widget, _, settings = make_widget(app, tmp_path / 'ui', root)
    discover = widget.discover
    backbone = widget.data_root / 'models' / 'emdino' / 'backbone_emdino_v1.pt'
    backbone.parent.mkdir(parents=True)
    backbone.touch()
    discover.encoder.setCurrentIndex(discover.encoder.findData('resnet18'))
    discover.weights.setText('/local/resnet18.safetensors')
    discover.encoder.setCurrentIndex(discover.encoder.findData('emdino'))
    assert discover.weights.text() == str(backbone)
    discover.weights.setText('/custom/emdino.pt')
    discover._encoder_selected()  # Operation completion must preserve current input.
    assert discover.weights.text() == '/custom/emdino.pt'
    discover.encoder.setCurrentIndex(discover.encoder.findData('resnet18'))
    assert discover.weights.text() == '/local/resnet18.safetensors'
    discover.encoder.setCurrentIndex(discover.encoder.findData('texture'))
    assert discover.weights.text() == ''
    discover.encoder.setCurrentIndex(discover.encoder.findData('emdino'))
    assert discover.weights.text() == '/custom/emdino.pt'
    assert settings.value('discover_weights_resnet18') == '/local/resnet18.safetensors'
    discover.config()
    assert settings.value('discover_weights_emdino') == '/custom/emdino.pt'
    widget.close()


def test_large_single_strip_tiff_uses_read_only_memmap(tmp_path, monkeypatch):
    import napari_browser_assistant._discovery_io as image_io
    path = tmp_path / 'single-strip.tif'
    data = np.arange(512 * 512 * 3, dtype='uint32').reshape(512, 512, 3).astype('uint8')
    tifffile.imwrite(path, data, photometric='rgb', rowsperstrip=512)
    before = path.read_bytes()
    monkeypatch.setattr(image_io, 'MAX_DECODE_BYTES', 128 * 1024)
    def forbidden(*args, **kwargs):
        raise AssertionError('Memory-mappable TIFF should not enter the chunk decoder')
    monkeypatch.setattr(tifffile.TiffPageSeries, 'aszarr', forbidden)
    with RegionSource(path) as source:
        result = source.read([[17, 49], [23, 55]])
        mapped = source._loaded[0]
        assert isinstance(mapped, np.memmap) and mapped.mode == 'r'
        assert np.array_equal(result, data[17:49, 23:55])
        assert not np.shares_memory(result, mapped)
        with pytest.raises(ValueError):
            mapped[0, 0, 0] = 0
    assert mapped._mmap.closed
    assert path.read_bytes() == before


def test_compressed_single_strip_tiff_keeps_decode_limit(tmp_path, monkeypatch):
    import napari_browser_assistant._discovery_io as image_io
    path = tmp_path / 'compressed.tif'
    tifffile.imwrite(path, np.zeros((512, 512, 3), dtype='uint8'),
                     photometric='rgb', rowsperstrip=512, compression='deflate')
    monkeypatch.setattr(image_io, 'MAX_DECODE_BYTES', 128 * 1024)
    with RegionSource(path) as source:
        with pytest.raises(ValueError, match='decode budget'):
            source.read([[0, 32], [0, 32]])
