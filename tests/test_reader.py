import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from napari_browser_assistant._library import scan_images, build_preview
from napari_browser_assistant._reader import (
    napari_get_reader, read_dataset, workspace_manifest, zarr_arrays,
)
from test_browser import app, make_widget, settle


def make_store(path, *, version=3, label=False):
    zarr = pytest.importorskip("zarr")
    group = zarr.open_group(str(path), mode="w", zarr_format=version)
    shapes = [(32, 48)] if label else [(32, 48), (8, 12)]
    for i, shape in enumerate(shapes):
        array = group.create_array(f"s{i}", shape=shape, chunks=(8, 12), dtype="uint16" if label else "uint8")
        array[:] = 7 if label else np.arange(np.prod(shape), dtype="uint8").reshape(shape)
    metadata = {"axes": [{"name": "y"}, {"name": "x"}], "datasets": [
        {"path": f"s{i}", "coordinateTransformations": [
            {"type": "scale", "scale": [2 * 4**i, 3 * 4**i]},
            {"type": "translation", "translation": [10, 20]}]}
        for i in range(len(shapes))]}
    if version == 3:
        group.attrs["ome"] = {"version": "0.5", "multiscales": [metadata]}
    else:
        group.attrs["multiscales"] = [metadata]
    return group


def make_workspace(root):
    manifest = root / "sample.label-assistant.json"
    companion = root / "sample_label_assistant_data"
    companion.mkdir()
    make_store(companion / "image.ome.zarr")
    make_store(companion / "labels.ome.zarr", label=True)
    document = {"format": "napari-label-assistant-workspace", "version": 1, "layers": [
        {"layer_type": "Image", "name": "Source EM", "scale": [1, 1], "translate": [4, 5],
         "rgb": False, "contrast_limits": [0, 255],
         "storage": {"kind": "zarr", "path": "sample_label_assistant_data/image.ome.zarr",
                     "array_path": "", "multiscale": True, "datasets": ["s0", "s1"], "mode": "r"}},
        {"layer_type": "Labels", "name": "Curated", "visible": False,
         "storage": {"kind": "zarr", "path": "sample_label_assistant_data/labels.ome.zarr",
                     "array_path": "s0", "mode": "r+"}},
        {"layer_type": "Shapes", "name": "ROI", "storage": {"kind": "inline"},
         "data": [[[1, 1], [1, 8], [8, 8], [8, 1]]], "shape_type": "rectangle"}]}
    manifest.write_text(json.dumps(document))
    return manifest, companion


@pytest.mark.parametrize("version", [2, 3])
def test_ngff_reader_and_preview(version, tmp_path):
    path = tmp_path / "image.ome.zarr"
    make_store(path, version=version)
    reader = napari_get_reader(str(path))
    assert reader is not None
    layers = reader(str(path))
    arrays, meta, kind = layers[0]
    assert kind == "image" and meta["multiscale"]
    assert [a.shape for a in arrays] == [(32, 48), (8, 12)]
    assert all(a.read_only for a in arrays)
    assert meta["scale"] == [2, 3]
    assert meta["translate"] == [10, 20]
    entry = scan_images(tmp_path, True)[0][0]
    target, updated = build_preview(entry, tmp_path / "cache")
    assert updated.shape == "32 x 48"
    with Image.open(target) as image:
        assert image.size == (12, 8)


def test_workspace_discovery_groups_companion_and_ignores_other_json(tmp_path):
    manifest, companion = make_workspace(tmp_path)
    (companion / "chunk.tif").touch()
    (tmp_path / "settings.json").write_text('{"layers": []}')
    (tmp_path / "invalid.json").write_text("{")
    entries, errors = scan_images(tmp_path, True)
    assert not errors
    assert [entry.path for entry in entries] == [manifest]
    assert scan_images(companion, True)[0][0].path == manifest
    assert scan_images(manifest, True)[0][0].path == manifest
    assert workspace_manifest(companion) == manifest
    assert napari_get_reader(str(tmp_path / "settings.json")) is None


def test_workspace_image_preview_and_optional_typed_annotations(tmp_path):
    manifest, companion = make_workspace(tmp_path)
    before = manifest.read_bytes()
    default = read_dataset(manifest)
    assert [kind for _, _, kind in default] == ["image"]
    assert default[0][1]["name"] == "Source EM"
    assert default[0][1]["scale"] == [1, 1]
    assert default[0][1]["translate"] == [4, 5]
    full = read_dataset(companion, include_annotations=True)
    assert [kind for _, _, kind in full] == ["image", "labels", "shapes"]
    assert full[1][0].read_only
    assert not full[1][1]["visible"]
    entry = scan_images(tmp_path, True)[0][0]
    target, updated = build_preview(entry, tmp_path / "cache")
    assert updated.shape == "32 x 48"
    with Image.open(target) as preview:
        assert preview.mode == "RGB"
    assert manifest.read_bytes() == before


def test_missing_annotation_does_not_prevent_opening_image(tmp_path):
    manifest, _ = make_workspace(tmp_path)
    document = json.loads(manifest.read_text())
    document["layers"][1]["storage"]["path"] = "missing.ome.zarr"
    manifest.write_text(json.dumps(document))
    assert read_dataset(manifest)[0][2] == "image"
    with pytest.raises(Exception):
        read_dataset(manifest, include_annotations=True)
    document["layers"][0]["storage"]["path"] = "missing.ome.zarr"
    manifest.write_text(json.dumps(document))
    assert scan_images(tmp_path, True)[0][0].path == manifest
    with pytest.raises(Exception):
        read_dataset(manifest)


def test_reader_rejects_unsafe_array_keys_and_future_workspace(tmp_path):
    manifest, companion = make_workspace(tmp_path)
    document = json.loads(manifest.read_text())
    document["layers"][0]["storage"]["datasets"] = ["../../escape"]
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Invalid OME-Zarr"):
        read_dataset(manifest)
    document["version"] = 999
    manifest.write_text(json.dumps(document))
    assert napari_get_reader(str(manifest)) is None
    assert workspace_manifest(companion) is None


def test_reader_keeps_pixels_lazy(tmp_path, monkeypatch):
    path = tmp_path / "image.ome.zarr"
    make_store(path)
    arrays, _, _ = zarr_arrays(path)
    def no_pixels(*args, **kwargs):
        raise AssertionError("Opening the reader eagerly decoded pixels")
    monkeypatch.setattr(type(arrays[0]), "__getitem__", no_pixels)
    assert read_dataset(path)[0][0][0].shape == (32, 48)


def test_workspace_browser_and_registered_reader(app, tmp_path, monkeypatch):
    from napari.components import ViewerModel
    from npe2 import PluginManager, PluginManifest
    manager = PluginManager()
    monkeypatch.setattr(PluginManager, "instance", classmethod(lambda cls: manager))
    manager.register(PluginManifest.from_file(
        Path(__file__).parents[1] / "src/napari_browser_assistant/napari.yaml"))
    manifest, companion = make_workspace(tmp_path)
    widget, _, _ = make_widget(app, tmp_path / "ui", tmp_path)
    widget.asset_view.setCurrentIndex(1)
    widget.kind.setCurrentText("Datasets / workspaces")
    settle(app, widget)
    assert widget.proxy.rowCount() == 1
    widget.viewer = ViewerModel()
    widget._open_paths([str(manifest)])
    assert len(widget.viewer.layers) == 1, widget.status_label.text()
    assert widget.viewer.layers[0]._type_string == "image"
    assert widget.viewer.layers[0].multiscale
    widget.viewer.layers.clear()
    widget._open_workspace_layers(manifest)
    assert [layer._type_string for layer in widget.viewer.layers] == ["image", "labels", "shapes"]
    assert not widget.viewer.layers[1].editable
    widget.viewer.layers.clear()
    widget.path_edit.setText(str(companion))
    widget._change_root()
    settle(app, widget)
    assert widget.root == tmp_path
    assert any(entry.path == manifest for entry in widget._entries.values())
    widget.close()
