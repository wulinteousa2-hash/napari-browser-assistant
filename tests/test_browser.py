import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from dataclasses import replace
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import tifffile
from qtpy.QtCore import QItemSelectionModel, QSettings, Qt
from qtpy.QtWidgets import QApplication, QAbstractItemView

from napari_browser_assistant._library import FolderIndex, scan_images, build_preview
from napari_browser_assistant._widget import BrowserAssistantWidget, ROLE_PATH


@pytest.fixture(scope='session')
def app():
    return QApplication.instance() or QApplication([])


def settle(app, widget, predicate=lambda: True):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        app.processEvents()
        if not widget._tasks and not widget._visible_timer.isActive() and predicate():
            return
        time.sleep(.01)
    raise AssertionError(f'Work did not settle: {len(widget._tasks)} tasks; {widget.status_label.text()}')


def make_widget(app, tmp_path, root):
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.IniFormat)
    settings.setValue('last_folder', str(root))
    settings.setValue('library_root', str(root))
    viewer = SimpleNamespace(opened=[], layers=SimpleNamespace(selection=SimpleNamespace(active=None)))
    viewer.open = lambda path, **kwargs: viewer.opened.append(path)
    widget = BrowserAssistantWidget(viewer, cache_root=tmp_path / 'cache', settings=settings)
    widget.resize(1100, 750)
    widget.show()
    settle(app, widget, lambda: not widget._scanning)
    return widget, viewer, settings


def test_scan_cache_formats_and_source_preservation(tmp_path):
    root = tmp_path / 'images'
    root.mkdir()
    sub = root / 'nested'
    sub.mkdir()
    for suffix in ('jpg', 'png', 'bmp', 'gif', 'webp'):
        Image.new('RGB', (400, 300), 'red').save(root / f'image.{suffix}')
    for suffix in ('tif', 'tiff', 'btf', 'tf8'):
        tifffile.imwrite(sub / f'stack.{suffix}', np.arange(8192,dtype='uint16').reshape(2,64,64), photometric='minisblack')
    (root / 'ignore.txt').write_text('ignore')
    assert len(scan_images(root, False)[0]) == 5
    entries, errors = scan_images(root, True)
    assert len(entries) == 9 and errors == []
    for entry in entries:
        before = entry.path.read_bytes()
        preview, updated = build_preview(entry, tmp_path / 'previews')
        assert updated.shape and updated.dtype
        with Image.open(preview) as img:
            assert max(img.size) <= 256
        assert build_preview(entry, tmp_path / 'previews') == (preview, updated)
        assert entry.path.read_bytes() == before
    entry = entries[0]
    old, _ = build_preview(entry, tmp_path / 'previews')
    Image.new('RGB',(30,30),'blue').save(entry.path)
    new_entry = next(e for e in scan_images(root, True)[0] if e.path == entry.path)
    assert build_preview(new_entry, tmp_path / 'previews')[0] != old


def test_legacy_index_migration(tmp_path):
    db = tmp_path / 'index.sqlite'
    root = tmp_path
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE indexed_files (folder TEXT, recursive INTEGER, path TEXT, size_bytes INTEGER, modified_sort REAL, modified_text TEXT, indexed_at REAL, PRIMARY KEY(folder,recursive,path))')
        conn.execute('INSERT INTO indexed_files VALUES (?,?,?,?,?,?,?)', (str(root), 1, str(root/'old.tif'), 1, 1., '', 1.))
    index = FolderIndex(db)
    assert index.load(root, True)[0].path.name == 'old.tif'
    Image.new('L',(10,10)).save(root / 'new.png')
    entries, _ = scan_images(root, True)
    index.save(root, True, entries)
    updated = replace(entries[0], shape='10 x 10', dtype='uint8')
    index.update_metadata(updated)
    assert index.load(root, True) == [updated]
    index.update_metadata(replace(updated, modified_ns=updated.modified_ns+1, shape='stale'))
    assert index.load(root, True) == [updated]


def test_gallery_table_filters_selection_and_settings(app, tmp_path):
    root = tmp_path / 'images'
    root.mkdir()
    for n in range(3):
        Image.new('RGB',(20,20)).save(root / f'image{n}.png')
    sub = root / 'special folder'
    sub.mkdir()
    Image.new('RGB',(20,20)).save(sub / 'nested.jpg')
    widget, viewer, settings = make_widget(app, tmp_path, root)
    assert widget.proxy.rowCount() == 4
    assert widget.grid.visualRect(widget.proxy.index(0,0)).height() >= widget.thumbnail_slider.value()
    assert widget.table.columnWidth(0) >= 300
    assert any(not widget.model.item(row,0).icon().isNull() for row in range(4))
    widget.search_edit.setText('special nested')
    settle(app, widget)
    assert widget.proxy.rowCount() == 1
    widget.search_edit.clear()
    widget.kind.setCurrentText('JPEG')
    settle(app, widget)
    assert widget.proxy.rowCount() == 1
    widget.selection.select(widget.proxy.index(0,0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
    widget.view_mode.setCurrentText('Table')
    settle(app, widget)
    assert widget.selection.selectedRows(0)
    widget._open_selected()
    assert viewer.opened == [str(sub / 'nested.jpg')]
    widget.kind.setCurrentText('All images')
    widget.recursive_button.setChecked(False)
    settle(app, widget)
    assert widget.proxy.rowCount() == 3
    widget._show_folder(sub)
    settle(app, widget)
    assert widget.proxy.rowCount() == 1
    widget._show_folder(root)
    settle(app, widget)
    widget.thumbnail_slider.setValue(192)
    assert settings.value('view_mode') == 'Table'
    assert int(settings.value('thumbnail_size')) == 192
    assert settings.value('last_folder') == str(root)
    widget.close()
    restored = BrowserAssistantWidget(viewer, cache_root=tmp_path / 'cache', settings=settings)
    restored.show()
    settle(app, restored, lambda: not restored._scanning)
    assert restored.folder == root
    assert restored.thumbnail_slider.value() == 192
    assert restored.view_mode.currentText() == 'Table'
    restored.close()


def test_visible_previews_past_200_and_refresh_races(app, tmp_path):
    root = tmp_path / 'many'
    root.mkdir()
    for n in range(230):
        Image.new('L',(20,20),n % 255).save(root / f'{n:03}.png')
    widget, viewer, _ = make_widget(app, tmp_path, root)
    assert widget.model.rowCount() == 230
    widget.grid.scrollTo(widget.proxy.index(229,0), QAbstractItemView.PositionAtBottom)
    settle(app, widget, lambda: not widget._items[str(root / '229.png')].icon().isNull())
    assert not widget._items[str(root / '229.png')].icon().isNull()
    widget.search_edit.setText('229')
    widget.refresh()
    widget.refresh(use_cache=False)
    settle(app, widget)
    assert widget.proxy.rowCount() == 1
    assert widget.proxy.index(0,0).data(ROLE_PATH) == str(root / '229.png')
    widget.close()


@pytest.mark.parametrize("query,path,expected", [
    ("*.tif", "/a/IMAGE.TIF", True),
    ("*.tif", "/a/image.tiff", False),
    ("*.tif !mask", "/a/cell_mask.tif", False),
    ('"cell one" !mask', "/a/cell one.tif", True),
    ('!"cell one"', "/a/cell one.tif", False),
    ("cell sample", "/sample/cell.tif", True),
    ("cell sample", "/other/cell.tif", False),
    ("ext:tif;tiff", "/a/cell.TIFF", True),
    ("ext:tif;tiff", "/a/cell.png", False),
    ("*.tif !mask | *.nd2", "/a/mask.nd2", True),
    ('"a|b"', "/a/a|b.tif", True),
    ('"cell one', "/a/cell one.tif", True),
    ("", "/a/file", True),
    ("*/sample/*.tif", "/data/sample/cell.tif", True),
])
def test_compiled_search(query, path, expected):
    from napari_browser_assistant._search import compile_query
    assert compile_query(query)(path) is expected


def test_backup_migration_and_empty_snapshot(tmp_path):
    from napari_browser_assistant._library import migrate_index
    legacy = tmp_path / "cache" / "folder_index.sqlite3"
    index = FolderIndex(legacy)
    index.save(tmp_path, True, [])
    with sqlite3.connect(legacy) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("INSERT INTO libraries VALUES (?)", ("/offline/library",))
        connection.commit()
        target = tmp_path / "data" / "folder_index.sqlite3"
        migrate_index(target, [legacy])
    migrated = FolderIndex(target)
    assert migrated.has_snapshot(tmp_path, True)
    assert Path("/offline/library") in migrated.libraries()
    assert legacy.exists()
    migrate_index(target, [tmp_path / "missing"])
    assert migrated.has_snapshot(tmp_path, True)


def test_microscopy_discovery_prunes_zarr_chunks(tmp_path):
    from napari_browser_assistant._library import capabilities
    for name in ("sample.nd2", "sample.czi", "sample.lsm", "sample.mrc"):
        (tmp_path / name).touch()
    dataset = tmp_path / "sample.ome.zarr"
    (dataset / "0").mkdir(parents=True)
    (dataset / "0" / "chunk.tif").touch()
    (dataset / ".zattrs").write_text("{}")
    entries, errors = scan_images(tmp_path, True)
    assert not errors
    assert len(entries) == 5
    assert dataset in [e.path for e in entries]
    assert scan_images(dataset, True)[0][0].path == dataset
    assert "metadata/preview unavailable" in capabilities(tmp_path / "sample.nd2")


def test_ome_zarr_bounded_preview(tmp_path):
    zarr = pytest.importorskip("zarr")
    root = tmp_path / "sample.ome.zarr"
    group = zarr.open_group(str(root), mode="w")
    data = np.arange(2 * 64 * 64, dtype="uint16").reshape(2, 64, 64)
    array = group.create_dataset("0", shape=data.shape, chunks=(1, 32, 32), dtype=data.dtype)
    array[:] = data
    group.attrs["multiscales"] = [{"axes": [{"name": a} for a in ("z", "y", "x")],
                                  "datasets": [{"path": "0"}]}]
    entry = scan_images(tmp_path, True)[0][0]
    preview, updated = build_preview(entry, tmp_path / "cache")
    assert updated.shape == "2 x 64 x 64"
    assert updated.dtype == "uint16"
    with Image.open(preview) as image:
        assert image.size == (64, 64)
    assert np.array_equal(array[:], data)


def test_cached_navigation_multiple_libraries_and_offline(app, tmp_path, monkeypatch):
    import napari_browser_assistant._widget as module
    root = tmp_path / "images"
    sub = root / "sub"
    sub.mkdir(parents=True)
    Image.new("L", (10, 10)).save(sub / "one.png")
    widget, viewer, _ = make_widget(app, tmp_path, root)
    original_scan = module.scan_images
    def forbidden(*args):
        raise AssertionError("Unexpected rescan")
    monkeypatch.setattr(module, "scan_images", forbidden)
    widget.refresh()
    settle(app, widget)
    widget._show_folder(sub)
    settle(app, widget)
    assert widget.proxy.rowCount() == 1
    assert "Unexpected" not in widget.status_label.text()
    monkeypatch.setattr(module, "scan_images", original_scan)
    second = tmp_path / "second"
    second.mkdir()
    Image.new("L", (10, 10)).save(second / "two.png")
    widget.path_edit.setText(str(second))
    widget._change_root()
    settle(app, widget)
    root.rename(tmp_path / "disconnected")
    widget._pick_library(0)
    settle(app, widget)
    assert widget.proxy.rowCount() == 2
    assert str(sub / "one.png") in widget._offline
    assert "Offline" in widget.status_label.text()
    assert widget.index.db_path.parent != widget.cache_root
    assert len(widget.index.libraries()) == 2
    widget.close()


def test_first_run_does_not_scan_home(app, tmp_path, monkeypatch):
    import napari_browser_assistant._widget as module
    def forbidden(*args):
        raise AssertionError("First run scanned a folder")
    monkeypatch.setattr(module, "scan_images", forbidden)
    settings = QSettings(str(tmp_path / "first.ini"), QSettings.IniFormat)
    widget = BrowserAssistantWidget(SimpleNamespace(), cache_root=tmp_path / "cache", settings=settings)
    widget.show()
    settle(app, widget)
    assert widget.model.rowCount() == 0
    assert widget.index.libraries() == []
    assert "Add an image folder" in widget.status_label.text()
    widget.close()


def test_empty_library_refresh_does_not_rescan(app, tmp_path, monkeypatch):
    import napari_browser_assistant._widget as module
    root = tmp_path / "empty"
    root.mkdir()
    widget, _, _ = make_widget(app, tmp_path, root)
    def forbidden(*args):
        raise AssertionError("Empty snapshot rescanned")
    monkeypatch.setattr(module, "scan_images", forbidden)
    widget.refresh()
    settle(app, widget)
    assert "rescanned" not in widget.status_label.text()
    assert widget.index.has_snapshot(root, True)
    widget.close()


def test_snapshot_reconciliation(tmp_path):
    root = tmp_path / "images"
    child = root / "child"
    child.mkdir(parents=True)
    Image.new("L", (10, 10)).save(child / "one.png")
    index = FolderIndex(tmp_path / "index.sqlite")
    index.add_library(root)
    index.save(root, True, scan_images(root, True)[0])
    index.save(child, True, scan_images(child, True)[0])
    (child / "one.png").unlink()
    Image.new("L", (10, 10)).save(child / "two.png")
    index.save(child, True, scan_images(child, True)[0])
    assert [e.path.name for e in index.load(root, True)] == ["two.png"]
    (child / "two.png").unlink()
    index.save(root, True, [])
    assert index.load(child, True) == []
    assert index.has_snapshot(child, True)
    assert FolderIndex(index.db_path).libraries() == [root]


def test_widget_automatic_storage_migration(app, tmp_path, monkeypatch):
    from qtpy.QtCore import QStandardPaths
    cache, data = tmp_path / "qt-cache", tmp_path / "qt-data"
    monkeypatch.setattr(QStandardPaths, "writableLocation",
                        lambda location: str(cache if location == QStandardPaths.CacheLocation else data))
    root = tmp_path / "offline"
    legacy = cache / "napari-browser-assistant" / "folder_index.sqlite3"
    from napari_browser_assistant._library import ImageEntry
    index = FolderIndex(legacy)
    index.save(root, True, [ImageEntry(root / "saved.tif", 100, 1)])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    settings.setValue("library_root", str(root))
    widget = BrowserAssistantWidget(SimpleNamespace(), settings=settings)
    widget.show()
    settle(app, widget)
    assert widget.index.db_path == data / "napari-browser-assistant" / "folder_index.sqlite3"
    assert widget.cache == cache / "napari-browser-assistant" / "thumbnails"
    assert widget.proxy.rowCount() == 1
    assert "Offline" in widget.proxy.index(0, 0).data()
    assert legacy.exists()
    widget.close()


def test_zarr_decode_budget_prevents_array_read(tmp_path, monkeypatch):
    import sys
    from napari_browser_assistant._library import _zarr_preview
    class Array:
        shape = (1, 10000, 10000)
        ndim = 3
        chunks = (1, 10000, 10000)
        dtype = np.dtype("uint16")
        def __getitem__(self, key):
            raise AssertionError("Oversized chunk was decoded")
    class Group:
        attrs = {"multiscales": [{"axes": ["z", "y", "x"], "datasets": [{"path": "0"}]}]}
        def __getitem__(self, key):
            return Array()
    monkeypatch.setitem(sys.modules, "zarr", SimpleNamespace(open_group=lambda *a, **k: Group(),
                                                           open_array=lambda *a, **k: Array()))
    with pytest.raises(ValueError, match="decode budget"):
        _zarr_preview(tmp_path)


def test_large_model_uses_lazy_cells(app, tmp_path):
    from napari_browser_assistant._model import LibraryModel
    from napari_browser_assistant._library import ImageEntry
    model = LibraryModel()
    entries = [ImageEntry(tmp_path / f"image{n}.tif", n, 1) for n in range(20000)]
    model.set_entries(entries, set())
    assert model.rowCount() == 20000
    assert model.overrides == {}
    assert model.index(19999, 0).data() == "image19999.tif"


@pytest.mark.parametrize("name,stem", [
    ("sample_mask.tif", "sample"), ("sample_m.tif", "sample"),
    ("sample_M.TIFF", "sample"), ("sample_MASK.ome.tif", "sample"),
    ("sample_m.ome.zarr", "sample"),
    ("sample_masked.tif", None), ("sample_more.tif", None),
    ("sample_m_extra.tif", None), ("_m.tif", None),
])
def test_mask_suffixes(name, stem):
    from napari_browser_assistant._masks import mask_source_stem
    assert mask_source_stem(Path(name)) == stem


def test_mask_pairing_ambiguity_and_manual_persistence(tmp_path):
    from napari_browser_assistant._library import ImageEntry
    index = FolderIndex(tmp_path / "index.sqlite")
    names = ["cell.tif", "cell_m.tif", "cell_mask.png", "ambiguous.tif",
             "ambiguous.png", "ambiguous_m.tif", "missing_m.tif", "cell_more.tif"]
    entries = [ImageEntry(tmp_path / name, 1, 1) for name in names]
    records = {e.path.name: e for e in index.annotate(entries)}
    assert records["cell_m.tif"].source_path == str(tmp_path / "cell.tif")
    assert records["cell_mask.png"].source_path == str(tmp_path / "cell.tif")
    assert records["cell_m.tif"].pairing == "Suggested pair"
    assert records["ambiguous_m.tif"].pairing == "Ambiguous source"
    assert records["ambiguous_m.tif"].source_path == ""
    assert records["missing_m.tif"].pairing == "Missing source"
    assert records["cell_more.tif"].role == "image"
    index.set_annotation(tmp_path / "cell_m.tif", "image")
    index.set_annotation(tmp_path / "ambiguous_m.tif", "mask", tmp_path / "ambiguous.png")
    restored = FolderIndex(index.db_path)
    records = {e.path.name: e for e in restored.annotate(entries)}
    assert records["cell_m.tif"].role == "image"
    assert records["ambiguous_m.tif"].pairing == "Confirmed pair"
    assert records["ambiguous_m.tif"].source_path.endswith("ambiguous.png")
    index.reset_annotation(tmp_path / "cell_m.tif")
    assert {e.path.name: e for e in index.annotate(entries)}["cell_m.tif"].role == "mask"
    index.set_annotation(tmp_path / "cell_mask.png", "mask")
    assert {e.path.name: e for e in index.annotate(entries)}["cell_mask.png"].pairing == "Unpaired"


def test_pairing_is_same_folder_and_handles_ome_names(tmp_path):
    from napari_browser_assistant._library import ImageEntry
    from napari_browser_assistant._masks import pair_masks
    paths = [tmp_path / "a" / "sample.ome.tif",
             tmp_path / "a" / "sample_m.tif",
             tmp_path / "b" / "sample_mask.tif"]
    records = pair_masks([ImageEntry(p, 1, 1) for p in paths], {})
    assert records[1].source_path == str(paths[0])
    assert records[2].pairing == "Missing source"


@pytest.mark.parametrize("suffix", ["tif", "png"])
def test_mask_previews_preserve_labels_and_transparent_background(tmp_path, suffix):
    from napari_browser_assistant._library import ImageEntry
    from napari_browser_assistant._masks import label_preview
    data = np.zeros((600, 600), dtype="uint16")
    data[100:300, 100:300] = 1
    data[300:550, 300:550] = 4097
    path = tmp_path / ("sample_m." + suffix)
    if suffix == "tif":
        tifffile.imwrite(path, data)
    else:
        Image.fromarray(data).save(path)
    before = path.read_bytes()
    entry = ImageEntry(path, path.stat().st_size, path.stat().st_mtime_ns, role="mask")
    target, updated = build_preview(entry, tmp_path / "cache")
    assert updated.role == "mask"
    with Image.open(target) as image:
        assert image.mode == "RGBA"
        assert max(image.size) <= 256
        colors = {tuple(color) for color in np.asarray(image).reshape(-1, 4)}
    expected = {tuple(color) for color in np.asarray(label_preview(np.array([[0, 1, 4097]], dtype="uint16"))).reshape(-1, 4)}
    assert colors == expected  # nearest-neighbor preview introduces no blended labels
    assert (0, 0, 0, 0) in colors
    assert path.read_bytes() == before
    assert build_preview(entry, tmp_path / "cache")[0] == target
    ordinary, _ = build_preview(replace(entry, role="image"), tmp_path / "cache")
    assert ordinary != target


@pytest.mark.parametrize("data", [
    np.zeros((4, 4, 3), dtype="uint8"),
    np.zeros((4, 4), dtype="float32"),
    np.full((4, 4), -1, dtype="int16"),
])
def test_invalid_label_previews_are_reported(data):
    from napari_browser_assistant._masks import label_preview
    with pytest.raises(ValueError, match="Mask"):
        label_preview(data)


def test_mask_browser_tabs_pair_filter_and_roles(app, tmp_path, monkeypatch):
    import napari_browser_assistant._widget as module
    root = tmp_path / "masks"
    root.mkdir()
    for name in ("cell.tif", "cell_m.tif", "cell_mask.tif", "orphan_m.tif"):
        tifffile.imwrite(root / name, np.ones((16, 16), dtype="uint16"))
    widget, viewer, settings = make_widget(app, tmp_path, root)
    def forbidden(*args):
        raise AssertionError("Mask management rescanned the folder")
    monkeypatch.setattr(module, "scan_images", forbidden)
    widget.asset_view.setCurrentIndex(1)
    settle(app, widget)
    assert widget.proxy.rowCount() == 1
    widget.asset_view.setCurrentIndex(2)
    settle(app, widget)
    assert widget.proxy.rowCount() == 3
    assert widget.mask_pair_filter.isVisible()
    widget.mask_pair_filter.setCurrentText("Needs pairing")
    assert widget.proxy.rowCount() == 1
    assert widget.proxy.index(0, 0).data(ROLE_PATH).endswith("orphan_m.tif")
    widget.mask_pair_filter.setCurrentText("All pairs")
    widget._save_annotation(str(root / "cell_m.tif"), "image")
    settle(app, widget)
    assert widget.proxy.rowCount() == 2
    widget._save_annotation(str(root / "cell_mask.tif"), "mask", str(root / "cell.tif"))
    settle(app, widget)
    widget.mask_pair_filter.setCurrentText("Confirmed pair")
    assert widget.proxy.rowCount() == 1
    assert settings.value("asset_view") == "Masks"
    assert "rescanned" not in widget.status_label.text()
    widget.close()


def test_open_mask_pair_as_image_and_labels_and_warn_mismatch(app, tmp_path):
    root = tmp_path / "masks"
    root.mkdir()
    for name in ("cell.tif", "cell_m.tif", "cell_mask.tif", "orphan_m.tif"):
        tifffile.imwrite(root / name, np.ones((16, 16), dtype="uint16"))
    widget, viewer, _ = make_widget(app, tmp_path, root)
    calls = []
    def open_file(path, **kwargs):
        calls.append((Path(path).name, kwargs))
        shape = (8, 8) if kwargs["layer_type"] == "labels" else (16, 16)
        return [SimpleNamespace(data=np.zeros(shape), rgb=False)]
    viewer.open = open_file
    widget._open_paths([str(root / n) for n in ("cell.tif", "cell_m.tif", "cell_mask.tif")])
    assert [name for name, _ in calls] == ["cell.tif", "cell_m.tif", "cell_mask.tif"]
    assert [kwargs["layer_type"] for _, kwargs in calls] == ["image", "labels", "labels"]
    assert all(kwargs["plugin"] is None for _, kwargs in calls)
    assert "Shape mismatch" in widget.status_label.text()
    calls.clear()
    widget._open_paths([str(root / "orphan_m.tif")])
    assert calls[0][1]["layer_type"] == "labels"
    assert "Missing source" in widget.status_label.text()
    widget.close()


def test_real_napari_mask_opening(app, tmp_path, monkeypatch):
    from napari.components import ViewerModel
    from npe2 import PluginManager
    # A bare ViewerModel does not initialize GUI plugin discovery.
    manager = PluginManager()
    monkeypatch.setattr(PluginManager, "instance", classmethod(lambda cls: manager))
    manager.register("napari")
    root = tmp_path / "paired"
    root.mkdir()
    image = np.arange(256, dtype="uint16").reshape(16, 16)
    labels = np.zeros((16, 16), dtype="uint16")
    labels[3:10, 4:12] = 12
    tifffile.imwrite(root / "cell.tif", image)
    tifffile.imwrite(root / "cell_m.tif", labels)
    widget, _, _ = make_widget(app, tmp_path, root)
    widget.viewer = ViewerModel()
    widget._open_paths([str(root / "cell_m.tif")])
    assert len(widget.viewer.layers) == 2, widget.status_label.text()
    assert widget.viewer.layers[0]._type_string == "image"
    assert widget.viewer.layers[1]._type_string == "labels"
    assert np.array_equal(widget.viewer.layers[1].data, labels)
    assert "Shape mismatch" not in widget.status_label.text()
    widget.viewer.layers.clear()
    widget.close()
