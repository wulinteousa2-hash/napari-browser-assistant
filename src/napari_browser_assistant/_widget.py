"""Shared gallery/table browser with bounded, visible-first background previews."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3

from qtpy.QtCore import (
    QDir, QEvent, QPoint, QItemSelectionModel, QObject, QRunnable, QSettings, QSize,
    QSortFilterProxyModel, QStandardPaths, Qt, QThreadPool, QTimer, QUrl, Signal, Slot,
)
from qtpy.QtGui import QDesktopServices, QIcon
from qtpy.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QFileSystemModel, QHBoxLayout,
    QLabel, QLineEdit, QListView, QMenu, QPushButton, QSlider, QSplitter,
    QStackedWidget, QStyledItemDelegate, QTabBar, QTabWidget, QTableView, QTreeView, QVBoxLayout, QWidget,
)
from ._library import (DEFAULT_ROOT, TIFF_SUFFIXES, FolderIndex, build_preview, scan_images,
                       migrate_index, capabilities, MICROSCOPY_SUFFIXES)
from ._search import compile_query
from ._model import LibraryModel, ROLE_ASSET, ROLE_PAIRING
from ._masks import pair_shape_warning
from ._reader import workspace_manifest, is_ome_zarr, read_dataset

ROLE_PATH = Qt.UserRole + 1
ROLE_SORT = Qt.UserRole + 2


class FilterModel(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.matches = compile_query('')
        self.extensions = None
        self.asset_role = None
        self.pair_filter = "All pairs"
        self.folder = Path.home()
        self.setSortRole(ROLE_SORT)

    def filterAcceptsRow(self, row, parent):
        text = self.sourceModel().index(row, 0, parent).data(ROLE_PATH)
        if not text:
            return False
        index = self.sourceModel().index(row, 0, parent)
        if self.asset_role and index.data(ROLE_ASSET) != self.asset_role:
            return False
        if self.asset_role == "mask" and self.pair_filter != "All pairs":
            pairing = index.data(ROLE_PAIRING)
            if self.pair_filter == "Needs pairing":
                if pairing not in ("Missing source", "Ambiguous source", "Unpaired"):
                    return False
            elif pairing != self.pair_filter:
                return False
        path = Path(text)
        return (self.matches(str(path))
                and (self.extensions is None or path.suffix.lower() in self.extensions))


class GalleryDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        # Keep empty and loaded previews the same size, avoiding cached tiny hints.
        return self.parent().gridSize()


class TaskSignals(QObject):
    done = Signal(int, object, object)


class Task(QRunnable):
    def __init__(self, identifier, function):
        super().__init__()
        self.identifier, self.function = identifier, function
        self.signals = TaskSignals()

    @Slot()
    def run(self):
        try:
            result = self.function()
        except Exception as error:
            self.signals.done.emit(self.identifier, None, str(error))
        else:
            self.signals.done.emit(self.identifier, result, None)


class BrowserAssistantWidget(QWidget):
    def __init__(self, napari_viewer, *, cache_root=None, data_root=None, settings=None):
        super().__init__()
        self.viewer = napari_viewer
        self.setWindowTitle('Browser Assistant')
        base = Path(QStandardPaths.writableLocation(QStandardPaths.CacheLocation)
                    or Path.home() / '.cache')
        self.cache_root = Path(cache_root) if cache_root is not None else base / 'napari-browser-assistant'
        persistent = Path(QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
                          or Path.home() / '.local' / 'share') / 'napari-browser-assistant'
        # An explicit cache root is also an isolated test/embedding default.
        self.data_root = Path(data_root) if data_root is not None else (
            Path(cache_root).parent / 'library-data' if cache_root is not None else persistent)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        if settings is None:
            self.settings = QSettings(str(self.data_root / 'settings.ini'), QSettings.IniFormat)
            if not self.settings.contains('migration_complete'):
                legacy_settings = QSettings('napari-browser-assistant', 'napari-browser-assistant')
                for key in legacy_settings.allKeys():
                    if not self.settings.contains(key):
                        self.settings.setValue(key, legacy_settings.value(key))
                self.settings.setValue('migration_complete', True)
                self.settings.sync()
        else:
            self.settings = settings
        db = self.data_root / 'folder_index.sqlite3'
        migrate_index(db, [self.cache_root / 'folder_index.sqlite3', base / 'folder_index.sqlite3'])
        self.index = FolderIndex(db)
        self.cache = self.cache_root / 'thumbnails'
        self.thread_pool = QThreadPool(self)
        self.thread_pool.setMaxThreadCount(2)
        self.index_pool = QThreadPool(self)
        self.index_pool.setMaxThreadCount(1)
        self._tasks = {}
        self._identifier = self._generation = 0
        self._pending = set()
        self._completed = set()
        self._entries, self._items = {}, {}
        self._scanning = False
        self._scan_error = ''
        self._all_libraries = False
        self._configured = False
        self._offline = set()
        self.root = self.folder = DEFAULT_ROOT if DEFAULT_ROOT.is_dir() else Path.home()
        self._initial_thumbnail_size = max(64, min(256, int(self.settings.value('thumbnail_size', 160))))
        self._build_ui()
        QTimer.singleShot(0, self._restore)

    def _build_ui(self):
        outer = QVBoxLayout(self)
        self.main_tabs = QTabWidget()
        outer.addWidget(self.main_tabs)
        library_page = QWidget()
        layout = QVBoxLayout(library_page)
        self.main_tabs.addTab(library_page, 'Library')
        heading = QLabel('Image Library')
        heading.setStyleSheet('font-size: 20px; font-weight: bold')
        layout.addWidget(heading)
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText('Image folder, OME-Zarr directory, or workspace JSON')
        self.path_edit.setToolTip('Enter a folder, OME-Zarr directory, or Label Assistant JSON and press Enter')
        self.path_edit.returnPressed.connect(self._change_root)
        for widget in [self.path_edit,
                       self._button('Add folder…', self._browse_folder),
                       self._button('Open JSON…', self._browse_workspace),
                       self._button('Refresh', self.refresh),
                       self._button('Refresh Index', lambda: self.refresh(use_cache=False))]:
            path_row.addWidget(widget)
        layout.addLayout(path_row)
        self.library_picker = QComboBox()
        self.library_picker.addItem('Choose an image folder…', None)
        self.library_picker.activated.connect(self._pick_library)
        layout.addWidget(self.library_picker)
        self.asset_view = QTabBar()
        for title in ("All files", "Images", "Masks"):
            self.asset_view.addTab(title)
        self.asset_view.currentChanged.connect(self._apply_filter)
        layout.addWidget(self.asset_view)
        mask_controls = QHBoxLayout()
        self.mask_hint = QLabel('Masks: _mask or _m · colored label IDs · background 0 transparent')
        self.mask_hint.setWordWrap(True)
        self.mask_pair_filter = QComboBox()
        self.mask_pair_filter.addItems(['All pairs', 'Needs pairing', 'Suggested pair', 'Confirmed pair'])
        self.mask_pair_filter.currentTextChanged.connect(self._apply_filter)
        self.pair_button = self._button('Pair source…', self._pair_selected)
        self.pair_button.setToolTip('Choose the source image for one selected mask; saved in the library')
        for widget in (self.mask_hint, self.mask_pair_filter, self.pair_button):
            mask_controls.addWidget(widget)
            widget.hide()
        layout.addLayout(mask_controls)
        filters = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText('*.tif !mask | ext:nd2;czi')
        self.search_edit.setToolTip('AND: multiple terms; OR: |; exclusions: !mask; phrases: "cell one"; extensions: ext:tif;tiff')
        self.search_edit.textChanged.connect(self._apply_filter)
        self.kind = QComboBox()
        self.kind.addItems(['All images', 'TIFF', 'JPEG', 'PNG', 'Datasets / workspaces', 'Other images'])
        self.kind.currentIndexChanged.connect(self._apply_filter)
        self.recursive_button = QPushButton('Include subfolders')
        self.recursive_button.setCheckable(True)
        self.recursive_button.setChecked(True)
        self.recursive_button.toggled.connect(lambda _: self.refresh())
        self.view_mode = QComboBox()
        self.view_mode.addItems(['Gallery', 'Table'])
        self.view_mode.currentIndexChanged.connect(self._set_view_mode)
        for widget in [self.search_edit, self.kind, self.recursive_button, self.view_mode]:
            filters.addWidget(widget)
        layout.addLayout(filters)
        navigation = QHBoxLayout()
        navigation.addWidget(self._button('Library root', lambda: self._show_folder(self.root)))
        self.breadcrumb = QLabel()
        self.breadcrumb.setWordWrap(True)
        navigation.addWidget(self.breadcrumb, 1)
        layout.addLayout(navigation)
        splitter = QSplitter()
        self.folders = QFileSystemModel(self)
        self.folders.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot)
        self.tree = QTreeView()
        self.tree.setModel(self.folders)
        self.tree.setHeaderHidden(True)
        for column in range(1, 4):
            self.tree.hideColumn(column)
        self.tree.clicked.connect(lambda idx: self._show_folder(Path(self.folders.filePath(idx))))
        splitter.addWidget(self.tree)
        self.model = LibraryModel(self)
        self.proxy = FilterModel(self)
        self.proxy.setSourceModel(self.model)
        self.selection = QItemSelectionModel(self.proxy, self)
        self.selection.selectionChanged.connect(self._update_mask_controls)
        self.grid = QListView()
        self.grid.setViewMode(QListView.IconMode)
        self.grid.setResizeMode(QListView.Adjust)
        self.grid.setMovement(QListView.Static)
        self.grid.setUniformItemSizes(True)
        self.grid.setLayoutMode(QListView.Batched)
        self.grid.setBatchSize(200)
        self.grid.setItemDelegate(GalleryDelegate(self.grid))
        self.grid.setWordWrap(True)
        self.grid.setSpacing(8)
        self.table = QTableView()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.stack = QStackedWidget()
        for view in (self.grid, self.table):
            view.setModel(self.proxy)
            view.setSelectionModel(self.selection)
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
            view.setEditTriggers(QAbstractItemView.NoEditTriggers)
            view.doubleClicked.connect(self._open_index)
            view.setContextMenuPolicy(Qt.CustomContextMenu)
            view.customContextMenuRequested.connect(
                lambda pos, v=view: self._context_menu(v, pos))
            view.verticalScrollBar().valueChanged.connect(self._queue_visible)
            view.horizontalScrollBar().valueChanged.connect(self._queue_visible)
            view.viewport().installEventFilter(self)
            self.stack.addWidget(view)
        for column, width in enumerate((300, 160, 100, 100, 150, 280, 80, 300, 160)):
            self.table.setColumnWidth(column, width)
        self.proxy.sort(0)
        splitter.addWidget(self.stack)
        splitter.setSizes([180, 850])
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        controls = QHBoxLayout()
        controls.addWidget(self._button('Use Current Layer Folder', self._use_current_layer_folder))
        controls.addWidget(self._button('Open Selected', self._open_selected))
        controls.addStretch()
        controls.addWidget(QLabel('Thumbnail size'))
        self.thumbnail_slider = QSlider(Qt.Horizontal)
        self.thumbnail_slider.setRange(64, 256)
        self.thumbnail_slider.setValue(self._initial_thumbnail_size)
        self.thumbnail_slider.valueChanged.connect(self._set_thumbnail_size)
        controls.addWidget(self.thumbnail_slider)
        layout.addLayout(controls)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self._visible_timer = QTimer(self)
        self._visible_timer.setSingleShot(True)
        self._visible_timer.setInterval(80)
        self._visible_timer.timeout.connect(self._schedule_visible)
        self._set_thumbnail_size(self._initial_thumbnail_size)
        from ._discover_widget import DiscoverWidget
        self.discover = DiscoverWidget(self)
        self.main_tabs.addTab(self.discover, 'Discover')
        from ._metadata_widget import MetadataWidget
        self.metadata_tab = MetadataWidget(self)
        self.main_tabs.addTab(self.metadata_tab, 'Metadata')
        self.main_tabs.currentChanged.connect(self._queue_visible)
        self.setMinimumWidth(500)

    def closeEvent(self, event):
        if hasattr(self, "discover"):
            self.discover.cancel_work()
        super().closeEvent(event)

    def _button(self, text, callback):
        button = QPushButton(text)
        button.clicked.connect(lambda _checked=False: callback())
        return button

    def _restore(self):
        saved = Path(str(self.settings.value('last_folder', '')))
        root = Path(str(self.settings.value('library_root', '')))
        if str(root) != '.':
            self.root = root.resolve()
        elif str(saved) != '.':
            self.root = saved.resolve()
        self.folder = self.root
        if str(saved) != '.' and saved.is_dir():
            try:
                saved.resolve().relative_to(self.root)
                self.folder = saved.resolve()
            except ValueError:
                pass
        self.path_edit.setText(str(self.root))
        self.folders.setRootPath(str(self.root))
        self.tree.setRootIndex(self.folders.index(str(self.root)))
        self.thumbnail_slider.setValue(max(64, min(256, int(self.settings.value('thumbnail_size', 160)))))
        asset_tab = str(self.settings.value('asset_view', 'All files'))
        self.asset_view.setCurrentIndex({'All files': 0, 'Images': 1, 'Masks': 2}.get(asset_tab, 0))
        self.view_mode.setCurrentIndex(1 if self.settings.value('view_mode', 'Gallery') == 'Table' else 0)
        configured = str(root) != '.' or str(saved) != '.' or bool(self.index.libraries())
        self._configured = configured
        if configured:
            if str(root) == '.' and str(saved) == '.':
                self.root = self.folder = self.index.libraries()[0]
                self.path_edit.setText(str(self.root))
                self.folders.setRootPath(str(self.root))
                self.tree.setRootIndex(self.folders.index(str(self.root)))
            self.index.add_library(self.root)
            self._update_libraries()
            self.refresh()
        else:
            self.path_edit.clear()
            self.status_label.setText('Add an image folder to build your library. No folders are scanned automatically.')

    def _update_libraries(self):
        self.library_picker.clear()
        self.library_picker.addItem('All libraries', None)
        for folder in self.index.libraries():
            self.library_picker.addItem(str(folder), str(folder))
        self.library_picker.setCurrentIndex(max(0, self.library_picker.findData(str(self.root))))

    def _pick_library(self, index):
        folder = self.library_picker.itemData(index)
        self._all_libraries = folder is None
        if folder is not None:
            self.root = self.folder = Path(folder)
            self.path_edit.setText(folder)
            self.folders.setRootPath(folder)
            self.tree.setRootIndex(self.folders.index(folder))
        self.refresh()

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Choose image library', str(self.root))
        if folder:
            self.path_edit.setText(folder)
            self._change_root()

    def _browse_workspace(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Label Assistant workspace image', str(self.root),
            'Label Assistant workspace (*.json)')
        if not path:
            return
        if workspace_manifest(path) is None:
            self.status_label.setText('Choose a valid Label Assistant workspace JSON (version 1).')
            return
        self.path_edit.setText(path)
        self._change_root()
        self._open_paths([path])

    def _change_root(self):
        if not self.path_edit.text().strip():
            self.status_label.setText('Choose an image folder to add.')
            return
        root = Path(self.path_edit.text().strip()).expanduser()
        manifest = workspace_manifest(root)
        if manifest is not None:
            root = manifest.parent
        if not root.is_dir():
            self.status_label.setText(f'Folder not found: {root}')
            return
        self.root = root.resolve()
        self.path_edit.setText(str(self.root))
        self._configured = True
        self.index.add_library(self.root)
        self._update_libraries()
        self.folders.setRootPath(str(self.root))
        self.tree.setRootIndex(self.folders.index(str(self.root)))
        self._show_folder(self.root, use_cache=manifest is None)

    def _show_folder(self, folder, use_cache=True):
        self._all_libraries = False
        self.folder = folder
        self.refresh(use_cache=use_cache)

    def refresh(self, use_cache=True):
        if not self._configured:
            self.status_label.setText('Add an image folder to build your library.')
            return
        self._generation += 1
        generation = self._generation
        self._scan_error = ''
        self._scanning = True
        self._completed.clear()
        self.settings.setValue('last_folder', str(self.folder))
        self.settings.setValue('library_root', str(self.root))
        self.breadcrumb.setText('All libraries' if self._all_libraries else str(self.folder))
        folder, recursive = self.folder, self.recursive_button.isChecked()
        all_libraries = self._all_libraries
        self.status_label.setText('Loading saved index…')
        def load():
            roots = self.index.libraries() if all_libraries else [folder]
            entries, errors, offline = {}, [], set()
            for root in roots:
                recurse = True if all_libraries else recursive
                cached = self.index.load(root, recurse)
                if not root.is_dir():
                    offline.update(str(e.path) for e in cached)
                    errors.append(f'Offline: {root}')
                elif not use_cache or not self.index.has_snapshot(root, recurse):
                    scanned, scan_errors = scan_images(root, recurse)
                    old = {e.signature: e for e in cached}
                    scanned = [replace(e, shape=old[e.signature].shape, dtype=old[e.signature].dtype)
                               if e.signature in old else e for e in scanned]
                    if scan_errors:
                        # Inaccessible subfolders must not erase saved entries.
                        merged = {str(e.path): e for e in cached}
                        merged.update({str(e.path): e for e in scanned})
                        scanned = list(merged.values())
                    self.index.save(root, recurse, scanned)
                    cached = scanned
                    errors.extend(scan_errors)
                entries.update({str(e.path): e for e in cached})
            decorated = self.index.annotate(sorted(entries.values(), key=lambda e: str(e.path).casefold()))
            return decorated, errors, offline
        self._submit('scan', generation, None, load)

    def _set_entries(self, entries):
        self._entries = {str(e.path): e for e in entries}
        self._items = {str(e.path): self.model.item(row) for row, e in enumerate(entries)}
        self.proxy.folder = self.folder
        self.model.set_entries(entries, self._offline)
        self._apply_filter()

    def _apply_filter(self, *_):
        tab = self.asset_view.currentIndex()
        self.proxy.asset_role = {1: 'image', 2: 'mask'}.get(tab)
        self.proxy.pair_filter = self.mask_pair_filter.currentText()
        self.settings.setValue('asset_view', self.asset_view.tabText(tab))
        for widget in (self.mask_hint, self.mask_pair_filter, self.pair_button):
            widget.setVisible(tab == 2)
        self._update_mask_controls()
        self.proxy.matches = compile_query(self.search_edit.text())
        self.proxy.extensions = {'TIFF': TIFF_SUFFIXES, 'JPEG': {'.jpg', '.jpeg'},
                                 'PNG': {'.png'}, 'Datasets / workspaces': {'.zarr', '.json'}, 'Other images': {'.bmp', '.gif', '.webp'} | MICROSCOPY_SUFFIXES | {'.zarr'}}.get(self.kind.currentText())
        self.proxy.invalidate()
        self._update_status()
        self._queue_visible()

    def _set_view_mode(self, index):
        self.stack.setCurrentIndex(index)
        self.settings.setValue('view_mode', self.view_mode.currentText())
        self._queue_visible()

    def _set_thumbnail_size(self, value):
        size = QSize(value, value)
        self.grid.setIconSize(size)
        self.grid.setGridSize(QSize(value + 32, value + 56))
        self.table.setIconSize(size)
        self.table.verticalHeader().setDefaultSectionSize(value + 12)
        self.settings.setValue('thumbnail_size', value)
        self._queue_visible()

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Resize, QEvent.Show):
            self._queue_visible()
        return super().eventFilter(watched, event)

    def _queue_visible(self, *_):
        if hasattr(self, '_visible_timer'):
            self._visible_timer.start()

    def _visible_entries(self):
        view = self.stack.currentWidget()
        rect = view.viewport().rect()
        if view is self.table:
            first = max(0, view.rowAt(0))
            last = view.rowAt(rect.bottom())
            rows = range(first, min(self.proxy.rowCount(), last + 1 if last >= 0 else first + rect.height() // max(1, view.verticalHeader().defaultSectionSize()) + 2))
        else:
            # Probe the viewport on a grid finer than an item; avoid walking the library.
            found = set()
            step_x = max(1, view.gridSize().width() // 2)
            step_y = max(1, view.gridSize().height() // 2)
            for y in range(0, rect.height(), step_y):
                for x in range(0, rect.width(), step_x):
                    idx = view.indexAt(QPoint(x, y))
                    if idx.isValid():
                        found.add(idx.row())
            rows = sorted(found)
        for row in rows:
            idx = self.proxy.index(row, 0)
            if view.visualRect(idx).intersects(rect):
                yield self._entries[idx.data(ROLE_PATH)]

    def _schedule_visible(self):
        if self.main_tabs.currentWidget() is self.discover:
            return
        if self._scanning:
            return
        active = sum(context[0] == 'preview' for context in self._tasks.values())
        for entry in self._visible_entries():
            if active >= 2:
                break
            if str(entry.path) in self._offline or entry.path.suffix.lower() in MICROSCOPY_SUFFIXES:
                continue
            key = (self._generation, entry.signature)
            if key in self._pending or key in self._completed:
                continue
            self._pending.add(key)
            self._submit('preview', self._generation, entry,
                         lambda e=entry: self._preview(e))
            active += 1

    def _preview(self, entry):
        path, updated = build_preview(entry, self.cache)
        warning = ''
        try:
            self.index.update_metadata(updated)
        except sqlite3.Error as error:
            warning = f'Index metadata update failed: {error}'
        return path, updated, warning

    def _submit(self, kind, generation, entry, function):
        self._identifier += 1
        identifier = self._identifier
        task = Task(identifier, function)
        self._tasks[identifier] = (kind, generation, entry, task)
        task.signals.done.connect(self._task_done)
        (self.index_pool if kind in ('scan', 'annotation') else self.thread_pool).start(task)

    @Slot(int, object, object)
    def _task_done(self, identifier, result, error):
        kind, generation, entry, task = self._tasks.pop(identifier)
        if kind == 'annotation':
            if error:
                self.status_label.setText(f'Could not save mask details: {error}')
            else:
                self.refresh()
            return
        if kind == 'preview':
            key = (generation, entry.signature)
            self._pending.discard(key)
        if generation != self._generation:
            self._queue_visible()
            return
        if kind == 'scan':
            self._scanning = False
            if error:
                self._scan_error = f'Scan failed: {error}'
            else:
                entries, errors, self._offline = result
                self._scan_error = '; '.join(errors[:3])
                self._completed.clear()
                self._set_entries(entries)
        else:
            self._completed.add(key)
            item = self._items.get(str(entry.path))
            if item is not None:
                if error:
                    item.setToolTip(f'{entry.path}\nPreview unavailable: {error}')
                else:
                    path, updated, warning = result
                    if warning:
                        self._scan_error = warning
                    self._entries[str(entry.path)] = updated
                    item.setIcon(QIcon(str(path)))
                    for column, value in [(1, updated.shape), (2, updated.dtype)]:
                        cell = self.model.item(item.row(), column)
                        cell.setText(value)
                        cell.setData(value, ROLE_SORT)
        self._update_status()
        self._queue_visible()

    def _update_status(self):
        if not hasattr(self, 'status_label'):
            return
        noun = 'masks' if self.proxy.asset_role == 'mask' else 'images' if self.proxy.asset_role == 'image' else 'files'
        text = f'{self.proxy.rowCount()} matching {noun} · {len(self._entries)} indexed files'
        if self._scanning:
            text += ' · scanning…'
        if self._scan_error:
            text += ' · ' + self._scan_error
        self.status_label.setText(text)

    def _open_index(self, index):
        if index.isValid():
            self._open_paths([self.proxy.index(index.row(), 0).data(ROLE_PATH)])

    def _open_selected(self):
        paths = [index.data(ROLE_PATH) for index in self.selection.selectedRows(0)]
        self._open_paths(paths)

    def _open_paths(self, paths):
        errors, warnings, opened, attempted = [], [], {}, set()

        def open_one(path, role):
            key = (path, role)
            if key in attempted:
                return opened.get(key, [])
            attempted.add(key)
            try:
                if not Path(path).exists():
                    raise FileNotFoundError(f'Offline or missing: {path}')
                own_reader = workspace_manifest(path) is not None or is_ome_zarr(path)
                kwargs = {'plugin': 'napari-browser-assistant' if own_reader else None,
                          'layer_type': 'labels' if role == 'mask' else 'image'}
                layers = self.viewer.open(path, **kwargs)
                if own_reader and role == 'mask':
                    for layer in layers or []:
                        layer.editable = False
                    warnings.append(f'{Path(path).name}: Zarr labels opened read-only')
                if role == 'image':
                    for image_index, layer in enumerate(layers or []):
                        if hasattr(layer, 'metadata'):
                            layer.metadata['browser_source'] = {'path': str(Path(path).resolve()),
                                                                'image_index': image_index}
                opened[key] = layers or []
                self.settings.setValue('last_open_file', path)
                return opened[key]
            except Exception as error:
                errors.append(f'{Path(path).name}: {error}')
                return []

        for path in paths:
            entry = self._entries.get(path)
            if entry is not None and entry.role == 'mask':
                source_layers = open_one(entry.source_path, 'image') if entry.source_path else []
                mask_layers = open_one(path, 'mask')
                if entry.source_path:
                    warning = pair_shape_warning(source_layers, mask_layers)
                    if warning:
                        warnings.append(f'{entry.path.name}: {warning}')
                else:
                    warnings.append(f'{entry.path.name}: {entry.pairing}; opened labels only')
            else:
                open_one(path, 'image')
        summary = f'Opened {len(opened)} file(s).' if paths else 'Select files to open.'
        self.status_label.setText('\n'.join([summary] + errors + warnings))
        return opened

    def _update_mask_controls(self, *_):
        if not hasattr(self, 'selection'):
            return
        selected = self.selection.selectedRows(0)
        self.pair_button.setEnabled(len(selected) == 1 and selected[0].data(ROLE_ASSET) == 'mask')

    def _save_annotation(self, path, role, source_path=''):
        self._submit('annotation', self._generation, None,
                     lambda: self.index.set_annotation(path, role, source_path))

    def _pair_selected(self):
        selected = self.selection.selectedRows(0)
        if len(selected) == 1:
            self._choose_source(selected[0].data(ROLE_PATH))

    def _choose_source(self, path):
        entry = self._entries[path]
        if entry.role != 'mask':
            return
        source, _ = QFileDialog.getOpenFileName(
            self, f'Choose source image for {entry.path.name}',
            entry.source_path or str(entry.path.parent))
        if source:
            known = self._entries.get(source)
            if source == path or (known and known.role == 'mask'):
                self.status_label.setText('Choose a source image, not a mask.')
                return
            self._save_annotation(path, 'mask', source)

    def _context_menu(self, view, pos):
        index = view.indexAt(pos)
        if not index.isValid():
            return
        path = Path(self.proxy.index(index.row(), 0).data(ROLE_PATH))
        entry = self._entries[str(path)]
        menu = QMenu(self)
        containing = menu.addAction('Open Containing Folder')
        workspace_action = (menu.addAction('Open saved image + annotations')
                            if workspace_manifest(path) is not None else None)
        menu.addSeparator()
        image_role = menu.addAction('Treat as image')
        mask_role = menu.addAction('Treat as mask')
        choose = confirm = unpair = None
        if entry.role == 'mask':
            choose = menu.addAction('Pair with source image…')
            if entry.source_path:
                confirm = menu.addAction('Confirm source pairing')
            unpair = menu.addAction('Unpair source')
        reset = menu.addAction('Reset role and pairing to filename suggestions')
        action = menu.exec_(view.viewport().mapToGlobal(pos))
        if action is None:
            return
        if action == containing:
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent))):
                self.status_label.setText(f'Could not open folder: {path.parent}')
        elif action == workspace_action:
            self._open_workspace_layers(path)
        elif action == image_role:
            self._save_annotation(str(path), 'image')
        elif action == mask_role:
            self._save_annotation(str(path), 'mask', entry.source_path)
        elif action == choose:
            self._choose_source(str(path))
        elif action == confirm:
            self._save_annotation(str(path), 'mask', entry.source_path)
        elif action == unpair:
            self._save_annotation(str(path), 'mask')
        elif action == reset:
            self._submit('annotation', self._generation, None,
                         lambda: self.index.reset_annotation(path))

    def _open_workspace_layers(self, path):
        try:
            # Resolve/validate all records before adding any layers. Arrays stay lazy.
            records = read_dataset(path, include_annotations=True)
            image_index = 0
            for data, metadata, kind in records:
                if kind == 'image':
                    metadata = dict(metadata)
                    metadata['metadata'] = dict(metadata.get('metadata', {}))
                    metadata['metadata']['browser_source'] = {'path': str(Path(path).resolve()),
                                                              'image_index': image_index}
                    image_index += 1
                layer = getattr(self.viewer, 'add_' + kind)(data, **metadata)
                if kind == 'labels':
                    layer.editable = False
            self.status_label.setText(f'Opened {len(records)} saved layers. Zarr labels are read-only.')
        except Exception as error:
            self.status_label.setText(f'Workspace could not be fully opened: {error}')

    def _use_current_layer_folder(self):
        active = getattr(getattr(self.viewer.layers, 'selection', None), 'active', None)
        source = getattr(getattr(active, 'source', None), 'path', None)
        if source and Path(source).parent.is_dir():
            self.path_edit.setText(str(Path(source).parent))
            self._change_root()
        else:
            self.status_label.setText('The active layer does not have a file path.')


def _format_size(size):
    value = float(size)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if value < 1024 or unit == 'TB':
            return f'{int(value)} B' if unit == 'B' else f'{value:.1f} {unit}'
        value /= 1024
