"""Image discovery, SQLite metadata, and atomic thumbnail caching."""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import numpy as np
from PIL import Image, ImageOps, PngImagePlugin
import tifffile
from ._masks import label_preview, pair_masks
from ._reader import (workspace_manifest, workspace_data_folder, primary_image,
                      storage_path, zarr_arrays, is_ome_zarr)

TIFF_SUFFIXES = {'.tif', '.tiff', '.btf', '.tf8', '.lsm'}
IMAGE_SUFFIXES = TIFF_SUFFIXES | {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
MICROSCOPY_SUFFIXES = {'.nd2', '.czi', '.lif', '.mrc', '.mrcs', '.stk', '.ims'}
IMAGE_SUFFIXES |= MICROSCOPY_SUFFIXES
DEFAULT_ROOT = Path.home()


def migrate_index(destination, candidates):
    """Copy through SQLite backup (including WAL), leaving the original intact."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    for candidate in candidates:
        if Path(candidate).is_file() and Path(candidate) != destination:
            temporary = destination.with_suffix('.migrating')
            try:
                with sqlite3.connect(candidate) as source, sqlite3.connect(temporary) as target:
                    source.backup(target)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            return


def is_zarr(path):
    return path.name.lower().endswith(('.zarr', '.ome.zarr')) or (
        path.is_dir() and any((path / name).is_file() for name in ('.zgroup', 'zarr.json')))


def capabilities(path):
    if path.suffix.lower() == '.json':
        return 'Label Assistant workspace · image preview and lazy image I/O · saved annotations available'
    if path.suffix.lower() in MICROSCOPY_SUFFIXES:
        return 'Discovered · metadata/preview unavailable · opening requires an installed napari reader'
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return 'OME-Zarr · bounded preview and lazy read-only I/O via optional zarr'
    return 'Metadata and preview available · opening uses installed napari readers'



@dataclass(frozen=True)
class ImageEntry:
    path: Path
    size_bytes: int
    modified_ns: int
    shape: str = ''
    dtype: str = ''
    role: str = 'image'
    source_path: str = ''
    pairing: str = ''

    @property
    def signature(self):
        return str(self.path), self.size_bytes, self.modified_ns


def scan_images(folder: Path, recursive: bool):
    manifest = workspace_manifest(folder)
    if manifest is not None:
        stat = manifest.stat()
        return [ImageEntry(manifest, stat.st_size, stat.st_mtime_ns)], []
    if not folder.is_dir():
        raise ValueError(f'Folder does not exist: {folder}')
    entries, errors = [], []
    if is_zarr(folder):
        stat = folder.stat()
        return [ImageEntry(folder, 0, stat.st_mtime_ns)], []

    for root, dirs, names in os.walk(folder, onerror=lambda e: errors.append(str(e))):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.')
                         and not (Path(root) / d).is_symlink())
        managed = set()
        for name in names:
            candidate = Path(root) / name
            if candidate.suffix.lower() != '.json' or candidate.is_symlink():
                continue
            manifest = workspace_manifest(candidate)
            if manifest is None:
                continue
            try:
                stat = manifest.stat()
                entries.append(ImageEntry(manifest, stat.st_size, stat.st_mtime_ns))
                companion = workspace_data_folder(manifest)
                if companion is not None and companion.parent == Path(root):
                    managed.add(companion.name)
            except (OSError, ValueError) as error:
                errors.append(str(error))
        dirs[:] = [d for d in dirs if d not in managed]
        datasets = [d for d in dirs if is_zarr(Path(root) / d)]
        for name in datasets:
            path = Path(root) / name
            try:
                stamp = max([path.stat().st_mtime_ns] + [
                    (path / marker).stat().st_mtime_ns for marker in ('.zattrs', '.zgroup', 'zarr.json')
                    if (path / marker).is_file()])
                entries.append(ImageEntry(path, 0, stamp))
            except OSError as error:
                errors.append(str(error))
        dirs[:] = [d for d in dirs if d not in datasets]
        for name in names:
            path = Path(root) / name
            if path.suffix.lower() not in IMAGE_SUFFIXES or path.is_symlink():
                continue
            try:
                stat = path.stat()
                entries.append(ImageEntry(path, stat.st_size, stat.st_mtime_ns))
            except OSError as error:
                errors.append(str(error))
        if not recursive:
            break
    return sorted(entries, key=lambda e: str(e.path).casefold()), errors


class FolderIndex:
    """Extends the original index schema without losing saved folder entries."""
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS indexed_files (
                folder TEXT NOT NULL, recursive INTEGER NOT NULL, path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL, modified_sort REAL NOT NULL,
                modified_text TEXT NOT NULL, indexed_at REAL NOT NULL,
                PRIMARY KEY(folder, recursive, path))""")
            migrate_libraries = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='libraries'").fetchone() is None
            conn.execute('CREATE TABLE IF NOT EXISTS libraries (folder TEXT PRIMARY KEY)')
            conn.execute('CREATE TABLE IF NOT EXISTS snapshots (folder TEXT, recursive INTEGER, PRIMARY KEY(folder, recursive))')
            conn.execute('INSERT OR IGNORE INTO snapshots SELECT DISTINCT folder, recursive FROM indexed_files')
            if migrate_libraries:
                conn.execute('INSERT OR IGNORE INTO libraries SELECT DISTINCT folder FROM indexed_files')
            conn.execute('''CREATE TABLE IF NOT EXISTS asset_annotations (
                path TEXT PRIMARY KEY, role TEXT NOT NULL CHECK(role IN ('image','mask')),
                source_path TEXT NOT NULL DEFAULT '',
                mode TEXT NOT NULL CHECK(mode IN ('auto','manual')))''')
            columns = {row[1] for row in conn.execute('PRAGMA table_info(indexed_files)')}
            for name, definition in [('modified_ns', 'INTEGER NOT NULL DEFAULT 0'),
                                     ('shape', "TEXT NOT NULL DEFAULT ''"),
                                     ('dtype', "TEXT NOT NULL DEFAULT ''")]:
                if name not in columns:
                    conn.execute(f'ALTER TABLE indexed_files ADD COLUMN {name} {definition}')

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=5)

    def annotate(self, entries):
        with self._connect() as conn:
            annotations = {path: (role, source, mode) for path, role, source, mode in
                           conn.execute('SELECT path, role, source_path, mode FROM asset_annotations')}
            result = pair_masks(entries, annotations)
            conn.executemany("""INSERT INTO asset_annotations VALUES (?, 'mask', ?, 'auto')
                ON CONFLICT(path) DO UPDATE SET source_path=excluded.source_path
                WHERE asset_annotations.mode='auto'""",
                [(str(e.path), e.source_path) for e in result if e.role == 'mask'
                 and (str(e.path) not in annotations or annotations[str(e.path)][2] == 'auto')])
        return result

    def set_annotation(self, path, role, source_path=''):
        if role not in ('image', 'mask'):
            raise ValueError('Role must be image or mask')
        if source_path and (role != 'mask' or Path(source_path) == Path(path)):
            raise ValueError('A mask must be paired with a different source image')
        with self._connect() as conn:
            conn.execute("""INSERT OR REPLACE INTO asset_annotations
                (path, role, source_path, mode) VALUES (?, ?, ?, 'manual')""",
                (str(path), role, str(source_path)))

    def reset_annotation(self, path):
        with self._connect() as conn:
            conn.execute('DELETE FROM asset_annotations WHERE path=?', (str(path),))

    def add_library(self, folder):
        with self._connect() as conn:
            conn.execute('INSERT OR IGNORE INTO libraries VALUES (?)', (str(folder),))

    def libraries(self):
        with self._connect() as conn:
            return [Path(row[0]) for row in conn.execute('SELECT folder FROM libraries ORDER BY folder')]

    def _snapshot(self, conn, folder, recursive):
        key = (str(folder), int(recursive))
        if conn.execute('SELECT 1 FROM snapshots WHERE folder=? AND recursive=?', key).fetchone():
            return key
        for ancestor in (Path(folder), *Path(folder).parents):
            if conn.execute('SELECT 1 FROM snapshots WHERE folder=? AND recursive=1',
                            (str(ancestor),)).fetchone():
                return str(ancestor), 1
        return None

    def has_snapshot(self, folder, recursive):
        with self._connect() as conn:
            return self._snapshot(conn, folder, recursive) is not None

    def load_all(self):
        entries = {}
        for folder in self.libraries():
            for entry in self.load(folder, True):
                entries[str(entry.path)] = entry
        return sorted(entries.values(), key=lambda e: str(e.path).casefold())

    def load(self, folder, recursive):
        with self._connect() as conn:
            scope = self._snapshot(conn, folder, recursive) or (str(folder), int(recursive))
            rows = conn.execute("""SELECT path, size_bytes, modified_ns, shape, dtype
                FROM indexed_files WHERE folder=? AND recursive=? ORDER BY lower(path)""",
                                scope).fetchall()
        result = [ImageEntry(Path(path), size, modified, shape, dtype)
                  for path, size, modified, shape, dtype in rows]
        if scope != (str(folder), int(recursive)):
            result = [e for e in result if (Path(folder) in e.path.parents if recursive
                                           else e.path.parent == Path(folder))]
        return result

    def save(self, folder, recursive, entries):
        import time
        with self._connect() as conn:
            if recursive:
                for saved_folder, saved_recursive in conn.execute('SELECT folder, recursive FROM snapshots').fetchall():
                    if Path(folder) in Path(saved_folder).parents or (saved_folder == str(folder) and not saved_recursive):
                        conn.execute('DELETE FROM snapshots WHERE folder=? AND recursive=?', (saved_folder, saved_recursive))
                        conn.execute('DELETE FROM indexed_files WHERE folder=? AND recursive=?', (saved_folder, saved_recursive))
            # Reconcile the same subtree in saved ancestor libraries as well.
            for ancestor, recurse in conn.execute('SELECT folder, recursive FROM snapshots').fetchall():
                if recurse and Path(ancestor) in Path(folder).parents:
                    paths = [row[0] for row in conn.execute(
                        'SELECT path FROM indexed_files WHERE folder=? AND recursive=1', (ancestor,))]
                    removed = [path for path in paths if (
                        Path(folder) in Path(path).parents if recursive else Path(path).parent == Path(folder))]
                    conn.executemany('DELETE FROM indexed_files WHERE folder=? AND recursive=1 AND path=?',
                                     [(ancestor, path) for path in removed])
                    conn.executemany('''INSERT OR REPLACE INTO indexed_files
                        (folder,recursive,path,size_bytes,modified_sort,modified_text,indexed_at,modified_ns,shape,dtype)
                        VALUES (?,?,?,?,?,?,?,?,?,?)''',
                        [(ancestor, 1, str(e.path), e.size_bytes, e.modified_ns / 1e9,
                          '', time.time(), e.modified_ns, e.shape, e.dtype) for e in entries])
            conn.execute('INSERT OR IGNORE INTO snapshots VALUES (?,?)', (str(folder), int(recursive)))
            conn.execute('DELETE FROM indexed_files WHERE folder=? AND recursive=?',
                         (str(folder), int(recursive)))
            conn.executemany("""INSERT INTO indexed_files
                (folder,recursive,path,size_bytes,modified_sort,modified_text,indexed_at,
                 modified_ns,shape,dtype) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                [(str(folder), int(recursive), str(e.path), e.size_bytes,
                  e.modified_ns / 1e9, '', time.time(), e.modified_ns, e.shape, e.dtype)
                 for e in entries])

    def update_metadata(self, entry):
        with self._connect() as conn:
            conn.execute("""UPDATE indexed_files SET shape=?, dtype=?
                WHERE path=? AND size_bytes=? AND modified_ns=?""",
                         (entry.shape, entry.dtype, str(entry.path),
                          entry.size_bytes, entry.modified_ns))


def _normalize(data):
    data = np.asarray(data)
    if np.iscomplexobj(data):
        data = np.abs(data)
    if data.dtype == np.uint8:
        return data
    finite = data[np.isfinite(data)]
    if not finite.size:
        return np.zeros(data.shape, dtype=np.uint8)
    low, high = np.percentile(finite, (1, 99))
    if high <= low:
        low, high = finite.min(), finite.max()
    scaled = (data.astype(np.float32) - low) * (255 / max(float(high - low), 1e-12))
    return np.clip(np.nan_to_num(scaled), 0, 255).astype(np.uint8)


def build_preview(entry: ImageEntry, cache: Path):
    dependency = ''
    if entry.path.suffix.lower() == '.json':
        manifest, record = primary_image(entry.path)
        storage = record.get('storage') or {}
        source = storage_path(manifest, storage)
        keys = [storage.get('array_path') or ''] + list(storage.get('datasets') or [])
        markers = [source / marker for marker in ('.zattrs', '.zgroup', 'zarr.json')]
        markers += [source / key / marker for key in keys for marker in ('.zarray', 'zarr.json')]
        dependency = repr([(str(path), path.stat().st_mtime_ns, path.stat().st_size)
                           for path in markers if path.is_file()])
    key = hashlib.sha256(f'v4:{entry.role}:{entry.signature}:{dependency}'.encode()).hexdigest()
    target = cache / f'{key}.png'
    if target.exists():
        try:
            with Image.open(target) as image:
                image.load()
                return target, replace(entry, shape=image.info['shape'], dtype=image.info['dtype'])
        except (OSError, KeyError):
            pass
    cache.mkdir(parents=True, exist_ok=True)
    if entry.path.suffix.lower() in MICROSCOPY_SUFFIXES:
        raise ValueError('Discovery only: metadata and preview adapter unavailable; use an installed napari reader to open')
    if entry.path.suffix.lower() == '.json':
        manifest, record = primary_image(entry.path)
        storage = record.get('storage') or {}
        if storage.get('kind') != 'zarr':
            raise ValueError('Workspace image preview currently requires Zarr storage')
        image, shape, dtype = _zarr_preview(storage_path(manifest, storage),
                                           storage=storage, rgb=record.get('rgb', False))
    elif is_zarr(entry.path):
        image, shape, dtype = _zarr_preview(entry.path, labels=entry.role == 'mask')
    elif entry.path.suffix.lower() in TIFF_SUFFIXES:
        with tifffile.TiffFile(entry.path) as tif:
            base = tif.series[0]
            shape, dtype = ' x '.join(map(str, base.shape)), str(base.dtype)
            series = base.levels[-1]
            page = series.pages[len(series.pages) // 2]
            data, axes = page.asarray(), page.axes
        data = data[tuple(slice(None) if axis in 'YXS' else dim // 2
                          for axis, dim in zip(axes, data.shape))]
        if ''.join(a for a in axes if a in 'YXS').startswith('S'):
            data = np.moveaxis(data, 0, -1)
        step = max(1, math.ceil(max(data.shape[:2]) / 1024))
        plane = data[::step, ::step]
        image = label_preview(plane) if entry.role == 'mask' else Image.fromarray(_normalize(plane))
    else:
        with Image.open(entry.path) as source:
            shape = f'{source.height} x {source.width}'
            if len(source.getbands()) > 1:
                shape += f' x {len(source.getbands())}'
            if entry.role == 'mask':
                # Preserve palette indices and label IDs. No EXIF rotation of labels.
                plane = np.asarray(source)
                dtype = str(plane.dtype)
                step = max(1, math.ceil(max(plane.shape[:2]) / 1024))
                image = label_preview(plane[::step, ::step])
            else:
                dtype = str(np.asarray(source).dtype)
                source.thumbnail((256, 256))
                image = ImageOps.exif_transpose(source).convert('RGB')
    image.thumbnail((256, 256), resample=Image.Resampling.NEAREST if entry.role == 'mask'
                    else Image.Resampling.LANCZOS)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text('shape', shape)
    metadata.add_text('dtype', dtype)
    fd, temporary = tempfile.mkstemp(dir=cache, suffix='.png')
    os.close(fd)
    try:
        image.convert('RGBA' if entry.role == 'mask' else 'RGB').save(temporary, pnginfo=metadata)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target, replace(entry, shape=shape, dtype=dtype)


def _zarr_preview(path, labels=False, storage=None, rgb=False):
    """Read one bounded representative plane, handling both NGFF metadata layouts."""
    arrays, axes, metadata = zarr_arrays(path, storage)
    array = arrays[-1]
    if not axes and storage is not None:
        axes = ([f"d{i}" for i in range(array.ndim - 3)] + ["y", "x", "s"]
                if rgb else [f"d{i}" for i in range(array.ndim - 2)] + ["y", "x"])
    if len(axes) != array.ndim or "y" not in axes or "x" not in axes:
        raise ValueError("Preview needs explicit x/y axes")
    if math.prod(array.chunks) * array.dtype.itemsize > 64 * 1024 * 1024:
        raise ValueError("Preview chunk exceeds 64 MiB decode budget")
    selection = tuple(slice(None, None, max(1, math.ceil(n / 512))) if a in ("x", "y")
                      else slice(None, None, 1) if a == "s" and rgb else n // 2
                      for a, n in zip(axes, array.shape))
    touched = math.prod(min(math.ceil(n / c), len(range(0, n, sel.step)))
                        if isinstance(sel, slice) else 1
                        for n, c, sel in zip(array.shape, array.chunks, selection))
    if touched * math.prod(array.chunks) * array.dtype.itemsize > 128 * 1024 * 1024:
        raise ValueError("Preview exceeds 128 MiB decode budget; add a coarser pyramid level")
    data = np.asarray(array[selection])
    kept = [a for a, sel in zip(axes, selection) if isinstance(sel, slice)]
    order = [kept.index("y"), kept.index("x")]
    if rgb and "s" in kept:
        order.append(kept.index("s"))
    data = data.transpose(order)
    return (label_preview(data) if labels else Image.fromarray(_normalize(data))), " x ".join(map(str, arrays[0].shape)), str(arrays[0].dtype)
