# napari-browser-assistant

**Version 1.0.0** · [Changelog](CHANGELOG.md) · [MIT license](LICENSE)

An image library inside napari, combining a gallery, metadata table, persistent
filename/path search, and visible-first cached thumbnails.

## Install and launch

In your napari environment, clone this repository and install the plugin:

```sh
git clone https://github.com/wulinteousa2-hash/napari-browser-assistant.git
cd napari-browser-assistant
python -m pip install -e .
napari
```

A working Qt backend is required by napari. For OME-Zarr v3 and Label Assistant
workspace I/O, use Python 3.11 or newer and install the optional dependencies below.

Open **Plugins → Browser Assistant**, then **Add folder…**. First launch does not
index your home directory. To enable OME-Zarr previews and I/O:

```sh
python -m pip install -e '.[ome-zarr]'
```

From a source checkout you can also run
`PYTHONPATH=src python -m napari_browser_assistant`.

## Libraries and search

- Add image folders using **Add folder…** or the path field. The library picker
  switches between saved folders and **All libraries** (recursive, deduplicated).
- Search saved filename/full-path entries without rescanning or decoding images.
  Queries are case-insensitive and compiled once per filter update.
- Multiple terms mean AND: `sample cell`.
- Wildcards match filenames: `*.tif`, `cell?.png`. Patterns containing a slash
  match the full path: `*/experiment/*.tif`.
- Exclude terms with `!`: `*.tif !mask`.
- Quote phrases: `"cell culture" !"bad sample"`.
- Combine AND groups with OR: `*.tif !mask | *.nd2`.
- Match extensions: `ext:tif;tiff`. Parentheses, regex, and the full Everything
  query language are not implemented.
- **Refresh** reloads the saved snapshot. New folders are indexed in the background.
  Folder navigation reuses an existing recursive snapshot.
- **Refresh Index** explicitly rescans the selected folder, or every library in
  All libraries. Use it to discover added/deleted/changed files.
- Disconnected roots retain their saved entries and display **Offline**. Reconnect
  and refresh to restore previews/opening. Partial scan errors retain saved entries
  rather than deleting inaccessible files; retry Refresh Index after resolving them.

Switch between **Gallery** and **Table**, adjust thumbnail size, and Ctrl/Shift-select
images. Selection is shared. Double-click or **Open Selected** delegates to
`viewer.open(path)`, including dataset directories. Reader errors appear in the
status area. Right-click to open the containing folder.

## OME-Zarr and Label Assistant JSON I/O

The plugin now includes a napari reader for local OME-Zarr stores and
`napari-label-assistant-workspace` version 1 JSON manifests. A workspace JSON is
displayed as an **image entry with an image thumbnail**, rather than JSON text.

- Click **Open JSON…** to choose a workspace and open its image. You can also paste
  its path into the library field, or add the containing folder and double-click
  the entry. Use **Refresh Index** once for previously indexed folders.
- A conventional `name_label_assistant_data` folder resolves to its sibling
  `name.label-assistant.json`. The JSON and companion data are presented as one
  dataset; discovery skips the companion tree and all Zarr chunks. Unrelated JSON
  files are ignored.
- Use the **Datasets / workspaces** format filter to find these entries. The entry
  remains in **Images**, with its first Image layer used for the gallery thumbnail.
- Default opening loads all saved **Image** layers as lazy Zarr arrays/pyramids.
  Right-click **Open saved image + annotations** to also load saved **Labels**,
  **Shapes**, and **Points**, with their names, visibility, and spatial transforms.
- Saved Labels storage is always opened read-only, even if the JSON requests
  `r+`; the annotation action disables label editing. Training state, model files,
  viewer position, and Label Assistant's workspace-save lifecycle are not restored.
  Use Label Assistant's workspace tools for a full editable training session.
- Standalone OME-Zarr directories also open through the bundled reader. The reader
  supports the first NGFF multiscale image group, pyramid levels, scale/translation
  transforms, and both root-level `multiscales` (NGFF 0.4) and
  `ome.multiscales` (NGFF 0.5).
- Zarr v3 datasets require Zarr 3 (the extra installs it on Python 3.11+). The JSON
  adapter currently supports Zarr-backed Image/Labels plus inline Shapes/Points;
  file-backed workspace layers produce an explicit unsupported-storage error.
  Only local stores are accepted. JSON metadata is limited to 64 MiB.

Image data stays lazy when opened. Thumbnail generation reads a bounded plane
from the smallest pyramid level. Missing referenced stores produce an opening or
preview error; the manifest entry is retained. Image metadata and thumbnail
creation are independent of opening the full dataset.

Restart napari after updating the plugin so the new reader contribution is
registered. Napari's file-opening workflow can also select the Browser Assistant
reader for these datasets.

## Masks and image–mask pairs

Use the **Masks** tab for a dedicated gallery/table of label files. **Images**
excludes masks; **All files** shows both. These tabs use the saved index and keep
the existing search, format filters, thumbnail controls, and shared selection.

- A trailing `_mask` or `_m` suggests a mask: `cell_mask.tif`, `cell_m.png`,
  or `cell_m.ome.tif`. Matching is case-insensitive; names such as
  `cell_more.tif` are ordinary images.
- Pairing suggests a source with the same base name in the same folder. Extensions
  can differ. Two possible sources produce **Ambiguous source**, never an arbitrary
  match. Both `cell_m.tif` and `cell_mask.tif` can refer to `cell.tif`.
- Filter the Masks view by **Needs pairing**, **Suggested pair**, or **Confirmed
  pair**. The table and tooltips show the source path and pairing state.
- Select one mask and click **Pair source…** to select or correct its source,
  including images in another folder. Right-click to **Confirm source pairing**,
  **Unpair source**, **Treat as image**, or **Treat as mask**. Reset returns the
  file to filename-based suggestions.
- Roles and manual pairing decisions are saved in SQLite and survive rescans and
  restarts. Suggested matches remain distinguishable from confirmed relationships.
- Masks preview categorical label IDs in stable colors, with background `0`
  transparent. Previews use nearest-neighbor resizing, without contrast stretching
  labels or creating interpolated IDs. Integer and boolean label planes are
  supported; RGB masks, negative IDs, and floating-point probability maps report
  a preview error rather than being silently converted.
- Double-click a mask or choose **Open Selected** to open its linked source as a
  napari **Image** layer and the mask as **Labels**. Ordinary image-file masks give an editable overlay
  in napari; Zarr label stores open read-only. Multiple selected masks sharing a source open that source once per
  action. Unpaired masks open as Labels alone.
- Opening compares reader-provided shapes and warns about mismatches or extra
  axes. Matching shapes and a confirmed filename relationship do not validate
  pixel registration, physical transforms, or annotation quality for training.

Semantic class IDs and instance IDs both receive categorical previews; the browser
does not infer which annotation convention you use. This initial mask workflow
covers discovery, pairing, browsing, and opening. Training approval status,
train/validation/test splits, export manifests, and gallery overlay previews are
not implemented yet. Editing/saving Labels uses napari; the browser itself never
rewrites your source images or masks.

## Format capabilities

Discovery, metadata/preview, and opening are separate capabilities. The table's
capability column and item tooltips describe these limits.

| Format | Discovery | Metadata and thumbnails | Opening |
| --- | --- | --- | --- |
| TIFF, OME-TIFF, BTF, TF8, LSM | Yes | tifffile, representative page/smallest pyramid level | Installed napari readers |
| JPEG, PNG, BMP, GIF, WebP | Yes | Pillow | Installed napari readers |
| ND2, CZI, LIF, MRC/MRCS, STK, IMS | Yes | Not implemented; discovery only | Requires an appropriate installed napari reader |
| OME-Zarr | One entry per store | Optional zarr; NGFF 0.4/0.5, bounded representative plane | Bundled lazy reader |
| Label Assistant workspace JSON | One entry per workspace | Thumbnail from first Zarr-backed image | Bundled image reader; optional saved annotations |

Zarr directories (including metadata-marked groups without a .zarr suffix) are
pruned from discovery: individual arrays/chunks never become image entries.
Previews select the last declared pyramid level, central nonspatial coordinates,
and at most 512 samples per spatial dimension. Chunks exceeding 64 MiB or estimated
total decoding above 128 MiB are rejected before reading. Unsupported stores retain
their library entry and report a preview error. Displayed Zarr shape/dtype describe
the highest-resolution level; store size is not recursively calculated.

TIFFs use a representative middle page. A large non-pyramidal TIFF may still require
decoding a full page. High-bit-depth previews use percentile contrast. Source
images and datasets are opened read-only by the preview code and never modified.

## Storage and responsiveness

The SQLite index, saved library list, and settings are persistent:
Qt **AppLocalDataLocation/napari-browser-assistant/** contains
`folder_index.sqlite3` and `settings.ini`. Users select image folders, not database
locations. Rebuildable thumbnails live separately in
Qt **CacheLocation/napari-browser-assistant/thumbnails/**.

On first use, a legacy SQLite database from either the previous plugin cache
directory or the older cache root is copied with SQLite backup (including WAL)
and schema-migrated. The original is retained. Existing QSettings preferences are
copied into the persistent settings file.

Index reads/scans run on a serialized background worker. Up to two preview workers
prioritize visible entries. Metadata persistence also runs off the GUI thread.
The table uses lazy cells instead of allocating Qt objects for every cell; gallery
layout is batched and visibility checks are bounded by the viewport.

Search/filtering and proxy sorting still run on the GUI thread over the loaded
snapshot; this is not yet a paged, million-file database search engine.
Thumbnail files are written atomically. Cache keys include path, size, modification
time, and preview version. Zarr cache invalidation uses store/metadata timestamps;
editing chunks without updating metadata may require clearing the thumbnail cache.

## Validation

Install the test and dataset extras in your napari environment:

```sh
python -m pip install -e '.[test,ome-zarr]'
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

Tests cover search operators, SQLite schema and storage migration, empty snapshots,
multiple/offline libraries, cached navigation, snapshot reconciliation, source
preservation, Zarr discovery/decode limits, lazy models, shared selection, visible
previews beyond 200 entries, refresh races, mask suffix recognition, persistent role/pair overrides,
label-color preservation, and real napari Image/Labels opening.

Reader tests cover NGFF 0.4/0.5, JSON/companion discovery, lazy arrays, read-only
annotations, missing references, internal array-path validation, and registered napari I/O.
