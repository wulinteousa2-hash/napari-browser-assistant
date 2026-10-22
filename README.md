# napari-browser-assistant

**Latest release: 1.0.1** · [Changelog](CHANGELOG.md) · [MIT license](LICENSE)

An image library inside napari with three connected tabs: **Library** for browsing
and filename/path search, **Discover** for saved studies and region similarity
search, and **Metadata** for local OCR extraction and review.

**Version 1.0.1** adds Discover, reviewed dataset export, and Tesseract OCR saved
in the active study. It also supports read-only mapped crops from large compatible
TIFFs and remembers separate checkpoint paths for ResNet18 and EM-DINO.
See the [Discover guide](docs/DISCOVER.md) and the Metadata workflow below.

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
covers discovery, pairing, browsing, and opening. Discover provides separate region-review decisions, specimen-group splits, and
crop/pair export manifests. Pixel-mask training approvals and gallery overlay
previews are not implemented. Editing/saving Labels uses napari; the browser itself never
rewrites your source images or masks.

## Discover studies

Open **Discover** beside **Library**, create a study, add the Library's image
entries, and prepare region embeddings. Draw a query rectangle in napari, find
similar regions, review the results, and save explicit dataset selections.

The built-in texture baseline works without model downloads. Optional local
ResNet18 and EM-DINO encoders provide learned features. Biological classes are
separate from relevance reviews and require human confirmation for classification
export. This workflow exports trainer-neutral crops and reviewed pairs; it does
not train a model automatically.

Studies autosave as versioned JSON and reopen with searches, annotations, and
selections. Vectors are rebuildable cache data. See the
[full workflow and study format](docs/DISCOVER.md).

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

### Metadata: local OCR of printed acquisition information

Open or create a study in **Discover**, add Library images, then open the
**Metadata** tab. It shares that study and saves OCR records in its existing
`.browser-analysis.json` document. Original images are never modified. File
metadata is read-only and displayed separately from printed acquisition fields;
printed calibration never replaces file-derived pixel calibration.

Install **Tesseract** and its English language data separately and put the
`tesseract` executable on PATH. On Ubuntu/Debian use
`sudo apt install tesseract-ocr`; on macOS use `brew install tesseract`.
On Windows, install Tesseract with English data and add its installation folder
to PATH. The app uses only this local executable, with no cloud service or
automatic model downloads. Missing installations produce guidance in the tab.

1. Select an image and use **Preview image** to see its saved representative
   plane. File metadata includes the source path, shape, axes, revision and any
   available file calibration.
2. Choose **Bottom 25% (footer)** or **Full image**, then **Extract selected** or
   **Extract all**. Progress and cancellation appear in Metadata and Discover.
   Discover study operations are disabled during OCR. Cancelling stops the current
   executable and saves already completed images; individual image failures are
   reported while the remaining batch continues.
3. Review original OCR text, word confidence/bounds and provenance. Correct the
   text and the printed acquisition fields (a JSON object with text values),
   check **Reviewed by user** when verified, and select **Save corrections to
   study**. Parsing recognizes calibration, scale bars, voltage, magnification,
   camera, exposure, acquisition date/time and TEM mode when labels/units are
   recognizable. All parsed values start unverified; unrecognized text remains
   in the original output. Editing text does not automatically replace fields.
4. Unchanged images with the same scope/settings and engine version are cached.
   **Re-extract explicitly** adds another generation. Use the generation selector
   to revisit previous OCR and corrections; reviewed corrections are retained.

Unsaved correction drafts survive image, generation and study navigation in the
current widget. Save them before closing the widget/application; drafts are not
persisted across restarts. Saves use Discover's atomic persistence and reject
external changes to the study file. If an OCR save encounters a disk conflict,
its completed result is retained in memory as `discover._unsaved_ocr_result` for
recovery; reopen the on-disk study before attempting further changes.

OCR uses the saved representative plane, English language data
and sparse-text segmentation. Word boxes are `[x, y, width, height]` in source
level-0 coordinates, including the footer offset. Existing Discover read limits
apply (4 million samples per region and 128 MiB decode/chunk budget).
**Resize large OCR regions to the memory budget** is enabled by default. Large
regions are read in bounded tiles and resized into an OCR image within the sample
budget; original files are unchanged. This can reduce small-text detail, so prefer
footer mode for footer text. Word boxes are mapped back to original source
coordinates, and the input dimensions and scale are recorded. Uncheck this option
to require native-resolution OCR. Preview also uses bounded resizing. Uncompressed, memory-mappable TIFFs use read-only mapped crops. Sources that
exceed the decode budget (including oversized individual chunks) still fail safely;
convert those to a tiled pyramid/OME-Zarr copy before adding that copy to the study. Re-add changed source images through Discover before extracting
again. Source revision caching uses the existing metadata/stat fingerprint, not
a full pixel-content hash (changes inside Zarr chunks may not invalidate it).
OCR quality and field parsing depend on image legibility and printed formatting;
review every scientific value. Discover embedding crops are unchanged.


### Discover preparation and model behavior

**Max regions** is the total crop budget across the study. It must be at least
the number of study images, with a current maximum of 10,000. For example, a study
with 544 images needs at least 544 regions; choose 1024 or more for broader
coverage. Tile size is 16–2048 pixels at the selected pyramid level. Level 0
uses original resolution; higher levels require an existing source pyramid.
Preparation can succeed partially: unreadable images are reported and skipped.

After preparation, open an image, draw a query rectangle, and click **Find similar
regions**. Results appear as thumbnails below **Saved search**. Click a thumbnail
to open and outline its source region. Searching uses the encoder/checkpoint of
the selected prepared run, rather than a newly selected encoder awaiting preparation.

EM-DINO requires `omniem` and the separately downloaded compatible
`backbone_emdino_v1.pt`. A ResNet18 checkpoint cannot be used for EM-DINO.
Place the backbone under `models/emdino/` in Browser Assistant's persistent data
directory for automatic detection, or use **Choose…**. Checkpoint paths are saved
separately for each encoder. Denoising, segmentation and super-resolution heads
are not used by Discover.

Discover runs inference with fixed weights. Preparation, relevance reviews and
class assignments do not train or fine-tune the model. Dataset exports provide
crops and reviewed pairs for an external training pipeline; a compatible changed
backbone requires a new preparation run. Retrieval scores describe appearance
similarity and are not biological identities or calibrated probabilities.
