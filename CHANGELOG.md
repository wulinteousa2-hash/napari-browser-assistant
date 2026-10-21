# Changelog

## [1.0.0] — 2026-10-21

First GitHub release of napari-browser-assistant.

### Added

- Image gallery and metadata table with shared selection, folder navigation,
  configurable thumbnail size, and opening selected files in napari.
- Compiled, case-insensitive filename/path search with multiple terms, wildcards,
  quoted phrases, exclusions, OR groups, and extension lists.
- Persistent SQLite indexing and settings in the application-data directory,
  separate thumbnail caching, and migration from earlier cache-based indexes.
- Background indexing and cached navigation, multiple library folders, an
  All libraries view, and retained entries for offline roots.
- Lazy table cells, batched gallery layout, and viewport-prioritized background
  thumbnail generation with atomic cache writes.
- Dedicated Images and Masks views; recognition of trailing `_mask` and `_m`;
  source-image suggestions; ambiguity indicators; and saved role/pair overrides.
- Categorical mask previews with transparent background label 0,
  nearest-neighbor resizing, Image/Labels opening, and shape-mismatch warnings.
- Microscopy discovery for ND2, CZI, LSM, LIF, MRC/MRCS, STK, and IMS, with
  explicit format capability indicators.
- A bundled lazy, read-only OME-Zarr reader supporting NGFF 0.4 and 0.5 metadata,
  multiscale image pyramids, and scale/translation transforms.
- Label Assistant workspace JSON image previews, JSON opening, companion-folder
  resolution, and optional opening of saved Labels, Shapes, and Points layers.
- Dataset discovery that prunes Zarr chunks and workspace companion trees.
- Bounded representative-plane Zarr previews and rejection of oversized decodes.
- An offscreen test suite covering discovery, migrations, search, mask pairing,
  caching, offline libraries, reader integration, and refresh races.

### Scope and limitations

- ND2/CZI and several other microscopy formats are discoverable but do not yet
  have preview adapters; opening depends on installed napari readers.
- Search/filtering and sorting still run over the loaded snapshot on the GUI
  thread; this is not a paged database search engine.
- Workspace I/O supports version 1 Zarr-backed Image/Labels and inline
  Shapes/Points records. Training state and the editable Label Assistant
  workspace-save lifecycle are not restored. Zarr labels open read-only.
- Training review states, dataset splits, and export manifests are not included.

### Validation

- 55 tests passed with Qt offscreen before release.
- Verified gallery preview and lazy six-level image opening using a real
  49,976 × 59,716 Label Assistant OME-Zarr workspace.

[1.0.0]: https://github.com/wulinteousa2-hash/napari-browser-assistant/tree/v1.0.0
