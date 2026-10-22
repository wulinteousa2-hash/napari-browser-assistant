# Changelog

## [1.0.1] — 2026-10-22

### Added

- Discover tab inside Browser Assistant with automatically saved/reopened studies.
- Bounded TIFF/image/OME-Zarr/workspace region reads using level-0 coordinates
  and explicit pyramid/plane settings.
- Texture, local ResNet18, and optional EM-DINO encoder adapters, checkpoint
  fingerprints, cached vectors, and exact cosine retrieval.
- Napari query rectangles, result galleries, search history, source highlighting,
  and saved-query restoration.
- Independent relevance reviews, proposed/confirmed biological region classes,
  dataset selections, and specimen/group split assignments.
- Immutable native-valued crop and reviewed query-relevance pair exports,
  retaining source and model provenance.
- Cancellation, atomic saves, writer-conflict detection, cache-loss recovery,
  source-revision checks, and checkpoint-symlink regression coverage.
- Discover usage and study-format documentation in docs/DISCOVER.md.
- Metadata tab sharing the active Discover study, with file metadata, bounded
  preview, local Tesseract OCR, original output, editable corrections and fields,
  review status, extraction history, word confidence and source-coordinate boxes.
- Footer/full-image OCR, acquisition-field parsing, revision/settings caching,
  explicit re-extraction, background progress, cancellation preserving completed
  results, and per-image errors. Original files and file calibration stay intact.
- Unsaved correction drafts retained across image/study navigation; explicit
  correction saves use study conflict detection and atomic persistence.

### Fixed

- Large OCR regions can be assembled from bounded tiles and resized within the
  sample budget, preserving word boxes in original source coordinates.
- Compatible uncompressed TIFFs use read-only memory mapping for bounded crops,
  avoiding full-strip decoding while retaining compressed TIFF/Zarr decode limits.
- Encoder switching remembers separate checkpoint paths and auto-detects a local
  EM-DINO backbone, avoiding accidental reuse of ResNet18 weights.

### Scope and limitations

- Discover uses fixed-weight inference; reviews do not automatically train models.
- Preparation currently requires at least one crop per image and supports a total
  budget of at most 10,000 regions. Source pyramids must already exist.
- OCR uses local English Tesseract and a representative plane; printed values
  remain unverified until reviewed. Unsaved drafts do not survive application exit.
- Checkpoints and OCR language data must be installed separately; no automatic
  model downloads or cloud OCR are included.

### Validation

- Offscreen tests exercise the real napari ROI/review/export workflow.
- Real EM smoke test: six-level 49,976 × 59,716 workspace, pretrained ResNet18,
  512-dimensional vectors, saved review, and source-valued level-5 crop export.
- Real EM-DINO backbone loaded and exercised on CPU: a 224 × 224 region produced
  a normalized 1024-dimensional embedding.
- Real Tesseract smoke test and deterministic OCR/parser/cache/cancellation/UI tests.
- All 86 sources in a mixed JPEG/TIFF study passed read-only 512 × 512 crop checks.
- 93 tests passed with Qt offscreen before release.


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

[1.0.1]: https://github.com/wulinteousa2-hash/napari-browser-assistant/blob/main/CHANGELOG.md
[1.0.0]: https://github.com/wulinteousa2-hash/napari-browser-assistant/tree/v1.0.0
