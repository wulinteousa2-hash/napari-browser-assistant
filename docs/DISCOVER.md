# Discover: saved studies and region retrieval

Discover is a tab in **napari-browser-assistant**, beside **Library**. It uses the
same indexed image collection and napari canvas. Similarity scores describe
appearance; they are not biological identities or calibrated probabilities.

## Install

The built-in texture baseline needs no neural-model dependency. For bounded
access to large TIFFs and OME-Zarr:

~~~sh
python -m pip install -e '.[discover]'
~~~

For local ResNet18 checkpoints:

~~~sh
python -m pip install -e '.[discover,encoders]'
~~~

For EM-DINO, use Python 3.10+ and install the optional package:

~~~sh
python -m pip install -e '.[discover,em-dino]'
~~~

Choose a local checkpoint; model weights are never downloaded automatically.
Checkpoint paths are remembered separately for each encoder. Switching from
ResNet18 to EM-DINO does not reuse the ResNet18 weights. EM-DINO also detects
`models/emdino/backbone_emdino_v1.pt` inside Browser Assistant's persistent data
directory (the parent directory of its `settings.ini`). Only the backbone is
used for similarity search; denoising, segmentation and super-resolution heads
are not used by Discover.
ResNet18 suggests a checkpoint if the standard `smp-hub/resnet18.imagenet`
Hugging Face cache already exists.

The EM-DINO adapter uses `omniem.EMEncoder.load("emdinov1", checkpoint)` and CLS
features. Obtain compatible weights through [EM-SSL](https://github.com/pku-maleilab/EM-SSL-project)
and consult [omniem's API guide](https://github.com/pku-maleilab/omniem-package/blob/main/docs/api.md).
Its API contract is tested, and a real EM-DINO backbone was validated with a
224×224 CPU inference producing a normalized 1024-dimensional embedding. Real
pretrained extraction/retrieval was also validated with the local ResNet18
ImageNet checkpoint. The texture baseline is a deterministic descriptor,
not a learned biological model.

## First study

1. In **Library**, choose/filter the image collection to investigate.
2. Open **Discover → New study**.
3. Click **Add Library images to study**. This adds saved image entries in the
   current Library filter, excluding masks, without rescanning image folders.
4. Choose the encoder, local checkpoint if required, CPU/CUDA, tile size,
   pyramid level, and maximum region count.
5. Click **Prepare study embeddings**. Progress and cancellation are available.
   The bounded grid sample is distributed across each image, rather than taking
   only the first tiles. The status reports the sample and full grid counts.
6. Open an image in Library and select its Image layer in napari.
7. Click **Draw query region**, then draw one rectangle on the canvas.
8. Click **Find similar regions**. The query is embedded and compared with saved
   vectors using exact cosine similarity. Indexed tiles are not re-encoded.
9. Click a result to open its source, outline its region, and zoom to it.
   **Show saved query** returns to the original query.
10. Select result thumbnails and review **Relevant**, **Not relevant**, or
    **Uncertain**. Separately choose **Include in dataset** for explicit selections.

Relevance is specific to a query. Including a crop does not assign a biological
class. Neither decision creates a pixel segmentation mask. Overlapping regions
on the same source plane are excluded from query results to avoid trivial matches.

## Large images and scale

Uncompressed, memory-mappable TIFFs are opened read-only so Discover can copy a
small crop even when the file stores the entire image in a strip larger than
128 MiB. Compressed/tiled TIFF and Zarr reads retain the 128 MiB chunk decode
budget; an oversized compressed chunk still requires a coarser existing pyramid
level or a tiled copy. Source files are never rewritten by region reading.

Choose a pyramid level that preserves the structures you want to investigate.
Level 5 in a six-level EM pyramid may be far cheaper than level 0. **Tile size**
is in pixels at the analysis level; saved ROI bounds remain in level-0 pixels.

A study uses one representative nonspatial plane per source, initially the middle
of each nonspatial axis. A query records its actual slice. This is not exhaustive
3D-volume indexing. File-backed sources and axis-aligned XY rectangles are required.

Source records include axes, full shape, pyramid shapes, dtype, and available
pixel scale/units. Missing calibration stays unknown. Viewer transforms are saved
when capturing a query and restored when reopening its source. Images are not
automatically resampled to a common physical field of view: compare magnification,
physical scale, preparation, and acquisition batches deliberately.

Encoders use a recorded grayscale/percentile/resizing preprocessing policy.
ResNet18 adds ImageNet normalization; EM-DINO uses its upstream normalization.
This can suppress absolute-intensity information. Evaluate whether retrieval
matches your biological question using reviewed, held-out specimens.

## Saved studies

A new study is automatically created at:

~~~text
AppLocalDataLocation/napari-browser-assistant/studies/<id>/study.browser-analysis.json
~~~

The location is visible in Discover. **Save As…** writes a new file and rebases
source/checkpoint references. **Open study…** restores it. The last study restores
automatically when the browser starts.

Successful preparations/searches and each review, class assignment, dataset
selection, and specimen-split change autosave atomically. Conflicting external
edits are reported rather than overwriting another writer's reviews.

The study is separate from Label Assistant's `.label-assistant.json`. The analysis
workflow reads that workspace as a source and never rewrites it.

| JSON field | Meaning |
| --- | --- |
| `format`, `version`, `id` | `napari-browser-analysis`, schema version 1, stable study ID |
| `images` | Relative paths, Image index, dimensions, axes, revision, specimen/group |
| `regions` | Stable IDs, image ID, level-0 YX bounds, slice coordinates, revision |
| `runs` | Encoder/backend versions, checkpoint SHA256, preprocessing, device, sampling, indexed region IDs |
| `sessions` | Query/run IDs, ordered results, original scores/ranks, excluded sources |
| `reviews` | Human relevance decisions keyed by session and result region |
| `classes` | Stable biological category IDs and names |
| `annotations` | Proposed/confirmed region categories, separate from relevance |
| `selections` | Explicitly included dataset region IDs |
| `groups` | Specimen/group assignments to unassigned/train/validation/test |
| `datasets` | References to immutable dataset exports |

An example region record, abbreviated from the surrounding study:

~~~json
{
  "id": "stable-region-id",
  "image_id": "stable-image-id",
  "bounds": [[1000, 1512], [2000, 2512]],
  "plane": [null, null],
  "coordinate_space": "level-0-pixels",
  "source_revision": "metadata-fingerprint"
}
~~~

For ZYX, `plane` could be `[25, null, null]`. Source pixels and model weights remain
external references rather than being embedded in the JSON.

## Caches and revisions

Rebuildable vectors and preview PNGs live under the plugin cache directory,
isolated by study ID. SQLite vectors are keyed by run and region. Changed encoder
weights, backend versions, or preprocessing require a new run.

Deleting the cache does not delete studies, rankings, reviews, annotations, or
dataset selections. Saved searches remain reviewable. Preparing the same regions
restores cached thumbnails; a new search needs prepared vectors.

Source revisions use metadata/stat fingerprints, not full hashes of giant pixel
or chunk contents. Changed/missing sources are excluded from new searches and
blocked from crop export. Saved decisions remain available. In-place Zarr chunk
edits without updated store metadata cannot be detected reliably; record a
metadata revision, re-add those sources, and rebuild.

Patch reads reject more than four million output samples or estimated chunk
decodes above 128 MiB. TIFFs use lazy Zarr access when available. Oversized reads
report an error suggesting a coarser level or tiled OME-Zarr. Cancellation is
checked between regions; one decode/model call must finish first.

## Classes, specimens, and training export

Expand **Dataset: classes, specimen splits, and export**.

- Enter a biological class, choose **Proposed** or **Confirmed by reviewer**, and
  **Assign class** to selected results.
- Select an image in the specimen controls. Replace its default image-specific
  group with the animal/specimen ID when images are biologically related.
- Assign the group to train, validation, test, or unassigned. All images in a
  group share its split. Crops are never randomly split independently.
- Choose the task, **Crop level**, and **Export selected dataset…**.

Each export creates a new directory:

~~~text
dataset-<unique-version-id>/
  dataset.json
  crops/
    <region-id>.tif
~~~

Crops preserve source pixel values at the explicitly chosen pyramid level.
They are not gallery thumbnails or normalized model inputs. Metadata retains
level-0 bounds, exported level/shape/pixel scale, source revision, group/split,
annotations, encoder runs, and retrieval provenance.

| Export | Requirements |
| --- | --- |
| Self-supervised crops | Explicitly selected regions; no biological classes needed |
| Region classification | Every selected crop needs a confirmed class; integer targets map to stable class IDs and names |
| Similarity pairs | Selected candidates with relevant/irrelevant reviews; query crops and relationships are included |

Uncertain reviews are not turned into positive/negative targets. Pairs spanning
incompatible splits are rejected. Unassigned samples need group assignments before
training. Query relevance is not automatically biological class equivalence.

Exports are trainer-neutral datasets. A later adapter can consume them for the
chosen training method. They are not checkpoints or direct input adapters for a
particular EM-DINO/U-Net command. Classification labels describe regions; no
segmentation masks are generated. Automatic training, RAG interpretation, and
automatic biological classification are outside this initial workflow.

## Validation

Offscreen tests cover bounded crop access, NGFF 0.4/0.5, retrieval, overlap
exclusion, persistent reviews, cache loss, revisions, cancellation, confirmations,
specimen splits, immutable exports, and real napari query/result interactions.

A separate smoke test used a real 49,976 × 59,716 six-level workspace, cached
pretrained ResNet18, 512-dimensional vectors, a query, saved review/selection,
and a source-valued level-5 crop export. This validates integration, not biological
retrieval accuracy.
