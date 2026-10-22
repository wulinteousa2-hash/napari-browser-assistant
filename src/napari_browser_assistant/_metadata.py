"""Local, bounded OCR. Printed acquisition values are always unverified initially."""
from __future__ import annotations

import csv
import math
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ._discovery_io import RegionSource, fingerprint
from ._study import now, uid

GUIDANCE = ('Tesseract is unavailable. Install the local Tesseract executable and English '
            'language data (Ubuntu: sudo apt install tesseract-ocr; macOS: brew install '
            'tesseract; Windows: install Tesseract and add it to PATH). No downloads are performed by this app.')


def parse_fields(text):
    """Conservative label/unit parsing; retain the full text separately."""
    number = r'\d+(?:[.,]\d+)?'
    patterns = {
        'calibration': rf'({number}\s*(?:nm|[µμu]m|mm)\s*/\s*(?:pix(?:el)?s?|px))',
        'scale_bar': rf'({number}\s*(?:nm|[µμu]m|mm))\b(?!\s*/\s*(?:pix|px))',
        'accelerating_voltage': rf'({number}\s*kV)\b',
        'direct_magnification': rf'({number}\s*[×x])(?!\w)',
        'camera': r'(?:Camera)\s*[:=]\s*([^\n;]+)',
        'exposure': rf'(?:Exposure(?: time)?)\s*[:=]?\s*({number}\s*(?:ms|s|sec|seconds?)\b)',
        'acquisition_datetime': r'((?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})(?:[ T]+\d{1,2}:\d{2}(?::\d{2})?)?)',
        'tem_mode': r'(?:TEM\s*mode|Mode)\s*[:=]\s*([^\n;]+)',
    }
    fields = {}
    # Remove calibration expressions before looking for a stand-alone scale length.
    for key, pattern in patterns.items():
        candidate = re.sub(patterns['calibration'], '', text, flags=re.I) if key == 'scale_bar' else text
        match = re.search(pattern, candidate, flags=re.I)
        if match:
            fields[key] = match.group(1).strip()
    return fields


def engine_version():
    executable = shutil.which('tesseract')
    if not executable:
        raise RuntimeError(GUIDANCE)
    try:
        result = subprocess.run([executable, '--version'], capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(GUIDANCE) from error
    return result.stdout.splitlines()[0]


def _uint8(data):
    if data.dtype != np.uint8:
        data = np.nan_to_num(data.astype(np.float32))
        lo, hi = float(data.min()), float(data.max())
        data = ((data - lo) * (255 / (hi - lo)) if hi > lo else np.zeros_like(data)).clip(0, 255).astype('uint8')
    return data


def read_ocr_scope(study, image, scope, resize=True, cancel=None):
    """Assemble a bounded OCR image from bounded native tiles when necessary.

    The output maps exactly to the requested level-0 rectangle. Decode limits
    remain enforced by RegionSource, including for a single oversized chunk.
    """
    from ._discovery_io import MAX_PATCH_PIXELS
    if scope not in ('footer', 'full'):
        raise ValueError('Unknown OCR scope')
    study.assert_current(image)
    with RegionSource(study.resolve(image['path']), image['image_index']) as source:
        height, width = (source.shapes[0][a] for a in (source.y_axis, source.x_axis))
        y0 = height * 3 // 4 if scope == 'footer' else 0
        crop_height = height - y0
        channels = source.shapes[0][-1] if source.rgb else 1
        samples = crop_height * width * channels
        if samples <= MAX_PATCH_PIXELS or not resize:
            data = source.read([[y0, height], [0, width]], image['plane'])
            return _uint8(data), y0, 1., 1.
        ratio = math.sqrt(MAX_PATCH_PIXELS / samples)
        out_width = max(1, int(width * ratio))
        out_height = max(1, min(int(crop_height * ratio), MAX_PATCH_PIXELS // (out_width * channels)))
        if out_width * out_height * channels > MAX_PATCH_PIXELS:
            raise ValueError('Image aspect ratio exceeds the OCR resize budget')
        shape = (out_height, out_width) + ((channels,) if source.rgb else ())
        dtype = np.uint8 if source.dtype == 'uint8' else np.float32
        data = np.empty(shape, dtype=dtype)
        # Output tiles correspond to native rectangles no larger than ~512x512.
        # Never load the entire large region just to shrink it afterwards.
        tile_height = max(1, int(512 * out_height / crop_height))
        tile_width = max(1, int(512 * out_width / width))
        for oy in range(0, out_height, tile_height):
            ey = min(out_height, oy + tile_height)
            for ox in range(0, out_width, tile_width):
                if cancel is not None and cancel.is_set():
                    raise InterruptedError('OCR cancelled')
                ex = min(out_width, ox + tile_width)
                bounds = [[y0 + oy * crop_height // out_height, y0 + ey * crop_height // out_height],
                          [ox * width // out_width, ex * width // out_width]]
                tile = source.read(bounds, image['plane'])
                target = (ex - ox, ey - oy)
                if source.rgb and tile.dtype != np.uint8:
                    # Pillow cannot resize multichannel uint16/float images directly.
                    for channel in range(channels):
                        resized = Image.fromarray(tile[..., channel].astype(np.float32)).resize(
                            target, Image.Resampling.BILINEAR)
                        data[oy:ey, ox:ex, channel] = np.asarray(resized)
                else:
                    if tile.dtype != np.uint8:
                        tile = tile.astype(np.float32)
                    resized = Image.fromarray(tile).resize(target, Image.Resampling.BILINEAR)
                    data[oy:ey, ox:ex] = np.asarray(resized)
    return _uint8(data), y0, width / out_width, crop_height / out_height


def read_scope(study, image, scope):
    data, y0, _, _ = read_ocr_scope(study, image, scope)
    return data, y0


def recognize(data, y0, cancel):
    """Poll the executable so cancellation also stops an in-flight image."""
    with tempfile.TemporaryDirectory(prefix='browser-ocr-') as directory:
        path = Path(directory) / 'input.png'
        Image.fromarray(data).save(path)
        # Files avoid subprocess pipe deadlocks on verbose output.
        with (Path(directory) / 'output.tsv').open('w+') as output, (Path(directory) / 'error.txt').open('w+') as error:
            process = subprocess.Popen(['tesseract', str(path), 'stdout', '-l', 'eng', '--psm', '11', 'tsv'], stdout=output, stderr=error)
            started = time.monotonic()
            try:
                while process.poll() is None:
                    if cancel.wait(.1):
                        return None
                    if time.monotonic() - started > 180:
                        raise RuntimeError('Tesseract timed out after 180 seconds')
                if process.returncode:
                    error.seek(0)
                    raise RuntimeError('Tesseract failed: ' + error.read(4000))
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            output.seek(0)
            rows = list(csv.DictReader(output, delimiter='\t'))
    words, lines = [], {}
    for row in rows:
        if row['level'] != '5' or not row['text'].strip():
            continue
        x, y, w, h = (int(row[k]) for k in ('left', 'top', 'width', 'height'))
        words.append({'text': row['text'], 'confidence': float(row['conf']),
                      'bbox': [x, y + y0, w, h]})
        key = tuple(row[k] for k in ('page_num', 'block_num', 'par_num', 'line_num'))
        lines.setdefault(key, []).append(row['text'])
    return '\n'.join(' '.join(line) for line in lines.values()), words


def extract_batch(study, identifiers, scope, cancel, progress, force=False, resize=True):
    version = engine_version()
    settings = {'scope': scope, 'language': 'eng', 'psm': 11, 'pipeline': 2, 'resize_to_budget': resize}
    records = study.document.setdefault('ocr_records', [])
    errors, completed, skipped = [], 0, 0
    for index, identifier in enumerate(identifiers):
        if cancel.is_set():
            break
        image = study.row('images', identifier)
        try:
            revision = fingerprint(study.resolve(image['path']))
            cached = any(r['image_id'] == identifier and r['source_revision'] == revision
                         and r['settings'] == settings and r['engine_version'] == version for r in records)
            if cached and not force:
                skipped += 1
            else:
                data, y0, scale_x, scale_y = read_ocr_scope(study, image, scope, resize, cancel)
                output = recognize(data, y0, cancel)
                if output is None:
                    break
                study.assert_current(image, revision)
                text, words = output
                for word in words:
                    x, y, w, h = word['bbox']
                    word['bbox'] = [x * scale_x, y0 + (y - y0) * scale_y,
                                    w * scale_x, h * scale_y]
                # Append an immutable extraction generation; never replace reviewed corrections.
                records.append({'id': uid(), 'image_id': identifier, 'original_text': text,
                                'corrected_text': text, 'original_fields': parse_fields(text),
                                'fields': parse_fields(text), 'words': words,
                                'coordinate_space': 'level-0-pixels-xywh', 'engine': 'tesseract',
                                'engine_version': version, 'extracted_at': now(),
                                'scope': scope, 'settings': dict(settings), 'plane': image['plane'],
                                'ocr_input_shape': list(data.shape),
                                'ocr_to_source_scale_xy': [scale_x, scale_y],
                                'source_revision': revision, 'review_status': 'unverified'})
                completed += 1
        except Exception as error:
            if cancel.is_set():
                break
            errors.append(f"{image['path']}: {error}")
        progress(index + 1, len(identifiers), f'OCR {index + 1}/{len(identifiers)}')
    return study, {'completed': completed, 'skipped': skipped, 'errors': errors, 'cancelled': cancel.is_set()}
