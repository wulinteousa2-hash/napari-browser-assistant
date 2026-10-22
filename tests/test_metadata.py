import threading
from copy import deepcopy
import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from napari_browser_assistant import _metadata as ocr
from napari_browser_assistant._study import Study
from test_browser import app, make_widget, settle


@pytest.fixture
def study(tmp_path):
    study = Study.create(tmp_path / 'study.json')
    for index in range(3):
        path = tmp_path / f'image{index}.png'
        Image.new('L', (200, 100), 'white').save(path)
        study.add_image(path)
    study.save()
    return study


@pytest.fixture
def mock_ocr(monkeypatch):
    monkeypatch.setattr(ocr, 'engine_version', lambda: 'tesseract test')
    calls = []
    def recognize(data, y0, cancel):
        calls.append((data.shape, y0))
        return '120 kV 4 µm', [{'text': '120', 'confidence': 94., 'bbox': [3, y0 + 2, 20, 10]}]
    monkeypatch.setattr(ocr, 'recognize', recognize)
    return calls


def run(study, **kwargs):
    return ocr.extract_batch(study.clone(), [i['id'] for i in study.document['images']],
                             'footer', kwargs.pop('cancel', threading.Event()),
                             kwargs.pop('progress', lambda *args: None), **kwargs)


def test_fields():
    text = ('Calibration: 6.875626 nm/pix\n4 µm 120 kV 1500×\nCamera: CCD 42\n'
            'Exposure: 250 ms\n2025-10-21 12:34:56\nTEM mode: Bright field\nunknown label')
    assert ocr.parse_fields(text) == {
        'calibration': '6.875626 nm/pix', 'scale_bar': '4 µm',
        'accelerating_voltage': '120 kV', 'direct_magnification': '1500×',
        'camera': 'CCD 42', 'exposure': '250 ms',
        'acquisition_datetime': '2025-10-21 12:34:56', 'tem_mode': 'Bright field'}
    assert ocr.parse_fields('6.875626 nm/pix') == {'calibration': '6.875626 nm/pix'}
    assert ocr.parse_fields('unrecognized') == {}


def test_persistence_cache_and_generations(study, mock_ocr):
    source = study.resolve(study.document['images'][0]['path'])
    before = source.read_bytes()
    extracted, detail = run(study)
    assert detail['completed'] == 3
    assert mock_ocr == [((25, 200), 75)] * 3
    record = extracted.document['ocr_records'][0]
    assert record['words'][0]['bbox'] == [3, 77, 20, 10]
    assert record['review_status'] == 'unverified'
    record.update(corrected_text='corrected', fields={'camera': 'manual'}, review_status='reviewed')
    extracted.save()
    loaded = Study.load(study.path)
    assert loaded.document == extracted.document
    cached, detail = run(loaded)
    assert detail['skipped'] == 3 and len(mock_ocr) == 3
    forced, detail = run(cached, force=True)
    assert len(forced.document['ocr_records']) == 6
    assert forced.document['ocr_records'][0]['corrected_text'] == 'corrected'
    assert forced.document['images'][0]['pixel_scale'] is None
    assert source.read_bytes() == before
    # A different setting is a separate extraction.
    _, detail = ocr.extract_batch(loaded.clone(), [loaded.document['images'][0]['id']], 'full', threading.Event(), lambda *a: None)
    assert detail['completed'] == 1


def test_cancel_and_individual_failures(study, mock_ocr, monkeypatch):
    cancel = threading.Event()
    extracted, detail = run(study, cancel=cancel, progress=lambda *a: cancel.set())
    assert detail['cancelled'] and detail['completed'] == 1
    extracted.save()
    assert len(Study.load(study.path).document['ocr_records']) == 1
    study.resolve(study.document['images'][0]['path']).unlink()
    extracted, detail = run(study)
    assert len(detail['errors']) == 1 and detail['completed'] == 2
    monkeypatch.setattr(ocr, 'recognize', lambda *a: None)
    extracted, _ = run(study)
    assert extracted.document['ocr_records'] == []


def test_missing_executable(monkeypatch):
    monkeypatch.setattr(ocr.shutil, 'which', lambda *a: None)
    with pytest.raises(RuntimeError, match='Install the local Tesseract'):
        ocr.engine_version()


def test_conflict_and_old_studies(study, mock_ocr):
    assert 'ocr_records' not in Study.load(study.path).document
    extracted, _ = run(study)
    study.document['name'] = 'external edit'
    study.save()
    with pytest.raises(RuntimeError, match='changed on disk'):
        extracted.save()
    invalid = deepcopy(extracted.document)
    invalid['ocr_records'][0]['image_id'] = 'missing'
    with pytest.raises(ValueError, match='Invalid OCR image'):
        Study(study.path, invalid)


def test_ui_shared_study_drafts_and_lock(app, tmp_path, study, mock_ocr):
    root = tmp_path / 'library'
    root.mkdir()
    widget, _, _ = make_widget(app, tmp_path / 'ui', root)
    discover, metadata = widget.discover, widget.metadata_tab
    assert widget.main_tabs.tabText(2) == 'Metadata'
    assert 'Open or create' in metadata.message.text()
    discover.open_study(study.path)
    metadata.extract(True)
    assert not metadata.panel.isEnabled()
    with pytest.raises(ValueError, match='current operation'):
        discover.new_study(tmp_path / 'blocked.json')
    settle(app, widget, lambda: discover._work is None)
    assert len(discover.study.document['ocr_records']) == 3
    metadata.text.setPlainText('my unsaved correction')
    metadata.images.setCurrentIndex(1)
    metadata.images.setCurrentIndex(0)
    assert metadata.text.toPlainText() == 'my unsaved correction'
    other = Study.create(tmp_path / 'other.json')
    discover.open_study(other.path)
    discover.open_study(study.path)
    assert metadata.text.toPlainText() == 'my unsaved correction'
    metadata.fields.setPlainText('{"camera": "Reviewed camera"}')
    metadata.reviewed.setChecked(True)
    metadata.save()
    record = Study.load(study.path).document['ocr_records'][0]
    assert record['corrected_text'] == 'my unsaved correction'
    assert record['original_text'] == '120 kV 4 µm'
    assert record['review_status'] == 'reviewed'
    metadata.show_preview()
    assert metadata.preview.pixmap() is not None
    widget.close()


@pytest.mark.skipif(not shutil.which('tesseract'), reason='Tesseract executable is not installed')
def test_real_ocr_smoke():
    image = Image.new('L', (900, 140), 'white')
    font = ImageFont.truetype('DejaVuSans.ttf', 48)
    ImageDraw.Draw(image).text((20, 20), '120 kV 1500x', fill='black', font=font)
    text, words = ocr.recognize(np.asarray(image), 70, threading.Event())
    assert '120' in text and words
    assert all(word['bbox'][1] >= 70 for word in words)


def test_executable_tsv_coordinates_and_cancel(monkeypatch):
    class Process:
        returncode = 0
        stopped = False
        def __init__(self, args, stdout, stderr):
            assert args[0] == 'tesseract' and args[-1] == 'tsv'
            stdout.write('level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n'
                         '5\t1\t1\t1\t1\t1\t10\t8\t30\t12\t92.5\t120\n'
                         '5\t1\t1\t1\t1\t2\t45\t8\t20\t12\t88\tkV\n')
        def poll(self):
            return self.returncode
        def terminate(self):
            self.stopped = True
        def wait(self, timeout=None):
            pass
    monkeypatch.setattr(ocr.subprocess, 'Popen', Process)
    text, words = ocr.recognize(np.zeros((40, 100), dtype='uint8'), 75, threading.Event())
    assert text == '120 kV'
    assert words[0] == {'text': '120', 'confidence': 92.5, 'bbox': [10, 83, 30, 12]}
    Process.returncode = None
    cancel = threading.Event()
    cancel.set()
    assert ocr.recognize(np.zeros((40, 100), dtype='uint8'), 0, cancel) is None
    Process.returncode = 1
    with pytest.raises(RuntimeError, match='Tesseract failed'):
        ocr.recognize(np.zeros((40, 100), dtype='uint8'), 0, threading.Event())


def test_source_change_and_memory_budget(study, mock_ocr, monkeypatch):
    import napari_browser_assistant._discovery_io as image_io
    extracted, _ = run(study)
    image = extracted.document['images'][0]
    Image.new('L', (210, 100)).save(extracted.resolve(image['path']))
    changed, detail = run(extracted)
    assert detail['skipped'] == 2 and len(detail['errors']) == 1
    changed.add_image(changed.resolve(image['path']))
    changed, detail = run(changed)
    assert detail['completed'] == 1 and detail['skipped'] == 2
    monkeypatch.setattr(image_io, 'MAX_PATCH_PIXELS', 10)
    failed, detail = run(study, force=True)
    assert detail['completed'] == 0 and len(detail['errors']) == 3


def test_ui_missing_engine_and_cancel(app, tmp_path, study, mock_ocr, monkeypatch):
    root = tmp_path / 'empty'
    root.mkdir()
    widget, _, _ = make_widget(app, tmp_path / 'ui', root)
    discover, metadata = widget.discover, widget.metadata_tab
    discover.open_study(study.path)
    def unavailable():
        raise RuntimeError(ocr.GUIDANCE)
    monkeypatch.setattr(ocr, 'engine_version', unavailable)
    metadata.extract(True)
    settle(app, widget, lambda: discover._work is None)
    assert 'Install the local Tesseract' in metadata.status.text()
    assert metadata.panel.isEnabled()
    monkeypatch.setattr(ocr, 'engine_version', lambda: 'test')
    def recognize(data, y0, cancel):
        cancel.set()
        return 'completed', []
    monkeypatch.setattr(ocr, 'recognize', recognize)
    metadata.extract(True)
    settle(app, widget, lambda: discover._work is None)
    records = Study.load(study.path).document['ocr_records']
    assert len(records) == 1
    assert 'cancelled' in metadata.status.text()
    assert metadata.panel.isEnabled() and discover.add_button.isEnabled()
    widget.close()


@pytest.mark.parametrize('scope,size,mode', [
    ('full', (2401, 2003), 'L'),
    ('footer', (4201, 4003), 'L'),
    ('full', (1801, 1003), 'RGB'),
])
def test_large_ocr_resize_and_source_coordinates(tmp_path, monkeypatch, scope, size, mode, mock_ocr):
    path = tmp_path / 'large.png'
    Image.new(mode, size, 'white').save(path)
    before = path.read_bytes()
    study = Study.create(tmp_path / 'large-study.json')
    image = study.add_image(path)
    original_read = ocr.RegionSource.read
    reads = []
    def bounded_read(self, bounds, *args, **kwargs):
        reads.append(bounds)
        assert (bounds[0][1] - bounds[0][0]) * (bounds[1][1] - bounds[1][0]) <= 513 ** 2
        return original_read(self, bounds, *args, **kwargs)
    monkeypatch.setattr(ocr.RegionSource, 'read', bounded_read)
    extracted, detail = ocr.extract_batch(study, [image['id']], scope, threading.Event(), lambda *a: None)
    assert detail['completed'] == 1, detail['errors']
    record = extracted.document['ocr_records'][0]
    assert np.prod(record['ocr_input_shape']) <= 4 * 1024 * 1024
    sx, sy = record['ocr_to_source_scale_xy']
    y0 = size[1] * 3 // 4 if scope == 'footer' else 0
    assert sx > 1 and sy > 1
    assert record['words'][0]['bbox'] == pytest.approx([3 * sx, y0 + 2 * sy, 20 * sx, 10 * sy])
    assert len(reads) > 1 and min(bounds[0][0] for bounds in reads) == y0
    assert max(bounds[0][1] for bounds in reads) == size[1]
    assert path.read_bytes() == before
    extracted.save()
    assert Study.load(extracted.path).document['ocr_records'][0] == record


def test_resize_cancellation_and_opt_out(tmp_path, mock_ocr):
    path = tmp_path / 'large.png'
    Image.new('L', (2401, 2003), 'white').save(path)
    study = Study.create(tmp_path / 'study.json')
    image = study.add_image(path)
    with pytest.raises(ValueError, match='4-million-sample'):
        ocr.read_ocr_scope(study, image, 'full', resize=False)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(InterruptedError):
        ocr.read_ocr_scope(study, image, 'full', cancel=cancel)


def test_large_uint16_rgb_resize(tmp_path):
    import tifffile
    path = tmp_path / 'rgb.tif'
    data = np.zeros((1003, 1801, 3), dtype='uint16')
    data[..., 0] = 60000
    data[..., 1] = 30000
    tifffile.imwrite(path, data, photometric='rgb')
    study = Study.create(tmp_path / 'study.json')
    image = study.add_image(path)
    resized, y0, sx, sy = ocr.read_ocr_scope(study, image, 'full')
    assert resized.dtype == np.uint8 and resized.size <= 4 * 1024 * 1024
    assert sx > 1 and sy > 1 and y0 == 0
    assert np.all(resized[..., 0] == 255)
    assert np.all(resized[..., 2] == 0)
    assert np.all((resized[..., 1] >= 127) & (resized[..., 1] <= 128))
