"""Review OCR generations in the active Discover study; drafts survive navigation."""
from copy import deepcopy
import json
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
                           QPushButton, QPlainTextEdit, QCheckBox, QProgressBar, QScrollArea)

from ._metadata import extract_batch, read_scope
from ._study import now


class MetadataWidget(QWidget):
    def __init__(self, browser):
        super().__init__(browser)
        self.discover = browser.discover
        self._key = None
        self._drafts = {}
        layout = QVBoxLayout(self)
        self.message = QLabel('Open or create a study in Discover, then add Library images.')
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.panel = QWidget()
        panel = QVBoxLayout(self.panel)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)
        layout.addWidget(scroll, 1)
        self.images = QComboBox()
        panel.addWidget(self.images)
        self.file_metadata = QPlainTextEdit()
        self.file_metadata.setReadOnly(True)
        self.file_metadata.setMaximumHeight(120)
        panel.addWidget(QLabel('File metadata (read-only; printed calibration is separate)'))
        panel.addWidget(self.file_metadata)
        self.preview = QLabel('Select Preview to view the saved representative plane.')
        self.preview.setAlignment(Qt.AlignCenter)
        panel.addWidget(self.preview)
        self.preview_button = self.button('Preview image', self.show_preview)
        panel.addWidget(self.preview_button)
        row = QHBoxLayout()
        self.scope = QComboBox()
        self.scope.addItem('Bottom 25% (footer)', 'footer')
        self.scope.addItem('Full image', 'full')
        row.addWidget(self.scope)
        row.addWidget(self.button('Extract selected', lambda: self.extract(False)))
        row.addWidget(self.button('Extract all', lambda: self.extract(True)))
        panel.addLayout(row)
        self.resize = QCheckBox('Resize large OCR regions to the memory budget (may reduce small-text detail)')
        self.resize.setChecked(True)
        panel.addWidget(self.resize)
        self.force = QCheckBox('Re-extract explicitly (adds a generation; keeps all corrections)')
        panel.addWidget(self.force)
        self.generations = QComboBox()
        panel.addWidget(self.generations)
        self.original = QPlainTextEdit()
        self.original.setReadOnly(True)
        panel.addWidget(QLabel('Original OCR output and provenance (read-only)'))
        panel.addWidget(self.original)
        self.text = QPlainTextEdit()
        panel.addWidget(QLabel('Corrected text — unsaved drafts are retained during navigation'))
        panel.addWidget(self.text)
        self.fields = QPlainTextEdit()
        panel.addWidget(QLabel('Printed acquisition fields (editable JSON object; unverified until reviewed)'))
        panel.addWidget(self.fields)
        self.reviewed = QCheckBox('Reviewed by user')
        panel.addWidget(self.reviewed)
        panel.addWidget(self.button('Save corrections to study', self.save))
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.cancel = self.button('Cancel OCR / current Discover operation', self.discover.cancel_work)
        layout.addWidget(self.cancel)
        self.status = QLabel('')
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.images.currentIndexChanged.connect(self.select_image)
        self.generations.currentIndexChanged.connect(self.select_generation)
        self.discover.study_changed.connect(self.refresh)
        self.discover.operation_changed.connect(self.set_busy)
        self.discover._work_progress.connect(self.update_progress)
        for combo in (self.images, self.generations):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
        self.refresh()

    def button(self, title, function):
        button = QPushButton(title)
        def invoke():
            try:
                function()
            except Exception as error:
                self.status.setText(str(error))
        button.clicked.connect(invoke)
        return button

    def set_busy(self, busy):
        self.panel.setEnabled(not busy)
        self.cancel.setEnabled(busy)
        if not busy:
            self.status.setText(self.discover.status.text())

    def update_progress(self, value, total, message):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(value)
        self.status.setText(message)

    def stash(self):
        if self._key is not None:
            self._drafts[self._key] = (self.text.toPlainText(), self.fields.toPlainText(), self.reviewed.isChecked())

    def refresh(self):
        self.stash()
        self._key = None
        study = self.discover.study
        previous = self.images.currentData()
        self.images.blockSignals(True)
        self.images.clear()
        if study:
            for image in study.document['images']:
                self.images.addItem(f"{Path(image['path']).name} [{image['image_index']}]", image['id'])
        self.images.setCurrentIndex(max(0, self.images.findData(previous)))
        self.images.blockSignals(False)
        self.message.setText(f'Active Discover study: {study.path}' if study else 'Open or create a study in Discover, then add Library images.')
        self.panel.setEnabled(study is not None and self.discover._work is None)
        self.cancel.setEnabled(self.discover._work is not None)
        self.select_image()

    def select_image(self, *_):
        self.stash()
        self._key = None
        self.preview.clear()
        self.file_metadata.clear()
        study, identifier = self.discover.study, self.images.currentData()
        self.generations.blockSignals(True)
        self.generations.clear()
        if study and identifier:
            image = study.row('images', identifier)
            self.file_metadata.setPlainText(json.dumps(image, indent=2))
            for record in study.document.get('ocr_records', []):
                if record['image_id'] == identifier:
                    self.generations.addItem(f"{record['extracted_at']} · {record['scope']} · {record['review_status']}", record['id'])
        self.generations.setCurrentIndex(self.generations.count() - 1)
        self.generations.blockSignals(False)
        self.select_generation()

    def record(self):
        study = self.discover.study
        identifier = self.generations.currentData()
        return next((r for r in study.document.get('ocr_records', []) if r['id'] == identifier), None) if study else None

    def select_generation(self, *_):
        self.stash()
        self._key = None
        record = self.record()
        self.original.clear()
        self.text.clear()
        self.fields.clear()
        self.reviewed.setChecked(False)
        if record:
            self._key = (str(self.discover.study.path), record['image_id'], record['id'])
            provenance = {k: v for k, v in record.items() if k not in ('corrected_text', 'fields')}
            self.original.setPlainText(json.dumps(provenance, indent=2, ensure_ascii=False))
            text, fields, reviewed = self._drafts.get(self._key, (record['corrected_text'], json.dumps(record['fields'], indent=2, ensure_ascii=False), record['review_status'] == 'reviewed'))
            self.text.setPlainText(text)
            self.fields.setPlainText(fields)
            self.reviewed.setChecked(reviewed)

    def show_preview(self):
        self.discover._require_study()
        image = self.discover.study.row('images', self.images.currentData())
        data, _ = read_scope(self.discover.study, image, 'full')
        from PIL import Image
        import numpy as np
        preview = Image.fromarray(data).convert('RGB')
        preview.thumbnail((600, 240))
        array = np.asarray(preview).copy()
        height, width = array.shape[:2]
        picture = QImage(array.data, width, height, width * 3, QImage.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(picture))

    def save(self):
        record = self.record()
        if record is None:
            raise ValueError('Extract OCR before saving corrections')
        fields = json.loads(self.fields.toPlainText())
        if not isinstance(fields, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in fields.items()):
            raise ValueError('Fields must be a JSON object with text values')
        identifier = record['id']
        text = self.text.toPlainText()
        status = 'reviewed' if self.reviewed.isChecked() else 'unverified'
        def edit(study):
            saved = next(r for r in study.document['ocr_records'] if r['id'] == identifier)
            saved.update(corrected_text=text, fields=deepcopy(fields), review_status=status, corrected_at=now())
        self.discover._edit(edit)
        self.status.setText('Corrections saved in the Discover study.')

    def extract(self, all_images):
        self.discover._require_study()
        self.stash()
        snapshot = self.discover.study.clone()
        identifiers = [image['id'] for image in snapshot.document['images']] if all_images else [self.images.currentData()]
        if not identifiers or identifiers == [None]:
            raise ValueError('Add images in Discover first')
        scope, force, resize = self.scope.currentData(), self.force.isChecked(), self.resize.isChecked()
        self.discover._start('OCR', lambda progress: extract_batch(snapshot, identifiers, scope, self.discover._cancel, progress, force, resize))
