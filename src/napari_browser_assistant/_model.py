"""Lazy table cells: one Python record per file, with icons only for viewed entries."""
from datetime import datetime

from qtpy.QtCore import QAbstractTableModel, QModelIndex, Qt
from qtpy.QtGui import QIcon

from ._library import capabilities

ROLE_PATH = Qt.UserRole + 1
ROLE_SORT = Qt.UserRole + 2
ROLE_ASSET = Qt.UserRole + 3
ROLE_PAIRING = Qt.UserRole + 4


class EntryItem:
    """Small compatibility handle shared by the gallery's preview callbacks."""
    def __init__(self, model, row, column=0):
        self.model, self._row, self.column = model, row, column

    def row(self):
        return self._row

    def _set(self, role, value):
        self.model.overrides[self._row, self.column, role] = value
        index = self.model.index(self._row, self.column)
        self.model.dataChanged.emit(index, index, [role])

    def setToolTip(self, value):
        self._set(Qt.ToolTipRole, value)

    def setIcon(self, value):
        self._set(Qt.DecorationRole, value)

    def icon(self):
        return self.model.data(self.model.index(self._row, self.column), Qt.DecorationRole) or QIcon()

    def setText(self, value):
        self._set(Qt.DisplayRole, value)

    def setData(self, value, role):
        self._set(role, value)


class LibraryModel(QAbstractTableModel):
    headers = ["File", "Shape", "Dtype", "Size", "Modified", "Capability / availability",
               "Role", "Source image", "Pairing"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.entries, self.offline, self.overrides = [], set(), {}

    def set_entries(self, entries, offline):
        self.beginResetModel()
        self.entries, self.offline, self.overrides = list(entries), set(offline), {}
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.entries)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.headers)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.headers[section]
        return None

    def item(self, row, column=0):
        return EntryItem(self, row, column)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        key = (row, col, role)
        if key in self.overrides:
            return self.overrides[key]
        entry = self.entries[row]
        path = str(entry.path)
        if role == ROLE_PATH:
            return path
        if role == ROLE_ASSET:
            return entry.role
        if role == ROLE_PAIRING:
            return entry.pairing
        if role == Qt.ToolTipRole:
            pairing = ("\n" + entry.pairing + (" · " + entry.source_path if entry.source_path else "")) if entry.role == "mask" else ""
            return path + pairing + ("\nOffline" if path in self.offline else "") + "\n" + capabilities(entry.path)
        if role not in (Qt.DisplayRole, ROLE_SORT):
            return None
        if col == 0:
            return entry.path.name.casefold() if role == ROLE_SORT else (
                entry.path.name + (' · Offline' if path in self.offline else '')
                + ('\n' + entry.pairing if entry.role == 'mask' else ''))
        if col in (1, 2):
            return entry.shape if col == 1 else entry.dtype
        if col == 3:
            if role == ROLE_SORT:
                return entry.size_bytes
            if entry.path.suffix.lower() == ".zarr":
                return "—"
            value = float(entry.size_bytes)
            for unit in ("B", "KB", "MB", "GB", "TB"):
                if value < 1024 or unit == "TB":
                    return f"{value:.1f} {unit}"
                value /= 1024
        if col == 4:
            if role == ROLE_SORT:
                return entry.modified_ns
            return datetime.fromtimestamp(entry.modified_ns / 1e9).strftime("%Y-%m-%d %H:%M") if entry.modified_ns else ""
        if col == 6:
            return entry.role.title()
        if col == 7:
            return entry.source_path or ("—" if entry.role == "mask" else "")
        if col == 8:
            return entry.pairing
        return "Offline" if path in self.offline else capabilities(entry.path)
