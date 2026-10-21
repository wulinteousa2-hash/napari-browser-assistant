import napari
from ._widget import BrowserAssistantWidget

viewer = napari.Viewer()
viewer.window.add_dock_widget(BrowserAssistantWidget(viewer), name='Browser Assistant', area='right')
napari.run()
