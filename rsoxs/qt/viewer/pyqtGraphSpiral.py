from nbs_viewer.views.display.plotDisplay import ImageGridDisplay
from nbs_viewer.views.plot.imageGridWidget import ImageGridWidget
from qtpy.QtCore import QThread, Signal, QEvent
from qtpy.QtWidgets import (
    QListView,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QPushButton,
    QHBoxLayout,
    QGridLayout,
)
from qtpy.QtCore import Qt
import pyqtgraph as pg
from pyqtgraph import ImageItem
import numpy as np
import PyHyperScattering as phs


class CustomImageItem(ImageItem):
    sigMouseClicked = Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Disable Selectable and Movable flags to prevent accepting mousePressEvent
        # self.setFlag(pg.QtWidgets.QGraphicsItem.ItemIsSelectable, False)
        # self.setFlag(pg.QtWidgets.QGraphicsItem.ItemIsMovable, False)

    def mouseClickEvent(self, event):
        if event.button() == pg.QtCore.Qt.MouseButton.LeftButton:
            self.sigMouseClicked.emit(event)
            print("CustomImageItem.mouseClickEvent")
        super().mouseClickEvent(event)

    def mousePressEvent(self, event):
        """Override to not accept press events, allowing scroll area events to work."""
        # Don't accept the press event to allow scroll area panning
        print("CustomImageItem.mousePressEvent")
        event.ignore()
        # super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Override to not accept release events, allowing scroll area events to work."""
        # Don't accept the release event to allow scroll area panning
        print("CustomImageItem.mouseReleaseEvent")
        event.ignore()
        # super().mouseReleaseEvent(event)


class PyHyperWorker(QThread):
    data_ready = Signal(object)  # (x, y, plotData, artist)
    error_occurred = Signal(str)

    def __init__(self, pyhyper_data, scan_id, parent=None):
        super().__init__(parent)
        self.pyhyper_data = pyhyper_data
        self.scan_id = scan_id

    def run(self):
        try:
            data = self.pyhyper_data.loadRun(self.scan_id, dims=["sam_x", "sam_y"]).unstack("system")
            self.data_ready.emit(data)
        except Exception as e:
            error_msg = f"Error fetching plot data: {str(e)}"
            print_debug("PyHyperWorker", error_msg, category="DEBUG_PLOTS")
            self.error_occurred.emit(error_msg)


class BestImageSelector(QWidget):
    """Control widget for selecting the best image from spiral plots."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selecting_mode = False
        self.spiral_widget = None
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.select_button = QPushButton("Select Best Image")
        self.select_button.setCheckable(True)
        self.select_button.clicked.connect(self._toggle_selection_mode)

        self.clear_button = QPushButton("Clear Selection")
        self.clear_button.clicked.connect(self._clear_selection)
        self.clear_button.setEnabled(False)

        layout.addWidget(self.select_button)
        layout.addStretch()

    def _clear_selection(self):
        """Clear the selection."""
        self.spiral_widget.clear_image_selection()

    def _toggle_selection_mode(self):
        """Toggle between selection mode and normal mode."""
        self.selecting_mode = self.select_button.isChecked()

        if self.selecting_mode:
            self.select_button.setText("Use Selected Images")
            self.clear_button.setEnabled(True)

            if self.spiral_widget:
                self.spiral_widget.enable_image_selection()
        else:
            self.select_button.setText("Select Best Image")
            self.clear_button.setEnabled(False)
            if self.spiral_widget:
                self.spiral_widget.disable_image_selection()

    def set_spiral_widget(self, spiral_widget):
        """Set the spiral widget to control."""
        self.spiral_widget = spiral_widget


class PyQtGraphSpiralWidget(ImageGridWidget):
    data_ready = Signal()

    def __init__(self, *args, **kwargs):
        self.data_arrays = {}
        self.selected_images = set()
        self.plot_widgets = {}  # Store individual plot widgets
        self.image_items = {}  # Store image items for highlighting
        self.signal_connections = {}  # Track signal connections for proper cleanup
        self.index_to_coords = {}  # Store image index to coordinates
        # Override parent class methods that expect matplotlib canvas
        self.canvas = None  # No matplotlib canvas in PyQtGraph version
        # PyQtGraph configuration
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.contrastLimits = [1e-1, 1e6]
        self.limitsInboardOutboardDownUp = [0, 61.7, 0, 61.4]
        self.pyHyperLoader = None
        super().__init__(*args, **kwargs)

    def _create_plot_controls(self):
        """Create the best image selector control widget."""
        self.plot_controls = BestImageSelector()
        self.plot_controls.set_spiral_widget(self)
        if len(self.run_list_model.available_runs) > 0:
            run = self.run_list_model.available_runs[0]
            self.pyHyperLoader = phs.load.SST1RSoXSDB(catalog=run._catalog)
        return self.plot_controls

    def _connect_signals(self):
        super()._connect_signals()
        self.run_list_model.run_added.connect(self._on_run_added)
        self.data_ready.connect(self._update_grid)

    def _setup_ui(self):
        """Set up the user interface with PyQtGraph."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create scroll area for the grid
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("background-color: white;")  # White background

        # Enable drag scrolling by setting up the viewport for panning
        self.scroll_area.viewport().setCursor(Qt.OpenHandCursor)
        self.scroll_area.viewport().installEventFilter(self)

        # Create container widget for the grid
        self.grid_container = QWidget()
        self.grid_container.setStyleSheet("background-color: white;")  # White background
        # Make grid container transparent to mouse events for panning
        # self.grid_container.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        # Override mouse events to track propagation
        # self.grid_container.mousePressEvent = self._grid_container_mouse_press
        # self.grid_container.mouseReleaseEvent = self._grid_container_mouse_release

        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(15)  # Consistent spacing between plots
        # Override grid layout mouse events to track propagation
        # self.grid_layout.mousePressEvent = self._grid_layout_mouse_press
        # self.grid_layout.mouseReleaseEvent = self._grid_layout_mouse_release
        self.grid_layout.setContentsMargins(10, 10, 10, 10)  # Margin around the grid

        # Set the grid container as the scroll area widget
        self.scroll_area.setWidget(self.grid_container)

        # Add to main layout
        main_layout.addWidget(self.scroll_area)
        self.setLayout(main_layout)

    def _on_run_added(self, run):
        if run.plan_name not in ["spiralsearch", "spiral_square"]:
            self.run_list_model.remove_run(run)
            QMessageBox.warning(
                self, "Unsupported run", "Unsupported run: " + run.plan_name + " for spiral viewer"
            )
        else:
            if run.scan_id not in self.data_arrays and run.scan_id not in self.workers:
                self._start_pyhyper_worker(run)

    def _start_pyhyper_worker(self, run):
        print(f"Starting worker for run {run.scan_id}")
        if self.pyHyperLoader is None:
            self.pyHyperLoader = phs.load.SST1RSoXSDB(catalog=run._catalog)
        worker = PyHyperWorker(self.pyHyperLoader, run.scan_id, self)
        worker.data_ready.connect(lambda data: self._handle_loaded_data(run.scan_id, data))
        worker.error_occurred.connect(lambda error_msg: self._handle_error(error_msg))
        worker.finished.connect(lambda: self._cleanup_worker(run.scan_id))
        self.workers[run.scan_id] = worker
        worker.start()

    def _cleanup_worker(self, key):
        """Clean up a worker thread."""
        if key in self.workers:
            worker = self.workers.pop(key)
            try:
                worker.data_ready.disconnect()
                worker.error_occurred.disconnect()
                worker.quit()
                worker.wait()
                worker.deleteLater()
            except Exception as e:
                print_debug("ImageGridWidget", f"Error cleaning up worker: {e}")

    def _handle_loaded_data(self, scan_id, data):
        print(f"_handle_loaded_data: Data for run {scan_id} loaded")
        self.data_arrays[scan_id] = data
        self.data_ready.emit()

    def _update_grid(self):
        if len(self.run_list_model.visible_models) == 0:
            print("No visible runs")
            return

        run = self.run_list_model.visible_models[0]

        if run.scan_id in self.data_arrays:
            scan = self.data_arrays[run.scan_id]
            print(f"Data for run {run.scan_id} loaded")
        elif run.scan_id in self.workers:
            print(f"Data for run {run.scan_id} being loaded by worker")
            return
        else:
            print(f"Starting worker for run {run.scan_id}")
            self._start_pyhyper_worker(run)
            return
        self._clear_grid()
        self._create_subplots_for_page(scan)

    def _create_subplots_for_page(self, scan):
        """Create PyQtGraph subplots for the spiral data."""
        xMotorPositions = scan.attrs["sam_x"]
        yMotorPositions = scan.attrs["sam_y"]
        timestamps = scan.attrs["time"]

        numberRows, numberColumns = len(scan["sam_y"]), len(scan["sam_x"])

        # Clear existing grid
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        self.plot_widgets.clear()
        self.image_items.clear()
        self.index_to_coords.clear()

        for indexPlotRow, yMotorPosition in enumerate(scan["sam_y"]):
            for indexPlotColumn, xMotorPosition in enumerate(scan["sam_x"]):
                image = scan.sel(sam_y=yMotorPosition, sam_x=xMotorPosition, method="nearest")

                # Find image number
                for indexImage, timestamp in enumerate(timestamps):
                    if (
                        xMotorPositions[indexImage] == xMotorPosition
                        and yMotorPositions[indexImage] == yMotorPosition
                    ):
                        break

                # Create PyQtGraph plot widget
                plot_widget = pg.PlotWidget()
                plot_widget.setAspectLocked(False)
                # Disable pan/zoom but keep mouse events for selection
                plot_widget.setMouseEnabled(x=False, y=False)
                # Configure ViewBox to disable pan/zoom but allow mouse events

                # Store reference to this plot widget for mouse event handling
                plot_widget.plot_id = (indexPlotRow, indexPlotColumn)

                # Set title
                title = f"Image {indexImage}, x = {float(xMotorPosition):.1f}, y = {float(yMotorPosition):.1f}"
                plot_widget.setTitle(title, size="10pt", color="k")

                # Style the plot
                self._style_pyqtgraph_plot(plot_widget, indexPlotRow, indexPlotColumn, numberRows, numberColumns)

                # Create image item
                image_data = np.array(image)
                image_item = pg.ImageItem(image_data)  # Use regular ImageItem instead of CustomImageItem

                # Set image bounds
                x_min, x_max = self.limitsInboardOutboardDownUp[0], self.limitsInboardOutboardDownUp[1]
                y_min, y_max = self.limitsInboardOutboardDownUp[2], self.limitsInboardOutboardDownUp[3]
                image_item.setRect(pg.QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min))

                # Set color map and levels with log normalization
                # Apply log transformation to the data
                log_data = np.log10(np.maximum(image_data, self.contrastLimits[0]))
                image_item.setImage(log_data)

                # Set levels for log-transformed data
                log_min = np.log10(self.contrastLimits[0])
                log_max = np.log10(self.contrastLimits[1])
                image_item.setLevels([log_min, log_max])

                colormap = pg.colormap.get("RdYlBu_r", source="matplotlib")
                image_item.setColorMap(colormap)

                # Add image to plot
                plot_widget.addItem(image_item)

                view_box = plot_widget.getPlotItem().getViewBox()
                view_box.setMouseEnabled(x=False, y=False)
                # Set ViewBox limits to match image bounds so images fill the entire plot area
                x_min, x_max = self.limitsInboardOutboardDownUp[0], self.limitsInboardOutboardDownUp[1]
                y_min, y_max = self.limitsInboardOutboardDownUp[2], self.limitsInboardOutboardDownUp[3]
                view_box.setRange(xRange=(x_min, x_max), yRange=(y_min, y_max), padding=0)

                # Store references
                self.index_to_coords[indexImage] = (indexPlotRow, indexPlotColumn)
                self.plot_widgets[indexImage] = plot_widget
                self.image_items[indexImage] = image_item

                # Override plot widget mouse events for panning and selection
                plot_widget.mousePressEvent = lambda event, pw=plot_widget: self._plot_widget_mouse_press(
                    event, pw
                )
                plot_widget.mouseReleaseEvent = lambda event, pw=plot_widget: self._plot_widget_mouse_release(
                    event, pw
                )
                plot_widget.mouseMoveEvent = lambda event, pw=plot_widget: self._plot_widget_mouse_move(event, pw)
                plot_widget.mouseDoubleClickEvent = (
                    lambda event, pw=plot_widget: self._plot_widget_mouse_double_click(event, pw)
                )

                # Add to grid
                self.grid_layout.addWidget(plot_widget, indexPlotRow, indexPlotColumn)

    def _style_pyqtgraph_plot(self, plot_widget, row_idx, col_idx, total_rows, total_cols):
        """Style a PyQtGraph plot widget to match Matplotlib reference."""
        # Set fixed size for consistent image areas
        plot_widget.setFixedSize(250, 250)  # Fixed size for all plots
        plot_widget.getPlotItem().hideButtons()
        plot_widget.getPlotItem().showAxes(True)
        # Set white background to match Matplotlib reference
        plot_widget.setBackground("w")

        # Set axis labels on all plots for consistency
        plot_widget.setLabel("bottom", "Outboard-inboard (mm)")
        plot_widget.setLabel("left", "Down-up (mm)")

        # Set axis limits
        plot_widget.setXRange(self.limitsInboardOutboardDownUp[0], self.limitsInboardOutboardDownUp[1])
        plot_widget.setYRange(self.limitsInboardOutboardDownUp[2], self.limitsInboardOutboardDownUp[3])

        # Style all axes with consistent borders and major ticks only
        for axis_name in ["left", "bottom", "top", "right"]:
            axis = plot_widget.getAxis(axis_name)
            axis.setPen(pg.mkPen(color="k", width=1))
            axis.setStyle(maxTickLevel=0, tickLength=4)  # Only major ticks

    def enable_image_selection(self):
        """Enable image selection mode."""
        self.selected_images.clear()
        self._clear_highlights()
        print("enable_image_selection")
        # Image selection is now handled by plot widget mouse events

    def disable_image_selection(self):
        """Disable image selection mode."""
        print("disable_image_selection")

        # Print selected images and clear highlights
        if self.selected_images:
            selected_list = sorted(list(self.selected_images))
            print(f"Selected images: {selected_list}")
            self.clear_image_selection()

    def clear_image_selection(self):
        self._clear_highlights()
        self.selected_images.clear()

    def _highlight_plot(self, plot_widget):
        """Highlight a plot by changing its border color."""
        # Create a red border by modifying the plot widget's border
        plot_widget.setStyleSheet("border: 3px solid red;")

    def _unhighlight_plot(self, plot_widget):
        """Remove highlighting from a plot."""
        plot_widget.setStyleSheet("")

    def _clear_highlights(self):
        """Clear all highlights."""
        for plot_widget in self.plot_widgets.values():
            self._unhighlight_plot(plot_widget)

    def _clear_grid(self):
        """Clear the grid and all plot widgets."""
        # Clear existing grid
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # Disconnect all signal connections before clearing
        for image_index, image_item in self.image_items.items():
            if image_index in self.signal_connections:
                try:
                    handler = self.signal_connections[image_index]
                    image_item.sigMouseClicked.disconnect(handler)
                except:
                    pass  # Already disconnected or item doesn't exist

        self.plot_widgets.clear()
        self.image_items.clear()
        self.index_to_coords.clear()
        self.selected_images.clear()
        self.signal_connections.clear()

    def _open_plot_popup(self, plot_widget):
        print("open_plot_popup")
        """Open a popup window with a larger version of the plot."""
        # Find the image number for this plot widget
        image_number = None
        for img_num, pw in self.plot_widgets.items():
            if pw == plot_widget:
                image_number = img_num
                break

        if image_number is None:
            print("image_number is None")
            return
        print(f"image_number: {image_number}")

        # Create popup window
        popup = QWidget(parent=self)  # Set parent to ensure proper window management
        popup.setWindowTitle(f"Image {image_number} - Detailed View")
        popup.setGeometry(100, 100, 800, 600)
        # Set window flags to make it a proper popup window
        popup.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        # Create layout
        layout = QVBoxLayout(popup)

        # Create a new plot widget with full controls
        detailed_plot = pg.PlotWidget()
        detailed_plot.setAspectLocked(False)

        # Enable full mouse controls for panning and zooming
        detailed_plot.setMouseEnabled(x=True, y=True)

        # Get the original image data
        original_image_item = self.image_items[image_number]
        image_data = original_image_item.image

        # Create new image item for the detailed view
        detailed_image_item = pg.ImageItem(image_data)

        # Set image bounds
        x_min, x_max = self.limitsInboardOutboardDownUp[0], self.limitsInboardOutboardDownUp[1]
        y_min, y_max = self.limitsInboardOutboardDownUp[2], self.limitsInboardOutboardDownUp[3]
        detailed_image_item.setRect(pg.QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min))

        # Set color map and levels with log normalization
        log_data = np.log10(np.maximum(image_data, self.contrastLimits[0]))
        detailed_image_item.setImage(log_data)

        # Set levels for log-transformed data
        log_min = np.log10(self.contrastLimits[0])
        log_max = np.log10(self.contrastLimits[1])
        detailed_image_item.setLevels([log_min, log_max])

        colormap = pg.colormap.get("RdYlBu_r", source="matplotlib")
        detailed_image_item.setColorMap(colormap)

        # Add image to plot
        detailed_plot.addItem(detailed_image_item)

        # Set title and labels
        title = f"Image {image_number}"
        detailed_plot.setTitle(title, size="14pt", color="k")
        detailed_plot.setLabel("bottom", "Outboard-inboard (mm)")
        detailed_plot.setLabel("left", "Down-up (mm)")

        # Set axis limits
        detailed_plot.setXRange(x_min, x_max)
        detailed_plot.setYRange(y_min, y_max)

        # Style axes
        for axis_name in ["left", "bottom", "top", "right"]:
            axis = detailed_plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(color="k", width=1))

        # Add plot to layout
        layout.addWidget(detailed_plot)

        # Show the popup
        popup.show()

    def _handle_plot_click(self, plot_widget):
        """Handle click on a plot widget for image selection."""
        # Find the image number for this plot widget
        image_number = None
        for img_num, pw in self.plot_widgets.items():
            if pw == plot_widget:
                image_number = img_num
                break

        if image_number is not None:
            # Toggle selection
            if image_number in self.selected_images:
                print(f"Removing image {image_number} from selection")
                self.selected_images.remove(image_number)
                self._unhighlight_plot(plot_widget)
            else:
                print(f"Adding image {image_number} to selection")
                self.selected_images.add(image_number)
                self._highlight_plot(plot_widget)

            # Update button text
            if hasattr(self, "plot_controls"):
                count = len(self.selected_images)
                if count == 0:
                    self.plot_controls.select_button.setText("Use Selected Images")
                else:
                    self.plot_controls.select_button.setText(f"Use Selected Images ({count})")

    def _grid_container_mouse_press(self, event):
        """Track mouse press events on grid container."""
        print(f"grid_container.mousePressEvent: pos={event.pos()}")
        event.ignore()  # Let event continue to children

    def _grid_container_mouse_release(self, event):
        """Track mouse release events on grid container."""
        print(f"grid_container.mouseReleaseEvent: pos={event.pos()}")
        event.ignore()  # Let event continue to children

    def _grid_layout_mouse_press(self, event):
        """Track mouse press events on grid layout."""
        print(f"grid_layout.mousePressEvent: pos={event.pos()}")
        event.ignore()  # Let event continue to children

    def _grid_layout_mouse_release(self, event):
        """Track mouse release events on grid layout."""
        print(f"grid_layout.mouseReleaseEvent: pos={event.pos()}")
        event.ignore()  # Let event continue to children

    def _plot_widget_mouse_press(self, event, plot_widget):
        """Handle mouse press events on plot widgets."""
        if event.button() == Qt.LeftButton:
            # Store the press position and start time for click detection
            plot_widget._press_pos = event.pos()
            plot_widget._press_time = event.timestamp()
            plot_widget._has_moved = False
            # Let event pass through for panning
            event.ignore()
        else:
            event.ignore()

    def _plot_widget_mouse_double_click(self, event, plot_widget):
        """Handle double-click events on plot widgets to open popup."""
        print("plot_widget_mouse_double_click")
        if event.button() == Qt.LeftButton:
            self._open_plot_popup(plot_widget)
            event.accept()
        else:
            event.ignore()

    def _plot_widget_mouse_move(self, event, plot_widget):
        """Handle mouse move events on plot widgets."""
        if hasattr(plot_widget, "_press_pos") and plot_widget._press_pos is not None:
            # Check if mouse has moved significantly from press position
            delta = (event.pos() - plot_widget._press_pos).manhattanLength()
            if delta > 5:  # Threshold for considering it a move vs click
                plot_widget._has_moved = True
        event.ignore()

    def _plot_widget_mouse_release(self, event, plot_widget):
        """Handle mouse release events on plot widgets."""
        if event.button() == Qt.LeftButton:
            if hasattr(plot_widget, "_has_moved") and not plot_widget._has_moved:
                # This was a click, not a drag
                if hasattr(self, "plot_controls") and self.plot_controls.selecting_mode:
                    # We're in selection mode, handle image selection
                    self._handle_plot_click(plot_widget)
                else:
                    # Not in selection mode, let event pass through for panning
                    event.ignore()
            else:
                # This was a drag, let event pass through for panning
                event.ignore()

            # Clear the press state
            plot_widget._press_pos = None
            plot_widget._press_time = None
            plot_widget._has_moved = False
        else:
            event.ignore()

    def eventFilter(self, obj, event):
        """Event filter for smooth panning of the scroll area."""
        # if obj == self.scroll_area.viewport():
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                self.scroll_area.viewport().setCursor(Qt.ClosedHandCursor)
                self._drag_start_pos = event.pos()
                self._dragging = True
                return True
        elif event.type() == QEvent.MouseMove:
            if hasattr(self, "_dragging") and self._dragging:
                # print("MouseMove")
                delta = event.pos() - self._drag_start_pos
                self.scroll_area.horizontalScrollBar().setValue(
                    self.scroll_area.horizontalScrollBar().value() - delta.x()
                )
                self.scroll_area.verticalScrollBar().setValue(
                    self.scroll_area.verticalScrollBar().value() - delta.y()
                )
                self._drag_start_pos = event.pos()
                return True
        elif event.type() == QEvent.MouseButtonRelease:
            if event.button() == Qt.LeftButton:
                self.scroll_area.viewport().setCursor(Qt.OpenHandCursor)
                self._dragging = False
                return True
        return super().eventFilter(obj, event)

    def draw(self):
        """Override parent class draw method - no-op for PyQtGraph."""
        pass

    def _do_draw(self):
        """Override parent class _do_draw method - no-op for PyQtGraph."""
        pass


class PyQtGraphSpiral(ImageGridDisplay):
    def setup_models(self):
        super().setup_models()

    def _create_plot_widget(self):
        return PyQtGraphSpiralWidget(self.run_list_model)
