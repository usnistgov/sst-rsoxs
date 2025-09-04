from nbs_viewer.views.display.plotDisplay import ImageGridDisplay
from nbs_viewer.views.plot.imageGridWidget import ImageGridWidget
from qtpy.QtCore import QThread, Signal, QEvent, QTimer
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
    QLabel,
    QFileDialog,
    QApplication,
)
from qtpy.QtCore import Qt
import pyqtgraph as pg
from pyqtgraph import ImageItem
import numpy as np
import PyHyperScattering as phs
from rsoxs.configuration_setup.configuration_load_save_sanitize import (
    save_configuration_spreadsheet_local,
    load_configuration_spreadsheet_local,
)
import os
import copy
import tracemalloc
from nbs_viewer.utils import print_debug


def pick_locations_from_spirals(
    locations_selected_indices,
    scan_spiral,
    configuration,
):
    """
    Intended to be an updated, data-security-compliant version of `resolve_spirals` function (used pre-2025).
    This workflow is meant to run in a python notebook.
    Unlike the legacy `resolve_spirals` function, the current function is intended to run for a single sample and can be rerun for all samples individually.  This way, not all spirals have to be resolved in one go.

    Parameters:
        locations_selected_indices: list of int values
            List of one or multiple locations on the sample.
        scan_spiral: xarray.dataarray
            Scan data that has been formatted using PyHyperScattering with dims = ["time"].
        configuration: list of dictionaries
            Configuration variable with the format of `rsoxs_config["bar"]` in which sample positions will be updated.

    Returns:
        configuration: list of dictionaries
            Configuration variable with the format of `rsoxs_config["bar"]` in which sample positions have been updated.
    """

    sample_id = scan_spiral.attrs["sample_name"]
    locations_outboard_inboard = scan_spiral.attrs["sam_x"]
    locations_down_up = scan_spiral.attrs["sam_y"]
    locations_upstream_downstream = scan_spiral.attrs["sam_z"]
    locations_theta = scan_spiral.attrs["sam_th"]

    ## Find the sample to update location
    ## TODO: probably better to deep copy configuration and search through the copy while updating the original configuration?
    for index_configuration, sample in enumerate(configuration):
        if sample["sample_id"] == sample_id:
            print(f"Found sample {sample_id} in configuration")
            # location_initial = sample["location"]
            for index_location_selected_indices, location_selected_indices in enumerate(
                locations_selected_indices
            ):
                # location_new_formatted = copy.deepcopy(location_initial)
                location_new_formatted = [
                    {"motor": "x", "position": locations_outboard_inboard[location_selected_indices]},
                    {"motor": "y", "position": locations_down_up[location_selected_indices]},
                    {"motor": "th", "position": locations_theta},
                    {"motor": "z", "position": locations_upstream_downstream},
                ]
                if index_location_selected_indices == 0:
                    sample["location"] = location_new_formatted
                else:
                    sample_new = copy.deepcopy(sample)
                    sample_new["location"] = location_new_formatted

                    sample_new["sample_name"] += f"_{index_location_selected_indices}"
                    sample_new["sample_id"] += f"_{index_location_selected_indices}"
                    configuration.append(sample_new)
            break  ## Exit after the sample is found and do not spend time looking through the other samples
    return configuration


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

    def __init__(self, pyhyper_data, scan_id, parent=None, contrastLimits=[1e-1, 1e6]):
        super().__init__(parent)
        self.pyhyper_data = pyhyper_data
        self.scan_id = scan_id
        self.contrastLimits = contrastLimits

    def run(self):
        try:
            data = self.pyhyper_data.loadRun(self.scan_id, dims=["sam_x", "sam_y"]).unstack("system")

            # Optimize datatype to save memory
            data = self._log_data(data)

            self.data_ready.emit(data)
        except Exception as e:
            error_msg = f"Error fetching plot data: {str(e)}"
            print_debug("PyHyperWorker", error_msg, category="DEBUG_PLOTS")
            self.error_occurred.emit(error_msg)

    def _log_data(self, data):
        data = data.astype(np.float32)
        data = np.log10(np.maximum(data, self.contrastLimits[0]))
        return data

    def _optimize_datatype(self, data):
        """
        Optimize the datatype of the data to use the smallest appropriate unsigned integer type.

        Parameters:
            data: xarray.DataArray with image data

        Returns:
            data: xarray.DataArray with optimized datatype
        """
        # Get the actual data values
        if hasattr(data, "values"):
            values = data.values
        else:
            values = data

        # Find the actual maximum value in the dataset
        actual_max = np.max(values)

        print(f"Original datatype: {values.dtype}")
        print(f"Actual maximum value in dataset: {actual_max}")

        # Determine the appropriate unsigned integer type
        if actual_max <= np.iinfo(np.uint8).max:
            target_dtype = np.uint8
            print(f"Optimizing to uint8 (max: {np.iinfo(np.uint8).max})")
        elif actual_max <= np.iinfo(np.uint16).max:
            target_dtype = np.uint16
            print(f"Optimizing to uint16 (max: {np.iinfo(np.uint16).max})")
        elif actual_max <= np.iinfo(np.uint32).max:
            target_dtype = np.uint32
            print(f"Optimizing to uint32 (max: {np.iinfo(np.uint32).max})")
        else:
            # If max value exceeds uint32, keep original type but warn
            print(f"Warning: Maximum value {actual_max} exceeds uint32 range, keeping original datatype")
            return data

        # Calculate memory savings
        original_size = values.nbytes
        optimized_size = values.astype(target_dtype).nbytes
        savings_mb = (original_size - optimized_size) / (1024 * 1024)

        print(f"Memory savings: {savings_mb:.2f} MB ({original_size} -> {optimized_size} bytes)")

        # Cast the data to the optimized type
        if hasattr(data, "astype"):
            # For xarray DataArray
            optimized_data = data.astype(target_dtype)
        else:
            # For numpy array
            optimized_data = data.astype(target_dtype)

        print(f"Optimized datatype: {optimized_data.dtype}")

        return optimized_data


class BestImageSelector(QWidget):
    """Control widget for selecting the best image from spiral plots."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selecting_mode = False
        self.spiral_widget = None
        self.configuration = None
        self.configuration_file = None
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.load_configuration_button = QPushButton("Load File")
        self.configuration_label = QLabel("File: None")
        self.load_configuration_button.clicked.connect(self._load_configuration)
        self.select_button = QPushButton("Select Best Image")
        self.select_button.setCheckable(True)
        self.select_button.clicked.connect(self._toggle_selection_mode)

        self.clear_button = QPushButton("Clear Selection")
        self.clear_button.clicked.connect(self._clear_selection)
        self.clear_button.setEnabled(False)

        self.next_run_button = QPushButton("Next Run")
        self.next_run_button.clicked.connect(self._next_run)
        self.previous_run_button = QPushButton("Previous Run")
        self.previous_run_button.clicked.connect(self._previous_run)
        self.save_configuration_button = QPushButton("Save Configuration")
        self.save_configuration_button.clicked.connect(self.save_configuration)

        layout1 = QHBoxLayout()
        layout2 = QHBoxLayout()
        layout3 = QHBoxLayout()

        layout1.addWidget(self.load_configuration_button)
        layout1.addWidget(self.configuration_label)

        layout2.addWidget(self.select_button)
        layout2.addWidget(self.clear_button)

        layout3.addWidget(self.previous_run_button)
        layout3.addWidget(self.next_run_button)

        layout.addLayout(layout1)
        layout.addLayout(layout2)
        layout.addLayout(layout3)
        layout.addWidget(self.save_configuration_button)

    def _load_configuration(self):
        filepicker = QFileDialog()
        filepicker.setNameFilter("Configuration files (*.xlsx)")
        filepicker.setFileMode(QFileDialog.FileMode.ExistingFile)
        filepicker.setWindowTitle("Load File")
        if filepicker.exec_():
            filename = os.path.basename(filepicker.selectedFiles()[0])
            self.configuration_label.setText(f"File: {filename}")
            self.configuration_file = filepicker.selectedFiles()[0]
            self.configuration = load_configuration_spreadsheet_local(self.configuration_file)

    def _clear_selection(self):
        """Clear the selection."""
        self.spiral_widget.clear_image_selection()
        self.select_button.setText("Use Selected Images")

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
                self.update_configuration()
                self.spiral_widget.disable_image_selection()

    def set_spiral_widget(self, spiral_widget):
        """Set the spiral widget to control."""
        self.spiral_widget = spiral_widget

    def update_configuration(self):
        if self.spiral_widget.selected_images:
            scan_id = self.spiral_widget.active_run.scan_id
            print(f"update_configuration for {scan_id}")

            scan = self.spiral_widget.data_arrays[scan_id]
            self.configuration = pick_locations_from_spirals(
                self.spiral_widget.selected_images, scan, self.configuration
            )

    def save_configuration(self):
        directory = os.path.dirname(self.configuration_file)
        print(f"Saving configuration to {directory}/SpiralSpotsPicked.xlsx")
        save_configuration_spreadsheet_local(
            self.configuration, file_label="SpiralSpotsPicked", file_path=directory
        )

    def _next_run(self):
        was_selecting = self.selecting_mode
        if was_selecting:
            self.update_configuration()
            self.spiral_widget.disable_image_selection()
        self.spiral_widget.advance_run()
        if was_selecting:
            self.spiral_widget.enable_image_selection()

    def _previous_run(self):
        was_selecting = self.selecting_mode
        if was_selecting:
            self.update_configuration()
            self.spiral_widget.disable_image_selection()
        self.spiral_widget.previous_run()
        if was_selecting:
            self.spiral_widget.enable_image_selection()


def memory_print(banner="Memory usage"):
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    print(banner)

    print("\n[ Top 10 memory users ]")
    for i, stat in enumerate(top_stats[:10]):
        if stat.size < 5 * 1024 * 1024:
            break
        print(f"{i+1}. {stat.count} blocks: {stat.size / 1024 / 1024:.1f} MiB")
        for line in stat.traceback.format():
            print(f"    {line}")


class SimpleCache(dict):
    def __init__(self, *args, max_size_bytes: int = 1e9, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size_bytes = max_size_bytes
        self.current_size = 0
        self.last_accessed = []

    def __getitem__(self, key, *args, **kwargs):
        if key in self.last_accessed:
            self.last_accessed.remove(key)
        if key in self:
            self.last_accessed.append(key)
        return super().__getitem__(key, *args, **kwargs)

    def __setitem__(self, key, value, *args, **kwargs):
        if key in self:
            self.current_size -= self[key].nbytes

        super().__setitem__(key, value, *args, **kwargs)
        self.current_size += value.nbytes
        if key in self.last_accessed:
            self.last_accessed.remove(key)
        self.last_accessed.append(key)
        if self.current_size > self.max_size_bytes:
            self.evict()

    def clear(self):
        super().clear()
        self.current_size = 0

    def get_size(self):
        return self.current_size

    def evict(self):
        while self.current_size > self.max_size_bytes and len(self.last_accessed) > 1:
            key = self.last_accessed.pop(0)
            del self[key]
            print("Evicting key: ", key)

    def __delitem__(self, key, *args, **kwargs):
        if key in self:
            self.current_size -= self[key].nbytes
            if key in self.last_accessed:
                self.last_accessed.remove(key)
        super().__delitem__(key, *args, **kwargs)

    def get_stats(self):
        return {
            "size": self.current_size,
            "max_size": self.max_size_bytes,
        }


class PyQtGraphSpiralWidget(ImageGridWidget):
    data_ready = Signal()

    def __init__(self, *args, **kwargs):
        # Start memory tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        self.data_arrays = SimpleCache(max_size_bytes=2 * 1024**3)
        self.selected_images = set()
        self.plot_widgets = {}  # Store individual plot widgets
        self.image_items = {}  # Store image items for highlighting
        self.original_image_data = {}  # Store original image data for popup creation
        self.signal_connections = {}  # Track signal connections for proper cleanup
        self.index_to_coords = {}  # Store image index to coordinates
        self.active_run = None
        # Override parent class methods that expect matplotlib canvas
        self.canvas = None  # No matplotlib canvas in PyQtGraph version
        # PyQtGraph configuration
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.contrastLimits = [1e-1, 1e6]
        self.limitsInboardOutboardDownUp = [0, 61.7, 0, 61.4]
        self.pyHyperLoader = None
        super().__init__(*args, **kwargs)

        memory_print(f"=== BASELINE MEMORY USAGE (PyQtGraphSpiralWidget initialized) ===")

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
        self.grid_layout.setSpacing(10)  # Consistent spacing between plots
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
            # If the run is visible, or next to a visible row, start the worker
            self._prefetch_data(run)

    def _prefetch_data(self, run):
        print(f"Prefetching data for run {run.scan_id}")
        uid = run.uid
        is_visible = uid in self.run_list_model.visible_runs

        if is_visible:
            self.start_pyhyper_worker(run)

        siblings = self.run_list_model.get_siblings_of_run(run)
        for sibling in siblings:
            if sibling is None:
                continue
            print(f"Prefetching data for sibling {sibling.scan_id}")
            sibling_uid = sibling.uid
            sibling_is_visible = sibling_uid in self.run_list_model.visible_runs
            if sibling_is_visible:
                print(f"Prefetching data for {run.scan_id} because the sibling {sibling.scan_id} is visible")
                self.start_pyhyper_worker(run)
            elif is_visible:
                print(f"Prefetching data for sibling {sibling.scan_id} because the run {run.scan_id} is visible")
                self.start_pyhyper_worker(sibling)

    def _on_visible_runs_changed(self, visible_runs):
        for uid in visible_runs:
            run = self.run_list_model._run_models[uid]
            self._prefetch_data(run)
        self._update_grid()

    def advance_run(self):
        current_run = self.active_run
        if current_run is None:
            run = self.run_list_model.get_first_run()
            self.run_list_model.set_run_visible(run, True)
            return
        siblings = self.run_list_model.get_siblings_of_run(current_run)
        next_run = siblings[1]
        if next_run is not None:
            self.run_list_model.set_run_visible(next_run, True)
            return

    def previous_run(self):
        current_run = self.active_run
        if current_run is None:
            run = self.run_list_model.get_first_run()
            self.run_list_model.set_run_visible(run, True)
            return
        siblings = self.run_list_model.get_siblings_of_run(current_run)
        previous_run = siblings[0]
        if previous_run is not None:
            self.run_list_model.set_run_visible(previous_run, True)
            return

    def start_pyhyper_worker(self, run):
        if run.scan_id not in self.data_arrays and run.scan_id not in self.workers:
            self._start_pyhyper_worker(run)

    def _start_pyhyper_worker(self, run):
        print(f"Starting worker for run {run.scan_id}")
        memory_print(f"=== STARTING WORKER FOR RUN {run.scan_id} ===")
        if self.pyHyperLoader is None:
            if hasattr(run, "run"):
                catalog = run.run._catalog
            else:
                catalog = run._catalog
            self.pyHyperLoader = phs.load.SST1RSoXSDB(catalog=catalog)
        worker = PyHyperWorker(self.pyHyperLoader, run.scan_id, self, contrastLimits=self.contrastLimits)
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

        # Memory tracking after data load
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        self.data_ready.emit()

    def _handle_error(self, error_msg):
        """Handle errors emitted by background workers.

        Shows a non-blocking warning and logs the error for debugging.
        """
        print_debug("PyQtGraphSpiralWidget", f"Worker error: {error_msg}", category="DEBUG_PLOTS")
        try:
            QMessageBox.warning(self, "Data load error", error_msg)
        except Exception:
            # If GUI is not available or warning fails, fallback to printing
            print(f"Data load error: {error_msg}")

    def _update_grid(self):
        # memory_print(f"=== UPDATING GRID ===")
        if len(self.run_list_model.visible_models) == 0:
            print("No visible runs")
            self.active_run = None
            return

        run = self.run_list_model.visible_models[0]
        if self.active_run == run:
            print("Already displaying this run")
            return
        self._clear_grid()
        if run.scan_id in self.data_arrays:
            scan = self.data_arrays[run.scan_id]
            print(f"Data for run {run.scan_id} loaded")
        else:
            print(f"Safely starting worker for run {run.scan_id}")
            self.start_pyhyper_worker(run)
            return
        # memory_print(f"=== CLEARED GRID ===")
        self._create_subplots_for_page(scan)
        memory_print(f"=== CREATED SUBPLOTS FOR PAGE ===")
        self.active_run = run

    def _create_subplots_for_page(self, scan):
        """Create PyQtGraph subplots for the spiral data."""
        xMotorPositions = scan.attrs["sam_x"]
        yMotorPositions = scan.attrs["sam_y"]
        timestamps = scan.attrs["time"]

        numberRows, numberColumns = scan.attrs["start"]["shape"]
        yidx = np.argsort(yMotorPositions, kind="stable")
        colgroups = [yidx[i : i + numberColumns] for i in range(0, len(yidx), numberColumns)]
        colidx = [np.argsort(xMotorPositions[colgroup], kind="stable") for colgroup in colgroups]
        sortidx = [colgroup[colidx[i]] for i, colgroup in enumerate(colgroups)]
        sortidx = np.concatenate(sortidx)
        # Clear existing grid
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        self.plot_widgets.clear()
        self.image_items.clear()
        self.index_to_coords.clear()

        for n, indexImage in enumerate(sortidx):
            yMotorPosition = yMotorPositions[indexImage]
            xMotorPosition = xMotorPositions[indexImage]
            indexPlotRow = n // numberRows
            indexPlotColumn = n % numberColumns
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

            # Create image item using common method
            # image_data = np.array(image)
            image_item = self._create_image_item(
                scan.sel(sam_y=yMotorPosition, sam_x=xMotorPosition, method="nearest").data
            )

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
            # Store original image data for popup creation
            # self.original_image_data[indexImage] = image_data

            # Override plot widget mouse events for panning and selection
            plot_widget.mousePressEvent = lambda event, pw=plot_widget: self._plot_widget_mouse_press(event, pw)
            plot_widget.mouseReleaseEvent = lambda event, pw=plot_widget: self._plot_widget_mouse_release(
                event, pw
            )
            plot_widget.mouseMoveEvent = lambda event, pw=plot_widget: self._plot_widget_mouse_move(event, pw)
            plot_widget.mouseDoubleClickEvent = lambda event, pw=plot_widget: self._plot_widget_mouse_double_click(
                event, pw
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
        """Clear the grid rapidly by hiding widgets first, then cleaning up in background."""
        print("Clearing grid")

        # Store widgets for cleanup before removing from layout
        widgets_to_cleanup = []

        # First, hide all widgets immediately for instant visual feedback
        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(i)
            if item.widget():
                widget = item.widget()
                widget.hide()  # Instant visual removal
                widgets_to_cleanup.append(widget)  # Store reference BEFORE removing

        # Clear the layout items (this is fast)
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            # Don't delete the widget here, just remove from layout

        print("Disconnecting signal connections")
        # Disconnect all signal connections before clearing
        for image_index, image_item in self.image_items.items():
            if image_index in self.signal_connections:
                try:
                    handler = self.signal_connections[image_index]
                    image_item.sigMouseClicked.disconnect(handler)
                except:
                    pass  # Already disconnected or item doesn't exist

        print("Clearing dictionaries")
        # Clear references immediately
        self.plot_widgets.clear()
        self.image_items.clear()
        self.original_image_data.clear()
        self.index_to_coords.clear()
        self.selected_images.clear()
        self.signal_connections.clear()

        # Schedule widget cleanup in background using stored references
        QTimer.singleShot(100, lambda w=widgets_to_cleanup: self._cleanup_widgets_in_background(w))

        print("Clear Done")

    def _cleanup_widgets_in_background(self, widgets):
        """Remove widgets in background to avoid blocking UI."""
        # Now we have the stored widget references
        print("Cleaning up widgets in background")
        for widget in widgets:
            if widget and not widget.parent():  # Check if widget still exists
                widget.deleteLater()
                QApplication.processEvents()
        print("Widget cleanup done")

    def _create_image_item(self, image_data):
        """Create and configure an ImageItem with common settings."""
        # Create standard ImageItem with original data (no transformations)
        image_item = pg.ImageItem(image_data)

        # Set image bounds
        x_min, x_max = self.limitsInboardOutboardDownUp[0], self.limitsInboardOutboardDownUp[1]
        y_min, y_max = self.limitsInboardOutboardDownUp[2], self.limitsInboardOutboardDownUp[3]
        image_item.setRect(pg.QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min))

        # Set levels for original data
        image_item.setLevels([np.log10(self.contrastLimits[0]), np.log10(self.contrastLimits[1])])

        # Create logarithmic colormap using logspace positions
        base_colormap = pg.colormap.get("RdYlBu_r", source="matplotlib")
        image_item.setColorMap(base_colormap)
        """
        base_colors = base_colormap.getColors()
        num_colors = len(base_colors)

        print(f"Creating log colormap with {num_colors} colors")

        cvalues = np.maximum(np.linspace(0, self.contrastLimits[1], 256), self.contrastLimits[0])
        logcvals = np.log10(cvalues)
        relog = (logcvals - min(logcvals)) / (max(logcvals) - min(logcvals))
        lut = [base_colormap[v].getRgb() for v in relog]

        image_item.setLookupTable(lut)
        
        # Create logarithmically spaced positions
        # This creates positions that are spaced logarithmically between 0 and 1
        lower_lim = np.log10(1 / self.contrastLimits[1])
        log_positions = np.power(np.linspace(lower_lim, 0, num_colors), 10)  # From 0.001 to 1.0
        log_positions = (log_positions - log_positions[0]) / (
            log_positions[-1] - log_positions[0]
        )  # Normalize to [0, 1]

        # Create the logarithmic colormap
        log_colormap = pg.ColorMap(pos=log_positions, color=base_colors)
        image_item.setColorMap(log_colormap) """

        return image_item

    def _open_plot_popup(self, plot_widget):
        """Open a popup window with a larger version of the plot."""
        # Find the image number for this plot widget
        image_number = None
        for img_num, pw in self.plot_widgets.items():
            if pw == plot_widget:
                image_number = img_num
                break

        if image_number is None:
            return

        # Create popup window
        popup = QWidget(parent=self)  # Set parent to ensure proper window management
        popup.setWindowTitle(f"Image {image_number} - Detailed View")
        popup.setGeometry(100, 100, 800, 600)
        # Set window flags to make it a proper popup window
        popup.setWindowFlags(Qt.Window)

        # Create layout
        layout = QVBoxLayout(popup)

        # Create a new plot widget with full controls
        detailed_plot = pg.PlotWidget()
        detailed_plot.setAspectLocked(False)

        # Enable full mouse controls for panning and zooming
        detailed_plot.setMouseEnabled(x=True, y=True)

        # Get the original image data and create new image item
        scan = self.data_arrays[self.active_run.scan_id]
        yMotorPosition = scan.attrs["sam_y"][image_number]
        xMotorPosition = scan.attrs["sam_x"][image_number]
        detailed_image_item = self._create_image_item(
            scan.sel(sam_y=yMotorPosition, sam_x=xMotorPosition, method="nearest").data
        )

        # Add image to plot
        detailed_plot.addItem(detailed_image_item)

        # Set title and labels
        title = f"Image {image_number}"
        detailed_plot.setTitle(title, size="14pt", color="k")
        detailed_plot.setLabel("bottom", "Outboard-inboard (mm)")
        detailed_plot.setLabel("left", "Down-up (mm)")

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
        self.run_list_model.set_auto_add(False)

    def _create_plot_widget(self):
        return PyQtGraphSpiralWidget(self.run_list_model)
