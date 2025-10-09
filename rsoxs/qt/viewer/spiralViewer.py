from nbs_viewer.views.display.plotDisplay import ImageGridDisplay
from nbs_viewer.views.plot.imageGridWidget import ImageGridWidget
from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import (
    QListView,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QPushButton,
    QHBoxLayout,
)
from qtpy.QtCore import Qt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib
import numpy as np
import PyHyperScattering as phs


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

        layout.addWidget(self.select_button)
        layout.addStretch()

    def _toggle_selection_mode(self):
        """Toggle between selection mode and normal mode."""
        self.selecting_mode = self.select_button.isChecked()

        if self.selecting_mode:
            self.select_button.setText("Use Selected Images")
            if self.spiral_widget:
                self.spiral_widget.enable_image_selection()
        else:
            self.select_button.setText("Select Best Image")
            if self.spiral_widget:
                self.spiral_widget.disable_image_selection()

    def set_spiral_widget(self, spiral_widget):
        """Set the spiral widget to control."""
        self.spiral_widget = spiral_widget


class SpiralWidget(ImageGridWidget):
    data_ready = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmap = matplotlib.colormaps["RdYlBu_r"]
        self.contrastLimits = [1e-1, 1e6]
        self.limitsInboardOutboardDownUp = [0, 61.7, 0, 61.4]
        self.pyHyperLoader = None
        if len(self.run_list_model.available_runs) > 0:
            run = self.run_list_model.available_runs[0]
            self.pyHyperLoader = phs.load.SST1RSoXSDB(catalog=run._catalog)
        self.data_arrays = {}
        self.selected_images = set()
        self.original_spine_colors = {}

    def _create_plot_controls(self):
        """Create the best image selector control widget."""
        self.plot_controls = BestImageSelector()
        self.plot_controls.set_spiral_widget(self)
        return self.plot_controls

    def _connect_signals(self):
        super()._connect_signals()
        self.run_list_model.run_added.connect(self._on_run_added)
        self.data_ready.connect(self._update_grid)

    def _setup_ui(self):
        """Set up the user interface with scrollable canvas."""
        # Main layout - grid and paging controls
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create single figure and canvas
        self.figure = Figure(figsize=(12, 12), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)

        # Set canvas size policy to allow it to grow beyond the scroll area
        self.canvas.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # Create navigation toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # Create scroll area for the canvas
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setWidgetResizable(False)  # Don't resize widget automatically
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Create paging controls
        # self._create_paging_controls()

        # Add widgets to main layout
        main_layout.addWidget(self.toolbar)  # Toolbar at top
        main_layout.addWidget(self.scroll_area)  # Scrollable canvas
        # main_layout.addWidget(self.paging_controls)  # Paging at bottom
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
        contrastLimits = self.contrastLimits
        limitsInboardOutboardDownUp = self.limitsInboardOutboardDownUp
        logScale = True
        xMotorPositions = scan.attrs["sam_x"]
        yMotorPositions = scan.attrs["sam_y"]
        timestamps = scan.attrs["time"]

        numberRows, numberColumns = len(scan["sam_y"]), len(scan["sam_x"])
        fig = self.figure

        # Clear existing subplots
        fig.clear()

        # Set figure size based on number of subplots
        fig.set_size_inches(numberColumns * 2.3, numberRows * 2.3)

        # Create subplots
        axs = self.figure.subplots(
            numberRows,
            numberColumns,
        )
        # figsize=(3.25, 3.25) for figure
        # fig.suptitle((""), color=(0, 0, 0, 1), fontname="Calibri", size=24)
        ## Enxure axs always stays a 2D array
        if numberRows == 1 and numberColumns == 1:
            axs = np.array([[axs]])
        if numberRows == 1 and numberColumns > 1:
            axs = axs.reshape(1, numberColumns)
        if numberRows > 1 and numberColumns == 1:
            axs = axs.reshape(numberRows, 1)

        # Store axes mapping for click detection
        self.axes = {}

        for indexPlotRow, yMotorPosition in enumerate(scan["sam_y"]):
            for indexPlotColumn, xMotorPosition in enumerate(scan["sam_x"]):
                image = scan.sel(sam_y=yMotorPosition, sam_x=xMotorPosition, method="nearest")

                ## Identify index number of image in order of time
                for indexImage, timestamp in enumerate(timestamps):
                    if (
                        xMotorPositions[indexImage] == xMotorPosition
                        and yMotorPositions[indexImage] == yMotorPosition
                    ):
                        break

                ## Plot
                ax = axs[indexPlotRow, indexPlotColumn]
                # Store the axis with its image number for click detection
                self.axes[ax] = indexImage

                ax.set_title(
                    (
                        "Image "
                        + str(indexImage)
                        + ", x = "
                        + str(float(xMotorPosition))
                        + ", y = "
                        + str(float(yMotorPosition))
                    ),
                    color=(0, 0, 0, 1),
                    size=10,
                )
                if logScale:
                    ax.imshow(
                        image,
                        extent=[
                            limitsInboardOutboardDownUp[0],
                            limitsInboardOutboardDownUp[1],
                            limitsInboardOutboardDownUp[3],
                            limitsInboardOutboardDownUp[2],
                        ],
                        cmap=matplotlib.colormaps["RdYlBu_r"],
                        norm=LogNorm(vmin=contrastLimits[0], vmax=contrastLimits[1]),
                    )  ## If needed, a pedestal could be added for better viewing of log-scale images
                else:
                    ax.imshow(
                        image,
                        extent=[
                            limitsInboardOutboardDownUp[0],
                            limitsInboardOutboardDownUp[1],
                            limitsInboardOutboardDownUp[3],
                            limitsInboardOutboardDownUp[2],
                        ],
                        cmap=matplotlib.colormaps["RdYlBu_r"],
                        vmin=contrastLimits[0],
                        vmax=contrastLimits[1],
                    )
                self._style_axes(ax, indexPlotRow, indexPlotColumn, numberRows, numberColumns)

        self.figure.tight_layout()  ## Ensures that subplots don't overlap

        # Update canvas size to match the figure content
        self.canvas.draw()

        # Resize canvas to match figure size
        fig_width, fig_height = fig.get_size_inches()
        dpi = fig.get_dpi()
        canvas_width = int(fig_width * dpi)
        canvas_height = int(fig_height * dpi)
        self.canvas.resize(canvas_width, canvas_height)

        # Update scroll area to show scrollbars if needed
        if hasattr(self, "scroll_area"):
            self.scroll_area.updateGeometry()

    def resizeEvent(self, event):
        """Handle resize events with scroll area."""
        super().resizeEvent(event)
        # Update scroll area when widget is resized
        if hasattr(self, "scroll_area"):
            self.scroll_area.updateGeometry()

    def _create_artist(self, ax, image_or_shape):
        extent = [
            self.limitsInboardOutboardDownUp[0],
            self.limitsInboardOutboardDownUp[1],
            self.limitsInboardOutboardDownUp[3],
            self.limitsInboardOutboardDownUp[2],
        ]
        if isinstance(image_or_shape, np.ndarray):
            artist = ax.imshow(
                image_or_shape,
                extent=extent,
                cmap=self.cmap,
                aspect="auto",
                norm=LogNorm(vmin=self.contrastLimits[0], vmax=self.contrastLimits[1]),
            )
        else:
            artist = ax.imshow(
                np.ones(image_or_shape),
                extent=extent,
                cmap=self.cmap,
                aspect="auto",
                norm=LogNorm(vmin=self.contrastLimits[0], vmax=self.contrastLimits[1]),
            )
        return artist

    def _style_axes(self, ax, row_idx=None, col_idx=None, total_rows=None, total_cols=None):
        color = (0, 0, 0, 1)
        size = 10

        # Only show x-axis labels on bottom row
        if row_idx == total_rows - 1 or total_rows == 1:
            ax.set_xlabel("Outboard-inboard (mm)", size=size, color=color)
        else:
            ax.set_xticklabels([])

        # Only show y-axis labels on leftmost column
        if col_idx == 0 or total_cols == 1:
            ax.set_ylabel("Down-up (mm)", size=size, color=color)
        else:
            ax.set_yticklabels([])

        ax.set_xscale("linear")
        ax.set_yscale("linear")
        for border in ["top", "bottom", "left", "right"]:
            ax.spines[border].set_linewidth(1)
            ax.spines[border].set_color(color)
        ax.tick_params(colors=color, width=1)
        ax.set_xlim([self.limitsInboardOutboardDownUp[0], self.limitsInboardOutboardDownUp[1]])
        ax.set_ylim([self.limitsInboardOutboardDownUp[2], self.limitsInboardOutboardDownUp[3]])

    def enable_image_selection(self):
        """Enable image selection mode."""
        if hasattr(self, "figure"):
            self.canvas.mpl_connect("button_press_event", self._on_plot_click)
            # Clear any previous selections
            self.selected_images.clear()
            self._clear_highlights()

    def disable_image_selection(self):
        """Disable image selection mode."""
        if hasattr(self, "figure"):
            self.canvas.mpl_disconnect("button_press_event")
            # Print selected images and clear highlights
            if self.selected_images:
                selected_list = sorted(list(self.selected_images))
                print(f"Selected images: {selected_list}")
            self._clear_highlights()
            self.selected_images.clear()

    def _on_plot_click(self, event):
        """Handle clicks on subplots during selection mode."""
        print(f"_on_plot_click: {event}")
        if not event.inaxes:
            return

        # Check if the clicked axis is in our mapping
        if hasattr(self, "axes") and event.inaxes in self.axes:
            image_number = self.axes[event.inaxes]

            # Toggle selection
            if image_number in self.selected_images:
                print(f"Removing image {image_number} from selection")
                self.selected_images.remove(image_number)
                self._unhighlight_axis(event.inaxes)
            else:
                print(f"Adding image {image_number} to selection")
                self.selected_images.add(image_number)
                self._highlight_axis(event.inaxes)

            # Update button text to show count
            if hasattr(self, "plot_controls"):
                print(f"Updating button text")
                count = len(self.selected_images)
                if count == 0:
                    self.plot_controls.select_button.setText("Use Selected Images")
                else:
                    self.plot_controls.select_button.setText(f"Use Selected Images ({count})")
            print(f"Done")

    def _highlight_axis(self, ax):
        """Highlight an axis by changing its border color."""
        if ax not in self.original_spine_colors:
            # Store original colors
            self.original_spine_colors[ax] = {}
            for spine_name in ["top", "bottom", "left", "right"]:
                self.original_spine_colors[ax][spine_name] = ax.spines[spine_name].get_edgecolor()

        # Set highlight color (bright red)
        highlight_color = (1, 0, 0, 1)  # Red
        for spine_name in ["top", "bottom", "left", "right"]:
            ax.spines[spine_name].set_color(highlight_color)
            ax.spines[spine_name].set_linewidth(3)

        # Only redraw this specific axis
        ax.figure.canvas.draw_idle()

    def _unhighlight_axis(self, ax):
        """Remove highlighting from an axis."""
        if ax in self.original_spine_colors:
            # Restore original colors
            for spine_name in ["top", "bottom", "left", "right"]:
                if spine_name in self.original_spine_colors[ax]:
                    original_color = self.original_spine_colors[ax][spine_name]
                    ax.spines[spine_name].set_color(original_color)
                    ax.spines[spine_name].set_linewidth(1)

            # Only redraw this specific axis
            ax.figure.canvas.draw_idle()

    def _clear_highlights(self):
        """Clear all highlights and restore original colors."""
        for ax in self.original_spine_colors:
            self._unhighlight_axis(ax)
        self.original_spine_colors.clear()
        if hasattr(self, "canvas"):
            self.canvas.draw_idle()

    def _clear_grid(self):
        if hasattr(self, "figure"):
            self.figure.clear()
            # Reset figure to default size when clearing
            self.figure.set_size_inches(12, 12)
            self.canvas.draw()
            # Reset canvas size
            self.canvas.resize(1200, 1200)  # 12 inches * 100 dpi

        # Clear plot data models
        for plot_data in self.plotArtists.values():
            plot_data.remove_artist_from_axes()

        # Clear axes reference and selection state
        if hasattr(self, "axes"):
            self.axes.clear()
        self.selected_images.clear()
        self._clear_highlights()

        # Update scroll area size after clearing
        if hasattr(self, "scroll_area"):
            self.scroll_area.updateGeometry()


class SpiralViewer(ImageGridDisplay):
    def setup_models(self):
        super().setup_models()

    def _create_plot_widget(self):
        return SpiralWidget(self.run_list_model)
