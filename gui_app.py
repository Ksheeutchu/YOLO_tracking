"""
Desktop interface for the counting system (step 2: live video + Start/Stop).

Usage:
    python gui_app.py                       # webcam, yolov8m
    python gui_app.py --source video.mp4 --weights yolov8s

Install:
    pip install PySide6
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import torch
import torchvision  # noqa: F401  (ultralytics imports this lazily on the first frame;
                    # its import chain is deep, so do it here on the main thread)
from PySide6.QtCore import QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog,
                               QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

from counting_engine import CountingEngine


class VideoWorker(QThread):
    """Runs the engine loop off the GUI thread so the window never freezes."""
    frame_ready = Signal(QImage)
    error = Signal(str)

    def __init__(self, engine):
        super().__init__()
        self.setStackSize(32 * 1024 * 1024)   # macOS gives Qt threads only 512 KB by default
        self.engine = engine
        self._running = True

    def request_stop(self):
        self._running = False

    def run(self):
        try:
            self.engine.start()
        except Exception as e:                      # e.g. camera not found
            self.error.emit(str(e))
            return
        while self._running:
            frame = self.engine.read_frame()
            if frame is None:                       # end of video file
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch * w,
                         QImage.Format.Format_RGB888).copy()   # copy: own the memory
            self.frame_ready.emit(img)
        self.engine.stop()


class ClassDialog(QDialog):
    """Checklist of the model's object classes, with a search box."""

    def __init__(self, names, selected, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Objects to detect")
        self.setMinimumSize(360, 480)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search...")
        self.search.textChanged.connect(self.filter_items)

        self.list = QListWidget()
        for class_id, name in sorted(names.items()):
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, class_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checked = selected is None or class_id in selected
            item.setCheckState(Qt.CheckState.Checked if checked
                               else Qt.CheckState.Unchecked)
            self.list.addItem(item)

        select_all = QPushButton("Select all")
        select_all.clicked.connect(lambda: self.set_all(Qt.CheckState.Checked))
        clear_all = QPushButton("Clear all")
        clear_all.clicked.connect(lambda: self.set_all(Qt.CheckState.Unchecked))
        quick_row = QHBoxLayout()
        quick_row.addWidget(select_all)
        quick_row.addWidget(clear_all)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.validate)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.search)
        layout.addLayout(quick_row)
        layout.addWidget(self.list)
        layout.addWidget(buttons)

    def filter_items(self, text):
        text = text.strip().lower()
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setHidden(text not in item.text().lower())

    def set_all(self, state):
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(state)

    def selected_ids(self):
        return [self.list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.CheckState.Checked]

    def validate(self):
        if not self.selected_ids():
            QMessageBox.warning(self, "Objects to detect",
                                "Select at least one object.")
            return
        self.accept()


MODEL_NAMES = ["yolov8n", "yolov8s", "yolov8m"]     # nano / small / medium


class MainWindow(QMainWindow):
    def __init__(self, engine, start_weights="yolov8m"):
        super().__init__()
        self.engine = engine
        self.worker = None

        self.setWindowTitle("Detection + Counting")

        self.video_label = QLabel("Press Start")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored,
                                       QSizePolicy.Policy.Ignored)
        self.video_label.setStyleSheet("background-color: black; color: white;")

        self.start_button = QPushButton("Start")
        self.start_button.setMinimumHeight(40)
        self.start_button.clicked.connect(self.toggle)

        self.reset_button = QPushButton("Reset counter")
        self.reset_button.setMinimumHeight(40)
        self.reset_button.clicked.connect(self.reset_counters)

        self.save_button = QPushButton("Save data")
        self.save_button.setMinimumHeight(40)
        self.save_button.clicked.connect(self.save_data)

        button_row = QHBoxLayout()
        button_row.addWidget(self.start_button, stretch=2)
        button_row.addWidget(self.reset_button, stretch=1)
        button_row.addWidget(self.save_button, stretch=1)

        # live date and time, top-right corner
        self.clock_label = QLabel()
        self.clock_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        # model selector (only usable while the video is stopped)
        self.model_combo = QComboBox()
        models_dir = Path(__file__).parent / "models"
        for name in MODEL_NAMES:
            if (models_dir / f"{name}.pt").exists():
                self.model_combo.addItem(name, name)
        if self.model_combo.findData(start_weights) < 0:     # custom .pt path from --weights
            self.model_combo.addItem(Path(start_weights).stem, start_weights)
        self.model_combo.setCurrentIndex(self.model_combo.findData(start_weights))
        self._model_index = self.model_combo.currentIndex()
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Model:"))
        top_bar.addWidget(self.model_combo)

        self.classes_button = QPushButton()
        self.classes_button.clicked.connect(self.choose_classes)
        self.update_classes_button()
        top_bar.addWidget(self.classes_button)

        top_bar.addStretch(1)
        top_bar.addWidget(self.clock_label)

        layout = QVBoxLayout()
        layout.addLayout(top_bar)
        layout.addWidget(self.video_label, stretch=1)
        layout.addLayout(button_row)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)
        self.update_clock()

    def update_clock(self):
        self.clock_label.setText(f"{datetime.now():%d.%m.%Y  %H:%M:%S}")

    # ---- start / stop ---------------------------------------------------
    def toggle(self):
        if self.worker is None:
            self.start_video()
        else:
            self.stop_video()

    def start_video(self):
        self.worker = VideoWorker(self.engine)
        self.worker.frame_ready.connect(self.show_frame)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()
        self.start_button.setText("Stop")
        self.model_combo.setEnabled(False)

    def stop_video(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.start_button.setEnabled(False)     # until the thread has ended

    def on_worker_finished(self):
        self.worker = None
        self.video_label.clear()                    # drop the last frame
        self.video_label.setText("Press Start")
        self.start_button.setText("Start")
        self.start_button.setEnabled(True)
        self.model_combo.setEnabled(True)

    # ---- model ----------------------------------------------------------
    def on_model_changed(self, index):
        weights = self.model_combo.itemData(index)
        name = self.model_combo.itemText(index)
        self.model_combo.setEnabled(False)
        self.start_button.setEnabled(False)
        self.statusBar().showMessage(f"Loading {name}...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()                 # show the message before loading
        try:
            self.engine.load_model(weights)
            self._model_index = index
            self.engine.classes = None               # class ids differ between models
            self.update_classes_button()
            self.statusBar().showMessage(
                f"Model: {name} (objects reset to all)", 4000)
        except Exception as e:
            self.model_combo.blockSignals(True)      # go back without re-triggering
            self.model_combo.setCurrentIndex(self._model_index)
            self.model_combo.blockSignals(False)
            self.statusBar().clearMessage()
            self.show_error(f"Could not load model {name}: {e}")
        finally:
            QApplication.restoreOverrideCursor()
            self.model_combo.setEnabled(True)
            self.start_button.setEnabled(True)

    # ---- objects to detect ----------------------------------------------
    def update_classes_button(self):
        names = self.engine.model.names
        selected = self.engine.classes
        if selected is None:
            self.classes_button.setText("Objects: all")
            self.classes_button.setToolTip("Detecting every class of this model")
        else:
            self.classes_button.setText(f"Objects: {len(selected)} of {len(names)}")
            self.classes_button.setToolTip(
                ", ".join(names[c] for c in sorted(selected)))

    def choose_classes(self):
        names = self.engine.model.names
        dialog = ClassDialog(names, self.engine.classes, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        ids = dialog.selected_ids()
        # all ticked = no filter at all (None); takes effect on the next frame
        self.engine.classes = None if len(ids) == len(names) else ids
        self.update_classes_button()

    # ---- counters -------------------------------------------------------
    def reset_counters(self):
        answer = QMessageBox.question(
            self, "Reset counter",
            "Set all counters to zero?\n\n"
            "The saved file for today is not changed until you press "
            "Save data (or close the app, which also saves).")
        if answer == QMessageBox.StandardButton.Yes:
            self.engine.reset_counters()
            self.statusBar().showMessage("Counters reset", 4000)

    def save_data(self):
        try:
            path = self.engine.save()
        except Exception as e:
            self.show_error(f"Could not save: {e}")
            return
        self.statusBar().showMessage(f"Saved to {path}", 5000)

    # ---- display --------------------------------------------------------
    def show_frame(self, img):
        # Retina/HiDPI: scale to physical pixels, otherwise the OS upscales
        # the pixmap a second time and it looks blurry.
        dpr = self.video_label.devicePixelRatioF()
        size = self.video_label.size()
        target = QSize(int(size.width() * dpr), int(size.height() * dpr))
        pix = QPixmap.fromImage(img).scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        pix.setDevicePixelRatio(dpr)
        self.video_label.setPixmap(pix)

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

    # ---- closing --------------------------------------------------------
    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.request_stop()
            self.worker.wait()
        self.engine.save()                          # same as the old script: save on exit
        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                        help="0 for webcam, or path to a video file")
    parser.add_argument("--weights", default="yolov8m",
                        help="yolov8n / yolov8s / yolov8m, or a path to a .pt file")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--classes", type=int, nargs="*", default=None)
    parser.add_argument("--line", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=1280,
                        help="requested webcam width (ignored for video files)")
    parser.add_argument("--height", type=int, default=720,
                        help="requested webcam height (ignored for video files)")
    parser.add_argument("--no-mirror", action="store_true",
                        help="do not mirror the webcam image (mirroring is on by default)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    engine = CountingEngine(source=source, weights=args.weights,
                            conf=args.conf, classes=args.classes,
                            line=args.line, cam_size=(args.width, args.height),
                            mirror=not args.no_mirror)

    app = QApplication(sys.argv)
    window = MainWindow(engine, start_weights=args.weights)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
