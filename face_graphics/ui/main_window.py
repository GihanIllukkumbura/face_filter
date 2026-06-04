from __future__ import annotations

import sys
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QColorDialog,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig
from ..capture.camera_worker import CameraWorker
from ..overlay.renderer import RenderSettings


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.setWindowTitle("Face Graphics")
        self._config = config
        self._settings = RenderSettings(
            show_sunglasses=True,
            show_hat=True,
            opacity=config.overlay_opacity,
            scale=config.overlay_scale,
            rotation_deg=config.overlay_rotation_deg,
            tint_strength=config.overlay_tint_strength,
            shadow_strength=config.overlay_shadow_strength,
            gesture_rotation=config.gesture_rotation,
        )
        self._last_image: QImage | None = None
        self._tint_color_name = ""
        self._accent_color_name = ""

        self._video_label = QLabel("Starting camera...")
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setMinimumSize(640, 360)
        self._video_label.setStyleSheet("background-color: #111; color: #ddd;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._video_label)
        self.setCentralWidget(container)

        self._build_controls()

        self._worker = CameraWorker(config, self._settings)
        self._worker.frameReady.connect(self._on_frame)
        self._worker.metricsReady.connect(self._on_metrics)
        self._worker.statusReady.connect(self.statusBar().showMessage)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _build_controls(self) -> None:
        controls = QWidget()
        layout = QVBoxLayout(controls)

        self._toggle_glasses = QCheckBox("Sunglasses")
        self._toggle_glasses.setChecked(True)
        self._toggle_hat = QCheckBox("Hat")
        self._toggle_hat.setChecked(True)
        self._toggle_mustache = QCheckBox("Mustache")
        self._toggle_halo = QCheckBox("Halo")
        self._toggle_sparkles = QCheckBox("Sparkles")
        self._toggle_bow_tie = QCheckBox("Bow tie")
        self._toggle_face_frame = QCheckBox("Face frame")

        self._opacity_slider = QSlider(Qt.Horizontal)
        self._opacity_slider.setRange(30, 100)
        self._opacity_slider.setValue(int(self._settings.opacity * 100))

        self._scale_slider = QSlider(Qt.Horizontal)
        self._scale_slider.setRange(50, 150)
        self._scale_slider.setValue(int(self._settings.scale * 100))

        self._rotation_slider = QSlider(Qt.Horizontal)
        self._rotation_slider.setRange(-180, 180)
        self._rotation_slider.setValue(int(self._settings.rotation_deg))

        self._rotation_target = QComboBox()
        self._rotation_target.addItem("All objects", "all")
        self._rotation_target.addItem("Sunglasses", "sunglasses")
        self._rotation_target.addItem("Hat", "hat")
        self._rotation_target.addItem("Mustache", "mustache")
        self._rotation_target.addItem("Halo", "halo")
        self._rotation_target.addItem("Sparkles", "sparkles")
        self._rotation_target.addItem("Bow tie", "bow_tie")
        self._rotation_target.addItem("Face frame", "face_frame")

        self._offset_x_slider = QSlider(Qt.Horizontal)
        self._offset_x_slider.setRange(-50, 50)
        self._offset_x_slider.setValue(0)

        self._offset_y_slider = QSlider(Qt.Horizontal)
        self._offset_y_slider.setRange(-50, 50)
        self._offset_y_slider.setValue(0)

        self._auto_motion = QCheckBox("Auto motion")

        self._rotation_motion = QComboBox()
        self._rotation_motion.addItem("Sway", "sway")
        self._rotation_motion.addItem("Spin", "spin")
        self._rotation_motion.addItem("Pulse", "pulse")

        self._rotation_speed_slider = QSlider(Qt.Horizontal)
        self._rotation_speed_slider.setRange(10, 300)
        self._rotation_speed_slider.setValue(100)

        self._rotation_amplitude_slider = QSlider(Qt.Horizontal)
        self._rotation_amplitude_slider.setRange(0, 180)
        self._rotation_amplitude_slider.setValue(int(self._settings.rotation_amplitude))

        self._gesture_rotation = QCheckBox("Hand pinch rotation")
        self._gesture_rotation.setChecked(self._settings.gesture_rotation)

        self._gesture_sensitivity_slider = QSlider(Qt.Horizontal)
        self._gesture_sensitivity_slider.setRange(25, 250)
        self._gesture_sensitivity_slider.setValue(100)

        self._tint_button = QPushButton("Tint Color")
        self._tint_strength_slider = QSlider(Qt.Horizontal)
        self._tint_strength_slider.setRange(0, 100)
        self._tint_strength_slider.setValue(int(self._settings.tint_strength * 100))

        self._accent_button = QPushButton("Accent Color")
        self._accent_strength_slider = QSlider(Qt.Horizontal)
        self._accent_strength_slider.setRange(0, 100)
        self._accent_strength_slider.setValue(int(self._settings.accent_strength * 100))

        self._style_combo = QComboBox()
        self._style_combo.addItem("Classic", "classic")
        self._style_combo.addItem("Neon", "neon")
        self._style_combo.addItem("Pop", "pop")
        self._style_combo.addItem("Mono", "mono")

        self._effect_combo = QComboBox()
        self._effect_combo.addItem("No effect", "none")
        self._effect_combo.addItem("Soft", "soft")
        self._effect_combo.addItem("Cinematic", "cinematic")
        self._effect_combo.addItem("Cartoon", "cartoon")
        self._effect_combo.addItem("Edges", "edges")
        self._effect_combo.addItem("Grayscale", "grayscale")

        self._reset_colors_button = QPushButton("Reset Colors")
        self._clean_preset_button = QPushButton("Clean")
        self._party_preset_button = QPushButton("Party")
        self._cyber_preset_button = QPushButton("Cyber")

        self._shadow_slider = QSlider(Qt.Horizontal)
        self._shadow_slider.setRange(0, 60)
        self._shadow_slider.setValue(int(self._settings.shadow_strength * 100))

        self._snapshot_button = QPushButton("Snapshot")

        layout.addWidget(self._toggle_glasses)
        layout.addWidget(self._toggle_hat)
        layout.addWidget(self._toggle_mustache)
        layout.addWidget(self._toggle_halo)
        layout.addWidget(self._toggle_sparkles)
        layout.addWidget(self._toggle_bow_tie)
        layout.addWidget(self._toggle_face_frame)
        layout.addWidget(QLabel("Opacity"))
        layout.addWidget(self._opacity_slider)
        layout.addWidget(QLabel("Scale"))
        layout.addWidget(self._scale_slider)
        layout.addWidget(QLabel("Rotation (deg)"))
        layout.addWidget(self._rotation_slider)
        layout.addWidget(QLabel("Rotation target"))
        layout.addWidget(self._rotation_target)
        layout.addWidget(QLabel("Target X offset"))
        layout.addWidget(self._offset_x_slider)
        layout.addWidget(QLabel("Target Y offset"))
        layout.addWidget(self._offset_y_slider)
        layout.addWidget(self._auto_motion)
        layout.addWidget(QLabel("Rotation motion"))
        layout.addWidget(self._rotation_motion)
        layout.addWidget(QLabel("Rotation speed"))
        layout.addWidget(self._rotation_speed_slider)
        layout.addWidget(QLabel("Rotation amplitude"))
        layout.addWidget(self._rotation_amplitude_slider)
        layout.addWidget(self._gesture_rotation)
        layout.addWidget(QLabel("Hand sensitivity"))
        layout.addWidget(self._gesture_sensitivity_slider)
        layout.addWidget(QLabel("Style"))
        layout.addWidget(self._style_combo)
        layout.addWidget(QLabel("Camera effect"))
        layout.addWidget(self._effect_combo)
        layout.addWidget(QLabel("Tint"))
        layout.addWidget(self._tint_button)
        layout.addWidget(QLabel("Tint strength"))
        layout.addWidget(self._tint_strength_slider)
        layout.addWidget(QLabel("Accent"))
        layout.addWidget(self._accent_button)
        layout.addWidget(QLabel("Accent strength"))
        layout.addWidget(self._accent_strength_slider)
        layout.addWidget(QLabel("Shadow strength"))
        layout.addWidget(self._shadow_slider)
        layout.addWidget(self._reset_colors_button)
        layout.addWidget(QLabel("Presets"))
        layout.addWidget(self._clean_preset_button)
        layout.addWidget(self._party_preset_button)
        layout.addWidget(self._cyber_preset_button)
        layout.addWidget(self._snapshot_button)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)

        dock = QDockWidget("Controls", self)
        dock.setWidget(scroll)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self._toggle_glasses.toggled.connect(self._apply_settings)
        self._toggle_hat.toggled.connect(self._apply_settings)
        self._toggle_mustache.toggled.connect(self._apply_settings)
        self._toggle_halo.toggled.connect(self._apply_settings)
        self._toggle_sparkles.toggled.connect(self._apply_settings)
        self._toggle_bow_tie.toggled.connect(self._apply_settings)
        self._toggle_face_frame.toggled.connect(self._apply_settings)
        self._opacity_slider.valueChanged.connect(self._apply_settings)
        self._scale_slider.valueChanged.connect(self._apply_settings)
        self._rotation_slider.valueChanged.connect(self._apply_settings)
        self._rotation_target.currentIndexChanged.connect(self._apply_settings)
        self._offset_x_slider.valueChanged.connect(self._apply_settings)
        self._offset_y_slider.valueChanged.connect(self._apply_settings)
        self._auto_motion.toggled.connect(self._apply_settings)
        self._rotation_motion.currentIndexChanged.connect(self._apply_settings)
        self._rotation_speed_slider.valueChanged.connect(self._apply_settings)
        self._rotation_amplitude_slider.valueChanged.connect(self._apply_settings)
        self._gesture_rotation.toggled.connect(self._toggle_gesture)
        self._gesture_sensitivity_slider.valueChanged.connect(self._apply_settings)
        self._style_combo.currentIndexChanged.connect(self._apply_settings)
        self._effect_combo.currentIndexChanged.connect(self._apply_settings)
        self._tint_button.clicked.connect(self._pick_tint_color)
        self._tint_strength_slider.valueChanged.connect(self._apply_settings)
        self._accent_button.clicked.connect(self._pick_accent_color)
        self._accent_strength_slider.valueChanged.connect(self._apply_settings)
        self._shadow_slider.valueChanged.connect(self._apply_settings)
        self._reset_colors_button.clicked.connect(self._reset_colors)
        self._clean_preset_button.clicked.connect(self._apply_clean_preset)
        self._party_preset_button.clicked.connect(self._apply_party_preset)
        self._cyber_preset_button.clicked.connect(self._apply_cyber_preset)
        self._snapshot_button.clicked.connect(self._save_snapshot)

    @Slot(QImage)
    def _on_frame(self, image: QImage) -> None:
        self._last_image = image
        self._render_frame()

    @Slot(float, bool)
    def _on_metrics(self, fps: float, has_face: bool) -> None:
        status = f"FPS: {fps:.1f} | {'Face' if has_face else 'No face'}"
        self.statusBar().showMessage(status)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        QMessageBox.critical(self, "Camera Error", message)

    def _apply_settings(self) -> None:
        self._settings.show_sunglasses = self._toggle_glasses.isChecked()
        self._settings.show_hat = self._toggle_hat.isChecked()
        self._settings.show_mustache = self._toggle_mustache.isChecked()
        self._settings.show_halo = self._toggle_halo.isChecked()
        self._settings.show_sparkles = self._toggle_sparkles.isChecked()
        self._settings.show_bow_tie = self._toggle_bow_tie.isChecked()
        self._settings.show_face_frame = self._toggle_face_frame.isChecked()
        self._settings.opacity = self._opacity_slider.value() / 100.0
        self._settings.scale = self._scale_slider.value() / 100.0
        self._settings.rotation_deg = float(self._rotation_slider.value())
        self._settings.rotation_target = str(self._rotation_target.currentData())
        self._settings.offset_x = self._offset_x_slider.value() / 100.0
        self._settings.offset_y = self._offset_y_slider.value() / 100.0
        self._settings.auto_motion = self._auto_motion.isChecked()
        self._settings.rotation_motion = str(self._rotation_motion.currentData())
        self._settings.rotation_speed = self._rotation_speed_slider.value() / 100.0
        self._settings.rotation_amplitude = float(self._rotation_amplitude_slider.value())
        self._settings.gesture_rotation = self._gesture_rotation.isChecked()
        self._settings.gesture_sensitivity = self._gesture_sensitivity_slider.value() / 100.0
        self._settings.style = str(self._style_combo.currentData())
        self._settings.frame_effect = str(self._effect_combo.currentData())
        self._settings.tint_strength = self._tint_strength_slider.value() / 100.0
        self._settings.accent_strength = self._accent_strength_slider.value() / 100.0
        self._settings.shadow_strength = self._shadow_slider.value() / 100.0
        self._worker.update_settings(self._settings)

    def _toggle_gesture(self, enabled: bool) -> None:
        self._apply_settings()

    def _pick_tint_color(self) -> None:
        color = QColorDialog.getColor(parent=self)
        if not color.isValid():
            return
        self._settings.tint_color = (color.blue(), color.green(), color.red())
        self._tint_color_name = color.name()
        self._tint_button.setStyleSheet(f"background-color: {self._tint_color_name};")
        self._apply_settings()

    def _pick_accent_color(self) -> None:
        color = QColorDialog.getColor(parent=self)
        if not color.isValid():
            return
        self._settings.accent_color = (color.blue(), color.green(), color.red())
        self._accent_color_name = color.name()
        self._accent_button.setStyleSheet(f"background-color: {self._accent_color_name};")
        self._apply_settings()

    def _reset_colors(self) -> None:
        self._settings.tint_color = None
        self._settings.accent_color = None
        self._tint_color_name = ""
        self._accent_color_name = ""
        self._tint_button.setStyleSheet("")
        self._accent_button.setStyleSheet("")
        self._tint_strength_slider.setValue(0)
        self._accent_strength_slider.setValue(0)
        self._apply_settings()

    def _apply_clean_preset(self) -> None:
        self._toggle_glasses.setChecked(True)
        self._toggle_hat.setChecked(False)
        self._toggle_mustache.setChecked(False)
        self._toggle_halo.setChecked(False)
        self._toggle_sparkles.setChecked(False)
        self._toggle_bow_tie.setChecked(False)
        self._toggle_face_frame.setChecked(False)
        self._style_combo.setCurrentIndex(self._style_combo.findData("classic"))
        self._effect_combo.setCurrentIndex(self._effect_combo.findData("soft"))
        self._opacity_slider.setValue(85)
        self._scale_slider.setValue(100)
        self._shadow_slider.setValue(30)
        self._auto_motion.setChecked(False)
        self._rotation_motion.setCurrentIndex(self._rotation_motion.findData("sway"))
        self._rotation_speed_slider.setValue(100)
        self._rotation_amplitude_slider.setValue(18)
        self._gesture_sensitivity_slider.setValue(100)
        self._reset_target_offsets()
        self._apply_settings()

    def _apply_party_preset(self) -> None:
        self._toggle_glasses.setChecked(True)
        self._toggle_hat.setChecked(True)
        self._toggle_mustache.setChecked(False)
        self._toggle_halo.setChecked(True)
        self._toggle_sparkles.setChecked(True)
        self._toggle_bow_tie.setChecked(True)
        self._toggle_face_frame.setChecked(False)
        self._style_combo.setCurrentIndex(self._style_combo.findData("pop"))
        self._effect_combo.setCurrentIndex(self._effect_combo.findData("cartoon"))
        self._opacity_slider.setValue(92)
        self._scale_slider.setValue(108)
        self._shadow_slider.setValue(38)
        self._auto_motion.setChecked(True)
        self._rotation_motion.setCurrentIndex(self._rotation_motion.findData("pulse"))
        self._rotation_speed_slider.setValue(135)
        self._rotation_amplitude_slider.setValue(34)
        self._gesture_sensitivity_slider.setValue(130)
        self._reset_target_offsets()
        self._apply_settings()

    def _apply_cyber_preset(self) -> None:
        self._toggle_glasses.setChecked(True)
        self._toggle_hat.setChecked(False)
        self._toggle_mustache.setChecked(False)
        self._toggle_halo.setChecked(True)
        self._toggle_sparkles.setChecked(True)
        self._toggle_bow_tie.setChecked(False)
        self._toggle_face_frame.setChecked(True)
        self._style_combo.setCurrentIndex(self._style_combo.findData("neon"))
        self._effect_combo.setCurrentIndex(self._effect_combo.findData("edges"))
        self._opacity_slider.setValue(88)
        self._scale_slider.setValue(104)
        self._shadow_slider.setValue(15)
        self._auto_motion.setChecked(True)
        self._rotation_motion.setCurrentIndex(self._rotation_motion.findData("spin"))
        self._rotation_speed_slider.setValue(80)
        self._rotation_amplitude_slider.setValue(45)
        self._gesture_sensitivity_slider.setValue(150)
        self._reset_target_offsets()
        self._apply_settings()

    def _reset_target_offsets(self) -> None:
        self._offset_x_slider.setValue(0)
        self._offset_y_slider.setValue(0)

    def _save_snapshot(self) -> None:
        if self._last_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Snapshot",
            "snapshot.png",
            "PNG (*.png);;JPEG (*.jpg)",
        )
        if path:
            self._last_image.save(path)

    def _render_frame(self) -> None:
        if self._last_image is None:
            return
        pixmap = QPixmap.fromImage(self._last_image)
        scaled = pixmap.scaled(
            self._video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._video_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_frame()

    def closeEvent(self, event) -> None:
        self._worker.stop()
        self._worker.wait(2000)
        super().closeEvent(event)


def run_app() -> None:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    window = MainWindow(AppConfig())
    window.resize(1100, 720)
    window.show()
    sys.exit(app.exec())
