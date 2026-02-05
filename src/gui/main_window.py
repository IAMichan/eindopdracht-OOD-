"""
Main window voor Pasfoto Validatie applicatie
"""
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QTextEdit,
    QMessageBox, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np
import logging

from ..services.camera_service import PhotoBoothCameraService
from ..services.validation_service import ValidationService
from ..services.storage_service import StorageService
from ..models.photo import Photo


logger = logging.getLogger(__name__)


class PhotoBoothWindow(QMainWindow):
    """
    Main window voor photobooth applicatie
    """

    def __init__(self):
        """Initialiseer main window"""
        super().__init__()

        # Services
        self._camera_service = PhotoBoothCameraService(camera_index=0, countdown_seconds=3)
        self._validation_service = ValidationService()
        self._storage_service = StorageService()

        # State
        self._current_photo: Photo = None
        self._capturing = False

        # Setup UI
        self._setup_ui()

        # Register as observer for validation updates
        self._validation_service.add_observer(self)

        # Start camera
        if not self._camera_service.initialize():
            QMessageBox.critical(
                self,
                "Camera Error",
                "Kon camera niet initialiseren. Controleer of een camera verbonden is."
            )

        # Start preview timer
        self._preview_timer = QTimer()
        self._preview_timer.timeout.connect(self._update_preview)
        self._preview_timer.start(33)  # ~30 FPS

        logger.info("PhotoBoothWindow initialized")

    def _setup_ui(self):
        """Setup gebruikersinterface"""
        self.setWindowTitle("Pasfoto Validatie - Fotohokje")
        self.setMinimumSize(1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Left panel - Camera preview
        left_panel = self._create_camera_panel()
        main_layout.addWidget(left_panel, stretch=2)

        # Right panel - Feedback en controls
        right_panel = self._create_control_panel()
        main_layout.addWidget(right_panel, stretch=1)

    def _create_camera_panel(self) -> QWidget:
        """Creëer camera preview panel"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)

        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Camera Preview")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # Camera preview label
        self._camera_label = QLabel()
        self._camera_label.setMinimumSize(640, 480)
        self._camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._camera_label.setStyleSheet("background-color: black; border: 2px solid #ccc;")
        layout.addWidget(self._camera_label)

        # Status label
        self._status_label = QLabel("Positioneer uw gezicht in het kader")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet("font-size: 16px; padding: 10px;")
        layout.addWidget(self._status_label)

        return panel

    def _create_control_panel(self) -> QWidget:
        """Creëer control panel"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)

        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Controles & Feedback")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # Capture button
        self._capture_button = QPushButton("Maak Pasfoto")
        self._capture_button.setMinimumHeight(60)
        self._capture_button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self._capture_button.clicked.connect(self._on_capture_clicked)
        layout.addWidget(self._capture_button)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        # Feedback text
        feedback_label = QLabel("Validatie Feedback:")
        feedback_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(feedback_label)

        self._feedback_text = QTextEdit()
        self._feedback_text.setReadOnly(True)
        self._feedback_text.setMaximumHeight(300)
        layout.addWidget(self._feedback_text)

        # Action buttons
        button_layout = QHBoxLayout()

        self._accept_button = QPushButton("Accepteren")
        self._accept_button.setEnabled(False)
        self._accept_button.clicked.connect(self._on_accept_clicked)
        button_layout.addWidget(self._accept_button)

        self._retry_button = QPushButton("Opnieuw")
        self._retry_button.setEnabled(False)
        self._retry_button.clicked.connect(self._on_retry_clicked)
        button_layout.addWidget(self._retry_button)

        layout.addLayout(button_layout)

        # Add stretch
        layout.addStretch()

        return panel

    def _update_preview(self):
        """Update camera preview"""
        if not self._camera_service.is_initialized or self._capturing:
            return

        # Get frame with overlay
        frame = self._camera_service.get_preview_with_overlay(overlay_type="guidelines")

        if frame is None:
            return

        # Convert to Qt format
        self._display_image(frame, self._camera_label)

    def _display_image(self, image: np.ndarray, label: QLabel):
        """
        Display image in label

        Args:
            image: NumPy array (BGR format)
            label: QLabel to display in
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to QImage
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale to label size while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        label.setPixmap(scaled_pixmap)

    def _on_capture_clicked(self):
        """Handle capture button click"""
        self._capturing = True
        self._capture_button.setEnabled(False)
        self._status_label.setText("Foto wordt gemaakt...")
        self._feedback_text.clear()

        # Capture photo
        photo = self._camera_service.capture_photo()

        if photo is None:
            QMessageBox.warning(self, "Fout", "Kon geen foto maken")
            self._reset_ui()
            return

        self._current_photo = photo

        # Display captured photo
        self._display_image(photo.image_data, self._camera_label)

        # Start validation
        self._status_label.setText("Foto wordt gevalideerd...")
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)

        # Validate in separate thread would be better, but for simplicity we do it directly
        validated_photo = self._validation_service.validate_photo(photo)
        self._current_photo = validated_photo

        # Save photo (approved or rejected)
        self._storage_service.save_photo(validated_photo)

        # Show results
        self._show_validation_results()

    def _show_validation_results(self):
        """Toon validatie resultaten"""
        if self._current_photo is None:
            return

        # Update progress
        self._progress_bar.setValue(100)

        # Build feedback text
        feedback = []
        feedback.append(f"<h3>Validatie Resultaat</h3>")
        feedback.append(f"<p><b>Status:</b> {self._current_photo.status.value.upper()}</p>")
        feedback.append(f"<p><b>Overall Confidence:</b> {self._current_photo.get_overall_confidence():.1%}</p>")
        feedback.append("<hr>")

        if self._current_photo.is_valid():
            feedback.append("<p style='color: green; font-weight: bold;'>✓ Alle validaties geslaagd!</p>")
            self._status_label.setText("Pasfoto geaccepteerd!")
            self._status_label.setStyleSheet("font-size: 16px; padding: 10px; color: green; font-weight: bold;")
        else:
            feedback.append("<p style='color: red; font-weight: bold;'>✗ Sommige validaties gefaald</p>")
            self._status_label.setText("Pasfoto afgekeurd - zie feedback")
            self._status_label.setStyleSheet("font-size: 16px; padding: 10px; color: red; font-weight: bold;")

        feedback.append("<h4>Details:</h4>")
        feedback.append("<ul>")

        for result in self._current_photo.validation_results:
            icon = "✓" if result.is_valid else "✗"
            color = "green" if result.is_valid else "red"
            feedback.append(
                f"<li><span style='color: {color};'>{icon} <b>{result.validator_name}:</b> "
                f"{result.message} ({result.confidence:.1%})</span></li>"
            )

        feedback.append("</ul>")

        self._feedback_text.setHtml("".join(feedback))

        # Enable action buttons
        if self._current_photo.is_valid():
            self._accept_button.setEnabled(True)
        self._retry_button.setEnabled(True)

    def _on_accept_clicked(self):
        """Handle accept button click"""
        if self._current_photo is None:
            return

        # Photo already saved after validation, just show confirmation
        QMessageBox.information(
            self,
            "Opgeslagen",
            f"Pasfoto succesvol opgeslagen!\n\nBestand: {self._current_photo.file_path}"
        )

        self._reset_ui()

    def _on_retry_clicked(self):
        """Handle retry button click"""
        self._reset_ui()

    def _reset_ui(self):
        """Reset UI naar initiële staat"""
        self._current_photo = None
        self._capturing = False
        self._capture_button.setEnabled(True)
        self._accept_button.setEnabled(False)
        self._retry_button.setEnabled(False)
        self._progress_bar.setVisible(False)
        self._progress_bar.setValue(0)
        self._status_label.setText("Positioneer uw gezicht in het kader")
        self._status_label.setStyleSheet("font-size: 16px; padding: 10px;")
        self._feedback_text.clear()

    def update(self, event_type: str, data):
        """
        Observer update methode

        Args:
            event_type: Type van event
            data: Event data
        """
        if event_type == "validation_progress":
            # Update status
            self._status_label.setText(data)
        elif event_type == "validation_result":
            # Update progress bar
            progress = len(self._current_photo.validation_results) / len(self._validation_service.get_validators()) * 100
            self._progress_bar.setValue(int(progress))

    def closeEvent(self, event):
        """Handle window close"""
        self._camera_service.release()
        event.accept()
