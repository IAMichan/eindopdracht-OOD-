"""
Camera service voor foto capture
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from datetime import datetime
import logging

from ..models.photo import Photo


logger = logging.getLogger(__name__)


class CameraService:
    """
    Service voor camera toegang en foto capture

    Handles camera initialization, frame capture, en resource management
    """

    def __init__(self, camera_index: int = 0):
        """
        Initialiseer camera service

        Args:
            camera_index: Index van camera device (default 0)
        """
        self._camera_index = camera_index
        self._camera: Optional[cv2.VideoCapture] = None
        self._is_initialized = False

    def initialize(self) -> bool:
        """
        Initialiseer camera

        Returns:
            True als succesvol, False anders
        """
        try:
            self._camera = cv2.VideoCapture(self._camera_index)

            if not self._camera.isOpened():
                logger.error(f"Could not open camera {self._camera_index}")
                return False

            # Configureer camera settings voor beste kwaliteit
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._camera.set(cv2.CAP_PROP_FPS, 30)

            self._is_initialized = True
            logger.info(f"Camera {self._camera_index} initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing camera: {e}", exc_info=True)
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Haal een frame op van de camera

        Returns:
            Frame als NumPy array, of None bij fout
        """
        if not self._is_initialized or self._camera is None:
            logger.warning("Camera not initialized")
            return None

        try:
            ret, frame = self._camera.read()

            if not ret:
                logger.warning("Failed to read frame from camera")
                return None

            return frame

        except Exception as e:
            logger.error(f"Error reading frame: {e}", exc_info=True)
            return None

    def capture_photo(self) -> Optional[Photo]:
        """
        Capture een foto en creëer Photo object

        Returns:
            Photo object met captured image, of None bij fout
        """
        frame = self.get_frame()

        if frame is None:
            return None

        try:
            photo = Photo(
                image_data=frame.copy(),  # Copy to avoid reference issues
                timestamp=datetime.now(),
                metadata={
                    "camera_index": self._camera_index,
                    "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                    "capture_mode": "photobooth"
                }
            )

            logger.info(f"Photo captured: {photo.timestamp}")
            return photo

        except Exception as e:
            logger.error(f"Error creating photo: {e}", exc_info=True)
            return None

    def get_camera_info(self) -> dict:
        """
        Haal camera informatie op

        Returns:
            Dictionary met camera properties
        """
        if not self._is_initialized or self._camera is None:
            return {}

        try:
            return {
                "index": self._camera_index,
                "width": int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": int(self._camera.get(cv2.CAP_PROP_FPS)),
                "backend": self._camera.getBackendName()
            }
        except Exception as e:
            logger.error(f"Error getting camera info: {e}", exc_info=True)
            return {}

    def release(self) -> None:
        """Release camera resources"""
        if self._camera is not None:
            self._camera.release()
            self._is_initialized = False
            logger.info("Camera released")

    def __del__(self):
        """Cleanup on destruction"""
        self.release()

    @property
    def is_initialized(self) -> bool:
        """Check of camera geïnitialiseerd is"""
        return self._is_initialized


class PhotoBoothCameraService(CameraService):
    """
    Extended camera service voor photobooth gebruik

    Voegt extra functionaliteit toe zoals countdown, preview effects, etc.
    """

    def __init__(
        self,
        camera_index: int = 0,
        countdown_seconds: int = 3
    ):
        """
        Initialiseer photobooth camera service

        Args:
            camera_index: Camera device index
            countdown_seconds: Countdown tijd voor foto capture
        """
        super().__init__(camera_index)
        self._countdown_seconds = countdown_seconds

    def capture_photo_with_countdown(self, callback=None) -> Optional[Photo]:
        """
        Capture foto met countdown

        Args:
            callback: Optionele callback functie die aangeroepen wordt
                     elke seconde met resterende tijd

        Returns:
            Photo object
        """
        import time

        if not self._is_initialized:
            logger.warning("Camera not initialized")
            return None

        # Countdown
        for i in range(self._countdown_seconds, 0, -1):
            if callback:
                callback(i)
            time.sleep(1)

        # Capture
        return self.capture_photo()

    def get_preview_with_overlay(
        self,
        overlay_type: str = "guidelines"
    ) -> Optional[np.ndarray]:
        """
        Haal preview frame op met overlay

        Args:
            overlay_type: Type overlay ("guidelines", "face_box", etc.)

        Returns:
            Frame met overlay
        """
        frame = self.get_frame()

        if frame is None:
            return None

        # Voeg overlay toe
        if overlay_type == "guidelines":
            frame = self._add_guidelines_overlay(frame)
        elif overlay_type == "face_box":
            frame = self._add_face_box_overlay(frame)

        return frame

    def _add_guidelines_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Voeg richtlijnen overlay toe

        Args:
            frame: Input frame

        Returns:
            Frame met overlay
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Teken ovaal voor gezichtspositie
        center = (w // 2, int(h * 0.45))
        axes = (int(w * 0.15), int(h * 0.25))
        cv2.ellipse(overlay, center, axes, 0, 0, 360, (0, 255, 0), 2)

        # Teken centrum kruisje
        cross_size = 20
        cv2.line(
            overlay,
            (w // 2 - cross_size, h // 2),
            (w // 2 + cross_size, h // 2),
            (0, 255, 0), 1
        )
        cv2.line(
            overlay,
            (w // 2, h // 2 - cross_size),
            (w // 2, h // 2 + cross_size),
            (0, 255, 0), 1
        )

        return overlay

    def _add_face_box_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Voeg face box overlay toe

        Args:
            frame: Input frame

        Returns:
            Frame met overlay
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Teken rechthoek voor gezicht (ideale positie/grootte)
        box_w = int(w * 0.4)
        box_h = int(h * 0.6)
        box_x = (w - box_w) // 2
        box_y = int(h * 0.15)

        cv2.rectangle(
            overlay,
            (box_x, box_y),
            (box_x + box_w, box_y + box_h),
            (0, 255, 0), 2
        )

        return overlay
