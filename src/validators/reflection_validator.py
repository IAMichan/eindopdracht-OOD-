"""
Validator voor reflectie detectie
"""
import numpy as np
import cv2
from typing import Optional

from ..models.photo import Photo, ValidationResult, FaceDetectionResult
from .base_validator import BaseValidator, ValidatorConfig


class ReflectionValidator(BaseValidator):
    """
    Valideert dat er geen reflecties in de foto zitten

    Controleert:
    - Geen bril reflectie
    - Geen flash reflectie op gezicht
    - Geen andere heldere spots
    """

    def __init__(
        self,
        threshold: float = ValidatorConfig.REFLECTION_THRESHOLD,
        brightness_threshold: int = ValidatorConfig.REFLECTION_BRIGHTNESS_THRESHOLD,
        max_reflection_ratio: float = ValidatorConfig.REFLECTION_MAX_RATIO
    ):
        """
        Initialiseer reflection validator

        Args:
            threshold: Minimum confidence threshold
            brightness_threshold: Brightness waarde voor reflectie detectie
            max_reflection_ratio: Maximum toegestane reflectie als ratio van gezicht
        """
        super().__init__(threshold)
        self._brightness_threshold = brightness_threshold
        self._max_reflection_ratio = max_reflection_ratio

    def validate(self, photo: Photo, face_detection: Optional[FaceDetectionResult] = None) -> ValidationResult:
        """
        Valideer aanwezigheid van reflecties

        Args:
            photo: Photo object om te valideren
            face_detection: Optioneel FaceDetectionResult om op gezicht te focussen

        Returns:
            ValidationResult met resultaat
        """
        self._validate_image_data(photo)

        # Converteer naar grayscale
        gray = self._get_grayscale_image(photo)

        # Als gezicht gedetecteerd is, focus op gezichtsregio
        face_region = None
        if face_detection and face_detection.face_found and face_detection.face_bbox:
            x, y, w, h = face_detection.face_bbox
            face_region = gray[y:y+h, x:x+w]
            analysis_region = face_region
        else:
            analysis_region = gray

        # Detect zeer heldere pixels (mogelijke reflecties)
        _, bright_mask = cv2.threshold(
            analysis_region,
            self._brightness_threshold,
            255,
            cv2.THRESH_BINARY
        )

        # Morfologische operaties om kleine ruis te verwijderen
        kernel = np.ones((3, 3), np.uint8)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

        # Tel reflectie pixels
        reflection_pixels = np.sum(bright_mask > 0)
        total_pixels = analysis_region.size
        reflection_ratio = reflection_pixels / total_pixels

        # Vind connected components (clusters van heldere pixels)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bright_mask, connectivity=8
        )

        # Filter kleine components (ruis)
        significant_reflections = 0
        large_reflections = []

        for i in range(1, num_labels):  # Skip 0 (background)
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 50:  # Minimum area voor significante reflectie
                significant_reflections += 1
                large_reflections.append({
                    'area': area,
                    'center': centroids[i].tolist()
                })

        # Bereken confidence score
        # Factor 1: Totale reflectie ratio
        if reflection_ratio <= self._max_reflection_ratio:
            ratio_score = 1.0
        else:
            excess = reflection_ratio - self._max_reflection_ratio
            ratio_score = max(0.0, 1.0 - (excess / self._max_reflection_ratio))

        # Factor 2: Aantal significante reflecties
        if significant_reflections == 0:
            count_score = 1.0
        elif significant_reflections <= 2:
            count_score = 0.7
        else:
            count_score = max(0.0, 1.0 - (significant_reflections - 2) * 0.15)

        # Combineer scores
        confidence = (ratio_score * 0.6 + count_score * 0.4)

        # Bepaal validatie resultaat
        is_valid = confidence >= self.threshold

        # Genereer feedback bericht
        if is_valid:
            message = "Geen significante reflecties gedetecteerd"
        elif significant_reflections > 3:
            message = "Meerdere reflecties gedetecteerd. Vermijd direct licht en bril reflectie"
        elif reflection_ratio > self._max_reflection_ratio * 2:
            message = "Sterke reflectie gedetecteerd. Pas belichting aan"
        else:
            message = "Lichte reflectie gedetecteerd. Pas positie of belichting aan"

        details = {
            "reflection_ratio": float(reflection_ratio),
            "reflection_pixels": int(reflection_pixels),
            "significant_reflections": significant_reflections,
            "large_reflections": large_reflections,
            "ratio_score": float(ratio_score),
            "count_score": float(count_score)
        }

        return self._create_result(is_valid, confidence, message, details)

    def get_name(self) -> str:
        """Haal naam van validator op"""
        return "ReflectionValidator"

    def get_description(self) -> str:
        """Haal beschrijving van validator op"""
        return "Controleert op ongewenste reflecties in de foto"
