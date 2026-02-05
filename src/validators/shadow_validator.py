"""
Validator voor schaduw detectie
"""
import numpy as np
import cv2
from typing import Optional

from ..models.photo import Photo, ValidationResult, FaceDetectionResult
from .base_validator import BaseValidator, ValidatorConfig


class ShadowValidator(BaseValidator):
    """
    Valideert dat er geen storende schaduwen in de foto zitten

    Controleert:
    - Geen harde schaduwen op gezicht
    - Geen schaduwen in achtergrond
    - Gelijkmatige belichting
    """

    def __init__(
        self,
        threshold: float = ValidatorConfig.SHADOW_THRESHOLD,
        darkness_threshold: int = ValidatorConfig.SHADOW_DARKNESS_THRESHOLD,
        max_shadow_ratio: float = ValidatorConfig.SHADOW_MAX_RATIO
    ):
        """
        Initialiseer shadow validator

        Args:
            threshold: Minimum confidence threshold
            darkness_threshold: Darkness waarde voor schaduw detectie
            max_shadow_ratio: Maximum toegestane schaduw als ratio van gezicht
        """
        super().__init__(threshold)
        self._darkness_threshold = darkness_threshold
        self._max_shadow_ratio = max_shadow_ratio

    def validate(self, photo: Photo, face_detection: Optional[FaceDetectionResult] = None) -> ValidationResult:
        """
        Valideer aanwezigheid van schaduwen

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
        if face_detection and face_detection.face_found and face_detection.face_bbox:
            x, y, w, h = face_detection.face_bbox
            # Voeg wat padding toe om nek/schouders mee te nemen
            padding = int(h * 0.2)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(gray.shape[1] - x, w + 2 * padding)
            h = min(gray.shape[0] - y, h + 2 * padding)
            analysis_region = gray[y:y+h, x:x+w]
        else:
            analysis_region = gray

        # Detect zeer donkere pixels (mogelijke schaduwen)
        _, dark_mask = cv2.threshold(
            analysis_region,
            self._darkness_threshold,
            255,
            cv2.THRESH_BINARY_INV
        )

        # Morfologische operaties om kleine ruis te verwijderen
        kernel = np.ones((5, 5), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

        # Tel schaduw pixels
        shadow_pixels = np.sum(dark_mask > 0)
        total_pixels = analysis_region.size
        shadow_ratio = shadow_pixels / total_pixels

        # Detect edges in shadow mask voor harde schaduw grenzen
        edges = cv2.Canny(dark_mask, 50, 150)
        edge_pixels = np.sum(edges > 0)

        # Bereken standaard deviatie van brightness (uniformiteit)
        std_dev = np.std(analysis_region)

        # Bereken confidence score
        # Factor 1: Totale schaduw ratio
        if shadow_ratio <= self._max_shadow_ratio:
            ratio_score = 1.0
        else:
            excess = shadow_ratio - self._max_shadow_ratio
            ratio_score = max(0.0, 1.0 - (excess / self._max_shadow_ratio))

        # Factor 2: Harde schaduw grenzen
        # Minder edges = zachtere overgangen = beter
        edge_ratio = edge_pixels / total_pixels
        if edge_ratio < 0.02:
            edge_score = 1.0
        else:
            edge_score = max(0.0, 1.0 - (edge_ratio - 0.02) / 0.05)

        # Factor 3: Uniformiteit (lagere std = uniformer = beter)
        if std_dev < 30:
            uniformity_score = 1.0
        elif std_dev < 50:
            uniformity_score = 0.8
        else:
            uniformity_score = max(0.0, 1.0 - (std_dev - 50) / 50)

        # Combineer scores
        confidence = (ratio_score * 0.5 + edge_score * 0.3 + uniformity_score * 0.2)

        # Bepaal validatie resultaat
        is_valid = confidence >= self.threshold

        # Genereer feedback bericht
        if is_valid:
            message = "Geen storende schaduwen gedetecteerd"
        elif shadow_ratio > self._max_shadow_ratio * 2:
            message = "Veel schaduw gedetecteerd. Verbeter de belichting"
        elif edge_score < 0.5:
            message = "Harde schaduwen gedetecteerd. Gebruik diffuus licht"
        elif uniformity_score < 0.6:
            message = "Ongelijkmatige belichting. Pas lichtbronnen aan"
        else:
            message = "Lichte schaduwen gedetecteerd. Pas belichting aan voor optimaal resultaat"

        details = {
            "shadow_ratio": float(shadow_ratio),
            "shadow_pixels": int(shadow_pixels),
            "edge_ratio": float(edge_ratio),
            "std_deviation": float(std_dev),
            "ratio_score": float(ratio_score),
            "edge_score": float(edge_score),
            "uniformity_score": float(uniformity_score)
        }

        return self._create_result(is_valid, confidence, message, details)

    def get_name(self) -> str:
        """Haal naam van validator op"""
        return "ShadowValidator"

    def get_description(self) -> str:
        """Haal beschrijving van validator op"""
        return "Controleert op storende schaduwen in de foto"
