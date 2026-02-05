"""
Validator voor scherpte controle
"""
import numpy as np
import cv2
from typing import Optional

from ..models.photo import Photo, ValidationResult, FaceDetectionResult
from .base_validator import BaseValidator, ValidatorConfig


class SharpnessValidator(BaseValidator):
    """
    Valideert of de foto voldoende scherp is

    Gebruikt Laplacian variance methode om blur te detecteren
    """

    def __init__(
        self,
        threshold: float = ValidatorConfig.SHARPNESS_THRESHOLD,
        min_variance: float = ValidatorConfig.SHARPNESS_MIN_VARIANCE
    ):
        """
        Initialiseer sharpness validator

        Args:
            threshold: Minimum confidence threshold
            min_variance: Minimum Laplacian variance voor scherpe foto
        """
        super().__init__(threshold)
        self._min_variance = min_variance

    def validate(self, photo: Photo, face_detection: Optional[FaceDetectionResult] = None) -> ValidationResult:
        """
        Valideer scherpte van foto

        Args:
            photo: Photo object om te valideren
            face_detection: Optioneel FaceDetectionResult voor focus op gezicht

        Returns:
            ValidationResult met resultaat
        """
        self._validate_image_data(photo)

        # Converteer naar grayscale
        gray = self._get_grayscale_image(photo)

        # Als gezicht gedetecteerd is, focus op gezichtsregio
        if face_detection and face_detection.face_found and face_detection.face_bbox:
            x, y, w, h = face_detection.face_bbox
            # Voeg wat padding toe
            padding = int(min(w, h) * 0.1)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(gray.shape[1] - x, w + 2 * padding)
            h = min(gray.shape[0] - y, h + 2 * padding)
            gray = gray[y:y+h, x:x+w]

        # Bereken Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Bereken confidence score
        # Lineaire mapping van variance naar confidence
        if variance >= self._min_variance:
            confidence = min(1.0, variance / (self._min_variance * 2))
        else:
            confidence = variance / self._min_variance

        # Bepaal validatie resultaat
        is_valid = confidence >= self.threshold

        # Genereer feedback bericht
        if is_valid:
            message = "Foto is voldoende scherp"
        elif variance < self._min_variance * 0.5:
            message = "Foto is wazig. Houd de camera stil en zorg voor goede focus"
        else:
            message = "Foto is niet scherp genoeg. Probeer opnieuw"

        details = {
            "laplacian_variance": float(variance),
            "min_variance_threshold": self._min_variance,
            "focused_on_face": face_detection is not None and face_detection.face_found
        }

        return self._create_result(is_valid, confidence, message, details)

    def get_name(self) -> str:
        """Haal naam van validator op"""
        return "SharpnessValidator"

    def get_description(self) -> str:
        """Haal beschrijving van validator op"""
        return "Controleert of de foto voldoende scherp is en niet wazig"
