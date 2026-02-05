"""
Validator voor belichting controle
"""
import numpy as np
import cv2
from typing import Optional

from ..models.photo import Photo, ValidationResult, FaceDetectionResult
from .base_validator import BaseValidator, ValidatorConfig


class BrightnessValidator(BaseValidator):
    """
    Valideert of de belichting van de foto correct is

    Controleert:
    - Gemiddelde brightness ligt binnen acceptabel bereik
    - Geen over- of onderbelichting
    - Histogram distributie is redelijk gebalanceerd
    """

    def __init__(
        self,
        threshold: float = ValidatorConfig.BRIGHTNESS_THRESHOLD,
        min_brightness: int = ValidatorConfig.BRIGHTNESS_MIN,
        max_brightness: int = ValidatorConfig.BRIGHTNESS_MAX
    ):
        """
        Initialiseer brightness validator

        Args:
            threshold: Minimum confidence threshold
            min_brightness: Minimum gemiddelde brightness waarde
            max_brightness: Maximum gemiddelde brightness waarde
        """
        super().__init__(threshold)
        # sla de min en max brightness op
        self._min_brightness = min_brightness
        self._max_brightness = max_brightness

    def validate(self, photo: Photo, face_detection: Optional[FaceDetectionResult] = None) -> ValidationResult:
        """
        Valideer belichting van foto

        Args:
            photo: Photo object om te valideren
            face_detection: Optioneel FaceDetectionResult (niet gebruikt door deze validator)

        Returns:
            ValidationResult met resultaat
        """
        # check of de foto valid is
        self._validate_image_data(photo)

        # zet de foto om naar grijstinten, dan kunnen we makkelijker brightness berekenen
        gray = self._get_grayscale_image(photo)

        # bereken de gemiddelde helderheid van alle pixels
        mean_brightness = np.mean(gray)

        # maak een histogram van alle pixelwaardes (0-255)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist / hist.sum()

        # kijk hoeveel pixels bijna wit zijn (overbelichting) of bijna zwart (onderbelichting)
        overexposed_ratio = hist_normalized[240:].sum()  # pixels tussen 240-255 (bijna wit)
        underexposed_ratio = hist_normalized[:15].sum()   # pixels tussen 0-15 (bijna zwart)

        # bereken hoe goed de foto belicht is
        # factor 1: zit de brightness binnen het goede bereik?
        if self._min_brightness <= mean_brightness <= self._max_brightness:
            brightness_score = 1.0  # perfect!
        else:
            # foto is te donker of te licht, bereken hoeveel te veel
            if mean_brightness < self._min_brightness:
                distance = self._min_brightness - mean_brightness
            else:
                distance = mean_brightness - self._max_brightness
            # hoe verder we ernaast zitten, hoe lager de score
            brightness_score = max(0.0, 1.0 - (distance / 30))

        # factor 2: zijn er te veel super donkere of super lichte plekken?
        extreme_score = 1.0 - min(1.0, (overexposed_ratio + underexposed_ratio) * 2)

        # combineer beide scores (brightness is belangrijker, dus 70%)
        confidence = (brightness_score * 0.7 + extreme_score * 0.3)

        # is de foto goed genoeg?
        is_valid = confidence >= self.threshold

        # maak een duidelijke feedback message
        if is_valid:
            message = "Belichting is correct"
        elif mean_brightness < self._min_brightness:
            message = "Foto is te donker. Zorg voor meer licht"
        elif mean_brightness > self._max_brightness:
            message = "Foto is te licht. Verminder de belichting"
        elif overexposed_ratio > 0.05:
            message = "Te veel overbelichting gedetecteerd"
        elif underexposed_ratio > 0.05:
            message = "Te veel onderbelichting gedetecteerd"
        else:
            message = "Belichting is niet optimaal"

        # sla alle details op voor debugging
        details = {
            "mean_brightness": float(mean_brightness),
            "overexposed_ratio": float(overexposed_ratio),
            "underexposed_ratio": float(underexposed_ratio),
            "brightness_score": float(brightness_score),
            "extreme_score": float(extreme_score)
        }

        return self._create_result(is_valid, confidence, message, details)

    def get_name(self) -> str:
        """Haal naam van validator op"""
        return "BrightnessValidator"

    def get_description(self) -> str:
        """Haal beschrijving van validator op"""
        return "Controleert of de belichting van de foto correct is"
