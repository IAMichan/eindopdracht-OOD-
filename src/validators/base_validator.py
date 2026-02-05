"""
Base validator interface en abstracte implementatie
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from ..models.photo import Photo, ValidationResult, FaceDetectionResult


class IValidator(ABC):
    """
    Interface voor foto validators

    Implementeert Strategy Pattern - elke validator is een strategie
    voor het valideren van een specifiek aspect van de pasfoto.
    """

    @abstractmethod
    def validate(self, photo: Photo, face_detection: Optional[FaceDetectionResult] = None) -> ValidationResult:
        """
        Valideer een foto

        Args:
            photo: Photo object om te valideren
            face_detection: Optioneel FaceDetectionResult voor validators die gezichtsdetectie nodig hebben

        Returns:
            ValidationResult met resultaat van validatie
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Haal naam van validator op

        Returns:
            Naam van de validator
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Haal beschrijving van validator op

        Returns:
            Beschrijving van wat de validator doet
        """
        pass


class BaseValidator(IValidator):
    """
    Abstracte base class voor validators met gemeenschappelijke functionaliteit
    """

    def __init__(self, threshold: float = 0.7):
        """
        Initialiseer base validator

        Args:
            threshold: Minimum confidence threshold voor validatie (0.0 - 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """Haal threshold op"""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """
        Zet threshold

        Args:
            value: Nieuwe threshold waarde

        Raises:
            ValueError: Als value niet tussen 0.0 en 1.0 ligt
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self._threshold = value

    def _create_result(
        self,
        is_valid: bool,
        confidence: float,
        message: str,
        details: Optional[dict] = None
    ) -> ValidationResult:
        """
        Helper methode om ValidationResult te maken

        Args:
            is_valid: Of validatie geslaagd is
            confidence: Confidence score
            message: Feedback bericht
            details: Optionele extra details

        Returns:
            ValidationResult object
        """
        return ValidationResult(
            validator_name=self.get_name(),
            is_valid=is_valid,
            confidence=confidence,
            message=message,
            details=details
        )

    def _get_grayscale_image(self, photo: Photo) -> np.ndarray:
        """
        Converteer foto naar grayscale

        Args:
            photo: Photo object

        Returns:
            Grayscale image als NumPy array
        """
        import cv2
        if len(photo.image_data.shape) == 2:
            # Already grayscale
            return photo.image_data
        return cv2.cvtColor(photo.image_data, cv2.COLOR_BGR2GRAY)

    def _validate_image_data(self, photo: Photo) -> None:
        """
        Valideer dat image data geldig is

        Args:
            photo: Photo object

        Raises:
            ValueError: Als image data niet geldig is
        """
        if photo.image_data is None:
            raise ValueError("Photo has no image data")
        if photo.image_data.size == 0:
            raise ValueError("Photo image data is empty")


class ValidatorConfig:
    """
    Configuratie class voor validators

    Bevat alle thresholds en parameters voor de verschillende validators.
    Waardes zijn aangepast voor realistische pasfoto validatie.
    """

    # Belichting - ruim bereik voor normale lichtomstandigheden
    BRIGHTNESS_MIN = 50       # accepteer donkerdere foto's
    BRIGHTNESS_MAX = 230      # accepteer lichtere foto's
    BRIGHTNESS_THRESHOLD = 0.5  # soepeler

    # Scherpte - minder streng
    SHARPNESS_MIN_VARIANCE = 50   # veel lager - webcams zijn vaak minder scherp
    SHARPNESS_THRESHOLD = 0.5     # soepeler

    # Gezichtspositie - ruim bereik
    FACE_CENTER_TOLERANCE = 0.30  # 30% afwijking van centrum toegestaan
    FACE_SIZE_MIN_RATIO = 0.05    # kleinere gezichten accepteren (verder weg)
    FACE_SIZE_MAX_RATIO = 0.70    # grotere gezichten accepteren (dichterbij)
    FACE_POSITION_THRESHOLD = 0.5  # soepeler

    # Gezichtsuitdrukking - soepel
    EXPRESSION_THRESHOLD = 0.4     # laag - kleine glimlach moet kunnen
    MOUTH_OPEN_THRESHOLD = 0.5     # mond mag iets open staan

    # Ogen zichtbaarheid - aangepast voor MediaPipe
    EYE_VISIBILITY_THRESHOLD = 0.4  # soepel
    EYE_ASPECT_RATIO_MIN = 0.12     # alleen echt dichte ogen afkeuren

    # Reflectie detectie - tolerant
    REFLECTION_BRIGHTNESS_THRESHOLD = 250  # alleen echt felle reflecties
    REFLECTION_MAX_RATIO = 0.12     # meer reflectie toestaan
    REFLECTION_THRESHOLD = 0.5      # soepeler

    # Schaduw detectie - tolerant
    SHADOW_DARKNESS_THRESHOLD = 35  # alleen echt donkere schaduwen
    SHADOW_MAX_RATIO = 0.20         # meer schaduw toestaan
    SHADOW_THRESHOLD = 0.5          # soepeler

    # Algemeen
    FACE_DETECTION_CONFIDENCE_MIN = 0.4  # makkelijker gezicht detecteren
