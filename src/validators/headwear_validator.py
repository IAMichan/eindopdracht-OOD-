"""
Validator voor hoofddeksel detectie
"""
import numpy as np
import cv2
from typing import Optional

from ..models.photo import Photo, ValidationResult, FaceDetectionResult
from .base_validator import BaseValidator, ValidatorConfig


class HeadwearValidator(BaseValidator):
    """
    Valideert dat er geen hoofddeksel wordt gedragen

    Controleert:
    - Geen pet/cap
    - Geen hoed
    - Geen hoofddoek (tenzij religieus)
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialiseer headwear validator

        Args:
            threshold: Minimum confidence threshold
        """
        super().__init__(threshold)

    def validate(self, photo: Photo, face_detection: Optional[FaceDetectionResult] = None) -> ValidationResult:
        """
        Valideer aanwezigheid van hoofddeksel

        Args:
            photo: Photo object om te valideren
            face_detection: FaceDetectionResult met gezichtsinfo

        Returns:
            ValidationResult met resultaat
        """
        self._validate_image_data(photo)

        if not face_detection or not face_detection.face_found or not face_detection.face_bbox:
            return self._create_result(
                is_valid=False,
                confidence=0.0,
                message="Geen gezicht gedetecteerd",
                details={"error": "no_face_detected"}
            )

        # Haal gezichtsinformatie op
        x, y, w, h = face_detection.face_bbox
        img_height, img_width = photo.image_data.shape[:2]

        # Check het gebied boven het voorhoofd
        # Als daar iets is (niet huid-kleurig), kan het een hoofddeksel zijn
        forehead_top = max(0, y - int(h * 0.3))  # 30% boven de face bbox
        forehead_region = photo.image_data[forehead_top:y, x:x+w]

        if forehead_region.size == 0:
            return self._create_result(
                is_valid=True,
                confidence=0.8,
                message="Geen hoofddeksel gedetecteerd",
                details={"note": "small_region"}
            )

        # Analyseer de kleuren in het voorhoofd gebied
        # Huid heeft typisch bepaalde HSV waarden
        hsv = cv2.cvtColor(forehead_region, cv2.COLOR_BGR2HSV)

        # Skin tone ranges in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Tweede range voor donkerdere huid
        lower_skin2 = np.array([0, 10, 40], dtype=np.uint8)
        upper_skin2 = np.array([25, 200, 255], dtype=np.uint8)

        skin_mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)

        # Bereken hoeveel van het gebied huidkleurig is
        skin_ratio = np.sum(skin_mask > 0) / skin_mask.size

        # Check ook voor donkere pixels (pet/hoed is vaak donker)
        gray = cv2.cvtColor(forehead_region, cv2.COLOR_BGR2GRAY)
        dark_ratio = np.sum(gray < 60) / gray.size

        # Check voor uniforme niet-huid kleuren (typisch voor petten)
        # Bereken kleur variatie
        std_color = np.std(forehead_region)

        # Bepaal of er een hoofddeksel is
        has_headwear = False
        confidence = 1.0

        # Als er weinig huid is EN veel donkere pixels, waarschijnlijk hoofddeksel
        if skin_ratio < 0.3 and dark_ratio > 0.4:
            has_headwear = True
            confidence = 0.2

        # Als er bijna geen huid is, zeer waarschijnlijk hoofddeksel
        if skin_ratio < 0.15:
            has_headwear = True
            confidence = 0.1

        # Als het gebied heel uniform is (lage std), waarschijnlijk stof/materiaal
        if std_color < 25 and skin_ratio < 0.4:
            has_headwear = True
            confidence = 0.3

        is_valid = not has_headwear

        if is_valid:
            message = "Geen hoofddeksel gedetecteerd"
        else:
            message = "Verwijder hoofddeksel (pet, hoed) voor de pasfoto"

        details = {
            "skin_ratio": float(skin_ratio),
            "dark_ratio": float(dark_ratio),
            "color_std": float(std_color),
            "has_headwear": has_headwear
        }

        return self._create_result(is_valid, confidence, message, details)

    def get_name(self) -> str:
        """Haal naam van validator op"""
        return "HeadwearValidator"

    def get_description(self) -> str:
        """Haal beschrijving van validator op"""
        return "Controleert of er geen hoofddeksel wordt gedragen"
