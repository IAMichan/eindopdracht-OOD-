"""
Validator voor gezichtspositie controle
"""
from typing import Optional

from ..models.photo import Photo, ValidationResult, FaceDetectionResult
from .base_validator import BaseValidator, ValidatorConfig


class FacePositionValidator(BaseValidator):
    """
    Valideert of het gezicht correct gepositioneerd is

    Controleert:
    - Gezicht is gecentreerd in de foto
    - Gezicht heeft juiste afstand (grootte)
    - Gezicht is frontaal (geen grote rotatie)
    """

    def __init__(
        self,
        threshold: float = ValidatorConfig.FACE_POSITION_THRESHOLD,
        center_tolerance: float = ValidatorConfig.FACE_CENTER_TOLERANCE,
        min_size_ratio: float = ValidatorConfig.FACE_SIZE_MIN_RATIO,
        max_size_ratio: float = ValidatorConfig.FACE_SIZE_MAX_RATIO
    ):
        """
        Initialiseer face position validator

        Args:
            threshold: Minimum confidence threshold
            center_tolerance: Toegestane afwijking van centrum (als ratio van image grootte)
            min_size_ratio: Minimum gezichtsgrootte als ratio van image
            max_size_ratio: Maximum gezichtsgrootte als ratio van image
        """
        super().__init__(threshold)
        self._center_tolerance = center_tolerance
        self._min_size_ratio = min_size_ratio
        self._max_size_ratio = max_size_ratio

    def validate(self, photo: Photo, face_detection: Optional[FaceDetectionResult] = None) -> ValidationResult:
        """
        Valideer positie van gezicht

        Args:
            photo: Photo object om te valideren
            face_detection: FaceDetectionResult met gezichtsdetectie info (vereist!)

        Returns:
            ValidationResult met resultaat
        """
        self._validate_image_data(photo)

        # Face detection is verplicht voor deze validator
        if not face_detection or not face_detection.face_found:
            return self._create_result(
                is_valid=False,
                confidence=0.0,
                message="Geen gezicht gedetecteerd. Zorg dat uw gezicht zichtbaar is",
                details={"error": "no_face_detected"}
            )

        if not face_detection.face_bbox:
            return self._create_result(
                is_valid=False,
                confidence=0.0,
                message="Gezicht kon niet worden gelokaliseerd",
                details={"error": "no_bounding_box"}
            )

        # Haal image dimensies op
        img_height, img_width = photo.image_data.shape[:2]
        img_center_x = img_width / 2
        img_center_y = img_height / 2

        # Haal gezichtsinformatie op
        face_center = face_detection.get_face_center()
        face_size = face_detection.get_face_size()
        x, y, w, h = face_detection.face_bbox

        if not face_center or not face_size:
            return self._create_result(
                is_valid=False,
                confidence=0.0,
                message="Gezicht kon niet worden geanalyseerd",
                details={"error": "invalid_face_data"}
            )

        face_x, face_y = face_center

        # 1. Check centering (horizontaal en verticaal)
        center_offset_x = abs(face_x - img_center_x) / img_width
        center_offset_y = abs(face_y - img_center_y) / img_height
        max_center_offset = max(center_offset_x, center_offset_y)

        if max_center_offset <= self._center_tolerance:
            centering_score = 1.0
        else:
            # Lineair afnemen bij grotere afwijking
            centering_score = max(0.0, 1.0 - (max_center_offset - self._center_tolerance) / self._center_tolerance)

        # 2. Check grootte (afstand)
        img_size = img_width * img_height
        face_size_ratio = face_size / img_size

        if self._min_size_ratio <= face_size_ratio <= self._max_size_ratio:
            size_score = 1.0
        else:
            if face_size_ratio < self._min_size_ratio:
                # Te klein (te ver weg)
                distance = self._min_size_ratio - face_size_ratio
                size_score = max(0.0, 1.0 - (distance / self._min_size_ratio))
            else:
                # Te groot (te dichtbij)
                distance = face_size_ratio - self._max_size_ratio
                size_score = max(0.0, 1.0 - (distance / self._max_size_ratio))

        # 3. Check aspect ratio (gezicht moet redelijk proportioneel zijn)
        aspect_ratio = w / h if h > 0 else 0
        # Gezicht bounding box is typisch breder dan hoog of ongeveer gelijk
        # Acceptabel bereik: 0.55 - 1.1 (gezicht is meestal smaller dan vierkant)
        if 0.55 <= aspect_ratio <= 1.1:
            aspect_score = 1.0
        else:
            # Buiten bereik - bereken penalty
            if aspect_ratio < 0.55:
                aspect_distance = 0.55 - aspect_ratio
            else:
                aspect_distance = aspect_ratio - 1.1
            aspect_score = max(0.0, 1.0 - aspect_distance * 2)

        # Combineer scores
        confidence = (centering_score * 0.4 + size_score * 0.4 + aspect_score * 0.2)

        # Bepaal validatie resultaat
        is_valid = confidence >= self.threshold

        # Genereer feedback bericht
        if is_valid:
            message = "Gezichtspositie is correct"
        elif centering_score < 0.7:
            if center_offset_x > center_offset_y:
                if face_x < img_center_x:
                    message = "Beweeg meer naar rechts in beeld"
                else:
                    message = "Beweeg meer naar links in beeld"
            else:
                if face_y < img_center_y:
                    message = "Beweeg meer naar beneden in beeld"
                else:
                    message = "Beweeg meer naar boven in beeld"
        elif size_score < 0.7:
            if face_size_ratio < self._min_size_ratio:
                message = "Kom dichter bij de camera"
            else:
                message = "Ga verder van de camera af"
        elif aspect_score < 0.7:
            message = "Houd uw hoofd recht en kijk frontaal in de camera"
        else:
            message = "Pas uw positie aan voor een betere pasfoto"

        details = {
            "center_offset_x": float(center_offset_x),
            "center_offset_y": float(center_offset_y),
            "face_size_ratio": float(face_size_ratio),
            "aspect_ratio": float(aspect_ratio),
            "centering_score": float(centering_score),
            "size_score": float(size_score),
            "aspect_score": float(aspect_score)
        }

        return self._create_result(is_valid, confidence, message, details)

    def get_name(self) -> str:
        """Haal naam van validator op"""
        return "FacePositionValidator"

    def get_description(self) -> str:
        """Haal beschrijving van validator op"""
        return "Controleert of het gezicht correct gepositioneerd en gecentreerd is"
