"""
Validator voor gezichtsuitdrukking controle
"""
from typing import Optional

from ..models.photo import Photo, ValidationResult, FaceDetectionResult
from .base_validator import BaseValidator, ValidatorConfig


class FacialExpressionValidator(BaseValidator):
    """
    Valideert of de gezichtsuitdrukking neutraal is

    Controleert:
    - Neutrale gezichtsuitdrukking
    - Mond is gesloten
    - Geen extreme emoties
    """

    def __init__(
        self,
        threshold: float = ValidatorConfig.EXPRESSION_THRESHOLD,
        mouth_open_threshold: float = ValidatorConfig.MOUTH_OPEN_THRESHOLD
    ):
        """
        Initialiseer facial expression validator

        Args:
            threshold: Minimum confidence threshold
            mouth_open_threshold: Maximum toegestane mond opening ratio
        """
        super().__init__(threshold)
        self._mouth_open_threshold = mouth_open_threshold

    def validate(self, photo: Photo, face_detection: Optional[FaceDetectionResult] = None) -> ValidationResult:
        """
        Valideer gezichtsuitdrukking

        Args:
            photo: Photo object om te valideren
            face_detection: FaceDetectionResult met landmarks (vereist!)

        Returns:
            ValidationResult met resultaat
        """
        self._validate_image_data(photo)

        # Face detection met landmarks is verplicht
        if not face_detection or not face_detection.face_found:
            return self._create_result(
                is_valid=False,
                confidence=0.0,
                message="Geen gezicht gedetecteerd",
                details={"error": "no_face_detected"}
            )

        if not face_detection.landmarks:
            return self._create_result(
                is_valid=False,
                confidence=0.0,
                message="Gezichtskenmerken konden niet worden gedetecteerd",
                details={"error": "no_landmarks"}
            )

        landmarks = face_detection.landmarks

        # Bereken mond aspect ratio (MAR)
        # MAR = hoogte / breedte van de mond
        # Gesloten mond: ~0.05-0.15
        # Licht open: ~0.15-0.30
        # Wijd open: ~0.30+
        mouth_aspect_ratio = self._calculate_mouth_aspect_ratio(landmarks)

        # Check of mond acceptabel is (niet wijd open)
        # threshold is 0.5, dus alleen echt wijd open is een probleem
        if mouth_aspect_ratio <= self._mouth_open_threshold:
            mouth_score = 1.0
            mouth_closed = True
        else:
            # Mond is te wijd open - geef lagere score
            excess = mouth_aspect_ratio - self._mouth_open_threshold
            mouth_score = max(0.0, 1.0 - (excess / 0.3))  # gradueel afnemen
            mouth_closed = False

        # Check voor extreme gezichtsuitdrukkingen via landmarks
        # Bijvoorbeeld: wenkbrauwen te hoog (verrassing), te laag (boos), etc.
        expression_score = self._analyze_facial_features(landmarks)

        # Combineer scores
        confidence = (mouth_score * 0.6 + expression_score * 0.4)

        # Bepaal validatie resultaat
        is_valid = confidence >= self.threshold

        # Genereer feedback bericht
        if is_valid:
            message = "Gezichtsuitdrukking is neutraal en correct"
        elif not mouth_closed:
            message = "Sluit uw mond voor de pasfoto"
        elif expression_score < 0.6:
            message = "Neem een neutrale gezichtsuitdrukking aan"
        else:
            message = "Gezichtsuitdrukking is niet volledig neutraal"

        details = {
            "mouth_aspect_ratio": float(mouth_aspect_ratio),
            "mouth_closed": mouth_closed,
            "mouth_score": float(mouth_score),
            "expression_score": float(expression_score)
        }

        return self._create_result(is_valid, confidence, message, details)

    def _calculate_mouth_aspect_ratio(self, landmarks: dict) -> float:
        """
        Bereken mouth aspect ratio (MAR)

        Args:
            landmarks: Dictionary met gezichtslandmarks

        Returns:
            Mouth aspect ratio (0.0 = gesloten, >0.3 = open)
        """
        # mouth_upper en mouth_lower zijn Y-coordinaten (pixels)
        # mouth_lower heeft HOGERE y-waarde dan mouth_upper (want y groeit naar beneden)
        # Dus: mouth_height = mouth_lower - mouth_upper

        if 'mouth_upper' in landmarks and 'mouth_lower' in landmarks and 'mouth_width' in landmarks:
            # Bereken hoogte: lower y - upper y (lower is groter in pixel coords)
            mouth_height = abs(landmarks['mouth_lower'] - landmarks['mouth_upper'])
            mouth_width = landmarks['mouth_width']
        else:
            # Geen landmarks beschikbaar - ga uit van gesloten mond
            return 0.1  # Conservatieve waarde voor gesloten mond

        if mouth_width == 0:
            return 0.0

        mar = mouth_height / mouth_width
        return mar

    def _analyze_facial_features(self, landmarks: dict) -> float:
        """
        Analyseer gezichtskenmerken voor extreme expressies

        Args:
            landmarks: Dictionary met gezichtslandmarks

        Returns:
            Score tussen 0.0 en 1.0 (1.0 = neutraal)
        """
        # Voor een neutrale expressie verwachten we:
        # - Wenkbrauwen in rustige positie
        # - Geen grote asymmetrie
        # - Ooghoeken niet te ver omhoog/omlaag

        score = 1.0

        # Check eyebrow position (if available)
        if 'eyebrow_raise' in landmarks:
            eyebrow_raise = landmarks['eyebrow_raise']
            if eyebrow_raise > 0.3:  # Te hoog opgetrokken
                score -= 0.3

        # Check voor asymmetrie (lachen, scheef kijken)
        if 'mouth_symmetry' in landmarks:
            symmetry = landmarks['mouth_symmetry']
            if symmetry < 0.7:  # Te asymmetrisch
                score -= 0.2

        return max(0.0, min(1.0, score))

    def get_name(self) -> str:
        """Haal naam van validator op"""
        return "FacialExpressionValidator"

    def get_description(self) -> str:
        """Haal beschrijving van validator op"""
        return "Controleert of de gezichtsuitdrukking neutraal is en de mond gesloten"
