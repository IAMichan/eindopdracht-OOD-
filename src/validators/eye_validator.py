"""
Validator voor ogen zichtbaarheid
"""
from typing import Optional
import numpy as np

from ..models.photo import Photo, ValidationResult, FaceDetectionResult
from .base_validator import BaseValidator, ValidatorConfig


class EyeVisibilityValidator(BaseValidator):
    """
    Valideert of beide ogen goed zichtbaar zijn

    Controleert:
    - Beide ogen zijn open
    - Ogen zijn niet bedekt (bril reflectie, haar, etc.)
    - Pupillen zijn zichtbaar
    """

    def __init__(
        self,
        threshold: float = ValidatorConfig.EYE_VISIBILITY_THRESHOLD,
        min_eye_aspect_ratio: float = ValidatorConfig.EYE_ASPECT_RATIO_MIN
    ):
        """
        Initialiseer eye visibility validator

        Args:
            threshold: Minimum confidence threshold
            min_eye_aspect_ratio: Minimum eye aspect ratio (EAR) voor open oog
        """
        super().__init__(threshold)
        # sla op hoe ver de ogen minimaal open moeten staan
        self._min_eye_aspect_ratio = min_eye_aspect_ratio

    def validate(self, photo: Photo, face_detection: Optional[FaceDetectionResult] = None) -> ValidationResult:
        """
        Valideer zichtbaarheid van ogen

        Args:
            photo: Photo object om te valideren
            face_detection: FaceDetectionResult met landmarks (vereist!)

        Returns:
            ValidationResult met resultaat
        """
        self._validate_image_data(photo)

        # voor deze validator hebben we echt face detection nodig
        if not face_detection or not face_detection.face_found:
            return self._create_result(
                is_valid=False,
                confidence=0.0,
                message="Geen gezicht gedetecteerd",
                details={"error": "no_face_detected"}
            )

        # zonder landmarks kunnen we de ogen niet vinden
        if not face_detection.landmarks:
            return self._create_result(
                is_valid=False,
                confidence=0.0,
                message="Ogen konden niet worden gedetecteerd",
                details={"error": "no_landmarks"}
            )

        landmarks = face_detection.landmarks

        # bereken hoe open elk oog is (eye aspect ratio)
        left_ear = self._calculate_eye_aspect_ratio(landmarks, eye='left')
        right_ear = self._calculate_eye_aspect_ratio(landmarks, eye='right')

        # MediaPipe EAR waarden (gebaseerd op testdata):
        # - Gesloten oog: ~0.15-0.22 (foto met dichte ogen gaf 0.175-0.212!)
        # - Half open: ~0.20-0.25
        # - Normaal open: ~0.25-0.35
        # - Wijd open: ~0.35+

        # Check of ogen open genoeg zijn - verhoogd na testen met echte data
        closed_threshold = 0.22  # onder dit zijn ogen waarschijnlijk dicht
        left_eye_open = left_ear >= closed_threshold
        right_eye_open = right_ear >= closed_threshold

        # Score berekening: alles boven 0.25 is prima
        min_acceptable = 0.25

        if left_ear >= min_acceptable:
            left_score = 1.0  # goed open
        elif left_ear >= closed_threshold:
            # half open - geef lagere maar acceptabele score
            left_score = 0.6 + (left_ear - closed_threshold) / (min_acceptable - closed_threshold) * 0.4
        else:
            # echt dicht
            left_score = max(0.0, left_ear / closed_threshold * 0.5)

        if right_ear >= min_acceptable:
            right_score = 1.0  # goed open
        elif right_ear >= closed_threshold:
            right_score = 0.6 + (right_ear - closed_threshold) / (min_acceptable - closed_threshold) * 0.4
        else:
            right_score = max(0.0, right_ear / closed_threshold * 0.5)

        # check of de ogen bedekt zijn (bv door bril reflectie of haar)
        left_coverage = self._check_eye_coverage(photo, face_detection, eye='left')
        right_coverage = self._check_eye_coverage(photo, face_detection, eye='right')

        coverage_score = (left_coverage + right_coverage) / 2

        # combineer alle scores
        # hoe open de ogen zijn is belangrijker (70%) dan bedekking (30%)
        eye_openness_score = (left_score + right_score) / 2
        confidence = (eye_openness_score * 0.7 + coverage_score * 0.3)

        # Validatie: ogen moeten open zijn EN confidence hoog genoeg
        # Als een oog echt dicht is (onder closed_threshold), is het NIET valid
        is_valid = confidence >= self.threshold and left_eye_open and right_eye_open

        # maak een duidelijke feedback message
        # Let op: "left" in MediaPipe is links vanuit de camera, dus rechts voor de gebruiker
        # We spiegelen dit zodat de feedback klopt voor de gebruiker
        if is_valid:
            message = "Beide ogen zijn goed zichtbaar"
        elif not left_eye_open and not right_eye_open:
            message = "Open uw ogen voor de pasfoto"
        elif not left_eye_open:
            # MediaPipe "left" = rechteroog van de gebruiker (gespiegeld)
            message = "Open uw rechteroog voor de pasfoto"
        elif not right_eye_open:
            # MediaPipe "right" = linkeroog van de gebruiker (gespiegeld)
            message = "Open uw linkeroog voor de pasfoto"
        elif coverage_score < 0.7:
            message = "Zorg dat uw ogen niet bedekt zijn door reflecties of haar"
        else:
            message = "Ogen zijn niet volledig zichtbaar"

        # sla alle details op
        details = {
            "left_eye_aspect_ratio": float(left_ear),
            "right_eye_aspect_ratio": float(right_ear),
            "left_eye_open": left_eye_open,
            "right_eye_open": right_eye_open,
            "left_score": float(left_score),
            "right_score": float(right_score),
            "coverage_score": float(coverage_score)
        }

        return self._create_result(is_valid, confidence, message, details)

    def _calculate_eye_aspect_ratio(self, landmarks: dict, eye: str) -> float:
        """
        Haal eye aspect ratio (EAR) op uit landmarks

        Args:
            landmarks: Dictionary met gezichtslandmarks
            eye: 'left' of 'right'

        Returns:
            Eye aspect ratio (>0.2 = open, <0.2 = gesloten)
        """
        # eye aspect ratio (EAR) is een formule om te meten hoe open een oog is
        # hoe hoger de waarde, hoe verder het oog open staat

        key_prefix = f'{eye}_eye'

        # De face detection model berekent al de ratio (height/width)
        # en slaat deze op als '{eye}_eye_height' (eigenlijk de ratio)
        if f'{key_prefix}_height' in landmarks:
            # Dit is al de eye aspect ratio (hoogte/breedte), niet de ruwe hoogte
            ear = landmarks[f'{key_prefix}_height']
            return ear
        else:
            # als we geen landmarks hebben, weten we het niet - return lage waarde
            # zodat het niet automatisch als "open" wordt gezien
            return 0.10  # lage waarde = onbekend/gesloten

    def _check_eye_coverage(
        self,
        photo: Photo,
        face_detection: FaceDetectionResult,
        eye: str
    ) -> float:
        """
        Check of het oog bedekt is (reflectie, haar, etc.)

        Args:
            photo: Photo object
            face_detection: FaceDetectionResult
            eye: 'left' of 'right'

        Returns:
            Score tussen 0.0 (volledig bedekt) en 1.0 (volledig zichtbaar)
        """
        import cv2

        landmarks = face_detection.landmarks
        key_prefix = f'{eye}_eye'

        # pak het stukje foto waar het oog zit
        if f'{key_prefix}_region' in landmarks:
            # als we weten waar het oog precies zit
            x, y, w, h = landmarks[f'{key_prefix}_region']
            eye_region = photo.image_data[y:y+h, x:x+w]

            # check of we wel wat hebben gevonden
            if eye_region.size == 0:
                return 0.5  # we weten het niet zeker

            # zet om naar grijstinten voor makkelijkere analyse
            if len(eye_region.shape) == 3:
                eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            else:
                eye_gray = eye_region

            # tel hoeveel pixels heel erg licht zijn (reflectie!)
            bright_pixels = np.sum(eye_gray > 240)
            total_pixels = eye_gray.size
            bright_ratio = bright_pixels / total_pixels

            # als er veel super lichte pixels zijn, is er waarschijnlijk reflectie
            if bright_ratio > 0.3:  # meer dan 30% = waarschijnlijk reflectie
                return 0.5
            elif bright_ratio > 0.15:  # tussen 15-30% = beetje reflectie
                return 0.7
            else:
                return 1.0  # ziet er goed uit!
        else:
            # als we geen expliciete regio hebben, wees dan voorzichtig
            return 0.8

    def get_name(self) -> str:
        """Haal naam van validator op"""
        return "EyeVisibilityValidator"

    def get_description(self) -> str:
        """Haal beschrijving van validator op"""
        return "Controleert of beide ogen goed zichtbaar zijn en niet bedekt"
