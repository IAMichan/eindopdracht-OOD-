"""
Validator voor achtergrond controle
"""
import numpy as np
import cv2
from typing import Optional

from ..models.photo import Photo, ValidationResult, FaceDetectionResult
from .base_validator import BaseValidator, ValidatorConfig


class BackgroundValidator(BaseValidator):
    """
    Valideert dat de achtergrond neutraal/uniform is

    Controleert:
    - Achtergrond is relatief uniform
    - Geen storende objecten
    - Bij voorkeur lichte/witte achtergrond
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialiseer background validator

        Args:
            threshold: Minimum confidence threshold
        """
        super().__init__(threshold)

    def validate(self, photo: Photo, face_detection: Optional[FaceDetectionResult] = None) -> ValidationResult:
        """
        Valideer achtergrond van foto

        Args:
            photo: Photo object om te valideren
            face_detection: FaceDetectionResult met gezichtsinfo

        Returns:
            ValidationResult met resultaat
        """
        self._validate_image_data(photo)

        if not face_detection or not face_detection.face_found or not face_detection.face_bbox:
            # Zonder gezicht kunnen we geen achtergrond bepalen
            return self._create_result(
                is_valid=True,
                confidence=0.5,
                message="Achtergrond kon niet worden geanalyseerd",
                details={"error": "no_face_for_reference"}
            )

        img_height, img_width = photo.image_data.shape[:2]
        x, y, w, h = face_detection.face_bbox

        # Maak een masker voor het gezicht (om uit te sluiten)
        face_mask = np.zeros((img_height, img_width), dtype=np.uint8)

        # Vergroot het gezichtsgebied een beetje om nek/schouders mee te nemen
        expanded_x = max(0, x - int(w * 0.3))
        expanded_y = max(0, y - int(h * 0.2))
        expanded_w = min(img_width - expanded_x, int(w * 1.6))
        expanded_h = min(img_height - expanded_y, int(h * 1.8))

        face_mask[expanded_y:expanded_y+expanded_h, expanded_x:expanded_x+expanded_w] = 255

        # Inverteer masker om achtergrond te krijgen
        bg_mask = cv2.bitwise_not(face_mask)

        # Haal achtergrond pixels op
        bg_pixels = photo.image_data[bg_mask > 0]

        if len(bg_pixels) < 100:
            # Te weinig achtergrond pixels
            return self._create_result(
                is_valid=True,
                confidence=0.7,
                message="Achtergrond is acceptabel",
                details={"note": "small_background"}
            )

        # Converteer naar grijstinten voor uniformiteits analyse
        gray = cv2.cvtColor(photo.image_data, cv2.COLOR_BGR2GRAY)
        bg_gray = gray[bg_mask > 0]

        # Bereken uniformiteit van achtergrond
        bg_std = np.std(bg_gray)
        bg_mean = np.mean(bg_gray)

        # Bereken kleur uniformiteit
        bg_color_std = np.std(bg_pixels, axis=0).mean()

        # Check voor edges in achtergrond (objecten)
        edges = cv2.Canny(gray, 50, 150)
        bg_edges = edges[bg_mask > 0]
        edge_ratio = np.sum(bg_edges > 0) / len(bg_edges) if len(bg_edges) > 0 else 0

        # Score berekening
        # 1. Uniformiteit (lagere std = beter)
        if bg_std < 20:
            uniformity_score = 1.0
        elif bg_std < 40:
            uniformity_score = 0.7
        elif bg_std < 60:
            uniformity_score = 0.5
        else:
            uniformity_score = 0.3

        # 2. Kleur uniformiteit
        if bg_color_std < 15:
            color_score = 1.0
        elif bg_color_std < 30:
            color_score = 0.7
        else:
            color_score = 0.4

        # 3. Edge score (minder edges = beter)
        if edge_ratio < 0.05:
            edge_score = 1.0
        elif edge_ratio < 0.10:
            edge_score = 0.7
        elif edge_ratio < 0.20:
            edge_score = 0.5
        else:
            edge_score = 0.3

        # 4. Lichte achtergrond bonus (voor pasfoto's)
        if bg_mean > 180:  # Lichte achtergrond
            brightness_bonus = 0.1
        elif bg_mean > 150:
            brightness_bonus = 0.05
        else:
            brightness_bonus = 0.0

        # Combineer scores
        confidence = (uniformity_score * 0.4 + color_score * 0.3 + edge_score * 0.3) + brightness_bonus
        confidence = min(1.0, confidence)

        is_valid = confidence >= self.threshold

        # Genereer feedback
        if is_valid:
            message = "Achtergrond is acceptabel"
        elif edge_score < 0.5:
            message = "Achtergrond bevat storende objecten. Gebruik een neutrale achtergrond"
        elif uniformity_score < 0.5:
            message = "Achtergrond is niet uniform genoeg. Gebruik een effen achtergrond"
        else:
            message = "Gebruik een neutrale, lichte achtergrond voor de pasfoto"

        details = {
            "bg_std": float(bg_std),
            "bg_mean": float(bg_mean),
            "bg_color_std": float(bg_color_std),
            "edge_ratio": float(edge_ratio),
            "uniformity_score": float(uniformity_score),
            "color_score": float(color_score),
            "edge_score": float(edge_score)
        }

        return self._create_result(is_valid, confidence, message, details)

    def get_name(self) -> str:
        """Haal naam van validator op"""
        return "BackgroundValidator"

    def get_description(self) -> str:
        """Haal beschrijving van validator op"""
        return "Controleert of de achtergrond neutraal en uniform is"
