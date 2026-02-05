"""
Domain model voor pasfoto's
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import numpy as np


class PhotoStatus(Enum):
    """Status van een pasfoto"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class ValidationResult:
    """
    Resultaat van een validatie check

    Attributes:
        validator_name: Naam van de validator
        is_valid: Of de validatie geslaagd is
        confidence: Confidence score (0.0 - 1.0)
        message: Feedback bericht voor gebruiker
        details: Optionele extra details
    """
    validator_name: str
    is_valid: bool
    confidence: float
    message: str
    details: Optional[dict] = None

    def __post_init__(self):
        """Valideer confidence score"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class Photo:
    """
    Domain model voor een pasfoto

    Attributes:
        id: Unieke identifier
        image_data: NumPy array met image data (optioneel, wordt later geladen)
        timestamp: Wanneer foto gemaakt is
        status: Huidige status van de foto
        validation_results: Lijst met validatie resultaten
        file_path: Pad naar opgeslagen bestand (indien opgeslagen)
        metadata: Extra metadata (camera settings, etc.)
    """
    image_data: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=datetime.now)
    id: Optional[str] = None
    status: PhotoStatus = PhotoStatus.PENDING
    validation_results: list[ValidationResult] = field(default_factory=list)
    file_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def add_validation_result(self, result: ValidationResult) -> None:
        """
        Voeg een validatie resultaat toe

        Args:
            result: ValidationResult object
        """
        self.validation_results.append(result)

    def is_valid(self) -> bool:
        """
        Check of alle validaties geslaagd zijn

        Returns:
            True als alle validaties geslaagd zijn
        """
        if not self.validation_results:
            return False
        return all(result.is_valid for result in self.validation_results)

    def get_overall_confidence(self) -> float:
        """
        Bereken gemiddelde confidence score

        Returns:
            Gemiddelde confidence (0.0 - 1.0)
        """
        if not self.validation_results:
            return 0.0
        return sum(r.confidence for r in self.validation_results) / len(self.validation_results)

    def get_failed_validations(self) -> list[ValidationResult]:
        """
        Haal alle gefaalde validaties op

        Returns:
            Lijst met gefaalde ValidationResult objecten
        """
        return [r for r in self.validation_results if not r.is_valid]

    def update_status(self) -> None:
        """Update status gebaseerd op validatie resultaten"""
        if self.is_valid():
            self.status = PhotoStatus.APPROVED
        elif self.validation_results:
            self.status = PhotoStatus.REJECTED
        else:
            self.status = PhotoStatus.PENDING


@dataclass
class FaceDetectionResult:
    """
    Resultaat van gezichtsdetectie

    Attributes:
        face_found: Of een gezicht is gedetecteerd
        face_bbox: Bounding box van gezicht (x, y, width, height)
        landmarks: Gezichtslandmarks (ogen, neus, mond, etc.)
        confidence: Confidence score van detectie
    """
    face_found: bool
    confidence: float
    face_bbox: Optional[tuple[int, int, int, int]] = None
    landmarks: Optional[dict] = None

    def get_face_center(self) -> Optional[tuple[int, int]]:
        """
        Bereken centrum van gezicht

        Returns:
            (x, y) coordinaten van centrum, of None
        """
        if not self.face_bbox:
            return None
        x, y, w, h = self.face_bbox
        return (x + w // 2, y + h // 2)

    def get_face_size(self) -> Optional[int]:
        """
        Bereken grootte van gezicht

        Returns:
            Oppervlakte van bounding box, of None
        """
        if not self.face_bbox:
            return None
        _, _, w, h = self.face_bbox
        return w * h
