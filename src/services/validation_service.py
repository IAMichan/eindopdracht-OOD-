"""
Validation service - orchestreert alle validators
"""
# importeer alle dingen die we nodig hebben
from typing import List, Optional
import logging

from ..models.photo import Photo, PhotoStatus, FaceDetectionResult
from ..models.face_detection_model import FaceDetectionModel, FaceDetectionModelFactory
from ..validators.base_validator import IValidator
# alle 7 validators importeren
from ..validators.brightness_validator import BrightnessValidator
from ..validators.sharpness_validator import SharpnessValidator
from ..validators.face_position_validator import FacePositionValidator
from ..validators.expression_validator import FacialExpressionValidator
from ..validators.eye_validator import EyeVisibilityValidator
from ..validators.reflection_validator import ReflectionValidator
from ..validators.shadow_validator import ShadowValidator


logger = logging.getLogger(__name__)


class ValidationService:
    """
    Service voor het valideren van pasfoto's

    Orchestreert alle validators en face detection model.
    Implementeert Observer Pattern voor real-time feedback.
    """

    def __init__(
        self,
        face_detection_model: Optional[FaceDetectionModel] = None,
        validators: Optional[List[IValidator]] = None
    ):
        """
        Initialiseer validation service

        Args:
            face_detection_model: Face detection model (optioneel, anders default)
            validators: Lijst met validators (optioneel, anders default set)
        """
        # als er geen model is meegegeven, maak dan een standaard model aan
        if face_detection_model is None:
            self._face_detection_model = FaceDetectionModelFactory.create_default_model()
        else:
            self._face_detection_model = face_detection_model

        # als er geen validators zijn meegegeven, maak ze dan zelf aan
        if validators is None:
            self._validators = self._create_default_validators()
        else:
            self._validators = validators

        # lijst voor observers (voor real-time updates naar de GUI)
        self._observers: List = []

        logger.info(f"ValidationService initialized with {len(self._validators)} validators")

    def _create_default_validators(self) -> List[IValidator]:
        """
        Creëer default set van validators

        Returns:
            Lijst met validator instances
        """
        # maak alle 7 validators aan, deze checken verschillende dingen
        return [
            BrightnessValidator(),      # check of de foto niet te donker/licht is
            SharpnessValidator(),       # kijk of de foto scherp genoeg is
            FacePositionValidator(),    # check of het gezicht goed gepositioneerd is
            FacialExpressionValidator(),  # kijk of de gezichts uitdrukking neutraal is
            EyeVisibilityValidator(),   # check of beide ogen zichtbaar zijn
            ReflectionValidator(),      # zoek naar reflecties op de foto
            ShadowValidator()           # check voor schaduwen
        ]

    def add_validator(self, validator: IValidator) -> None:
        """
        Voeg een validator toe

        Args:
            validator: Validator om toe te voegen
        """
        # voeg een extra validator toe aan de lijst
        self._validators.append(validator)
        logger.info(f"Added validator: {validator.get_name()}")

    def remove_validator(self, validator_name: str) -> bool:
        """
        Verwijder een validator op basis van naam

        Args:
            validator_name: Naam van validator om te verwijderen

        Returns:
            True als verwijderd, False als niet gevonden
        """
        # zoek de validator en gooi hem eruit
        for i, validator in enumerate(self._validators):
            if validator.get_name() == validator_name:
                del self._validators[i]
                logger.info(f"Removed validator: {validator_name}")
                return True
        return False

    def get_validators(self) -> List[IValidator]:
        """
        Haal alle validators op

        Returns:
            Lijst met validators
        """
        return self._validators.copy()

    def validate_photo(self, photo: Photo) -> Photo:
        """
        Valideer een foto met alle validators

        Args:
            photo: Photo object om te valideren

        Returns:
            Photo object met validation results
        """
        logger.info("Starting photo validation")

        # stap 1: zoek eerst het gezicht in de foto
        self._notify_observers("face_detection", "Detecting face...")
        face_detection = self._face_detection_model.detect_face(photo.image_data)

        if not face_detection.face_found:
            logger.warning("No face detected in photo")

        # stap 2: run alle validators over de foto
        for i, validator in enumerate(self._validators):
            validator_name = validator.get_name()
            # stuur een update naar de GUI
            self._notify_observers(
                "validation_progress",
                f"Running {validator_name}... ({i+1}/{len(self._validators)})"
            )

            try:
                # laat de validator zijn ding doen
                result = validator.validate(photo, face_detection)

                # voeg het resultaat toe aan de foto
                photo.add_validation_result(result)

                # vertel de observers wat er gebeurd is
                self._notify_observers("validation_result", {
                    "validator": validator_name,
                    "result": result
                })

                logger.info(
                    f"{validator_name}: {'PASS' if result.is_valid else 'FAIL'} "
                    f"(confidence: {result.confidence:.2f})"
                )

            except Exception as e:
                logger.error(f"Error in {validator_name}: {e}", exc_info=True)
                # ga gwn door met de volgende validator

        # stap 3: update de status van de foto (approved/rejected/pending)
        photo.update_status()

        # stap 4: vertel iedereen dat we klaar zijn
        self._notify_observers("validation_complete", {
            "photo": photo,
            "is_valid": photo.is_valid(),
            "confidence": photo.get_overall_confidence()
        })

        logger.info(
            f"Validation complete. Status: {photo.status.value}, "
            f"Overall confidence: {photo.get_overall_confidence():.2f}"
        )

        return photo

    def add_observer(self, observer) -> None:
        """
        Voeg een observer toe voor real-time updates

        Args:
            observer: Observer object met update(event_type, data) methode
        """
        # voeg observer toe zodat die updates krijgt
        self._observers.append(observer)

    def remove_observer(self, observer) -> None:
        """
        Verwijder een observer

        Args:
            observer: Observer om te verwijderen
        """
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify_observers(self, event_type: str, data) -> None:
        """
        Notify alle observers van een event

        Args:
            event_type: Type van event
            data: Event data
        """
        # vertel alle observers wat er gebeurt (bv de GUI)
        for observer in self._observers:
            try:
                observer.update(event_type, data)
            except Exception as e:
                logger.error(f"Error notifying observer: {e}", exc_info=True)

    def get_validation_summary(self, photo: Photo) -> dict:
        """
        Haal een samenvatting van validatie resultaten op

        Args:
            photo: Gevalideerde photo

        Returns:
            Dictionary met samenvatting
        """
        # maak een mooie samenvatting van alle resultaten
        return {
            "status": photo.status.value,
            "is_valid": photo.is_valid(),
            "overall_confidence": photo.get_overall_confidence(),
            "total_validators": len(photo.validation_results),
            "passed_validators": sum(1 for r in photo.validation_results if r.is_valid),
            "failed_validators": len(photo.get_failed_validations()),
            "results": [
                {
                    "validator": r.validator_name,
                    "valid": r.is_valid,
                    "confidence": r.confidence,
                    "message": r.message
                }
                for r in photo.validation_results
            ]
        }
