"""
Unit tests voor Photo domain model
"""
import pytest
import numpy as np
from datetime import datetime

from src.models.photo import Photo, PhotoStatus, ValidationResult, FaceDetectionResult


class TestPhoto:
    """Test suite voor Photo class"""

    def test_create_photo_with_valid_data(self):
        """UT-DM-01: Creëer Photo object met geldige data"""
        image_data = np.zeros((480, 640, 3), dtype=np.uint8)
        photo = Photo(image_data=image_data)

        assert photo.image_data is not None
        assert photo.status == PhotoStatus.PENDING
        assert isinstance(photo.timestamp, datetime)
        assert len(photo.validation_results) == 0

    def test_add_validation_result(self):
        """UT-DM-02: add_validation_result() voegt result toe"""
        photo = Photo(image_data=np.zeros((480, 640, 3), dtype=np.uint8))
        result = ValidationResult(
            validator_name="TestValidator",
            is_valid=True,
            confidence=0.9,
            message="Test passed"
        )

        photo.add_validation_result(result)

        assert len(photo.validation_results) == 1
        assert photo.validation_results[0] == result

    def test_is_valid_with_all_passed(self):
        """UT-DM-03: is_valid() met alle passed validators"""
        photo = Photo(image_data=np.zeros((480, 640, 3), dtype=np.uint8))

        # Add multiple passed validations
        for i in range(3):
            result = ValidationResult(
                validator_name=f"Validator{i}",
                is_valid=True,
                confidence=0.9,
                message="Passed"
            )
            photo.add_validation_result(result)

        assert photo.is_valid() is True

    def test_is_valid_with_failed_validators(self):
        """UT-DM-04: is_valid() met gefaalde validators"""
        photo = Photo(image_data=np.zeros((480, 640, 3), dtype=np.uint8))

        # Add one passed, one failed
        photo.add_validation_result(ValidationResult(
            validator_name="Validator1",
            is_valid=True,
            confidence=0.9,
            message="Passed"
        ))
        photo.add_validation_result(ValidationResult(
            validator_name="Validator2",
            is_valid=False,
            confidence=0.3,
            message="Failed"
        ))

        assert photo.is_valid() is False

    def test_get_overall_confidence(self):
        """UT-DM-05: get_overall_confidence() berekening"""
        photo = Photo(image_data=np.zeros((480, 640, 3), dtype=np.uint8))

        # Add results with known confidences
        photo.add_validation_result(ValidationResult(
            validator_name="V1", is_valid=True, confidence=0.8, message="ok"
        ))
        photo.add_validation_result(ValidationResult(
            validator_name="V2", is_valid=True, confidence=0.6, message="ok"
        ))
        photo.add_validation_result(ValidationResult(
            validator_name="V3", is_valid=True, confidence=1.0, message="ok"
        ))

        expected_confidence = (0.8 + 0.6 + 1.0) / 3
        assert photo.get_overall_confidence() == pytest.approx(expected_confidence)

    def test_update_status_approved(self):
        """UT-DM-06: update_status() met passed validations"""
        photo = Photo(image_data=np.zeros((480, 640, 3), dtype=np.uint8))

        photo.add_validation_result(ValidationResult(
            validator_name="V1", is_valid=True, confidence=0.9, message="ok"
        ))

        photo.update_status()

        assert photo.status == PhotoStatus.APPROVED

    def test_update_status_rejected(self):
        """UT-DM-07: update_status() met failed validations"""
        photo = Photo(image_data=np.zeros((480, 640, 3), dtype=np.uint8))

        photo.add_validation_result(ValidationResult(
            validator_name="V1", is_valid=False, confidence=0.3, message="failed"
        ))

        photo.update_status()

        assert photo.status == PhotoStatus.REJECTED

    def test_get_failed_validations(self):
        """Test get_failed_validations() methode"""
        photo = Photo(image_data=np.zeros((480, 640, 3), dtype=np.uint8))

        passed_result = ValidationResult(
            validator_name="V1", is_valid=True, confidence=0.9, message="ok"
        )
        failed_result1 = ValidationResult(
            validator_name="V2", is_valid=False, confidence=0.3, message="failed"
        )
        failed_result2 = ValidationResult(
            validator_name="V3", is_valid=False, confidence=0.4, message="failed"
        )

        photo.add_validation_result(passed_result)
        photo.add_validation_result(failed_result1)
        photo.add_validation_result(failed_result2)

        failed = photo.get_failed_validations()

        assert len(failed) == 2
        assert failed_result1 in failed
        assert failed_result2 in failed
        assert passed_result not in failed


class TestValidationResult:
    """Test suite voor ValidationResult class"""

    def test_create_with_valid_confidence(self):
        """Test ValidationResult met geldige confidence"""
        result = ValidationResult(
            validator_name="Test",
            is_valid=True,
            confidence=0.85,
            message="Good"
        )

        assert result.confidence == 0.85

    def test_create_with_confidence_below_zero(self):
        """UT-DM-08: Creëer met confidence < 0"""
        with pytest.raises(ValueError, match="Confidence must be between"):
            ValidationResult(
                validator_name="Test",
                is_valid=True,
                confidence=-0.1,
                message="Invalid"
            )

    def test_create_with_confidence_above_one(self):
        """UT-DM-09: Creëer met confidence > 1"""
        with pytest.raises(ValueError, match="Confidence must be between"):
            ValidationResult(
                validator_name="Test",
                is_valid=True,
                confidence=1.5,
                message="Invalid"
            )

    def test_validation_result_with_details(self):
        """Test ValidationResult met extra details"""
        details = {"score": 0.9, "reason": "test"}
        result = ValidationResult(
            validator_name="Test",
            is_valid=True,
            confidence=0.9,
            message="ok",
            details=details
        )

        assert result.details == details


class TestFaceDetectionResult:
    """Test suite voor FaceDetectionResult class"""

    def test_get_face_center(self):
        """UT-DM-10: get_face_center() berekening"""
        result = FaceDetectionResult(
            face_found=True,
            confidence=0.95,
            face_bbox=(100, 200, 300, 400)  # x, y, width, height
        )

        center = result.get_face_center()

        assert center == (250, 400)  # (100 + 300/2, 200 + 400/2)

    def test_get_face_center_no_bbox(self):
        """Test get_face_center() zonder bounding box"""
        result = FaceDetectionResult(
            face_found=False,
            confidence=0.0
        )

        center = result.get_face_center()

        assert center is None

    def test_get_face_size(self):
        """Test get_face_size() berekening"""
        result = FaceDetectionResult(
            face_found=True,
            confidence=0.95,
            face_bbox=(100, 200, 300, 400)
        )

        size = result.get_face_size()

        assert size == 120000  # 300 * 400

    def test_get_face_size_no_bbox(self):
        """Test get_face_size() zonder bounding box"""
        result = FaceDetectionResult(
            face_found=False,
            confidence=0.0
        )

        size = result.get_face_size()

        assert size is None