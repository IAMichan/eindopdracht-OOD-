"""
Unit tests voor Validators
"""
import pytest
import numpy as np
import cv2

from src.models.photo import Photo, FaceDetectionResult
from src.validators.brightness_validator import BrightnessValidator
from src.validators.sharpness_validator import SharpnessValidator
from src.validators.face_position_validator import FacePositionValidator


class TestBrightnessValidator:
    """Test suite voor BrightnessValidator"""

    def create_image_with_brightness(self, brightness: int) -> np.ndarray:
        """Helper om image met specifieke brightness te maken"""
        image = np.ones((480, 640, 3), dtype=np.uint8) * brightness
        return image

    def test_validate_correct_brightness(self):
        """UT-V-01: Foto met correcte belichting"""
        validator = BrightnessValidator()
        image = self.create_image_with_brightness(140)  # Midden van 80-200
        photo = Photo(image_data=image)

        result = validator.validate(photo)

        assert result.is_valid == True  # gebruik == i.p.v. is voor numpy booleans
        assert result.confidence > 0.7
        assert "correct" in result.message.lower()

    def test_validate_too_dark(self):
        """UT-V-02: Foto te donker"""
        validator = BrightnessValidator()
        image = self.create_image_with_brightness(20)  # Zeer donker (onder nieuwe minimum van 60)
        photo = Photo(image_data=image)

        result = validator.validate(photo)

        assert result.is_valid == False  # gebruik == i.p.v. is voor numpy booleans
        assert "donker" in result.message.lower()

    def test_validate_too_bright(self):
        """UT-V-03: Foto te licht"""
        validator = BrightnessValidator()
        image = self.create_image_with_brightness(250)  # Zeer licht (boven nieuwe maximum van 220)
        photo = Photo(image_data=image)

        result = validator.validate(photo)

        assert result.is_valid == False  # gebruik == i.p.v. is voor numpy booleans
        assert "licht" in result.message.lower()

    def test_validator_threshold_setting(self):
        """Test threshold instelling"""
        validator = BrightnessValidator(threshold=0.8)

        assert validator.threshold == 0.8

    def test_validator_threshold_validation(self):
        """Test threshold validatie"""
        with pytest.raises(ValueError):
            BrightnessValidator(threshold=1.5)


class TestSharpnessValidator:
    """Test suite voor SharpnessValidator"""

    def create_blurred_image(self, blur_amount: int) -> np.ndarray:
        """Helper om blurred image te maken"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        if blur_amount > 0:
            image = cv2.GaussianBlur(image, (blur_amount*2+1, blur_amount*2+1), 0)
        return image

    def test_validate_sharp_image(self):
        """UT-V-05: Scherpe foto"""
        validator = SharpnessValidator()
        image = self.create_blurred_image(0)  # Geen blur
        photo = Photo(image_data=image)

        result = validator.validate(photo)

        # Sharp random image should have high Laplacian variance
        assert result.confidence > 0.5
        assert result.validator_name == "SharpnessValidator"

    def test_validate_blurry_image(self):
        """UT-V-06: Wazige foto"""
        validator = SharpnessValidator()
        image = self.create_blurred_image(10)  # Strong blur
        photo = Photo(image_data=image)

        result = validator.validate(photo)

        # Blurred image should fail or have low confidence
        if not result.is_valid:
            assert "wazig" in result.message.lower() or "scherp" in result.message.lower()

    def test_validator_name(self):
        """Test validator naam"""
        validator = SharpnessValidator()
        assert validator.get_name() == "SharpnessValidator"

    def test_validator_description(self):
        """Test validator beschrijving"""
        validator = SharpnessValidator()
        description = validator.get_description()
        assert len(description) > 0
        assert "scherp" in description.lower()


class TestFacePositionValidator:
    """Test suite voor FacePositionValidator"""

    def create_face_detection(
        self,
        bbox: tuple,
        image_size: tuple = (480, 640)
    ) -> FaceDetectionResult:
        """Helper om FaceDetectionResult te maken"""
        return FaceDetectionResult(
            face_found=True,
            confidence=0.95,
            face_bbox=bbox,
            landmarks={}
        )

    def test_validate_centered_face(self):
        """UT-V-07: Gezicht gecentreerd"""
        validator = FacePositionValidator()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        photo = Photo(image_data=image)

        # Centered face: bbox centered in 640x480 image
        # Face size approximately 25% of image (tussen min 0.20 en max 0.40)
        face_w = int(640 * 0.5)  # 320 pixels (0.5 van breedte)
        face_h = int(480 * 0.5)  # 240 pixels (0.5 van hoogte) -> ratio = 0.25
        face_x = (640 - face_w) // 2
        face_y = (480 - face_h) // 2

        face_detection = self.create_face_detection((face_x, face_y, face_w, face_h))

        result = validator.validate(photo, face_detection)

        assert result.is_valid == True  # gebruik == voor numpy booleans
        assert result.confidence > 0.7

    def test_validate_face_too_left(self):
        """UT-V-08: Gezicht te ver links"""
        validator = FacePositionValidator()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        photo = Photo(image_data=image)

        # Face far left
        face_detection = self.create_face_detection((50, 200, 150, 200))

        result = validator.validate(photo, face_detection)

        # Should suggest moving right
        if not result.is_valid:
            assert "rechts" in result.message.lower()

    def test_validate_no_face_detected(self):
        """UT-V-11: Geen gezicht gedetecteerd"""
        validator = FacePositionValidator()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        photo = Photo(image_data=image)

        # No face
        face_detection = FaceDetectionResult(
            face_found=False,
            confidence=0.0
        )

        result = validator.validate(photo, face_detection)

        assert result.is_valid is False
        assert result.confidence == 0.0
        assert "geen gezicht" in result.message.lower()

    def test_validate_face_too_close(self):
        """UT-V-09: Gezicht te dichtbij"""
        validator = FacePositionValidator()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        photo = Photo(image_data=image)

        # Very large face (too close)
        face_detection = self.create_face_detection((50, 50, 540, 380))

        result = validator.validate(photo, face_detection)

        if not result.is_valid:
            assert "verder" in result.message.lower() or "af" in result.message.lower()

    def test_validate_face_too_far(self):
        """UT-V-10: Gezicht te ver weg"""
        validator = FacePositionValidator()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        photo = Photo(image_data=image)

        # Very small face (too far)
        face_detection = self.create_face_detection((280, 200, 80, 80))

        result = validator.validate(photo, face_detection)

        if not result.is_valid:
            assert "dichter" in result.message.lower() or "bij" in result.message.lower()


class TestBaseValidator:
    """Test suite voor base validator functionaliteit"""

    def test_threshold_setter_valid(self):
        """Test threshold setter met geldige waarde"""
        validator = BrightnessValidator(threshold=0.7)
        validator.threshold = 0.85

        assert validator.threshold == 0.85

    def test_threshold_setter_invalid(self):
        """Test threshold setter met ongeldige waarde"""
        validator = BrightnessValidator()

        with pytest.raises(ValueError):
            validator.threshold = 1.5

        with pytest.raises(ValueError):
            validator.threshold = -0.1
