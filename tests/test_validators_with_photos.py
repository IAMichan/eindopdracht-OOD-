"""
Test script om validators te testen met echte foto's
"""
import cv2
import os
import sys
from pathlib import Path

# Voeg src toe aan path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.photo import Photo
from src.models.face_detection_model import FaceDetectionModel
from src.validators.brightness_validator import BrightnessValidator
from src.validators.sharpness_validator import SharpnessValidator
from src.validators.face_position_validator import FacePositionValidator
from src.validators.expression_validator import FacialExpressionValidator
from src.validators.eye_validator import EyeVisibilityValidator
from src.validators.reflection_validator import ReflectionValidator
from src.validators.shadow_validator import ShadowValidator
from src.validators.headwear_validator import HeadwearValidator
from src.validators.background_validator import BackgroundValidator


def validate_photo(image_path: str, expected_valid: bool):
    """Test een foto met alle validators (helper functie, geen pytest test)"""
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(image_path)}")
    print(f"Expected: {'APPROVED' if expected_valid else 'REJECTED'}")
    print(f"{'='*60}")

    # Laad image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return False

    # Maak Photo object
    photo = Photo(image_data=image)

    # Face detection
    face_model = FaceDetectionModel(min_detection_confidence=0.4)
    face_result = face_model.detect_face(image)

    print(f"\nFace Detection:")
    print(f"  - Face found: {face_result.face_found}")
    print(f"  - Confidence: {face_result.confidence:.2f}")
    if face_result.landmarks:
        if 'left_eye_height' in face_result.landmarks:
            print(f"  - Left eye EAR: {face_result.landmarks['left_eye_height']:.3f}")
        if 'right_eye_height' in face_result.landmarks:
            print(f"  - Right eye EAR: {face_result.landmarks['right_eye_height']:.3f}")

    # Initialiseer validators
    validators = [
        BrightnessValidator(),
        SharpnessValidator(),
        FacePositionValidator(),
        FacialExpressionValidator(),
        EyeVisibilityValidator(),
        ReflectionValidator(),
        ShadowValidator(),
        HeadwearValidator(),
        BackgroundValidator(),
    ]

    # Run alle validators
    all_valid = True
    print(f"\nValidator Results:")
    for validator in validators:
        result = validator.validate(photo, face_result)
        status = "PASS" if result.is_valid else "FAIL"
        print(f"  [{status}] {validator.get_name()}: {result.message} (conf: {result.confidence:.2f})")

        # Print details voor debugging
        if not result.is_valid and result.details:
            for key, value in result.details.items():
                if isinstance(value, float):
                    print(f"         - {key}: {value:.3f}")
                elif key != 'error':
                    print(f"         - {key}: {value}")

        if not result.is_valid:
            all_valid = False

    # Check resultaat
    actual_result = "APPROVED" if all_valid else "REJECTED"
    expected_result = "APPROVED" if expected_valid else "REJECTED"
    match = actual_result == expected_result

    print(f"\nFinal Result: {actual_result}")
    print(f"Match with expected: {'YES' if match else 'NO'}")

    return match


def main():
    """Test alle foto's in de data folders"""
    base_path = Path(__file__).parent.parent / "data" / "photos"

    approved_path = base_path / "approved"
    rejected_path = base_path / "rejected"

    results = []

    # Test approved photos
    print("\n" + "="*70)
    print("TESTING APPROVED PHOTOS (should all pass)")
    print("="*70)

    if approved_path.exists():
        for photo_file in approved_path.glob("*.jpg"):
            match = validate_photo(str(photo_file), expected_valid=True)
            results.append(("approved", photo_file.name, match))
    else:
        print(f"Approved folder not found: {approved_path}")

    # Test rejected photos
    print("\n" + "="*70)
    print("TESTING REJECTED PHOTOS (should all fail)")
    print("="*70)

    if rejected_path.exists():
        for photo_file in rejected_path.glob("*.jpg"):
            match = validate_photo(str(photo_file), expected_valid=False)
            results.append(("rejected", photo_file.name, match))
    else:
        print(f"Rejected folder not found: {rejected_path}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    correct = sum(1 for r in results if r[2])
    total = len(results)

    print(f"\nTotal photos tested: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {correct/total*100:.1f}%" if total > 0 else "N/A")

    print("\nDetails:")
    for category, filename, match in results:
        status = "OK" if match else "WRONG"
        print(f"  [{status}] {category}/{filename}")


if __name__ == "__main__":
    main()
