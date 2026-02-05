"""
Script om validators te testen met BioID dataset

Dit script test de nauwkeurigheid van validators op de BioID dataset
"""
import sys
import os
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.photo import Photo
from src.models.face_detection_model import FaceDetectionModelFactory
from src.validators.brightness_validator import BrightnessValidator
from src.validators.sharpness_validator import SharpnessValidator
from src.validators.face_position_validator import FacePositionValidator
from src.validators.expression_validator import FacialExpressionValidator
from src.validators.eye_validator import EyeVisibilityValidator
from src.services.validation_service import ValidationService


class BioIDTester:
    """Test validators met BioID dataset"""

    def __init__(self, dataset_path: str = "data/bioid/images"):
        """
        Initialiseer tester

        Args:
            dataset_path: Pad naar BioID images
        """
        self.dataset_path = Path(dataset_path)
        self.face_model = FaceDetectionModelFactory.create_default_model()
        self.validation_service = ValidationService()

        # Statistieken
        self.stats = {
            'total_images': 0,
            'faces_detected': 0,
            'validation_results': {}
        }

    def load_images(self, limit: int = None):
        """
        Laad BioID images

        Args:
            limit: Optioneel limit op aantal images

        Returns:
            Lijst met image paths
        """
        if not self.dataset_path.exists():
            print(f"✗ Dataset not found at {self.dataset_path}")
            print("Run: python scripts/download_bioid_dataset.py")
            return []

        image_files = list(self.dataset_path.glob("*.pgm"))

        if limit:
            image_files = image_files[:limit]

        print(f"Found {len(image_files)} images")
        return image_files

    def test_single_image(self, image_path: Path):
        """
        Test validators op één image

        Args:
            image_path: Pad naar image

        Returns:
            Dictionary met resultaten
        """
        # Laad image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            return None

        # Converteer grayscale naar BGR (validators verwachten BGR)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Maak Photo object
        photo = Photo(image_data=img_bgr)

        # Detect face
        face_detection = self.face_model.detect_face(img_bgr)

        # Validate
        validated_photo = self.validation_service.validate_photo(photo)

        return {
            'filename': image_path.name,
            'face_detected': face_detection.face_found,
            'face_confidence': face_detection.confidence,
            'overall_valid': validated_photo.is_valid(),
            'overall_confidence': validated_photo.get_overall_confidence(),
            'validation_results': [
                {
                    'validator': r.validator_name,
                    'valid': r.is_valid,
                    'confidence': r.confidence,
                    'message': r.message
                }
                for r in validated_photo.validation_results
            ]
        }

    def test_all(self, limit: int = None):
        """
        Test alle images

        Args:
            limit: Optioneel limit op aantal images
        """
        print("=" * 70)
        print("BioID Dataset Validation Test")
        print("=" * 70)
        print()

        image_files = self.load_images(limit)

        if not image_files:
            return

        print(f"Testing {len(image_files)} images...\n")

        results = []

        for i, image_path in enumerate(image_files, 1):
            print(f"\rProcessing {i}/{len(image_files)}: {image_path.name}", end='')

            result = self.test_single_image(image_path)

            if result:
                results.append(result)

                # Update stats
                self.stats['total_images'] += 1
                if result['face_detected']:
                    self.stats['faces_detected'] += 1

        print("\n")

        # Analyze results
        self.analyze_results(results)

        return results

    def analyze_results(self, results):
        """
        Analyseer test resultaten

        Args:
            results: Lijst met test resultaten
        """
        print("\n" + "=" * 70)
        print("Results Analysis")
        print("=" * 70)

        # Face detection rate
        face_detection_rate = (self.stats['faces_detected'] / self.stats['total_images']) * 100
        print(f"\nFace Detection:")
        print(f"  Total images: {self.stats['total_images']}")
        print(f"  Faces detected: {self.stats['faces_detected']}")
        print(f"  Detection rate: {face_detection_rate:.1f}%")

        # Validator statistics
        validator_stats = {}

        for result in results:
            if not result['face_detected']:
                continue

            for vr in result['validation_results']:
                validator_name = vr['validator']

                if validator_name not in validator_stats:
                    validator_stats[validator_name] = {
                        'total': 0,
                        'passed': 0,
                        'confidences': []
                    }

                validator_stats[validator_name]['total'] += 1
                if vr['valid']:
                    validator_stats[validator_name]['passed'] += 1
                validator_stats[validator_name]['confidences'].append(vr['confidence'])

        print(f"\nValidator Performance:")
        print(f"{'Validator':<30} {'Pass Rate':<15} {'Avg Confidence':<15}")
        print("-" * 70)

        for validator_name, stats in sorted(validator_stats.items()):
            pass_rate = (stats['passed'] / stats['total']) * 100
            avg_conf = np.mean(stats['confidences'])

            print(f"{validator_name:<30} {pass_rate:>6.1f}%         {avg_conf:>6.3f}")

        # Overall validation
        overall_valid = sum(1 for r in results if r.get('overall_valid', False))
        overall_rate = (overall_valid / len(results)) * 100 if results else 0

        print(f"\nOverall Validation:")
        print(f"  Images passing all validators: {overall_valid}/{len(results)} ({overall_rate:.1f}%)")

        # Average confidence
        avg_overall_conf = np.mean([r['overall_confidence'] for r in results if r['face_detected']])
        print(f"  Average overall confidence: {avg_overall_conf:.3f}")

        print("\n" + "=" * 70)

    def test_specific_validator(self, validator_name: str, limit: int = 100):
        """
        Test een specifieke validator in detail

        Args:
            validator_name: Naam van validator
            limit: Aantal images om te testen
        """
        print(f"\nDetailed test for: {validator_name}")
        print("=" * 70)

        image_files = self.load_images(limit)

        failed_cases = []

        for image_path in image_files:
            result = self.test_single_image(image_path)

            if not result or not result['face_detected']:
                continue

            # Find validator result
            vr = next((v for v in result['validation_results'] if v['validator'] == validator_name), None)

            if vr and not vr['valid']:
                failed_cases.append({
                    'filename': result['filename'],
                    'confidence': vr['confidence'],
                    'message': vr['message']
                })

        print(f"\nFailed cases for {validator_name}:")
        print(f"Total: {len(failed_cases)}")

        if failed_cases:
            print("\nTop 10 failures:")
            for case in sorted(failed_cases, key=lambda x: x['confidence'])[:10]:
                print(f"  {case['filename']}: {case['message']} (conf: {case['confidence']:.3f})")


def main():
    """Main functie"""
    import argparse

    parser = argparse.ArgumentParser(description='Test validators met BioID dataset')
    parser.add_argument('--limit', type=int, default=None, help='Limit aantal images')
    parser.add_argument('--validator', type=str, default=None, help='Test specifieke validator')

    args = parser.parse_args()

    tester = BioIDTester()

    if args.validator:
        tester.test_specific_validator(args.validator, limit=args.limit or 100)
    else:
        tester.test_all(limit=args.limit)


if __name__ == "__main__":
    main()
