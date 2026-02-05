"""
Demo script voor Pasfoto Validatie Applicatie

Test het complete systeem zonder GUI/camera
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

from src.models.photo import Photo, PhotoStatus
from src.models.face_detection_model import FaceDetectionModelFactory
from src.services.validation_service import ValidationService
from src.services.storage_service import StorageService
from src.repositories.photo_repository import SQLitePhotoRepository


def create_test_image():
    """Maak een test image met een gezicht"""
    # Voor nu een simpel beeld, later vervangen met echte foto
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128

    # Teken een simpel "gezicht" voor testing
    cv2.circle(img, (320, 240), 100, (200, 200, 200), -1)  # Hoofd
    cv2.circle(img, (280, 220), 15, (50, 50, 50), -1)       # Linker oog
    cv2.circle(img, (360, 220), 15, (50, 50, 50), -1)       # Rechter oog
    cv2.ellipse(img, (320, 270), (30, 15), 0, 0, 180, (50, 50, 50), 2)  # Mond

    return img


def demo_basic_validation():
    """Demo: Basis validatie zonder face detection"""
    print("=" * 70)
    print("DEMO 1: Basis Validatie (zonder face detection)")
    print("=" * 70)

    # Maak test image
    img = create_test_image()
    photo = Photo(image_data=img)

    # Valideer met alleen brightness en sharpness (geen face detection nodig)
    from src.validators.brightness_validator import BrightnessValidator
    from src.validators.sharpness_validator import SharpnessValidator

    validators = [
        BrightnessValidator(),
        SharpnessValidator()
    ]

    for validator in validators:
        result = validator.validate(photo)
        status = "PASS" if result.is_valid else "FAIL"
        print(f"\n{validator.get_name()}:")
        print(f"  Status: {status}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Message: {result.message}")

        photo.add_validation_result(result)

    photo.update_status()
    print(f"\nOverall Status: {photo.status.value}")
    print(f"Overall Confidence: {photo.get_overall_confidence():.3f}")


def demo_storage():
    """Demo: Storage systeem"""
    print("\n" + "=" * 70)
    print("DEMO 2: Storage Systeem")
    print("=" * 70)

    # Maak test photo
    img = create_test_image()
    photo = Photo(image_data=img)

    # Valideer (simplified)
    from src.validators.brightness_validator import BrightnessValidator
    validator = BrightnessValidator()
    result = validator.validate(photo)
    photo.add_validation_result(result)
    photo.update_status()

    # Sla op
    storage = StorageService()
    success = storage.save_photo(photo)

    print(f"\nSave photo: {'SUCCESS' if success else 'FAILED'}")
    if success:
        print(f"Photo ID: {photo.id}")
        print(f"File path: {photo.file_path}")

        # Load terug
        loaded = storage.load_photo(photo.id)
        print(f"\nLoad photo: {'SUCCESS' if loaded else 'FAILED'}")

        if loaded:
            print(f"Loaded {len(loaded.validation_results)} validation results")

        # Statistics
        stats = storage.get_storage_statistics()
        print(f"\nStorage Statistics:")
        print(f"  Total photos: {stats.get('total_photos', 0)}")
        print(f"  Storage size: {stats.get('total_size_mb', 0)} MB")


def demo_repository():
    """Demo: Repository pattern"""
    print("\n" + "=" * 70)
    print("DEMO 3: Repository Pattern")
    print("=" * 70)

    repo = SQLitePhotoRepository()

    # Get statistics
    stats = repo.get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"  Total photos: {stats.get('total_photos', 0)}")
    print(f"  Approved: {stats.get('by_status', {}).get('approved', 0)}")
    print(f"  Rejected: {stats.get('by_status', {}).get('rejected', 0)}")
    print(f"  Pending: {stats.get('by_status', {}).get('pending', 0)}")
    print(f"  Avg confidence: {stats.get('average_confidence', 0):.3f}")


def demo_validation_service():
    """Demo: Validation service met observer pattern"""
    print("\n" + "=" * 70)
    print("DEMO 4: Validation Service (Observer Pattern)")
    print("=" * 70)

    class ConsoleObserver:
        """Observer die updates print naar console"""
        def update(self, event_type, data):
            if event_type == "validation_progress":
                print(f"  {data}")
            elif event_type == "validation_result":
                result = data['result']
                status = "PASS" if result.is_valid else "FAIL"
                print(f"    [{status}] {data['validator']}: {result.message}")

    # Maak service met observer
    service = ValidationService()
    observer = ConsoleObserver()
    service.add_observer(observer)

    # Maak test photo
    img = create_test_image()
    photo = Photo(image_data=img)

    print("\nValidating photo...")
    validated = service.validate_photo(photo)

    print(f"\nFinal Result:")
    print(f"  Valid: {validated.is_valid()}")
    print(f"  Confidence: {validated.get_overall_confidence():.3f}")
    print(f"  Status: {validated.status.value}")


def demo_design_patterns():
    """Demo: Design patterns in actie"""
    print("\n" + "=" * 70)
    print("DEMO 5: Design Patterns")
    print("=" * 70)

    # 1. Factory Pattern
    print("\n1. Factory Pattern (FaceDetectionModelFactory):")
    model1 = FaceDetectionModelFactory.create_default_model()
    model2 = FaceDetectionModelFactory.create_mediapipe_model(
        min_detection_confidence=0.8
    )
    print(f"   [OK] Created 2 models via factory")

    # 2. Strategy Pattern
    print("\n2. Strategy Pattern (Validators as strategies):")
    from src.validators.brightness_validator import BrightnessValidator
    from src.validators.sharpness_validator import SharpnessValidator

    validators = [BrightnessValidator(), SharpnessValidator()]
    print(f"   [OK] Loaded {len(validators)} validator strategies")

    # Validators zijn interchangeable
    for validator in validators:
        print(f"     - {validator.get_name()}: {validator.get_description()}")

    # 3. Repository Pattern
    print("\n3. Repository Pattern (Data access abstraction):")
    from src.repositories.photo_repository import IPhotoRepository, SQLitePhotoRepository

    repo: IPhotoRepository = SQLitePhotoRepository()  # Could be swapped
    print(f"   [OK] Using SQLitePhotoRepository (implements IPhotoRepository)")
    print(f"   [OK] Can be swapped with PostgreSQL or other implementation")

    # 4. Observer Pattern
    print("\n4. Observer Pattern (ValidationService -> Observers):")
    service = ValidationService()

    class DemoObserver:
        def __init__(self, name):
            self.name = name
        def update(self, event_type, data):
            pass  # Quiet for demo

    obs1 = DemoObserver("GUI Observer")
    obs2 = DemoObserver("Logging Observer")
    service.add_observer(obs1)
    service.add_observer(obs2)
    print(f"   [OK] Attached 2 observers to ValidationService")
    print(f"       - {obs1.name}")
    print(f"       - {obs2.name}")

    # 5. Singleton Pattern (Config)
    print("\n5. Singleton Pattern (ValidatorConfig):")
    from src.validators.base_validator import ValidatorConfig

    print(f"   [OK] Centralized configuration:")
    print(f"       - BRIGHTNESS_THRESHOLD: {ValidatorConfig.BRIGHTNESS_THRESHOLD}")
    print(f"       - SHARPNESS_THRESHOLD: {ValidatorConfig.SHARPNESS_THRESHOLD}")


def demo_solid_principles():
    """Demo: SOLID principles"""
    print("\n" + "=" * 70)
    print("DEMO 6: SOLID Principles")
    print("=" * 70)

    # S - Single Responsibility
    print("\n[S] Single Responsibility Principle:")
    print("    - BrightnessValidator: only validates brightness")
    print("    - CameraService: only handles camera")
    print("    - StorageService: only handles storage")

    # O - Open/Closed
    print("\n[O] Open/Closed Principle:")
    print("    - New validators can be added without modifying existing code")
    print("    - IValidator interface enables extension")

    # L - Liskov Substitution
    print("\n[L] Liskov Substitution Principle:")
    print("    - All validators can substitute IValidator")
    print("    - SQLitePhotoRepository can substitute IPhotoRepository")

    # I - Interface Segregation
    print("\n[I] Interface Segregation Principle:")
    print("    - IValidator: small, focused interface")
    print("    - IPhotoRepository: only essential CRUD operations")

    # D - Dependency Inversion
    print("\n[D] Dependency Inversion Principle:")
    print("    - ValidationService depends on IValidator abstraction")
    print("    - StorageService depends on IPhotoRepository abstraction")


def main():
    """Run all demos"""
    print("\n")
    print("=" * 70)
    print(" " * 15 + "PASFOTO VALIDATIE APPLICATIE DEMO")
    print(" " * 20 + "Complete System Demonstration")
    print("=" * 70)

    try:
        demo_basic_validation()
        demo_storage()
        demo_repository()
        demo_validation_service()
        demo_design_patterns()
        demo_solid_principles()

        print("\n" + "=" * 70)
        print("SUCCESS: ALL DEMOS COMPLETED")
        print("=" * 70)

        print("\nNext steps:")
        print("1. Run tests: .\\venv\\Scripts\\pytest.exe")
        print("2. Download BioID: .\\venv\\Scripts\\python.exe scripts\\download_bioid_dataset.py")
        print("3. Test with BioID: .\\venv\\Scripts\\python.exe scripts\\test_with_bioid.py")
        print("4. Start GUI: .\\venv\\Scripts\\python.exe main.py")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
