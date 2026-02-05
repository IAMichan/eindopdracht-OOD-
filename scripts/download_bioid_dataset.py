"""
Script om BioID Face Database te downloaden en voorbereiden

BioID Face Database: https://www.bioid.com/face-database/
- 1521 grayscale images
- 23 verschillende personen
- Verschillende poses, belichting, achtergronden
"""
import os
import urllib.request
import zipfile
from pathlib import Path
import shutil


class BioIDDatasetDownloader:
    """Download en prepareer BioID Face Database"""

    def __init__(self, data_dir: str = "data/bioid"):
        """
        Initialiseer downloader

        Args:
            data_dir: Directory waar dataset opgeslagen wordt
        """
        self.data_dir = Path(data_dir)
        self.dataset_url = "https://ftp.uni-erlangen.de/pub/facedb/BioID-FaceDatabase-V1.2.zip"
        self.zip_path = self.data_dir / "BioID-FaceDatabase-V1.2.zip"
        self.extract_dir = self.data_dir / "extracted"

    def download(self):
        """Download de dataset"""
        print("BioID Face Database Downloader")
        print("=" * 50)

        # Maak directory aan
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Check of al gedownload
        if self.zip_path.exists():
            print(f"Dataset zip already exists at: {self.zip_path}")
            response = input("Re-download? (y/n): ")
            if response.lower() != 'y':
                print("Skipping download...")
                return

        print(f"\nDownloading from: {self.dataset_url}")
        print(f"Saving to: {self.zip_path}")
        print("This may take a few minutes (approx 80 MB)...\n")

        try:
            # Download met progress
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100)
                print(f"\rProgress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB)", end='')

            urllib.request.urlretrieve(
                self.dataset_url,
                self.zip_path,
                reporthook=report_progress
            )
            print("\n\n[OK] Download complete!")

        except Exception as e:
            print(f"\n[FAIL] Download failed: {e}")
            print("\nAlternative: Download manually from:")
            print("https://www.bioid.com/face-database/")
            print(f"And place the zip file at: {self.zip_path}")
            return False

        return True

    def extract(self):
        """Extract de dataset"""
        if not self.zip_path.exists():
            print("[FAIL] Zip file not found. Run download() first.")
            return False

        print("\nExtracting dataset...")
        self.extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_dir)
            print(f"[OK] Extracted to: {self.extract_dir}")
            return True
        except Exception as e:
            print(f"[FAIL] Extraction failed: {e}")
            return False

    def organize(self):
        """Organiseer dataset in structured folders"""
        print("\nOrganizing dataset...")

        # Maak directories
        images_dir = self.data_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Find BioID images (meestal in BioID-FaceDatabase-V1.2/BioID_***.pgm)
        source_dir = self.extract_dir / "BioID-FaceDatabase-V1.2"

        if not source_dir.exists():
            # Probeer alternative locaties
            for item in self.extract_dir.iterdir():
                if item.is_dir() and "BioID" in item.name:
                    source_dir = item
                    break

        if not source_dir.exists():
            print(f"[FAIL] Could not find BioID images in {self.extract_dir}")
            print("Contents:", list(self.extract_dir.iterdir()))
            return False

        # Copy images
        image_files = list(source_dir.glob("*.pgm"))

        if not image_files:
            print(f"[FAIL] No .pgm images found in {source_dir}")
            return False

        print(f"Found {len(image_files)} images")

        for img_file in image_files:
            dest = images_dir / img_file.name
            if not dest.exists():
                shutil.copy2(img_file, dest)

        print(f"[OK] Copied {len(image_files)} images to {images_dir}")

        # Copy landmarks if available
        landmark_files = list(source_dir.glob("*.pts"))
        if landmark_files:
            landmarks_dir = self.data_dir / "landmarks"
            landmarks_dir.mkdir(exist_ok=True)

            for lm_file in landmark_files:
                dest = landmarks_dir / lm_file.name
                if not dest.exists():
                    shutil.copy2(lm_file, dest)

            print(f"[OK] Copied {len(landmark_files)} landmark files to {landmarks_dir}")

        return True

    def get_statistics(self):
        """Toon dataset statistieken"""
        images_dir = self.data_dir / "images"
        landmarks_dir = self.data_dir / "landmarks"

        if not images_dir.exists():
            print("Dataset not yet organized. Run organize() first.")
            return

        num_images = len(list(images_dir.glob("*.pgm")))
        num_landmarks = len(list(landmarks_dir.glob("*.pts"))) if landmarks_dir.exists() else 0

        print("\n" + "=" * 50)
        print("BioID Dataset Statistics")
        print("=" * 50)
        print(f"Total images: {num_images}")
        print(f"Landmark files: {num_landmarks}")
        print(f"Location: {self.data_dir}")
        print(f"Images dir: {images_dir}")
        if landmarks_dir.exists():
            print(f"Landmarks dir: {landmarks_dir}")
        print("=" * 50)

    def cleanup_zip(self):
        """Verwijder zip en extracted folders (save space)"""
        response = input("\nDelete zip file and extracted folder to save space? (y/n): ")
        if response.lower() == 'y':
            if self.zip_path.exists():
                self.zip_path.unlink()
                print(f"[OK] Deleted {self.zip_path}")

            if self.extract_dir.exists():
                shutil.rmtree(self.extract_dir)
                print(f"[OK] Deleted {self.extract_dir}")

            print("[OK] Cleanup complete. Dataset preserved in organized structure.")


def main():
    """Main functie"""
    print("BioID Face Database Setup")
    print("=" * 50)
    print()

    downloader = BioIDDatasetDownloader()

    # Download
    if downloader.download():
        # Extract
        if downloader.extract():
            # Organize
            if downloader.organize():
                # Show stats
                downloader.get_statistics()

                # Optional cleanup
                downloader.cleanup_zip()

                print("\n[OK] Dataset ready for use!")
                print("\nNext steps:")
                print("1. Use images in data/bioid/images/ for testing")
                print("2. Run validation tests with: pytest tests/integration/test_bioid.py")
                print("3. See docs/TESTPLAN.md for AI accuracy testing strategy")
            else:
                print("\n[FAIL] Failed to organize dataset")
        else:
            print("\n[FAIL] Failed to extract dataset")
    else:
        print("\n[FAIL] Failed to download dataset")
        print("\nManual download instructions:")
        print("1. Visit: https://www.bioid.com/face-database/")
        print("2. Download BioID-FaceDatabase-V1.2.zip")
        print("3. Place in: data/bioid/")
        print("4. Run this script again")


if __name__ == "__main__":
    main()
