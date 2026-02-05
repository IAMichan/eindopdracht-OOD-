"""
Storage service voor foto opslag
"""
# imports voor bestands operaties en foto opslag
import os
import cv2
import logging
from typing import Optional
from datetime import datetime
import numpy as np

from ..models.photo import Photo
from ..repositories.photo_repository import IPhotoRepository, SQLitePhotoRepository


logger = logging.getLogger(__name__)


class StorageService:
    """
    Service voor foto opslag (bestanden + metadata)

    Combineert file system storage met repository voor metadata
    """

    def __init__(
        self,
        storage_path: str = "data/photos",
        repository: Optional[IPhotoRepository] = None
    ):
        """
        Initialiseer storage service

        Args:
            storage_path: Pad naar directory voor foto opslag
            repository: Photo repository (optioneel, anders SQLite default)
        """
        self._storage_path = storage_path
        # als er geen repo is meegegeven, maak er dan een sqlite database
        self._repository = repository or SQLitePhotoRepository()

        # maak de hoofd directory aan waar we alles opslaan
        os.makedirs(storage_path, exist_ok=True)

        # maak subdirectories voor elke status
        # approved = goedgekeurd, rejected = afgekeurd, pending = wachten op beslissing
        os.makedirs(os.path.join(storage_path, "approved"), exist_ok=True)
        os.makedirs(os.path.join(storage_path, "rejected"), exist_ok=True)
        os.makedirs(os.path.join(storage_path, "pending"), exist_ok=True)

        logger.info(f"StorageService initialized with path: {storage_path}")

    def save_photo(self, photo: Photo) -> bool:
        """
        Sla photo op (bestand + metadata)

        Args:
            photo: Photo object om op te slaan

        Returns:
            True als succesvol, False anders
        """
        try:
            # maak een unieke ID voor de foto als die er nog geen heeft
            if photo.id is None:
                import uuid
                photo.id = str(uuid.uuid4())

            filename = f"{photo.id}.jpg"

            # kijk wat de status is (approved/rejected/pending)
            # en stop hem in de juiste folder
            status_dir = photo.status.value
            file_path = os.path.join(self._storage_path, status_dir, filename)

            # schrijf de foto naar disk
            success = cv2.imwrite(file_path, photo.image_data)

            if not success:
                logger.error(f"Failed to write image to {file_path}")
                return False

            # update het photo object met het pad
            photo.file_path = file_path

            # sla ook de metadata op in de database
            success = self._repository.save(photo)

            if success:
                logger.info(f"Photo {photo.id} saved successfully to {file_path}")
            else:
                logger.error(f"Failed to save photo metadata for {photo.id}")
                # als metadata niet gelukt is, gooi dan de foto ook weg
                os.remove(file_path)

            return success

        except Exception as e:
            logger.error(f"Error saving photo: {e}", exc_info=True)
            return False

    def load_photo(self, photo_id: str) -> Optional[Photo]:
        """
        Laad photo (metadata + bestand)

        Args:
            photo_id: ID van photo

        Returns:
            Photo object of None als niet gevonden
        """
        try:
            # haal eerst de metadata op uit de database
            photo = self._repository.get_by_id(photo_id)

            if photo is None:
                logger.warning(f"Photo {photo_id} not found in repository")
                return None

            # laad dan de echte foto van disk
            if photo.file_path and os.path.exists(photo.file_path):
                image_data = cv2.imread(photo.file_path)

                if image_data is None:
                    logger.error(f"Failed to read image from {photo.file_path}")
                    return None

                photo.image_data = image_data
            else:
                logger.warning(f"Image file not found: {photo.file_path}")
                return None

            return photo

        except Exception as e:
            logger.error(f"Error loading photo {photo_id}: {e}", exc_info=True)
            return None

    def delete_photo(self, photo_id: str) -> bool:
        """
        Verwijder photo (bestand + metadata)

        Args:
            photo_id: ID van photo

        Returns:
            True als succesvol
        """
        try:
            # zoek de foto op
            photo = self._repository.get_by_id(photo_id)

            success = True

            # verwijder het bestand van disk
            if photo and photo.file_path and os.path.exists(photo.file_path):
                os.remove(photo.file_path)
                logger.info(f"Deleted file: {photo.file_path}")
            else:
                logger.warning(f"File not found for photo {photo_id}")
                success = False

            # verwijder ook de metadata uit database
            if not self._repository.delete(photo_id):
                logger.error(f"Failed to delete metadata for {photo_id}")
                success = False

            return success

        except Exception as e:
            logger.error(f"Error deleting photo {photo_id}: {e}", exc_info=True)
            return False

    def export_approved_photos(self, export_path: str) -> int:
        """
        Exporteer alle approved photos naar een directory

        Args:
            export_path: Pad waar photos geëxporteerd worden

        Returns:
            Aantal geëxporteerde photos
        """
        try:
            # maak de export directory aan
            os.makedirs(export_path, exist_ok=True)

            from ..models.photo import PhotoStatus
            # haal alle goedgekeurde foto's op
            approved_photos = self._repository.get_by_status(PhotoStatus.APPROVED)

            count = 0
            # kopieer ze allemaal naar de export directory
            for photo in approved_photos:
                if photo.file_path and os.path.exists(photo.file_path):
                    import shutil
                    dest_path = os.path.join(export_path, os.path.basename(photo.file_path))
                    shutil.copy2(photo.file_path, dest_path)
                    count += 1

            logger.info(f"Exported {count} approved photos to {export_path}")
            return count

        except Exception as e:
            logger.error(f"Error exporting photos: {e}", exc_info=True)
            return 0

    def cleanup_old_photos(self, days: int = 30) -> int:
        """
        Verwijder photos ouder dan X dagen (behalve approved)

        Args:
            days: Aantal dagen

        Returns:
            Aantal verwijderde photos
        """
        try:
            from datetime import timedelta
            from ..models.photo import PhotoStatus

            # bereken de cutoff datum
            cutoff_date = datetime.now() - timedelta(days=days)

            # haal alle foto's op
            all_photos = self._repository.get_all()

            count = 0
            for photo in all_photos:
                # goedgekeurde foto's houden we altijd
                if photo.status == PhotoStatus.APPROVED:
                    continue

                # als de foto ouder is dan de cutoff datum, gooi hem weg
                if photo.timestamp < cutoff_date:
                    if self.delete_photo(photo.id):
                        count += 1

            logger.info(f"Cleaned up {count} old photos")
            return count

        except Exception as e:
            logger.error(f"Error cleaning up old photos: {e}", exc_info=True)
            return 0

    def get_storage_statistics(self) -> dict:
        """
        Haal storage statistieken op

        Returns:
            Dictionary met statistieken
        """
        try:
            # haal stats op uit de database
            repo_stats = self._repository.get_statistics()

            # bereken hoeveel ruimte alle foto's innemen
            def get_dir_size(path):
                total = 0
                if os.path.exists(path):
                    for entry in os.scandir(path):
                        if entry.is_file():
                            total += entry.stat().st_size
                        elif entry.is_dir():
                            total += get_dir_size(entry.path)  # recursief voor subdirs
                return total

            total_size = get_dir_size(self._storage_path)

            # combineer alles in een mooie dictionary
            return {
                **repo_stats,
                "storage_path": self._storage_path,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }

        except Exception as e:
            logger.error(f"Error getting storage statistics: {e}", exc_info=True)
            return {}
