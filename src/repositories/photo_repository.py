"""
Repository interface en implementatie voor photo storage
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
import sqlite3
import json
import logging
import os
import uuid
import numpy as np

from ..models.photo import Photo, PhotoStatus, ValidationResult


logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """
    Converteer numpy types naar standaard Python types voor JSON serialisatie

    Args:
        obj: Object om te converteren

    Returns:
        Geconverteerd object
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


class IPhotoRepository(ABC):
    """
    Interface voor photo repository

    Implementeert Repository Pattern voor data toegang abstractie
    """

    @abstractmethod
    def save(self, photo: Photo) -> bool:
        """
        Sla een photo op

        Args:
            photo: Photo object om op te slaan

        Returns:
            True als succesvol, False anders
        """
        pass

    @abstractmethod
    def get_by_id(self, photo_id: str) -> Optional[Photo]:
        """
        Haal een photo op via ID

        Args:
            photo_id: Unique photo identifier

        Returns:
            Photo object of None als niet gevonden
        """
        pass

    @abstractmethod
    def get_all(self, limit: Optional[int] = None) -> List[Photo]:
        """
        Haal alle photos op

        Args:
            limit: Optionele limiet op aantal resultaten

        Returns:
            Lijst met Photo objecten
        """
        pass

    @abstractmethod
    def get_by_status(self, status: PhotoStatus) -> List[Photo]:
        """
        Haal photos op via status

        Args:
            status: PhotoStatus filter

        Returns:
            Lijst met Photo objecten
        """
        pass

    @abstractmethod
    def delete(self, photo_id: str) -> bool:
        """
        Verwijder een photo

        Args:
            photo_id: ID van photo om te verwijderen

        Returns:
            True als succesvol, False anders
        """
        pass


class SQLitePhotoRepository(IPhotoRepository):
    """
    SQLite implementatie van photo repository
    """

    def __init__(self, db_path: str = "data/photos.db"):
        """
        Initialiseer SQLite repository

        Args:
            db_path: Pad naar SQLite database bestand
        """
        self._db_path = db_path
        self._ensure_database_exists()

    def _ensure_database_exists(self) -> None:
        """Maak database en tables aan als ze niet bestaan"""
        # Zorg dat directory bestaat
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)

        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # Create photos table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS photos (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                file_path TEXT,
                metadata TEXT,
                overall_confidence REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create validation_results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS validation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id TEXT NOT NULL,
                validator_name TEXT NOT NULL,
                is_valid INTEGER NOT NULL,
                confidence REAL NOT NULL,
                message TEXT NOT NULL,
                details TEXT,
                FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
            )
        ''')

        # Create indices
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_photos_status
            ON photos(status)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_photos_timestamp
            ON photos(timestamp DESC)
        ''')

        conn.commit()
        conn.close()

        logger.info(f"Database initialized at {self._db_path}")

    def save(self, photo: Photo) -> bool:
        """Sla photo op in database"""
        try:
            # Generate ID if not set
            if photo.id is None:
                photo.id = str(uuid.uuid4())

            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()

            # Insert photo
            cursor.execute('''
                INSERT OR REPLACE INTO photos
                (id, timestamp, status, file_path, metadata, overall_confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                photo.id,
                photo.timestamp.isoformat(),
                photo.status.value,
                photo.file_path,
                json.dumps(convert_numpy_types(photo.metadata)),
                photo.get_overall_confidence()
            ))

            # Delete existing validation results (in case of update)
            cursor.execute(
                'DELETE FROM validation_results WHERE photo_id = ?',
                (photo.id,)
            )

            # Insert validation results
            for result in photo.validation_results:
                cursor.execute('''
                    INSERT INTO validation_results
                    (photo_id, validator_name, is_valid, confidence, message, details)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    photo.id,
                    result.validator_name,
                    1 if result.is_valid else 0,
                    result.confidence,
                    result.message,
                    json.dumps(convert_numpy_types(result.details)) if result.details else None
                ))

            conn.commit()
            conn.close()

            logger.info(f"Photo {photo.id} saved successfully")
            return True

        except Exception as e:
            logger.error(f"Error saving photo: {e}", exc_info=True)
            return False

    def get_by_id(self, photo_id: str) -> Optional[Photo]:
        """Haal photo op via ID"""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get photo metadata
            cursor.execute(
                'SELECT * FROM photos WHERE id = ?',
                (photo_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            # Get validation results
            cursor.execute(
                'SELECT * FROM validation_results WHERE photo_id = ?',
                (photo_id,)
            )
            validation_rows = cursor.fetchall()

            conn.close()

            # Reconstruct photo (zonder image_data - moet apart geladen worden)
            photo = Photo(
                image_data=None,  # Image data moet geladen worden via file_path
                timestamp=datetime.fromisoformat(row['timestamp']),
                id=row['id'],
                status=PhotoStatus(row['status']),
                file_path=row['file_path'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )

            # Add validation results
            for vr in validation_rows:
                # Converteer confidence naar float (kan bytes zijn door SQLite type affinity)
                confidence_value = vr['confidence']
                if isinstance(confidence_value, bytes):
                    # Parse bytes als little-endian float
                    import struct
                    confidence_value = struct.unpack('<f', confidence_value)[0]
                else:
                    confidence_value = float(confidence_value)

                result = ValidationResult(
                    validator_name=vr['validator_name'],
                    is_valid=bool(vr['is_valid']),
                    confidence=confidence_value,
                    message=vr['message'],
                    details=json.loads(vr['details']) if vr['details'] else None
                )
                photo.add_validation_result(result)

            return photo

        except Exception as e:
            logger.error(f"Error getting photo {photo_id}: {e}", exc_info=True)
            return None

    def get_all(self, limit: Optional[int] = None) -> List[Photo]:
        """Haal alle photos op"""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = 'SELECT id FROM photos ORDER BY timestamp DESC'
            if limit:
                query += f' LIMIT {limit}'

            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()

            # Load each photo
            photos = []
            for row in rows:
                photo = self.get_by_id(row['id'])
                if photo:
                    photos.append(photo)

            return photos

        except Exception as e:
            logger.error(f"Error getting all photos: {e}", exc_info=True)
            return []

    def get_by_status(self, status: PhotoStatus) -> List[Photo]:
        """Haal photos op via status"""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                'SELECT id FROM photos WHERE status = ? ORDER BY timestamp DESC',
                (status.value,)
            )
            rows = cursor.fetchall()
            conn.close()

            # Load each photo
            photos = []
            for row in rows:
                photo = self.get_by_id(row['id'])
                if photo:
                    photos.append(photo)

            return photos

        except Exception as e:
            logger.error(f"Error getting photos by status: {e}", exc_info=True)
            return []

    def delete(self, photo_id: str) -> bool:
        """Verwijder photo"""
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()

            cursor.execute('DELETE FROM photos WHERE id = ?', (photo_id,))

            conn.commit()
            deleted = cursor.rowcount > 0
            conn.close()

            if deleted:
                logger.info(f"Photo {photo_id} deleted")
            else:
                logger.warning(f"Photo {photo_id} not found for deletion")

            return deleted

        except Exception as e:
            logger.error(f"Error deleting photo {photo_id}: {e}", exc_info=True)
            return False

    def get_statistics(self) -> dict:
        """
        Haal statistieken op

        Returns:
            Dictionary met statistieken
        """
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()

            # Total count
            cursor.execute('SELECT COUNT(*) FROM photos')
            total = cursor.fetchone()[0]

            # Count by status
            cursor.execute('''
                SELECT status, COUNT(*) as count
                FROM photos
                GROUP BY status
            ''')
            status_counts = {row[0]: row[1] for row in cursor.fetchall()}

            # Average confidence
            cursor.execute('SELECT AVG(overall_confidence) FROM photos')
            avg_confidence = cursor.fetchone()[0] or 0.0

            conn.close()

            return {
                "total_photos": total,
                "by_status": status_counts,
                "average_confidence": avg_confidence
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}", exc_info=True)
            return {}
