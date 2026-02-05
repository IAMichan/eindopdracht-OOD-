"""
Face detection model using MediaPipe
"""
# imports voor de gezichts herkenning
import cv2
import mediapipe as mp # Google MediaPipe bibliotheek
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from .photo import FaceDetectionResult


class FaceDetectionModel:
    """
    Face detection en landmark extraction met MediaPipe

    Gebruikt MediaPipe Face Mesh voor accurate gezichtsdetectie
    en landmark extraction.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialiseer face detection model

        Args:
            min_detection_confidence: Minimum confidence voor face detectie
            min_tracking_confidence: Minimum confidence voor tracking
        """
        # sla de confidence waardes op voor later gebruik
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence

        # hier maken we het MediaPipe gezichts mesh model aan
        # dit ding kan 468 punten op je gezicht vinden, best veel eigenlijk
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=True,  # we doen losse foto's, geen video
            max_num_faces=1,  # we willen maar 1 gezicht per foto
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # dit zijn de indices van belangrijke punten op het gezicht
        # MediaPipe heeft 468 punten, maar we hebben niet allemaal nodig want sommige zijn best dicht bij elkaar
        self._LANDMARK_INDICES = {
            # Ogen
            'left_eye_top': 159,
            'left_eye_bottom': 145,
            'left_eye_left': 33,
            'left_eye_right': 133,
            'right_eye_top': 386,
            'right_eye_bottom': 374,
            'right_eye_left': 362,
            'right_eye_right': 263,
            # Mond
            'mouth_top': 13,
            'mouth_bottom': 14,
            'mouth_left': 61,
            'mouth_right': 291,
            'upper_lip_top': 13,
            'lower_lip_bottom': 14,
            # Wenkbrauwen
            'left_eyebrow_inner': 107,
            'left_eyebrow_outer': 66,
            'right_eyebrow_inner': 336,
            'right_eyebrow_outer': 296,
            # Gezicht outline
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                         397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                         172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        }

    def detect_face(self, image: np.ndarray) -> FaceDetectionResult:
        """
        Detecteer gezicht en extract landmarks

        Args:
            image: Input image als NumPy array (BGR format)

        Returns:
            FaceDetectionResult met detectie informatie
        """
        # Converteer BGR naar RGB (MediaPipe verwacht RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image
        results = self._face_mesh.process(image_rgb)

        # Check of gezicht gedetecteerd is
        if not results.multi_face_landmarks:
            return FaceDetectionResult(
                face_found=False,
                confidence=0.0,
                face_bbox=None,
                landmarks=None
            )

        # Haal eerste (en enige) gezicht op
        face_landmarks = results.multi_face_landmarks[0]

        # Bereken bounding box
        bbox = self._calculate_bounding_box(face_landmarks, image.shape)

        # Extract belangrijke landmarks
        landmarks = self._extract_landmarks(face_landmarks, image.shape)

        # Bereken confidence (MediaPipe geeft geen directe confidence, gebruik landmark count)
        confidence = min(1.0, len(face_landmarks.landmark) / 468)  # 468 is totaal aantal landmarks

        return FaceDetectionResult(
            face_found=True,
            confidence=confidence,
            face_bbox=bbox,
            landmarks=landmarks
        )

    def _calculate_bounding_box(
        self,
        face_landmarks,
        image_shape: Tuple[int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Bereken bounding box van gezicht

        Args:
            face_landmarks: MediaPipe face landmarks
            image_shape: Shape van image (height, width, channels)

        Returns:
            (x, y, width, height) tuple
        """
        h, w, _ = image_shape

        # Haal alle landmark coordinaten op
        x_coords = [lm.x * w for lm in face_landmarks.landmark]
        y_coords = [lm.y * h for lm in face_landmarks.landmark]

        # Bereken min/max
        x_min = int(min(x_coords))
        x_max = int(max(x_coords))
        y_min = int(min(y_coords))
        y_max = int(max(y_coords))

        # Voeg wat padding toe
        padding_x = int((x_max - x_min) * 0.1)
        padding_y = int((y_max - y_min) * 0.1)

        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(w, x_max + padding_x)
        y_max = min(h, y_max + padding_y)

        width = x_max - x_min
        height = y_max - y_min

        return (x_min, y_min, width, height)

    def _extract_landmarks(
        self,
        face_landmarks,
        image_shape: Tuple[int, int, int]
    ) -> Dict:
        """
        Extract belangrijke landmarks en bereken features

        Args:
            face_landmarks: MediaPipe face landmarks
            image_shape: Shape van image

        Returns:
            Dictionary met landmark features
        """
        h, w, _ = image_shape
        landmarks_dict = {}

        # Helper functie om landmark coordinaten op te halen
        def get_point(idx):
            lm = face_landmarks.landmark[idx]
            return (int(lm.x * w), int(lm.y * h))

        # Extract oog landmarks
        left_eye_top = get_point(self._LANDMARK_INDICES['left_eye_top'])
        left_eye_bottom = get_point(self._LANDMARK_INDICES['left_eye_bottom'])
        left_eye_left = get_point(self._LANDMARK_INDICES['left_eye_left'])
        left_eye_right = get_point(self._LANDMARK_INDICES['left_eye_right'])

        right_eye_top = get_point(self._LANDMARK_INDICES['right_eye_top'])
        right_eye_bottom = get_point(self._LANDMARK_INDICES['right_eye_bottom'])
        right_eye_left = get_point(self._LANDMARK_INDICES['right_eye_left'])
        right_eye_right = get_point(self._LANDMARK_INDICES['right_eye_right'])

        # Bereken eye aspect ratios
        left_eye_height = abs(left_eye_top[1] - left_eye_bottom[1])
        left_eye_width = abs(left_eye_right[0] - left_eye_left[0])
        right_eye_height = abs(right_eye_top[1] - right_eye_bottom[1])
        right_eye_width = abs(right_eye_right[0] - right_eye_left[0])

        landmarks_dict['left_eye_height'] = left_eye_height / left_eye_width if left_eye_width > 0 else 0
        landmarks_dict['left_eye_width'] = left_eye_width
        landmarks_dict['right_eye_height'] = right_eye_height / right_eye_width if right_eye_width > 0 else 0
        landmarks_dict['right_eye_width'] = right_eye_width

        # Eye regions voor coverage check
        left_eye_x = min(left_eye_left[0], left_eye_right[0])
        left_eye_y = min(left_eye_top[1], left_eye_bottom[1])
        left_eye_w = abs(left_eye_right[0] - left_eye_left[0])
        left_eye_h = abs(left_eye_bottom[1] - left_eye_top[1])

        right_eye_x = min(right_eye_left[0], right_eye_right[0])
        right_eye_y = min(right_eye_top[1], right_eye_bottom[1])
        right_eye_w = abs(right_eye_right[0] - right_eye_left[0])
        right_eye_h = abs(right_eye_bottom[1] - right_eye_top[1])

        landmarks_dict['left_eye_region'] = (left_eye_x, left_eye_y, left_eye_w, left_eye_h)
        landmarks_dict['right_eye_region'] = (right_eye_x, right_eye_y, right_eye_w, right_eye_h)

        # Extract mond landmarks
        mouth_top = get_point(self._LANDMARK_INDICES['mouth_top'])
        mouth_bottom = get_point(self._LANDMARK_INDICES['mouth_bottom'])
        mouth_left = get_point(self._LANDMARK_INDICES['mouth_left'])
        mouth_right = get_point(self._LANDMARK_INDICES['mouth_right'])

        # Bereken mouth aspect ratio
        mouth_height = abs(mouth_top[1] - mouth_bottom[1])
        mouth_width = abs(mouth_right[0] - mouth_left[0])

        landmarks_dict['mouth_upper'] = mouth_top[1]
        landmarks_dict['mouth_lower'] = mouth_bottom[1]
        landmarks_dict['mouth_width'] = mouth_width

        # Extract wenkbrauw landmarks
        left_eyebrow_inner = get_point(self._LANDMARK_INDICES['left_eyebrow_inner'])
        left_eyebrow_outer = get_point(self._LANDMARK_INDICES['left_eyebrow_outer'])

        # Bereken eyebrow raise (afstand tussen wenkbrauw en oog)
        eyebrow_eye_distance = abs(left_eyebrow_inner[1] - left_eye_top[1])
        # Normaliseer op basis van gezichts hoogte
        face_height = h
        landmarks_dict['eyebrow_raise'] = eyebrow_eye_distance / face_height

        # Bereken mouth symmetry
        mouth_center_x = (mouth_left[0] + mouth_right[0]) / 2
        face_center_x = w / 2
        mouth_offset = abs(mouth_center_x - face_center_x) / w
        landmarks_dict['mouth_symmetry'] = 1.0 - min(1.0, mouth_offset * 10)

        return landmarks_dict

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_face_mesh'):
            self._face_mesh.close()


class FaceDetectionModelFactory:
    """
    Factory voor het creëren van face detection models

    Implementeert Factory Pattern
    """

    @staticmethod
    def create_mediapipe_model(
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5
    ) -> FaceDetectionModel:
        """
        Creëer MediaPipe face detection model

        Args:
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence

        Returns:
            FaceDetectionModel instance
        """
        return FaceDetectionModel(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    @staticmethod
    def create_default_model() -> FaceDetectionModel:
        """
        Creëer default face detection model

        Returns:
            FaceDetectionModel instance met default settings
        """
        return FaceDetectionModel()
