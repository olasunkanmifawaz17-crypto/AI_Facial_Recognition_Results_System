
import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMBEDDINGS_FILE = os.path.join(BASE_DIR, "face_embeddings.npy")
YUNET_MODEL = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
SFACE_MODEL = os.path.join(BASE_DIR, "models", "face_recognition_sface_2021dec.onnx")


class FaceRecognitionService:

    def __init__(self):

        self.detector = cv2.FaceDetectorYN_create(
            YUNET_MODEL, "", (320, 320), 0.8, 0.3, 5000
        )

        self.recognizer = cv2.FaceRecognizerSF_create(
            SFACE_MODEL, ""
        )

        self.student_embeddings = {}
        self.normalized_embeddings = {}

        self.reload_embeddings()

    def reload_embeddings(self):

        if not os.path.isfile(EMBEDDINGS_FILE):
            raise FileNotFoundError(
                "Facial embeddings file not found."
            )

        loaded_embeddings = np.load(
            EMBEDDINGS_FILE,
            allow_pickle=True
        ).item()

        normalized_embeddings = {}

        for matric_number, features in loaded_embeddings.items():

            features = np.asarray(
                features,
                dtype=np.float32
            )

            if features.ndim != 2 or len(features) == 0:
                normalized_embeddings[matric_number] = np.empty(
                    (0, 128), dtype=np.float32
                )
                continue

            norms = np.linalg.norm(
                features,
                axis=1,
                keepdims=True
            )

            valid = (
                np.isfinite(features).all(axis=1)
                & (norms.flatten() > 0)
            )

            features = features[valid]
            norms = norms[valid]

            if len(features) == 0:
                normalized_embeddings[matric_number] = np.empty(
                    (0, features.shape[1]), dtype=np.float32
                )
                continue

            normalized_embeddings[matric_number] = (
                features / norms
            )

        # Update both dictionaries only after successful loading.
        self.student_embeddings = loaded_embeddings
        self.normalized_embeddings = normalized_embeddings

        return len(loaded_embeddings)

    def check_face_quality(self, frame, face):

        x, y, w, h = face[:4].astype(int)

        height, width = frame.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return False, "Invalid face area"

        if min(face_crop.shape[:2]) < 120:
            return False, "Move closer to camera"

        gray = cv2.cvtColor(
            face_crop,
            cv2.COLOR_BGR2GRAY
        )

        blur_score = cv2.Laplacian(
            gray, cv2.CV_64F
        ).var()

        if blur_score < 50:
            return False, "Image too blurry"

        brightness = np.mean(gray)

        if not 45 <= brightness <= 210:
            return False, "Adjust lighting"

        return True, "Face quality acceptable"

    def compare_frame(self, matric_number, frame):

        if matric_number not in self.normalized_embeddings:
            return {
                "valid": False,
                "message": "Student embeddings unavailable."
            }

        stored_features = self.normalized_embeddings[matric_number]

        if len(stored_features) == 0:
            return {
                "valid": False,
                "message": "No valid embeddings available."
            }

        height, width = frame.shape[:2]

        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(frame)

        if faces is None or len(faces) == 0:
            return {
                "valid": False,
                "message": "No face detected."
            }

        if len(faces) > 1:
            return {
                "valid": False,
                "message": "Multiple faces detected."
            }

        face = faces[0]

        quality_ok, quality_message = self.check_face_quality(
            frame, face
        )

        if not quality_ok:
            return {
                "valid": False,
                "message": quality_message
            }

        aligned_face = self.recognizer.alignCrop(
            frame, face
        )

        live_feature = self.recognizer.feature(
            aligned_face
        ).flatten()

        live_norm = np.linalg.norm(live_feature)

        if live_norm == 0 or not np.all(np.isfinite(live_feature)):
            return {
                "valid": False,
                "message": "Invalid facial features."
            }

        live_feature = live_feature / live_norm

        similarities = np.dot(
            stored_features, live_feature
        )

        top_k = min(5, len(stored_features))

        top_scores = np.sort(similarities)[-top_k:]

        score = float(np.mean(top_scores))

        return {
            "valid": True,
            "score": score,
            "message": "Frame processed successfully."
        }
