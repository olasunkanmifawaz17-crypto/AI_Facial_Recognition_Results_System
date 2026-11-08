
import os
import cv2
import hashlib
import shutil
import tempfile
import numpy as np

from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "dataset")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "face_embeddings.npy")

DETECTOR_MODEL = os.path.join(
    BASE_DIR, "models", "face_detection_yunet_2023mar.onnx"
)

RECOGNIZER_MODEL = os.path.join(
    BASE_DIR, "models", "face_recognition_sface_2021dec.onnx"
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

MIN_FACE_SIZE = 120
MIN_BLUR_SCORE = 50
MIN_BRIGHTNESS = 45
MAX_BRIGHTNESS = 210


def save_embeddings_atomically(embeddings):

    fd, temporary_file = tempfile.mkstemp(
        dir=BASE_DIR,
        suffix=".tmp"
    )

    try:
        with os.fdopen(fd, "wb") as file:
            np.save(file, embeddings, allow_pickle=True)

        os.replace(temporary_file, EMBEDDINGS_FILE)

    except Exception:
        if os.path.exists(temporary_file):
            os.remove(temporary_file)
        raise


def backup_embeddings():

    if not os.path.isfile(EMBEDDINGS_FILE):
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    backup_file = os.path.join(
        BASE_DIR,
        f"face_embeddings_{timestamp}.backup.npy"
    )

    shutil.copy2(EMBEDDINGS_FILE, backup_file)

    return backup_file


def check_image_quality(image, face):

    x, y, w, h = face[:4].astype(int)

    height, width = image.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    face_crop = image[y1:y2, x1:x2]

    if face_crop.size == 0:
        return False

    if min(face_crop.shape[:2]) < MIN_FACE_SIZE:
        return False

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(
        gray, cv2.CV_64F
    ).var()

    if blur_score < MIN_BLUR_SCORE:
        return False

    brightness = np.mean(gray)

    if not MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS:
        return False

    return True


def extract_student_embeddings(matric_number):

    student_folder = os.path.join(DATASET_PATH, matric_number)

    if not os.path.isdir(student_folder):
        raise ValueError("Student facial dataset was not found.")

    if not os.path.isfile(DETECTOR_MODEL):
        raise FileNotFoundError("YuNet model not found.")

    if not os.path.isfile(RECOGNIZER_MODEL):
        raise FileNotFoundError("SFace model not found.")

    image_files = [
        filename
        for filename in sorted(os.listdir(student_folder))
        if filename.lower().endswith(IMAGE_EXTENSIONS)
    ]

    if not image_files:
        raise ValueError("No facial images were found.")

    detector = cv2.FaceDetectorYN_create(
        DETECTOR_MODEL,
        "",
        (320, 320),
        0.8,
        0.3,
        5000
    )

    recognizer = cv2.FaceRecognizerSF_create(
        RECOGNIZER_MODEL,
        ""
    )

    embeddings = []
    seen_hashes = set()

    for filename in image_files:

        image_path = os.path.join(student_folder, filename)

        with open(image_path, "rb") as file:
            image_hash = hashlib.sha256(file.read()).hexdigest()

        if image_hash in seen_hashes:
            continue

        seen_hashes.add(image_hash)

        image = cv2.imread(image_path)

        if image is None:
            continue

        height, width = image.shape[:2]
        detector.setInputSize((width, height))

        _, faces = detector.detect(image)

        if faces is None or len(faces) != 1:
            continue

        face = faces[0]

        if not check_image_quality(image, face):
            continue

        aligned_face = recognizer.alignCrop(image, face)
        feature = recognizer.feature(aligned_face)

        feature = np.asarray(
            feature,
            dtype=np.float32
        ).flatten()

        if not np.all(np.isfinite(feature)):
            continue

        norm = np.linalg.norm(feature)

        if norm == 0:
            continue

        embeddings.append(feature / norm)

    if not embeddings:
        raise ValueError(
            "No usable facial embeddings could be extracted."
        )

    return np.asarray(embeddings, dtype=np.float32)


def update_student_embeddings(matric_number):

    new_embeddings = extract_student_embeddings(matric_number)

    if os.path.exists(EMBEDDINGS_FILE):

        existing_embeddings = np.load(
            EMBEDDINGS_FILE,
            allow_pickle=True
        ).item()

        if not isinstance(existing_embeddings, dict):
            raise ValueError("Existing embeddings file is invalid.")

        updated_embeddings = existing_embeddings.copy()

        backup_embeddings()

    else:
        updated_embeddings = {}

    updated_embeddings[matric_number] = new_embeddings

    save_embeddings_atomically(updated_embeddings)

    return {
        "matric_number": matric_number,
        "images_processed": len(new_embeddings),
        "total_students": len(updated_embeddings)
    }


def delete_student_facial_data(matric_number):
    """
    Remove one student's embeddings and facial dataset.

    Other students' embeddings and dataset folders remain untouched.
    """

    student_folder = os.path.realpath(
        os.path.join(DATASET_PATH, matric_number)
    )

    dataset_root = os.path.realpath(DATASET_PATH)

    # Prevent accidental deletion outside the dataset directory.
    if os.path.commonpath(
        [dataset_root, student_folder]
    ) != dataset_root:
        raise ValueError("Invalid student dataset path.")

    embedding_removed = False
    folder_removed = False

    # Remove only this student's embedding.
    if os.path.isfile(EMBEDDINGS_FILE):

        existing_embeddings = np.load(
            EMBEDDINGS_FILE,
            allow_pickle=True
        ).item()

        if not isinstance(existing_embeddings, dict):
            raise ValueError("Existing embeddings file is invalid.")

        if matric_number in existing_embeddings:

            updated_embeddings = existing_embeddings.copy()

            # Back up before modifying.
            backup_embeddings()

            updated_embeddings.pop(matric_number)

            save_embeddings_atomically(updated_embeddings)

            embedding_removed = True

    # Remove only this student's image folder.
    if os.path.isdir(student_folder):

        shutil.rmtree(student_folder)

        folder_removed = True

    return {
        "matric_number": matric_number,
        "embedding_removed": embedding_removed,
        "folder_removed": folder_removed
    }
