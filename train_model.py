
import cv2
import numpy as np
import os
import hashlib
import shutil
from datetime import datetime
from collections import Counter

# ==============================
# CONFIGURATION
# ==============================

DATASET_PATH = "dataset"

DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "models/face_recognition_sface_2021dec.onnx"

OUTPUT_FILE = "face_embeddings.npy"

MIN_FACE_SIZE = 120
MIN_BLUR_SCORE = 50
MIN_BRIGHTNESS = 45
MAX_BRIGHTNESS = 210

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# ==============================
# INITIAL VALIDATION
# ==============================

if not os.path.isdir(DATASET_PATH):
    raise SystemExit(f"Dataset folder not found: {DATASET_PATH}")

if not os.path.isfile(DETECTOR_MODEL):
    raise SystemExit(f"YuNet model not found: {DETECTOR_MODEL}")

if not os.path.isfile(RECOGNIZER_MODEL):
    raise SystemExit(f"SFace model not found: {RECOGNIZER_MODEL}")

# ==============================
# LOAD MODELS
# ==============================

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

student_embeddings = {}

# ==============================
# IMAGE QUALITY CHECK
# ==============================


def check_image_quality(image, face):

    x, y, w, h = face[:4].astype(int)

    height, width = image.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    face_crop = image[y1:y2, x1:x2]

    if face_crop.size == 0:
        return False, "invalid_crop"

    if min(face_crop.shape[:2]) < MIN_FACE_SIZE:
        return False, "face_too_small"

    gray = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2GRAY
    )

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    if blur_score < MIN_BLUR_SCORE:
        return False, "blurry"

    brightness = np.mean(gray)

    if not MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS:
        return False, "poor_brightness"

    return True, "accepted"


# ==============================
# PROCESS STUDENT DATASETS
# ==============================

student_folders = sorted(os.listdir(DATASET_PATH))

total_students = 0
total_images = 0
total_usable = 0

for matric_number in student_folders:

    student_folder = os.path.join(
        DATASET_PATH,
        matric_number
    )

    if not os.path.isdir(student_folder):
        continue

    total_students += 1

    embeddings = []
    seen_hashes = set()

    rejection_counts = Counter()

    image_files = [
        filename
        for filename in sorted(os.listdir(student_folder))
        if filename.lower().endswith(IMAGE_EXTENSIONS)
    ]

    print(f"\nProcessing: {matric_number}")

    for filename in image_files:

        total_images += 1

        image_path = os.path.join(
            student_folder,
            filename
        )

        # Detect exact duplicate files
        with open(image_path, "rb") as file:
            image_hash = hashlib.sha256(
                file.read()
            ).hexdigest()

        if image_hash in seen_hashes:
            rejection_counts["duplicate"] += 1
            continue

        seen_hashes.add(image_hash)

        # Read image
        image = cv2.imread(image_path)

        if image is None:
            rejection_counts["unreadable"] += 1
            continue

        height, width = image.shape[:2]

        detector.setInputSize((width, height))

        _, faces = detector.detect(image)

        # Require exactly one face
        if faces is None or len(faces) == 0:
            rejection_counts["no_face"] += 1
            continue

        if len(faces) > 1:
            rejection_counts["multiple_faces"] += 1
            continue

        face = faces[0]

        # Check image quality
        valid, reason = check_image_quality(
            image,
            face
        )

        if not valid:
            rejection_counts[reason] += 1
            continue

        # Align face
        aligned_face = recognizer.alignCrop(
            image,
            face
        )

        # Extract SFace embedding
        feature = recognizer.feature(
            aligned_face
        )

        feature = np.asarray(
            feature,
            dtype=np.float32
        ).flatten()

        # Validate embedding
        if not np.all(np.isfinite(feature)):
            rejection_counts["invalid_embedding"] += 1
            continue

        norm = np.linalg.norm(feature)

        if norm == 0:
            rejection_counts["invalid_embedding"] += 1
            continue

        # Normalize embedding
        feature = feature / norm

        embeddings.append(feature)

    # Save embeddings for this student
    if embeddings:

        student_embeddings[matric_number] = np.asarray(
            embeddings,
            dtype=np.float32
        )

        usable = len(embeddings)
        total_usable += usable

        print(f"Images found: {len(image_files)}")
        print(f"Usable embeddings: {usable}")
        print(f"Rejected images: {sum(rejection_counts.values())}")

        if rejection_counts:
            print("Rejection details:")

            for reason, count in rejection_counts.items():
                print(f"  {reason}: {count}")

    else:
        print("No usable facial samples found.")

# ==============================
# SAVE EMBEDDINGS
# ==============================

if not student_embeddings:
    raise SystemExit(
        "\nNo valid embeddings generated. "
        "Existing embedding file was not changed."
    )

# Back up previous embeddings before replacing
if os.path.exists(OUTPUT_FILE):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_file = f"face_embeddings_{timestamp}.backup.npy"

    shutil.copy2(
        OUTPUT_FILE,
        backup_file
    )

    print(f"\nPrevious embeddings backed up to: {backup_file}")

# Save new embeddings
np.save(
    OUTPUT_FILE,
    student_embeddings,
    allow_pickle=True
)

# ==============================
# FINAL SUMMARY
# ==============================

print("\n" + "=" * 40)
print("EMBEDDING GENERATION COMPLETED")
print("=" * 40)

print("Student folders processed:", total_students)
print("Students with usable embeddings:", len(student_embeddings))
print("Total images examined:", total_images)
print("Total usable embeddings:", total_usable)
print("Output file:", OUTPUT_FILE)

for matric_number, embeddings in student_embeddings.items():
    print(f"{matric_number}: {len(embeddings)} embeddings")
