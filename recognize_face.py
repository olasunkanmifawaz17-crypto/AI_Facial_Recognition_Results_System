
import cv2
import numpy as np
from collections import deque

# ==============================
# CONFIGURATION
# ==============================

EMBEDDINGS_FILE = "face_embeddings.npy"

THRESHOLD = 0.55

TOP_K = 5

FRAMES_TO_CHECK = 15
REQUIRED_MATCHES = 12

MIN_AVERAGE_SCORE = 0.55

MIN_FACE_SIZE = 120
MIN_BLUR_SCORE = 50
MIN_BRIGHTNESS = 45
MAX_BRIGHTNESS = 210

# ==============================
# LOAD EMBEDDINGS
# ==============================

try:
    student_embeddings = np.load(
        EMBEDDINGS_FILE,
        allow_pickle=True
    ).item()

except Exception as error:
    raise SystemExit(f"Could not load embeddings: {error}")

matric_number = input(
    "Enter your matric number: "
).strip()

if matric_number not in student_embeddings:
    raise SystemExit(
        "Student not registered or embeddings unavailable."
    )

stored_features = np.asarray(
    student_embeddings[matric_number],
    dtype=np.float32
)

if stored_features.ndim != 2 or len(stored_features) == 0:
    raise SystemExit("Invalid student embeddings.")

# Normalize stored embeddings
norms = np.linalg.norm(
    stored_features,
    axis=1,
    keepdims=True
)

valid_embeddings = (
    np.isfinite(stored_features).all(axis=1)
    & (norms.flatten() > 0)
)

stored_features = stored_features[valid_embeddings]
norms = norms[valid_embeddings]

if len(stored_features) == 0:
    raise SystemExit("No valid embeddings available.")

stored_features = stored_features / norms

TOP_K = min(TOP_K, len(stored_features))

# ==============================
# LOAD MODELS
# ==============================

detector = cv2.FaceDetectorYN_create(
    "models/face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF_create(
    "models/face_recognition_sface_2021dec.onnx",
    ""
)

# ==============================
# OPEN CAMERA
# ==============================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit("Could not access webcam.")

# Stores actual similarity scores
score_history = deque(maxlen=FRAMES_TO_CHECK)

verified = False

print("\nLook at the camera.")
print("Press Q to quit.")
print("Verification in progress...\n")

# ==============================
# IMAGE QUALITY CHECK
# ==============================


def check_face_quality(frame, face):

    x, y, w, h = face[:4].astype(int)

    height, width = frame.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size == 0:
        return False, "Invalid face area"

    if min(face_crop.shape[:2]) < MIN_FACE_SIZE:
        return False, "Move closer to camera"

    gray = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2GRAY
    )

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    if blur_score < MIN_BLUR_SCORE:
        return False, "Image too blurry"

    brightness = np.mean(gray)

    if not MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS:
        return False, "Adjust lighting"

    return True, "Face quality acceptable"


# ==============================
# FACE VERIFICATION
# ==============================

try:

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Camera error.")
            break

        height, width = frame.shape[:2]

        detector.setInputSize((width, height))

        _, faces = detector.detect(frame)

        status = "No face detected"
        color = (0, 0, 255)

        current_score = None
        average_score = None

        if faces is not None and len(faces) == 1:

            face = faces[0]

            quality_ok, quality_message = check_face_quality(
                frame,
                face
            )

            if not quality_ok:

                score_history.clear()
                status = quality_message

            else:

                aligned_face = recognizer.alignCrop(
                    frame,
                    face
                )

                live_feature = recognizer.feature(
                    aligned_face
                ).flatten()

                live_norm = np.linalg.norm(live_feature)

                if (
                    live_norm == 0
                    or not np.all(np.isfinite(live_feature))
                ):

                    score_history.clear()
                    status = "Invalid facial features"

                else:

                    # Normalize live embedding
                    live_feature = live_feature / live_norm

                    # Compare with stored embeddings
                    similarities = np.dot(
                        stored_features,
                        live_feature
                    )

                    # Average the five highest scores
                    top_scores = np.sort(similarities)[-TOP_K:]

                    current_score = float(
                        np.mean(top_scores)
                    )

                    score_history.append(current_score)

                    # Count scores meeting the threshold
                    matching_scores = [
                        value for value in score_history
                        if value >= THRESHOLD
                    ]

                    match_count = len(matching_scores)

                    # Average ALL scores in the window,
                    # including scores below the threshold.
                    if len(score_history) > 0:
                        average_score = float(
                            np.mean(score_history)
                        )

                    status = "Checking identity..."
                    color = (0, 255, 255)

                    # Final verification conditions
                    if (
                        len(score_history) == FRAMES_TO_CHECK
                        and match_count >= REQUIRED_MATCHES
                        and average_score >= MIN_AVERAGE_SCORE
                    ):

                        status = "IDENTITY VERIFIED"
                        color = (0, 255, 0)
                        verified = True

            x, y, w, h = face[:4].astype(int)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

        elif faces is not None and len(faces) > 1:

            score_history.clear()
            status = "Multiple faces detected"

        else:

            score_history.clear()

        # ==============================
        # DISPLAY INFORMATION
        # ==============================

        match_count = sum(
            value >= THRESHOLD
            for value in score_history
        )

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        if current_score is not None:

            cv2.putText(
                frame,
                f"Current score: {current_score:.3f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        if average_score is not None:

            cv2.putText(
                frame,
                f"Window average: {average_score:.3f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        cv2.putText(
            frame,
            f"Matching frames: {match_count}/{FRAMES_TO_CHECK}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "SFace Student Verification",
            frame
        )

        if verified:
            cv2.waitKey(1500)
            print(f"Identity verified for {matric_number}")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Verification cancelled.")
            break

finally:

    camera.release()
    cv2.destroyAllWindows()
