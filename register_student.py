
import cv2
import os
import time
import shutil
import numpy as np

from pathlib import Path
from collections import deque


# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"

TARGET_TOTAL = 30

STABLE_FRAMES_REQUIRED = 3
POSE_FRAMES_REQUIRED = 3
CAPTURE_COOLDOWN = 0.35

MIN_FACE_SIZE = 120
MIN_BLUR_SCORE = 50
MIN_BRIGHTNESS = 45
MAX_BRIGHTNESS = 210

CALIBRATION_SAMPLES = 8
MIN_SAMPLE_DIFFERENCE = 3.0

LEFT_THRESHOLD = 0.20
RIGHT_THRESHOLD = -0.18

FRONT_HORIZONTAL_LIMIT = 0.12
FRONT_VERTICAL_LIMIT = 0.12

MAX_OTHER_AXIS_MOVEMENT = 0.25

STAGES = [
    ("Look directly at the camera", 12, "front"),
    ("Turn your head slightly LEFT", 9, "left"),
    ("Turn your head slightly RIGHT", 9, "right")
]


# ---------------- HELPER FUNCTIONS ----------------

def get_face_crop(frame, face):

    x, y, w, h = face[:4].astype(int)

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame.shape[1], x + w)
    y2 = min(frame.shape[0], y + h)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop


def assess_quality(crop):

    if crop is None:
        return False, "Invalid face image"

    height, width = crop.shape[:2]

    if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
        return False, "Move closer to camera"

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    brightness = np.mean(gray)

    if blur_score < MIN_BLUR_SCORE:
        return False, "Image too blurry"

    if brightness < MIN_BRIGHTNESS:
        return False, "Lighting too dark"

    if brightness > MAX_BRIGHTNESS:
        return False, "Lighting too bright"

    return True, "Image quality acceptable"


def get_landmark_measurements(face):

    points = face[4:14].reshape(5, 2)

    right_eye = points[0]
    left_eye = points[1]
    nose = points[2]

    eye_mid = (right_eye + left_eye) / 2

    eye_vector = left_eye - right_eye
    eye_distance = np.linalg.norm(eye_vector)

    if eye_distance < 1:
        return None

    eye_axis = eye_vector / eye_distance

    perpendicular = np.array([
        -eye_axis[1],
        eye_axis[0]
    ])

    nose_vector = nose - eye_mid

    horizontal = np.dot(
        nose_vector,
        eye_axis
    ) / eye_distance

    vertical = np.dot(
        nose_vector,
        perpendicular
    ) / eye_distance

    return np.array([horizontal, vertical])


def is_stable(stable_frames):

    if len(stable_frames) < STABLE_FRAMES_REQUIRED:
        return False

    boxes = np.array(stable_frames)

    centers = boxes[:, :2] + boxes[:, 2:4] / 2

    center_difference = np.max(
        np.linalg.norm(
            centers - centers[-1],
            axis=1
        )
    )

    areas = boxes[:, 2] * boxes[:, 3]

    area_difference = (
        np.max(areas) - np.min(areas)
    ) / max(np.mean(areas), 1)

    return (
        center_difference < 8
        and area_difference < 0.12
    )


def check_pose(direction, horizontal, vertical):

    if direction == "front":

        return (
            abs(horizontal) <= FRONT_HORIZONTAL_LIMIT
            and abs(vertical) <= FRONT_VERTICAL_LIMIT
        )

    elif direction == "left":

        return (
            horizontal >= LEFT_THRESHOLD
            and abs(vertical) <= MAX_OTHER_AXIS_MOVEMENT
        )

    elif direction == "right":

        return (
            horizontal <= RIGHT_THRESHOLD
            and abs(vertical) <= MAX_OTHER_AXIS_MOVEMENT
        )

    return False


def get_pose_instruction(direction):

    instructions = {
        "front": "Face the camera directly",
        "left": "Turn your head slightly LEFT",
        "right": "Turn your head slightly RIGHT"
    }

    return instructions.get(
        direction,
        "Follow the instruction"
    )


def is_different_sample(crop, last_sample):

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(gray, (100, 100))

    if last_sample is None:
        return True, gray

    difference = np.mean(
        cv2.absdiff(gray, last_sample)
    )

    return difference >= MIN_SAMPLE_DIFFERENCE, gray


# ---------------- MAIN REGISTRATION FUNCTION ----------------

def register_student(matric_number):

    matric_number = str(matric_number).strip()

    if (
        not matric_number
        or not matric_number.isalnum()
        or matric_number.startswith("0")
    ):
        raise ValueError("Invalid student matric number.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YuNet model not found: {MODEL_PATH}"
        )

    student_folder = DATASET_PATH / matric_number

    if student_folder.exists():
        raise FileExistsError(
            "This student already has a facial dataset. "
            "Existing facial data will not be overwritten."
        )

    # Create detector only when registration is requested.
    detector = cv2.FaceDetectorYN_create(
        str(MODEL_PATH),
        "",
        (320, 320),
        0.8,
        0.3,
        5000
    )

    DATASET_PATH.mkdir(parents=True, exist_ok=True)
    student_folder.mkdir()

    camera = None
    completed = False
    total_count = 0

    try:

        # ---------------- OPEN CAMERA ----------------

        camera = cv2.VideoCapture(0)

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not camera.isOpened():
            raise RuntimeError("Could not access webcam.")

        # ---------------- STATE VARIABLES ----------------

        stage_index = 0
        stage_count = 0

        stable_frames = deque(
            maxlen=STABLE_FRAMES_REQUIRED
        )

        pose_history = deque(maxlen=5)

        pose_match_frames = deque(
            maxlen=POSE_FRAMES_REQUIRED
        )

        baseline_samples = deque(
            maxlen=CALIBRATION_SAMPLES
        )

        baseline = None

        last_capture_time = 0
        last_sample = None

        horizontal = 0.0
        vertical = 0.0

        print("\nAUTOMATIC FACIAL REGISTRATION")
        print("--------------------------------")
        print("Student:", matric_number)
        print("Total images required: 30")
        print("1. Look directly at the camera.")
        print("2. Turn your head slightly LEFT.")
        print("3. Turn your head slightly RIGHT.")
        print("The system captures images automatically.")
        print("Press Q to stop registration.\n")

        # ---------------- REGISTRATION LOOP ----------------

        while stage_index < len(STAGES):

            ret, frame = camera.read()

            if not ret:
                raise RuntimeError("Camera error.")

            height, width = frame.shape[:2]

            detector.setInputSize((width, height))
            _, faces = detector.detect(frame)

            display = frame.copy()

            instruction, required, direction = STAGES[stage_index]

            status = "Position your face"
            color = (0, 0, 255)

            if faces is not None and len(faces) == 1:

                face = faces[0]

                x, y, w, h = face[:4].astype(int)

                crop = get_face_crop(frame, face)

                quality_ok, quality_message = assess_quality(crop)

                if quality_ok:

                    measurements = get_landmark_measurements(face)

                    if measurements is not None:

                        current_box = np.array(
                            [x, y, w, h],
                            dtype=np.float32
                        )

                        stable_frames.append(current_box)

                        status = "Hold your position..."
                        color = (0, 255, 255)

                        # -------- INITIAL CALIBRATION --------

                        if baseline is None:

                            if is_stable(stable_frames):

                                baseline_samples.append(measurements)

                                status = (
                                    f"Calibrating: "
                                    f"{len(baseline_samples)}/"
                                    f"{CALIBRATION_SAMPLES}"
                                )

                                if len(baseline_samples) == CALIBRATION_SAMPLES:

                                    calibration_data = np.array(
                                        baseline_samples
                                    )

                                    spread = np.ptp(
                                        calibration_data,
                                        axis=0
                                    )

                                    if np.max(spread) <= 0.08:

                                        baseline = np.median(
                                            calibration_data,
                                            axis=0
                                        )

                                        print(
                                            "Face calibration completed:",
                                            baseline
                                        )

                                        baseline_samples.clear()
                                        pose_history.clear()
                                        stable_frames.clear()

                                        status = "Calibration completed!"
                                        color = (0, 255, 0)

                                    else:

                                        baseline_samples.clear()

                                        status = (
                                            "Hold your head still "
                                            "while facing forward"
                                        )

                            else:

                                baseline_samples.clear()
                                status = "Hold still for calibration"

                        # -------- POSE CHECKING --------

                        else:

                            pose_history.append(measurements)

                            average_pose = np.median(
                                np.array(pose_history),
                                axis=0
                            )

                            horizontal = average_pose[0] - baseline[0]
                            vertical = average_pose[1] - baseline[1]

                            pose_ok = check_pose(
                                direction,
                                horizontal,
                                vertical
                            )

                            pose_match_frames.append(pose_ok)

                            if not pose_ok:

                                status = get_pose_instruction(direction)
                                color = (0, 165, 255)

                            elif len(pose_match_frames) < POSE_FRAMES_REQUIRED:

                                status = "Correct direction - hold briefly"
                                color = (0, 255, 255)

                            elif not all(pose_match_frames):

                                status = "Hold the requested direction"
                                color = (0, 255, 255)

                            elif not is_stable(stable_frames):

                                status = "Keep your face steady"
                                color = (0, 255, 255)

                            else:

                                current_time = time.monotonic()

                                if (
                                    current_time - last_capture_time
                                    >= CAPTURE_COOLDOWN
                                ):

                                    different, gray_sample = (
                                        is_different_sample(
                                            crop,
                                            last_sample
                                        )
                                    )

                                    if different:

                                        image_path = (
                                            student_folder
                                            / f"face_{total_count + 1}.jpg"
                                        )

                                        if cv2.imwrite(
                                            str(image_path),
                                            frame
                                        ):

                                            stage_count += 1
                                            total_count += 1

                                            last_sample = gray_sample
                                            last_capture_time = current_time

                                            status = "Image captured!"
                                            color = (0, 255, 0)

                                            print(
                                                f"Stage {stage_index + 1}: "
                                                f"{stage_count}/{required}"
                                            )

                                        else:

                                            status = "Could not save image"
                                            color = (0, 0, 255)

                                    else:

                                        status = "Move slightly for a new sample"
                                        color = (0, 165, 255)

                else:

                    stable_frames.clear()
                    pose_history.clear()
                    pose_match_frames.clear()

                    status = quality_message

            elif faces is not None and len(faces) > 1:

                stable_frames.clear()
                pose_history.clear()
                pose_match_frames.clear()

                status = "Multiple faces detected"

            else:

                stable_frames.clear()
                pose_history.clear()
                pose_match_frames.clear()

                status = "No face detected"

            # ---------------- DISPLAY ----------------

            if faces is not None and len(faces) == 1:

                cv2.rectangle(
                    display,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )

            cv2.putText(
                display,
                instruction,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                status,
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

            cv2.putText(
                display,
                f"Stage: {stage_count}/{required}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Total: {total_count}/{TARGET_TOTAL}",
                (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            if baseline is not None and len(pose_history) > 0:

                cv2.putText(
                    display,
                    f"H: {horizontal:+.3f}  V: {vertical:+.3f}",
                    (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1
                )

            cv2.putText(
                display,
                "Press Q to stop",
                (10, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "Automatic Student Registration",
                display
            )

            # -------- MOVE TO NEXT STAGE --------

            if stage_count >= required:

                print(f"Stage {stage_index + 1} completed.")

                stage_index += 1
                stage_count = 0

                stable_frames.clear()
                pose_history.clear()
                pose_match_frames.clear()
                last_sample = None

                if stage_index < len(STAGES):

                    print(
                        "\nNext instruction:",
                        STAGES[stage_index][0]
                    )

                time.sleep(0.3)

            if cv2.waitKey(1) & 0xFF == ord("q"):

                print("Registration stopped by user.")
                break

        # ---------------- CHECK COMPLETION ----------------

        if total_count == TARGET_TOTAL:

            completed = True

            print("\nRegistration completed successfully.")
            print("Matric number:", matric_number)
            print("Total images:", total_count)

            return str(student_folder)

        print("\nRegistration incomplete.")
        print("Images captured:", total_count)
        print("Incomplete facial data will be removed.")

        return None

    finally:

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()

        # Never leave incomplete facial datasets behind.
        if not completed and student_folder.exists():
            shutil.rmtree(student_folder)
