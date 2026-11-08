
import cv2
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


class RegistrationService:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YuNet model not found: {MODEL_PATH}"
            )

        self.detector = cv2.FaceDetectorYN_create(
            str(MODEL_PATH),
            "",
            (320, 320),
            0.8,
            0.3,
            5000
        )

        # Separate registration state for each browser session.
        self.sessions = {}

    # ---------------- IMAGE HELPERS ----------------

    def get_face_crop(self, frame, face):

        x, y, w, h = face[:4].astype(int)

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return crop

    def assess_quality(self, crop):

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

    def get_landmark_measurements(self, face):

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

    def is_stable(self, stable_frames):

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

    def check_pose(self, direction, horizontal, vertical):

        if direction == "front":
            return (
                abs(horizontal) <= FRONT_HORIZONTAL_LIMIT
                and abs(vertical) <= FRONT_VERTICAL_LIMIT
            )

        if direction == "left":
            return (
                horizontal >= LEFT_THRESHOLD
                and abs(vertical) <= MAX_OTHER_AXIS_MOVEMENT
            )

        if direction == "right":
            return (
                horizontal <= RIGHT_THRESHOLD
                and abs(vertical) <= MAX_OTHER_AXIS_MOVEMENT
            )

        return False

    def is_different_sample(self, crop, last_sample):

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

    # ---------------- REGISTRATION CONTROL ----------------

    def start_registration(self, session_key, matric_number):

        matric_number = str(matric_number).strip()

        if (
            not matric_number
            or not matric_number.isalnum()
            or matric_number.startswith("0")
        ):
            raise ValueError("Invalid student matric number.")

        if session_key in self.sessions:
            self.stop_registration(session_key)

        student_folder = DATASET_PATH / matric_number

        if student_folder.exists():
            raise FileExistsError(
                "This student already has a facial dataset. "
                "Existing facial data will not be overwritten."
            )

        DATASET_PATH.mkdir(parents=True, exist_ok=True)
        student_folder.mkdir()

        self.sessions[session_key] = {
            "matric_number": matric_number,
            "student_folder": student_folder,

            "stage_index": 0,
            "stage_count": 0,
            "total_count": 0,

            "stable_frames": deque(
                maxlen=STABLE_FRAMES_REQUIRED
            ),

            "pose_history": deque(maxlen=5),

            "pose_match_frames": deque(
                maxlen=POSE_FRAMES_REQUIRED
            ),

            "baseline_samples": deque(
                maxlen=CALIBRATION_SAMPLES
            ),

            "baseline": None,

            "last_capture_time": 0,
            "last_sample": None,

            "horizontal": 0.0,
            "vertical": 0.0,
        }

        return {
            "message": "Camera registration is ready.",
            "instruction": STAGES[0][0],
            "progress": 0,
            "stage_progress": 0,
            "stage_required": STAGES[0][1],
            "stage": 1,
            "total": TARGET_TOTAL,
            "complete": False
        }

    def stop_registration(self, session_key):

        session = self.sessions.pop(session_key, None)

        if session is not None:

            folder = session["student_folder"]

            if folder.exists():
                shutil.rmtree(folder)

    # ---------------- FRAME PROCESSING ----------------

    def process_frame(self, session_key, frame):

        session = self.sessions.get(session_key)

        if session is None:
            raise ValueError(
                "Registration session not found. Start registration again."
            )

        if frame is None:
            raise ValueError("Invalid camera frame.")

        height, width = frame.shape[:2]

        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(frame)

        stage_index = session["stage_index"]

        instruction, required, direction = STAGES[stage_index]

        status = "Position your face"

        face_detected = faces is not None and len(faces) == 1

        if face_detected:

            face = faces[0]

            x, y, w, h = face[:4].astype(int)

            crop = self.get_face_crop(frame, face)

            quality_ok, quality_message = self.assess_quality(crop)

            if quality_ok:

                measurements = self.get_landmark_measurements(face)

                if measurements is not None:

                    current_box = np.array(
                        [x, y, w, h],
                        dtype=np.float32
                    )

                    session["stable_frames"].append(current_box)

                    stable_frames = session["stable_frames"]

                    status = "Hold your position..."

                    # -------- INITIAL CALIBRATION --------

                    if session["baseline"] is None:

                        if self.is_stable(stable_frames):

                            session["baseline_samples"].append(
                                measurements
                            )

                            count = len(session["baseline_samples"])

                            status = (
                                f"Calibrating: {count}/"
                                f"{CALIBRATION_SAMPLES}"
                            )

                            if count == CALIBRATION_SAMPLES:

                                calibration_data = np.array(
                                    session["baseline_samples"]
                                )

                                spread = np.ptp(
                                    calibration_data,
                                    axis=0
                                )

                                if np.max(spread) <= 0.08:

                                    session["baseline"] = np.median(
                                        calibration_data,
                                        axis=0
                                    )

                                    session["baseline_samples"].clear()
                                    session["pose_history"].clear()
                                    session["stable_frames"].clear()

                                    status = "Calibration completed!"

                                else:

                                    session["baseline_samples"].clear()

                                    status = (
                                        "Hold your head still "
                                        "while facing forward"
                                    )

                        else:

                            session["baseline_samples"].clear()
                            status = "Hold still for calibration"

                    # -------- POSE CHECKING --------

                    else:

                        session["pose_history"].append(measurements)

                        average_pose = np.median(
                            np.array(session["pose_history"]),
                            axis=0
                        )

                        horizontal = (
                            average_pose[0] - session["baseline"][0]
                        )

                        vertical = (
                            average_pose[1] - session["baseline"][1]
                        )

                        session["horizontal"] = horizontal
                        session["vertical"] = vertical

                        pose_ok = self.check_pose(
                            direction,
                            horizontal,
                            vertical
                        )

                        session["pose_match_frames"].append(pose_ok)

                        if not pose_ok:

                            status = instruction

                        elif len(session["pose_match_frames"]) < POSE_FRAMES_REQUIRED:

                            status = "Correct direction - hold briefly"

                        elif not all(session["pose_match_frames"]):

                            status = "Hold the requested direction"

                        elif not self.is_stable(stable_frames):

                            status = "Keep your face steady"

                        else:

                            current_time = time.monotonic()

                            if (
                                current_time - session["last_capture_time"]
                                >= CAPTURE_COOLDOWN
                            ):

                                different, gray_sample = (
                                    self.is_different_sample(
                                        crop,
                                        session["last_sample"]
                                    )
                                )

                                if different:

                                    total_count = session["total_count"]

                                    image_path = (
                                        session["student_folder"]
                                        / f"face_{total_count + 1}.jpg"
                                    )

                                    if cv2.imwrite(
                                        str(image_path),
                                        frame
                                    ):

                                        session["stage_count"] += 1
                                        session["total_count"] += 1

                                        session["last_sample"] = gray_sample
                                        session["last_capture_time"] = current_time

                                        status = "Image captured!"

                                    else:

                                        status = "Could not save image"

                                else:

                                    status = "Move slightly for a new sample"

            else:

                session["stable_frames"].clear()
                session["pose_history"].clear()
                session["pose_match_frames"].clear()

                status = quality_message

        elif faces is not None and len(faces) > 1:

            session["stable_frames"].clear()
            session["pose_history"].clear()
            session["pose_match_frames"].clear()

            status = "Multiple faces detected"

        else:

            session["stable_frames"].clear()
            session["pose_history"].clear()
            session["pose_match_frames"].clear()

            status = "No face detected"

        # -------- MOVE TO NEXT STAGE --------

        if session["stage_count"] >= required:

            session["stage_index"] += 1
            session["stage_count"] = 0

            session["stable_frames"].clear()
            session["pose_history"].clear()
            session["pose_match_frames"].clear()

            session["last_sample"] = None

            if session["stage_index"] >= len(STAGES):

                dataset_path = str(session["student_folder"])

                self.sessions.pop(session_key, None)

                return {
                    "message": "Facial registration completed successfully.",
                    "instruction": "Registration completed!",
                    "status": "Registration completed successfully.",
                    "progress": TARGET_TOTAL,
                    "stage_progress": TARGET_TOTAL,
                    "stage_required": TARGET_TOTAL,
                    "stage": 3,
                    "total": TARGET_TOTAL,
                    "complete": True,
                    "dataset_path": dataset_path
                }

            instruction = STAGES[session["stage_index"]][0]
            status = "Stage completed. Follow the next instruction."

        # -------- RESPONSE --------

        current_stage = session["stage_index"]
        current_instruction, current_required, _ = STAGES[current_stage]

        return {
            "message": status,
            "status": status,
            "instruction": current_instruction,
            "progress": session["total_count"],
            "stage_progress": session["stage_count"],
            "stage_required": current_required,
            "stage": current_stage + 1,
            "total": TARGET_TOTAL,
            "complete": False,
            "face_detected": face_detected
        }


# One shared service instance for Django.
registration_service = RegistrationService()
