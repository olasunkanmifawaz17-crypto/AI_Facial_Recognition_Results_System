
import cv2
import numpy as np
from collections import deque

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

detector = cv2.FaceDetectorYN_create(
    MODEL_PATH,
    "",
    (640, 480),
    0.8,
    0.3,
    5000
)

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    print("Could not access webcam.")
    raise SystemExit

baseline = None
history = deque(maxlen=10)
paused = False
display = None

print("\nFACIAL LANDMARK MOVEMENT TEST")
print("--------------------------------")
print("1. Look straight and press C to calibrate.")
print("2. Move your head left, right, up and down.")
print("3. SPACE freezes/resumes.")
print("4. Q quits.")

while True:

    if not paused:

        ret, frame = camera.read()

        if not ret:
            print("Camera error.")
            break

        height, width = frame.shape[:2]

        detector.setInputSize((width, height))
        _, faces = detector.detect(frame)

        display = frame.copy()

        if faces is not None and len(faces) == 1:

            face = faces[0]

            # YuNet landmark order:
            # Right eye, left eye, nose,
            # right mouth corner, left mouth corner.

            points = face[4:14].reshape(5, 2)

            right_eye = points[0]
            left_eye = points[1]
            nose = points[2]

            eye_mid = (right_eye + left_eye) / 2

            eye_vector = left_eye - right_eye
            eye_distance = np.linalg.norm(eye_vector)

            if eye_distance > 1:

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

                history.append([horizontal, vertical])

                average = np.mean(history, axis=0)

                x, y, w, h = face[:4].astype(int)

                cv2.rectangle(
                    display,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    display,
                    tuple(nose.astype(int)),
                    5,
                    (0, 0, 255),
                    -1
                )

                if baseline is not None:

                    horizontal_change = (
                        average[0] - baseline[0]
                    )

                    vertical_change = (
                        average[1] - baseline[1]
                    )

                    cv2.putText(
                        display,
                        f"Horizontal: {horizontal_change:+.3f}",
                        (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        display,
                        f"Vertical: {vertical_change:+.3f}",
                        (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )

                else:

                    cv2.putText(
                        display,
                        "Press C while looking straight",
                        (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

        else:

            history.clear()

            cv2.putText(
                display,
                "Keep one face in view",
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    if display is not None:

        cv2.putText(
            display,
            "C: Calibrate | SPACE: Freeze | Q: Quit",
            (15, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display,
            "FROZEN" if paused else "LIVE",
            (15, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow("Facial Landmark Test", display)

    key = cv2.waitKey(30) & 0xFF

    if key == ord("c") and not paused:

        if len(history) >= 5:
            baseline = np.mean(history, axis=0)
            print("Straight position calibrated:", baseline)
        else:
            print("Keep your face visible before calibrating.")

    elif key == ord(" "):
        paused = not paused

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
