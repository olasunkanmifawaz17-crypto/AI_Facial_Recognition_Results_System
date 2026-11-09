import cv2
import numpy as np

from collections import deque
from decimal import Decimal, ROUND_HALF_UP

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import ensure_csrf_cookie
from django.contrib.admin.views.decorators import staff_member_required
from django.utils import timezone

from .models import Student

from recognition_service import FaceRecognitionService
from registration_service import registration_service
from embedding_service import update_student_embeddings


# ---------------- FACIAL RECOGNITION ----------------

# Load recognition models once when Django starts.
recognition_service = FaceRecognitionService()

# Recognition settings.
FRAMES_TO_CHECK = 8
REQUIRED_MATCHES = 6
MATCH_THRESHOLD = 0.55
MIN_AVERAGE_SCORE = 0.55

# Store verification progress separately for each session.
verification_sessions = {}


@ensure_csrf_cookie
def camera_test(request):

    if not request.session.session_key:
        request.session.create()

    session_key = request.session.session_key

    verification_sessions.pop(session_key, None)

    request.session.pop("verified_matric_number", None)
    request.session.pop("verified_at", None)

    return render(request, 'results/camera.html')


@require_POST
def receive_frame(request):

    matric_number = request.POST.get("matric_number", "").strip()

    if not matric_number:
        return JsonResponse({
            "message": "Enter your matric number first."
        }, status=400)

    student_exists = Student.objects.filter(
        matric_number=matric_number
    ).exists()

    if not student_exists:
        return JsonResponse({
            "message": "Student record not found."
        }, status=404)

    # Refresh embeddings only if this student's data
    # is missing from the recognition service's memory.
    if matric_number not in recognition_service.student_embeddings:

        recognition_service.reload_embeddings()

    if matric_number not in recognition_service.student_embeddings:
        return JsonResponse({
            "message": "Student facial data unavailable."
        }, status=404)

    uploaded_file = request.FILES.get("frame")

    if not uploaded_file:
        return JsonResponse({
            "message": "No frame received."
        }, status=400)

    image_bytes = np.frombuffer(
        uploaded_file.read(),
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_bytes,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return JsonResponse({
            "message": "Invalid image received."
        }, status=400)

    session_key = request.session.session_key

    if not session_key:
        request.session.create()
        session_key = request.session.session_key

    session = verification_sessions.get(session_key)

    if session is None or session["matric_number"] != matric_number:

        request.session.pop("verified_matric_number", None)
        request.session.pop("verified_at", None)

        session = {
            "matric_number": matric_number,
            "scores": deque(maxlen=FRAMES_TO_CHECK),
            "verified": False
        }

        verification_sessions[session_key] = session

    # Every submitted frame must be processed.
    result = recognition_service.compare_frame(
        matric_number,
        frame
    )

    if not result["valid"]:

        session["scores"].clear()

        return JsonResponse({
            "verified": False,
            "message": result["message"],
            "progress": 0,
            "matches": 0
        })

    score = result["score"]
    session["scores"].append(score)

    scores = list(session["scores"])

    matches = sum(
        value >= MATCH_THRESHOLD for value in scores
    )

    average_score = float(np.mean(scores))
    progress = len(scores)

    verified = (
        progress == FRAMES_TO_CHECK
        and matches >= REQUIRED_MATCHES
        and average_score >= MIN_AVERAGE_SCORE
    )

    if verified:

        session["verified"] = True

        request.session["verified_matric_number"] = matric_number
        request.session["verified_at"] = timezone.now().timestamp()

    message = (
        "IDENTITY VERIFIED"
        if verified
        else "Checking identity..."
    )

    return JsonResponse({
        "verified": verified,
        "message": message,
        "progress": progress,
        "matches": matches,
        "average_score": round(average_score, 3)
    })


# ---------------- GPA CALCULATIONS ----------------

def get_grade_details(score):

    if score >= Decimal("70"):
        return "A", 5

    elif score >= Decimal("60"):
        return "B", 4

    elif score >= Decimal("50"):
        return "C", 3

    elif score >= Decimal("45"):
        return "D", 2

    elif score >= Decimal("40"):
        return "E", 1

    return "F", 0


def calculate_gpa(courses):

    # GPA cannot be calculated accurately if any course
    # in the semester is missing its credit unit.
    if not courses or any(
        course.credit_unit is None
        for course in courses
    ):
        return {
            "available": False,
            "gpa": None,
            "total_credit_units": None
        }

    total_quality_points = Decimal("0")
    total_credit_units = 0

    for course in courses:

        credit_unit = course.credit_unit

        quality_points = (
            Decimal(course.grade_point) *
            Decimal(credit_unit)
        )

        total_quality_points += quality_points
        total_credit_units += credit_unit

    gpa = (
        total_quality_points /
        Decimal(total_credit_units)
    ).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP
    )

    return {
        "available": True,
        "gpa": gpa,
        "total_credit_units": total_credit_units
    }


# ---------------- STUDENT RESULTS ----------------

def student_results(request):

    matric_number = request.session.get(
        "verified_matric_number"
    )

    verified_at = request.session.get("verified_at")

    if not matric_number or not verified_at:
        return redirect("camera_test")

    current_time = timezone.now().timestamp()

    if current_time - verified_at > 300:

        request.session.pop("verified_matric_number", None)
        request.session.pop("verified_at", None)

        session_key = request.session.session_key

        if session_key:
            verification_sessions.pop(session_key, None)

        return redirect("camera_test")

    student = get_object_or_404(
        Student,
        matric_number=matric_number
    )

    # Check payment clearance before retrieving results.
    if not student.fee_cleared:

        return render(
            request,
            "results/student_results.html",
            {
                "student": student,
                "fee_cleared": False,
                "results": [],
                "academic_sessions": {},
                "cgpa_available": False,
                "cgpa": None,
            },
            status=403
        )

    results = list(
        student.results.all().order_by(
            "-academic_session",
            "semester",
            "course_code"
        )
    )

    academic_sessions = {}

    # Assign grades and grade points to individual courses.
    for result in results:

        result.grade, result.grade_point = get_grade_details(
            result.score
        )

    # Group results by academic session and semester.
    for result in results:

        session_name = result.academic_session
        semester_name = result.semester

        if session_name not in academic_sessions:

            academic_sessions[session_name] = {
                "First": [],
                "Second": []
            }

        academic_sessions[session_name][semester_name].append(
            result
        )

    # Calculate semester averages and GPAs.
    for session_name, semesters in academic_sessions.items():

        for semester_name, courses in semesters.items():

            if courses:

                total_score = sum(
                    (course.score for course in courses),
                    Decimal("0")
                )

                average = (
                    total_score / Decimal(len(courses))
                ).quantize(
                    Decimal("0.01"),
                    rounding=ROUND_HALF_UP
                )

            else:
                average = Decimal("0")

            gpa_details = calculate_gpa(courses)

            semesters[semester_name] = {
                "courses": courses,
                "average": average,
                "course_count": len(courses),
                "gpa": gpa_details["gpa"],
                "gpa_available": gpa_details["available"],
                "total_credit_units": gpa_details["total_credit_units"],
            }

    # Calculate cumulative GPA across all academic sessions.
    cgpa_details = calculate_gpa(results)

    return render(
        request,
        "results/student_results.html",
        {
            "student": student,
            "fee_cleared": True,
            "results": results,
            "academic_sessions": academic_sessions,
            "cgpa": cgpa_details["gpa"],
            "cgpa_available": cgpa_details["available"],
            "total_credit_units": cgpa_details["total_credit_units"],
        }
    )


# ---------------- BROWSER FACIAL REGISTRATION ----------------

@staff_member_required
@require_POST
def start_face_registration(request):

    student_id = request.POST.get("student_id")

    student = get_object_or_404(
        Student,
        pk=student_id
    )

    if not request.session.session_key:
        request.session.create()

    session_key = request.session.session_key

    try:

        result = registration_service.start_registration(
            session_key,
            student.matric_number
        )

        request.session["face_registration_student_id"] = student.pk

        return JsonResponse(result)

    except (ValueError, FileExistsError) as error:

        return JsonResponse({
            "message": str(error)
        }, status=400)


@staff_member_required
@require_POST
def receive_registration_frame(request):

    student_id = request.session.get(
        "face_registration_student_id"
    )

    session_key = request.session.session_key

    if not student_id or not session_key:

        return JsonResponse({
            "message": "No active registration session."
        }, status=400)

    student = get_object_or_404(
        Student,
        pk=student_id
    )

    uploaded_file = request.FILES.get("frame")

    if not uploaded_file:

        return JsonResponse({
            "message": "No camera frame received."
        }, status=400)

    image_bytes = np.frombuffer(
        uploaded_file.read(),
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_bytes,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        return JsonResponse({
            "message": "Invalid camera frame."
        }, status=400)

    try:

        result = registration_service.process_frame(
            session_key,
            frame
        )

        if result["complete"]:

            # Save the completed facial dataset path.
            student.facial_data_path = result["dataset_path"]

            student.save(
                update_fields=["facial_data_path"]
            )

            # End the registration session.
            request.session.pop(
                "face_registration_student_id",
                None
            )

            # Automatically generate embeddings for this student.
            try:

                embedding_result = update_student_embeddings(
                    student.matric_number
                )

                # Refresh the recognition service immediately.
                total_students = recognition_service.reload_embeddings()

                result["embedding_ready"] = True

                result["embeddings_count"] = (
                    embedding_result["images_processed"]
                )

                result["total_students"] = total_students

                result["message"] = (
                    "Registration completed successfully. "
                    "Facial embeddings generated automatically."
                )

                print(
                    f"Automatic embedding generation successful: "
                    f"{student.matric_number} - "
                    f"{embedding_result['images_processed']} embeddings"
                )

            except Exception as error:

                print(
                    "AUTOMATIC EMBEDDING ERROR:",
                    str(error)
                )

                result["embedding_ready"] = False

                result["message"] = (
                    "Facial registration completed, but embedding "
                    "generation failed. Check the Django terminal."
                )

        return JsonResponse(result)

    except ValueError as error:

        return JsonResponse({
            "message": str(error)
        }, status=400)


@staff_member_required
@require_POST
def stop_face_registration(request):

    session_key = request.session.session_key

    if session_key:

        registration_service.stop_registration(
            session_key
        )

    request.session.pop(
        "face_registration_student_id",
        None
    )

    return JsonResponse({
        "message": "Registration stopped. Incomplete facial data removed."
    })


@staff_member_required
def register_face_page(request, student_id):

    student = get_object_or_404(
        Student,
        pk=student_id
    )

    return render(
        request,
        "results/register_face.html",
        {"student": student}
    )
