# PROJECT DOCUMENTATION

## Project Title

**Development of a Lightweight AI-Based Facial Recognition System for Automated Examination Results Management**

---

## 1. Project Overview

This project involves the development of a lightweight AI-based facial recognition system for automated examination results management.

The system is designed to provide secure access to students' examination results through facial identity verification. It uses computer vision and artificial intelligence techniques to detect and recognize registered students.

Each student is identified using their matriculation number and previously registered facial data. During verification, the system captures the student's live facial image and compares it with their stored facial embeddings to determine whether their identity can be confirmed.

The application also includes an examination results management platform developed with Django. Administrators can register students, manage academic records, upload examination results, and update students' school fee clearance status.

Students must successfully complete facial verification and have their fee clearance confirmed before they can access their examination results.

The primary focus of this project is the facial recognition component, while the Django-based results management module provides the administrative and academic platform through which the system operates.

---

## 2. System Components

The system consists of four major components:

1. Student Facial Registration
2. Facial Embedding Extraction
3. Identity Verification
4. Examination Results Management and Payment Clearance

The application combines standalone Python-based facial recognition scripts with a Django web application.

---

## 3. Student Facial Registration – `register_student.py`

### Purpose

This component is responsible for registering students and collecting facial images for future identity verification.

### How It Works

1. The administrator provides the student's matriculation number and other required details.
2. The system activates the webcam.
3. The student is guided through different head positions.
4. YuNet detects the student's face in the camera frames.
5. Facial images are captured when the required head position and image quality conditions are satisfied.
6. The captured images are stored in a folder named after the student's matriculation number.

### Facial Capture Positions

The current registration process captures facial images from three main positions:

- Looking directly at the camera.
- Turning the head to the left.
- Turning the head to the right.

Multiple samples are collected to provide different facial representations for subsequent recognition.

### Image Quality Checks

The registration process uses facial detection and image quality conditions to ensure that suitable facial samples are collected.

### Technologies Used

- Python
- OpenCV
- YuNet face detection model
- Webcam

### Output

A collection of registered facial images stored in the corresponding student's dataset folder.

---

## 4. Facial Embedding Extraction – `train_model.py`

### Purpose

This component processes registered facial images and converts them into numerical representations known as facial embeddings.

These embeddings are used during identity verification.

### How It Works

1. The system scans the facial dataset directory for registered student folders.
2. It loads the facial images associated with each student.
3. YuNet detects faces in the images.
4. The system checks image quality and rejects unsuitable samples.
5. Detected faces are aligned using the SFace recognition model.
6. SFace extracts numerical facial embeddings from the accepted images.
7. The embeddings are normalized and stored alongside their corresponding matriculation numbers.

### Image Quality Checks

The system performs checks including:

- Successful face detection.
- Minimum acceptable face size.
- Acceptable image brightness.
- Minimum image sharpness.
- Rejection of unsuitable facial samples.

### Technologies Used

- Python
- OpenCV
- YuNet
- SFace
- NumPy

### Output

A file named `face_embeddings.npy`, containing the extracted facial embeddings associated with registered students.

**Important:** SFace is a pretrained facial recognition model. This project uses it for facial feature extraction and comparison rather than training a new facial recognition model from scratch.

---

## 5. Identity Verification – `recognize_face.py`

### Purpose

This component verifies a student's identity before granting access to their examination results.

It is the primary biometric security component of the system.

### How It Works

1. The student enters their matriculation number.
2. The system retrieves the facial embeddings associated with the supplied matriculation number.
3. The webcam captures the student's live facial images.
4. YuNet detects the face in the camera frames.
5. The system performs facial image quality checks.
6. SFace extracts an embedding from the live facial image.
7. The live embedding is compared with the stored embeddings belonging to the supplied matriculation number.
8. Cosine similarity is used to measure the similarity between the live facial representation and the registered facial data.
9. The system evaluates multiple consecutive frames before making a final verification decision.

### Current Verification Configuration

The current experimental configuration includes:

- Similarity threshold: 0.55
- Number of frames evaluated: 15
- Minimum matching frames required: 12
- Minimum average similarity score: 0.55
- Top-K comparison: 5

The system uses multiple matching frames and similarity scores to reduce the possibility of accepting an incorrect identity.

These parameters are configurable and may be adjusted following further testing.

### Verification Outcome

**Successful Verification:**

The student's identity is confirmed, allowing the application to proceed to the results access stage.

**Failed Verification:**

The student's identity cannot be confirmed, and access to the examination results is denied.

### Technologies Used

- Python
- OpenCV
- YuNet
- SFace
- NumPy
- Cosine similarity

### Future Improvement: Liveness Detection

A lightweight liveness detection mechanism is planned to help distinguish a real person presenting themselves to the camera from photographs or replayed videos.

---

## 6. Examination Results Management and Payment Clearance Module

### Purpose

This component manages students' academic records and controls access to examination results based on successful identity verification and payment clearance.

It is implemented using the Django web framework.

### 6.1 Administrator Functions

The administrator can:

- Register and manage student records.
- View student information.
- Register students for facial recognition.
- Manage individual examination results.
- Add new course results manually.
- Edit existing examination results.
- Delete individual examination results.
- Upload examination results through CSV and Excel files.
- Update students' school fee clearance status.
- Manage academic records through Django Admin.

### 6.2 Results Upload Functionality

The system supports bulk examination results import through CSV and Excel files.

The expected file structure contains the following columns:

- `matric_number`
- `course_code`
- `course_title`
- `score`
- `academic_session`
- `semester`

The upload functionality allows administrators to manage examination results for multiple students.

The system also includes an upload mode designed to replace the existing examination results dataset with the newly uploaded dataset.

### 6.3 Academic Performance Calculations

The application automatically calculates students' academic performance using their examination scores and course credit units.

The calculations include:

- Letter grades.
- Grade points.
- Total credit units.
- Semester Grade Point Average (GPA).
- Cumulative Grade Point Average (CGPA).

The system applies credit-unit weighting when calculating GPA and CGPA.

The grading configuration is:

| Score | Grade | Grade Point |
|---|---|---|
| 70–100 | A | 5 |
| 60–69 | B | 4 |
| 50–59 | C | 3 |
| 45–49 | D | 2 |
| 40–44 | E | 1 |
| 0–39 | F | 0 |

The calculations are updated when examination results are added, edited or deleted.

### 6.4 Student Functions

Students can:

1. Enter their matriculation number.
2. Complete facial identity verification.
3. Have their payment clearance status checked.
4. Access their examination results if all requirements are satisfied.

### 6.5 Payment Clearance

The payment clearance functionality is managed through an administrator-controlled Boolean status field.

The available states are:

- Cleared.
- Not Cleared.

Students whose payment clearance has not been confirmed are denied access to their examination results.

The system does not require a real online payment gateway. Administrators manually update the student's clearance status.

### 6.6 Student Results Interface

After successful facial verification and payment clearance, students are redirected to their academic results page.

The results interface displays:

- Student's full name.
- Matriculation number.
- Department.
- Academic sessions.
- Semester results.
- Course codes and titles.
- Examination scores.
- Letter grades.
- Grade points.
- Semester GPA.
- Cumulative GPA (CGPA).

The results are organised according to academic session and semester.

The interface also supports printing the student's academic results.

### Technologies Used

- Python
- Django
- HTML
- CSS
- JavaScript
- SQLite
- Pandas
- OpenPyXL

---

## 7. Authentication and Authorization

The system uses Django's built-in authentication and authorization functionality to manage administrative access.

This functionality provides:

- Administrator login.
- User account management.
- Group management.
- Permission management.
- Superuser privileges.

It allows the application to support different administrative accounts and controlled access to administrative functions.

Further configuration of role-based administrative permissions can be introduced as the application moves towards deployment.

Student facial verification and Django administrator authentication serve separate purposes.

---

## 8. Overall System Workflow

The complete system follows these stages:

### Stage 1 – Student Registration

- Administrator creates a student record.
- Student facial images are captured.
- Facial images are stored in the student's dataset folder.

### Stage 2 – Facial Embedding Extraction

- Registered facial images are processed.
- YuNet detects faces.
- SFace extracts facial embeddings.
- The embeddings are normalized and stored.

### Stage 3 – Student Identity Verification

- Student enters their matriculation number.
- The webcam captures the student's live face.
- YuNet detects the face.
- SFace extracts a live facial embedding.
- The live embedding is compared with the registered facial embeddings.
- Multiple frames are evaluated.
- The student's identity is accepted or rejected.

### Stage 4 – Payment Clearance Verification

- The system retrieves the student's payment clearance status.
- If the student is not cleared, access to examination results is denied.
- If cleared, the application proceeds to the results display stage.

### Stage 5 – Examination Results Access

- The student's academic records are retrieved.
- Results are organised by academic session and semester.
- GPA and CGPA are displayed.
- The student can view and print their academic results.

---

## 9. Current Development Progress

| Component | Development Status |
|---|---|
| Student facial registration | Implemented |
| Facial embedding extraction | Implemented |
| Identity verification | Implemented and tested |
| Multi-frame facial comparison | Implemented |
| Liveness detection | Pending |
| Django web application | Implemented |
| Student record management | Implemented |
| Administrator authentication | Implemented |
| Individual result creation | Implemented and tested |
| Individual result editing | Implemented and tested |
| Individual result deletion | Implemented and tested |
| CSV/Excel results upload | Implemented |
| Dataset replacement upload mode | Implemented; further testing pending |
| Automatic grade calculation | Implemented and tested |
| Semester GPA calculation | Implemented and tested |
| Cumulative GPA calculation | Implemented and tested |
| Credit-unit weighting | Implemented |
| Payment clearance integration | Implemented and tested |
| Facial verification and results integration | Implemented and tested |
| Student academic results interface | Implemented |
| Internet deployment | Pending |
| Final security and deployment testing | Pending |

---

## 10. Future Improvements

The following improvements may be considered:

1. Implement lightweight liveness detection to improve biometric security.
2. Improve facial recognition reliability under different lighting conditions and camera angles.
3. Configure administrative groups and permissions for different staff responsibilities.
4. Improve the security of stored facial data and application credentials.
5. Deploy the application to a suitable web hosting environment.
6. Perform additional testing under real-world usage conditions.

---

## 11. Summary

The project combines artificial intelligence, computer vision and web application development to provide a facial recognition-based examination results management system.

The biometric component uses YuNet for face detection and SFace for facial feature extraction and comparison.

The web application, developed using Django, manages student records, examination results and payment clearance.

Students must successfully complete facial identity verification and satisfy the payment clearance requirement before accessing their academic records.

The application has progressed beyond the initial development stage, with the major biometric and results management components implemented and integrated.

Further work will focus on liveness detection, additional security testing, repository preparation and internet deployment.