# Lightweight AI-Based Facial Recognition System for Automated Examination Results Management

An AI-powered examination results management system that uses facial recognition to verify student identities before granting access to academic results.

This project was developed as a final-year Computer Science project at Ladoke Akintola University of Technology (LAUTECH).

## Project Overview

The system combines computer vision, facial recognition and web-based academic results management to provide a more secure approach to accessing student examination results.

Students enter their matriculation number and undergo facial verification. Once their identity is verified, the system checks their fee-clearance status before granting access to their academic records.

## Key Features

- **Facial Registration:** Captures student facial images through a webcam.
- **Face Detection:** Uses OpenCV YuNet to detect faces.
- **Face Recognition:** Uses OpenCV SFace to generate and compare facial embeddings.
- **Identity Verification:** Compares a student's live facial data against registered facial records.
- **Results Management:** Allows administrators to create, update and delete student examination results.
- **Bulk Results Upload:** Supports CSV and Excel (.xlsx) result uploads.
- **Fee Clearance Verification:** Restricts results access until an administrator confirms fee clearance.
- **Academic Performance Calculation:** Automatically calculates grades, GPA and CGPA using course credit units.
- **Student Results Portal:** Displays academic records grouped by academic session and semester.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Django | Web application framework |
| OpenCV | Face detection and recognition |
| YuNet | Face detection model |
| SFace | Face recognition model |
| NumPy | Numerical operations and facial embeddings |
| SQLite | Development database |
| HTML/CSS | Web interface |
| JavaScript | Frontend interactions |
| OpenPyXL | Excel file processing |

## System Workflow

1. An administrator registers student information.
2. The student's face is captured and stored for recognition.
3. Facial embeddings are generated using the recognition model.
4. The student enters their matriculation number.
5. The system verifies their face using the webcam.
6. The system checks the student's fee-clearance status.
7. If clearance is confirmed, the student can access their examination results.

## Project Structure

```text
AI_Facial_Recognition_Results_System/
│
├── config/
├── results/
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
│
├── register_student.py
├── train_model.py
├── recognize_face.py
├── embedding_service.py
├── recognition_service.py
├── registration_service.py
│
├── manage.py
├── requirements.txt
├── PROJECT_DOCUMENTATION.md
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/olasunkanmifawaz17-crypto/AI_Facial_Recognition_Results_System
cd AI_Facial_Recognition_Results_System
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root using `.env.example` as a guide.

Generate a new Django secret key and configure it in the `.env` file.

### 5. Apply database migrations

```bash
python manage.py migrate
```

### 6. Create an administrator account

```bash
python manage.py createsuperuser
```

### 7. Start the development server

```bash
python manage.py runserver
```

Open the local development address:

`http://127.0.0.1:8000/`

## Academic Grading System

The application uses the following grade scale:

| Score | Grade | Grade Point |
|---|---|---:|
| 70–100 | A | 5 |
| 60–69 | B | 4 |
| 50–59 | C | 3 |
| 45–49 | D | 2 |
| 40–44 | E | 1 |
| 0–39 | F | 0 |

GPA and CGPA calculations take course credit units into account.

## Privacy and Security

- Student biometric datasets and facial embeddings are excluded from the public repository.
- Database files and environment secrets are excluded from version control.
- Student results access is protected by facial identity verification and fee-clearance checks.

## Current Limitations and Future Improvements

- Implementing stronger liveness detection to improve resistance to spoofing.
- Further testing with a larger and more diverse student dataset.
- Preparing the application for online deployment.
- Improving production security and database configuration.

## Documentation

For detailed technical information about the system, implementation, components and testing, see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).

## Project Status

The core facial recognition and examination results management workflows have been implemented and tested locally. Further improvements and deployment preparation are ongoing.

---

**Academic Project:** Final-Year Computer Science Project  
**Institution:** Ladoke Akintola University of Technology (LAUTECH)
