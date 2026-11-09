# PROJECT DOCUMENTATION

## Project Title

**Development of a Lightweight AI-Based Facial Recognition System for Automated Examination Results Management**

---

## 1. Project Overview

This project involves the development of a lightweight AI-based facial recognition system for automated examination results management.

The system provides students with secure access to their examination results through facial identity verification. It combines computer vision, artificial intelligence and web application development.

Students identify themselves using their matriculation numbers and complete facial verification through a webcam. The system compares their live facial features with previously registered facial embeddings stored in the application's database.

Following successful identity verification, the system checks the student's school fee clearance status. Only students whose clearance has been confirmed by an administrator can access their examination results.

The web application is developed using Django, while the facial recognition component uses OpenCV, YuNet and SFace.

The system also provides administrative functionality for student registration, facial registration, examination result management, bulk result uploads and academic performance calculations.

The primary focus of the project is the facial recognition component, while the results management module provides the administrative and academic platform.

---

## 2. System Architecture and Components

The application consists of four major functional components:

1. Student Facial Registration.
2. Facial Embedding Extraction and Storage.
3. Facial Identity Verification.
4. Examination Results Management and Payment Clearance.

The system uses a modular architecture in which separate Python services handle facial registration, embedding extraction and recognition.

Django connects these services to the web interface and manages student information, academic records and administrative operations.

### Technologies and Tools

| Component | Technology |
|---|---|
| Programming language | Python |
| Web framework | Django |
| Face detection | OpenCV YuNet |
| Facial recognition | OpenCV SFace |
| Numerical processing | NumPy |
| Database | PostgreSQL (Neon) |
| Frontend | HTML, CSS, JavaScript |
| Excel processing | OpenPyXL |
| CSV processing | Python CSV utilities |
| Deployment target | Render |
| Database hosting | Neon |

---

## 3. Student Facial Registration

**Main file:** `registration_service.py`

**Web interface:** `register_face.html`

### Purpose

This component collects facial images of students and prepares their facial data for recognition.

### How It Works

1. An administrator selects a registered student from the administration interface.
2. The administrator starts the facial registration process.
3. The student's webcam is activated through the browser.
4. The registration service processes the camera frames.
5. YuNet detects the student's face.
6. The system evaluates facial position and image quality.
7. Suitable facial images are captured and saved in a folder associated with the student's matriculation number.
8. After registration is completed, the system automatically generates facial embeddings.

### Facial Capture Positions

The current registration process captures facial samples from three main positions:

- Looking directly at the camera.
- Turning the head to the left.
- Turning the head to the right.

Multiple samples are collected to provide different facial representations for subsequent recognition.

### Image Quality Checks

The registration process includes:

- Face detection.
- Minimum face-size requirements.
- Image sharpness checks.
- Brightness checks.
- Head-position checks.

### Technologies Used

- Python.
- OpenCV.
- YuNet.
- NumPy.
- JavaScript.
- Browser webcam API.

---

## 4. Facial Embedding Extraction and Storage

**Main file:** `embedding_service.py`

### Purpose

This component converts registered facial images into numerical representations known as facial embeddings.

These embeddings represent important facial characteristics and are used during