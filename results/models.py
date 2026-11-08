from django.db import models
from django.core.validators import (
    MinValueValidator,
    MaxValueValidator
)


class Student(models.Model):
    matric_number = models.CharField(max_length=20, unique=True)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    department = models.CharField(max_length=100)
    facial_data_path = models.CharField(max_length=255, blank=True)

    # Admin-managed payment clearance
    fee_cleared = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.matric_number} - {self.first_name} {self.last_name}"


class StudentResult(models.Model):

    SEMESTER_CHOICES = [
        ('First', 'First Semester'),
        ('Second', 'Second Semester'),
    ]

    student = models.ForeignKey(
        Student,
        on_delete=models.CASCADE,
        related_name='results'
    )

    course_code = models.CharField(max_length=20)
    course_title = models.CharField(max_length=200)

    score = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        validators=[
            MinValueValidator(0),
            MaxValueValidator(100)
        ]
    )

    # Nullable temporarily for results imported before this feature.
    credit_unit = models.PositiveSmallIntegerField(
        null=True,
        blank=True,
        validators=[
            MinValueValidator(1)
        ]
    )

    academic_session = models.CharField(max_length=20)

    semester = models.CharField(
        max_length=20,
        choices=SEMESTER_CHOICES
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=[
                    'student',
                    'course_code',
                    'academic_session',
                    'semester'
                ],
                name='unique_student_course_result'
            )
        ]

        ordering = [
            'student__matric_number',
            'course_code'
        ]

    def __str__(self):
        return (
            f"{self.student.matric_number} - "
            f"{self.course_code} - {self.score}"
        )