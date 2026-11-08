from django import forms

from .models import StudentResult, Student
from .result_import import (
    normalize_matric_number,
    normalize_course_code,
    normalize_course_title,
    normalize_score,
    normalize_credit_unit,
    normalize_academic_session,
    normalize_semester,
)


class ResultUploadForm(forms.Form):

    IMPORT_MODES = [
        (
            'update',
            'Update existing results (add new records and overwrite matching records)'
        ),
        (
            'replace',
            'Replace entire results dataset (delete all existing results first)'
        ),
    ]

    file = forms.FileField(
        label='Select results file',
        help_text='Upload a CSV or XLSX file.'
    )

    import_mode = forms.ChoiceField(
        choices=IMPORT_MODES,
        initial='update',
        widget=forms.RadioSelect,
        label='Import mode'
    )

    def clean_file(self):

        uploaded_file = self.cleaned_data['file']

        filename = uploaded_file.name.lower()

        if not filename.endswith(('.csv', '.xlsx')):
            raise forms.ValidationError(
                'Only CSV and XLSX files are supported.'
            )

        if uploaded_file.size == 0:
            raise forms.ValidationError(
                'The uploaded file is empty.'
            )

        return uploaded_file


class StudentResultForm(forms.ModelForm):

    matric_number = forms.CharField(
        label='Student Matric Number',
        max_length=20
    )

    credit_unit = forms.IntegerField(
        label='Course Credit Unit',
        min_value=1,
        required=True
    )

    class Meta:
        model = StudentResult
        fields = [
            'matric_number',
            'course_code',
            'course_title',
            'score',
            'credit_unit',
            'academic_session',
            'semester',
        ]

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if self.instance and self.instance.pk:

            self.fields['matric_number'].initial = (
                self.instance.student.matric_number
            )

            self.fields['credit_unit'].initial = (
                self.instance.credit_unit
            )

    @staticmethod
    def normalize_value(normalizer, value):

        try:
            return normalizer(value)

        except ValueError as error:
            raise forms.ValidationError(str(error))

    def clean_matric_number(self):

        value = self.normalize_value(
            normalize_matric_number,
            self.cleaned_data['matric_number']
        )

        try:
            return Student.objects.get(matric_number=value)

        except Student.DoesNotExist:
            raise forms.ValidationError(
                f'Student with matric number {value} does not exist.'
            )

    def clean_course_code(self):

        return self.normalize_value(
            normalize_course_code,
            self.cleaned_data['course_code']
        )

    def clean_course_title(self):

        return self.normalize_value(
            normalize_course_title,
            self.cleaned_data['course_title']
        )

    def clean_score(self):

        return self.normalize_value(
            normalize_score,
            self.cleaned_data['score']
        )

    def clean_credit_unit(self):

        return self.normalize_value(
            normalize_credit_unit,
            self.cleaned_data['credit_unit']
        )

    def clean_academic_session(self):

        return self.normalize_value(
            normalize_academic_session,
            self.cleaned_data['academic_session']
        )

    def clean_semester(self):

        return self.normalize_value(
            normalize_semester,
            self.cleaned_data['semester']
        )

    def clean(self):

        cleaned_data = super().clean()

        student = cleaned_data.get('matric_number')
        course_code = cleaned_data.get('course_code')
        academic_session = cleaned_data.get('academic_session')
        semester = cleaned_data.get('semester')

        if all([
            student,
            course_code,
            academic_session,
            semester
        ]):

            existing = StudentResult.objects.filter(
                student=student,
                course_code=course_code,
                academic_session=academic_session,
                semester=semester
            )

            if self.instance.pk:
                existing = existing.exclude(pk=self.instance.pk)

            if existing.exists():

                raise forms.ValidationError(
                    'A result already exists for this student, course, '
                    'academic session and semester.'
                )

        return cleaned_data

    def save(self, commit=True):

        instance = super().save(commit=False)

        instance.student = self.cleaned_data['matric_number']

        if commit:
            instance.save()
            self.save_m2m()

        return instance