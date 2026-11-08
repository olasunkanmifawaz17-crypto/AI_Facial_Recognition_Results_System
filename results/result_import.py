import csv
import io
import re

from decimal import Decimal, InvalidOperation

from django.core.exceptions import ValidationError
from django.db import DatabaseError, transaction

from .models import Student, StudentResult


REQUIRED_COLUMNS = {
    'matric_number',
    'course_code',
    'course_title',
    'score',
    'credit_unit',
    'academic_session',
    'semester',
}


def validate_headers(header_row, location):

    headers = [
        str(value).strip().lower() if value is not None else ''
        for value in header_row
    ]

    if len(headers) != len(set(headers)):
        raise ValueError(
            f'{location}: Duplicate column headings were found.'
        )

    if set(headers) != REQUIRED_COLUMNS:

        missing = REQUIRED_COLUMNS - set(headers)
        extra = set(headers) - REQUIRED_COLUMNS

        message = []

        if missing:
            message.append(
                f'Missing columns: {", ".join(sorted(missing))}'
            )

        if extra:
            message.append(
                f'Unexpected columns: {", ".join(sorted(extra))}'
            )

        raise ValueError(
            f'{location}: {"; ".join(message)}'
        )

    return headers


def extract_sheet_rows(sheet, sheet_name):

    if not sheet:
        return []

    if all(
        not row or all(
            value is None or str(value).strip() == ''
            for value in row
        )
        for row in sheet
    ):
        return []

    location = f"Sheet '{sheet_name}'"

    headers = validate_headers(
        sheet[0],
        location
    )

    rows = []

    for row_number, values in enumerate(sheet[1:], start=2):

        if not values or all(
            value is None or str(value).strip() == ''
            for value in values
        ):
            continue

        values = list(values)

        row_location = (
            f"Sheet '{sheet_name}', row {row_number}"
        )

        if len(values) != len(headers):

            rows.append((
                row_location,
                None,
                'Incorrect number of cells.'
            ))

            continue

        rows.append((
            row_location,
            dict(zip(headers, values)),
            None
        ))

    return rows


def load_spreadsheet(uploaded_file):

    filename = uploaded_file.name.lower()

    all_rows = []

    if filename.endswith('.csv'):

        content = uploaded_file.read().decode('utf-8-sig')

        sheet = list(
            csv.reader(io.StringIO(content))
        )

        all_rows.extend(
            extract_sheet_rows(sheet, 'CSV')
        )

    elif filename.endswith('.xlsx'):

        from openpyxl import load_workbook

        workbook = load_workbook(
            uploaded_file,
            read_only=True,
            data_only=True
        )

        try:

            for worksheet in workbook.worksheets:

                sheet = list(
                    worksheet.iter_rows(values_only=True)
                )

                all_rows.extend(
                    extract_sheet_rows(
                        sheet,
                        worksheet.title
                    )
                )

        finally:
            workbook.close()

    else:

        raise ValueError(
            'Only CSV and XLSX files are supported.'
        )

    if not all_rows:

        raise ValueError(
            'The uploaded file contains no result records.'
        )

    return all_rows


def normalize_matric_number(value):

    if value is None or isinstance(value, bool):
        raise ValueError(
            'Matric number is missing or invalid.'
        )

    if isinstance(value, (int, float, Decimal)):

        number = Decimal(str(value))

        if not number.is_finite():
            raise ValueError(
                'Matric number must be numeric.'
            )

        if number != number.to_integral_value():
            raise ValueError(
                'Matric number must contain whole digits only.'
            )

        value = str(int(number))

    else:

        value = str(value).strip()

    if not re.fullmatch(r'[1-9][0-9]*', value):

        raise ValueError(
            'Matric number must contain digits only '
            'and cannot start with zero.'
        )

    return value


def normalize_course_code(value):

    value = (
        str(value).strip()
        if value is not None else ''
    )

    if not re.fullmatch(
        r'(?=.*[A-Za-z])(?=.*[0-9])[A-Za-z0-9]+',
        value
    ):

        raise ValueError(
            'Course code must contain letters and numbers only.'
        )

    return value.upper()


def normalize_course_title(value):

    if value is None or str(value).strip() == '':

        raise ValueError(
            'Course title cannot be empty.'
        )

    return str(value).strip().title()


def normalize_score(value):

    if value is None or isinstance(value, bool):

        raise ValueError(
            'Score must be numeric.'
        )

    try:

        score = Decimal(str(value).strip())

    except (InvalidOperation, ValueError):

        raise ValueError(
            'Score must be numeric.'
        )

    if not score.is_finite():

        raise ValueError(
            'Score must be a valid finite number.'
        )

    if not Decimal('0') <= score <= Decimal('100'):

        raise ValueError(
            'Score must be between 0 and 100.'
        )

    return score


def normalize_credit_unit(value):

    if value is None or isinstance(value, bool):

        raise ValueError(
            'Credit unit is required.'
        )

    try:

        credit_unit = Decimal(str(value).strip())

    except (InvalidOperation, ValueError):

        raise ValueError(
            'Credit unit must be a whole number.'
        )

    if not credit_unit.is_finite():

        raise ValueError(
            'Credit unit must be a valid number.'
        )

    if credit_unit != credit_unit.to_integral_value():

        raise ValueError(
            'Credit unit must be a whole number.'
        )

    if credit_unit < 1 or credit_unit > 32767:

        raise ValueError(
            'Credit unit must be between 1 and 32767.'
        )

    return int(credit_unit)


def normalize_academic_session(value):

    value = (
        str(value).strip()
        if value is not None else ''
    )

    match = re.fullmatch(
        r'(\d{4})\s*/\s*(\d{4})',
        value
    )

    if not match:

        raise ValueError(
            'Academic session must follow the format YYYY/YYYY.'
        )

    start_year = int(match.group(1))
    end_year = int(match.group(2))

    if end_year != start_year + 1:

        raise ValueError(
            'The ending year must be exactly one year '
            'after the starting year.'
        )

    return f'{start_year}/{end_year}'


def normalize_semester(value):

    value = (
        str(value).strip().casefold()
        if value is not None else ''
    )

    if value not in ('first', 'second'):

        raise ValueError(
            'Semester must be either First or Second.'
        )

    return value.capitalize()


def process_result_upload(uploaded_file, mode='update'):

    if mode not in ('update', 'replace'):

        raise ValueError(
            'Invalid upload mode.'
        )

    rows = load_spreadsheet(uploaded_file)

    summary = {
        'created': 0,
        'updated': 0,
        'rejected': 0,
        'errors': [],
    }

    prepared_results = []
    seen_records = set()

    students = {
        student.matric_number: student
        for student in Student.objects.all()
    }

    # Validate every record across every worksheet first.
    for location, row, row_error in rows:

        if row_error:

            summary['rejected'] += 1

            summary['errors'].append(
                f'{location}: {row_error}'
            )

            continue

        try:

            matric_number = normalize_matric_number(
                row['matric_number']
            )

            course_code = normalize_course_code(
                row['course_code']
            )

            course_title = normalize_course_title(
                row['course_title']
            )

            score = normalize_score(
                row['score']
            )

            credit_unit = normalize_credit_unit(
                row['credit_unit']
            )

            academic_session = normalize_academic_session(
                row['academic_session']
            )

            semester = normalize_semester(
                row['semester']
            )

            student = students.get(matric_number)

            if student is None:

                raise ValueError(
                    f'Student with matric number '
                    f'{matric_number} does not exist.'
                )

            record_key = (
                matric_number,
                course_code,
                academic_session,
                semester
            )

            if record_key in seen_records:

                raise ValueError(
                    'Duplicate result entry for this student, '
                    'course, academic session and semester.'
                )

            seen_records.add(record_key)

            prepared_results.append(
                StudentResult(
                    student=student,
                    course_code=course_code,
                    course_title=course_title,
                    score=score,
                    credit_unit=credit_unit,
                    academic_session=academic_session,
                    semester=semester
                )
            )

        except (
            ValueError,
            InvalidOperation,
            ValidationError
        ) as error:

            summary['rejected'] += 1

            summary['errors'].append(
                f'{location}: {error}'
            )

    # Never modify existing results if any record is invalid.
    if summary['rejected'] > 0:
        return summary

    if not prepared_results:

        raise ValueError(
            'No valid result records were found.'
        )

    # Apply the selected import mode atomically.
    try:

        with transaction.atomic():

            if mode == 'replace':

                StudentResult.objects.all().delete()

                StudentResult.objects.bulk_create(
                    prepared_results,
                    batch_size=500
                )

                summary['created'] = len(
                    prepared_results
                )

            else:

                for result_data in prepared_results:

                    result, created = (
                        StudentResult.objects.update_or_create(
                            student=result_data.student,
                            course_code=result_data.course_code,
                            academic_session=result_data.academic_session,
                            semester=result_data.semester,
                            defaults={
                                'course_title': result_data.course_title,
                                'score': result_data.score,
                                'credit_unit': result_data.credit_unit,
                            }
                        )
                    )

                    if created:
                        summary['created'] += 1
                    else:
                        summary['updated'] += 1

    except DatabaseError as error:

        summary['created'] = 0
        summary['updated'] = 0
        summary['rejected'] = len(prepared_results)

        summary['errors'].append(
            f'Database error. No changes were saved: {error}'
        )

    return summary