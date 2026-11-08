from django.contrib import admin, messages
from django.urls import path, reverse
from django.shortcuts import redirect
from django.template.response import TemplateResponse
from django.http import HttpResponseNotAllowed

from .models import Student, StudentResult
from .forms import ResultUploadForm, StudentResultForm
from .result_import import process_result_upload

from embedding_service import delete_student_facial_data


@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):

    change_form_template = (
        'admin/results/student/change_form.html'
    )

    list_display = (
        'matric_number',
        'first_name',
        'last_name',
        'department',
        'fee_cleared',
    )

    search_fields = (
        'matric_number',
        'first_name',
        'last_name'
    )

    list_filter = (
        'department',
        'fee_cleared',
    )

    list_editable = ('fee_cleared',)

    def get_urls(self):

        urls = super().get_urls()

        custom_urls = [
            path(
                '<path:object_id>/register-face/',
                self.admin_site.admin_view(self.register_face),
                name='results_student_register_face'
            ),
        ]

        return custom_urls + urls

    def register_face(self, request, object_id):

        if request.method != 'POST':
            return HttpResponseNotAllowed(['POST'])

        student = self.get_object(request, object_id)

        if student is None:

            messages.error(
                request,
                'Student record not found.'
            )

            return redirect(
                'admin:results_student_changelist'
            )

        return redirect(
            'register_face_page',
            student_id=student.pk
        )

    def cleanup_student_facial_data(self, request, matric_numbers):

        for matric_number in matric_numbers:

            try:

                result = delete_student_facial_data(matric_number)

                from .views import recognition_service

                recognition_service.reload_embeddings()

                messages.success(
                    request,
                    (
                        f"Facial data cleanup completed for "
                        f"{matric_number}."
                    )
                )

                if not result["embedding_removed"] and not result["folder_removed"]:

                    messages.info(
                        request,
                        "No remaining facial data was found for this student."
                    )

            except Exception as error:

                messages.warning(
                    request,
                    (
                        f"Student {matric_number} was deleted, "
                        f"but facial data cleanup encountered an error: "
                        f"{error}"
                    )
                )

    def delete_model(self, request, obj):

        matric_number = obj.matric_number

        super().delete_model(request, obj)

        self.cleanup_student_facial_data(
            request,
            [matric_number]
        )

    def delete_queryset(self, request, queryset):

        matric_numbers = list(
            queryset.values_list(
                'matric_number',
                flat=True
            )
        )

        super().delete_queryset(request, queryset)

        self.cleanup_student_facial_data(
            request,
            matric_numbers
        )


@admin.register(StudentResult)
class StudentResultAdmin(admin.ModelAdmin):

    form = StudentResultForm

    list_display = (
        'student',
        'course_code',
        'course_title',
        'score',
        'credit_unit',
        'academic_session',
        'semester'
    )

    search_fields = (
        'student__matric_number',
        'student__first_name',
        'student__last_name',
        'course_code',
        'course_title'
    )

    list_filter = (
        'academic_session',
        'semester',
        'course_code'
    )

    list_editable = (
        'score',
        'credit_unit'
    )

    ordering = ('student__matric_number', 'course_code')

    change_list_template = (
        'admin/results/studentresult/change_list.html'
    )

    def get_urls(self):

        urls = super().get_urls()

        custom_urls = [
            path(
                'upload-results/',
                self.admin_site.admin_view(self.upload_results),
                name='results_studentresult_upload'
            ),
        ]

        return custom_urls + urls

    def upload_results(self, request):

        if request.method == 'POST':

            form = ResultUploadForm(request.POST, request.FILES)

            if form.is_valid():

                try:

                    mode = form.cleaned_data['import_mode']

                    summary = process_result_upload(
                        form.cleaned_data['file'],
                        mode=mode
                    )

                    if summary['rejected'] > 0:

                        messages.error(
                            request,
                            (
                                f"Upload rejected: "
                                f"{summary['rejected']} records rejected. "
                                "No existing results were changed."
                            )
                        )

                    else:

                        if mode == 'replace':

                            messages.success(
                                request,
                                (
                                    f"Dataset replaced successfully: "
                                    f"{summary['created']} results imported."
                                )
                            )

                        else:

                            messages.success(
                                request,
                                (
                                    f"Results updated successfully: "
                                    f"{summary['created']} created, "
                                    f"{summary['updated']} updated."
                                )
                            )

                    for error in summary['errors']:
                        messages.error(request, error)

                    return redirect(
                        reverse('admin:results_studentresult_changelist')
                    )

                except (ValueError, UnicodeDecodeError) as error:

                    form.add_error('file', str(error))

        else:
            form = ResultUploadForm()

        context = {
            **self.admin_site.each_context(request),
            'opts': self.model._meta,
            'title': 'Upload Student Results',
            'form': form,
        }

        return TemplateResponse(
            request,
            'admin/results/studentresult/upload_results.html',
            context
        )