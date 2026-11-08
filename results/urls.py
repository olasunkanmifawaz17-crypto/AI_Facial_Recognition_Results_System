from django.urls import path
from . import views
from . import home_views


urlpatterns = [

    # Public homepage
    path('', home_views.home, name='home'),

    # Student facial authentication
    path(
        'camera/',
        views.camera_test,
        name='camera_test'
    ),

    path(
        'camera/frame/',
        views.receive_frame,
        name='receive_frame'
    ),

    # Student examination results
    path(
        'results/',
        views.student_results,
        name='student_results'
    ),

    # Admin browser-based facial registration
    path(
        'registration/<int:student_id>/',
        views.register_face_page,
        name='register_face_page'
    ),

    path(
        'registration/start/',
        views.start_face_registration,
        name='start_face_registration'
    ),

    path(
        'registration/frame/',
        views.receive_registration_frame,
        name='receive_registration_frame'
    ),

    path(
        'registration/stop/',
        views.stop_face_registration,
        name='stop_face_registration'
    ),
]