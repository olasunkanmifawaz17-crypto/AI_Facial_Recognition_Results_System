from .models import Student, StudentResult


def admin_dashboard_stats(request):

    if not request.user.is_authenticated or not request.user.is_staff:
        return {}

    return {
        'dashboard_stats': {
            'total_students': Student.objects.count(),

            'total_results': StudentResult.objects.count(),

            'cleared_students': Student.objects.filter(
                fee_cleared=True
            ).count(),

            'uncleared_students': Student.objects.filter(
                fee_cleared=False
            ).count(),
        }
    }