from django.apps import AppConfig


class AyushmaConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ayushma"

    def ready(self):
        # from .signals import   # noqa: F401
        pass
