from django.db.models import IntegerChoices


class ChatMessageType(IntegerChoices):
    USER = 1
    SYSTEM = 2
    AYUSHMA = 3


class DocumentType(IntegerChoices):
    FILE = 1
    URL = 2
    TEXT = 3


class STTEngine(IntegerChoices):
    WHISPER = 1
    GOOGLE = 2
    SELF_HOSTED = 3

    @classmethod
    def get_id_from_name(cls, name):
        for member in cls:
            if member.name.lower() == name.lower():
                return member.value
        return None


class TTSEngine(IntegerChoices):
    OPENAI = (1, "openai")
    GOOGLE = (2, "google")


class FeedBackRating(IntegerChoices):
    HALLUCINATING = 1
    WRONG = 2
    UNSATISFACTORY = 3
    SATISFACTORY = 4
    GOOD = 5
    EXCELLENT = 6


class ModelType(IntegerChoices):
    GPT_3_5 = 1
    GPT_3_5_16K = 2
    GPT_4 = 3
    GPT_4_32K = 4
    GPT_4_VISUAL = 5
    GPT_4_TURBO = 6
    GPT_4_OMNI = 7
    GPT_4_OMNI_MINI = 8


class StatusChoices(IntegerChoices):
    RUNNING = 1
    COMPLETED = 2
    CANCELED = 3
    FAILED = 4
