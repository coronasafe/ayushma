import time

from django.conf import settings
from drf_spectacular.utils import extend_schema
from rest_framework import filters, status
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.exceptions import ValidationError
from rest_framework.mixins import (
    CreateModelMixin,
    DestroyModelMixin,
    ListModelMixin,
    RetrieveModelMixin,
)
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from ayushma.models import Chat, ChatFeedback, Project
from ayushma.permissions import IsTempTokenOrAuthenticated
from ayushma.serializers import (
    ChatDetailSerializer,
    ChatFeedbackSerializer,
    ChatSerializer,
    ConverseSerializer,
    SpeechToTextSerializer,
)
from ayushma.utils.converse import converse_api
from ayushma.utils.speech_to_text import speech_to_text
from utils.views.base import BaseModelViewSet
from utils.views.mixins import PartialUpdateModelMixin


class ChatViewSet(
    BaseModelViewSet,
    CreateModelMixin,
    PartialUpdateModelMixin,
    RetrieveModelMixin,
    ListModelMixin,
    DestroyModelMixin,
):
    queryset = Chat.objects.all()
    serializer_class = ChatSerializer
    filter_backends = (filters.SearchFilter,)
    search_fields = ("title",)
    serializer_action_classes = {
        "retrieve": ChatDetailSerializer,
        "list_all": ChatDetailSerializer,
        "converse": ConverseSerializer,
        "speech_to_text": SpeechToTextSerializer,
    }
    permission_classes = (IsTempTokenOrAuthenticated,)
    lookup_field = "external_id"

    def initialize_request(self, request, *args, **kwargs):
        self.action = self.action_map.get(request.method.lower())
        return super().initialize_request(request, *args, **kwargs)

    def get_parsers(self):
        if self.action == "converse":
            return [MultiPartParser()]
        return super().get_parsers()

    def get_queryset(self):
        user = self.request.user
        project_id = self.kwargs["project_external_id"]
        queryset = self.queryset.filter(project__external_id=project_id).order_by(
            "-created_at"
        )

        if user.is_superuser and (
            self.request.query_params.get("fetch") == "all" or self.action == "list_all"
        ):
            return queryset

        return queryset.filter(user=user).order_by("-created_at")

    def perform_create(self, serializer):
        project_id = self.kwargs["project_external_id"]
        project = Project.objects.get(external_id=project_id)

        if (
            not project.assistant_id
            and not self.request.headers.get("OpenAI-Key")
            and not project.open_ai_key
            and not (self.request.user.allow_key and settings.OPENAI_API_KEY)
        ):
            raise ValidationError(
                {"error": "OpenAI-Key header is required to create a chat"}
            )

        if project.archived:
            raise ValidationError(
                {"non_field_errors": "Project is archived. Cannot create chat."}
            )

        serializer.save(user=self.request.user, project=project)
        super().perform_create(serializer)

    @extend_schema(
        tags=("chats",),
    )
    @action(detail=False, methods=["get"])
    def list_all(self, *args, **kwarg):
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    @extend_schema(
        tags=("chats",),
    )
    @action(detail=True, methods=["post"])
    def speech_to_text(self, *args, **kwarg):
        serializer = self.get_serializer(data=self.request.data)
        serializer.is_valid()

        project_id = kwarg["project_external_id"]
        audio = serializer.validated_data["audio"]
        language = serializer.validated_data.get("language", "en")

        stats = {}
        try:
            stt_engine = Project.objects.get(external_id=project_id).stt_engine
        except Project.DoesNotExist:
            return Response(
                {"error": "Project not found"}, status=status.HTTP_400_BAD_REQUEST
            )
        try:
            stats["transcript_start_time"] = time.time()
            transcript = speech_to_text(stt_engine, audio, language + "-IN")
            stats["transcript_end_time"] = time.time()
            translated_text = transcript
        except Exception as e:
            print(f"Failed to transcribe speech with {stt_engine} engine: {e}")
            return Response(
                {
                    "error": "Something went wrong in getting transcription, please try again later"
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {"transcript": translated_text, "stats": stats}, status=status.HTTP_200_OK
        )

    @extend_schema(
        tags=("chats",),
    )
    @action(detail=True, methods=["post"])
    def converse(self, *args, **kwarg):
        chat: Chat = Chat.objects.get(external_id=kwarg["external_id"])
        try:
            response = converse_api(
                request=self.request,
                chat=chat,
                is_thread=bool(chat.project.assistant_id),
            )
            return response
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class ChatFeedbackViewSet(
    BaseModelViewSet, CreateModelMixin, RetrieveModelMixin, ListModelMixin
):
    queryset = ChatFeedback.objects.all()
    permission_classes = [IsAuthenticated]
    serializer_class = ChatFeedbackSerializer

    def get_queryset(self):
        user = self.request.user
        queryset = self.queryset

        if user.is_superuser or user.is_staff or user.is_reviewer:
            return queryset

        return queryset.filter(chat_message__chat__user=user)

    lookup_field = "external_id"
    filterset_fields = [
        "liked",
        "chat_message",
        "chat_message__chat",
        "chat_message__chat__project",
    ]
