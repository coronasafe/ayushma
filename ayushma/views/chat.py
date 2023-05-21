import openai
from django.conf import settings
from django.http import StreamingHttpResponse
from drf_spectacular.utils import extend_schema, extend_schema_view, inline_serializer
from rest_framework import permissions, status
from rest_framework.decorators import action
from rest_framework.exceptions import ValidationError
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.serializers import CharField, IntegerField

from ayushma.models import Chat, ChatMessage, Project
from ayushma.serializers import ChatDetailSerializer, ChatSerializer
from ayushma.utils.language_helpers import translate_text
from ayushma.utils.openaiapi import converse
from utils.views.base import BaseModelViewSet


@extend_schema_view(
    destroy=extend_schema(exclude=True),
    partial_update=extend_schema(exclude=False),
    create=extend_schema(exclude=False),
    retrieve=extend_schema(
        description="Get Chats",
    ),
)
class ChatViewSet(BaseModelViewSet):
    queryset = Chat.objects.all()
    serializer_class = ChatSerializer
    serializer_action_classes = {
        "retrieve": ChatDetailSerializer,
    }
    permission_classes = (permissions.IsAuthenticated,)
    lookup_field = "external_id"

    def initialize_request(self, request, *args, **kwargs):
        self.action = self.action_map.get(request.method.lower())
        return super().initialize_request(request, *args, **kwargs)

    def get_queryset(self):
        user = self.request.user
        project_id = self.kwargs["project_external_id"]
        queryset = self.queryset.filter(project__external_id=project_id)

        if user.is_superuser:
            return queryset

        return queryset.filter(user=user)

    def get_parsers(self):
        if self.action == "audio_converse":
            return [MultiPartParser()]
        return super().get_parsers()

    def perform_create(self, serializer):
        if (
            not self.request.headers.get("OpenAI-Key")
            and not self.request.user.allow_key
        ):
            raise ValidationError(
                {"error": "OpenAI-Key header is required to create a chat"}
            )

        project_id = self.kwargs["project_external_id"]
        project = Project.objects.get(external_id=project_id)

        serializer.save(user=self.request.user, project=project)
        super().perform_create(serializer)

    @extend_schema(
        tags=("chats",),
        request=inline_serializer(
            name="ConverseRequest",
            fields={
                "audio": CharField(),
                "match_number": IntegerField(default=10),
            },
        ),
        responses={status.HTTP_200_OK: None},
    )
    @action(detail=True, methods=["post"])
    def audio_converse(self, *args, **kwarg):
        if not self.request.data.get("audio"):
            return Response(
                {"error": "Please provide audio to generate embedding"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        audio = self.request.data.get("audio")

        chat: Chat = Chat.objects.get(external_id=kwarg["external_id"])

        openai_key = self.request.headers.get("OpenAI-Key") or (
            self.request.user.allow_key and settings.OPENAI_API_KEY
        )

        match_number = self.request.data.get("match_number") or 100

        if not openai_key:
            return Response(
                {"error": "OpenAI-Key header is required to create a chat"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            transcript = openai.Audio.translate(
                "whisper-1", file=audio, api_key=openai_key
            )
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        trasnlated_text = transcript.text
        if chat.language != "en":
            trasnlated_text = translate_text(chat.language + "-IN", transcript.text)

        if not ChatMessage.objects.filter(chat=chat).exists():
            chat.title = trasnlated_text[0:50]
            chat.save()

        response = StreamingHttpResponse(content_type="text/event-stream")
        try:
            response.streaming_content = converse(
                english_text=transcript.text,
                local_translated_text=trasnlated_text,
                openai_key=openai_key,
                chat=chat,
                match_number=match_number,
                user_language=chat.language + "-IN",
            )
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        return response

    @extend_schema(
        tags=("chats",),
        request=inline_serializer(
            name="ConverseRequest",
            fields={
                "text": CharField(),
                "match_number": IntegerField(default=10),
            },
        ),
        responses={status.HTTP_200_OK: None},
    )
    @action(detail=True, methods=["post"])
    def converse(self, *args, **kwarg):
        if not self.request.data.get("text"):
            return Response(
                {"error": "Please provide text to generate embedding"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        text = self.request.data.get("text")
        chat: Chat = Chat.objects.get(external_id=kwarg["external_id"])
        openai_key = self.request.headers.get("OpenAI-Key") or (
            self.request.user.allow_key and settings.OPENAI_API_KEY
        )

        if not openai_key:
            return Response(
                {"error": "OpenAI-Key header is required to create a chat"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        match_number = self.request.data.get("match_number") or 100
        response = StreamingHttpResponse(content_type="text/event-stream")

        english_text = text
        if chat.language != "en":
            english_text = translate_text("en-IN", text)

        try:
            response.streaming_content = converse(
                english_text=english_text,
                local_translated_text=text,
                openai_key=openai_key,
                chat=chat,
                match_number=match_number,
                user_language=chat.language + "-IN",
            )

        except Exception as e:
            # delete chat object if first conversation
            previous_chats = ChatMessage.objects.filter(chat=chat.id)
            if len(previous_chats) == 1:
                chat.delete()
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        return response
