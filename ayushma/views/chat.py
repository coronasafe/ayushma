import time

import openai
from django.conf import settings
from django.http import StreamingHttpResponse
from drf_spectacular.utils import extend_schema, extend_schema_view, inline_serializer
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.exceptions import ValidationError
from rest_framework.filters import OrderingFilter, SearchFilter
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from rest_framework.response import Response
from rest_framework.serializers import CharField, IntegerField

from ayushma.models import APIKey, Chat, ChatFeedback, ChatMessage, Project
from ayushma.serializers import (
    ChatDetailSerializer,
    ChatFeedbackSerializer,
    ChatSerializer,
    ConverseSerializer,
)
from ayushma.utils.converse import converse_api
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
        "list_all": ChatDetailSerializer,
        "converse": ConverseSerializer,
    }
    permission_classes = (IsAuthenticated,)
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
        queryset = self.queryset.filter(project__external_id=project_id)

        if user.is_superuser and self.action == "list_all":
            return queryset

        return queryset.filter(user=user)

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
    def converse(self, *args, **kwarg):
        chat: Chat = Chat.objects.get(external_id=kwarg["external_id"])
        try:
            response = converse_api(
                request=self.request,
                chat=chat,
            )
            return response
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        
class ChatFeedbackViewSet(BaseModelViewSet):
    queryset=ChatFeedback.objects.all()
    permission_classes = [IsAuthenticated, IsAdminUser]
    serializer_class = ChatFeedbackSerializer

    lookup_field = "external_id"
    filterset_fields = ['liked', 'chat_message', 'chat_message__chat', 'chat_message__chat__project']
    

