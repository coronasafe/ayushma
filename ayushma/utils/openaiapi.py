import json
from queue import Queue
from typing import List

import openai
import tiktoken
from anyio.from_thread import start_blocking_portal
from django.conf import settings
from langchain.schema import AIMessage, HumanMessage
from pinecone import QueryResponse

from ayushma.models import ChatMessage
from ayushma.models.enums import ChatMessageType
from ayushma.utils.langchain import LangChainHelper


def get_embedding(
    text: List[str],
    model: str = "text-embedding-ada-002",
    openai_api_key: str = settings.OPENAI_API_KEY,
) -> List[List[float]]:
    """
    Generates embeddings for the given list of texts using the OpenAI API.

    Args:
        text (List[str]): A list of strings to be embedded.
        model (str, optional): The name of the OpenAI model to use for embedding.
            Defaults to "text-embedding-ada-002".

    Returns:
        A list of embeddings generated by the OpenAI API for the input texts.

    Raises:
        OpenAIError: If there was an error communicating with the OpenAI API.

    Example usage:
        >>> get_embedding(["Hello, world!", "How are you?"])
        [[-0.123, 0.456, 0.789, ...], [0.123, -0.456, 0.789, ...]]

    """
    openai.api_key = openai_api_key
    res = openai.Embedding.create(input=text, model=model)
    return [record["embedding"] for record in res["data"]]


def get_sanitized_reference(pinecone_references: List[QueryResponse]) -> str:
    """
    Extracts the text from the Pinecone QueryResponse object and sanitizes it.

    Args:
        pinecone_reference (List[QueryResponse]): The similar documents retrieved from
            the Pinecone index.

    Returns:
        A string containing the text from the Pinecone QueryResponse object.

    Example usage:
        >>> get_sanitized_reference([QueryResponse(...), QueryResponse(...)])
        "Hello, world! How are you?,I am fine. Thank you."
    """
    sanitized_reference = ""

    for reference in pinecone_references:
        for match in reference.matches:
            sanitized_reference += str(match.metadata["text"]).replace("\n", " ") + ","

    return sanitized_reference


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def split_text(text):
    """Returns one string split into n equal length strings"""
    n = len(text)
    number_of_chars = 8192
    parts = []

    for i in range(0, n, number_of_chars):
        part = text[i : i + number_of_chars]
        parts.append(part)

    return parts


def create_json_response(input_text, chat_id, delta, message, stop):
    json_data = {
        "chat": str(chat_id),
        "input": input_text,
        "delta": delta,
        "message": message,
        "stop": stop,
    }
    return "data: " + json.dumps(json_data) + "\n\n"


def get_reference(text, openai_key, chat, top_k):
    num_tokens = num_tokens_from_string(text, "cl100k_base")
    embeddings: List[List[List[float]]] = []
    if num_tokens < 8192:
        try:
            embeddings.append(get_embedding(text=[text], openai_api_key=openai_key))
        except Exception as e:
            return Exception(
                e.__str__(),
            )
    else:
        parts = split_text(text)
        for part in parts:
            try:
                embeddings.append(get_embedding(text=[part], openai_api_key=openai_key))
            except Exception as e:
                raise Exception(
                    e.__str__(),
                )
    # find similar embeddings from pinecone index for each embedding
    pinecone_references: List[QueryResponse] = []

    for embedding in embeddings:
        try:
            similar: QueryResponse = settings.PINECONE_INDEX_INSTANCE.query(
                vector=embedding,
                top_k=top_k,
                namespace=str(chat.project.external_id),
                include_metadata=True,
            )
            pinecone_references.append(similar)
        except Exception as e:
            raise Exception(
                e.__str__(),
            )
    return get_sanitized_reference(pinecone_references=pinecone_references)


def converse(text, openai_key, chat, match_number):
    if not openai_key:
        raise Exception("OpenAI-Key header is required to create a chat or converse")
    openai.api_key = openai_key

    text = text.replace("\n", " ")
    nurse_query = ChatMessage.objects.create(message=text, chat=chat, messageType=1)

    reference = get_reference(text, openai_key, chat, match_number)

    token_queue = Queue()
    RESPONSE_END = object()
    lang_chain_helper = LangChainHelper(
        token_queue=token_queue,
        end=RESPONSE_END,
        openai_api_key=openai_key,
        prompt_template=chat.project.prompt,
    )

    # excluding the latest query since it is not a history
    previous_messages = (
        ChatMessage.objects.filter(chat=chat)
        .exclude(id=nurse_query.id)
        .order_by("created_at")
    )
    chat_history = []
    for message in previous_messages:
        if message.messageType == ChatMessageType.USER:
            chat_history.append(HumanMessage(content=f"Nurse: {message.message}"))
        elif message.messageType == ChatMessageType.AYUSHMA:
            chat_history.append(AIMessage(content=f"Ayushma: {message.message}"))

    with start_blocking_portal() as portal:
        portal.start_task_soon(
            lang_chain_helper.get_response,
            RESPONSE_END,
            token_queue,
            text,
            reference,
            chat_history,
        )
        chat_response = ""
        skip_token = len("Ayushma: ")
        try:
            while True:
                if token_queue.empty():
                    continue
                next_token = token_queue.get(True, timeout=10)
                if skip_token > 0:
                    skip_token -= 1
                    continue
                if next_token is RESPONSE_END:
                    ChatMessage.objects.create(
                        message=chat_response,
                        chat=chat,
                        messageType=ChatMessageType.AYUSHMA,
                    )

                    # language_code = chat_response[chat_response.rfind(":")
                    yield create_json_response(
                        text, chat.external_id, "", chat_response, True
                    )
                    break
                chat_response += next_token
                yield create_json_response(
                    text, chat.external_id, next_token, chat_response, False
                )
        except Exception as e:
            print(e)
            ChatMessage.objects.create(
                message=str(e),
                chat=chat,
                messageType=ChatMessageType.AYUSHMA,
            )
            yield create_json_response(text, chat.external_id, "", str(e), True)
