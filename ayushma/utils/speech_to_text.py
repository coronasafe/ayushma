import os

import requests
from django.conf import settings
from google.cloud import speech
from openai import OpenAI

from ayushma.models.enums import STTEngine


class WhisperEngine:
    def __init__(self, api_key, language_code):
        self.api_key = api_key
        self.language_code = language_code

    def recognize(self, audio):
        try:
            client = OpenAI(api_key=self.api_key)
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                # https://github.com/openai/openai-python/tree/main#file-uploads
                file=(audio.name, audio.read()),
                language=self.language_code.replace("-IN", ""),
                # api_version="2020-11-07",
            )
            return transcription.text
        except Exception as e:
            raise ValueError(f"Failed to recognize speech with whisper engine: {e}")


class GoogleEngine:
    def __init__(self, api_key, language_code):
        self.api_key = api_key
        self.language_code = language_code

    def recognize(self, audio):
        try:
            client = speech.SpeechClient()
            audio_content = audio.file.read()
            audio_data = speech.RecognitionAudio(content=audio_content)

            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
                language_code=self.language_code,
            )

            response = client.recognize(config=config, audio=audio_data)
            if not response.results:
                return ""
            return response.results[0].alternatives[0].transcript
        except Exception as e:
            raise ValueError(f"Failed to recognize speech with google engine: {e}")


class SelfHostedEngine:
    def __init__(self, api_key, language_code):
        self.language_code = language_code

    def recognize(self, audio):
        try:
            response = requests.post(
                settings.SELF_HOSTED_ENDPOINT,
                files={"audio": audio},
                data={
                    # change this model to get faster results see: https://github.com/coronasafe/care-whisper
                    "model": "small",
                    "language": self.language_code.replace("-IN", ""),
                },
            )

            if not response.ok:
                print("Failed to recognize speech with self hosted engine")
                return ""
            response = response.json()
            return response["data"]["transcription"].strip()
        except Exception as e:
            raise ValueError(f"Failed to recognize speech with self hosted engine: {e}")


engines = {
    "whisper": WhisperEngine,
    "google": GoogleEngine,
    "self_hosted": SelfHostedEngine,
    # Add new engines here
}


def speech_to_text(engine_id, audio, language_code):
    api_key = os.environ.get("STT_API_KEY", "")
    engine_name = STTEngine(engine_id).name.lower()
    engine_class = engines.get(engine_name)

    if not engine_class:
        raise ValueError(f"Invalid STT engine ID: {engine_id}")

    try:
        engine = engine_class(api_key, language_code)
        recognized_text = engine.recognize(audio)
        if not recognized_text:
            raise ValueError("Speech-to-text: Failed to recognize speech")
        return recognized_text
    except Exception as e:
        raise ValueError(f"Speech-to-text failed with {engine_name} engine: {e}")
