import re

from google.cloud import texttospeech
from google.cloud import translate_v2 as translate
from rest_framework.exceptions import APIException


def translate_text(target, text):
    try:
        translate_client = translate.Client()

        result = translate_client.translate(text, target_language=target)
        return result["translatedText"]
    except Exception as e:
        print(e)
        raise APIException("Translation failed")


language_code_voice_map = {
    "bn-IN": "bn-IN-Wavenet-A",
    "en-US": "en-US-Neural2-C",
    "gu-IN": "gu-IN-Wavenet-A",
    "hi-IN": "hi-IN-Neural2-D",
    "kn-IN": "kn-IN-Wavenet-A",
    "ml-IN": "ml-IN-Wavenet-C",
    "mr-IN": "mr-IN-Wavenet-C",
    "pa-IN": "pa-IN-Wavenet-C",
    "ta-IN": "ta-IN-Wavenet-C",
    "te-IN": "te-IN-Standard-A",
}


def sanitize_text(text):
    sanitized_text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)  # Remove bold
    sanitized_text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)  # Remove italic

    return sanitized_text


def text_to_speech(text, language_code):
    try:
        # in en-IN neural voice is not available
        if language_code == "en-IN":
            language_code = "en-US"

        client = texttospeech.TextToSpeechClient()

        text = sanitize_text(text)
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code, name=language_code_voice_map[language_code]
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        return response.audio_content
    except Exception as e:
        print(e)
        return None
