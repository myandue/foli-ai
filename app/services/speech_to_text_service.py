import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
clova_stt_url = os.getenv("CLOVASPEECH_INVOKE_URL") + "/recognizer/upload"
secret_key = os.getenv("CLOVASPEECH_SECRET_KEY")


def speech_to_text(audio_file):
    params = {"language": "ko-KR", "completion": "sync"}
    headers = {"X-CLOVASPEECH-API-KEY": secret_key}
    files = {
        "media": (
            audio_file.filename,
            audio_file.file,
            audio_file.content_type,
        ),
        "params": (None, json.dumps(params), "application/json"),
        "format": "JSON",
    }

    response = requests.post(clova_stt_url, headers=headers, files=files)
    print(response.json())

    return {
        "status": response.status_code,
        "result": (
            response.json()["text"]
            if response.headers.get("Content-Type", "").startswith(
                "application/json"
            )
            else response.text
        ),
    }
