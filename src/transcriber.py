import ssl
import certifi
from pathlib import Path

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import whisper

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

def transcribe_audio(path: str) -> str:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / "transcription.txt"

    if cache_file.exists():
        print("       (using cached transcription)")
        return cache_file.read_text()

    model = whisper.load_model("small")
    result = model.transcribe(path, fp16=False)
    text = str(result["text"])

    cache_file.write_text(text)
    return text
