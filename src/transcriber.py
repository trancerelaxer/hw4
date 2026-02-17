import ssl
import certifi
import os
import platform
from pathlib import Path

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import torch
import whisper

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def _get_device() -> str:
    # Optional override: WHISPER_DEVICE=mps|cpu|cuda
    env_device = os.getenv("WHISPER_DEVICE")
    if env_device:
        return env_device

    # On macOS, actively probe MPS because some setups report availability inconsistently.
    if platform.system() == "Darwin" and torch.backends.mps.is_built():
        try:
            _ = torch.zeros(1, device="mps")
            return "mps"
        except Exception:
            pass

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def transcribe_audio(path: str) -> str:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / "transcription.txt"

    if cache_file.exists():
        print("       (using cached transcription)")
        return cache_file.read_text()

    device = _get_device()
    print(f"       (whisper device: {device})")
    model = whisper.load_model("small", device=device)
    result = model.transcribe(path, fp16=False)
    text = str(result["text"])

    cache_file.write_text(text)
    return text
