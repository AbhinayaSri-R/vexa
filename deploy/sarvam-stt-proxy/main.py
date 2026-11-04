"""OpenAI-compatible /v1/audio/transcriptions facade over Sarvam AI speech-to-text.

Vexa's bot POSTs multipart (file, model, response_format, language, ...) with
`Authorization: Bearer <TRANSCRIPTION_SERVICE_TOKEN>` and expects a Whisper-style
verbose_json reply. Sarvam wants `api-subscription-key` and answers with its own shape.
This service translates both directions.

Env:
  SARVAM_API_KEY   key used when the caller's Bearer token is not itself a Sarvam key (optional)
  SARVAM_MODEL     default saaras:v3 (used unless the request's `model` starts with "saaras")
  SARVAM_URL       default https://api.sarvam.ai/speech-to-text
  PROXY_TOKEN      if set, callers must send exactly this Bearer token (recommended)
"""
import hmac
import os
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

SARVAM_URL = os.getenv("SARVAM_URL", "https://api.sarvam.ai/speech-to-text")
DEFAULT_MODEL = os.getenv("SARVAM_MODEL", "saaras:v3")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
PROXY_TOKEN = os.getenv("PROXY_TOKEN", "")

# Whisper-style ISO-639-1 -> Sarvam BCP-47 (the codes Sarvam documents).
LANG = {
    "hi": "hi-IN", "bn": "bn-IN", "kn": "kn-IN", "ml": "ml-IN", "mr": "mr-IN",
    "or": "od-IN", "od": "od-IN", "pa": "pa-IN", "ta": "ta-IN", "te": "te-IN",
    "en": "en-IN", "gu": "gu-IN", "as": "as-IN", "ur": "ur-IN", "ne": "ne-IN",
    "sa": "sa-IN", "sd": "sd-IN", "ks": "ks-IN",
}
SARVAM_CODES = set(LANG.values()) | {"unknown", "kok-IN", "sat-IN", "mni-IN", "brx-IN", "mai-IN", "doi-IN"}

app = FastAPI(title="sarvam-stt-proxy")


def _sarvam_lang(lang: Optional[str]) -> str:
    if not lang:
        return "unknown"
    if lang in SARVAM_CODES:
        return lang
    return LANG.get(lang.split("-")[0].lower(), "unknown")


def _check_caller(authorization: Optional[str]) -> str:
    bearer = (authorization or "").removeprefix("Bearer ").strip()
    if PROXY_TOKEN and not hmac.compare_digest(bearer, PROXY_TOKEN):
        raise HTTPException(401, "invalid token")
    key = SARVAM_API_KEY or (bearer if not PROXY_TOKEN else "")
    if not key:
        raise HTTPException(500, "no Sarvam key: set SARVAM_API_KEY")
    return key


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(""),
    language: Optional[str] = Form(None),
    response_format: str = Form("verbose_json"),
    authorization: Optional[str] = Header(None),
):
    key = _check_caller(authorization)
    audio = await file.read()
    sarvam_model = model if model.startswith("saaras") else DEFAULT_MODEL

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            SARVAM_URL,
            headers={"api-subscription-key": key},
            files={"file": (file.filename or "audio.wav", audio, file.content_type or "audio/wav")},
            data={
                "model": sarvam_model,
                "language_code": _sarvam_lang(language),
                "with_timestamps": "true",
            },
        )

    if r.status_code != 200:
        # Pass status + Retry-After through so the bot's retry/back-off classifies it correctly.
        headers = {k: v for k, v in r.headers.items() if k.lower() == "retry-after"}
        return JSONResponse({"error": r.text[:500]}, status_code=r.status_code, headers=headers)

    d = r.json()
    text = (d.get("transcript") or "").strip()
    ts = d.get("timestamps") or {}
    words = ts.get("words") or []
    starts = ts.get("start_time_seconds") or []
    ends = ts.get("end_time_seconds") or []

    segments = [
        {"start": s, "end": e, "text": w}
        for w, s, e in zip(words, starts, ends)
        if (w or "").strip()
    ]
    duration = max(ends) if ends else 0.0
    if not segments and text:
        segments = [{"start": 0.0, "end": duration, "text": text}]

    lang_code = d.get("language_code") or ""
    print(
        f"[sarvam] in_bytes={len(audio)} asked_lang={language} -> lang={lang_code} "
        f"text_len={len(text)} has_ts={bool(words)} segs={len(segments)} dur={duration} "
        f"text={text[:60]!r}",
        flush=True,
    )
    out = {
        "text": text,
        "language": lang_code.split("-")[0] if lang_code else (language or "unknown"),
        "language_probability": d.get("language_probability") or 0,
        "duration": duration,
        "segments": segments,
    }
    if response_format == "json":
        return {"text": text}
    return out
