from fastapi import APIRouter, HTTPException, UploadFile, File

from app.schemas.speech_to_text import TranscriptionResponse
from app.services.speech_to_text_service import speech_to_text

router = APIRouter(prefix="/stt", tags=["stt"])


@router.post("/transcription", response_model=TranscriptionResponse)
async def transcription(audio_file: UploadFile = File(...)):
    try:
        transcription_data = speech_to_text(audio_file)
        return TranscriptionResponse(**transcription_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
