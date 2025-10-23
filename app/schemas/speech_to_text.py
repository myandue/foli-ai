from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    status: int
    transcription: str
