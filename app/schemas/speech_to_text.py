from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    status: int
    result: str
