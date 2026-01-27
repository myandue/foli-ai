from pydantic import BaseModel


class SummaryRequest(BaseModel):
    text: str


class SummaryResponse(BaseModel):
    summary: str


class QnARequest(BaseModel):
    text: str
    history: list[str]
    question: str


class QnAResponse(BaseModel):
    answer: str
