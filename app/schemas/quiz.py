from pydantic import BaseModel
from typing import List


class KeywordQuizRequest(BaseModel):
    keyword: str
    amount: int
    level: str


class DocsQuizRequest(BaseModel):
    docs: str
    amount: int
    level: str


class Answer(BaseModel):
    answer: str
    correct: bool


class Question(BaseModel):
    question: str
    answers: List[Answer]


class QuizResponse(BaseModel):
    questions: List[Question]
