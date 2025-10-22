from fastapi import APIRouter

from app.api.routes import documents, quiz, speech_to_text

api_router = APIRouter()

# api_router.include_router(documents.router)
api_router.include_router(quiz.router)
api_router.include_router(speech_to_text.router)
