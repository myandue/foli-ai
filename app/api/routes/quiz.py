from fastapi import APIRouter, HTTPException


from app.schemas.quiz import (
    KeywordQuizRequest,
    DocsQuizRequest,
    QuizResponse,
)
from app.services.quiz_service import (
    generate_quiz_by_keyword,
    generate_quiz_by_docs,
)

router = APIRouter(prefix="/quiz", tags=["quiz"])


@router.post("/keyword", response_model=QuizResponse)
async def generate_quiz_by_keyword(request: KeywordQuizRequest):
    try:
        quiz_data = generate_quiz_by_keyword(request)
        return QuizResponse(**quiz_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents", response_model=QuizResponse)
async def generate_quiz_by_documents(request: DocsQuizRequest):
    try:
        quiz_data = generate_quiz_by_docs(request)
        return QuizResponse(**quiz_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
