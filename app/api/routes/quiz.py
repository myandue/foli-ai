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
from app.services.documents_service import split_n_return_docs

router = APIRouter(prefix="/quiz", tags=["quiz"])


@router.post("/keyword", response_model=QuizResponse)
def quiz_by_keyword(request: KeywordQuizRequest):
    try:
        quiz_data = generate_quiz_by_keyword(*request)
        return QuizResponse(
            **(
                quiz_data["generate_quiz"]
                if "generate_quiz" in quiz_data
                else quiz_data
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents", response_model=QuizResponse)
async def quiz_by_documents(request: DocsQuizRequest):
    try:
        docs = await split_n_return_docs(text=request.docs)
        quiz_data = generate_quiz_by_docs(
            docs=docs, amount=request.amount, level=request.level
        )
        return QuizResponse(
            **(
                quiz_data["generate_quiz"]
                if "generate_quiz" in quiz_data
                else quiz_data
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
