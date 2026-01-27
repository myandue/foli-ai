from fastapi import APIRouter, HTTPException

from app.schemas.documents import (
    SummaryRequest,
    SummaryResponse,
    QnARequest,
    QnAResponse,
)
from app.services.documents_service import (
    generate_summary,
    respond_to_question,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/summary", response_model=SummaryResponse)
async def summarize_docs(request: SummaryRequest):
    try:
        summary = await generate_summary(text=request.text)
        return SummaryResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/qna", response_model=QnAResponse)
async def qna(request: QnARequest):
    try:
        answer = await respond_to_question(
            question=request.question,
            history=request.history,
            text=request.text,
        )
        return QnAResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
