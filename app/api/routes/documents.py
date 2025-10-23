from fastapi import APIRouter, HTTPException

from app.schemas.documents import SummaryRequest, SummaryResponse
from app.services.documents_service import generate_summary

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/summary", response_model=SummaryResponse)
async def summarize_docs(request: SummaryRequest):
    try:
        summary = await generate_summary(text=request.text)
        return SummaryResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
