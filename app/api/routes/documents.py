from fastapi import APIRouter, HTTPException, UploadFile

from app.schemas.documents import SummaryResponse
from app.services.documents_service import generate_summary

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/summary", response_model=SummaryResponse)
async def summarize_docs(text_file: UploadFile):
    try:
        summary = await generate_summary(text_file)
        return SummaryResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
