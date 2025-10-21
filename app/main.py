from fastapi import FastAPI

from app.api.router import api_router

app = FastAPI()

# 라우터 등록
app.include_router(api_router, prefix="/api")
