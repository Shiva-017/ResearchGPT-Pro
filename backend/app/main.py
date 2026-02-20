# backend/app/main.py

"""
FastAPI application entry point.

Run:
    uvicorn backend.app.main:app --reload
    uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.app.routes.search import router as search_router

app = FastAPI(
    title="ResearchGPT-Pro API",
    description="Hybrid RAG search for academic papers — HyDE + BM25 + cross-encoder reranking",
    version="0.1.0",
)

# CORS — allow frontend (Next.js dev server) and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://research-gpt-frontend.vercel.app",
        "https://research-gpt-frontend-*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(search_router)


@app.on_event("startup")
async def startup():
    logger.info("ResearchGPT-Pro API starting…")


@app.on_event("shutdown")
async def shutdown():
    logger.info("ResearchGPT-Pro API shutting down.")
