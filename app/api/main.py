from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import settings
from app.api.routes_bandit import router as bandit_router
from app.api.routes_causal import router as causal_router

app = FastAPI(
    title="Causal Inference & Multi-Armed Bandit Optimization Engine",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(bandit_router)
app.include_router(causal_router)

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "processed_data_ready": settings.CRITEO_CAUSAL_SAMPLE_PARQUET.exists(),
        "bandit_db": str(settings.BANDIT_DB_PATH),
    }
