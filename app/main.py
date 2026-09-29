from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.model import build_model_payload

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Regressão Linear — Função de Custo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache(maxsize=1)
def model_payload() -> dict:
    return build_model_payload()


# Compute once at import/startup.
model_payload()


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/model")
def api_model() -> dict:
    return model_payload()
