import hashlib
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.model import build_model_payload

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Regressão Linear — Função de Custo")
# The model payload is ~300 KB of JSON; gzip cuts it to a fraction.
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Cache busting. Browsers would otherwise reuse old style.css/app.js with a new
# index.html after a deploy, which breaks the page. Asset links carry a hash of
# the files' contents, and every response asks the browser to revalidate
# (cheap: unchanged files come back as 304 via their ETag).
ASSETS = ("style.css", "i18n.js", "app.js")
ASSET_VERSION = hashlib.sha256(
    b"".join((STATIC_DIR / name).read_bytes() for name in ASSETS if (STATIC_DIR / name).exists())
).hexdigest()[:12]


@app.middleware("http")
async def revalidate(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-cache")
    return response
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache(maxsize=1)
def model_payload() -> dict:
    return build_model_payload()


# Compute once at import/startup.
model_payload()


@lru_cache(maxsize=1)
def index_html() -> str:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for name in ASSETS:
        html = html.replace(f"/static/{name}", f"/static/{name}?v={ASSET_VERSION}")
    return html


@app.get("/", include_in_schema=False)
def index():
    return HTMLResponse(index_html())


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/api/model")
def api_model() -> dict:
    return model_payload()
