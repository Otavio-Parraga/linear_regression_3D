# Linear Regression 3D

Interactive explorer of the linear regression cost function
J(θ₀, θ₁) = (1/2m) Σ (θ₀ + θ₁x − y)², built for teaching.

- 3D cost surface you can rotate freely, with the current θ, its tangent plane,
  the ∂J/∂θ slices and the −∇J direction
- Editable regression function (slider or typed value), live predictions table
  and regression line
- Gradient descent: single step, 30 steps, or run to convergence, with the
  stable learning-rate bound shown
- Original vs normalized feature scale, light/dark theme, English/Portuguese

Stack: Python (FastAPI + NumPy) managed with [uv](https://docs.astral.sh/uv/);
the page uses Plotly.js and KaTeX from a CDN. The server computes the dataset
and cost surface once; everything interactive runs in the browser.

## Run locally

```bash
uv run uvicorn app.main:app --port 8123   # http://127.0.0.1:8123
./serve.sh                                # same, plus a public Cloudflare quick tunnel
uv run pytest -q                          # tests
```

## Deploy on Render

The repo includes a [Blueprint](render.yaml):

1. In Render: **New → Blueprint**, pick this repository, and apply.
2. Render builds with `uv sync --frozen --no-dev` and starts
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT`; the health check is `/healthz`.

To set it up by hand instead (**New → Web Service**, runtime Python):

| Setting | Value |
|---|---|
| Build command | `pip install uv && uv sync --frozen --no-dev` |
| Start command | `uv run --no-dev uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/healthz` |
| Env var | `PYTHON_VERSION=3.12.3` |

`requirements.txt` is exported from `uv.lock` for platforms that expect it
(`pip install -r requirements.txt`). Regenerate it after changing dependencies:
`uv export --no-dev --no-hashes --no-emit-project -o requirements.txt`.

On Render's free plan the service sleeps after about 15 minutes without
traffic; the first visit afterwards takes roughly a minute to wake it.
