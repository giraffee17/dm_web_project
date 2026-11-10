"""
main.py: the FastAPI web server.

  GET /api/forecast   live 3 day forecast for Ulaanbaatar (cached once per day)
  GET /api/meta       model information and test metrics (+ the data source check)
  GET /               the website (files in static/)

Run from the repo folder:
  python -m uvicorn app.main:app --port 7860
then open http://localhost:7860 in a browser.
"""

import json

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from src import config
from src.live import LiveDataError, get_forecast

app = FastAPI(title="Forecasting the Coldest Capital on Earth",
              description="3 day temperature forecast for Ulaanbaatar (CatBoost, ERA5)")


@app.get("/api/forecast")
def forecast():
    # A plain 'def' (not async): FastAPI runs it in a worker thread, so a slow
    # weather API call never blocks other visitors.
    try:
        return get_forecast()
    except LiveDataError as e:
        raise HTTPException(status_code=503, detail=f"Live forecast unavailable: {e}")


@app.get("/api/meta")
def meta():
    info = json.loads((config.MODELS_DIR / "meta.json").read_text())
    if config.SOURCE_CHECK_PATH.exists():
        info["source_check"] = json.loads(config.SOURCE_CHECK_PATH.read_text())["summary"]
    return info


# The website itself. Mounted last, so /api/... routes take priority.
app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")
