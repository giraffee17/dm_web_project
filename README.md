# Forecasting the Coldest Capital on Earth

Forecasting Ulaanbaatar's daily mean temperature for the next 3 days with gradient boosting,
deployed behind a minimal website. (Full README with results comes at the end of the project.)

## Setup (Windows, from this folder)

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Phase 1: download and check the data

```
.venv\Scripts\python -m src.download      # cached: re-running skips finished chunks
.venv\Scripts\python -m src.check_data    # should end with ALL CHECKS PASSED
```
