Create Env

`python -m venv .venv`

Activate venv

`.\venv\Scripts\Activate.ps1`

Install Dependencies

`pip install fastapi numpy`

run server

`uvicorn app:app --host 0.0.0.0 --port 8000 --reload`
