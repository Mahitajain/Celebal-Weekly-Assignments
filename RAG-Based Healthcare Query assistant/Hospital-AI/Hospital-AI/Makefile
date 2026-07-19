.PHONY: install load-data test run-api run-frontend docker-up docker-down lint

install:
	pip install -r requirements.txt

load-data:
	python -m app.database.load_data

test:
	pytest --cov=app --cov-report=term-missing

run-api:
	uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	streamlit run frontend/streamlit_app.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down
