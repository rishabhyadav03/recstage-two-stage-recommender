.PHONY: help dev install data posters train evaluate map api web test clean

help:
	@echo "RecStage: Two-Stage Recommendation System"
	@echo "=========================================="
	@echo "Available commands:"
	@echo "  make dev       Run backend and frontend together (one-command startup)"
	@echo "  make install   Install Python and Node.js dependencies"
	@echo "  make data      Download the MovieLens-1M dataset"
	@echo "  make train     Train Two-Tower retrieval and LightGBM LambdaRank models"
	@echo "  make evaluate  Evaluate baselines and two-stage pipeline against test split"
	@echo "  make map       Generate 2D UMAP projection for the embedding map"
	@echo "  make posters   Fetch posters and backdrops from TMDB API"
	@echo "  make api       Run FastAPI backend on http://localhost:8000"
	@echo "  make web       Run Vite frontend dev server on http://localhost:5173"
	@echo "  make test      Run backend test suite with pytest"
	@echo "  make clean     Remove build caches and temporary files"

dev:
	@echo "Starting RecStage development servers..."
	@echo "  Backend API:  http://127.0.0.1:8000 (docs: http://127.0.0.1:8000/docs)"
	@echo "  Frontend UI:  http://localhost:5173"
	@trap 'kill 0' INT TERM EXIT; uvicorn api.main:app --port 8000 & (cd frontend && npm run dev)

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	cd frontend && npm install

data:
	python ml/data/download.py

posters:
	python scripts/fetch_posters.py

train:
	python ml/train_retrieval.py
	python ml/train_ranker.py

evaluate:
	python ml/evaluate.py

map:
	python ml/build_embedding_map.py

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

web:
	cd frontend && npm run dev

test:
	pytest ml/tests -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf frontend/dist 2>/dev/null || true
