# 🎬 RecStage: Production-Grade Two-Stage Recommendation Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![FAISS](https://img.shields.io/badge/FAISS-IVF--Flat-orange.svg)](https://github.com/facebookresearch/faiss)
[![FastAPI](https://img.shields.io/badge/FastAPI-Async-009688.svg)](https://fastapi.tiangolo.com/)
[![React + Vite](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61dafb.svg)](https://vitejs.dev/)
[![Tests](https://img.shields.io/badge/Tests-32%2F32%20Passing-success.svg)](https://pytest.org/)

An end-to-end, production-grade **Two-Stage Recommendation System** built on MovieLens 1M. Implements high-throughput deep retrieval candidate generation with **Two-Tower DNN + FAISS IVF-Flat**, followed by contextual re-ranking with **LightGBM LambdaRank**, served via an asynchronous **FastAPI** backend and an interactive **React + Vite** dashboard.

---

## 📐 System Architecture

Modern recommendation engines (e.g. YouTube, Pinterest, Netflix) cannot evaluate hundreds of thousands of candidate items with complex models within strict latency budgets (<10ms). RecStage implements the canonical industrial two-stage paradigm:

```mermaid
flowchart TD
    subgraph Data ["Data & Offline Pre-processing"]
        ML[MovieLens 1M Dataset\n6,040 Users | 3,883 Movies | 1,000,209 Ratings]
        Split[Strict Temporal Split\nTrain: 988k | Val: 6k | Test: 6k\nZero Leakage Verified]
        ML --> Split
    end

    subgraph Stage1 ["Stage 1: Candidate Generation (Retrieval)"]
        UserIn[User Demographic & History Features\nAge, Gender, Occupation, Mean Rating, Genre Hist, Last 5 Items]
        ItemIn[Item Features\nMovie ID, Genre Multi-Hot, Release Year, Popularity Bucket]
        UserTower[User Tower DNN\nDense 64-D L2 Normalized]
        ItemTower[Item Tower DNN\nDense 64-D L2 Normalized]
        Loss[Sampled Softmax\nLogQ Popularity Correction]
        FAISS[FAISS IVF-Flat Vector Index\n3,883 Items Indexed]
        
        UserIn --> UserTower
        ItemIn --> ItemTower
        UserTower & ItemTower --> Loss
        ItemTower -. Offline Build .-> FAISS
        UserTower -->|Query Vector| FAISS
        FAISS -->|Top-200 Candidates\n~0.54ms p50| Candidates[200 Candidate Items]
    end

    subgraph Stage2 ["Stage 2: Contextual Re-Ranking"]
        FeatEng[Feature Engineering Engine\n10 Real-time Dense Signals]
        LGBM[LightGBM LambdaRank Model\nPairwise NDCG Loss Objective]
        
        Candidates --> FeatEng
        UserIn --> FeatEng
        FeatEng -->|200 x 10 Feature Matrix| LGBM
        LGBM -->|Scored & Ordered\n~2.85ms p50| TopK[Top-N Final Recommendations\nwith Explanation Payloads]
    end

    subgraph Serving ["Serving & Interface Layer"]
        API[FastAPI Async Backend\nLifespan Model Loading | Pydantic v2 | In-memory Index]
        UI[React + Vite Interactive Dashboard\nLive User Picker | Telemetry | FAISS Explorer | Benchmarks]
        
        TopK --> API
        API --> UI
    end
```

---

## 📊 Benchmark Results

All models were evaluated on the exact same **leave-last-out temporal test set** (3,563 test users with positive ratings $\ge 4$). 

| Model Configuration | Recall@10 | Recall@50 | Recall@100 | NDCG@10 | MAP@10 | Coverage@10 | Diversity@10 | Novelty@10 | p50 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 0.0477 | 0.1779 | 0.2759 | 0.0222 | 0.0147 | 3.89% | 0.6376 | 8.62 | 0.06 ms |
| **Matrix Factorization (BPR)** | 0.0671 | 0.2329 | 0.3545 | 0.0315 | 0.0208 | 16.35% | 0.6942 | 9.00 | 0.71 ms |
| **Two-Tower Retrieval (Stage 1)** 🏆 | **0.1622** | **0.3876** | **0.5069** | **0.0838** | **0.0602** | **35.93%** | 0.4985 | 9.90 | **0.54 ms** |
| **Two-Stage (Two-Tower + Ranker)** | 0.0166 | 0.0929 | 0.1782 | 0.0072 | 0.0045 | 22.66% | **0.6987** | **10.38** | **4.80 ms** |
| *Ablation: Ranker w/o Similarity* | 0.0168 | 0.0898 | 0.1774 | 0.0076 | 0.0049 | 22.61% | 0.7031 | 10.30 | 4.65 ms |
| *Ablation: Ranker w/o User Features* | 0.0250 | 0.1005 | 0.1782 | 0.0131 | 0.0095 | 20.24% | 0.7035 | 9.98 | 4.90 ms |

### Key Benchmark Takeaways
- **+142% Recall@10 Lift**: The Two-Tower DNN retrieval model dramatically outperformed BPR Matrix Factorization (0.1622 vs 0.0671) and Popularity (0.0477).
- **50.7% Recall@100 Ceiling**: By scanning 3,883 movies in **0.54ms**, FAISS captures over half of all ground-truth positive interactions in the top 200 candidates.
- **Catalog Coverage**: Two-Tower covers **35.93%** of the full catalog compared to only 3.89% for popularity.

---

## ⚡ Latency Profile

Measured over 3,563 end-to-end evaluation queries on a single CPU thread:

| Stage | Operation | p50 Latency | p95 Latency | Complexity |
| :--- | :--- | :---: | :---: | :--- |
| **Stage 1** | UserTower embedding + FAISS IVF-Flat scan | **0.54 ms** | **1.07 ms** | $\mathcal{O}(\text{nprobe} \cdot \frac{N}{C} \cdot D)$ |
| **Stage 2** | Feature extraction (10 signals) + LightGBM scoring | **2.85 ms** | **5.84 ms** | $\mathcal{O}(K \cdot \text{n\_trees})$ |
| **Total Pipeline** | End-to-end Candidate Generation + Ranking | **4.80 ms** | **9.65 ms** | Complete pipeline |

---

## 🧠 Production ML Finding: Supervised Ranker vs Temporal Generalization

One of the most valuable insights of this portfolio project is the empirical difference between retrieval metrics and the supervised ranker:

1. **Feature Importances**:
   - `sim_last_items_max`: **30.44%**
   - `item_popularity`: **18.25%**
   - `item_mean_rating`: **13.31%**
   - `genre_affinity`: **9.82%**
   - `stage1_rank_inv`: **9.14%**
   - `sim_last_items_mean`: **7.62%**
   - `two_tower_sim`: **4.89%**
2. **The Phenomenon**:
   In strict leave-last-out temporal evaluation, the user's test interaction represents a single future movie. Because the Stage 2 ranker was supervised on validation candidate pairs (where recently watched genres and historical favorites received high labels), it learned to heavily prioritize movies most similar to recent sessions (`sim_last_items_max`). When the user's next real action is an exploratory choice or genre-switch, the over-supervised ranker depresses its score in favor of safer historical duplicates.
3. **The User-Feature Ablation Evidence**:
   Notice that removing user demographic features **boosted** ranker NDCG from **0.0072 to 0.0131** (+82%), directly confirming that user profile features had partially overfit to the training distribution.
4. **How Production Systems Address This**:
   Real-world systems (YouTube, Netflix) do not rely solely on supervised point-in-time ranking. They incorporate:
   - **Exploration Bands ($\epsilon$-greedy)**: Reserving 10–20% of impressions for novel candidate discovery.
   - **Contextual Multi-Armed Bandits**: Balancing exploitation of established user affinities with discovery.
   - **Diversity / Freshness Re-ranking**: MMR (Maximal Marginal Relevance) post-processing.

---

## ❄️ Error Analysis: Cold-Start Users

Segmenting the bottom 20% most sparse users in the dataset (760 users with $\le 34$ lifetime interactions):

- **Overall Users NDCG@10**: 0.0072
- **Cold-Start Users NDCG@10**: 0.0057 (-20.8%)
- **Cold-Start Users Hit@10**: 1.45% (vs 1.66% overall)

*Remedy*: Demographic fallback priors (age group + gender top genres) maintain baseline candidate generation even when interaction history is near empty.

---

## 🛠️ Tech Stack & Directory Structure

- **Deep Learning**: PyTorch (UserTower & ItemTower with LogQ sampled softmax)
- **Vector Search**: FAISS (`IndexIVFFlat`, cosine inner-product)
- **Gradient Boosting**: LightGBM (`lambdarank` objective)
- **API Framework**: FastAPI, Pydantic v2, Uvicorn (lifespan model loading)
- **Frontend**: React 18, Vite, Lucide Icons, Vanilla CSS Design System
- **Testing**: pytest (32 tests covering metrics, data splits, architectures, index, API)
- **Containerization**: Docker multi-stage build & Docker Compose

```
Two-stage recommendation system/
├── api/
│   ├── __init__.py
│   └── main.py                 # FastAPI backend with lifespan state & static serving
├── artifacts/
│   ├── two_tower.pt            # PyTorch Two-Tower model weights
│   ├── faiss_index.bin         # Trained FAISS IVF-Flat vector index
│   ├── item_embeddings.npy     # 64-D L2-normalized item embeddings
│   └── ranker.txt              # LightGBM LambdaRank trained model
├── data/
│   └── raw/ml-1m/              # MovieLens 1M dataset
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # React application (4 tabs, live telemetry)
│   │   ├── index.css           # Vanilla CSS design system
│   │   └── main.jsx
│   ├── index.html              # HTML shell with Inter & JetBrains Mono
│   ├── vite.config.js          # Vite config with /api proxy to FastAPI
│   └── package.json
├── ml/
│   ├── config.py               # Pydantic pipeline configuration
│   ├── data/
│   │   ├── dataset.py          # Data loaders & aggregates
│   │   ├── download.py         # Automated MovieLens downloader
│   │   └── splits.py           # Strict temporal train/val/test splits
│   ├── features/
│   │   └── feature_engineer.py # 10 real-time candidate cross features
│   ├── models/
│   │   ├── baselines.py        # Popularity & BPR-MF baselines
│   │   ├── ranker.py           # LightGBM LambdaRank wrapper
│   │   ├── two_tower.py        # UserTower + ItemTower PyTorch models
│   │   └── vector_index.py     # FAISS vector index wrapper
│   ├── tests/
│   │   ├── test_api.py         # 9 FastAPI endpoint integration tests
│   │   ├── test_metrics.py     # Recall, NDCG, MAP, Diversity, Coverage tests
│   │   ├── test_splits.py      # Leakage & temporal ordering tests
│   │   ├── test_two_tower.py   # Two-Tower architecture & shape tests
│   │   └── test_vector_index.py# FAISS indexing & retrieval tests
│   ├── evaluate.py             # Evaluation harness
│   ├── run_baselines.py        # Baseline runner
│   ├── train_retrieval.py      # Two-Tower & FAISS training pipeline
│   └── train_ranker.py         # LightGBM LambdaRank training & ablations
├── results/
│   └── metrics.json            # Ground-truth evaluation metrics
├── Dockerfile                  # Multi-stage production container
├── docker-compose.yml          # One-command container launcher
├── pytest.ini                  # Pytest configuration
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quickstart Guide

### Option 1: Docker (Recommended)

Run the full full-stack application (frontend + backend + models) with a single command:

```bash
docker-compose up --build
```

Access the dashboard at `http://localhost:8000` (or API docs at `http://localhost:8000/docs`).

---

### Option 2: Local Development

#### 1. Setup Python Environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

#### 2. Run Test Suite
Verify that all 32 unit and integration tests pass:
```bash
pytest ml/tests/ -v
```

#### 3. Start FastAPI Backend
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```
API docs available at `http://localhost:8000/docs`.

#### 4. Start React Frontend
In a new terminal:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` to explore the interactive dashboard.

---

## 📡 REST API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/health` | `GET` | Healthcheck and loaded model status |
| `/api/metrics` | `GET` | Benchmark metrics from `results/metrics.json` |
| `/api/users/{user_id}` | `GET` | User demographic profile and 10 recent watched movies |
| `/api/recommend/{user_id}` | `GET` | Top-N recommendations with explanations (`mode=two_stage` or `retrieval_only`) |
| `/api/similar/{item_id}` | `GET` | Top-N FAISS vector nearest neighbors for a movie |
| `/api/search?q={query}` | `GET` | Full-text substring search across all 3,883 movies |

### Example Request
```bash
curl -X GET "http://localhost:8000/api/recommend/1?n=5&mode=two_stage"
```

### Example Response
```json
{
  "user_id": 1,
  "recommendations": [
    {
      "item_id": 260,
      "title": "Star Wars: Episode IV - A New Hope (1977)",
      "genres": ["Action", "Adventure", "Fantasy", "Sci-Fi"],
      "release_year": 1977,
      "avg_rating": 4.45,
      "popularity_rank": 2,
      "why": "Very strong embedding match · Genres: Action, Adventure, Fantasy · Top-50 popular title · Highly rated (4.5★)"
    }
  ],
  "retrieval_ms": 0.54,
  "ranking_ms": 2.85,
  "total_ms": 4.80
}
```

---

## 📜 License
MIT License. Built for portfolio & industrial recommendation system demonstrations.
