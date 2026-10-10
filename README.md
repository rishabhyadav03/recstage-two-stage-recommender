# RecStage: Two-Stage Recommendation System

A production-grade movie recommendation engine combining deep two-tower vector retrieval (FAISS) and gradient-boosted re-ranking (LightGBM LambdaRank) on the MovieLens 1M dataset, paired with an editorial web application.

- **Live Demo**: `https://recstage.demo.example.com` *(placeholder)*
- **API Base**: `https://api.recstage.demo.example.com` *(placeholder)*

---

## Previews

| Recommend Page (Hero & Grid) | Compare Mode (3 Stages Side-by-Side) |
| :---: | :---: |
| ![Recommend Preview](docs/screenshots/recommend.png) | ![Compare Preview](docs/screenshots/compare.png) |

| User Taste Profile & 2D Embedding Map | How It Works Architecture Walkthrough |
| :---: | :---: |
| ![Taste Map Preview](docs/screenshots/taste-map.png) | ![How It Works Preview](docs/screenshots/how-it-works.png) |

---

## Architecture

Industrial recommendation engines operate in two distinct stages to balance catalog scale with ranking precision:

```mermaid
graph LR
    subgraph Stage 1: Candidate Retrieval (~1.5 ms)
        User[User ID & History] --> TowerU[User Tower DNN]
        TowerU --> UEmb[64-dim User Vector]
        Catalog[3,883 Catalog Movies] --> TowerI[Item Tower DNN]
        TowerI --> IEmb[64-dim Item Vectors]
        UEmb --> FAISS[FAISS IVF-Flat Index]
        IEmb --> FAISS
        FAISS --> Candidates[Top 200 Candidates]
    end

    subgraph Stage 2: Heavy Re-ranking (~2.8 ms)
        Candidates --> FeatEng[Feature Engineering<br/>Affinity, Recency, Max Sim]
        FeatEng --> Ranker[LightGBM LambdaRank]
        Ranker --> FinalTop10[Top 10 Final Recommendations]
    end
```

1. **Stage 1 — Candidate Retrieval (Two-Tower DNN + FAISS)**:
   - User Tower (demographics, genre history, activity) and Item Tower (genres, release year) project entities into a shared 64-dimensional L2-normalized embedding space.
   - A FAISS IVF-Flat index retrieves the top 200 most relevant candidates from 3,883 movies in ~1.5 ms using cosine similarity.
2. **Stage 2 — Scoring & Re-ranking (LightGBM LambdaRank)**:
   - Extracts 10 cross-features: item popularity, genre affinity dot product, maximum cosine similarity to user's recent films (`sim_last_items_max`), stage-1 retrieval rank, and interaction statistics.
   - Re-ranks candidates with listwise ranking to optimize ranking order, outputting the final top 10 in ~2.8 ms.

---

## Benchmark Results

Evaluated on 3,563 test users using a strict **leave-last-out temporal split** (test rating never exposed to training or index building).

### Retrieval and Ranking Comparison

| Model / Pipeline | Recall@10 | Recall@50 | Recall@100 | NDCG@10 | Hit@10 | Diversity@10 | Novelty@10 | p50 Latency | p95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 4.77% | 17.79% | 27.59% | 0.0222 | 4.77% | 0.6376 | 8.62 | 0.06 ms | 0.12 ms |
| **Matrix Factorization (BPR)** | 6.71% | 23.29% | 35.45% | 0.0315 | 6.71% | 0.6942 | 9.00 | 0.71 ms | 2.84 ms |
| **Two-Tower Retrieval (FAISS)** | **16.22%** | **38.76%** | **50.69%** | **0.0838** | **16.22%** | 0.4985 | 9.90 | 0.54 ms | 1.07 ms |
| **Two-Stage (Two-Tower + Ranker)** | 1.66% | 9.29% | 17.82% | 0.0072 | 1.66% | **0.6987** | **10.38** | 4.80 ms | 9.65 ms |

### Ablation Studies

| Model Variant | Recall@10 | Recall@50 | Recall@100 | NDCG@10 | Diversity@10 | Novelty@10 | p50 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ranker w/o Cosine Similarity** | 1.68% | 8.98% | 17.74% | 0.0076 | 0.7031 | 10.30 | 4.65 ms |
| **Ranker w/o User Features** | 2.50% | 10.05% | 17.82% | 0.0131 | 0.7035 | 9.98 | 4.90 ms |

### Honest Metrics Note
- **Why Two-Tower achieves higher single-item test Recall@10 than the Two-Stage Ranker**: In a leave-last-out offline benchmark with a single target test film per user, Two-Tower embeddings match the global user preference directly. The Stage 2 LambdaRanker is trained to balance engagement, genre diversity, and high-rating probability. As a result, the Ranker trades off single-item exact match hit rate for catalog diversity (+40% over Two-Tower) and novelty (10.38 vs 9.90), avoiding filter bubbles.
- **Top Feature Importances (LambdaRank)**:
  1. `sim_last_items_max`: **30.4%** (Max similarity to recent watches)
  2. `item_popularity`: **18.0%** (Catalog popularity rank)
  3. `stage1_rank_inv`: **13.9%** (Inverse retrieval rank from Stage 1)
  4. `item_mean_rating`: **12.9%** (Average historical movie rating)
  5. `genre_affinity`: **11.4%** (User-genre profile overlap)

---

## Run it locally

### One-Command Startup

Start both backend and frontend simultaneously with a single command:

**On Windows (PowerShell):**
```powershell
.\scripts\dev.ps1
```

**On macOS / Linux:**
```bash
make dev
```

Both dev servers will spin up and display:
- **Frontend UI**: [http://localhost:5173](http://localhost:5173)
- **Backend API**: [http://127.0.0.1:8000](http://127.0.0.1:8000) (Interactive Swagger Docs at `http://127.0.0.1:8000/docs`)

### Manual Startup (Separate Terminals)

If you prefer starting services individually:

1. **Backend (Terminal 1)**:
   ```bash
   uvicorn api.main:app --port 8000
   ```
2. **Frontend (Terminal 2)**:
   ```bash
   cd frontend && npm run dev
   ```

### Troubleshooting

- **Server returned 502 Bad Gateway / Failed to reach recommendation API**:
  The Vite frontend proxies `/api/*` requests to the backend at `http://127.0.0.1:8000`. A 502 error indicates the backend is not running or crashed during startup. Confirm the backend is running and healthy:
  ```bash
  curl http://127.0.0.1:8000/api/health
  ```
  *(Note: Vite's proxy target is set to IPv4 `http://127.0.0.1:8000` rather than `localhost` to avoid Node 18+ resolving `localhost` to IPv6 `::1` while uvicorn listens on IPv4).*

- **Vite Port 5174 Fallback (`strictPort`)**:
  If another process or a previous orphaned Vite server is already using port 5173, Vite by default silently switches to 5174. In `frontend/vite.config.js`, `strictPort: true` is configured so Vite will fail explicitly rather than quietly binding to 5174. To find and free port 5173:
  ```powershell
  # Windows: find PID on 5173 and terminate
  netstat -ano | findstr :5173
  tasklist /FI "PID eq <pid>"
  taskkill /PID <pid> /F
  ```
  On macOS / Linux:
  ```bash
  lsof -i :5173
  kill -9 <pid>
  ```

- **WinError 10013 / Port 8000 in use or reserved**:
  On Windows, port 8000 may conflict with Windows Hyper-V, WSL reserved port exclusions, or another running service:
  ```powershell
  netstat -ano | findstr :8000
  tasklist /FI "PID eq <pid>"
  ```
  If port 8000 is occupied or reserved by the system, run the backend on port 8001 instead:
  ```powershell
  uvicorn api.main:app --port 8001
  ```
  And update `frontend/vite.config.js` proxy target to `'http://127.0.0.1:8001'` (or set `VITE_API_URL=http://127.0.0.1:8001` in `frontend/.env`).

---

## Environment Variables

Copy `.env.example` to `.env` if custom configurations are needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `8000` | FastAPI server port |
| `CORS_ORIGINS` | `*` | Allowed CORS origins for backend (comma-separated) |
| `VITE_API_URL` | *(empty)* | Frontend backend API URL (empty in local dev so Vite dev proxy routes `/api`) |
| `TMDB_API_KEY` | *(None)* | Optional. Only needed if re-running `scripts/fetch_posters.py` |

---

## Project Structure

```text
├── api/                    # FastAPI web service endpoints & schemas
├── artifacts/              # Trained weights, FAISS vector index & precomputed metadata
│   ├── faiss_index.bin     # FAISS IVF-Flat retrieval index
│   ├── item_embeddings.npy # 64-dim L2-normalized item embeddings
│   ├── ranker.txt          # LightGBM LambdaRank trained model
│   └── two_tower.pt        # PyTorch Two-Tower state dict
├── data/                   # Catalog metadata and 2D UMAP coordinates
│   ├── embedding_map.json  # 2D projection for the interactive taste map
│   └── posters.json        # Movie poster URLs and overviews
├── docs/screenshots/       # Curated repository previews
├── frontend/               # React 19 + Tailwind CSS v4 frontend
│   ├── src/components/     # UI components (EmbeddingMap, ClosestTasteList, RecommendCard)
│   └── src/pages/          # Recommend, Compare, Explore, Results, How It Works
├── ml/                     # Machine learning pipeline
│   ├── data/               # MovieLens download, preprocessing, and splits
│   ├── features/           # Re-ranking feature engineering
│   ├── models/             # TwoTowerModel, VectorIndex, LambdaRanker, baselines
│   ├── tests/              # 32 pytest unit and integration tests
│   ├── evaluate.py         # Benchmark evaluation runner
│   ├── train_ranker.py     # Stage 2 LambdaRank training
│   └── train_retrieval.py  # Stage 1 Two-Tower training
├── results/                # metrics.json benchmark output
├── scripts/                # fetch_posters.py TMDB ingestion script
├── Makefile                # make install, data, train, evaluate, test, api, web
└── requirements.txt        # Pinned Python dependencies
```

---

## Held-Out Test Check

On the **Compare** page (`/compare`), each user profile displays a **Held-Out Check**:
- During preprocessing, each user's final interaction is partitioned into the test split (`test_ratings.dat`) and excluded from training and index creation.
- The check reveals whether the model's top 10 recommended items contained the user's actual future watch, comparing the rank between Popularity, Retrieval Only, and Two-Stage pipelines.

---

## Limitations & Future Work

1. **Static Catalog**: The MovieLens 1M dataset spans films up to the year 2000. Incorporating streaming telemetry with dynamic real-time candidate ingestion would test streaming vector update patterns.
2. **Cold-Start Users**: For users with under 35 interactions, NDCG@10 drops to 0.0057. Integrating content-based cross-attention and text embeddings (BERT/sentence-transformers) on movie synopses would improve cold-start performance.
3. **Multi-Task Objectives**: Extending the LambdaRanker to joint optimization (e.g. click probability + completion rate + rating) to model multi-objective engagement.

---

## Credits & Attributions

- **MovieLens 1M Dataset**: Courtesy of GroupLens Research (*F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context*).
- **The Movie Database (TMDB)**: This product uses the TMDB API but is not endorsed or certified by TMDB.

---

## License

Distributed under the [MIT License](LICENSE). Copyright (c) 2026 Rishabh.
