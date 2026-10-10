#!/usr/bin/env bash
# Deploy RecStage backend to Hugging Face Spaces (Docker)
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <space-repo-url>"
    echo "Example: $0 https://huggingface.co/spaces/username/recstage-api"
    exit 1
fi

SPACE_REPO_URL="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== RecStage Hugging Face Spaces Deployment Helper ==="
echo "Source project: ${PROJECT_ROOT}"
echo "Target Space:   ${SPACE_REPO_URL}"
echo ""

# 1. Create a clean temporary directory
TEMP_DIR="$(mktemp -d 2>/dev/null || mktemp -d -t 'recstage_hf')"
echo "[1/6] Created temporary directory: ${TEMP_DIR}"

cleanup() {
    if [ "$?" -ne 0 ]; then
        echo "Deployment script encountered an error."
    fi
}
trap cleanup EXIT

# 2. Clone the Hugging Face Space repository
echo "[2/6] Cloning Space repository..."
git clone "${SPACE_REPO_URL}" "${TEMP_DIR}"

# 3. Copy runtime assets
echo "[3/6] Copying runtime files..."
cp "${PROJECT_ROOT}/Dockerfile.hf" "${TEMP_DIR}/Dockerfile"
cp "${PROJECT_ROOT}/requirements-deploy.txt" "${TEMP_DIR}/requirements-deploy.txt"

# api/
mkdir -p "${TEMP_DIR}/api"
cp -r "${PROJECT_ROOT}/api/"* "${TEMP_DIR}/api/"
find "${TEMP_DIR}/api" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# ml/ (without tests)
mkdir -p "${TEMP_DIR}/ml"
for item in "${PROJECT_ROOT}/ml/"*; do
    basename_item="$(basename "${item}")"
    if [ "${basename_item}" != "tests" ] && [ "${basename_item}" != "__pycache__" ]; then
        cp -r "${item}" "${TEMP_DIR}/ml/"
    fi
done
find "${TEMP_DIR}/ml" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# artifacts/ (runtime only)
mkdir -p "${TEMP_DIR}/artifacts"
for art in two_tower.pt faiss_index.bin faiss_index.npy item_embeddings.npy ranker.txt winning_users.json dataset_cache.pkl; do
    if [ -f "${PROJECT_ROOT}/artifacts/${art}" ]; then
        cp "${PROJECT_ROOT}/artifacts/${art}" "${TEMP_DIR}/artifacts/"
    fi
done

# data/
mkdir -p "${TEMP_DIR}/data"
[ -f "${PROJECT_ROOT}/data/embedding_map.json" ] && cp "${PROJECT_ROOT}/data/embedding_map.json" "${TEMP_DIR}/data/"
[ -f "${PROJECT_ROOT}/data/posters.json" ] && cp "${PROJECT_ROOT}/data/posters.json" "${TEMP_DIR}/data/"

# results/
mkdir -p "${TEMP_DIR}/results"
[ -f "${PROJECT_ROOT}/results/metrics.json" ] && cp "${PROJECT_ROOT}/results/metrics.json" "${TEMP_DIR}/results/"

# 4. Write README.md with HF front matter
echo "[4/6] Writing Hugging Face Space README.md..."
cat << 'EOF' > "${TEMP_DIR}/README.md"
---
title: RecStage API
emoji: 🎬
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# RecStage Recommendation API

High-performance two-stage recommendation API combining PyTorch Two-Tower candidate retrieval with LightGBM LambdaMART re-ranking on MovieLens-1M. Built with FastAPI and containerized for Hugging Face Spaces.
EOF

# 5. Check file sizes and warn if > 10 MB
echo "[5/6] Checking staged file sizes..."
OVER_10MB=0
while IFS= read -r file; do
    if [[ "$file" != *"/.git/"* ]]; then
        size_bytes=$(wc -c < "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)
        rel_path="${file#"${TEMP_DIR}/"}"
        size_mb=$(awk -v b="$size_bytes" 'BEGIN { printf "%.2f", b/1048576 }')
        size_kb=$(awk -v b="$size_bytes" 'BEGIN { printf "%.1f", b/1024 }')
        if [ "$size_bytes" -gt 10485760 ]; then
            echo "  [WARN > 10MB] ${rel_path} (${size_mb} MB)"
            OVER_10MB=1
        else
            echo "  ${rel_path} (${size_mb} MB / ${size_kb} KB)"
        fi
    fi
done < <(find "${TEMP_DIR}" -type f)

if [ "$OVER_10MB" -eq 1 ]; then
    echo "WARNING: Some files exceed 10 MB and may require Git LFS on Hugging Face Spaces."
else
    echo "All files are under 10 MB (safe for standard git push without Git LFS)."
fi

# 6. Git commit
echo "[6/6] Committing staged files..."
cd "${TEMP_DIR}"
git config user.name "Rishabh Yadav"
git config user.email "214611814+rishabhyadav03@users.noreply.github.com"
git add .
if ! git diff-index --quiet HEAD --; then
    git commit -m "chore(deploy): stage RecStage backend for Hugging Face Spaces"
    echo "Commit created successfully."
else
    echo "No changes to commit (repository already up to date)."
fi

CURRENT_BRANCH="$(git branch --show-current 2>/dev/null || echo "main")"
[ -z "$CURRENT_BRANCH" ] && CURRENT_BRANCH="main"

echo ""
echo "=========================================================="
echo "Deployment package is ready! To deploy, run the following:"
echo "=========================================================="
echo "cd \"${TEMP_DIR}\""
echo "git push origin ${CURRENT_BRANCH}"
echo "=========================================================="
echo "Note: git will prompt for your Hugging Face credentials."
echo "(Username: your HF username | Password: HF Access Token with write permissions)"
echo ""
