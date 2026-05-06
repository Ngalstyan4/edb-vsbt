export DATASET_LOCAL_DIR=/mnt/data/datasets/
export PGPASSWORD=postgres

echo ""
echo "============================================================"
echo "  EXPERIMENT: OpenAI 1536-dim 5M vectors (cosine)"
echo "============================================================"
echo ""

echo "--- [pgvector] ivfflat_bq_rerank lists=32k (single client) ---"
uv run python pgvector_suite.py -s ./config/openai-1536-5m-angular/ivfflat_bq_rerank-32k.yaml
echo ""
echo "--- [pgvector] ivfflat_bq_rerank lists=32k (16 clients) ---"
uv run python pgvector_suite.py -s ./config/openai-1536-5m-angular/ivfflat_bq_rerank-32k.yaml \
  --skip-index-creation --query-clients 16

echo ""
echo "--- [vectorchord] lists=[190,35k] (single client) ---"
uv run python vectorchord_suite.py -s ./config/openai-1536-5m-angular/vectorchord-190-35k.yaml
echo ""
echo "--- [vectorchord] lists=[190,35k] (16 clients) ---"
uv run python vectorchord_suite.py -s ./config/openai-1536-5m-angular/vectorchord-190-35k.yaml \
  --skip-index-creation --query-clients 16

echo ""
echo "--- [pgvector] ivfflat_bq_rerank lists=8k (single client) ---"
uv run python pgvector_suite.py -s ./config/openai-1536-5m-angular/ivfflat_bq_rerank-8k.yaml
echo ""
echo "--- [pgvector] ivfflat_bq_rerank lists=8k (16 clients) ---"
uv run python pgvector_suite.py -s ./config/openai-1536-5m-angular/ivfflat_bq_rerank-8k.yaml \
  --skip-index-creation --query-clients 16

echo ""
echo "--- [vectorchord] lists=[50,8k] (single client) ---"
uv run python vectorchord_suite.py -s ./config/openai-1536-5m-angular/vectorchord-50-8k.yaml
echo ""
echo "--- [vectorchord] lists=[50,8k] (16 clients) ---"
uv run python vectorchord_suite.py -s ./config/openai-1536-5m-angular/vectorchord-50-8k.yaml \
  --skip-index-creation --query-clients 16

# other vectorchord settings
# uv run python vectorchord_suite.py -s ./config/openai-1536-5m-angular/vectorchord-50-8k.yaml
# uv run python vectorchord_suite.py -s ./config/openai-1536-5m-angular/vectorchord-50-8k.yaml \
#   --skip-index-creation --query-clients 16

echo ""
echo "============================================================"
echo "  EXPERIMENT: Cohere 768-dim 10M vectors (cosine)"
echo "============================================================"
echo ""

echo "--- [pgvector] ivfflat_bq_rerank lists=32k (single client) ---"
uv run python pgvector_suite.py -s ./config/cohere-768-10m-cos/ivfflat_bq_rerank-32k.yaml
echo ""
echo "--- [pgvector] ivfflat_bq_rerank lists=32k (16 clients) ---"
uv run python pgvector_suite.py -s ./config/cohere-768-10m-cos/ivfflat_bq_rerank-32k.yaml \
  --skip-index-creation --query-clients 16

echo ""
echo "--- [vectorchord] lists=[180,32k] (single client) ---"
uv run python vectorchord_suite.py -s ./config/cohere-768-10m-cos/vectorchord-180-32k.yaml
echo ""
echo "--- [vectorchord] lists=[180,32k] (16 clients) ---"
uv run python vectorchord_suite.py -s ./config/cohere-768-10m-cos/vectorchord-180-32k.yaml \
  --skip-index-creation --query-clients 16

echo ""
echo "--- [pgvector] ivfflat_bq_rerank lists=10k (single client) ---"
uv run python pgvector_suite.py -s ./config/cohere-768-10m-cos/ivfflat_bq_rerank-10k.yaml
echo ""
echo "--- [pgvector] ivfflat_bq_rerank lists=10k (16 clients) ---"
uv run python pgvector_suite.py -s ./config/cohere-768-10m-cos/ivfflat_bq_rerank-10k.yaml \
  --skip-index-creation --query-clients 16

echo ""
echo "--- [vectorchord] lists=[100,10k] (single client) ---"
uv run python vectorchord_suite.py -s ./config/cohere-768-10m-cos/vectorchord-100-10k.yaml
echo ""
echo "--- [vectorchord] lists=[100,10k] (16 clients) ---"
uv run python vectorchord_suite.py -s ./config/cohere-768-10m-cos/vectorchord-100-10k.yaml \
  --skip-index-creation --query-clients 16

echo ""
echo "============================================================"
echo "  ALL BENCHMARKS COMPLETE"
echo "============================================================"
