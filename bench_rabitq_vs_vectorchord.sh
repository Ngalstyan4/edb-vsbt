#!/usr/bin/env bash
# Compare edb-pgvector RaBitQ path against VectorChord on the same dataset.
#
# Runs both indexes back-to-back on OpenAI 1536d 5M (cosine) at the same
# operating points, single-client and 16-client. Designed to answer two
# questions in one go:
#
#   1. Where does edb-pgvector RaBitQ (vector_rabitq_cosine_ops) sit on the
#      recall/QPS curve relative to VectorChord's rabitq?
#   2. Does the difference change between single- and multi-client?
#
# Prerequisites (server side):
#   - pgvector built and installed from ~/workspace/pgvector (narek/dev tip).
#   - VectorChord installed.
#   - Dataset openai-5m-cos staged at $DATASET_LOCAL_DIR.
#   - configs:
#       config/openai-1536-5m-angular/ivfflat_rabitq_rerank-32k.yaml  (already on narek/misc)
#       config/openai-1536-5m-angular/vectorchord-190-35k.yaml        (on narek/edb-pgvector-bench;
#                                                                       cherry-pick if missing)
#
# Index build is the long part; the script does NOT --skip-index-creation,
# so first run builds, second run reuses. To re-bench without rebuild,
# add --skip-index-creation to the relevant invocation.

set -euo pipefail

export DATASET_LOCAL_DIR="${DATASET_LOCAL_DIR:-/mnt/data/datasets/}"
export PGPASSWORD="${PGPASSWORD:-postgres}"

RABITQ_YAML=./config/openai-1536-5m-angular/ivfflat_rabitq_rerank-32k.yaml
VC_YAML=./config/openai-1536-5m-angular/vectorchord-190-35k.yaml

for f in "$RABITQ_YAML" "$VC_YAML"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing config: $f" >&2
    echo "If you're on narek/misc, the vectorchord yaml lives on" >&2
    echo "narek/edb-pgvector-bench. Cherry-pick it or copy it over." >&2
    exit 1
  fi
done

banner() {
  echo ""
  echo "============================================================"
  echo "  $*"
  echo "============================================================"
  echo ""
}

banner "OpenAI 1536d 5M (cosine) — RaBitQ vs VectorChord"

echo "--- [pgvector] vector_rabitq_cosine_ops, lists=32k (single client) ---"
uv run python pgvector_suite.py -s "$RABITQ_YAML"

echo ""
echo "--- [pgvector] vector_rabitq_cosine_ops, lists=32k (16 clients) ---"
uv run python pgvector_suite.py -s "$RABITQ_YAML" \
  --skip-index-creation --query-clients 16

echo ""
echo "--- [vectorchord] lists=[190,35k] (single client) ---"
uv run python vectorchord_suite.py -s "$VC_YAML"

echo ""
echo "--- [vectorchord] lists=[190,35k] (16 clients) ---"
uv run python vectorchord_suite.py -s "$VC_YAML" \
  --skip-index-creation --query-clients 16

banner "Done"
