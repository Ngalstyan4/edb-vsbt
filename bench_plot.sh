export DATASET_LOCAL_DIR=/mnt/data/datasets/
export PGPASSWORD=postgres

# bench openai 5m
#
uv run python pgvector_suite.py -s ./config/openai-1536-5m-angular/ivfflat_bq_rerank-32k.yaml \
  --query-clients 16
uv run python vectorchord_suite.py -s ./config/openai-1536-5m-angular/vectorchord-190-35k.yaml \
  --query-clients 16
