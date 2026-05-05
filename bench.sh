DATASET_LOCAL_DIR=/mnt/data/datasets/
PGPASSWORD=postgres

# bench openai 5m
uv run python pgvector_suite.py -s config_v2/ivfflat_bq_rerank-5m-openai.yaml
uv run python pgvector_suite.py -s config_v2/ivfflat_bq_rerank-5m-openai.yaml \
  --skip-index-creation --query-clients 10
uv run python vectorchord_suite.py -s config_v2/vectorchord_5m-openai.yaml
uv run python vectorchord_suite.py -s config_v2/vectorchord_5m-openai.yaml \
  --skip-index-creation --query-clients 10
