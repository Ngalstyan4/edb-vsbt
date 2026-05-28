"""
pgvector Benchmark Suite

Benchmarks vector search using the pgvector extension with HNSW or
IVFFlat indexes (vanilla and BQ + rerank variants) for PostgreSQL.
"""

import argparse
import math
import time

import numpy as np
import psycopg
import pgvector.psycopg
from tqdm import tqdm

import common
from results import ResultsManager


def build_arg_parse():
    """Build argument parser for pgvector benchmark suite."""
    parser = argparse.ArgumentParser(description="pgvector Benchmark Suite")
    common.build_arg_parse(parser)
    return parser


class PgvectorBaseTestSuite(common.TestSuite):
    """
    Shared base for all pgvector-backed suites (HNSW, IVFFlat, IVFFlat-BQ-Rerank).

    Holds the bits that are identical across index types: extension setup,
    connection helper that registers pgvector codecs, index prewarm, and the
    metric → operator / opclass lookups.
    """

    @staticmethod
    def _get_metric_operator(metric: str) -> str:
        """Convert metric name to PostgreSQL operator."""
        operators = {
            "l2": "<->",
            "euclidean": "<->",
            "cos": "<=>",
            "angular": "<=>",
            "dot": "<#>",
            "ip": "<#>",
        }
        if metric not in operators:
            raise ValueError(f"Unsupported metric type: {metric}")
        return operators[metric]

    @staticmethod
    def _get_metric_func(metric: str) -> str:
        """Convert metric name to pgvector operator class."""
        funcs = {
            "l2": "vector_l2_ops",
            "euclidean": "vector_l2_ops",
            "cos": "vector_cosine_ops",
            "ip": "vector_ip_ops",
            "dot": "vector_ip_ops",
        }
        if metric not in funcs:
            raise ValueError(f"Unsupported metric type: {metric}")
        return funcs[metric]

    def create_connection(self):
        """Create a database connection with pgvector support."""
        conn = super().create_connection()
        pgvector.psycopg.register_vector(conn)
        return conn

    def init_ext(self, suite_name: str = None):
        """Initialize required PostgreSQL extensions."""
        conn = super().create_connection()
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
        conn.close()
        self.debug_log("Extensions initialized successfully.")

    def prewarm_index(self, table_name: str):
        """Prewarm the index into memory for consistent benchmarking."""
        index_name = f"{table_name}_embedding_idx"
        conn = self.create_connection()
        self.check_index_fits_shared_buffers(conn, index_name, table_name)
        print("Prewarming the index into shared_buffers...", end="", flush=True)
        try:
            prewarm_start = time.perf_counter()
            conn.execute(f"SELECT pg_prewarm('{index_name}')")
            prewarm_time = time.perf_counter() - prewarm_start
            print(f" done! ({prewarm_time:.1f}s)")
        except psycopg.Error as e:
            print(f" failed! ({e.diag.message_primary})")
            self.debug_log(f"Prewarm failed: {e}")

        finally:
            conn.close()


class HNSWTestSuite(PgvectorBaseTestSuite):
    """
    Test suite for pgvector HNSW indexing.

    Uses the pgvector extension to build HNSW indexes and perform
    approximate nearest neighbor searches.
    """

    @staticmethod
    def process_batch(args):
        """Process a batch of queries in parallel."""
        test, answer, top, metric_ops, url, table_name, ef_search, warmup_n = args

        conn = psycopg.connect(url)
        pgvector.psycopg.register_vector(conn)
        conn.execute(f"SET hnsw.ef_search={ef_search}")
        conn.execute("SET enable_seqscan = off")

        query_sql = f"SELECT id FROM {table_name} ORDER BY embedding {metric_ops} %s LIMIT {top}"

        cursor = conn.cursor()

        # Per-worker warmup (warms this process's CPU caches); latencies
        # and results never enter the returned list.
        if warmup_n:
            n_test = len(test)
            for j in range(warmup_n):
                cursor.execute(query_sql, (test[j % n_test],))
                cursor.fetchall()

        results = []
        for query, ground_truth in zip(test, answer):
            start = time.perf_counter()
            cursor.execute(query_sql, (query,))
            result = cursor.fetchall()
            end = time.perf_counter()

            result_ids = {p[0] for p in result[:top]}
            gt_ids = ground_truth[:top]
            ground_truth_ids = set(gt_ids.tolist() if hasattr(gt_ids, "tolist") else gt_ids)
            hit = len(result_ids & ground_truth_ids)
            results.append((hit, (start, end)))

        cursor.close()
        conn.close()
        return results

    def make_batch_args(self, test, answer, top, metric, table_name, benchmark,
                        warmup_n=0):
        """Prepare arguments for parallel batch processing."""
        metric_ops = self._get_metric_operator(metric)
        return (
            test,
            answer,
            top,
            metric_ops,
            self.url,
            table_name,
            benchmark["efSearch"],
            warmup_n,
        )

    def apply_session_guc(self, conn, benchmark):
        conn.execute(f"SET hnsw.ef_search={benchmark['efSearch']}")
        conn.execute("SET enable_seqscan = off")

    @staticmethod
    def estimate_hnsw_graph_memory(num_vectors: int, dim: int, m: int) -> int:
        """Estimate maintenance_work_mem needed for an in-memory HNSW build.

        Based on pgvector's in-memory graph layout (HnswElementData, neighbor
        arrays, and vector storage). Each node at level L consumes:

          MAXALIGN(sizeof(HnswElementData))        ~128 bytes
          MAXALIGN(8 + 4*dim)                      vector value
          MAXALIGN(8 * (L+1))                      neighbor list pointers
          MAXALIGN(8 + 32*m)                       layer 0 neighbor array
          L * MAXALIGN(8 + 16*m)                   upper layer neighbor arrays

        Levels follow P(level >= L) = (1/m)^L, so the expected upper-layer
        overhead per node is (1/(m-1)) * (8 + MAXALIGN(8 + 16*m)).
        """
        def maxalign(x):
            return (x + 7) & ~7

        element_size = 128  # sizeof(HnswElementData) after alignment
        vector_size = maxalign(8 + 4 * dim)
        layer0_neighbors = maxalign(8 + 32 * m)
        layer0_ptrs = maxalign(8)  # neighbor list pointer for layer 0
        upper_layer_cost = maxalign(8) + maxalign(8 + 16 * m)
        upper_layer_fraction = 1.0 / (m - 1) if m > 1 else 0

        avg_per_node = (
            element_size
            + vector_size
            + layer0_ptrs
            + layer0_neighbors
            + upper_layer_fraction * upper_layer_cost
        )

        return int(num_vectors * avg_per_node)

    @staticmethod
    def estimate_hnsw_index_size(num_vectors: int, dim: int, m: int) -> int:
        """Estimate on-disk HNSW index size based on pgvector's page layout.

        Each node on disk stores the vector, layer-0 neighbors (2*M entries),
        and a fraction of upper-layer neighbors. Nodes are packed into 8KB
        PostgreSQL pages, and page fragmentation waste is accounted for.

        Validated against:
          dim=96,  m=16, 1B vectors  → predicts 632 GB (actual 646 GB, ~2% off)
          dim=768, m=16, 5M vectors  → predicts 19.0 GB (actual 18.8 GB, ~1% off)
        """
        def maxalign(x):
            return (x + 7) & ~7

        USABLE_PAGE = 8192 - 40  # page header + HNSW special space
        TUPLE_OVERHEAD = 32      # line pointer (4) + tuple header (~28)
        NEIGHBOR_SIZE = 6        # ItemPointerData: BlockIdData(4) + OffsetNumber(2)

        vector_bytes = maxalign(8 + 4 * dim)
        neighbor_bytes_l0 = maxalign(4 + 2 * m * NEIGHBOR_SIZE)
        upper_neighbor_avg = maxalign(4 + m * NEIGHBOR_SIZE) / (m - 1) if m > 1 else 0
        raw_node_size = TUPLE_OVERHEAD + vector_bytes + neighbor_bytes_l0 + int(upper_neighbor_avg)

        nodes_per_page = max(1, USABLE_PAGE // raw_node_size)
        actual_bytes_per_node = USABLE_PAGE / nodes_per_page

        return int(actual_bytes_per_node * num_vectors)

    def create_index(self, suite_name: str, table_name: str, dataset: dict):
        """Create an HNSW index using pgvector."""
        event, index_monitor_thread = super().create_index(
            suite_name, table_name, dataset
        )

        config = self.config[suite_name]
        pg_parallel_workers = config["pg_parallel_workers"]
        m = config["m"]
        ef_construction = config["efConstruction"]
        metric = dataset["metric"]
        metric_func = self._get_metric_func(metric)

        num_vectors = dataset.get("num", 0)
        dim = dataset.get("dim", 0)
        if num_vectors and dim:
            est_bytes = self.estimate_hnsw_graph_memory(num_vectors, dim, m)
            est_gb = est_bytes / (1024 ** 3)
            est_mwm = f"{int(est_gb + 1)}GB"
            est_idx_bytes = self.estimate_hnsw_index_size(num_vectors, dim, m)
            est_idx_gb = est_idx_bytes / (1024 ** 3)
            print(f"Estimated HNSW graph memory: {est_gb:.1f} GB "
                  f"(recommended maintenance_work_mem >= '{est_mwm}')")
            print(f"Estimated on-disk index size: {est_idx_gb:.1f} GB "
                  f"(recommended shared_buffers >= '{int(est_idx_gb + 1)}GB' for query serving)")

        if self.debug:
            print(f"\n🔧 Index Configuration (HNSW):")
            print(f"    • M:               {m}")
            print(f"    • EF Construction: {ef_construction}")
            print(f"    • Metric Function: {metric_func}")
            print()

        conn = self.create_connection()
        start_time = time.perf_counter()

        conn.execute(f"SET max_parallel_maintenance_workers TO {pg_parallel_workers}")
        conn.execute(f"SET max_parallel_workers TO {pg_parallel_workers}")
        conn.execute(
            f"CREATE INDEX {table_name}_embedding_idx ON {table_name} "
            f"USING hnsw (embedding {metric_func}) WITH (m = {m}, ef_construction = {ef_construction})"
        )

        build_time = int(round(time.perf_counter() - start_time))
        self.results[suite_name]["index_build_time"] = build_time

        event.set()
        index_monitor_thread.join()

        print(f"Index build time: {build_time}s")

        conn.execute("CHECKPOINT")
        conn.close()
        print("Index built successfully.")

    def sequential_bench(
        self,
        name: str,
        table_name: str,
        conn: psycopg.Connection,
        metric: str,
        top: int,
        benchmark: dict,
        dataset: dict,
    ) -> tuple[list[tuple[int, float]], str]:
        """Run sequential benchmark queries."""
        conn.execute(f"SET hnsw.ef_search={benchmark['efSearch']}")
        conn.execute("SET enable_seqscan = off")

        metric_ops = self._get_metric_operator(metric)

        self.debug_log(
            f"Benchmark config: ef_search={benchmark['efSearch']}, "
            f"metric={metric}, metric_ops={metric_ops}"
        )

        self.warmup_for_benchmark(
            conn, table_name, dataset, metric_ops, top, name
        )

        return super().sequential_bench(
            name, table_name, conn, metric_ops, top, benchmark, dataset
        )

    def generate_markdown_result(self):
        """Generate benchmark results with charts and consolidated CSV."""
        self.debug_log(f"Results: {self.results}")

        results_manager = ResultsManager()

        # Get monitoring data for each suite
        for suite_name in self.config:
            system_metrics, pg_stats, dashboard_path = self.get_monitoring_data(suite_name)

            results_manager.process_suite_results(
                suite_type="pgvector",
                config={suite_name: self.config[suite_name]},
                results={suite_name: self.results.get(suite_name, {})},
                query_clients=self.query_clients,
                system_metrics=system_metrics,
                pg_stats=pg_stats,
                system_dashboard_path=dashboard_path,
            )


class IVFFlatTestSuite(PgvectorBaseTestSuite):
    """
    Test suite for vanilla IVFFlat indexing.

    Single-stage `ORDER BY embedding <op> q LIMIT k` against an IVFFlat
    index on the float vector column.
    """

    @staticmethod
    def process_batch(args):
        """Process a batch of queries in parallel."""
        test, answer, top, metric_ops, url, table_name, probes, warmup_n = args

        conn = psycopg.connect(url)
        pgvector.psycopg.register_vector(conn)
        conn.execute(f"SET ivfflat.probes TO {probes}")
        conn.execute("SET enable_seqscan = off")

        query_sql = f"SELECT id FROM {table_name} ORDER BY embedding {metric_ops} %s LIMIT {top}"

        cursor = conn.cursor()

        if warmup_n:
            n_test = len(test)
            for j in range(warmup_n):
                cursor.execute(query_sql, (test[j % n_test],))
                cursor.fetchall()

        results = []
        for query, ground_truth in zip(test, answer):
            start = time.perf_counter()
            cursor.execute(query_sql, (query,))
            result = cursor.fetchall()
            end = time.perf_counter()

            result_ids = {p[0] for p in result[:top]}
            gt_ids = ground_truth[:top]
            ground_truth_ids = set(gt_ids.tolist() if hasattr(gt_ids, "tolist") else gt_ids)
            hit = len(result_ids & ground_truth_ids)
            results.append((hit, (start, end)))

        cursor.close()
        conn.close()
        return results

    def make_batch_args(self, test, answer, top, metric, table_name, benchmark,
                        warmup_n=0):
        metric_ops = self._get_metric_operator(metric)
        return (
            test,
            answer,
            top,
            metric_ops,
            self.url,
            table_name,
            benchmark["probes"],
            warmup_n,
        )

    def apply_session_guc(self, conn, benchmark):
        conn.execute(f"SET ivfflat.probes TO {benchmark['probes']}")
        conn.execute("SET enable_seqscan = off")

    def create_index(self, suite_name: str, table_name: str, dataset: dict):
        """Create a vanilla IVFFlat index on the float vector column."""
        event, index_monitor_thread = super().create_index(
            suite_name, table_name, dataset
        )

        config = self.config[suite_name]
        pg_parallel_workers = config.get("pg_parallel_workers", 2)
        lists = config["lists"]
        if lists == "auto":
            lists = max(1, int(math.sqrt(dataset["num"])))
        metric = dataset["metric"]
        metric_func = self._get_metric_func(metric)
        maintenance_work_mem = config.get("maintenance_work_mem")

        if self.debug:
            print(f"\n🔧 Index Configuration (IVFFlat):")
            print(f"    • Lists:           {lists}")
            print(f"    • Metric Function: {metric_func}")
            print()

        self.results[suite_name]["lists"] = lists

        conn = self.create_connection()
        start_time = time.perf_counter()

        if maintenance_work_mem:
            conn.execute(f"SET maintenance_work_mem TO '{maintenance_work_mem}'")
        conn.execute(f"SET max_parallel_maintenance_workers TO {pg_parallel_workers}")
        conn.execute(f"SET max_parallel_workers TO {pg_parallel_workers}")
        conn.execute(
            f"CREATE INDEX {table_name}_embedding_idx ON {table_name} "
            f"USING ivfflat (embedding {metric_func}) WITH (lists = {lists})"
        )

        build_time = int(round(time.perf_counter() - start_time))
        self.results[suite_name]["index_build_time"] = build_time

        event.set()
        index_monitor_thread.join()

        print(f"Index build time: {build_time}s")

        conn.execute("CHECKPOINT")
        conn.close()
        print("Index built successfully.")

    def sequential_bench(
        self,
        name: str,
        table_name: str,
        conn: psycopg.Connection,
        metric: str,
        top: int,
        benchmark: dict,
        dataset: dict,
    ) -> tuple[list[tuple[int, float]], str]:
        """Run sequential benchmark queries with IVFFlat probes set."""
        self.apply_session_guc(conn, benchmark)
        metric_ops = self._get_metric_operator(metric)

        self.debug_log(
            f"Benchmark config: probes={benchmark['probes']}, "
            f"metric={metric}, metric_ops={metric_ops}"
        )

        self.warmup_for_benchmark(
            conn, table_name, dataset, metric_ops, top, name
        )

        return super().sequential_bench(
            name, table_name, conn, metric_ops, top, benchmark, dataset
        )

    def generate_markdown_result(self):
        """Generate benchmark results with charts and consolidated CSV."""
        self.debug_log(f"Results: {self.results}")

        results_manager = ResultsManager()

        for suite_name in self.config:
            system_metrics, pg_stats, dashboard_path = self.get_monitoring_data(suite_name)

            results_manager.process_suite_results(
                suite_type="ivfflat",
                config={suite_name: self.config[suite_name]},
                results={suite_name: self.results.get(suite_name, {})},
                query_clients=self.query_clients,
                system_metrics=system_metrics,
                pg_stats=pg_stats,
                system_dashboard_path=dashboard_path,
            )


class IVFFlatBQRerankTestSuite(PgvectorBaseTestSuite):
    """
    Test suite for IVFFlat with binary quantization and exact rerank.

    Builds an IVFFlat index on `binary_quantize(embedding)::bit(dim)` with
    Hamming distance, then queries in two stages: a fast BQ scan returning
    `top * rerank_limit_amplify_factor` candidates, followed by an exact
    rerank using the configured metric on the original `vector` column.
    """

    DEFAULT_RERANK_AMP = 20

    @staticmethod
    def _two_stage_query_sql(table_name: str, dim: int, rerank_op: str, top: int) -> str:
        return (
            f"SELECT id FROM ("
            f"SELECT id, embedding FROM {table_name} "
            f"ORDER BY binary_quantize(embedding)::bit({dim}) <~> "
            f"binary_quantize(%s::vector({dim}))::bit({dim}) "
            f"LIMIT %s::int"
            f") sub "
            f"ORDER BY embedding {rerank_op} %s::vector({dim}) "
            f"LIMIT {top}"
        )

    @staticmethod
    def process_batch(args):
        """Process a batch of queries in parallel (two-stage BQ rerank)."""
        (test, answer, top, rerank_op, url, table_name, probes, dim,
         rerank_amp, warmup_n) = args

        conn = psycopg.connect(url)
        pgvector.psycopg.register_vector(conn)
        conn.execute(f"SET ivfflat.probes TO {probes}")

        rerank_limit = top * rerank_amp
        query_sql = IVFFlatBQRerankTestSuite._two_stage_query_sql(
            table_name, dim, rerank_op, top
        )
        bind = lambda q: (q, rerank_limit, q)

        cursor = conn.cursor()

        if warmup_n:
            n_test = len(test)
            for j in range(warmup_n):
                cursor.execute(query_sql, bind(test[j % n_test]))
                cursor.fetchall()

        results = []
        for query, ground_truth in zip(test, answer):
            start = time.perf_counter()
            cursor.execute(query_sql, bind(query))
            result = cursor.fetchall()
            end = time.perf_counter()

            result_ids = {p[0] for p in result[:top]}
            gt_ids = ground_truth[:top]
            ground_truth_ids = set(gt_ids.tolist() if hasattr(gt_ids, "tolist") else gt_ids)
            hit = len(result_ids & ground_truth_ids)
            results.append((hit, (start, end)))

        cursor.close()
        conn.close()
        return results

    def make_batch_args(self, test, answer, top, metric, table_name, benchmark,
                        warmup_n=0):
        rerank_op = self._get_metric_operator(metric)
        dim = test.shape[1]
        return (
            test,
            answer,
            top,
            rerank_op,
            self.url,
            table_name,
            benchmark["probes"],
            dim,
            benchmark.get("rerank_limit_amplify_factor", self.DEFAULT_RERANK_AMP),
            warmup_n,
        )

    def apply_session_guc(self, conn, benchmark):
        conn.execute(f"SET ivfflat.probes TO {benchmark['probes']}")

    def warmup_query(self, table_name, dataset, metric_ops, top, benchmark):
        dim = dataset["dim"]
        rerank_amp = (benchmark or {}).get(
            "rerank_limit_amplify_factor", self.DEFAULT_RERANK_AMP
        )
        rerank_limit = top * rerank_amp
        sql = self._two_stage_query_sql(table_name, dim, metric_ops, top)
        return sql, (lambda q: (q, rerank_limit, q))

    def create_index(self, suite_name: str, table_name: str, dataset: dict):
        """Create an IVFFlat expression index on binary_quantize(embedding)."""
        event, index_monitor_thread = super().create_index(
            suite_name, table_name, dataset
        )

        config = self.config[suite_name]
        pg_parallel_workers = config.get("pg_parallel_workers", 2)
        lists = config["lists"]
        if lists == "auto":
            lists = max(1, int(math.sqrt(dataset["num"])))
        dim = dataset["dim"]
        maintenance_work_mem = config.get("maintenance_work_mem")

        if self.debug:
            print(f"\n🔧 Index Configuration (IVFFlat BQ Rerank):")
            print(f"    • Lists:           {lists}")
            print(f"    • Dimensions:      {dim}")
            print()

        self.results[suite_name]["lists"] = lists

        conn = self.create_connection()
        start_time = time.perf_counter()

        if maintenance_work_mem:
            conn.execute(f"SET maintenance_work_mem TO '{maintenance_work_mem}'")
        conn.execute(f"SET max_parallel_maintenance_workers TO {pg_parallel_workers}")
        conn.execute(f"SET max_parallel_workers TO {pg_parallel_workers}")
        conn.execute(
            f"CREATE INDEX {table_name}_embedding_idx ON {table_name} "
            f"USING ivfflat ((binary_quantize(embedding)::bit({dim})) bit_hamming_ops) "
            f"WITH (lists = {lists})"
        )

        build_time = int(round(time.perf_counter() - start_time))
        self.results[suite_name]["index_build_time"] = build_time

        event.set()
        index_monitor_thread.join()

        print(f"Index build time: {build_time}s")

        conn.execute("CHECKPOINT")
        conn.close()
        print("Index built successfully.")

    def sequential_bench(
        self,
        name: str,
        table_name: str,
        conn: psycopg.Connection,
        metric: str,
        top: int,
        benchmark: dict,
        dataset: dict,
    ) -> tuple[list[tuple[int, float]], str]:
        """Run sequential benchmark with two-stage BQ rerank query."""
        conn.execute("SET jit=false")
        self.apply_session_guc(conn, benchmark)

        rerank_op = self._get_metric_operator(metric)
        dim = dataset["dim"]
        rerank_amp = benchmark.get(
            "rerank_limit_amplify_factor", self.DEFAULT_RERANK_AMP
        )
        rerank_limit = top * rerank_amp

        self.debug_log(
            f"Benchmark config: probes={benchmark['probes']}, "
            f"rerank_limit_amplify_factor={rerank_amp}, "
            f"metric={metric}, rerank_op={rerank_op}, dim={dim}"
        )

        self.warmup_for_benchmark(
            conn, table_name, dataset, rerank_op, top, name,
            benchmark=benchmark,
        )

        query_sql = self._two_stage_query_sql(table_name, dim, rerank_op, top)

        m = dataset["test"].shape[0]

        if self.debug_single_query:
            print(f"Running DEBUG single-query benchmark ({m} iterations of same query)")
            single_query = dataset["test"][0]
            single_answer = dataset["answer"][0][:top]
            if hasattr(single_answer, "tolist"):
                single_answer = single_answer.tolist()
        else:
            print(f"Running sequential benchmark with {m} queries")

        answers_list = dataset["answer"]
        if hasattr(answers_list, "tolist"):
            answers_list = [a[:top].tolist() if hasattr(a, "tolist") else a[:top] for a in answers_list]

        results = []
        latencies = []
        total_hits = 0
        total_time = 0.0

        cursor = conn.cursor()

        pbar = tqdm(range(m), total=m, ncols=80,
                    bar_format="{desc} {n}/{total}: {percentage:3.0f}%|{bar}|")
        for i in pbar:
            query = single_query if self.debug_single_query else dataset["test"][i]

            start = time.perf_counter()
            cursor.execute(query_sql, (query, rerank_limit, query))
            result = cursor.fetchall()
            end = time.perf_counter()

            query_time = end - start
            latencies.append(query_time)
            total_time += query_time

            if self.debug_single_query:
                answers = single_answer
            else:
                answers = answers_list[i] if isinstance(answers_list, list) else answers_list[i][:top]
                if hasattr(answers, "tolist"):
                    answers = answers.tolist()

            hit = len({p[0] for p in result[:top]} & set(answers))
            total_hits += hit
            results.append((hit, query_time))

            if (i + 1) % 50 == 0 or i == m - 1:
                curr_recall = total_hits / (top * (i + 1))
                curr_qps = (i + 1) / total_time
                curr_p50 = np.percentile(latencies, 50) * 1000
                recall_color = "\033[92m" if curr_recall >= 0.95 else "\033[91m"
                pbar.set_description(f"recall: {recall_color}{curr_recall:.4f}\033[0m QPS: {curr_qps:.2f} P50: {curr_p50:.2f}ms")

        cursor.close()
        pbar.close()
        return results, rerank_op

    def generate_markdown_result(self):
        """Generate benchmark results with charts and consolidated CSV."""
        self.debug_log(f"Results: {self.results}")

        results_manager = ResultsManager()

        for suite_name in self.config:
            system_metrics, pg_stats, dashboard_path = self.get_monitoring_data(suite_name)

            results_manager.process_suite_results(
                suite_type="ivfflat_bq_rerank",
                config={suite_name: self.config[suite_name]},
                results={suite_name: self.results.get(suite_name, {})},
                query_clients=self.query_clients,
                system_metrics=system_metrics,
                pg_stats=pg_stats,
                system_dashboard_path=dashboard_path,
            )


SUITE_DISPATCH = {
    "hnsw": HNSWTestSuite,
    "ivfflat": IVFFlatTestSuite,
    "ivfflat_bq_rerank": IVFFlatBQRerankTestSuite,
}


def main():
    """Main entry point for pgvector benchmark suite."""
    parser = build_arg_parse()
    args = parser.parse_args()

    config = common.load_suite_config(args.suite)
    first_suite = next(iter(config.values()))
    index_type = first_suite.get("indexType", "hnsw")
    if index_type not in SUITE_DISPATCH:
        raise ValueError(
            f"Unknown indexType {index_type!r}; "
            f"expected one of {sorted(SUITE_DISPATCH)}"
        )
    suite_cls = SUITE_DISPATCH[index_type]

    test_suite = suite_cls(
        suite_file=args.suite,
        url=args.url,
        devices=args.devices,
        chunk_size=args.chunk_size,
        skip_add_embeddings=args.skip_add_embeddings,
        centroids=args.centroids_file,
        centroids_table=args.centroids_table,
        skip_index_creation=args.skip_index_creation,
        query_clients=args.query_clients,
        max_load_threads=args.max_load_threads,
        debug=args.debug,
        overwrite_table=args.overwrite_table,
        debug_single_query=args.debug_single_query,
        build_only=args.build_only,
        max_queries=args.max_queries,
        warmup=args.warmup,
    )

    test_suite.run()
    print("Test suite completed.")


if __name__ == "__main__":
    main()
