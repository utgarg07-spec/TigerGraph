"""
Benchmark Runner for GraphRAG Evaluation Harness (Phase 2).

No official evaluator was found in the repository; strict_exact_match is the reproducible internal benchmark metric.
Evaluates the GraphRAG /query API against data/eval_public.jsonl without modifying underlying retrieval algorithms or agent logic.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests


def load_local_config(config_path: str = "configs/local_server_config.json") -> Dict[str, Any]:
    """Loads TigerGraph/GraphRAG local server configuration."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        cfg_file = Path("configs/server_config.json")
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(cfg_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_olympic_doc_ids(events_path: str = "data/processed/events.jsonl") -> set:
    """Loads the set of verified Olympic-event doc_ids for intrusion calculation."""
    events_file = Path(events_path)
    if not events_file.exists():
        print(f"Warning: {events_path} not found. Intrusion metrics will be disabled.")
        return set()
    
    olympic_ids = set()
    with open(events_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("doc_id"):
                olympic_ids.add(rec["doc_id"].upper())
    return olympic_ids


def load_evaluation_dataset(eval_path: str = "data/eval_public.jsonl") -> List[Dict[str, Any]]:
    """Loads and validates evaluation dataset records."""
    eval_file = Path(eval_path)
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")

    records = []
    qids = set()
    duplicate_qids = []
    missing_answers = []

    with open(eval_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = rec.get("qid")
            if qid in qids:
                duplicate_qids.append(qid)
            else:
                qids.add(qid)
            
            ans = rec.get("answer")
            if ans is None or ans == "" or (isinstance(ans, list) and len(ans) == 0):
                missing_answers.append(qid)
                
            records.append(rec)

    if duplicate_qids:
        raise ValueError(f"Data Integrity Error: Duplicate QIDs found in {eval_path}: {duplicate_qids}")
    if missing_answers:
        raise ValueError(f"Data Integrity Error: Missing gold answers found in {eval_path}: {missing_answers}")

    return records


def extract_retrieval_info(query_sources: Dict[str, Any], olympic_doc_ids: set) -> Tuple[List[str], List[str], Optional[float]]:
    """
    Extracts retrieved chunk IDs, source document IDs, and calculates retrieval_intrusion_at_k.
    """
    if not query_sources or not isinstance(query_sources, dict):
        return [], [], None

    agent_steps = query_sources.get("agent_steps", [])
    retrieved_chunk_ids = []
    retrieved_doc_ids = []

    for step in agent_steps:
        if not isinstance(step, dict):
            continue
        output_str = step.get("output", "")
        if not output_str or not isinstance(output_str, str):
            continue
        
        try:
            output_json = json.loads(output_str)
            ctx = output_json.get("context", {})
            if isinstance(ctx, dict):
                res = ctx.get("result", {})
                if isinstance(res, dict):
                    fin_ret = res.get("final_retrieval", {})
                    if isinstance(fin_ret, dict):
                        retrieved_chunk_ids.extend(list(fin_ret.keys()))
        except Exception:
            pass

    if not retrieved_chunk_ids:
        qs_str = json.dumps(query_sources)
        matches = re.findall(r'"(q\d+_chunk_\d+)"', qs_str)
        retrieved_chunk_ids = list(set(matches))

    doc_id_set = set()
    for chunk_id in retrieved_chunk_ids:
        match = re.match(r'^(q\d+)_chunk_', chunk_id, re.IGNORECASE)
        if match:
            doc_id = match.group(1).upper()
            doc_id_set.add(doc_id)

    retrieved_doc_ids = sorted(list(doc_id_set))

    intrusion_pct = None
    if retrieved_doc_ids and olympic_doc_ids:
        non_olympic_count = sum(1 for d in retrieved_doc_ids if d not in olympic_doc_ids)
        intrusion_pct = round((non_olympic_count / len(retrieved_doc_ids)) * 100, 2)

    return retrieved_chunk_ids, retrieved_doc_ids, intrusion_pct


def compute_normalized_match(pred_str: str, target_gold: str) -> bool:
    """
    Diagnostic matching rule:
    - Removes Markdown emphasis markers (** * __)
    - Lowercases both strings
    - Checks whether canonical gold answer string occurs within normalized prediction
    """
    if not pred_str or not target_gold:
        return False
    clean_pred = re.sub(r'\*\*|\*|__', '', pred_str).lower()
    clean_gold = re.sub(r'\*\*|\*|__', '', target_gold).lower()
    return clean_gold in clean_pred


def run_benchmark(
    eval_path: str = "data/eval_public.jsonl",
    graph_name: str = "Olympics",
    base_url: str = "http://localhost:8000",
    mode: str = "classic",
    rag_method: str = "similaritysearch",
    pass_qtype: bool = False,
    selected_qid: Optional[str] = None,
    selected_qtype: Optional[str] = None,
    limit: Optional[int] = None,
    run_name: str = "smoke_test",
    output_dir: str = "benchmark/results",
) -> Dict[str, Any]:
    """
    Executes benchmark runner over specified evaluation questions.
    """
    cfg = load_local_config()
    db_cfg = cfg.get("db_config", {})
    username = db_cfg.get("username", "__GSQL__secret")
    password = db_cfg.get("password", "")

    auth_str = base64.b64encode(f"{username}:{password}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_str}",
        "Content-Type": "application/json",
    }

    olympic_doc_ids = load_olympic_doc_ids()
    eval_records = load_evaluation_dataset(eval_path)

    filtered_records = []
    for rec in eval_records:
        if selected_qid and rec["qid"] != selected_qid:
            continue
        if selected_qtype and rec["qtype"] != selected_qtype:
            continue
        filtered_records.append(rec)

    if limit is not None and limit > 0:
        filtered_records = filtered_records[:limit]

    print(f"Starting benchmark run '{run_name}' with {len(filtered_records)} questions...")
    print(f"Configuration: mode={mode}, rag_method={rag_method}, pass_qtype={pass_qtype}\n")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    jsonl_output_path = Path(output_dir) / f"{run_name}.jsonl"
    summary_output_path = Path(output_dir) / f"{run_name}_summary.json"

    results = []
    qtype_counts = {}
    qtype_strict = {}
    qtype_norm = {}

    target_url = f"{base_url.rstrip('/')}/{graph_name}/query"

    with open(jsonl_output_path, "w", encoding="utf-8") as f_out:
        for idx, rec in enumerate(filtered_records, 1):
            qid = rec["qid"]
            question = rec["question"]
            qtype = rec["qtype"]
            gold_answers = rec.get("answer", [])
            gold_str = gold_answers[0] if isinstance(gold_answers, list) and len(gold_answers) > 0 else str(gold_answers)

            payload = {
                "query": question,  # Preserve natural language question untouched
                "mode": mode,
                "rag_method": rag_method,
                "include_fields": ["all"],
            }
            if pass_qtype:
                payload["qtype"] = qtype

            start_time = time.time()
            prediction = ""
            error_info = None
            resp_data = {}

            try:
                response = requests.post(target_url, json=payload, headers=headers, timeout=120)
                latency = round(time.time() - start_time, 3)

                if response.status_code == 200:
                    resp_data = response.json()
                    prediction = resp_data.get("natural_language_response", "")
                else:
                    error_info = f"HTTP {response.status_code}: {response.text}"
            except Exception as e:
                latency = round(time.time() - start_time, 3)
                error_info = f"Request Exception: {str(e)}"

            # Scoring:
            # 1. Primary Metric: strict_exact_match (exact string equality after trimming whitespace)
            norm_pred = prediction.strip() if prediction else ""
            norm_gold = gold_str.strip() if gold_str else ""
            strict_exact_match = (norm_pred == norm_gold)

            # 2. Diagnostic Metric: normalized_match
            normalized_match = compute_normalized_match(prediction, gold_str)

            query_sources = resp_data.get("query_sources", {}) if isinstance(resp_data, dict) else {}
            token_usage = query_sources.get("token_usage") if isinstance(query_sources, dict) else None
            citations = query_sources.get("citations") if isinstance(query_sources, dict) else None
            agent_steps = query_sources.get("agent_steps") if isinstance(query_sources, dict) else None
            
            plan = None
            if isinstance(agent_steps, list):
                for step in agent_steps:
                    if isinstance(step, dict) and step.get("node") == "plan_question":
                        plan = step.get("output")

            chunk_ids, doc_ids, intrusion_pct = extract_retrieval_info(query_sources, olympic_doc_ids)

            result_record = {
                "qid": qid,
                "qtype": qtype,
                "question": question,
                "prediction": prediction,
                "gold": gold_str,
                "gold_doc_ids": rec.get("gold_doc_ids", []),
                "strict_exact_match": strict_exact_match,
                "normalized_match": normalized_match,
                "correct": strict_exact_match,  # Alias for backward compatibility
                "latency": latency,
                "token_usage": token_usage,
                "retrieved_chunk_ids": chunk_ids,
                "retrieved_doc_ids": doc_ids,
                "retrieval_intrusion_at_k": intrusion_pct,
                "citations": citations,
                "agent_steps": agent_steps,
                "plan": plan,
                "error": error_info,
            }

            f_out.write(json.dumps(result_record, ensure_ascii=False) + "\n")
            f_out.flush()
            results.append(result_record)

            if qtype not in qtype_counts:
                qtype_counts[qtype] = 0
                qtype_strict[qtype] = 0
                qtype_norm[qtype] = 0
            qtype_counts[qtype] += 1
            if strict_exact_match:
                qtype_strict[qtype] += 1
            if normalized_match:
                qtype_norm[qtype] += 1

            status_str = "EXACT_MATCH" if strict_exact_match else ("NORM_MATCH" if normalized_match else "FAIL")
            err_str = f" [ERROR: {error_info}]" if error_info else ""
            print(f"[{idx}/{len(filtered_records)}] QID: {qid} | QType: {qtype} | Result: {status_str} | Latency: {latency}s{err_str}")

    total_attempted = len(results)
    total_successful = sum(1 for r in results if r["error"] is None)
    total_failed = sum(1 for r in results if r["error"] is not None)
    
    total_strict_correct = sum(1 for r in results if r["strict_exact_match"])
    strict_accuracy = round((total_strict_correct / total_attempted) * 100, 2) if total_attempted > 0 else 0.0

    total_norm_correct = sum(1 for r in results if r["normalized_match"])
    norm_accuracy = round((total_norm_correct / total_attempted) * 100, 2) if total_attempted > 0 else 0.0

    http_success_rate = round((total_successful / total_attempted), 4) if total_attempted > 0 else 0.0
    avg_latency = round(sum(r["latency"] for r in results) / total_attempted, 3) if total_attempted > 0 else 0.0
    
    intrusions = [r["retrieval_intrusion_at_k"] for r in results if r["retrieval_intrusion_at_k"] is not None]
    avg_intrusion = round(sum(intrusions) / len(intrusions), 2) if intrusions else None

    qtype_summary = {}
    for qt, count in qtype_counts.items():
        s_cnt = qtype_strict[qt]
        n_cnt = qtype_norm[qt]
        qtype_summary[qt] = {
            "attempted": count,
            "strict_exact_match_count": s_cnt,
            "strict_exact_match_pct": round((s_cnt / count) * 100, 2),
            "normalized_match_count": n_cnt,
            "normalized_match_pct": round((n_cnt / count) * 100, 2),
        }

    summary_report = {
        "run_name": run_name,
        "number_attempted": total_attempted,
        "number_successful": total_successful,
        "number_failed": total_failed,
        "http_success_rate": http_success_rate,
        "strict_exact_match": strict_accuracy,
        "normalized_match": norm_accuracy,
        "avg_latency_seconds": avg_latency,
        "average_retrieval_intrusion_pct": avg_intrusion,
        "qtype_summary": qtype_summary,
        "pipeline_configuration": {
            "mode": mode,
            "rag_method": rag_method,
            "pass_qtype": pass_qtype,
            "graph_name": graph_name,
        },
    }

    with open(summary_output_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary_report, f_sum, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("BENCHMARK RUN SUMMARY")
    print("=" * 60)
    print(f"Run Name:            {run_name}")
    print(f"Attempted:           {total_attempted}")
    print(f"HTTP Success Rate:   {http_success_rate * 100:.2f}% ({total_successful}/{total_attempted})")
    print(f"strict_exact_match:  {total_strict_correct} / {total_attempted} ({strict_accuracy}%) [Primary Metric]")
    print(f"normalized_match:    {total_norm_correct} / {total_attempted} ({norm_accuracy}%) [Diagnostic Only]")
    print(f"Avg Latency Seconds: {avg_latency}s")
    print(f"Average Intrusion:   {avg_intrusion}%" if avg_intrusion is not None else "Average Intrusion: N/A")
    print("\nAccuracy by QType:")
    for qt, stats in qtype_summary.items():
        print(f"  - {qt:20s}: strict={stats['strict_exact_match_count']}/{stats['attempted']} ({stats['strict_exact_match_pct']}%) | norm={stats['normalized_match_count']}/{stats['attempted']} ({stats['normalized_match_pct']}%)")
    print("=" * 60)
    print(f"Detailed results: {jsonl_output_path}")
    print(f"Summary report:   {summary_output_path}")
    print("=" * 60)

    return summary_report


def main():
    parser = argparse.ArgumentParser(description="GraphRAG Benchmark Runner (Phase 2)")
    parser.add_argument("--all", action="store_true", help="Run all questions in dataset")
    parser.add_argument("--qid", type=str, default=None, help="Run specific QID (e.g. pub-001)")
    parser.add_argument("--qtype", type=str, default=None, help="Filter by qtype (e.g. aggregation)")
    parser.add_argument("--limit", type=int, default=None, help="Limit max questions to process")
    parser.add_argument("--mode", type=str, default="classic", choices=["classic", "agentic"], help="Query mode")
    parser.add_argument("--rag-method", type=str, default="similaritysearch", choices=["similaritysearch", "hybridsearch"], help="RAG method")
    parser.add_argument("--pass-qtype", action="store_true", help="Pass qtype field in POST payload to /query")
    parser.add_argument("--run-name", type=str, default="smoke_test", help="Custom name for benchmark run output files")

    args = parser.parse_args()

    run_benchmark(
        mode=args.mode,
        rag_method=args.rag_method,
        pass_qtype=args.pass_qtype,
        selected_qid=args.qid,
        selected_qtype=args.qtype,
        limit=args.limit,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
