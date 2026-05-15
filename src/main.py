import os
import json
import time
import argparse
from pathlib import Path
from langgraph.graph import StateGraph, END

from nodes import data_loader_node, generation_node, injection_node, mutation_node
from state import AgentState
from config import DATA_PROJECT_DIR, CLONED_REPOS_DIR

# GRAPH SETUP
workflow = StateGraph(AgentState)
workflow.add_node("data_loader", data_loader_node)
workflow.add_node("generation", generation_node)
workflow.add_node("injection", injection_node)
workflow.add_node("mutation", mutation_node)

workflow.set_entry_point("data_loader")
workflow.add_edge("data_loader", "generation")
workflow.add_edge("generation", "injection")
workflow.add_edge("injection", "mutation")
workflow.add_edge("mutation", END)

app = workflow.compile()

def find_project_root(base_dir, project_keyword):
    project_keyword = project_keyword.lower()
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            if d.lower() == project_keyword:
                return Path(root) / d
    return None

def run_evaluation(config, json_files, limit=None):
    """Helper to run a specific configuration and return metrics."""
    total_test_strength = 0.0
    total_time = 0.0
    
    processed_count = 0
    completed_runs = 0
    quarantined_runs = 0

    # Cache setup
    cache_file = DATA_PROJECT_DIR / "scripts" / "dataset" / "output" / "compilation_cache.json"
    compile_cache = {}
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            compile_cache = json.load(f)

    for dataset_path in json_files:
        if limit and processed_count >= limit: break
        with open(dataset_path, "r", encoding="utf-8") as f:    
            raw_data = json.load(f)
        project_objs = raw_data if isinstance(raw_data, list) else [raw_data]
        
        for project_obj in project_objs:
            if limit and processed_count >= limit: break
            
            package_id = project_obj.get("testClass", {}).get("packageIdentifier", "")
            if "twilio" in str(package_id).lower(): keyword = "twilio-java"
            elif "liqp" in str(package_id).lower(): keyword = "liqp"
            elif "zuul" in str(package_id).lower(): keyword = "zuul"
            else: keyword = "dnsjava"

            repo_path = find_project_root(CLONED_REPOS_DIR, keyword)
            if not repo_path: continue
            proj_name = str(repo_path.relative_to(CLONED_REPOS_DIR))

            # Extract the focal class from the parent object
            parent_focal_class = project_obj.get("focalClass", {})

            for dp in project_obj.get("datapoints", []):
                if limit and processed_count >= limit: break

                # --- TEMPORARY SKIP: Remove or comment out to run all tests ---
                if processed_count < 7:
                    processed_count += 1
                    continue
                
                item_id = str(dp.get("testPrefix", {}).get("identifier", "UNKNOWN"))

                # --- THE SKIP LOGIC ---
                # If we are NOT in human mode, check the cache. 
                # If the human baseline didn't compile, skip this datapoint.
                if config["run_mode"] != "human":
                    if not compile_cache.get(item_id, False):
                        print(f"    [SKIP] {item_id} (Human baseline failed to compile)")
                        continue

                dp["_test_file_path"] = project_obj.get("testClass", {}).get("filePath", "").lstrip("/")
                dp["_project_name"] = proj_name
                
                # Attach the focal class to the individual datapoint
                dp["_focalClass"] = parent_focal_class
                
                state = {
                    "raw_datapoint": dp,
                    "run_mode": config["run_mode"],
                    "use_summarizer": config["sum"],
                    "use_planner": config["plan"],
                    "use_evaluator_loop": config["loop"],
                    "iteration": 0,
                    "max_iterations": 3,
                    "feedback_history": [],
                    "best_score": 0.0
                }

                start = time.time()
                final = app.invoke(state)
                dur = time.time() - start

                # --- THE SAVE LOGIC ---
                # If we ARE in human mode, record if it compiled and wasn't quarantined
                if config["run_mode"] == "human":
                    # It's successful if injection compiled it AND mutation node didn't quarantine it
                    is_valid = final.get("is_compiled", False) and not final.get("is_quarantined", False)
                    compile_cache[item_id] = is_valid
                    
                    # Save to disk immediately so progress isn't lost if the script crashes
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(compile_cache, f, indent=4)

                # --- METRICS & PRINTING ---
                processed_count += 1
                total_time += dur
                
                is_quarantined = final.get("is_quarantined", False)
                score = final.get("mutation_score", 0.0) if final.get("mutation_score") else 0.0
                
                if is_quarantined:
                    quarantined_runs += 1
                    print(f"    [{processed_count}] {item_id} | QUARANTINED | Time: {dur:.2f}s")
                else:
                    completed_runs += 1
                    total_test_strength += score
                    running_avg = total_test_strength / completed_runs
                    
                    print(f"    [{processed_count}] {item_id} | Test Strength: {score:.4f} | Running Avg: {running_avg:.4f} | Time: {dur:.2f}s")

    final_avg = (total_test_strength / completed_runs) if completed_runs > 0 else 0.0
    avg_time = (total_time / processed_count) if processed_count > 0 else 0.0
    
    return final_avg, avg_time, completed_runs

def main():
    parser = argparse.ArgumentParser(description="LLM Test Evaluator")
    parser.add_argument("--mode", type=str, choices=["human", "oneshot", "agentic", "ablation"], default="human")
    parser.add_argument("--limit", type=int, default=None, help="Limit datapoints (useful for ablation)")
    args = parser.parse_args()

    base_dataset_dir = DATA_PROJECT_DIR / "scripts" / "dataset" / "output" / "raw-oracles-dataset"
    json_files = list(base_dataset_dir.rglob("oracles-datapoints-*.json"))

    if args.mode == "ablation":
        variants = [
            {"name": "V2 (Loop Only)", "run_mode": "agentic", "sum": False, "plan": False, "loop": True},
            {"name": "V3 (Thought Only)", "run_mode": "agentic", "sum": True, "plan": True, "loop": False},
            {"name": "V4 (Full Agent)", "run_mode": "agentic", "sum": True, "plan": True, "loop": True},
        ]
        results = {}
        for v in variants:
            print(f"\n>>> RUNNING ABLATION: {v['name']}")
            avg_s, avg_t, completed = run_evaluation(v, json_files, args.limit)
            results[v['name']] = (avg_s, avg_t, completed)

        print("\n" + "="*60 + "\n FINAL ABLATION RESULTS (TEST STRENGTH) \n" + "="*60)
        for name, metrics in results.items():
            print(f"{name:<20} | Avg Test Strength: {metrics[0]:.4f} | Valid Runs: {metrics[2]} | Avg Time: {metrics[1]:.2f}s")
    
    else:
        # Standard Single Run
        config = {
            "run_mode": args.mode,
            "sum": (args.mode == "agentic"),
            "plan": (args.mode == "agentic"),
            "loop": (args.mode == "agentic")
        }
        print(f"\n>>> STARTING BATCH RUN | MODE: {args.mode.upper()}")
        avg_s, avg_t, completed = run_evaluation(config, json_files, args.limit)
        print(f"\nDONE. Final Avg Test Strength: {avg_s:.4f} | Valid Runs: {completed} | Avg Time: {avg_t:.2f}s")

if __name__ == "__main__":
    main()