import os
import json
import time
import argparse

from pathlib import Path
from langgraph.graph import StateGraph, END

from nodes import (
    data_loader_node,
    generation_node,
    injection_node,
    mutation_node
)

from state import AgentState
from config import DATA_PROJECT_DIR, CLONED_REPOS_DIR

# =============================================================================
# BROKEN TEST CACHE
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
BROKEN_FILE = (
    PROJECT_ROOT
    / "data"
    / "broken_tests.json"
)

# =============================================================================
# GRAPH SETUP
# =============================================================================

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

# =============================================================================
# HELPERS
# =============================================================================

def find_project_root(base_dir, project_keyword):
    """
    Finds the project root folder inside cloned repos.
    """

    project_keyword = project_keyword.lower()

    for root, dirs, files in os.walk(base_dir):

        for d in dirs:

            if d.lower() == project_keyword:
                return Path(root) / d

    return None


def is_valid_generated_assertion(assertion_text):
    """
    Filters obviously useless/generated garbage assertions.
    """

    if not assertion_text:
        return False

    cleaned = assertion_text.strip()

    invalid_patterns = [
        "assertTrue(true)",
        "assertFalse(false)",
        "assertNotNull(null)",
        "TODO",
        "expectedResource"
    ]

    for pattern in invalid_patterns:

        if pattern in cleaned:
            return False

    return True


# =============================================================================
# EVALUATION
# =============================================================================

def load_broken_tests(broken_file):

    if not broken_file.exists():

        return {}

    try:

        with open(
            broken_file,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return {}

def save_broken_tests(
    broken_file,
    broken_tests
):
    broken_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        broken_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            broken_tests,
            f,
            indent=4
        )


def build_ablation_variants():
    component_matrix = [
        (
            "None",
            False,
            False,
            False
        ),
        (
            "Summarizer",
            True,
            False,
            False
        ),
        (
            "Planner",
            False,
            True,
            False
        ),
        (
            "Refining Loop",
            False,
            False,
            True
        ),
        (
            "Summarizer + Planner",
            True,
            True,
            False
        ),
        (
            "Summarizer + Loop",
            True,
            False,
            True
        ),
        (
            "Planner + Loop",
            False,
            True,
            True
        ),
        (
            "Full Agent",
            True,
            True,
            True
        ),
    ]

    return [
        {
            "name": name,
            "run_mode": "agentic",
            "sum": use_summarizer,
            "plan": use_planner,
            "loop": use_loop
        }
        for (
            name,
            use_summarizer,
            use_planner,
            use_loop
        ) in component_matrix
    ]


def evaluate_datapoint(
    state,
    item_id,
    compiled_tests,
    pit_executed
):

    start = time.time()
    print(f"################ New datapoint #{compiled_tests}   #################")
    final = app.invoke(state)

    dur = time.time() - start

    if final.get("is_compiled"):

        compiled_tests += 1

    if (
        final.get("mutation_score") is not None
        or final.get(
            "pit_metrics",
            {}
        ).get(
            "generated",
            0
        ) > 0
    ):

        pit_executed += 1

    generated_assertions = str(
        final.get(
            "prediction",
            ""
        )
    )

    return (
        final,
        dur,
        generated_assertions,
        compiled_tests,
        pit_executed
    )


def is_zero_coverage_result(
    covered_mutants,
    pit_metrics
):
    return (
        covered_mutants == 0
        and pit_metrics.get(
            "generated",
            0
        ) > 0
    )


def is_non_evaluable_result(
    final,
    covered_mutants,
    pit_metrics
):
    if final.get("is_evaluable") is False:
        return True

    return (
        not final.get(
            "is_compiled",
            False
        )
        or covered_mutants is None
        or is_zero_coverage_result(
            covered_mutants,
            pit_metrics
        )
    )


def update_human_filtering(
    config,
    item_id,
    final,
    covered_mutants,
    pit_metrics,
    broken_tests
):

    if config["run_mode"] != "human":

        return

    is_broken = False

    if not final.get(
        "is_compiled",
        False
    ):

        is_broken = True

        print(
            "\n    >> [COMPILE FAILURE]"
        )

    elif is_zero_coverage_result(
        covered_mutants,
        pit_metrics
    ):

        is_broken = True

        print(
            "\n    >> [ZERO COVERAGE]"
        )

        print(
            f"    >> Marking as broken/non-evaluable: "
            f"{item_id}"
        )

    broken_tests[item_id] = is_broken



def extract_metrics(final):

    mutation_score = (
        final.get(
            "mutation_score",
            0.0
        )
        if final.get(
            "mutation_score"
        ) is not None
        else 0.0
    )

    test_strength = (
        final.get(
            "test_strength",
            0.0
        )
        if final.get(
            "test_strength"
        ) is not None
        else 0.0
    )

    covered_mutants = final.get(
        "covered_mutants",
        None
    )

    pit_metrics = final.get(
        "pit_metrics",
        {}
    )

    return (
        mutation_score,
        test_strength,
        covered_mutants,
        pit_metrics
    )


def run_evaluation(config, json_files, limit=None):

    total_test_strength = 0.0
    total_mutation_score = 0.0
    total_time = 0.0

    processed_count = 0
    completed_runs = 0

    skipped_broken = 0
    skipped_invalid = 0

    total_seen = 0
    split0_seen = 0
    compiled_tests = 0
    pit_executed = 0

    broken_tests = load_broken_tests(
        BROKEN_FILE
    )

    for dataset_path in json_files:

        if limit and processed_count >= limit:
            break

        with open(
            dataset_path,
            "r",
            encoding="utf-8"
        ) as f:

            raw_data = json.load(f)

        project_objs = (
            raw_data
            if isinstance(raw_data, list)
            else [raw_data]
        )

        for project_obj in project_objs:

            if limit and processed_count >= limit:
                break

            package_id = project_obj.get(
                "testClass",
                {}
            ).get(
                "packageIdentifier",
                ""
            )

            package_id_lower = str(
                package_id
            ).lower()

            if "twilio" in package_id_lower:

                keyword = "twilio-java"

            elif "liqp" in package_id_lower:

                keyword = "liqp"

            elif "zuul" in package_id_lower:

                keyword = "zuul"

            else:

                keyword = "dnsjava"

            repo_path = find_project_root(
                CLONED_REPOS_DIR,
                keyword
            )

            if not repo_path:
                continue

            proj_name = str(
                repo_path.relative_to(
                    CLONED_REPOS_DIR
                )
            )

            parent_focal_class = project_obj.get(
                "focalClass",
                {}
            )

            for dp in project_obj.get(
                "datapoints",
                []
            ):

                total_seen += 1

                if limit and processed_count >= limit:
                    break

                item_id = str(
                    dp.get(
                        "testPrefix",
                        {}
                    ).get(
                        "identifier",
                        "UNKNOWN"
                    )
                )

                if not item_id.endswith(
                    "_split_0"
                ):
                    continue

                split0_seen += 1

                print(
                    f"\n{'=' * 50}"
                )

                print(
                    f"[PROCESSING] "
                    f"{item_id}"
                )

                print(
                    f"{'=' * 50}\n"
                )

                if (
                    config["run_mode"]
                    != "human"
                ):

                    if broken_tests.get(
                        item_id,
                        False
                    ):

                        skipped_broken += 1

                        print(
                            f"    [SKIP BROKEN] "
                            f"{item_id}"
                        )

                        continue

                dp["_test_file_path"] = (
                    project_obj.get(
                        "testClass",
                        {}
                    ).get(
                        "filePath",
                        ""
                    ).lstrip("/")
                )

                dp["_project_name"] = (
                    proj_name
                )

                dp["_focalClass"] = (
                    parent_focal_class
                )

                dp["_testClass"] = (
                    project_obj.get(
                        "testClass",
                        {}
                    )
                )

                state = {
                    "raw_datapoint": dp,
                    "run_mode": config[
                        "run_mode"
                    ],
                    "use_summarizer": config[
                        "sum"
                    ],
                    "use_planner": config[
                        "plan"
                    ],
                    "use_evaluator_loop": config[
                        "loop"
                    ],
                    "iteration": 0,
                    "max_iterations": 4,
                    "feedback_history": [],
                    "best_score": 0.0,
                    "mutation_score": None,
                    "test_strength": None
                }

                (
                    final,
                    dur,
                    generated_assertions,
                    compiled_tests,
                    pit_executed
                ) = evaluate_datapoint(
                    state,
                    item_id,
                    compiled_tests,
                    pit_executed
                )

                if not is_valid_generated_assertion(
                    generated_assertions
                ):

                    skipped_invalid += 1

                    print(
                        f"    [SKIP INVALID] "
                        f"{item_id}"
                    )

                    continue

                (
                    mutation_score,
                    test_strength,
                    covered_mutants,
                    pit_metrics
                ) = extract_metrics(
                    final
                )

                update_human_filtering(
                    config,
                    item_id,
                    final,
                    covered_mutants,
                    pit_metrics,
                    broken_tests
                )

                if is_non_evaluable_result(
                    final,
                    covered_mutants,
                    pit_metrics
                ):

                    skipped_broken += 1

                    continue

                processed_count += 1

                completed_runs += 1

                total_time += dur

                total_test_strength += (
                    test_strength
                )

                total_mutation_score += (
                    mutation_score
                )

                running_avg_ts = (
                    total_test_strength
                    / completed_runs
                )

                running_avg_ms = (
                    total_mutation_score
                    / completed_runs
                )

                print(
                    f"    [{processed_count}] "
                    f"{item_id} "
                    f"| Test Strength: "
                    f"{test_strength:.4f} "
                    f"| Mutation Score: "
                    f"{mutation_score:.4f} "
                    f"| Covered Mutants: "
                    f"{covered_mutants} "
                    f"| Running Avg TS: "
                    f"{running_avg_ts:.4f} "
                    f"| Running Avg MS: "
                    f"{running_avg_ms:.4f} "
                    f"| Time: "
                    f"{dur:.2f}s"
                )

    if config["run_mode"] == "human":

        save_broken_tests(
            BROKEN_FILE,
            broken_tests
        )

    final_avg_ts = (
        total_test_strength
        / completed_runs
        if completed_runs > 0
        else 0.0
    )

    final_avg_ms = (
        total_mutation_score
        / completed_runs
        if completed_runs > 0
        else 0.0
    )

    avg_time = (
        total_time
        / processed_count
        if processed_count > 0
        else 0.0
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "RUN SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        f"Total JSON Datapoints: "
        f"{total_seen}"
    )

    print(
        f"split_0 Datapoints Seen: "
        f"{split0_seen}"
    )

    print(
        f"Tests Successfully Compiled: "
        f"{compiled_tests}"
    )

    print(
        f"PIT Successfully Executed: "
        f"{pit_executed}"
    )

    print(
        f"Completed Runs: "
        f"{completed_runs}"
    )

    print(
        f"Skipped Broken: "
        f"{skipped_broken}"
    )

    print(
        f"Skipped Invalid: "
        f"{skipped_invalid}"
    )

    print(
        f"Average Test Strength: "
        f"{final_avg_ts:.4f}"
    )

    print(
        f"Average Mutation Score: "
        f"{final_avg_ms:.4f}"
    )

    print(
        f"Average Runtime: "
        f"{avg_time:.2f}s"
    )

    return (
        final_avg_ts,
        final_avg_ms,
        avg_time,
        completed_runs
    )



# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="LLM Test Evaluator"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "human",
            "oneshot",
            "agentic",
            "ablation"
        ],
        default="human"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit datapoints"
    )

    args = parser.parse_args()

    base_dataset_dir = (
        DATA_PROJECT_DIR
        / "scripts"
        / "dataset"
        / "output"
        / "raw-oracles-dataset"
    )

    json_files = list(
        base_dataset_dir.rglob(
            "oracles-datapoints-*.json"
        )
    )

    if args.mode == "ablation":

        variants = build_ablation_variants()

        results = {}

        for v in variants:

            print(
                f"\n>>> RUNNING ABLATION: "
                f"{v['name']}"
                f" | summarizer={v['sum']}"
                f" planner={v['plan']}"
                f" loop={v['loop']}"
            )

            avg_s, avg_m, avg_t, completed = run_evaluation(
                v,
                json_files,
                args.limit
            )

            results[v["name"]] = {
                "avg_test_strength": avg_s,
                "avg_mutation_score": avg_m,
                "avg_time": avg_t,
                "completed": completed,
                "summarizer": v["sum"],
                "planner": v["plan"],
                "loop": v["loop"],
            }

        print(
            "\n"
            + "=" * 60
            + "\n FINAL ABLATION RESULTS "
            + "\n"
            + "=" * 60
        )

        for name, metrics in results.items():

            print(
                f"{name:<22} "
                f"| S={int(metrics['summarizer'])} "
                f"P={int(metrics['planner'])} "
                f"L={int(metrics['loop'])} "
                f"| Avg Test Strength: "
                f"{metrics['avg_test_strength']:.4f} "
                f"| Avg Mutation Score: "
                f"{metrics['avg_mutation_score']:.4f} "
                f"| Valid Runs: "
                f"{metrics['completed']} "
                f"| Avg Time: "
                f"{metrics['avg_time']:.2f}s"
            )

    else:

        config = {
            "run_mode": args.mode,
            "sum": (args.mode == "agentic"),
            "plan": (args.mode == "agentic"),
            "loop": (args.mode == "agentic")
        }

        print(
            f"\n>>> STARTING BATCH RUN "
            f"| MODE: {args.mode.upper()}"
        )

        avg_s, avg_m, avg_t, completed = run_evaluation(
            config,
            json_files,
            args.limit
        )

        print(
            f"\nDONE. "
            f"Final Avg Test Strength: "
            f"{avg_s:.4f} "
            f"| Final Avg Mutation Score: "
            f"{avg_m:.4f} "
            f"| Valid Runs: {completed} "
            f"| Avg Time: {avg_t:.2f}s"
        )

if __name__ == "__main__":
    main()
