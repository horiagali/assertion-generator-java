import os
import re
import csv
import json
import shutil
import subprocess
import time
import traceback
from generators.agentic_logic import (
    execute_sandbox,
    inject_prediction_into_test_file
)
from typing import Dict
from pathlib import Path
from sandbox_utils import cleanup_star_files
from config import (
    MUTATION_JAR_PATH,
    JAVA_HOME,
    ORACLE_CONFIG_JSON,
    SANDBOX_DIR,
    CLONED_REPOS_DIR,
    DATA_PROJECT_DIR
)

from state import AgentState

from generators.human_logic import run_human_logic
from generators.oneshot_logic import run_oneshot_logic
from generators.agentic_logic import run_agentic_logic


# =============================================================================
# BROKEN TEST CACHE
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

BROKEN_FILE = (
    PROJECT_ROOT
    / "data"
    / "broken_tests.json"
)


def load_broken_tests():

    if BROKEN_FILE.exists():

        with open(
            BROKEN_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    return {}


def save_broken_test(item_id):

    broken = load_broken_tests()

    broken[item_id] = True

    with open(
        BROKEN_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            broken,
            f,
            indent=4
        )


def remove_broken_test(item_id):

    broken = load_broken_tests()

    if item_id in broken:

        del broken[item_id]

        with open(
            BROKEN_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                broken,
                f,
                indent=4
            )






# =============================================================================
# DATA LOADER
# =============================================================================
def data_loader_node(state: AgentState) -> Dict:

    import json

    dp = state.get("raw_datapoint")

    test_prefix = dp.get("testPrefix", {})

    item_id = str(
        test_prefix.get(
            "identifier",
            "UNKNOWN"
        )
    )

    # =========================================================
    # ONLY PROCESS split_0
    # SKIP FIRST TEST FOR DEBUGGING
    # =========================================================

    if not item_id.endswith("_split_0"):

        print(
            f"    >> [SKIP NON-SPLIT0] "
            f"{item_id}"
        )

        return {
            "item_id": item_id,
            "is_broken": True
        }

    # =========================================================
    # BASIC METADATA
    # =========================================================

    project_name = dp.get(
        "_project_name",
        "dnsjava"
    )

    project_id = str(
        project_name
    ).split("/")[-1]

    file_path = dp.get(
        "_test_file_path",
        ""
    )

    focal_class = dp.get(
        "_focalClass",
        {}
    )

    test_class = dp.get(
        "_testClass",
        {}
    )

    broken_tests = load_broken_tests()

    is_broken = False

    # if (
    #     state.get("run_mode") != "human"
    #     and broken_tests.get(item_id, False)
    # ):

    #     print(
    #         f"    >> [BROKEN TEST SKIPPED] "
    #         f"{item_id}"
    #     )

    #     is_broken = True

    # =========================================================
    # TEST BODY
    # =========================================================

    test_signature = test_prefix.get(
        "signature",
        ""
    )

    test_body = test_prefix.get(
        "body",
        ""
    )

    invoked_methods = test_prefix.get(
        "invokedMethods",
        []
    ) or []

    # =========================================================
    # FULL RAW CONTEXT
    # ONLY SUMMARIZER SEES THIS
    # =========================================================

    full_context_json = {

        "project_name": project_name,

        "testPrefix": {

            "identifier": test_prefix.get(
                "identifier"
            ),

            "signature": test_signature,

            "body": test_body,

            "invokedMethods": invoked_methods
        },

        "testClass": {

            "identifier": test_class.get(
                "identifier"
            ),

            "packageIdentifier": test_class.get(
                "packageIdentifier"
            ),

            "filePath": test_class.get(
                "filePath"
            ),

            "fields": test_class.get(
                "fields",
                []
            ),

            "setupTearDownMethods": test_class.get(
                "setupTearDownMethods",
                []
            ),

            "auxiliaryMethods": test_class.get(
                "auxiliaryMethods",
                []
            )
        },

        "_focalClass": {

            "identifier": focal_class.get(
                "identifier"
            ),

            "packageIdentifier": focal_class.get(
                "packageIdentifier"
            ),

            "superclasses": focal_class.get(
                "superclasses"
            ),

            "interfaces": focal_class.get(
                "interfaces"
            ),

            "fields": focal_class.get(
                "fields",
                []
            ),

            "constructors": focal_class.get(
                "constructors",
                []
            ),

            # =================================================
            # NO FILTERING
            # SUMMARIZER DECIDES RELEVANCE
            # =================================================

            "methods": focal_class.get(
                "methods",
                []
            )
        }
    }

    # =========================================================
    # PREVENT TARGET LEAKAGE
    # =========================================================

    full_context_json.pop(
        "target",
        None
    )

    # =========================================================
    # FULL PROMPT CONTEXT
    # ONLY FOR SUMMARIZER
    # =========================================================

    full_prompt_context = (
        "You are given structured Java semantic metadata in JSON form.\n"
        "Use ALL available information.\n"
        "Do NOT invent unavailable variables or APIs.\n\n"
        + json.dumps(
            full_context_json,
            indent=2,
            default=str
        )
    )

    # =========================================================
    # COMPACT EXECUTABLE CONTEXT
    # FOR CODER / PLANNER / IMPROVER
    # =========================================================

    focal_method_signatures = []

    for method in focal_class.get(
        "methods",
        []
    ):

        signature = method.get(
            "signature",
            ""
        )

        if signature:

            focal_method_signatures.append(
                f"- {signature}"
            )

    constructor_lines = []

    for ctor in focal_class.get(
        "constructors",
        []
    ):

        signature = ctor.get(
            "signature",
            ""
        )

        if signature:

            constructor_lines.append(
                f"- {signature}"
            )

    field_lines = []

    for field in focal_class.get(
        "fields",
        []
    ):

        if isinstance(field, str):

            field_lines.append(
                f"- {field}"
            )

        elif isinstance(field, dict):

            signature = field.get(
                "signature",
                ""
            )

            if signature:

                field_lines.append(
                    f"- {signature}"
                )

    test_field_lines = []

    for field in test_class.get(
        "fields",
        []
    ):

        if isinstance(field, str):

            test_field_lines.append(
                f"- {field}"
            )

        elif isinstance(field, dict):

            signature = field.get(
                "signature",
                ""
            )

            if signature:

                test_field_lines.append(
                    f"- {signature}"
                )

    setup_lines = []

    for method in test_class.get(
        "setupTearDownMethods",
        []
    ):

        if isinstance(method, dict):

            signature = method.get(
                "signature",
                ""
            )

            body = method.get(
                "body",
                ""
            )

            if signature or body:

                setup_lines.append(
                    f"- {signature}\n{body}".strip()
                )

    helper_lines = []

    for method in test_class.get(
        "auxiliaryMethods",
        []
    ):

        if isinstance(method, dict):

            signature = method.get(
                "signature",
                ""
            )

            return_type = method.get(
                "returnType",
                ""
            )

            if signature:

                helper_lines.append(
                    f"- {signature}"
                    + (
                        f" -> {return_type}"
                        if return_type
                        else ""
                    )
                )

    compact_prompt_context = f"""
TARGET TEST METHOD:
{test_body}

TEST CLASS FIELDS:
{chr(10).join(test_field_lines)}

SETUP / TEARDOWN METHODS:
{chr(10).join(setup_lines)}

TEST CLASS HELPER METHODS:
{chr(10).join(helper_lines)}

VISIBLE FOCAL METHODS:
{chr(10).join(focal_method_signatures)}

VISIBLE CONSTRUCTORS:
{chr(10).join(constructor_lines)}

VISIBLE FIELDS:
{chr(10).join(field_lines)}
""".strip()

    # =========================================================
    # DEBUG PRINTS
    # =========================================================

    # print(
    #     "\n=================================================="
    # )

    # print(
    #     f"[PROCESSING] {item_id}"
    # )

    # print(
    #     "==================================================\n"
    # )

    # print(
    #     "\n========= COMPACT EXECUTABLE CONTEXT =========\n"
    # )

    # print(compact_prompt_context)

    # print(
    #     "\n==============================================\n"
    # )

    # print(
    #     "\n========= FULL SUMMARIZER CONTEXT =========\n"
    # )

    # print(
    #     f"[DEBUG] Full context chars = "
    #     f"{len(full_prompt_context)}"
    # )

    # print(
    #     f"[DEBUG] Total methods exposed to summarizer = "
    #     f"{len(focal_class.get('methods', []))}"
    # )

    # print(
    #     f"[DEBUG] Total constructors exposed = "
    #     f"{len(focal_class.get('constructors', []))}"
    # )

    # print(
    #     f"[DEBUG] Total fields exposed = "
    #     f"{len(focal_class.get('fields', []))}"
    # )

    # =========================================================
    # RETURN STATE
    # =========================================================

    return {

        "item_id": item_id,

        "project_name": project_name,

        "project_id": project_id,

        "info_file_path": dp.get(
            "_source_file_path"
        ),

        "full_prompt_context": full_prompt_context,

        "compact_prompt_context": compact_prompt_context,

        "prompt_context": full_prompt_context,

        "ground_truth": dp.get(
            "target"
        ),

        "file_path": file_path,

        "method_signature": test_signature,

        "is_broken": is_broken
    }
    
# =============================================================================
# GENERATION
# =============================================================================

def generation_node(state: AgentState) -> Dict:

    if state.get("is_broken"):

        return {
            "prediction": None
        }

    run_mode = state.get(
        "run_mode",
        "oneshot"
    )

    if run_mode == "human":

        return run_human_logic(state)

    elif run_mode == "oneshot":

        return run_oneshot_logic(state)

    elif run_mode == "agentic":

        return run_agentic_logic(state)

    else:

        print(
            f"    [ERROR] Unknown run mode: "
            f"{run_mode}"
        )

        return {
            "prediction": None
        }


# =============================================================================
# INJECTION
# =============================================================================

def injection_node(state: AgentState) -> Dict:

    if state.get("is_broken"):

        return {
            "is_compiled": False
        }

    prediction = state.get("prediction")

    project_name = state.get("project_name")

    file_path = state.get("file_path")

    dp = state.get("raw_datapoint")
    import json


    test_prefix = (
        dp.get("testPrefix", {})
        if dp else {}
    )

    if (
        not prediction
        or not project_name
        or not file_path
    ):

        print(
            "    >> [DIAGNOSTIC] "
            "Injection aborted due to missing state variables."
        )

        return {
            "is_compiled": False
        }

    result = inject_prediction_into_test_file(
        state
    )

    if (
        not result.get("is_compiled", False)
        and state.get("run_mode") == "human"
    ):
        save_broken_test(
            state.get("item_id")
        )

    return result


# =============================================================================
# MUTATION
# =============================================================================
def mutation_node(state: AgentState) -> Dict:

    print(
        "      >> [AGENT] Running Mutation Analysis..."
    )

    result = execute_sandbox(state)

    mutation_score = result.get(
        "mutation_score",
        0.0
    )

    test_strength = result.get(
        "test_strength",
        0.0
    )

    sandbox_feedback = result.get(
        "sandbox_feedback",
        ""
    )

    compile_error = result.get(
        "compile_error",
        ""
    )

    test_failure = result.get(
        "test_failure",
        ""
    )

    failure_stage = result.get(
        "failure_stage",
        ""
    )

    surviving_mutants = result.get(
        "surviving_mutants",
        ""
    )

    covered_mutants = result.get(
        "covered_mutants",
        None
        )

    pit_metrics = result.get(
        "pit_metrics",
        {}
    )

    generated_mutants = pit_metrics.get(
        "generated",
        0
    )

    # =====================================================
    # AUTO-SKIP ZERO-COVERAGE TESTS
    # =====================================================

    if (
        state.get("run_mode") == "human"
        and covered_mutants == 0
        and generated_mutants > 0
    ):

        print(
            "\n    >> [ZERO COVERAGE TEST DETECTED]"
        )

        print(
            f"    >> Marking as broken/non-evaluable: "
            f"{state.get('item_id')}"
        )

        save_broken_test(
            state.get("item_id")
        )

        return {
            "mutation_score": None,
            "test_strength": None,
            "covered_mutants": covered_mutants,
            "compile_time": result.get(
                "compile_time",
                0.0
            ),
            "is_compiled": result.get(
                "is_compiled",
                True
            ),
            "is_evaluable": False,
            "compile_error": compile_error,
            "test_failure": test_failure,
            "failure_stage": failure_stage,
            "sandbox_feedback": sandbox_feedback,
            "surviving_mutants": surviving_mutants,
            "pit_metrics": pit_metrics
        }
    return {
        "mutation_score": mutation_score,
        "test_strength": test_strength,
        "covered_mutants": covered_mutants,
        "pit_metrics": pit_metrics,

        "compile_time": result.get(
            "compile_time",
            0.0
        ),
        "is_compiled": result.get(
            "is_compiled",
            False
        ),
        "is_evaluable": (
            covered_mutants is not None
            and not (
                covered_mutants == 0
                and generated_mutants > 0
            )
        ),
        "compile_error": compile_error,
        "test_failure": test_failure,
        "failure_stage": failure_stage,
        "sandbox_feedback": sandbox_feedback,
        "surviving_mutants": surviving_mutants
    }
