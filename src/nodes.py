import os
import re
import csv
import json
import shutil
import subprocess
import time
import traceback

from typing import Dict
from pathlib import Path

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

BROKEN_FILE = (
    DATA_PROJECT_DIR
    / "scripts"
    / "dataset"
    / "output"
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


def cleanup_star_files(repo_path):

    for root, dirs, files in os.walk(repo_path):

        for file in files:

            if (
                "STAR" in file
                and file.endswith(".java")
            ):

                try:

                    os.remove(
                        os.path.join(root, file)
                    )

                except Exception:

                    pass



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

    if item_id == "testGetEndpoint_split_0":

        print(
            f"    >> [SKIP FIRST TEST] "
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

    broken_tests = load_broken_tests()

    is_broken = False

    if (
        state.get("run_mode") != "human"
        and broken_tests.get(item_id, False)
    ):

        print(
            f"    >> [BROKEN TEST SKIPPED] "
            f"{item_id}"
        )

        is_broken = True

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

    compact_prompt_context = f"""
TARGET TEST METHOD:
{test_body}

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

    print(
        "\n=================================================="
    )

    print(
        f"[PROCESSING] {item_id}"
    )

    print(
        "==================================================\n"
    )

    print(
        "\n========= COMPACT EXECUTABLE CONTEXT =========\n"
    )

    print(compact_prompt_context)

    print(
        "\n==============================================\n"
    )

    print(
        "\n========= FULL SUMMARIZER CONTEXT =========\n"
    )

    print(
        f"[DEBUG] Full context chars = "
        f"{len(full_prompt_context)}"
    )

    print(
        f"[DEBUG] Total methods exposed to summarizer = "
        f"{len(focal_class.get('methods', []))}"
    )

    print(
        f"[DEBUG] Total constructors exposed = "
        f"{len(focal_class.get('constructors', []))}"
    )

    print(
        f"[DEBUG] Total fields exposed = "
        f"{len(focal_class.get('fields', []))}"
    )

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

    repo_path = (
        CLONED_REPOS_DIR
        / str(project_name)
    )

    cleanup_star_files(repo_path)

    file_path_str = str(file_path)

    # =========================================================
    # STAR FILE NORMALIZATION
    # =========================================================

    if "STAR" in file_path_str:

        file_path_str = re.sub(
            r"STAR(?:Split|Normalized)?Test",
            "",
            file_path_str
        )

        if not file_path_str.endswith(
            "Test.java"
        ):

            file_path_str = file_path_str.replace(
                ".java",
                "Test.java"
            )

    original_file = (
        repo_path
        / file_path_str
    )

    print(
        f"    >> [INJECTION TARGET] "
        f"{original_file}"
    )

    # =========================================================
    # HARD RESET TEST FILE
    # =========================================================

    if original_file.exists():

        try:

            relative_git_path = (
                original_file.relative_to(
                    repo_path
                )
            )

            subprocess.run(
                [
                    "git",
                    "checkout",
                    "HEAD",
                    "--",
                    str(relative_git_path)
                ],
                cwd=str(repo_path),
                capture_output=True,
                text=True
            )

        except Exception as e:

            print(
                f"    >> [GIT WARNING] "
                f"{e}"
            )

    if not original_file.exists():

        print(
            f"    [ERROR] Original test file "
            f"not found at path: "
            f"{original_file}"
        )

        return {
            "is_compiled": False
        }

    try:

        # =====================================================
        # BACKUP
        # =====================================================

        backup_file = (
            repo_path
            / (str(file_path_str) + ".bak")
        )

        shutil.copyfile(
            original_file,
            backup_file
        )

        with open(
            original_file,
            "r",
            encoding="utf-8"
        ) as f:

            content = f.read()

        # =====================================================
        # DISABLE ALL OTHER TESTS
        # =====================================================

        content = re.sub(
            r'@Test\b',
            '// @Test',
            content
        )

        # =====================================================
        # ASSERT IMPORTS
        # =====================================================

        is_junit5 = (
            "jupiter" in content
            or "org.junit.jupiter" in content
        )

        if is_junit5:

            static_imports = (
                "\nimport static "
                "org.junit.jupiter.api.Assertions.*;"
            )

        else:

            static_imports = (
                "\nimport static "
                "org.junit.Assert.*;"
            )

        if static_imports not in content:

            package_match = re.search(
                r'package\s+[\w.]+;',
                content
            )

            if package_match:

                content = content.replace(
                    package_match.group(0),
                    package_match.group(0)
                    + static_imports
                )

            else:

                content = (
                    static_imports
                    + "\n"
                    + content
                )

        # =====================================================
        # CLEAN GENERATED OUTPUT
        # =====================================================

        cleaned_prediction = (
            prediction.strip()
        )

        if re.search(
            r'(?:public\s+|private\s+)?void\s+\w+\s*\([^)]*\)\s*\{',
            cleaned_prediction
        ):

            start_idx = cleaned_prediction.find('{')

            end_idx = cleaned_prediction.rfind('}')

            if (
                start_idx != -1
                and end_idx != -1
                and end_idx > start_idx
            ):

                cleaned_prediction = (
                    cleaned_prediction[
                        start_idx + 1:end_idx
                    ].strip()
                )

        # =====================================================
        # INJECT ASSERTIONS INTO MASK
        # =====================================================

        method_body = test_prefix.get(
            "body",
            ""
        )

        pattern = re.compile(
            r'/\*.*?MASK_PLACEHOLDER.*?\*/'
        )

        if pattern.search(method_body):

            method_stub = pattern.sub(
                cleaned_prediction,
                method_body
            )

        else:

            method_stub = (
                method_body
                .replace(
                    "/*<MASK_PLACEHOLDER>*/",
                    cleaned_prediction
                )
                .replace(
                    "/*MASK_PLACEHOLDER*/",
                    cleaned_prediction
                )
            )

        full_method_injection = (
            f"\n    @Test\n"
            f"    {method_stub}\n"
        )

     


        # =====================================================
        # APPEND TEST
        # =====================================================

        rbrace_idx = content.rfind('}')

        if rbrace_idx != -1:

            content = (
                content[:rbrace_idx]
                + full_method_injection
                + content[rbrace_idx:]
            )

        with open(
            original_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(content)

        return {
            "is_compiled": True,
            "file_path": file_path_str
        }

    except Exception:

        traceback.print_exc()

        if state.get("run_mode") == "human":

            save_broken_test(
                state.get("item_id")
            )

        return {
            "is_compiled": False
        }


# =============================================================================
# MUTATION
# =============================================================================

def mutation_node(state: AgentState) -> Dict:

    if state.get("is_broken"):

        return {
            "mutation_score": None,
            "test_strength": None
        }

    if not state.get("is_compiled"):

        if state.get("run_mode") == "human":

            save_broken_test(
                state.get("item_id")
            )

        return {
            "mutation_score": None,
            "test_strength": None,
            "is_compiled": False
        }

    project_name = state.get("project_name")

    file_path = state.get("file_path")

    item_id = state.get("item_id")

    repo_path = (
        CLONED_REPOS_DIR
        / str(project_name)
    )

    cleanup_star_files(repo_path)

    target_classes = "*.*"

    test_fqn = ""

    # =========================================================
    # BUILD TEST FQN
    # =========================================================

    if file_path:

        parts = (
            str(file_path)
            .replace("\\", "/")
            .split("src/test/java/")
        )

        if len(parts) > 1:

            class_path = (
                parts[-1]
                .replace(".java", "")
            )

            test_fqn = class_path.replace(
                "/",
                "."
            )

            pkg_parts = test_fqn.split(".")

            class_name = pkg_parts[-1]

            if class_name.endswith("Test"):

                class_name = class_name[:-4]

            target_classes = (
                ".".join(pkg_parts[:-1])
                + "."
                + class_name
                + "*"
            )

    pit_reports_path = (
        repo_path
        / "target"
        / "pit-reports"
    )

    if pit_reports_path.exists():

        shutil.rmtree(
            pit_reports_path
        )

    mutation_score = 0.0

    test_strength = 0.0

    try:

        env = os.environ.copy()

        env["JAVA_HOME"] = str(
            JAVA_HOME
        )



        start_time = time.time()






        # =====================================================
        # PITEST
        # =====================================================

        cmd_mutate = [
            "mvn",
            "org.pitest:pitest-maven:1.16.1:mutationCoverage",
            "-DjvmArgs=--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
            f"-DtargetClasses={target_classes}",
            f"-DtargetTests={test_fqn}",
            "-Dmutators=ALL",
            "-DoutputFormats=CSV",
            "-DtimestampedReports=false"
        ]



        mutate_result = subprocess.run(
            cmd_mutate,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            env=env,
            timeout=1200
        )

        compile_time = (
            time.time()
            - start_time
        )

        # =====================================================
        # PITEST FAILURE DETECTION
        # =====================================================

        if (
            "BUILD FAILURE" in mutate_result.stdout
            or "BUILD FAILURE" in mutate_result.stderr
            or "COMPILATION ERROR" in mutate_result.stdout
            or "COMPILATION ERROR" in mutate_result.stderr
        ):

            print(
                "\n      >> [PIT FAILURE]"
            )

            print(
                mutate_result.stdout[-2000:]
            )

            print(
                mutate_result.stderr[-1000:]
            )

            return {
                "mutation_score": 0.0,
                "test_strength": 0.0,
                "compile_time": compile_time,
                "is_compiled": False
            }

        # =====================================================
        # PARSE CSV
        # =====================================================

        pit_csv_path = (
            repo_path
            / "target"
            / "pit-reports"
            / "mutations.csv"
        )

        if pit_csv_path.exists():

            with open(
                pit_csv_path,
                "r",
                encoding="utf-8"
            ) as csvfile:

                reader = csv.reader(
                    csvfile
                )

                generated = 0
                killed = 0
                no_coverage = 0

                for row in reader:

                    if len(row) >= 6:

                        status = row[5].strip()

                        generated += 1

                        if status == "NO_COVERAGE":

                            no_coverage += 1

                        elif status in [
                            "KILLED",
                            "TIMED_OUT",
                            "MEMORY_ERROR"
                        ]:

                            killed += 1

                covered = (
                    generated
                    - no_coverage
                )

                mutation_score = (
                    killed / generated
                    if generated > 0
                    else 0.0
                )

                if covered > 0:

                    test_strength = (
                        killed / covered
                    )

                else:

                    test_strength = (
                        mutation_score
                    )

                print(
                    f"    >> [PIT METRICS] "
                    f"Generated={generated} "
                    f"Killed={killed} "
                    f"NoCoverage={no_coverage} "
                    f"Covered={covered}"
                )

                print(
                    f"    >> [FINAL SCORES] "
                    f"TS={test_strength:.4f} "
                    f"MS={mutation_score:.4f}"
                )

        else:

            print(
                "\n    >> [WARNING] "
                "mutations.csv not found"
            )

            print(
                mutate_result.stdout[-2000:]
            )

            print(
                mutate_result.stderr[-1000:]
            )

        if state.get("run_mode") == "human":

            remove_broken_test(item_id)

        return {
            "mutation_score": float(
                mutation_score
            ),
            "test_strength": float(
                test_strength
            ),
            "compile_time": compile_time,
            "is_compiled": True
        }

    except Exception as e:

        print(
            f"      >> [NODE ERROR] "
            f"{e}"
        )

        traceback.print_exc()

        if state.get("run_mode") == "human":

            save_broken_test(
                item_id
            )

        return {
            "mutation_score": None,
            "test_strength": None,
            "compile_time": 0.0,
            "is_compiled": False
        }

    finally:

        # =====================================================
        # CLEANUP
        # =====================================================

        if file_path:

            clean_cleanup_path = str(
                file_path
            )

            if "STAR" in clean_cleanup_path:

                clean_cleanup_path = re.sub(
                    r"STAR(?:Split|Normalized)?Test",
                    "",
                    clean_cleanup_path
                )

                if not clean_cleanup_path.endswith(
                    "Test.java"
                ):

                    clean_cleanup_path = (
                        clean_cleanup_path.replace(
                            ".java",
                            "Test.java"
                        )
                    )

            original_file = (
                repo_path
                / clean_cleanup_path
            )

            backup_file = (
                repo_path
                / (clean_cleanup_path + ".bak")
            )

            if backup_file.exists():

                if original_file.exists():

                    original_file.unlink()

                backup_file.rename(
                    original_file
                )

        cleanup_star_files(repo_path)