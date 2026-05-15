import os
import re
import csv
import shutil
import subprocess
import time
from typing import Dict
from pathlib import Path 
from state import AgentState
from config import (
    MUTATION_JAR_PATH, JAVA_HOME, ORACLE_CONFIG_JSON,
    SANDBOX_DIR, CLONED_REPOS_DIR, DATA_PROJECT_DIR
)

from generators.oneshot_logic import run_oneshot_logic
from generators.agentic_logic import run_agentic_logic
from generators.human_logic import run_human_logic

def data_loader_node(state: AgentState) -> Dict:
    dp = state.get("raw_datapoint")
    test_prefix = dp.get("testPrefix", {})
    focal_info = dp.get("_focalClass", {})
    
    # Extract method bodies
    method_codes = []
    for method in focal_info.get("methods", []):
        m_body = method.get("body")
        if m_body:
            method_codes.append(m_body)
    
    focal_code = "\n\n".join(method_codes)
    test_stub = test_prefix.get('body', '')
    method_sig = test_prefix.get("signature")

    # --- NEW PRINT STATEMENTS FOR VISIBILITY ---
    print("\n" + "="*80)
    print(f"TARGET METHOD: {method_sig}")
    print("-" * 80)
    print("### FOCAL CLASS IMPLEMENTATION ###")
    print(focal_code if focal_code else "NO IMPLEMENTATION FOUND")
    print("-" * 80)
    print("### TEST FILE CONTEXT (STUB) ###")
    print(test_stub)
    print("="*80 + "\n")
    # --------------------------------------------

    combined_context = (
        "### FOCAL CLASS IMPLEMENTATION ###\n"
        f"{focal_code if focal_code else 'NO IMPLEMENTATION FOUND'}\n\n"
        "### TEST FILE CONTEXT (STUB) ###\n"
        f"{test_stub}"
    )
    
    return {
        "item_id": str(test_prefix.get("identifier", "UNKNOWN")),
        "project_name": dp.get("_project_name"),
        "project_id": str(dp.get("_project_name", "")).split(".")[-1], 
        "prompt_context": combined_context,
        "ground_truth": dp.get("target"),
        "file_path": dp.get("_test_file_path"),
        "method_signature": method_sig
    }

def generation_node(state: AgentState) -> Dict:
    mode = state.get("run_mode")
    if mode == "human":
        return run_human_logic(state)
    elif mode == "agentic":
        return run_agentic_logic(state)
    return run_oneshot_logic(state)

def injection_node(state: AgentState) -> Dict:
    prediction = state.get("prediction")
    project_name = state.get("project_name")
    project_id = state.get("project_id")
    file_path = state.get("file_path")
    
    if not prediction: 
        return {"is_compiled": False}

    repo_path = CLONED_REPOS_DIR / str(project_name)
    miner_json_path = DATA_PROJECT_DIR / "scripts" / "test-miner" / "output" / "miner" / f"{project_name}.json"

    # 1. PRE-RUN CLEANUP
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if "STAR" in file and file.endswith(".java"):
                try: os.remove(os.path.join(root, file))
                except: pass

    subprocess.run(["git", "checkout", "--", "src/test"], cwd=str(repo_path), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    prediction_file = SANDBOX_DIR / f"{state.get('item_id')}_prediction.txt"
    with open(prediction_file, "w", encoding="utf-8") as f:
        f.write(f"[oracle]{prediction}[/oracle]")

    cmd = [
        str(JAVA_HOME / "bin" / "java"), "-jar", str(MUTATION_JAR_PATH),
        str(project_id), str(repo_path), str(miner_json_path),
        str(SANDBOX_DIR), str(DATA_PROJECT_DIR / "scripts" / "dataset"), str(ORACLE_CONFIG_JSON),
        "INFERENCE", str(prediction_file)
    ]

    try:
        env = os.environ.copy()
        env["JAVA_HOME"] = str(JAVA_HOME)
        result = subprocess.run(cmd, cwd=str(repo_path), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
        
        # 2. POST-RUN CLEANUP (The Missing Piece)
        # The JAR generates many files. Keep ONLY the one we are testing right now.
        if file_path:
            target_stem = Path(file_path).stem 
            for root, dirs, files in os.walk(repo_path / "src" / "test"):
                for file in files:
                    if "STAR" in file and not file.startswith(target_stem):
                        try: os.remove(os.path.join(root, file))
                        except: pass

        return {"is_compiled": (result.returncode == 0)}
    except:
        return {"is_compiled": False}
def mutation_node(state: AgentState) -> Dict:
    if not state.get("is_compiled"): 
        return {"mutation_score": 0.0, "compile_time": 0.0, "critic_feedback": "[COMPILATION FAILED] Ensure all variables are declared and the code compiles."}

    project_name = state.get("project_name")
    repo_path = CLONED_REPOS_DIR / str(project_name)
    file_path = state.get("file_path", "")
    
    # CLEAR PIT REPORTS CACHE
    pit_reports_path = repo_path / "target" / "pit-reports"
    if pit_reports_path.exists():
        shutil.rmtree(pit_reports_path)

    target_tests, target_classes = "*", "*"
    if file_path:
        parts = str(file_path).replace("\\", "/").split("src/test/java/")
        if len(parts) > 1:
            class_path = parts[-1].replace(".java", "")
            test_fqn = class_path.replace("/", ".")
            target_tests = test_fqn + "*" 
            pkg_parts = test_fqn.split(".")
            class_name = pkg_parts[-1]
            if class_name.endswith("STARSplitTest"): class_name = class_name[:-13]
            if class_name.endswith("Test"): class_name = class_name[:-4]
            target_classes = ".".join(pkg_parts[:-1]) + "." + class_name if len(pkg_parts) > 1 else class_name

    # --- STEP 1: Verify Test Compiles and Passes ---
    cmd_verify = [
        "mvn", "test",
        "-DjvmArgs=--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
        f"-Dtest={target_tests}"
    ]

    try:
        env = os.environ.copy()
        env["JAVA_HOME"] = str(JAVA_HOME)
        start_time = time.time()
        
        verify_result = subprocess.run(cmd_verify, cwd=str(repo_path), capture_output=True, text=True, env=env, timeout=600)
        
        if verify_result.returncode != 0:
            compile_time = time.time() - start_time
            return {"mutation_score": None, "is_quarantined": True, "compile_time": compile_time, "critic_feedback": "[TEST FAILED] The injected assertions caused the test suite to fail or throw an exception."}

        # --- STEP 2: Run PITest Mutation Coverage ---
        cmd_mutate = [
            "mvn", "test-compile", "org.pitest:pitest-maven:mutationCoverage",
            "-DjvmArgs=--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
            f"-DtargetClasses={target_classes}", f"-DtargetTests={target_tests}",
            "-Dmutators=ALL", 
            "-DoutputFormats=CSV", 
            "-DtimestampedReports=false"
        ]

        mutate_result = subprocess.run(cmd_mutate, cwd=str(repo_path), capture_output=True, text=True, env=env, timeout=1200)
        compile_time = time.time() - start_time

        if "unreported exception" in mutate_result.stdout or "COMPILATION ERROR" in mutate_result.stdout:
            return {"mutation_score": None, "is_quarantined": True, "compile_time": compile_time, "critic_feedback": "[COMPILATION ERROR DURING PIT]"}

        # --- STEP 3: Extract Test Strength and Filtered Critic Feedback ---
        pit_csv_path = repo_path / "target" / "pit-reports" / "mutations.csv"
        
        test_strength = 0.0
        critic_details = ""
        
        if pit_csv_path.exists():
            with open(pit_csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                generated = 0
                killed = 0
                no_coverage = 0
                survived_list = []
                
                for row in reader:
                    if len(row) >= 6:
                        generated += 1
                        mutator = row[2].split('.')[-1]
                        method = row[3]
                        line = row[4]
                        status = row[5].strip()
                        
                        if status == 'NO_COVERAGE':
                            no_coverage += 1
                            # Silently ignore NO_COVERAGE for feedback purposes
                        elif status in ['KILLED', 'TIMED_OUT', 'MEMORY_ERROR']:
                            killed += 1
                        elif status == 'SURVIVED':
                            # We only notify the agent about mutants on lines it ACTUALLY reached
                            survived_list.append(f"Line {line} ({method}): {mutator} survived")
                
                covered = generated - no_coverage
                test_strength = (killed / covered) if covered > 0 else 0.0
                
                if survived_list:
                    # Provide feedback ONLY on survived mutants (the ones the agent can fix)
                    survived_str = "\n".join(survived_list[:15])
                    if len(survived_list) > 15:
                        survived_str += f"\n...and {len(survived_list) - 15} more."
                    critic_details = f"[MUTATIONS SURVIVED] Your assertions reached the code but failed to distinguish the mutant from the original. Strengthen your assertions for:\n{survived_str}"
                else:
                    critic_details = f"[GOAL ACHIEVED] 100% Test Strength for covered lines ({covered}/{generated} mutants covered)."
        else:
            # Fallback
            match_strength = re.search(r"Test strength\s+(\d+)%", mutate_result.stdout)
            test_strength = (float(match_strength.group(1)) / 100.0) if match_strength else 0.0
            critic_details = "CSV report not found. Use better values in assertions to improve coverage score."

        return {
            "mutation_score": test_strength, 
            "compile_time": compile_time, 
            "critic_feedback": critic_details
        }
        
    except Exception as e:
        return {"mutation_score": 0.0, "compile_time": 0.0, "critic_feedback": f"Error: {str(e)}"}