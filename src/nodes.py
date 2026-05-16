import os
import re
import csv
import shutil
import subprocess
import time
import traceback
from typing import Dict
from pathlib import Path 
from config import (
    MUTATION_JAR_PATH, JAVA_HOME, ORACLE_CONFIG_JSON,
    SANDBOX_DIR, CLONED_REPOS_DIR, DATA_PROJECT_DIR
)
from state import AgentState

# Import the isolated generation logic
from generators.human_logic import run_human_logic
from generators.oneshot_logic import run_oneshot_logic
from generators.agentic_logic import run_agentic_logic

def data_loader_node(state: AgentState) -> Dict:
    dp = state.get("raw_datapoint")
    test_prefix = dp.get("testPrefix", {})
    item_id = str(test_prefix.get("identifier", "UNKNOWN"))
    project_name = dp.get("_project_name", "dnsjava")
    project_id = str(project_name).split("/")[-1] 
    file_path = dp.get("_test_file_path", "")

    return {
        "item_id": item_id,
        "project_name": project_name,
        "project_id": project_id,
        "info_file_path": dp.get("_source_file_path"),
        "prompt_context": test_prefix.get("body"),
        "ground_truth": dp.get("target"),
        "file_path": file_path,
        "method_signature": test_prefix.get("signature"),
        "is_quarantined": False
    }

def generation_node(state: AgentState) -> Dict:
    if state.get("is_quarantined"):
        return {"prediction": None}

    run_mode = state.get("run_mode", "oneshot")

    if run_mode == "human":
        return run_human_logic(state)
    elif run_mode == "oneshot":
        return run_oneshot_logic(state)
    elif run_mode == "agentic":
        return run_agentic_logic(state)
    else:
        print(f"    [ERROR] Unknown run mode: {run_mode}")
        return {"prediction": None}

def injection_node(state: AgentState) -> Dict:
    if state.get("is_quarantined"):
        return {"is_compiled": False}

    prediction = state.get("prediction")
    project_name = state.get("project_name")
    file_path = state.get("file_path")
    dp = state.get("raw_datapoint")
    test_prefix = dp.get("testPrefix", {}) if dp else {}
    
    if not prediction or not project_name or not file_path: 
        print("    >> [DIAGNOSTIC] Injection aborted due to missing state variables.")
        return {"is_compiled": False}

    repo_path = CLONED_REPOS_DIR / str(project_name)
    
    file_path_str = str(file_path)
    if "STAR" in file_path_str:
        file_path_str = re.sub(r'STAR(?:Split|Normalized)?Test', '', file_path_str)
        if not file_path_str.endswith("Test.java"):
            file_path_str = file_path_str.replace(".java", "Test.java")

    original_file = repo_path / file_path_str
    print(f"    >> [INJECTION TARGET] {original_file}")

    if original_file.exists():
        try:
            relative_git_path = original_file.relative_to(repo_path)
            git_res = subprocess.run(
                ["git", "checkout", "HEAD", "--", str(relative_git_path)], 
                cwd=str(repo_path), 
                capture_output=True,
                text=True
            )
            if git_res.returncode != 0:
                print(f"    >> [GIT DIAGNOSTIC ERROR] Git checkout failed: {git_res.stderr.strip()}")
        except Exception as e:
            print(f"    >> [GIT WARNING] Could not verify clean baseline state: {e}")

    if not original_file.exists():
        print(f"    [ERROR] Original test file not found at path: {original_file}")
        return {"is_compiled": False}

    try:
        # 1. Create a clean backup copy from the freshly reset file
        backup_file = repo_path / (str(file_path_str) + ".bak")
        shutil.copyfile(original_file, backup_file)

        # 2. Read original file contents
        with open(original_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 3. Disarm all other existing test annotations to isolate execution
        content = re.sub(r'@Test\b', '// @Test', content)

        # 4. Inject static assertion imports conditionally to avoid collisions
        is_junit5 = "jupiter" in content or "org.junit.jupiter" in content
        if is_junit5:
            static_imports = "\nimport static org.junit.jupiter.api.Assertions.*;"
        else:
            static_imports = "\nimport static org.junit.Assert.*;"

        package_match = re.search(r'package\s+[\w.]+;', content)
        if package_match:
            content = content.replace(package_match.group(0), package_match.group(0) + static_imports)
        else:
            content = static_imports + "\n" + content

        # 5. Clean and format the prediction block
        cleaned_prediction = prediction.strip()
        if re.search(r'(?:public\s+|private\s+)?void\s+\w+\s*\([^)]*\)\s*\{', cleaned_prediction):
            start_idx = cleaned_prediction.find('{')
            end_idx = cleaned_prediction.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned_prediction = cleaned_prediction[start_idx + 1:end_idx].strip()

        # 6. Build standalone targeted test scenario method body
        method_body = test_prefix.get('body', '')
        pattern = re.compile(r'/\*.*?MASK_PLACEHOLDER.*?\*/')
        if pattern.search(method_body):
            method_stub = pattern.sub(cleaned_prediction, method_body)
        else:
            method_stub = method_body.replace("/*<MASK_PLACEHOLDER>*/", cleaned_prediction).replace("/*MASK_PLACEHOLDER*/", cleaned_prediction)

        full_method_injection = f"\n    @Test\n    {method_stub}\n"

        # 7. Append the fresh scenario before class closure
        rbrace_idx = content.rfind('}')
        if rbrace_idx != -1:
            content = content[:rbrace_idx] + full_method_injection + content[rbrace_idx:]

        # 8. Write code back to live target file
        with open(original_file, "w", encoding="utf-8") as f:
            f.write(content)

        return {"is_compiled": True, "file_path": file_path_str}

    except Exception as e:
        print(f"    >> [INJECTION NODE CRITICAL EXCEPTION]")
        traceback.print_exc()
        return {"is_compiled": False}

def mutation_node(state: AgentState) -> Dict:
    if state.get("is_quarantined"):
        return {"mutation_score": None}

    if not state.get("is_compiled"): 
        print("    >> [DIAGNOSTIC] Mutation node skipped because is_compiled is False.")
        return {"mutation_score": 0.0, "is_compiled": False}

    project_name = state.get("project_name")
    file_path = state.get("file_path")
    item_id = state.get("item_id")
    repo_path = CLONED_REPOS_DIR / str(project_name)

    # Resolve package matching and class filters using broader wildcards
    target_classes = "*.*"
    test_fqn = ""
    if file_path:
        parts = str(file_path).replace("\\", "/").split("src/test/java/")
        if len(parts) > 1:
            class_path = parts[-1].replace(".java", "")
            test_fqn = class_path.replace("/", ".")
            pkg_parts = test_fqn.split(".")
            class_name = pkg_parts[-1]
            if class_name.endswith("Test"): 
                class_name = class_name[:-4]
            target_classes = ".".join(pkg_parts[:-1]) + "." + class_name + "*"

    pit_reports_path = repo_path / "target" / "pit-reports"
    if pit_reports_path.exists():
        shutil.rmtree(pit_reports_path)

    score = 0.0
    critic_details = ""

    try:
        env = os.environ.copy()
        env["JAVA_HOME"] = str(JAVA_HOME)

        # STEP 1: Verify Injected Test Method Compiles and Passes
        cmd_verify = [
            "mvn", "test",
            "-DjvmArgs=--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
            f"-Dtest={test_fqn}#{item_id}"
        ]
        
        print(f"    >> [DIAGNOSTIC] Running verification command: {' '.join(cmd_verify)}")
        start_time = time.time()
        verify_result = subprocess.run(cmd_verify, cwd=str(repo_path), capture_output=True, text=True, env=env, timeout=600)
        
        if verify_result.returncode != 0:
            compile_time = time.time() - start_time
            filtered_lines = [line for line in verify_result.stdout.split('\n') if not line.strip().startswith('[WARNING]')]
            filtered_stdout = "\n".join(filtered_lines)
            error_lines = [line for line in filtered_lines if "<<< FAILURE!" in line or "<<< ERROR!" in line or "[ERROR]" in line or "Exception" in line]
            error_snippet = "\n".join(error_lines[:15]) if error_lines else filtered_stdout[-500:]
            
            print(f"      >> [VERIFICATION FAILED] Console Output:\n{filtered_stdout[-1500:]}\n")
            critic_details = f"[TEST SUITE FAILED] The test crashed during execution. Fix the error:\n{error_snippet}"
            return {"mutation_score": 0.0, "compile_time": compile_time, "critic_feedback": critic_details, "is_compiled": False}

        # STEP 2: Run Micro-Targeted Mutation Analysis via explicit PITest injection
        cmd_mutate = [
            "mvn", \
            "org.pitest:pitest-maven:1.16.1:mutationCoverage",
            "-DjvmArgs=--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
            f"-DtargetClasses={target_classes}",
            f"-DtargetTests={test_fqn}",
            "-Dmutators=ALL",
            "-DoutputFormats=CSV",
            "-DtimestampedReports=false"
        ]

        print(f"    >> [DIAGNOSTIC] Running mutation command: {' '.join(cmd_mutate)}")
        mutate_result = subprocess.run(cmd_mutate, cwd=str(repo_path), capture_output=True, text=True, env=env, timeout=1200)
        compile_time = time.time() - start_time

        if "unreported exception" in mutate_result.stdout or "COMPILATION ERROR" in mutate_result.stdout:
            filtered_lines = [line for line in mutate_result.stdout.split('\n') if not line.strip().startswith('[WARNING]')]
            filtered_stdout = "\n".join(filtered_lines)
            error_lines = [line for line in filtered_lines if "[ERROR]" in line or "COMPILATION ERROR" in line]
            error_snippet = "\n".join(error_lines[:15]) if error_lines else filtered_stdout[-500:]

            print(f"      >> [PIT COMPILATION ERROR] Crash during mutation phase! Console Output:\n{filtered_stdout[-1500:]}\n")
            critic_details = f"[COMPILATION ERROR DURING PIT] Your code failed to compile:\n{error_snippet}"
            return {"mutation_score": 0.0, "compile_time": compile_time, "critic_feedback": critic_details, "is_compiled": False}

        # STEP 3: Extract Scores from Generated Mutation Reports
        pit_csv_path = repo_path / "target" / "pit-reports" / "mutations.csv"
        
        if pit_csv_path.exists():
            print(f"      >> [DIAGNOSTIC] PITest CSV report successfully generated. Extracting values...")
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
                        elif status in ['KILLED', 'TIMED_OUT', 'MEMORY_ERROR']:
                            killed += 1
                        elif status == 'SURVIVED':
                            survived_list.append(f"Line {line} ({method}): {mutator} survived")
                
                covered = generated - no_coverage
                score = (killed / covered) if covered > 0 else 0.0
                
                print(f"      >> [DIAGNOSTIC METRICS] Total Mutations Generated: {generated}")
                print(f"      >> [DIAGNOSTIC METRICS] Mutations Killed: {killed}")
                print(f"      >> [DIAGNOSTIC METRICS] Mutations Lacking Coverage: {no_coverage}")
                print(f"      >> [DIAGNOSTIC METRICS] Mutations Covered: {covered}")
                print(f"      >> [DIAGNOSTIC METRICS] Final Score Ratio: {score}")
                
                if survived_list:
                    survived_str = "\n".join(survived_list[:15])
                    if len(survived_list) > 15:
                        survived_str += f"\n...and {len(survived_list) - 15} more."
                    critic_details = f"[MUTATIONS SURVIVED] Your assertions reached the code but failed to distinguish the mutant from the original. Strengthen your assertions for:\n{survived_str}"
                else:
                    critic_details = f"[GOAL ACHIEVED] 100% Test Strength for covered lines ({covered}/{generated} mutants covered)."
        else:
            print(f"\n      >> [DIAGNOSTIC CRITICAL] PITest CSV report completely missing at {pit_csv_path}")
            print(f"      >> [PITEST STDOUT LOGS]:\n{mutate_result.stdout[-1500:]}\n")
            print(f"      >> [PITEST STDERR LOGS]:\n{mutate_result.stderr[-800:]}\n")
            
            match_strength = re.search(r"Test strength\s+(\d+)%", mutate_result.stdout)
            score = (float(match_strength.group(1)) / 100.0) if match_strength else 0.0
            critic_details = "CSV report not found. Use better values in assertions to improve coverage score."

        return {"mutation_score": score, "compile_time": compile_time, "critic_feedback": critic_details, "is_compiled": True}

    except Exception as e:
        print(f"      >> [NODE ERROR] Exception in mutation_node: {e}")
        traceback.print_exc()
        return {"mutation_score": 0.0, "compile_time": 0.0, "critic_feedback": f"Error: {str(e)}", "is_compiled": False}

    finally:
        # STEP 4: Reset Workspace Environment for Next Runs
        if file_path:
            clean_cleanup_path = str(file_path)
            if "STAR" in clean_cleanup_path:
                clean_cleanup_path = re.sub(r'STAR(?:Split|Normalized)?Test', '', clean_cleanup_path)
                if not clean_cleanup_path.endswith("Test.java"):
                    clean_cleanup_path = clean_cleanup_path.replace(".java", "Test.java")
                    
            original_file = repo_path / clean_cleanup_path
            backup_file = repo_path / (clean_cleanup_path + ".bak")
            if backup_file.exists():
                if original_file.exists():
                    original_file.unlink()
                backup_file.rename(original_file)