import os
import re
import csv
import subprocess
import shutil

from pathlib import Path
from typing import Dict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from state import AgentState
from utils.eval_tools import parse_maven_compile_log

from config import (
    MUTATION_JAR_PATH,
    JAVA_HOME,
    ORACLE_CONFIG_JSON,
    SANDBOX_DIR,
    CLONED_REPOS_DIR,
    DATA_PROJECT_DIR
)

# =========================================================
# LLM INITIALIZATION
# =========================================================

llm = ChatOpenAI(
    base_url="http://172.18.96.1:11434/v1",
    api_key="ollama",
    model="qwen3-coder:480b-cloud",
    temperature=0
)

# =========================================================
# PIT MUTANT FEEDBACK STRATEGY
# =========================================================

def normalize_assertions(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        lines.append(stripped)
    return lines

def extract_pit_mutant_details(stdout: str, csv_path: Path = None, limit=15) -> str:
    """
    Extracts precise mutant feedback from the CSV report.
    """
    mutants = []

    if csv_path and csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 6:
                        continue

                    mutated_class = row[1].split('.')[-1]
                    mutator = row[3]
                    line = row[4]
                    status = row[5].strip()

                    if status == "SURVIVED":
                        mutants.append(f"Line {line} ({mutated_class}): {mutator} survived")
        except Exception:
            pass

    if not mutants:
        return "No mutants survived on the covered lines."

    return "\n".join(mutants[:limit])

# =========================================================
# SANDBOX EXECUTION
# =========================================================

def execute_sandbox(state: AgentState) -> Dict:

    prediction = state.get("prediction")
    project_name = state.get("project_name")
    project_id = state.get("project_id")
    item_id = state.get("item_id")
    file_path = state.get("file_path", "")

    java_bin = JAVA_HOME / "bin" / "java"
    repo_path = CLONED_REPOS_DIR / str(project_name)

    miner_json_path = (
        DATA_PROJECT_DIR / "scripts" / "test-miner" / "output" / "miner" / f"{project_name}.json"
    )

    # 1. CLEANUP
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if "STAR" in file and file.endswith(".java"):
                try: os.remove(os.path.join(root, file))
                except Exception: pass

    pit_reports_path = repo_path / "target" / "pit-reports"
    if pit_reports_path.exists():
        shutil.rmtree(pit_reports_path)

    try:
        subprocess.run(["git", "checkout", "--", "src/test"], cwd=str(repo_path), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    # 2. INJECT
    prediction_file = SANDBOX_DIR / f"{item_id}_agentic_prediction.txt"
    with open(prediction_file, "w", encoding="utf-8") as f:
        f.write(f"[oracle]{prediction}[/oracle]")

    cmd_inject = [
        str(java_bin), "-jar", str(MUTATION_JAR_PATH),
        str(project_id), str(repo_path), str(miner_json_path),
        str(SANDBOX_DIR), str(DATA_PROJECT_DIR / "scripts" / "dataset"),
        str(ORACLE_CONFIG_JSON), "INFERENCE", str(prediction_file)
    ]

    env = os.environ.copy()
    env["JAVA_HOME"] = str(JAVA_HOME)

    inject_res = subprocess.run(cmd_inject, cwd=str(repo_path), env=env, capture_output=True, text=True, timeout=600)

    # Clean Extra STAR Files
    if file_path:
        target_stem = Path(file_path).stem
        for root, dirs, files in os.walk(repo_path / "src" / "test"):
            for file in files:
                if "STAR" in file and not file.startswith(target_stem):
                    try: os.remove(os.path.join(root, file))
                    except Exception: pass

    if inject_res.returncode != 0:
        return {"is_compiled": False, "compile_error": "Injection Failed. Syntax error in assertions."}

    # 3. RUN PIT
    target_tests = "*"
    target_classes = "*"

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

    cmd_mvn = [
        "mvn", "test-compile", "org.pitest:pitest-maven:mutationCoverage",
        "-DjvmArgs=--add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.lang.reflect=ALL-UNNAMED",
        f"-DtargetClasses={target_classes}",
        f"-DtargetTests={target_tests}",
        "-Dmutators=ALL",
        "-DoutputFormats=CSV",
        "-DtimestampedReports=false",
        "-q" # Maven Quiet Mode to suppress standard info/warnings
    ]

    mvn_res = subprocess.run(cmd_mvn, cwd=str(repo_path), env=env, capture_output=True, text=True, timeout=1200)
    stdout = mvn_res.stdout

    # Clean compile error extraction (No Console Spam)
    if "Compilation failure" in stdout or "COMPILATION ERROR" in stdout:
        errors = [line.strip() for line in stdout.splitlines() if "[ERROR]" in line and "COMPILATION ERROR" not in line]
        error_msg = "\n".join(errors[:3]) if errors else "Maven compilation failure due to invalid Java code."
        return {"is_compiled": False, "compile_error": error_msg}

    # 4. SCORE
    mutations_csv_path = None
    target_dir = repo_path / "target" / "pit-reports"

    if target_dir.exists():
        for root, dirs, files in os.walk(target_dir):
            if "mutations.csv" in files:
                mutations_csv_path = Path(root) / "mutations.csv"
                break

    score = 0.0
    if mutations_csv_path and mutations_csv_path.exists():
        with open(mutations_csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            generated = 0
            killed = 0
            for row in reader:
                if len(row) < 6: continue
                status = row[5].strip()
                if status == "NO_COVERAGE": continue
                generated += 1
                if status in ("KILLED", "TIMED_OUT", "MEMORY_ERROR"):
                    killed += 1
            if generated > 0:
                score = killed / generated

    return {
        "is_compiled": True,
        "stdout": stdout,
        "score": score,
        "mutations_csv": str(mutations_csv_path) if mutations_csv_path else None
    }


# =========================================================
# AGENT NODES
# =========================================================

def summarizer_node(state: AgentState) -> Dict:
    print("      >> [AGENT] Summarizing Context...")
    system_msg = (
        "You are a strict Java Static Analysis Tool. Your task is to build a Variable Manifest.\n"
        "RULES:\n"
        "1. DO NOT GUESS method names. Only list methods explicitly visible in the Focal Class source code.\n"
        "2. INFER FROM CONSTRUCTORS: If the Focal Class returns 'new Response(a, b, c)', assume the result object has properties for a, b, and c.\n"
        "3. TEST STUB ANCHOR: List every variable declared in the 'TEST FILE CONTEXT (STUB)' and its visible type.\n"
        "4. ASSERTION ONLY: Identify only those elements that can be passed into an assertEquals() or assertTrue()."
    )

    response = llm.invoke([("system", system_msg), ("human", state["prompt_context"])])
    summary = response.content.strip()
    
    print(f"         [MANIFEST]:\n{summary}\n")
    return {"summary": summary}


def planner_node(state: AgentState) -> Dict:
    print("      >> [AGENT] Planning Assertions...")
    system_msg = (
        "You are a Mutation Testing Strategist. Design an assertion strategy to maximize Test Strength.\n"
        "STRICT RULE: Do NOT provide any Java code or snippets. Use only high-level natural language instructions."
    )
    context = f"Manifest:\n{state.get('summary', 'None')}\n\nTarget Code:\n{state['prompt_context']}"
    
    response = llm.invoke([("system", system_msg), ("human", context)])
    plan = response.content.strip()
    
    print(f"         [PLAN]:\n{plan}\n")
    return {"plan": plan}


def coder_node(state: AgentState) -> Dict:
    print("      >> [AGENT] Coding Assertions...")
    manifest = state.get("summary", "")
    strategy = state.get("improvement_plan") or state.get("plan", "Generate specific, high-precision assertions.")
    previous_code = state.get("prediction", "")

    prompt = (
        f"CONTEXT:\n{state['prompt_context']}\n\n"
        f"TRUTH MANIFEST (Variables/Methods):\n{manifest}\n\n"
        f"PLAN TO FOLLOW:\n{strategy}\n\n"
    )

    if previous_code:
        prompt += f"PREVIOUS CODE (FIX ERRORS IF ANY):\n{previous_code}\n\n"

    prompt += (
        "INSTRUCTION: Output ONLY pure Java assertion lines (e.g. assertEquals(x, y);).\n"
        "CRITICAL: If the previous code failed to compile, change the method names to match the MANIFEST.\n"
        "Unless the Manifest explicitly shows a custom accessor, assume standard Java Getters (e.g., use getCode() instead of code())."
    )

    system_msg = "You are a JUnit Expert. Output ONLY pure Java assertion lines. No markdown. No chatter. No logic setup."
    response = llm.invoke([("system", system_msg), ("human", prompt)])

    new_prediction = response.content.strip().replace("```java", "").replace("```", "").strip()

    previous = normalize_assertions(previous_code)
    new = normalize_assertions(new_prediction)
    combined = list(previous)

    for n in new:
        if n not in combined:
            combined.append(n)

    combined_prediction = "\n".join(combined)
    print(f"         [CODE]:\n{combined_prediction}\n")
    return {"prediction": combined_prediction}


def critic_node(state: AgentState) -> Dict:
    print("      >> [AGENT] Critic Executing Sandbox...")
    result = execute_sandbox(state)

    if not result.get("is_compiled", True):
        compile_err = result.get("compile_error", "Unknown Compilation Error.")
        feedback = f"COMPILATION FAILED:\n{compile_err}"
        score = 0.0
    else:
        score = result["score"]
        mutant_details = extract_pit_mutant_details(
            result["stdout"],
            Path(result.get("mutations_csv")) if result.get("mutations_csv") else None
        )
        feedback = f"PIT Score: {score:.2f}\n\nSURVIVING MUTANTS:\n{mutant_details}"

    print(f"         [CRITIC RESULT]:\n{feedback}\n")

    best_score = state.get("best_score", 0.0)
    best_pred = state.get("best_prediction", "")

    if score > best_score:
        best_score = score
        best_pred = state["prediction"]

    return {
        "iteration": state.get("iteration", 0) + 1,
        "latest_feedback": feedback,
        "best_score": best_score,
        "best_prediction": best_pred
    }


def improver_node(state: AgentState) -> Dict:
    print("      >> [AGENT] Improving...")
    system_msg = (
        "You are a Mutation Testing Strategist. Analyze why the last attempt failed or scored poorly.\n"
        "If it was a COMPILATION ERROR, identify the illegal method call.\n"
        "If it was surviving mutants, explain how to make the assertions more strict.\n"
        "STRICT: Natural language only. No code snippets."
    )
    context = (
        f"MANIFEST:\n{state.get('summary', 'None')}\n\n"
        f"LAST CODE ATTEMPT:\n{state.get('prediction', 'None')}\n\n"
        f"EXECUTION FEEDBACK:\n{state.get('latest_feedback', '')}"
    )

    response = llm.invoke([("system", system_msg), ("human", context)])
    plan = response.content.strip()
    
    print(f"         [IMPROVEMENT PLAN]:\n{plan}\n")
    return {"improvement_plan": plan}


# =========================================================
# ROUTING + GRAPH (UNCHANGED)
# =========================================================

def route_start(state: AgentState) -> str:
    if state.get("use_summarizer"):
        return "summarizer"
    if state.get("use_planner"):
        return "planner"
    return "coder"


def route_summarizer(state: AgentState) -> str:
    if state.get("use_planner"):
        return "planner"
    return "coder"


def route_critic(state: AgentState) -> str:
    if state.get("best_score", 0.0) >= 1.0 or state.get("iteration", 0) >= state.get("max_iterations", 3):
        return END
    return "improver"


agent_workflow = StateGraph(AgentState)

agent_workflow.add_node("summarizer", summarizer_node)
agent_workflow.add_node("planner", planner_node)
agent_workflow.add_node("coder", coder_node)
agent_workflow.add_node("critic", critic_node)
agent_workflow.add_node("improver", improver_node)

agent_workflow.set_conditional_entry_point(route_start)
agent_workflow.add_conditional_edges("summarizer", route_summarizer)
agent_workflow.add_edge("planner", "coder")
agent_workflow.add_edge("coder", "critic")
agent_workflow.add_conditional_edges("critic", route_critic)
agent_workflow.add_edge("improver", "coder")

agent_app = workflow_compiled = agent_workflow.compile()

def run_agentic_logic(state: AgentState) -> Dict:
    print("    >> [AGENTIC] Initializing Agentic Loop...")
    
    state["best_score"] = 0.0
    state["best_prediction"] = ""
    state["iteration"] = 0
    
    final_state = agent_app.invoke(state)

    return {
        "prediction": final_state.get("best_prediction", final_state.get("prediction")),
        "mutation_score": final_state.get("best_score", 0.0)
    }