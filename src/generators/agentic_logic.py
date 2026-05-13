import os
import re
import subprocess
import shutil
from pathlib import Path
from typing import Dict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from state import AgentState
from utils.eval_tools import parse_maven_compile_log, parse_pitest_csv

from config import (
    MUTATION_JAR_PATH, JAVA_HOME, ORACLE_CONFIG_JSON,
    SANDBOX_DIR, CLONED_REPOS_DIR, DATA_PROJECT_DIR
)

# LLM Initialization
llm = ChatOpenAI(
    base_url="http://172.18.96.1:11434/v1",
    api_key="ollama",
    model="qwen3-coder:480b-cloud",
    temperature=0
)

# --- SANDBOX EXECUTION HELPER ---
def execute_sandbox(state: AgentState) -> Dict:
    """Runs the injection and Maven mutation process strictly for the agent's feedback loop."""
    prediction = state.get("prediction")
    project_name = state.get("project_name")
    project_id = state.get("project_id")
    item_id = state.get("item_id")
    file_path = state.get("file_path", "")
    
    java_bin = JAVA_HOME / "bin" / "java"
    repo_path = CLONED_REPOS_DIR / str(project_name)
    miner_json_path = DATA_PROJECT_DIR / "scripts" / "test-miner" / "output" / "miner" / f"{project_name}.json"
    
    # 1. Clean previous STAR files
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if "STAR" in file and file.endswith(".java"):
                try: os.remove(os.path.join(root, file))
                except: pass

    # 2. Clear old Pitest reports
    pit_reports_path = repo_path / "target" / "pit-reports"
    if pit_reports_path.exists():
        shutil.rmtree(pit_reports_path)

    # 3. Revert modifications
    try:
        subprocess.run(["git", "checkout", "--", "src/test"], cwd=str(repo_path), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    # 4. Inject prediction
    prediction_file = SANDBOX_DIR / f"{item_id}_agentic_prediction.txt"
    with open(prediction_file, "w", encoding="utf-8") as f:
        f.write(f"[oracle]{prediction}[/oracle]")
        
    cmd_inject = [
        str(java_bin), "-jar", str(MUTATION_JAR_PATH),
        str(project_id), str(repo_path), str(miner_json_path),
        str(SANDBOX_DIR), str(DATA_PROJECT_DIR / "scripts" / "dataset"), str(ORACLE_CONFIG_JSON),
        "INFERENCE", str(prediction_file)
    ]
    
    env = os.environ.copy()
    env["JAVA_HOME"] = str(JAVA_HOME)
    
    inject_res = subprocess.run(cmd_inject, cwd=str(repo_path), env=env, capture_output=True, text=True, timeout=600)
    
    # 5. Post-JAR Cleanup
    if file_path:
        target_stem = Path(file_path).stem 
        for root, dirs, files in os.walk(repo_path / "src" / "test"):
            for file in files:
                if "STAR" in file and not file.startswith(target_stem):
                    try: os.remove(os.path.join(root, file))
                    except: pass

    if inject_res.returncode != 0:
        return {"is_compiled": False, "stdout": inject_res.stdout + "\n" + inject_res.stderr, "score": 0.0, "csv_path": None}
        
    # 6. Precise Pitest Targeting
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
                
    cmd_mvn = [
        "mvn", "test-compile", "org.pitest:pitest-maven:mutationCoverage",
        "-DjvmArgs=--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
        f"-DtargetClasses={target_classes}",
        f"-DtargetTests={target_tests}",
        "-Dmutators=ALL", "-DoutputFormats=CSV"
    ]
    
    mvn_res = subprocess.run(cmd_mvn, cwd=str(repo_path), env=env, capture_output=True, text=True, timeout=1200)
    stdout = mvn_res.stdout
    
    if "Compilation failure" in stdout or "cannot find symbol" in stdout:
        return {"is_compiled": False, "stdout": stdout, "score": 0.0, "csv_path": None}
        
    score = 0.0
    match = re.search(r"Generated\s+\d+\s+mutations\s+Killed\s+\d+\s+\((\d+)%\)", stdout)
    if match:
        score = float(match.group(1)) / 100.0
        
    mutations_csv_path = None
    target_dir = repo_path / "target" / "pit-reports"
    if target_dir.exists():
        for root, dirs, files in os.walk(target_dir):
            if "mutations.csv" in files:
                mutations_csv_path = Path(root) / "mutations.csv"
                break
                
    return {"is_compiled": True, "stdout": stdout, "score": score, "csv_path": mutations_csv_path}

# --- NODES ---

def summarizer_node(state: AgentState) -> Dict:
    print("      >> [AGENT] Summarizing Context...")
    system_msg = (
            """You are a Java Code Analyzer. Analyze the Focal Class provided in the context. Create a Manifest that lists:

    The public methods available to be called.

    The constructor parameters required to instantiate the object.

    The return types of the primary methods.
    Ignore the empty test method body; focus on what can be tested based on the Focal Class logic."""
    )
    response = llm.invoke([("system", system_msg), ("human", state["prompt_context"])])
    summary = response.content.strip()
    print(f"         [MANIFEST]:\n         {summary}")
    return {"summary": summary}

def planner_node(state: AgentState) -> Dict:
    print("      >> [AGENT] Planning Assertions...")
    context = f"Manifest:\n{state.get('summary', 'None')}\n\nTarget Code:\n{state['prompt_context']}"
    system_msg = (
        "Based on the Manifest, identify which specific fields or return values can be checked. "
        "Plan assertions that target edge cases (nulls, empty strings, boundary values) "
        "found in the logic of the target code. Plain text only."
    )
    response = llm.invoke([("system", system_msg), ("human", context)])
    plan = response.content.strip()
    print(f"         [THOUGHT: PLAN]\n         {plan}")
    return {"plan": plan}

def coder_node(state: AgentState) -> Dict:
    print(f"      >> [AGENT] Coding Assertions (Iteration {state.get('iteration', 0) + 1})...")
    
    prompt = f"Target Code Context:\n{state['prompt_context']}\n\n"
    if state.get("summary"): prompt += f"Manifest (USE ONLY THESE NAMES):\n{state['summary']}\n\n"
    if state.get("plan"): prompt += f"Assertion Plan:\n{state['plan']}\n\n"
    
    if state.get("feedback_history"):
        prompt += (
            "PREVIOUS ATTEMPTS AND RESULTS:\n" + "\n".join(state["feedback_history"]) + 
            "\n\nINSTRUCTION: Your previous assertions achieved a baseline score. "
            "DO NOT delete working assertions. KEEP existing successful code and ADD NEW, "
            "more detailed assertions to kill surviving mutations."
        )

    system_msg = (
        "You are an expert Java Test Engineer. Output ONLY assertion lines. "
        "RULES: 1. DO NOT declare new variables. 2. Use ONLY variable names from the Manifest. "
        "3. ADD to existing successful assertions rather than replacing them. "
        "4. If a compilation error occurred previously, fix the syntax while keeping the logic. "
        "5. No markdown, no backticks, no explanations."
    )
    
    response = llm.invoke([("system", system_msg), ("human", prompt)])
    prediction = response.content.strip().replace("```java", "").replace("```", "").strip()
    print(f"         [THOUGHT: CODE]\n         {prediction}")
    return {"prediction": prediction}

def critic_node(state: AgentState) -> Dict:
    print("      >> [AGENT] Critic Executing Sandbox...")
    
    result = execute_sandbox(state)
    current_iter = state.get("iteration", 0)
    
    if not result["is_compiled"]:
        log_snippet = parse_maven_compile_log(result["stdout"])
        feedback = f"[COMPILATION FAILED] {log_snippet}"
        score = 0.0
    else:
        score = result["score"]
        # Use explicit status codes instead of the word 'SUCCESS' to avoid route confusion
        if score < 1.0:
            csv_feedback = parse_pitest_csv(result["csv_path"]) if result["csv_path"] else "No CSV details."
            feedback = f"[MUTATIONS SURVIVED] Score: {score}. Details: {csv_feedback}"
        else:
            feedback = "[GOAL ACHIEVED] 100% Mutation Coverage."

    print(f"         [CRITIC FEEDBACK]: {feedback}")
    print(f"      >> [AGENT] Iteration {current_iter + 1} Result: {score * 100:.1f}%")
    
    best_score = state.get("best_score", 0.0)
    best_pred = state.get("best_prediction", "")
    history = state.get("feedback_history", [])
    history.append(f"Attempt {current_iter + 1}: Score {score}. Feedback: {feedback}")
    
    if score > best_score:
        new_best_score = score
        new_best_pred = state["prediction"]
    else:
        new_best_score = best_score
        new_best_pred = best_pred

    return {
        "iteration": current_iter + 1,
        "feedback_history": history,
        "best_score": new_best_score,
        "best_prediction": new_best_pred
    }

# --- ROUTING & BUILD ---

def route_start(state: AgentState) -> str:
    if state.get("use_summarizer"): return "summarizer"
    if state.get("use_planner"): return "planner"
    return "coder"

def route_summarizer(state: AgentState) -> str:
    if state.get("use_planner"): return "planner"
    return "coder"

def route_critic(state: AgentState) -> str:
    # Logic Fix: Route based on numeric score and iteration count
    current_score = state.get("best_score", 0.0)
    current_iter = state.get("iteration", 0)
    max_iters = state.get("max_iterations", 3)
    
    if current_score >= 1.0 or current_iter >= max_iters or not state.get("use_evaluator_loop"):
        return END
    return "coder"

agent_workflow = StateGraph(AgentState)
agent_workflow.add_node("summarizer", summarizer_node)
agent_workflow.add_node("planner", planner_node)
agent_workflow.add_node("coder", coder_node)
agent_workflow.add_node("critic", critic_node)

agent_workflow.set_conditional_entry_point(route_start)
agent_workflow.add_conditional_edges("summarizer", route_summarizer)
agent_workflow.add_edge("planner", "coder")
agent_workflow.add_edge("coder", "critic")
agent_workflow.add_conditional_edges("critic", route_critic)

agent_app = agent_workflow.compile()

def run_agentic_logic(state: AgentState) -> Dict:
    print("    >> [AGENTIC] Initializing Agentic Loop...")
    final_agent_state = agent_app.invoke(state)
    
    # Return the best prediction found across all iterations
    return {
        "prediction": final_agent_state.get("best_prediction", final_agent_state.get("prediction")),
        "mutation_score": final_agent_state.get("best_score", 0.0)
    }