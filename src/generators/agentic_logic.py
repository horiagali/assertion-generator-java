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
# ASSERTION HELPERS
# =========================================================

def normalize_assertions(text: str) -> list[str]:

    cleaned = []

    for line in text.splitlines():

        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("```"):
            continue

        if stripped not in cleaned:
            cleaned.append(stripped)

    return cleaned


def merge_assertions(
    previous_assertions: str,
    new_assertions: str
) -> str:

    previous = normalize_assertions(previous_assertions)

    new = normalize_assertions(new_assertions)

    merged = list(previous)

    for assertion in new:

        if assertion not in merged:
            merged.append(assertion)

    return "\n".join(merged)


def clean_llm_output(text: str) -> str:

    text = text.replace("```java", "")
    text = text.replace("```", "")

    return text.strip()


def extract_assertions_only(text: str) -> str:

    cleaned = []

    forbidden_prefixes = (
        "```",
        "Here",
        "Explanation",
        "The assertion",
        "This assertion",
        "Note:",
        "#"
    )

    forbidden_patterns = [
        r"^\s*public\s+class",
        r"^\s*class\s+",
        r"^\s*import\s+",
        r"^\s*package\s+",
        r"^\s*@\w+",
        r"^\s*public\s+void",
        r"^\s*private\s+void",
        r"^\s*protected\s+void"
    ]

    for line in text.splitlines():

        stripped = line.strip()

        if not stripped:
            continue

        # remove markdown/prose
        if stripped.startswith(
            forbidden_prefixes
        ):
            continue

        # remove forbidden Java structures
        blocked = False

        for pattern in forbidden_patterns:

            if re.search(pattern, stripped):

                blocked = True
                break

        if blocked:
            continue

        # keep only statement-like lines
        if (
            stripped.endswith(";")
            or stripped.endswith("}")
            or stripped.endswith("{")
        ):

            cleaned.append(stripped)

    return "\n".join(cleaned)

# =========================================================
# PIT FEEDBACK EXTRACTION
# =========================================================

def extract_pit_mutant_details(
    stdout: str,
    csv_path: Path = None,
    limit: int = 25
) -> str:

    import csv
    from collections import defaultdict

    if not csv_path or not csv_path.exists():

        return "No mutations.csv found."

    try:

        # =====================================================
        # PARSE CSV
        # =====================================================

        survived = []

        with open(
            csv_path,
            newline="",
            encoding="utf-8"
        ) as f:

            reader = csv.reader(f)

            for row in reader:

                if len(row) < 7:
                    continue

                source_file = row[0]
                mutated_class = row[1]
                mutator = row[2]
                method = row[3]
                line = row[4]
                status = row[5]

                # ONLY SURVIVED MUTANTS
                if status != "SURVIVED":
                    continue

                short_mutator = (
                    mutator
                    .split(".")[-1]
                    .replace("Mutator", "")
                )

                survived.append({
                    "source_file": source_file,
                    "class": mutated_class,
                    "method": method,
                    "line": int(line),
                    "mutator": short_mutator
                })

        # =====================================================
        # EMPTY CASE
        # =====================================================

        if not survived:

            return (
                "No surviving mutants detected."
            )

        # =====================================================
        # GROUP BY METHOD + MUTATOR
        # =====================================================

        grouped = defaultdict(list)

        for mutant in survived:

            key = (
                mutant["method"],
                mutant["mutator"]
            )

            grouped[key].append(
                mutant["line"]
            )

        # =====================================================
        # BUILD FEEDBACK
        # =====================================================

        feedback = []

        feedback.append(
            "SURVIVING MUTANT SUMMARY:\n"
        )

        sorted_groups = sorted(
            grouped.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        for (
            (method, mutator),
            lines
        ) in sorted_groups[:limit]:

            unique_lines = sorted(
                set(lines)
            )

            feedback.append(
                f"- Method: {method}"
            )

            feedback.append(
                f"  Mutator: {mutator}"
            )

            feedback.append(
                f"  Survived mutants: {len(lines)}"
            )

            feedback.append(
                f"  Lines: {unique_lines}"
            )

            # =================================================
            # SEMANTIC HINTS
            # =================================================

            semantic_hint = None

            if "Conditional" in mutator:

                semantic_hint = (
                    "Conditional logic is weakly constrained."
                )

            elif "ReturnVals" in mutator:

                semantic_hint = (
                    "Return values are insufficiently verified."
                )

            elif "Null" in mutator:

                semantic_hint = (
                    "Null handling behavior is under-tested."
                )

            elif "Math" in mutator:

                semantic_hint = (
                    "Arithmetic behavior is weakly validated."
                )

            elif "MemberVariable" in mutator:

                semantic_hint = (
                    "Field propagation/state validation is weak."
                )

            elif "NonVoidMethodCall" in mutator:

                semantic_hint = (
                    "Method return semantics are weakly asserted."
                )

            elif "<init>" in method:

                semantic_hint = (
                    "Constructor behavior is under-constrained."
                )

            if semantic_hint:

                feedback.append(
                    f"  Insight: {semantic_hint}"
                )

            feedback.append("")

        return "\n".join(feedback)

    except Exception as e:

        return (
            f"Failed parsing mutations.csv: {e}"
        )


# =========================================================
# SANDBOX EXECUTION
# =========================================================

def execute_sandbox(state: AgentState) -> Dict:

    prediction = state.get("prediction", "")

    project_name = state.get("project_name")
    project_id = state.get("project_id")
    item_id = state.get("item_id")

    file_path = state.get("file_path", "")

    java_bin = JAVA_HOME / "bin" / "java"

    repo_path = CLONED_REPOS_DIR / str(project_name)

    miner_json_path = (
        DATA_PROJECT_DIR
        / "scripts"
        / "test-miner"
        / "output"
        / "miner"
        / f"{project_name}.json"
    )

    # =====================================================
    # CLEANUP
    # =====================================================

    for root, dirs, files in os.walk(repo_path):

        for file in files:

            if "STAR" in file and file.endswith(".java"):

                try:
                    os.remove(os.path.join(root, file))
                except Exception:
                    pass

    pit_reports_path = repo_path / "target" / "pit-reports"

    if pit_reports_path.exists():
        shutil.rmtree(pit_reports_path)

    try:

        subprocess.run(
            ["git", "checkout", "--", "src/test"],
            cwd=str(repo_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    except Exception:
        pass

    # =====================================================
    # CLEAN ASSERTIONS
    # =====================================================

    prediction = extract_assertions_only(
        prediction
    )

    # =====================================================
    # INJECTION
    # =====================================================

    prediction_file = (
        SANDBOX_DIR
        / f"{item_id}_agentic_prediction.txt"
    )

    with open(prediction_file, "w", encoding="utf-8") as f:

        f.write(f"[oracle]{prediction}[/oracle]")

    cmd_inject = [
        str(java_bin),
        "-jar",
        str(MUTATION_JAR_PATH),

        str(project_id),
        str(repo_path),
        str(miner_json_path),

        str(SANDBOX_DIR),

        str(
            DATA_PROJECT_DIR
            / "scripts"
            / "dataset"
        ),

        str(ORACLE_CONFIG_JSON),

        "INFERENCE",
        str(prediction_file)
    ]

    env = os.environ.copy()

    env["JAVA_HOME"] = str(JAVA_HOME)

    inject_res = subprocess.run(
        cmd_inject,
        cwd=str(repo_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=600
    )

    if inject_res.returncode != 0:

        return {
            "is_compiled": False,
            "compile_error": (
                "Injection failed due to invalid assertions."
            )
        }

    # =====================================================
    # CLEAN EXTRA STAR FILES
    # =====================================================

    if file_path:

        target_stem = Path(file_path).stem

        for root, dirs, files in os.walk(
            repo_path / "src" / "test"
        ):

            for file in files:

                if (
                    "STAR" in file
                    and not file.startswith(target_stem)
                ):

                    try:
                        os.remove(os.path.join(root, file))
                    except Exception:
                        pass

    # =====================================================
    # PIT CONFIG
    # =====================================================

    target_tests = "*"
    target_classes = "*"

    if file_path:

        parts = (
            str(file_path)
            .replace("\\", "/")
            .split("src/test/java/")
        )

        if len(parts) > 1:

            class_path = parts[-1].replace(".java", "")

            test_fqn = class_path.replace("/", ".")

            target_tests = test_fqn + "*"

            pkg_parts = test_fqn.split(".")

            class_name = pkg_parts[-1]

            if class_name.endswith("STARSplitTest"):
                class_name = class_name[:-13]

            if class_name.endswith("Test"):
                class_name = class_name[:-4]

            if len(pkg_parts) > 1:

                target_classes = (
                    ".".join(pkg_parts[:-1])
                    + "."
                    + class_name
                )

            else:
                target_classes = class_name

    # =====================================================
    # PIT EXECUTION
    # =====================================================

    cmd_mvn = [
        "mvn",

        "test-compile",

        "org.pitest:pitest-maven:mutationCoverage",

        "-DjvmArgs=--add-opens java.base/java.lang=ALL-UNNAMED "
        "--add-opens java.base/java.lang.reflect=ALL-UNNAMED",

        f"-DtargetClasses={target_classes}",
        f"-DtargetTests={target_tests}",

        "-Dmutators=ALL",

        "-DoutputFormats=CSV",

        "-DtimestampedReports=false",

        "-q"
    ]

    mvn_res = subprocess.run(
        cmd_mvn,
        cwd=str(repo_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=1200
    )

    stdout = mvn_res.stdout

    # =====================================================
    # COMPILE ERROR EXTRACTION
    # =====================================================

    if (
        "Compilation failure" in stdout
        or "COMPILATION ERROR" in stdout
    ):

        errors = []

        for line in stdout.splitlines():

            if (
                "[ERROR]" in line
                and "COMPILATION ERROR" not in line
            ):

                errors.append(line.strip())

        error_msg = (
            "\n".join(errors[:8])
            if errors
            else "Maven compilation failure."
        )

        return {
            "is_compiled": False,
            "compile_error": error_msg
        }

    # =====================================================
    # SCORE COMPUTATION
    # =====================================================

    mutations_csv_path = None

    target_dir = repo_path / "target" / "pit-reports"

    if target_dir.exists():

        for root, dirs, files in os.walk(target_dir):

            if "mutations.csv" in files:

                mutations_csv_path = (
                    Path(root) / "mutations.csv"
                )

                break


    

    mutation_score = 0.0
    test_strength = 0.0

    if (
        mutations_csv_path
        and mutations_csv_path.exists()
    ):

        generated = 0
        killed = 0

        with open(
            mutations_csv_path,
            "r",
            encoding="utf-8"
        ) as csvfile:

            reader = csv.reader(csvfile)

            for row in reader:

                if len(row) < 6:
                    continue

                status = row[5].strip()

                if status == "NO_COVERAGE":
                    continue

                generated += 1

                if status in (
                    "KILLED",
                    "TIMED_OUT",
                    "MEMORY_ERROR"
                ):

                    killed += 1

        if generated > 0:

            mutation_score = killed / generated
            test_strength = mutation_score

    return {
        "is_compiled": True,
        "stdout": stdout,
        "mutation_score": mutation_score,
        "test_strength": test_strength,
        "mutations_csv": (
            str(mutations_csv_path)
            if mutations_csv_path
            else None
        )
    }


# =========================================================
# AGENT NODES
# =========================================================

def summarizer_node(state: AgentState) -> Dict:

    print("      >> [AGENT] Summarizing Context...")

    system_msg = """
You are an expert Java mutation-testing semantic analyzer.

TASK:
Compress the provided Java project context into ONLY the information
required to generate high-quality mutation-killing assertions.

Your summary MUST preserve:
- executable test scope
- visible variables
- mutation-relevant APIs
- state propagation paths
- observable behaviors
- assertion opportunities
- branch-sensitive behaviors
- collection/map semantics
- null-handling semantics
- builder propagation semantics

OUTPUT FORMAT MUST BE EXACTLY:

TEST_SCOPE_VARIABLES:
- variable : type : origin

FOCAL_METHODS:
- methodSignature -> returnType

MUTATION_RELEVANT_APIS:
- ...

STATE_PROPAGATION:
- ...

ASSERTABLE_BEHAVIORS:
- ...

RETURN_SEMANTICS:
- ...

COLLECTION_SEMANTICS:
- ...

NULLABILITY:
- ...

MUTATION_HOTSPOTS:
- ...

CONSTRAINTS:
- ...

RULES:
1. ONLY summarize behavior explicitly derivable from the context.
2. NEVER invent APIs.
3. NEVER invent variables.
4. NEVER invent hidden state.
5. Preserve builder setter methods if they affect observable outputs.
6. Preserve methods relevant for surviving mutants.
7. Preserve APIs needed to construct meaningful object states.
8. Prefer semantic compression over implementation detail copying.
9. Omit irrelevant helper methods.
10. NO prose paragraphs.
11. NO markdown.
12. NO explanations outside bullet lists.
"""

    print("\n========= PROMPT CONTEXT =========\n")

    print(state["full_prompt_context"])

    print("\n==================================\n")

    response = llm.invoke([
        ("system", system_msg),
        ("human", state["full_prompt_context"])])

    summary = response.content.strip()

    print(f"         [MANIFEST]\n{summary}\n")

    return {
        "summary": summary
    }


def planner_node(state: AgentState) -> Dict:

    print("      >> [AGENT] Planning Assertions...")

    system_msg = """
You are a mutation testing strategist.

TASK:
Design assertion objectives that maximize PIT mutation score.

OUTPUT FORMAT MUST BE EXACTLY:

PRIMARY_TARGETS:
- ...

HIGH_VALUE_ASSERTIONS:
- ...

MUTATION_RISKS:
- ...

INVALID_ASSERTION_RISKS:
- ...

ASSERTION_PRIORITIES:
1. ...
2. ...
3. ...

RULES:
1. NO Java code.
2. NO assertions.
3. NO prose paragraphs.
4. Use compact bullet points only.
5. Focus on semantic verification.
6. Use ONLY visible variables and methods.
7. Flag invalid scope assumptions.
8. Prefer exact semantic checks over null checks.
"""

    context = (
        f"MANIFEST:\n"
        f"{state.get('summary', '')}\n\n"

        f"CRUCIAL CONTEXT:\n"
        f"{state['compact_prompt_context']}"
    )

    response = llm.invoke([
        ("system", system_msg),
        ("human", context)
    ])

    plan = response.content.strip()

    print(f"         [PLAN]\n{plan}\n")

    return {
        "plan": plan
    }


def coder_node(state: AgentState) -> Dict:

    print("      >> [AGENT] Generating Assertions...")

    manifest = state.get("summary", "")

    strategy = (
        state.get("improvement_plan")
        or state.get("plan")
        or ""
    )

    previous_assertions = state.get("prediction", "")

    prompt = (
        f"CRUCIAL CONTEXT:\n"
        f"{state['compact_prompt_context']}\n\n"

        f"ASSERTION MANIFEST:\n"
        f"{manifest}\n\n"

        f"ASSERTION STRATEGY:\n"
        f"{strategy}\n\n"
    )

    if previous_assertions:

        prompt += (
            f"PREVIOUS ASSERTIONS:\n"
            f"{previous_assertions}\n\n"
        )

    prompt += """
TASK:
Generate ONLY valid Java test-body code.

OUTPUT REQUIREMENTS:
1. Every semantic validation MUST use a JUnit assertion API.
2. NEVER output naked boolean expressions.
3. NEVER output passive method calls without assertions.
4. EVERY generated line must compile inside the target test body.
5. EVERY line must end with ';'

ALLOWED ASSERTION APIS:
- assertEquals(...)
- assertNotEquals(...)
- assertTrue(...)
- assertFalse(...)
- assertNotNull(...)
- assertNull(...)
- assertSame(...)
- assertNotSame(...)
- assertArrayEquals(...)
- fail(...)

ALLOWED CODE:
- local variable declarations
- inline object construction
- method calls
- collection/map inspection
- intermediate values required for assertions

FORBIDDEN CODE:
- naked expressions like:
  x != null;
  foo.isEmpty();
  a == b;

- helper methods
- classes
- imports
- packages
- annotations
- markdown
- prose
- comments
- explanation text
- method wrappers

SCOPE RULES:
1. Use ONLY identifiers explicitly visible in:
   - TEST_SCOPE_VARIABLES
   - visible constructors
   - visible methods
   - visible constants
   - visible fields

2. NEVER invent:
   - mocks
   - services
   - factories
   - builders not explicitly visible
   - hidden state
   - undeclared variables
   - unavailable APIs

3. NEVER call instance methods without a receiver object.

4. If TEST_SCOPE_VARIABLES is empty:
   - instantiate objects inline using ONLY visible constructors
   - prefer inline constructor expressions over temporary variables

5. You MAY instantiate ONLY classes whose constructors are explicitly visible.

ASSERTION QUALITY RULES:
1. Prefer semantic assertions over trivial assertions.
2. Prefer exact value verification over null checks.
3. Prefer behavioral verification over existence checks.
4. Prefer mutation-killing assertions over broad assertions.
5. Avoid duplicate assertions.
6. Avoid redundant assertions.
7. Avoid weak assertions that always pass.

MUTATION-GUIDED RULES:
1. Focus assertions on surviving mutant behavior.
2. Strengthen validation around:
   - return values
   - conditionals
   - field propagation
   - constructor effects
   - collection contents
   - null handling

3. If surviving mutants involve:
   - ReturnVals mutators:
       validate exact returned values

   - Conditional mutators:
       validate branch-sensitive behavior

   - MemberVariable mutators:
       validate propagated state and attributes

   - Null mutators:
       validate null-sensitive semantics

OUTPUT FORMAT:
- Output ONLY raw Java test-body statements.
- No surrounding text.
- No markdown fences.
- No explanations.

If uncertain, output fewer but stronger assertions.
"""

    system_msg = (
        "You are an expert Java mutation-testing engineer."
    )

    response = llm.invoke([
        ("system", system_msg),
        ("human", prompt)
    ])

    raw_output = clean_llm_output(
        response.content
    )

    raw_output = extract_assertions_only(
        raw_output
    )

    combined_assertions = merge_assertions(
        previous_assertions,
        raw_output
    )

    print(
        f"         [ASSERTIONS]\n"
        f"{combined_assertions}\n"
    )

    return {
        "prediction": combined_assertions
    }

def critic_node(state: AgentState) -> Dict:

    print("      >> [AGENT] Executing Sandbox...")

    result = execute_sandbox(state)

    test_strength = 0.0
    mutation_score = 0.0

    if not result.get("is_compiled", True):

        compile_error = result.get(
            "compile_error",
            "Unknown compilation error."
        )

        feedback = (
            "COMPILATION FAILURE\n\n"
            f"{compile_error}"
        )

    else:

        mutation_score = result.get(
            "mutation_score",
            0.0
        )

        test_strength = result.get(
            "test_strength",
            0.0
        )



        mutant_details = extract_pit_mutant_details(
            result.get("stdout", ""),
            (
                Path(result["mutations_csv"])
                if result.get("mutations_csv")
                else None
            )
        )

        print("\n         [SURVIVING MUTANTS RAW]\n")
        print(mutant_details)

        feedback = (
            f"TEST STRENGTH: {test_strength:.4f}\n"
            f"MUTATION SCORE: {mutation_score:.4f}\n"
            f"SURVIVING MUTANTS:\n"
            f"{mutant_details}"
        )

    print(f"         [CRITIC]\n{feedback}\n")

    best_score = state.get("best_score", 0.0)

    best_prediction = state.get(
        "best_prediction",
        ""
    )

    if test_strength > best_score:

        best_score = test_strength

        best_prediction = state.get(
            "prediction",
            ""
        )

    return {
        "iteration": state.get("iteration", 0) + 1,
        "latest_feedback": feedback,
        "best_score": best_score,
        "best_prediction": best_prediction,
        "mutation_score": mutation_score,
        "test_strength": test_strength,

    }

def improver_node(state: AgentState) -> Dict:

    print("      >> [AGENT] Refining Strategy...")

    system_msg = """
You are a senior Java mutation-testing reviewer.

TASK:
Analyze why the previous assertion set failed or was weak.

OUTPUT FORMAT MUST BE EXACTLY:

FAILURE_ANALYSIS:
- ...

SURVIVING_MUTANT_ANALYSIS:
- ...

MISSING_VALIDATIONS:
- ...

INVALID_SCOPE_USAGE:
- ...

NEXT_ASSERTION_GOALS:
- ...

RULES:
1. NO Java code.
2. NO prose paragraphs.
3. NO explanations outside bullet points.
4. Focus on semantic gaps.
5. Focus on scope errors.
6. Focus on compile failures.
7. Focus on surviving mutant causes.
"""

    context = (
        f"CRUCIAL CONTEXT:\n"
        f"{state['compact_prompt_context']}\n\n"

        f"MANIFEST:\n"
        f"{state.get('summary', '')}\n\n"

        f"PREVIOUS ASSERTIONS:\n"
        f"{state.get('prediction', '')}\n\n"

        f"EXECUTION FEEDBACK:\n"
        f"{state.get('latest_feedback', '')}"
    )

    response = llm.invoke([
        ("system", system_msg),
        ("human", context)
    ])

    improvement_plan = response.content.strip()

    print(
        f"         [REFINEMENT PLAN]\n"
        f"{improvement_plan}\n"
    )

    return {
        "improvement_plan": improvement_plan
    }


# =========================================================
# ROUTING
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

def route_critic(state: AgentState):

    best_score = state.get("best_score", 0.0)

    iteration = state.get("iteration", 0)

    max_iterations = state.get(
        "max_iterations",
        3
    )

    if best_score >= 1.0:
        return END

    if iteration >= max_iterations:
        return END

    return "improver"

# =========================================================
# GRAPH
# =========================================================

agent_workflow = StateGraph(AgentState)

agent_workflow.add_node(
    "summarizer",
    summarizer_node
)

agent_workflow.add_node(
    "planner",
    planner_node
)

agent_workflow.add_node(
    "coder",
    coder_node
)

agent_workflow.add_node(
    "critic",
    critic_node
)

agent_workflow.add_node(
    "improver",
    improver_node
)

agent_workflow.set_conditional_entry_point(
    route_start
)

agent_workflow.add_conditional_edges(
    "summarizer",
    route_summarizer
)

agent_workflow.add_edge(
    "planner",
    "coder"
)

agent_workflow.add_edge(
    "coder",
    "critic"
)

agent_workflow.add_conditional_edges(
    "critic",
    route_critic
)

agent_workflow.add_edge(
    "improver",
    "coder"
)

agent_app = agent_workflow.compile()


# =========================================================
# ENTRYPOINT
# =========================================================

def run_agentic_logic(state: AgentState) -> Dict:

    print(
        "    >> [AGENTIC] "
        "Starting Iterative Assertion Refinement..."
    )

    state["best_score"] = 0.0
    state["best_prediction"] = ""
    state["iteration"] = 0
    state["is_compiled"] = False

    final_state = agent_app.invoke(state)

    return {
        "prediction": final_state.get(
            "best_prediction",
            final_state.get("prediction", "")
        ),

        "mutation_score": final_state.get(
            "mutation_score",
            0.0
        ),

        "test_strength": final_state.get(
            "best_score",
            0.0
        )
    }