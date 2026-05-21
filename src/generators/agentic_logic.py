import csv
import os
import re
import shutil
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from sandbox_utils import cleanup_star_files
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
    model="gpt-oss:120b-cloud",
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
# PIT REPORT PARSING
# =========================================================

PIT_STATUSES = {
    "KILLED",
    "SURVIVED",
    "NO_COVERAGE",
    "TIMED_OUT",
    "MEMORY_ERROR",
    "NON_VIABLE",
    "RUN_ERROR",
}

PIT_KILLED_STATUSES = {
    "KILLED",
    "TIMED_OUT",
    "MEMORY_ERROR",
}


@dataclass
class PitMutation:
    source_file: str
    mutated_class: str
    mutator: str
    method: str
    line: Optional[int]
    status: str
    killing_test: str = ""


@dataclass
class PitMetrics:
    generated: int = 0
    killed: int = 0
    no_coverage: int = 0
    covered: int = 0
    mutation_score: float = 0.0
    test_strength: float = 0.0

    def as_dict(self) -> Dict:
        return {
            "generated": self.generated,
            "killed": self.killed,
            "no_coverage": self.no_coverage,
            "covered": self.covered,
            "mutation_score": self.mutation_score,
            "test_strength": self.test_strength,
        }


@dataclass
class SandboxTarget:
    repo_path: Path
    test_file_path: str
    target_tests: str
    target_classes: str


def _status_index(row: List[str]) -> Optional[int]:
    for index, value in enumerate(row):
        if value.strip().upper() in PIT_STATUSES:
            return index

    return None


def _first_int(values: List[str]) -> Optional[int]:
    for value in values:
        stripped = value.strip()

        if stripped.isdigit():
            return int(stripped)

    return None


def _short_mutator(mutator: str) -> str:
    return (
        mutator
        .split(".")[-1]
        .replace("Mutator", "")
    )


def _looks_like_mutator(value: str) -> bool:
    lower = value.lower()

    return (
        "mutator" in lower
        or ".mutators." in lower
    )


def _parse_mutation_row(row: List[str]) -> Optional[PitMutation]:
    status_idx = _status_index(row)

    if status_idx is None:
        return None

    status = row[status_idx].strip().upper()

    if status_idx == 5 and len(row) >= 6:
        return PitMutation(
            source_file=row[0].strip(),
            mutated_class=row[1].strip(),
            mutator=row[2].strip(),
            method=row[3].strip(),
            line=_first_int([row[4]]),
            status=status,
            killing_test=(
                row[6].strip()
                if len(row) > 6
                else ""
            ),
        )

    if status_idx >= 9 and len(row) >= 10:
        return PitMutation(
            source_file=row[0].strip(),
            mutated_class=row[1].strip(),
            method=row[2].strip(),
            line=_first_int([row[4]]),
            mutator=row[5].strip(),
            status=status,
            killing_test=(
                row[8].strip()
                if len(row) > 8
                else ""
            ),
        )

    mutator_idx = next(
        (
            index
            for index, value in enumerate(row)
            if _looks_like_mutator(value)
        ),
        None
    )

    return PitMutation(
        source_file=(
            row[0].strip()
            if row
            else ""
        ),
        mutated_class=(
            row[1].strip()
            if len(row) > 1
            else ""
        ),
        mutator=(
            row[mutator_idx].strip()
            if mutator_idx is not None
            else ""
        ),
        method=(
            row[3].strip()
            if len(row) > 3
            else ""
        ),
        line=_first_int(row[:status_idx]),
        status=status,
        killing_test=(
            row[status_idx + 1].strip()
            if len(row) > status_idx + 1
            else ""
        ),
    )


def parse_pit_mutations(csv_path: Path) -> List[PitMutation]:
    if not csv_path or not csv_path.exists():
        return []

    mutations = []

    with open(
        csv_path,
        newline="",
        encoding="utf-8"
    ) as f:
        reader = csv.reader(f)

        for row in reader:
            mutation = _parse_mutation_row(row)

            if mutation:
                mutations.append(mutation)

    return mutations


def compute_pit_metrics(
    mutations: List[PitMutation]
) -> PitMetrics:
    generated = len(mutations)

    killed = sum(
        1
        for mutation in mutations
        if mutation.status in PIT_KILLED_STATUSES
    )

    no_coverage = sum(
        1
        for mutation in mutations
        if mutation.status == "NO_COVERAGE"
    )

    covered = (
        generated
        - no_coverage
    )

    mutation_score = (
        killed / generated
        if generated > 0
        else 0.0
    )

    test_strength = (
        killed / covered
        if covered > 0
        else 0.0
    )

    return PitMetrics(
        generated=generated,
        killed=killed,
        no_coverage=no_coverage,
        covered=covered,
        mutation_score=mutation_score,
        test_strength=test_strength,
    )


def extract_pit_mutant_details(
    stdout: str,
    csv_path: Path = None,
    limit: int = 25
) -> str:
    from collections import defaultdict

    try:
        mutations = parse_pit_mutations(csv_path)

        if not mutations:
            return "No mutations.csv rows found."

        survived = [
            mutation
            for mutation in mutations
            if mutation.status == "SURVIVED"
        ]

        if not survived:
            no_coverage_methods = sorted({
                mutation.method
                for mutation in mutations
                if mutation.status == "NO_COVERAGE"
                and mutation.method
            })

            if no_coverage_methods:
                methods = ", ".join(
                    no_coverage_methods[:limit]
                )

                return (
                    "No surviving mutants detected.\n"
                    "NO_COVERAGE mutants in methods: "
                    f"{methods}"
                )

            return "No surviving mutants detected."

        grouped = defaultdict(list)

        for mutant in survived:
            key = (
                mutant.method,
                _short_mutator(mutant.mutator)
            )

            grouped[key].append(
                mutant.line
            )

        feedback = [
            "SURVIVING MUTANT SUMMARY:\n"
        ]

        sorted_groups = sorted(
            grouped.items(),
            key=lambda item: len(item[1]),
            reverse=True
        )

        for (
            (method, mutator),
            lines
        ) in sorted_groups[:limit]:
            unique_lines = sorted({
                line
                for line in lines
                if line is not None
            })

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
# SANDBOX TARGETING / EXECUTION
# =========================================================

def normalize_test_file_path(file_path: str) -> str:
    file_path_str = str(file_path)

    if "STAR" not in file_path_str:
        return file_path_str

    file_path_str = re.sub(
        r"STAR(?:Split|Normalized)?Test",
        "",
        file_path_str
    )

    if not file_path_str.endswith("Test.java"):
        file_path_str = file_path_str.replace(
            ".java",
            "Test.java"
        )

    return file_path_str


def build_test_fqn(file_path: str) -> str:
    normalized_path = (
        normalize_test_file_path(file_path)
        .replace("\\", "/")
    )

    parts = normalized_path.split(
        "src/test/java/"
    )

    if len(parts) <= 1:
        return ""

    return (
        parts[-1]
        .replace(".java", "")
        .strip("/")
        .replace("/", ".")
    )


def fallback_target_classes(test_fqn: str) -> str:
    if not test_fqn:
        return "*.*"

    pkg_parts = test_fqn.split(".")

    class_name = pkg_parts[-1]

    if class_name.endswith("Test"):
        class_name = class_name[:-4]

    return (
        ".".join(pkg_parts[:-1])
        + "."
        + class_name
        + "*"
    )


def build_sandbox_target(
    state: AgentState
) -> SandboxTarget:
    project_name = state.get("project_name")
    file_path = state.get("file_path")

    repo_path = (
        CLONED_REPOS_DIR
        / str(project_name)
    )

    test_file_path = normalize_test_file_path(
        file_path
    )

    target_tests = build_test_fqn(
        test_file_path
    )

    target_classes = fallback_target_classes(
        target_tests
    )

    return SandboxTarget(
        repo_path=repo_path,
        test_file_path=test_file_path,
        target_tests=target_tests,
        target_classes=target_classes,
    )


def build_maven_env() -> Dict[str, str]:
    env = os.environ.copy()

    env["JAVA_HOME"] = str(
        JAVA_HOME
    )

    return env


def run_maven(
    repo_path: Path,
    command: List[str],
    env: Dict[str, str]
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(repo_path),
        capture_output=True,
        text=True,
        env=env,
        timeout=1200
    )


def maven_output(
    result: subprocess.CompletedProcess
) -> str:
    return (
        result.stdout
        + "\n"
        + result.stderr
    )


def is_build_failure(
    result: subprocess.CompletedProcess,
    stdout: str
) -> bool:
    return (
        result.returncode != 0
        or "BUILD FAILURE" in stdout
        or "COMPILATION ERROR" in stdout
    )


def remove_stale_target(repo_path: Path) -> None:
    target_dir = repo_path / "target"

    if target_dir.exists():
        shutil.rmtree(target_dir)


def find_pit_csv(repo_path: Path) -> Optional[Path]:
    pit_reports_path = (
        repo_path
        / "target"
        / "pit-reports"
    )

    direct_path = (
        pit_reports_path
        / "mutations.csv"
    )

    if direct_path.exists():
        return direct_path

    if not pit_reports_path.exists():
        return None

    candidates = list(
        pit_reports_path.rglob(
            "mutations.csv"
        )
    )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime
    )


def restore_test_backup(
    repo_path: Path,
    test_file_path: str
) -> None:
    if not test_file_path:
        return

    original_file = (
        repo_path
        / test_file_path
    )

    backup_file = (
        repo_path
        / (test_file_path + ".bak")
    )

    if not backup_file.exists():
        return

    if original_file.exists():
        original_file.unlink()

    backup_file.rename(
        original_file
    )


def restore_all_test_backups(
    repo_path: Path,
    exclude_backup: Optional[Path] = None
) -> None:
    test_root = (
        repo_path
        / "src"
        / "test"
        / "java"
    )

    if not test_root.exists():
        return

    excluded = (
        exclude_backup.resolve()
        if exclude_backup
        else None
    )

    for backup_file in test_root.rglob("*.java.bak"):
        if (
            excluded
            and backup_file.resolve() == excluded
        ):
            continue

        original_file = backup_file.with_suffix("")

        try:
            if original_file.exists():
                original_file.unlink()

            backup_file.rename(
                original_file
            )

        except Exception:
            pass


def pitest_command(
    target: SandboxTarget
) -> List[str]:
    return [
        "mvn",
        "org.pitest:pitest-maven:1.16.1:mutationCoverage",
        "-DjvmArgs=--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
        f"-DtargetClasses={target.target_classes}",
        f"-DtargetTests={target.target_tests}",
        "-Dmutators=ALL",
        "-DoutputFormats=CSV",
        "-DtimestampedReports=false"
    ]


def sandbox_result(
    metrics: PitMetrics,
    compile_time: float,
    is_compiled: bool,
    sandbox_feedback: str = "",
    surviving_mutants: str = "",
    stdout: str = "",
    target: Optional[SandboxTarget] = None,
    pit_csv_path: Optional[Path] = None,
) -> Dict:
    mutation_score = (
        float(metrics.mutation_score)
        if is_compiled
        else None
    )

    test_strength = (
        float(metrics.test_strength)
        if is_compiled
        else None
    )

    result = {
        "mutation_score": mutation_score,
        "test_strength": test_strength,
        "covered_mutants": metrics.covered,
        "pit_metrics": metrics.as_dict(),
        "compile_time": compile_time,
        "is_compiled": is_compiled,
        "sandbox_feedback": sandbox_feedback,
        "surviving_mutants": surviving_mutants,
        "stdout": stdout,
    }

    if target:
        result.update({
            "pit_target_classes": target.target_classes,
            "pit_target_tests": target.target_tests,
        })

    if pit_csv_path:
        result["pit_csv_path"] = str(
            pit_csv_path
        )

    return result


def execute_sandbox(state: AgentState) -> Dict:
    if state.get("is_broken"):
        return {
            "mutation_score": None,
            "test_strength": None,
            "covered_mutants": 0,
            "pit_metrics": PitMetrics().as_dict(),
        }

    target = build_sandbox_target(state)
    active_backup = (
        target.repo_path
        / (target.test_file_path + ".bak")
    )
    metrics = PitMetrics()
    stdout = ""
    compile_time = 0.0

    try:
        restore_all_test_backups(
            target.repo_path,
            exclude_backup=active_backup
        )

        cleanup_star_files(target.repo_path)
        remove_stale_target(target.repo_path)

        env = build_maven_env()

        compile_cmd = [
            "mvn",
            "test-compile",
            "-DskipTests"
        ]

        start_time = time.time()

        compile_result = run_maven(
            target.repo_path,
            compile_cmd,
            env
        )

        compile_time = (
            time.time()
            - start_time
        )

        compile_stdout = maven_output(
            compile_result
        )

        if is_build_failure(
            compile_result,
            compile_stdout
        ):
            print(
                "\n========= COMPILATION / BUILD FAILURE =========\n"
            )

            print(compile_stdout)

            print(
                "\n========= END FAILURE OUTPUT =========\n"
            )

            return sandbox_result(
                metrics=metrics,
                compile_time=compile_time,
                is_compiled=False,
                sandbox_feedback=compile_stdout,
                stdout=compile_stdout,
                target=target,
            )

        start_time = time.time()

        mutate_result = run_maven(
            target.repo_path,
            pitest_command(target),
            env
        )

        compile_time = (
            time.time()
            - start_time
        )

        stdout = maven_output(
            mutate_result
        )

        if is_build_failure(
            mutate_result,
            stdout
        ):
            print(
                "\n========= PIT / BUILD FAILURE =========\n"
            )

            print(stdout)

            print(
                "\n========= END FAILURE OUTPUT =========\n"
            )

            return sandbox_result(
                metrics=metrics,
                compile_time=compile_time,
                is_compiled=False,
                sandbox_feedback=stdout,
                stdout=stdout,
                target=target,
            )

        pit_csv_path = find_pit_csv(
            target.repo_path
        )

        if not pit_csv_path:
            feedback = (
                "mutations.csv NOT FOUND\n\n"
                f"{stdout}"
            )

            print(
                "\n========= mutations.csv NOT FOUND =========\n"
            )

            print(stdout)

            print(
                "\n========= END DEBUG =========\n"
            )

            return sandbox_result(
                metrics=metrics,
                compile_time=compile_time,
                is_compiled=False,
                sandbox_feedback=feedback,
                stdout=stdout,
                target=target,
            )

        mutations = parse_pit_mutations(
            pit_csv_path
        )

        metrics = compute_pit_metrics(
            mutations
        )

        surviving_summary = extract_pit_mutant_details(
            stdout,
            pit_csv_path
        )

        print(
            "\n========= SURVIVING MUTANTS =========\n"
        )

        print(surviving_summary)

        print(
            "\n========= END SURVIVING MUTANTS =========\n"
        )

        print(
            f"\n>> [PIT METRICS] "
            f"Generated={metrics.generated} "
            f"Killed={metrics.killed} "
            f"NoCoverage={metrics.no_coverage} "
            f"Covered={metrics.covered}"
        )

        print(
            f">> [FINAL SCORES] "
            f"TS={metrics.test_strength:.4f} "
            f"MS={metrics.mutation_score:.4f}"
        )

        return sandbox_result(
            metrics=metrics,
            compile_time=compile_time,
            is_compiled=True,
            sandbox_feedback="",
            surviving_mutants=surviving_summary,
            stdout=stdout,
            target=target,
            pit_csv_path=pit_csv_path,
        )

    except Exception:
        return sandbox_result(
            metrics=metrics,
            compile_time=compile_time,
            is_compiled=False,
            sandbox_feedback=traceback.format_exc(),
            surviving_mutants="",
            stdout=stdout,
            target=target,
        )

    finally:
        restore_test_backup(
            target.repo_path,
            target.test_file_path
        )

        restore_all_test_backups(
            target.repo_path
        )

        cleanup_star_files(
            target.repo_path
        )

        remove_stale_target(
            target.repo_path
        )
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

Infer likely Builder/fluent state-construction APIs
when constructor propagation patterns strongly imply them.

Example inference pattern:
- this.language = b.language;
- private String language;

→ likely Builder API:
- Builder.language(String)

Inferred APIs should ONLY be emitted when strongly
supported by visible field names and constructor propagation.

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

    # print("\n========= PROMPT CONTEXT =========\n")

    # print(state["full_prompt_context"])

    # print("\n==================================\n")

    response = llm.invoke([
        ("system", system_msg),
        ("human", state["full_prompt_context"])])

    summary = response.content.strip()

    # print(f"         [MANIFEST]\n{summary}\n")

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

    # print(f"         [PLAN]\n{plan}\n")

    return {
        "plan": plan
    }


def coder_node(state: AgentState) -> Dict:

    # print("      >> [AGENT] Generating Assertions...")

    manifest = state.get("summary", "")

    strategy = (
        state.get("improvement_plan")
        or state.get("plan")
        or ""
    )

    previous_assertions = state.get("prediction", "")

    prompt = ( ""
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
            "sandbox_feedback",
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



        mutant_details = result.get(
            "surviving_mutants",
            "No surviving mutant details."
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
