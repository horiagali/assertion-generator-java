import csv
import os
import re
import shutil
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET
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


def clean_prediction_body(prediction: str) -> str:
    cleaned_prediction = prediction.strip()

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

    return cleaned_prediction


def assertion_static_import(content: str) -> str:
    is_junit5 = (
        "jupiter" in content
        or "org.junit.jupiter" in content
    )

    if is_junit5:
        return (
            "\nimport static "
            "org.junit.jupiter.api.Assertions.*;"
        )

    return (
        "\nimport static "
        "org.junit.Assert.*;"
    )


def add_assertion_imports(content: str) -> str:
    static_imports = assertion_static_import(
        content
    )

    if static_imports in content:
        return content

    package_match = re.search(
        r'package\s+[\w.]+;',
        content
    )

    if package_match:
        return content.replace(
            package_match.group(0),
            package_match.group(0)
            + static_imports
        )

    return (
        static_imports
        + "\n"
        + content
    )


def build_injected_method(
    state: AgentState,
    prediction: str
) -> str:
    dp = state.get("raw_datapoint") or {}

    test_prefix = dp.get(
        "testPrefix",
        {}
    )

    method_body = test_prefix.get(
        "body",
        ""
    )

    cleaned_prediction = clean_prediction_body(
        prediction
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

    return (
        f"\n    @Test\n"
        f"    {method_stub}\n"
    )


def reset_repo_test_file(
    repo_path: Path,
    test_file_path: str
) -> None:
    original_file = (
        repo_path
        / test_file_path
    )

    if not original_file.exists():
        return

    try:
        relative_git_path = original_file.relative_to(
            repo_path
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


def inject_prediction_into_test_file(
    state: AgentState
) -> Dict:
    prediction = state.get("prediction")

    if not prediction:
        return {
            "is_compiled": False,
            "sandbox_feedback": "Missing prediction."
        }

    target = build_sandbox_target(state)
    repo_path = target.repo_path
    test_file_path = target.test_file_path

    restore_all_test_backups(
        repo_path
    )

    cleanup_star_files(repo_path)
    reset_repo_test_file(
        repo_path,
        test_file_path
    )

    original_file = (
        repo_path
        / test_file_path
    )

    print(
        f"    >> [INJECTION TARGET] "
        f"{original_file}"
    )

    if not original_file.exists():
        return {
            "is_compiled": False,
            "sandbox_feedback": (
                "Original test file not found at path: "
                f"{original_file}"
            )
        }

    try:
        backup_file = (
            repo_path
            / (test_file_path + ".bak")
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

        content = re.sub(
            r'@Test\b',
            '// @Test',
            content
        )

        content = add_assertion_imports(
            content
        )

        full_method_injection = build_injected_method(
            state,
            prediction
        )

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
            "file_path": test_file_path
        }

    except Exception:
        return {
            "is_compiled": False,
            "sandbox_feedback": traceback.format_exc()
        }


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


def strip_ansi(text: str) -> str:
    return re.sub(
        r"\x1b\[[0-9;]*m",
        "",
        text
    )


def extract_maven_compile_error(stdout: str) -> str:
    cleaned_stdout = strip_ansi(stdout)

    error_lines = []

    for line in cleaned_stdout.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if "error:" in stripped:
            error_lines.append(stripped)
            continue

        if re.search(
            r"^\[ERROR\]\s+/.+\.java(?::|\[)",
            stripped
        ):
            error_lines.append(stripped)

    if not error_lines:
        return ""

    deduped = []

    for line in error_lines:
        if line not in deduped:
            deduped.append(line)

    return "\n".join(deduped)


def truncate_lines(
    text: str,
    max_lines: int = 140,
    max_chars: int = 12000
) -> str:
    if not text:
        return ""

    lines = text.splitlines()
    truncated = "\n".join(lines[:max_lines])

    if len(lines) > max_lines:
        truncated += (
            f"\n... truncated {len(lines) - max_lines} lines ..."
        )

    if len(truncated) > max_chars:
        truncated = (
            truncated[:max_chars]
            + "\n... truncated output ..."
        )

    return truncated


def normalize_failure_line(line: str) -> str:
    stripped = line.strip()

    stripped = re.sub(
        r"^\[ERROR\]\s+",
        "",
        stripped
    )

    stripped = re.sub(
        r"^\[\w+\]\s+",
        "",
        stripped
    )

    return stripped


def is_assertion_failure_line(line: str) -> bool:
    normalized = normalize_failure_line(
        line
    )

    markers = (
        "org.opentest4j.AssertionFailedError",
        "junit.framework.AssertionFailedError",
        "org.junit.ComparisonFailure",
        "junit.framework.ComparisonFailure",
        "java.lang.AssertionError",
        "AssertionFailedError",
        "ComparisonFailure",
        "AssertionError",
    )

    if any(
        marker in normalized
        for marker in markers
    ):
        return True

    lower = normalized.lower()

    return (
        "expected:" in lower
        and "but was:" in lower
    )


def is_stack_trace_line(line: str) -> bool:
    normalized = normalize_failure_line(
        line
    )

    return (
        normalized.startswith("at ")
        or normalized.startswith("Caused by:")
        or normalized.startswith("Suppressed:")
        or bool(
            re.match(
                r"^[\w.$]+(?:Exception|Error):",
                normalized
            )
        )
    )


def extract_assertion_stack_trace(
    text: str,
    max_lines: int = 80,
    max_chars: int = 8000
) -> str:
    cleaned_text = strip_ansi(
        text or ""
    )

    lines = cleaned_text.splitlines()

    start_index = None

    for index, line in enumerate(lines):
        if is_assertion_failure_line(line):
            start_index = index
            break

    if start_index is None:
        return ""

    selected = []

    for line in lines[start_index:]:
        normalized = normalize_failure_line(
            line
        )

        if not normalized:
            if selected:
                break

            continue

        if (
            selected
            and not is_stack_trace_line(normalized)
            and not is_assertion_failure_line(normalized)
            and not normalized.startswith("...")
        ):
            break

        selected.append(normalized)

        if len(selected) >= max_lines:
            selected.append(
                "... truncated stack trace ..."
            )
            break

    return truncate_lines(
        "\n".join(selected),
        max_lines=max_lines,
        max_chars=max_chars
    )


def is_pit_green_suite_failure(stdout: str) -> bool:
    return (
        "did not pass without mutation" in stdout
        or "Mutation testing requires a green suite" in stdout
        or "Tests failing without mutation" in stdout
    )


def extract_pit_green_suite_failure(stdout: str) -> str:
    cleaned_stdout = strip_ansi(stdout)
    lines = cleaned_stdout.splitlines()
    details = []
    capture = False

    for line in lines:
        stripped = line.strip()

        if (
            "did not pass without mutation" in stripped
            or "Mutation testing requires a green suite" in stripped
            or "Tests failing without mutation" in stripped
        ):
            capture = True

        if capture and stripped:
            details.append(stripped)

        if (
            capture
            and stripped.startswith("Description [")
        ):
            continue

    if details:
        return truncate_lines(
            "\n".join(details),
            max_lines=40,
            max_chars=4000
        )

    return ""


def target_method_name(state: AgentState) -> str:
    item_id = str(
        state.get(
            "item_id",
            ""
        )
    ).strip()

    if item_id:
        return item_id

    dp = state.get("raw_datapoint") or {}
    test_prefix = dp.get(
        "testPrefix",
        {}
    )

    return str(
        test_prefix.get(
            "identifier",
            ""
        )
    ).strip()


def surefire_test_selector(
    target: SandboxTarget,
    state: AgentState
) -> str:
    method_name = target_method_name(
        state
    )

    if target.target_tests and method_name:
        return (
            f"{target.target_tests}"
            f"#{method_name}"
        )

    return target.target_tests


def surefire_command(
    target: SandboxTarget,
    state: AgentState
) -> List[str]:
    command = [
        "mvn",
        "test",
        "-DfailIfNoTests=false",
        "-DfailIfNoSpecifiedTests=false"
    ]

    selector = surefire_test_selector(
        target,
        state
    )

    if selector:
        command.insert(
            2,
            f"-Dtest={selector}"
        )

    return command


def collect_surefire_text_reports(
    repo_path: Path,
    target_tests: str
) -> str:
    reports_dir = (
        repo_path
        / "target"
        / "surefire-reports"
    )

    if not reports_dir.exists():
        return ""

    candidates = []

    if target_tests:
        candidates.extend([
            reports_dir / f"{target_tests}.txt",
            reports_dir / f"TEST-{target_tests}.xml",
        ])

    candidates.extend(
        sorted(
            reports_dir.glob("*.txt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True
        )[:3]
    )

    chunks = []
    seen = set()

    for path in candidates:
        if not path.exists():
            continue

        resolved = path.resolve()

        if resolved in seen:
            continue

        seen.add(resolved)

        if path.suffix == ".xml":
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="replace"
            )
        except Exception:
            continue

        if text.strip():
            assertion_trace = extract_assertion_stack_trace(
                text
            )

            if assertion_trace:
                text = (
                    "ASSERTION STACK TRACE:\n"
                    f"{assertion_trace}\n\n"
                    "FULL REPORT EXCERPT:\n"
                    f"{text}"
                )

            chunks.append(
                f"--- {path.name} ---\n"
                f"{truncate_lines(text)}"
            )

    return "\n\n".join(chunks)


def collect_surefire_xml_failures(
    repo_path: Path,
    target_tests: str,
    method_name: str
) -> str:
    reports_dir = (
        repo_path
        / "target"
        / "surefire-reports"
    )

    if not reports_dir.exists():
        return ""

    candidates = []

    if target_tests:
        candidates.append(
            reports_dir / f"TEST-{target_tests}.xml"
        )

    candidates.extend(
        sorted(
            reports_dir.glob("TEST-*.xml"),
            key=lambda path: path.stat().st_mtime,
            reverse=True
        )[:3]
    )

    chunks = []
    seen = set()

    for path in candidates:
        if not path.exists():
            continue

        resolved = path.resolve()

        if resolved in seen:
            continue

        seen.add(resolved)

        try:
            root = ET.parse(path).getroot()
        except Exception:
            continue

        for testcase in root.iter("testcase"):
            name = testcase.attrib.get(
                "name",
                ""
            )

            if method_name and name != method_name:
                continue

            for child in testcase:
                if child.tag not in {
                    "failure",
                    "error"
                }:
                    continue

                message = child.attrib.get(
                    "message",
                    ""
                )

                child_type = child.attrib.get(
                    "type",
                    child.tag
                )

                body = child.text or ""

                exception_line = child_type

                if message:
                    exception_line = (
                        f"{exception_line}: {message}"
                    )

                stack_trace = (
                    extract_assertion_stack_trace(body)
                    or truncate_lines(
                        body,
                        max_lines=80,
                        max_chars=8000
                    )
                )

                chunks.append(
                    "\n".join([
                        f"--- {path.name} :: {name} ---",
                        f"{child.tag.upper()}: {child_type}",
                        f"MESSAGE: {message}",
                        f"EXCEPTION: {exception_line}",
                        "STACKTRACE:",
                        stack_trace
                    ]).strip()
                )

    return "\n\n".join(chunks)


def extract_maven_test_failure_lines(stdout: str) -> str:
    cleaned_stdout = strip_ansi(stdout)
    lines = cleaned_stdout.splitlines()
    selected = []
    capture_until = -1

    interesting_patterns = (
        "<<< FAILURE!",
        "<<< ERROR!",
        "[ERROR] Failures:",
        "[ERROR] Errors:",
        "[ERROR] Tests run:",
        "There are test failures",
        "Please refer to",
    )

    for index, line in enumerate(lines):
        if any(
            pattern in line
            for pattern in interesting_patterns
        ):
            capture_until = max(
                capture_until,
                index + 25
            )

        if index <= capture_until:
            selected.append(line)

    if not selected:
        return ""

    return truncate_lines(
        "\n".join(selected),
        max_lines=100,
        max_chars=9000
    )


def collect_green_suite_failure_details(
    repo_path: Path,
    target: SandboxTarget,
    state: AgentState,
    stdout: str
) -> str:
    method_name = target_method_name(
        state
    )

    xml_details = collect_surefire_xml_failures(
        repo_path,
        target.target_tests,
        method_name
    )

    text_details = collect_surefire_text_reports(
        repo_path,
        target.target_tests
    )

    output_details = extract_maven_test_failure_lines(
        stdout
    )

    assertion_trace = extract_assertion_stack_trace(
        "\n\n".join([
            xml_details,
            text_details,
            output_details,
            stdout
        ])
    )

    sections = []

    if assertion_trace:
        sections.append(
            "ASSERTION STACK TRACE:\n"
            f"{assertion_trace}"
        )

    if xml_details:
        sections.append(
            "SUREFIRE XML FAILURE:\n"
            f"{xml_details}"
        )

    if text_details:
        sections.append(
            "SUREFIRE TEXT REPORT:\n"
            f"{text_details}"
        )

    if output_details:
        sections.append(
            "MAVEN TEST OUTPUT:\n"
            f"{output_details}"
        )

    if not sections:
        sections.append(
            "MAVEN TEST OUTPUT:\n"
            f"{truncate_lines(stdout)}"
        )

    return "\n\n".join(sections)


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
    compile_error: str = "",
    test_failure: str = "",
    failure_stage: str = "",
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
        "compile_error": compile_error,
        "test_failure": test_failure,
        "failure_stage": failure_stage,
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
            compile_error = extract_maven_compile_error(
                compile_stdout
            )

            # print(
            #     "\n========= COMPILATION / BUILD FAILURE =========\n"
            # )

            # print(compile_stdout)

            # print(
            #     "\n========= END FAILURE OUTPUT =========\n"
            # )

            return sandbox_result(
                metrics=metrics,
                compile_time=compile_time,
                is_compiled=False,
                sandbox_feedback=compile_stdout,
                compile_error=compile_error,
                failure_stage="compile",
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
            if is_pit_green_suite_failure(
                stdout
            ):
                green_result = run_maven(
                    target.repo_path,
                    surefire_command(target, state),
                    env
                )

                green_stdout = maven_output(
                    green_result
                )

                pit_failure = extract_pit_green_suite_failure(
                    stdout
                )

                test_failure = collect_green_suite_failure_details(
                    target.repo_path,
                    target,
                    state,
                    green_stdout
                )

                feedback = (
                    "PIT GREEN SUITE FAILURE\n\n"
                    "PIT BASELINE CONTEXT:\n"
                    f"{pit_failure}\n\n"
                    "TARGET TEST FAILURE DETAILS:\n"
                    f"{test_failure}"
                )

                return sandbox_result(
                    metrics=metrics,
                    compile_time=compile_time,
                    is_compiled=False,
                    sandbox_feedback=feedback,
                    test_failure=test_failure,
                    failure_stage="green_suite",
                    stdout=(
                        stdout
                        + "\n\n"
                        + green_stdout
                    ),
                    target=target,
                )

            compile_error = extract_maven_compile_error(
                stdout
            )

            # print(
            #     "\n========= PIT / BUILD FAILURE =========\n"
            # )

            # print(stdout)

            # print(
            #     "\n========= END FAILURE OUTPUT =========\n"
            # )

            return sandbox_result(
                metrics=metrics,
                compile_time=compile_time,
                is_compiled=False,
                sandbox_feedback=stdout,
                compile_error=compile_error,
                failure_stage="pit",
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
                compile_error=extract_maven_compile_error(
                    feedback
                ),
                failure_stage="pit_missing_csv",
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
            compile_error=traceback.format_exc(),
            failure_stage="exception",
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

    previous_code = state.get("prediction", "")

    prompt = ( ""
        f"CRUCIAL CONTEXT:\n"
        f"{state['compact_prompt_context']}\n\n"

        f"ASSERTION MANIFEST:\n"
        f"{manifest}\n\n"

        f"ASSERTION STRATEGY:\n"
        f"{strategy}\n\n"
    )

    if previous_code:

        prompt += (
            f"PREVIOUS CODE (FOR REFERENCE/FIXING):\n"
            f"{previous_code}\n\n"
        )

    prompt += """ 
TASK:
Generate ONLY valid Java test-body code.
Write the COMPLETE, fully corrected assertion block.
If previous code caused a compilation or runtime failure,
REMOVE or FIX the failing lines.
The output replaces the entire previous assertion block.

OUTPUT REQUIREMENTS:
1. Every semantic validation MUST use a JUnit assertion API.
2. NEVER output naked boolean expressions.
3. NEVER output passive method calls without assertions.
4. EVERY generated line must compile inside the target test body.
5. EVERY line must end with ';'
6. EVERY assertion must pass on the original unmutated implementation.
7. If feedback reports BASELINE TEST FAILURE, prioritize green-suite
   correctness over mutation score.

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
8. Avoid object-wide toString() or broad string contains assertions
   unless the focal behavior is explicitly a toString contract and the
   exact expected string is visible in the context.
9. Prefer dedicated getters, fields, collections, maps, or returned values
   over indirect debug/string representations.
10. Never infer expected values only from the test method name.

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

    new_prediction = clean_llm_output(
        response.content
    )

    return {
        "prediction": new_prediction
    }

def critic_node(state: AgentState) -> Dict:

    print("      >> [AGENT] Executing Sandbox...")

    injection_result = inject_prediction_into_test_file(
        state
    )

    if not injection_result.get(
        "is_compiled",
        False
    ):
        result = {
            "is_compiled": False,
            "mutation_score": None,
            "test_strength": None,
            "compile_error": injection_result.get(
                "sandbox_feedback",
                "Failed to inject prediction."
            ),
            "sandbox_feedback": injection_result.get(
                "sandbox_feedback",
                "Failed to inject prediction."
            ),
            "surviving_mutants": "",
            "pit_metrics": {},
            "failure_stage": "injection",
            "test_failure": "",
        }
    else:
        result = execute_sandbox(state)

    test_strength = 0.0
    mutation_score = 0.0

    if not result.get("is_compiled", True):

        failure_stage = result.get(
            "failure_stage",
            "compile"
        )

        if failure_stage == "green_suite":
            test_failure = result.get(
                "test_failure",
                ""
            )

            if not test_failure:
                test_failure = result.get(
                    "sandbox_feedback",
                    "Generated test failed before mutation analysis."
                )

            feedback = (
                "BASELINE TEST FAILURE\n\n"
                "The assertion block compiled, but failed on the "
                "original unmutated code. The next revision must "
                "remove or fix the failing assertion before trying "
                "to kill more mutants.\n\n"
                "RAW TEST FAILURE:\n"
                f"{test_failure}"
            )

        elif failure_stage == "pit":
            pit_feedback = result.get(
                "sandbox_feedback",
                "Unknown PIT failure."
            )

            feedback = (
                "PIT EXECUTION FAILURE\n\n"
                "RAW PIT FAILURE:\n"
                f"{truncate_lines(pit_feedback)}"
            )

        else:
            compile_error = result.get(
                "compile_error",
                ""
            )

            if not compile_error:
                compile_error = result.get(
                    "sandbox_feedback",
                    "Unknown compilation error."
                )

            feedback = (
                "COMPILATION FAILURE\n\n"
                "RAW COMPILE ERROR:\n"
                f"{truncate_lines(compile_error)}"
            )

    else:

        mutation_score = (
            result.get(
                "mutation_score",
                0.0
            )
            or 0.0
        )

        test_strength = (
            result.get(
                "test_strength",
                0.0
            )
            or 0.0
        )



        mutant_details = result.get(
            "surviving_mutants",
            "No surviving mutant details."
        )


        feedback = (
            f"TEST STRENGTH: {test_strength:.4f}\n"
            f"MUTATION SCORE: {mutation_score:.4f}\n"
            f"SURVIVING MUTANTS:\n"
            f"{mutant_details}"
        )

    print(f"         [CRITIC]\n{feedback}\n")

    feedback_signature = (
        f"{result.get('is_compiled', False)}|"
        f"{test_strength:.6f}|"
        f"{mutation_score:.6f}|"
        f"{result.get('surviving_mutants', '')}"
    )

    previous_signature = state.get(
        "latest_feedback_signature",
        ""
    )

    plateau_count = (
        state.get("plateau_count", 0) + 1
        if previous_signature == feedback_signature
        else 0
    )

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
        "latest_feedback_signature": feedback_signature,
        "plateau_count": plateau_count,
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

NEXT_ASSERTION_GOALS: (keep in mind that we can only write assertions, not write any new tests or modify any other code)
- ...

RULES:
1. NO Java code.
2. NO prose paragraphs.
3. NO explanations outside bullet points.
4. Focus on semantic gaps.
5. Focus on scope errors.
6. Focus on compile failures.
7. Focus on surviving mutant causes.
8. If feedback says BASELINE TEST FAILURE:
   - treat the previous assertions as invalid on the original code
   - identify the exact failing assertion or exception from RAW TEST FAILURE
   - recommend removing or correcting failing lines before adding strength
   - do not discuss surviving mutants until the baseline test is green
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

    plateau_count = state.get(
        "plateau_count",
        0
    )

    max_iterations = state.get(
        "max_iterations",
        3
    )

    if best_score >= 1.0:
        return END

    if not state.get("use_evaluator_loop", False):
        return END

    if plateau_count >= 1:
        print(
            "      >> [AGENT] "
            "Stopping refinement: sandbox feedback plateau."
        )

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
    state["plateau_count"] = 0
    state["latest_feedback_signature"] = ""
    state["is_compiled"] = False

    final_state = agent_app.invoke(state)

    best_prediction = (
        final_state.get("best_prediction")
        or final_state.get("prediction", "")
    )

    return {
        "prediction": best_prediction,

        "mutation_score": final_state.get(
            "mutation_score",
            0.0
        ),

        "test_strength": final_state.get(
            "best_score",
            0.0
        )
    }
