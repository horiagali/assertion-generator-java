import re
import csv
from pathlib import Path

def parse_maven_compile_log(stdout: str) -> str:
    """Extracts the specific error message from a Maven compilation failure."""
    lines = stdout.splitlines()
    error_lines = [l for l in lines if "[ERROR]" in l and "cannot find symbol" in l]
    if not error_lines:
        error_lines = [l for l in lines if "[ERROR]" in l]
    return "\n".join(error_lines[:5])

def parse_pitest_csv(csv_path: Path) -> str:
    if not csv_path or not csv_path.exists():
        return "Mutation CSV not found"
    
    survived_details = []
    no_coverage_details = []
    
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6: continue
                
                # Pitest CSV: 0:File, 1:Class, 2:Mutator, 3:Method, 4:Line, 5:Status
                status = row[5] 
                method = row[3]
                line = row[4]
                mutator = row[2].split(".")[-1]

                if status == "SURVIVED":
                    survived_details.append(f"Line {line} ({method}): {mutator} survived")
                elif status == "NO_COVERAGE":
                    # We want to tell the agent which methods it isn't even touching
                    no_coverage_details.append(method)
                    
    except Exception as e:
        return f"Error parsing CSV: {str(e)}"

    summary = []
    if survived_details:
        summary.append("SURVIVED (Test passed but should have failed):")
        summary.extend(list(set(survived_details))[:5])
    
    if no_coverage_details:
        methods = ", ".join(list(set(no_coverage_details)))
        summary.append(f"NO_COVERAGE (You are not calling these methods): {methods}")

    return "\n".join(summary) if summary else "NO_SURVIVORS_REPORTED"