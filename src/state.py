from typing import TypedDict, Optional, List


class AgentState(TypedDict, total=False):

    # =========================================================
    # RAW DATA
    # =========================================================

    raw_datapoint: dict

    item_id: str

    project_name: str

    project_id: str

    info_file_path: str

    file_path: str

    method_signature: str

    ground_truth: str

    full_prompt_context: str

    compact_prompt_context: str

    # =========================================================
    # EXECUTION MODES
    # =========================================================

    run_mode: str

    use_summarizer: bool

    use_planner: bool

    use_evaluator_loop: bool

    # =========================================================
    # GENERATION
    # =========================================================

    prediction: Optional[str]

    # =========================================================
    # COMPILATION / EXECUTION
    # =========================================================

    is_compiled: bool

    is_broken: bool

    compile_time: float

    compile_error: str

    test_failure: str

    failure_stage: str

    is_evaluable: bool

    # =========================================================
    # MUTATION METRICS
    # =========================================================

    mutation_score: Optional[float]

    test_strength: Optional[float]

    covered_mutants: Optional[int]

    pit_metrics: dict

    # =========================================================
    # AGENTIC LOOP
    # =========================================================

    iteration: int

    max_iterations: int

    feedback_history: List[str]

    critic_feedback: str

    latest_feedback: str

    latest_feedback_signature: str

    plateau_count: int

    best_score: float
