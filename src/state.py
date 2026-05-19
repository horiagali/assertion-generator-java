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

    prompt_context: str

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

    # =========================================================
    # MUTATION METRICS
    # =========================================================

    mutation_score: Optional[float]

    test_strength: Optional[float]

    # =========================================================
    # AGENTIC LOOP
    # =========================================================

    iteration: int

    max_iterations: int

    feedback_history: List[str]

    critic_feedback: str

    best_score: float