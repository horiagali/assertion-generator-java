from typing import Dict, Any, Optional, Literal, List
from typing_extensions import TypedDict


class AgentState(TypedDict):

    raw_datapoint: Dict[str, Any]

    item_id: str

    project_name: str

    project_id: str

    info_file_path: str

    prompt_context: str

    ground_truth: str

    file_path: str

    method_signature: str

    is_quarantined: bool

    prediction: Optional[str]

    is_compiled: bool

    mutation_score: Optional[float]

    run_mode: Literal["human", "oneshot", "agentic"]

    # =====================================================
    # ABLATION FLAGS
    # =====================================================

    use_summarizer: bool

    use_planner: bool

    use_evaluator_loop: bool

    # =====================================================
    # AGENT MEMORY
    # =====================================================

    summary: Optional[str]

    plan: Optional[str]

    improvement_plan: Optional[str]

    latest_feedback: Optional[str]

    available_variables: List[str]

    iteration: int

    max_iterations: int

    feedback_history: List[str]

    best_prediction: Optional[str]

    best_score: float