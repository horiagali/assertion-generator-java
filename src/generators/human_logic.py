from typing import Dict
from state import AgentState

def run_human_logic(state: AgentState) -> Dict:
    print("    >> [HUMAN] Injecting original ground truth assertions.")
    ground_truth = state.get("ground_truth", "")
    return {"prediction": ground_truth}