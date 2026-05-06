from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentMemory:
    """Stores the full state of the agent across its reasoning loop."""
    dataset_path: str = ""
    user_goal: str = ""
    df_raw: Any = None
    df_clean: Any = None
    data_summary: dict = field(default_factory=dict)
    eda_results: dict = field(default_factory=dict)
    eda_analysis: dict = field(default_factory=dict)
    ml_selection: dict = field(default_factory=dict)
    ml_results: dict = field(default_factory=dict)
    insights: str = ""
    report_path: str = ""
    iteration: int = 0
    reasoning_log: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def log(self, step: str, message: str):
        entry = f"[Step {self.iteration} | {step}] {message}"
        self.reasoning_log.append(entry)
        print(entry)
