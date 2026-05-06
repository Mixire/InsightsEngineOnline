from tools.data_loader import load_data, get_data_summary
from tools.data_cleaner import auto_clean, encode_target
from tools.eda_tool import run_eda
from tools.ml_selector import select_ml_approach
from tools.ml_runner import run_model
from tools.viz_tool import generate_all_charts
from tools.report_generator import generate_report

# Tool registry maps string names to actual functions
TOOL_REGISTRY = {
    "load_data": load_data,
    "get_data_summary": get_data_summary,
    "auto_clean": auto_clean,
    "encode_target": encode_target,
    "run_eda": run_eda,
    "select_ml_approach": select_ml_approach,
    "run_model": run_model,
    "generate_all_charts": generate_all_charts,
    "generate_report": generate_report,
}

def get_tool(name: str):
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Tool '{name}' not found in registry.")
    return TOOL_REGISTRY[name]
