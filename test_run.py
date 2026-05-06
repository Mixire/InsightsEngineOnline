# Quick test script — save as test_run.py

import os
import sys

# macOS Fix for XGBoost / libomp
if sys.platform == "darwin":
    libomp_path = "/opt/homebrew/opt/libomp/lib"
    if os.path.exists(libomp_path):
        os.environ["DYLD_LIBRARY_PATH"] = libomp_path + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")

from agent.core_agent import AutonomousAnalyticsAgent

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Note: You need to have data/sample_data.csv downloaded for this to work.
# curl -o data/sample_data.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

agent = AutonomousAnalyticsAgent()

# Use any real CSV — e.g., Titanic, Iris, Housing prices
report = agent.run(
    dataset_path="data/sample_data.csv",
    user_goal="predict survival based on passenger features"
)

print(f"Report: {report}")
