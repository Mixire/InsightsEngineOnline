import os
from dotenv import load_dotenv

load_dotenv()

# Cloud LLM Config (Google Gemini)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL = "gemini-2.0-flash"  # Latest and fastest model
MAX_TOKENS = 4096

# Agent Config
MAX_ITERATIONS = 10           # Max agent loop iterations before forced stop
ML_ACCURACY_THRESHOLD = 0.75  # Agent retries if model accuracy is below this
ENABLE_CODE_EXECUTION = True  # Allow agent to write and execute Python code

# Paths
DATA_DIR = "data/"
OUTPUT_DIR = "outputs/"
CHARTS_DIR = "outputs/charts/"
REPORTS_DIR = "outputs/reports/"
PROMPTS_DIR = "prompts/"
