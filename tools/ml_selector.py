import json
import google.generativeai as genai
from config import GEMINI_API_KEY, LLM_MODEL, MAX_TOKENS, PROMPTS_DIR
from loguru import logger


def select_ml_approach(data_summary: dict, user_goal: str, columns: list,
                        target_column: str = None, shape: tuple = None) -> dict:
    """
    Uses Google Gemini API to select the best ML task and algorithm for the given data.
    Returns a dict with task, algorithm, reason, target_column, feature_columns.
    """
    with open(f"{PROMPTS_DIR}ml_selection_prompt.txt") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(
        data_summary=json.dumps(data_summary, indent=2),
        user_goal=user_goal,
        columns=columns,
        target_column=target_column or "Not specified",
        shape=shape or "Unknown"
    )

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(LLM_MODEL)
    
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=MAX_TOKENS,
            temperature=0.1,
        )
    )

    raw = response.text.strip()
    # Strip markdown if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines[0].startswith("```json"):
            raw = "\n".join(lines[1:-1])
        elif lines[0].startswith("```"):
            raw = "\n".join(lines[1:-1])
    
    try:
        result = json.loads(raw.strip())
        logger.success(f"ML selection: {result['task']} → {result['algorithm']}")
        return result
    except Exception as e:
        logger.error(f"Failed to parse JSON from Gemini: {e}")
        logger.debug(f"Raw output: {raw}")
        raise
