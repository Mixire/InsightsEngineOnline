import json
import google.generativeai as genai
from rich.console import Console
from loguru import logger
from config import GEMINI_API_KEY, LLM_MODEL, MAX_TOKENS, MAX_ITERATIONS, ML_ACCURACY_THRESHOLD, PROMPTS_DIR
from agent.memory import AgentMemory
from agent.tool_registry import get_tool

console = Console()


class AutonomousAnalyticsAgent:
    """
    The master agent that orchestrates the full analytics pipeline.
    Uses Google Gemini API to reason and decide.
    """

    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(LLM_MODEL)
        self.memory = AgentMemory()
        with open(f"{PROMPTS_DIR}system_prompt.txt") as f:
            self.system_prompt = f.read()

    def run(self, dataset_path: str, user_goal: str):
        """Entry point. Runs the full agent pipeline."""
        self.memory.dataset_path = dataset_path
        self.memory.user_goal = user_goal

        console.rule("[bold blue]🚀 Data Analytics Pipeline Starting")
        logger.info(f"Objective: {user_goal}")

        # ── Phase 1: Load
        self._phase_load()

        # ── Phase 2: Clean
        self._phase_clean()

        # ── Phase 3: EDA
        self._phase_eda()

        # ── Phase 4: ML Selection + Training (with retry loop)
        self._phase_ml()

        # ── Phase 5: Visualizations
        self._phase_visualize()

        # ── Phase 6: Insight Generation
        self._phase_insights()

        # ── Phase 7: Report
        self._phase_report()

        console.rule("[bold green]✅ Agent Complete")
        return self.memory.report_path

    def _phase_load(self):
        self.memory.log("LOAD", "Loading dataset...")
        load_fn = get_tool("load_data")
        self.memory.df_raw = load_fn(self.memory.dataset_path)
        summary_fn = get_tool("get_data_summary")
        self.memory.data_summary = summary_fn(self.memory.df_raw)
        self.memory.log("LOAD", f"Loaded {self.memory.df_raw.shape}")

    def _phase_clean(self):
        self.memory.log("CLEAN", "Cleaning data...")
        clean_fn = get_tool("auto_clean")
        self.memory.df_clean = clean_fn(
            self.memory.df_raw.copy(),
            target_col=self.memory.ml_selection.get("target_column")
        )
        self.memory.log("CLEAN", f"Clean shape: {self.memory.df_clean.shape}")

    def _phase_eda(self):
        self.memory.log("EDA", "Running exploratory analysis...")
        eda_fn = get_tool("run_eda")
        self.memory.eda_results = eda_fn(self.memory.df_clean)

        # Ask LLM to interpret EDA
        with open(f"{PROMPTS_DIR}eda_prompt.txt") as f:
            prompt_template = f.read()
        prompt = prompt_template.format(
            data_profile=json.dumps(self.memory.data_summary, indent=2, default=str)
        )
        
        full_prompt = f"{self.system_prompt}\n\n{prompt}"
        
        resp = self.model.generate_content(full_prompt)
        raw = resp.text.strip().replace("```json", "").replace("```", "")
        self.memory.eda_analysis = json.loads(raw.strip())
        self.memory.log("EDA", f"Domain: {self.memory.eda_analysis.get('domain')}")

    def _phase_ml(self):
        self.memory.log("ML", "Selecting ML approach...")
        select_fn = get_tool("select_ml_approach")
        self.memory.ml_selection = select_fn(
            data_summary=self.memory.data_summary,
            user_goal=self.memory.user_goal,
            columns=list(self.memory.df_clean.columns),
            target_column=None,
            shape=self.memory.df_clean.shape
        )

        best_result = None
        for attempt in range(MAX_ITERATIONS):
            self.memory.iteration = attempt + 1
            algo = self.memory.ml_selection.get("algorithm")
            task = self.memory.ml_selection.get("task")
            target = self.memory.ml_selection.get("target_column")
            features = self.memory.ml_selection.get("feature_columns", [])

            self.memory.log("ML", f"Attempt {attempt + 1}: {algo}")

            try:
                run_fn = get_tool("run_model")
                result = run_fn(
                    df=self.memory.df_clean,
                    task=task,
                    algorithm=algo,
                    target_col=target,
                    feature_cols=features
                )
                best_result = result

                # Check quality threshold for supervised tasks
                metrics = result["metrics"]
                acc = metrics.get("accuracy") or metrics.get("r2_score") or 1.0
                if acc >= ML_ACCURACY_THRESHOLD:
                    self.memory.log("ML", f"✅ Threshold met: {acc}")
                    break
                else:
                    self.memory.log("ML", f"⚠️ Below threshold ({acc}). Retrying with different algorithm...")
                    self.memory.ml_selection = self._ask_for_alternative(task, algo, metrics)

            except Exception as e:
                self.memory.errors.append(str(e))
                logger.error(f"ML error: {e}")
                break

        self.memory.ml_results = best_result or {}

    def _ask_for_alternative(self, task: str, failed_algo: str, metrics: dict) -> dict:
        """Asks LLM to suggest a different algorithm."""
        prompt = (
            f"The algorithm '{failed_algo}' for task '{task}' achieved metrics: {metrics}. "
            f"This is below the acceptable threshold. "
            f"Suggest a different and potentially better algorithm for the same task. "
            f"Return ONLY a JSON with keys: task, algorithm, reason, target_column, feature_columns. "
            f"Use the same target and feature columns as before."
        )
        resp = self.model.generate_content(prompt)
        raw = resp.text.strip().replace("```json", "").replace("```", "")
        new_selection = json.loads(raw.strip())
        new_selection["target_column"] = self.memory.ml_selection["target_column"]
        new_selection["feature_columns"] = self.memory.ml_selection["feature_columns"]
        return new_selection

    def _phase_visualize(self):
        self.memory.log("VIZ", "Generating charts...")
        viz_fn = get_tool("generate_all_charts")
        viz_fn(
            df=self.memory.df_clean,
            target_col=self.memory.ml_selection.get("target_column"),
            task=self.memory.ml_selection.get("task")
        )

    def _phase_insights(self):
        self.memory.log("AI_GEN", "Generating autonomous insights...")
        with open(f"{PROMPTS_DIR}insight_prompt.txt") as f:
            prompt_template = f.read()

        metrics = self.memory.ml_results.get("metrics", {})
        fi = self.memory.ml_results.get("feature_importance", {})

        prompt = prompt_template.format(
            task=self.memory.ml_selection.get("task"),
            algorithm=self.memory.ml_selection.get("algorithm"),
            metrics=json.dumps(metrics, indent=2),
            feature_importance=json.dumps(dict(list(fi.items())[:10]), indent=2),
            domain=self.memory.eda_analysis.get("domain", "Unknown"),
            eda_findings=json.dumps(self.memory.eda_analysis.get("patterns", []), indent=2)
        )
        resp = self.model.generate_content(prompt)
        self.memory.insights = resp.text.strip()

    def _phase_report(self):
        self.memory.log("REPORT", "Generating final report...")
        report_fn = get_tool("generate_report")
        self.memory.report_path = report_fn(
            task=self.memory.ml_selection.get("task", ""),
            algorithm=self.memory.ml_selection.get("algorithm", ""),
            metrics=self.memory.ml_results.get("metrics", {}),
            feature_importance=self.memory.ml_results.get("feature_importance", {}),
            insights=self.memory.insights,
            eda_summary=self.memory.eda_analysis,
            dataset_name=self.memory.dataset_path.split("/")[-1]
        )
