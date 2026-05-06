import sys
import os

# macOS Fix for XGBoost / libomp
if sys.platform == "darwin":
    libomp_path = "/opt/homebrew/opt/libomp/lib"
    if os.path.exists(libomp_path):
        os.environ["DYLD_LIBRARY_PATH"] = libomp_path + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")

from rich.console import Console
from agent.core_agent import AutonomousAnalyticsAgent

console = Console()

def main():
    console.print("\n[bold cyan]InsightsEngine v1.0[/bold cyan]")
    console.print("[dim]Powered by Python + Google Gemini API[/dim]\n")

    # Get inputs
    dataset_path = input("📂 Enter path to your dataset (CSV/Excel/JSON): ").strip()
    user_goal = input("🎯 What is your analytics goal? (e.g., 'predict customer churn'): ").strip()

    # Run agent
    try:
        agent = AutonomousAnalyticsAgent()
        report_path = agent.run(dataset_path=dataset_path, user_goal=user_goal)

        console.print(f"\n[bold green]✅ Done! Report saved to: {report_path}[/bold green]")
        console.print("\n[bold]Reasoning Log:[/bold]")
        for entry in agent.memory.reasoning_log:
            console.print(f"  [dim]{entry}[/dim]")
    except Exception as e:
        console.print(f"\n[bold red]❌ An error occurred:[/bold red] {e}")


if __name__ == "__main__":
    main()
