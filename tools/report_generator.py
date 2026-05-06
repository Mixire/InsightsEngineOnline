import os
from fpdf import FPDF
from datetime import datetime
from loguru import logger
from config import REPORTS_DIR, CHARTS_DIR

os.makedirs(REPORTS_DIR, exist_ok=True)


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 100, 200)
        self.cell(0, 10, "Autonomous AI Analytics Report", align="C", ln=True)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C", ln=True)
        self.ln(5)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def generate_report(task: str, algorithm: str, metrics: dict,
                    feature_importance: dict, insights: str,
                    eda_summary: dict, dataset_name: str = "dataset") -> str:
    """
    Generates a full PDF analytics report.
    Returns the path to the saved report.
    """
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Section 1: Overview
    _section_title(pdf, "1. Project Overview")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 7, f"Dataset: {dataset_name}\nML Task: {task}\nAlgorithm: {algorithm}")
    pdf.ln(5)

    # ── Section 2: Key Metrics
    _section_title(pdf, "2. Model Performance Metrics")
    for key, val in metrics.items():
        if isinstance(val, dict):
            continue
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(70, 8, f"{key.replace('_', ' ').title()}:", ln=False)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, str(val), ln=True)
    pdf.ln(5)

    # ── Section 3: AI-Generated Insights
    _section_title(pdf, "3. AI-Generated Insights")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, insights)
    pdf.ln(5)

    # ── Section 4: Feature Importance
    if feature_importance:
        _section_title(pdf, "4. Top Influencing Features")
        top5 = list(feature_importance.items())[:5]
        for feat, score in top5:
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 7, f"  - {feat}: {round(score, 4)}", ln=True)
        pdf.ln(5)

    # ── Section 5: Charts
    _section_title(pdf, "5. Visualizations")
    chart_files = [
        ("distributions.png", "Feature Distributions"),
        ("correlation_heatmap.png", "Correlation Heatmap"),
        ("feature_importance.png", "Feature Importance"),
        ("class_balance.png", "Class Balance"),
        ("boxplots.png", "Box Plots"),
    ]
    for filename, title in chart_files:
        path = f"{CHARTS_DIR}{filename}"
        if os.path.exists(path):
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 8, title, ln=True)
            pdf.image(path, w=180)
            pdf.ln(5)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{REPORTS_DIR}analytics_report_{timestamp}.pdf"
    pdf.output(output_path)
    logger.success(f"Report saved: {output_path}")
    return output_path


def _section_title(pdf: FPDF, title: str):
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 100, 200)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_text_color(50, 50, 50)
    pdf.ln(2)
