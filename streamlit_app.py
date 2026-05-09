import streamlit as st
import os
import sys
import time
import pandas as pd
from agent.core_agent import AutonomousAnalyticsAgent
from config import LLM_MODEL, CHARTS_DIR, REPORTS_DIR

# macOS Fix for XGBoost / libomp
if sys.platform == "darwin":
    libomp_path = "/opt/homebrew/opt/libomp/lib"
    if os.path.exists(libomp_path):
        os.environ["DYLD_LIBRARY_PATH"] = libomp_path + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")

# Page Config
# Build: 2026-05-09 17:45
st.set_page_config(page_title="InsightsEngine v1.0", page_icon="📊", layout="wide")

# Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4F8EF7;
        color: white;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Settings")
    st.info(f"Model: {LLM_MODEL}")
    st.divider()
    st.markdown("### About")
    st.caption("InsightsEngine v1.0 is an autonomous data science platform powered by Google Gemini.")

# Main Header
st.title("📊 InsightsEngine v1.0")
st.caption(f"Active Model: {LLM_MODEL}")
st.subheader("Next-Generation Automated Data Analysis")
st.write("Upload a dataset and define your goal. The AI will handle cleaning, EDA, modeling, and reporting.")

# File Uploader
uploaded_file = st.file_uploader("📂 Choose a dataset (CSV, Excel, JSON)", type=["csv", "xlsx", "json"])

if uploaded_file:
    # Save uploaded file to data directory
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"Loaded: {uploaded_file.name}")
    
    # Goal Input
    user_goal = st.text_input("🎯 What is your analytics goal?", placeholder="e.g., 'predict customer churn' or 'analyze sales trends'")

    if st.button("🚀 Start Autonomous Analysis"):
        if not user_goal:
            st.error("Please enter an analytics goal first.")
        else:
            # Check for API Key
            if not os.getenv("GEMINI_API_KEY"):
                st.error("Missing GEMINI_API_KEY. Please set it in your environment or Streamlit Secrets.")
            else:
                # Initialize Agent
                agent = AutonomousAnalyticsAgent()
                
                # Create a placeholder for logs
                log_container = st.empty()
                with st.spinner("Pipeline running..."):
                    try:
                        # Run the agent
                        report_path = agent.run(dataset_path=file_path, user_goal=user_goal)
                        
                        st.success("✅ Analysis Complete!")
                        
                        # Layout for Results
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("### 📜 Reasoning Log")
                            for entry in agent.memory.reasoning_log:
                                st.text(entry)
                        
                        with col2:
                            st.markdown("### 📄 Final Report")
                            if os.path.exists(report_path):
                                with open(report_path, "rb") as f:
                                    st.download_button(
                                        label="📥 Download PDF Report",
                                        data=f,
                                        file_name=os.path.basename(report_path),
                                        mime="application/pdf"
                                    )
                        
                        # Display Charts
                        st.divider()
                        st.markdown("### 📈 Visualizations")
                        chart_files = [f for f in os.listdir(CHARTS_DIR) if f.endswith(".png")]
                        if chart_files:
                            cols = st.columns(2)
                            for idx, chart in enumerate(chart_files):
                                with cols[idx % 2]:
                                    st.image(os.path.join(CHARTS_DIR, chart), caption=chart.replace(".png", "").title())
                                    
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.exception(e)
else:
    st.info("Please upload a dataset to begin.")

# Footer
st.divider()
st.caption("InsightsEngine | Powered by Google Gemini | Professional Data Analysis")
