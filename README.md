# 📊 InsightsEngine v1.0

InsightsEngine is a professional-grade, autonomous data analytics platform that handles the entire data science pipeline—from raw data ingestion to AI-narrated executive reports. 

Powered by **Google Gemini**, it offers high-performance, cloud-ready autonomous analysis.

## 🚀 Key Features

- **Autonomous Data Cleaning**: Automatically handles missing values, duplicates, and encoding.
- **Quantitative EDA**: Generates statistical profiles, correlation heatmaps, and feature distributions.
- **Smart ML Orchestration**: Uses Google Gemini to reason about your data and select the optimal machine learning task and algorithm.
- **Iterative Improvement**: Automatically retries and optimizes models if accuracy thresholds aren't met.
- **Executive Reporting**: Produces professional PDF reports with charts, metrics, and AI-powered business insights.
- **Cloud-Ready**: Optimized for deployment on Streamlit Community Cloud.

## 🏗️ Technical Architecture

- **Core Engine**: Python 3.13+
- **Intelligence Layer**: Google Gemini 1.5 Flash
- **Data Stack**: Pandas, NumPy, Scikit-learn, XGBoost, LightGBM
- **Visuals**: Seaborn, Matplotlib, Plotly
- **Reporting**: FPDF2

## 🛠️ Installation

### 1. Prerequisites
- Get a [Google Gemini API Key](https://aistudio.google.com/app/apikey)

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-username/InsightsEngine.git
cd InsightsEngine

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your API Key
export GEMINI_API_KEY='your_api_key_here'
```

## 📈 Usage

### 🖥️ Option 1: Terminal (CLI)
```bash
python3 main.py
```

### 🌐 Option 2: Web Dashboard (Streamlit)
For a more interactive experience, use the web interface:
```bash
streamlit run streamlit_app.py
```

## 🌍 Deployment (Hosting Online)

To host this project for free on **Streamlit Community Cloud**:
1. Push your code to a GitHub repository.
2. Sign in to [Streamlit Cloud](https://share.streamlit.io/).
3. Click "New app" and select your repository.
4. **Important**: Go to "Advanced settings" -> "Secrets" and paste your API key:
   ```toml
   GEMINI_API_KEY = "your_actual_api_key_here"
   ```
5. Click **Deploy**!

## 📂 Project Structure

```
analytics_agent/
├── agent/            # Core orchestration & memory logic
├── tools/            # Specialized data science modules
├── prompts/          # LLM reasoning templates
├── data/             # Input datasets
└── outputs/          # Generated charts and PDF reports
```

## 📜 License
MIT License. Created by [Your Name].
