# 🧠 Fake News Detection with DistilBERT – MLOps Pipeline

A reproducible end-to-end Machine Learning workflow for **Fake News Detection**, built with:
- **DistilBERT** for language understanding
- **DVC** for data versioning and reproducible pipelines
- **MLflow** for experiment tracking and model registry
- **GitHub Actions** for CI/CD automation

---

## 🚀 Project Overview

This project detects whether a given news article is **fake or real** using NLP-based models.
The pipeline includes:
1. **EDA** – Explore the data to understand patterns
2. **Preprocessing** – Clean, normalize, and prepare data
3. **Model Training** – Baseline (TF-IDF) → Fine-tuned DistilBERT
4. **Explainability** – SHAP/LIME-based interpretation
5. **MLOps Integration** – DVC + MLflow + GitHub Actions

---

## 📁 Project Structure

```
fake-news-mlops/
├── data/ <- raw and processed datasets (tracked via DVC)
├── src/ <- source scripts for each pipeline stage
├── models/ <- saved models and checkpoints
├── reports/ <- evaluation metrics and visualizations
├── notebooks/ <- EDA and prototype notebooks
├── params.yaml <- configuration file for pipeline parameters
├── requirements.txt <- Python dependencies
└── dvc.yaml <- DVC pipeline definition (added later)
```

---

## 🧰 Tech Stack

| Component | Tool |
|------------|------|
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| Model | DistilBERT (Hugging Face Transformers) |
| CI/CD | GitHub Actions |
| Visualization | Matplotlib, SHAP, Pandas-Profiling |

---

## 🧪 Getting Started

```bash
# 1️⃣ Clone the repo
git clone https://github.com/<yourusername>/fake-news-mlops.git
cd fake-news-mlops

# 2️⃣ Create environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3️⃣ Initialize DVC
dvc init
```

---

## 📜 License

This project is released under the **MIT License** — free to use and modify for educational and research purposes.