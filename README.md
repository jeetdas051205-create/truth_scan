# TRUTHSCAN AI — AI-Powered Fake News Detection System
> **"Separate Facts From Fiction."**  
> Complete B.Tech CSE (AI & ML) Capstone Project & Production-Quality Web Application.

![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-98.81%25-brightgreen.svg)
![F1-Score](https://img.shields.io/badge/F1--Score-98.81%25-green.svg)
![Models](https://img.shields.io/badge/Models-6%20Benchmarked-indigo.svg)
![Status](https://img.shields.io/badge/Production-Ready-emerald.svg)

---

## 📌 Executive Summary

**TruthScan AI** is an industrial-grade end-to-end Machine Learning and Natural Language Processing (NLP) system designed to detect misinformation and deceptive reporting in digital news articles. Built upon rigorous academic standards from the NUS CS3244 benchmark, TruthScan AI evaluates **6 distinct Machine Learning and Deep Learning architectures** on over **34,100 verified news articles** from the University of Victoria ISOT dataset, achieving **98.81% test accuracy** and **98.81% F1-score** with sub-25ms inference latency.

---

## 🌟 Key Features

1. **Multi-Model Machine Learning Engine:**
   - **Support Vector Machine (LinearSVC):** 98.81% Accuracy (Best Classical Model, calibrated probabilities).
   - **Bidirectional LSTM (Deep Learning):** 98.81% Accuracy (4.14M parameters, Keras sequential).
   - **Passive Aggressive Classifier:** 98.70% Accuracy (Sub-20ms ultra-fast stream learning).
   - **Logistic Regression:** 98.48% Accuracy (Balanced L2-regularized log-odds).
   - **Random Forest Classifier:** 96.48% Accuracy (100 bagged orthogonal trees).
   - **Multinomial Naive Bayes:** 95.18% Accuracy (Laplace smoothed baseline).

2. **Linguistic Attribution & Explainability:**
   - Highlights top influential positive n-grams (`breaking`, `secret fbi`, `bombshell`) indicating sensationalist patterns.
   - Highlights top negative n-grams (`tuesday`, `senate`, `spokesman said`) indicating verified journalistic attribution.

3. **SaaS-Grade Modern Web Application:**
   - **Page 1: Overview & Landing Page** (Hero with instant interactive news evaluation preview).
   - **Page 2: Detection Studio** (Live analysis, sample loader, word/character counter, calibrated confidence meter).
   - **Page 3: Analytics Dashboard** (Genuine dataset balance metrics, confusion matrix, and distribution charts).
   - **Page 4: Model Performance** (Interactive multi-metric radar and comparison table).
   - **Page 5: Academic Project Report** (Formal B.Tech capstone documentation, methodology, limitations).
   - **Page 6: How It Works** (Step-by-step pipeline architecture diagram).

4. **Production REST API:**
   - `POST /api/predict`: Real-time news article classification.
   - `GET /api/health`: Health status & loaded model verification.
   - `GET /api/model-info`: Active algorithms & vectorizer specifications.
   - `GET /api/metrics`: Exact academic evaluation figures and confusion matrices.
   - `GET /api/dataset-stats`: Actual corpus counts and class ratios.
   - `GET /api/model-comparison`: Pre-formatted comparative benchmark arrays.

---

## 📊 Evaluation & Comparative Benchmarks

Evaluated on the 20% holdout test partition ($N = 6,821$ unseen articles) of the ISOT dataset:

| Algorithm | Test Accuracy | Precision | Recall | F1-Score | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (LinearSVC)** | **98.81%** | **98.97%** | 98.65% | **98.81%** | **~21 ms** |
| **Bidirectional LSTM (Deep Learning)** | **98.81%** | 98.75% | **98.87%** | **98.81%** | ~45 ms |
| **Passive Aggressive Classifier** | 98.70% | **99.11%** | 98.27% | 98.69% | **~19 ms** |
| **Logistic Regression** | 98.48% | 98.96% | 97.98% | 98.47% | ~22 ms |
| **Random Forest Classifier** | 96.48% | 98.68% | 94.22% | 96.40% | ~85 ms |
| **Multinomial Naive Bayes** | 95.18% | 95.19% | 95.16% | 95.18% | **~18 ms** |

---

## 🚀 Quickstart & Execution Guide

### 1. Prerequisites
Ensure you are in the project root directory where the virtual environment is located:

```powershell
cd "c:\Users\asus\Downloads\CS3244-Fake-News-Detection-master (1)"
```

### 2. Run the Full Integration Test Suite
Verify that all 11 unit and integration tests pass:

```powershell
.\venv\Scripts\python.exe ".\CS3244-Fake-News-Detection-master\test_app.py"
```

### 3. Launch the TruthScan AI Web Application
Start the Flask web server:

```powershell
.\venv\Scripts\python.exe ".\CS3244-Fake-News-Detection-master\app.py"
```

Open your browser and navigate to:
```
http://localhost:5000
```

### 4. Test Single Predictions from CLI
You can test inference directly from Python:

```powershell
.\venv\Scripts\python.exe ".\CS3244-Fake-News-Detection-master\predict.py"
```

### 5. Generate Academic Evaluation Reports
Display formatted confusion matrices and test metrics in terminal:

```powershell
.\venv\Scripts\python.exe ".\CS3244-Fake-News-Detection-master\evaluate_model.py"
```

---

## 📁 Project Directory Structure

```
CS3244-Fake-News-Detection-master (1)/
├── app.py                     --> Production Flask Application & REST API
├── train_model.py             --> Complete ML Training & Benchmarking Pipeline
├── predict.py                 --> Single & Batch Inference Engine
├── evaluate_model.py          --> Academic Diagnostic & Metrics Reporter
├── test_app.py                --> Automated 11-Test Integration Suite
├── templates/
│   ├── index.html             --> Modern Responsive SPA Container (All 6 Pages)
│   └── result.html            --> Legacy Form POST Result View
├── model/
│   ├── best_model.joblib      --> Serialized Best Model (LinearSVC)
│   ├── all_models.joblib      --> Serialized Dictionary of All 5 Classical Models
│   ├── tfidf_vectorizer.joblib--> 10,000-feature TF-IDF Vectorizer
│   ├── fake_news_bilstm.keras --> 4.14M-parameter Bi-LSTM Deep Learning Model
│   └── tokenizer.pkl          --> Keras Tokenizer (17,250 tokens)
├── data/
│   └── raw/
│       ├── isot/              --> ISOT Benchmark (Fake.csv & True.csv: 34,103 samples)
│       └── fakenewsnet/       --> FakeNewsNet Benchmark (PolitiFact & GossipCop: 23,196 samples)
├── reports/
│   ├── metrics.json           --> Comprehensive evaluation metrics
│   ├── model_comparison.json  --> Benchmark comparison table
│   ├── dataset_stats.json     --> Dataset counts and ratios
│   └── feature_importance.json--> Top influential n-grams
├── PROJECT_AUDIT.md           --> Initial discovery and resource audit
├── PROJECT_REPORT.md          --> Formal B.Tech Capstone Academic Paper
├── SYSTEM_ARCHITECTURE.md     --> Detailed architectural specification & diagrams
├── MODEL_TRAINING_GUIDE.md    --> Retraining & hyperparameter instructions
├── API_DOCUMENTATION.md       --> Complete REST API specification
├── requirements.txt           --> Python dependencies
└── .env.example               --> Environment configuration template
```

---

## 🎓 Academic Defense Notes

- **Data Leakage Prevention:** Preprocessing and TF-IDF vocabulary fitting were executed strictly on the 80% training split (`X_train`), transforming test samples independently.
- **Calibrated Probabilities:** Rather than displaying arbitrary confidence figures, `CalibratedClassifierCV` (Platt sigmoid scaling via 3-fold CV) transforms raw hyperplane distances into true probabilities.
- **Explainability:** Linear model coefficients map directly back to TF-IDF feature tokens, satisfying transparent AI ethics guidelines.

