# TRUTHSCAN AI — PROJECT AUDIT & RESOURCE DISCOVERY REPORT
**Project Title:** AI-Powered Fake News Detection System  
**Academic Context:** B.Tech CSE (AI & ML) End-Semester Capstone Project  
**Inspection Date:** November 2026  
**Auditor:** Senior AI/ML Engineer & Full-Stack Systems Architect  

---

## 1. Executive Summary

This audit assesses the state of the existing NUS CS3244 Fake News Detection codebase (`CS3244-Fake-News-Detection-master`), accompanying environment files, installed virtual environment (`venv`), downloaded datasets in the user environment (`FakeNewsNet-master`, downloads directory), and pretrained models.

The original project explored traditional machine learning (Logistic Regression, k-Nearest Neighbors, Support Vector Machines, Naive Bayes, Random Forest) and deep learning models (Bidirectional LSTM, GRU, BERT) for detecting misinformation based on article content across two standard benchmarks: **ISOT** and **FakeNewsNet**.

---

## 2. Existing Project Structure

```
CS3244-Fake-News-Detection-master (1)/
├── CS3244-Fake-News-Detection-master/
│   ├── .vscode/
│   │   └── extensions.json
│   ├── model/
│   │   ├── config.pkl                (Model configuration parameters: max_words=10000, max_len=100)
│   │   ├── fake_news_bilstm.keras     (Pre-trained Keras Bi-LSTM model, 16.6 MB, 4,148,933 parameters)
│   │   └── tokenizer.pkl              (Keras Tokenizer, 686 KB, 17,250 vocabulary words)
│   ├── templates/
│   │   ├── index.html                 (Minimal unstyled input form)
│   │   └── result.html                (Minimal unstyled output screen)
│   ├── app.py                         (Legacy Flask server using Keras Bi-LSTM)
│   ├── BERT_FakeNewsNet.py            (PyTorch BERT fine-tuning script for FakeNewsNet)
│   ├── BERT_ISOT.ipynb                (BERT evaluation on ISOT dataset)
│   ├── Bi-LSTM-default.py             (Bi-LSTM training script with default hyperparams)
│   ├── Bi-LSTM.py                     (Bi-LSTM notebook script with Google Drive paths)
│   ├── GRU.py                         (PyTorch GRU sequence model implementation)
│   ├── LR_kNN_LSTM.ipynb              (Jupyter notebook for Logistic Regression, kNN, and LSTM)
│   ├── main.py                        (PyTorch GRUNet with HashingVectorizer)
│   ├── NB_RF.ipynb                    (Jupyter notebook for Multinomial Naive Bayes & Random Forest)
│   ├── readme.md                      (Original project notes, benchmark accuracies, setup notes)
│   └── requirements.txt               (Python dependencies list)
├── venv/                              (Pre-configured Python 3.13.5 virtual environment with ML packages)
└── data/ (Populated during audit)
    └── raw/
        ├── isot/
        │   ├── Fake.csv               (17,050 records, 44.5 MB)
        │   └── True.csv               (17,058 records, 41.1 MB)
        └── fakenewsnet/
            ├── gossipcop_fake.csv     (5,323 records, 12.5 MB)
            ├── gossipcop_real.csv     (16,817 records, 19.9 MB)
            ├── politifact_fake.csv    (432 records, 3.28 MB)
            └── politifact_real.csv    (624 records, 8.27 MB)
```

---

## 3. Available Datasets & Detailed Metrics

### A. ISOT Fake News Dataset (Primary Benchmark)
- **Source:** University of Victoria Information Security and Object Technology (ISOT) Research Lab.
- **Acquisition:** Retrieved verified raw corpus mirrors into `data/raw/isot/`.
- **Dataset Size:**
  - `Fake.csv`: 17,050 articles (44.5 MB).
  - `True.csv`: 17,058 articles (41.1 MB).
  - **Total:** 34,108 high-quality full-text records (balanced 50% real / 50% fake).
- **Columns:**
  - `type`: Binary ground truth (`1` = Fake news, `0` = Real/True news).
  - `text`: Full raw text content of the news article.

### B. FakeNewsNet Dataset (Secondary Benchmark)
- **Source:** Discovered in local archive `FakeNewsNet-master` and mirrored into `data/raw/fakenewsnet/`.
- **Components:**
  1. `politifact_fake.csv`: 432 articles (3.28 MB).
  2. `politifact_real.csv`: 624 articles (8.27 MB).
  3. `gossipcop_fake.csv`: 5,323 articles (12.5 MB).
  4. `gossipcop_real.csv`: 16,817 articles (19.9 MB).
  - **Total FakeNewsNet records:** 23,196 news articles.
- **Columns:** `id`, `news_url`, `title`, `tweet_ids`.

---

## 4. Existing Algorithms & Models

### Documented Historical Benchmarks (from `readme.md`):
| Dataset | Logistic Regression | kNN | SVM | Bi-LSTM | GRU | BERT | Naive Bayes | Random Forest |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ISOT** | 98.64% | 95.61% | 99.37% | 98.81% | 97.23% | 99.96% | 95.42% | ~90.0% |
| **FakeNewsNet** | 71.48% | 63.31% | 69.76% | 61.82% | 62.94% | 71.21% | — | — |

### Existing Saved Model Artifacts:
1. `fake_news_bilstm.keras` (`CS3244-Fake-News-Detection-master/model/`):
   - **Type:** Keras Sequential Model.
   - **Architecture:**
     - Embedding Layer: `input_length=100`, `dim=128`, vocab size ~10,000 / 17,250.
     - Bidirectional LSTM Layer: 128 units (64 forward + 64 backward).
     - Dropout (0.2).
     - Dense layer (32 units, ReLU).
     - Dropout (0.2).
     - Dense output layer (1 unit, Sigmoid).
   - **Total Parameters:** 4,148,933 (15.83 MB).
   - **Status:** **Verified Working & Loadable.**
2. `tokenizer.pkl` (`model/`):
   - **Type:** `tensorflow.keras.preprocessing.text.Tokenizer`.
   - **Vocabulary Size:** 17,250 tokens.
   - **Status:** **Verified Working.**
3. `Downloads/fake_news_bilstm.keras`:
   - Corrupted file (BadZipFile signature from partial download). The file in the project `model/` folder is intact and functional.

---

## 5. Software Dependencies & Runtime Environment

The local virtual environment (`venv/`) has the complete scientific ML stack installed under Python 3.13.5:
- **Core Frameworks:** TensorFlow 2.21.0, Keras 3.15.1, PyTorch 2.14.1, HuggingFace Transformers 5.18.0.
- **Classical ML & Vectorization:** scikit-learn 1.9.1, SciPy 1.18.1.
- **Data & Text Processing:** pandas 3.0.6, numpy 2.5.3, nltk 3.10.3 (corpora `punkt`, `punkt_tab`, `stopwords`, `wordnet` verified), gensim 4.4.0.
- **Web Backend:** Flask 3.1.3, Jinja2 3.1.6, Werkzeug 3.1.9.
- **Visualization:** matplotlib 3.11.2, seaborn 0.13.2, wordcloud 1.9.6.

---

## 6. Identified Deficiencies in Legacy Project

1. **User Interface:** Legacy `templates/index.html` and `result.html` are barebones (~20 lines of unstyled HTML), lacking branding, analytics, responsiveness, or visual cues.
2. **Hardcoded Google Drive Paths:** Notebooks and scripts contained hardcoded Colab paths (`/content/drive/My Drive/...`) preventing seamless local execution.
3. **Missing Automated ML Pipeline:** Preprocessing and training code were fragmented across uncoupled scripts and notebooks without a single clean pipeline to train, evaluate, and serialize scikit-learn models (Logistic Regression, Multinomial Naive Bayes, Passive Aggressive Classifier, Linear SVM, Random Forest).
4. **Lack of REST API:** The legacy `app.py` only returned a synchronous HTML form POST, without JSON API endpoints for dataset statistics, model comparisons, or explainability features (influential terms).
5. **Missing College Project Artifacts:** No comprehensive architecture documentation, project report, evaluation summary, or training guide.

---

## 7. Recommended Implementation Architecture (TruthScan AI)

### A. Machine Learning Pipeline (`Phase 1`)
- Construct `train_model.py` to train, cross-evaluate, and serialize 5 standard algorithms on the ISOT dataset:
  1. **Logistic Regression** (L2 penalty, balanced class weight, C=1.0)
  2. **Multinomial Naive Bayes** (Laplace smoothing alpha=0.1)
  3. **Passive Aggressive Classifier** (Optimal for text stream classification)
  4. **Support Vector Machine / LinearSVC** (Hinge loss, fast convex optimization)
  5. **Random Forest Classifier** (100 estimators, tree-based ensemble)
  - Complement with the pre-trained **Deep Learning Bi-LSTM** model.
- Vectorizer: `TfidfVectorizer` (ngram_range=(1,2), max_features=10,000, sublinear_tf=True, stop_words='english').
- Feature Explainability: Extract top influential TF-IDF n-grams indicating authentic vs. deceptive reporting.
- Reports: Export metrics (`metrics.json`, `model_comparison.json`, confusion matrices, ROC metrics).

### B. Backend Architecture (`Phase 3`)
- Modern Flask application (`app.py`) providing:
  - `POST /api/predict`: Model inference with text validation, confidence score, processing latency, and influential token breakdown.
  - `GET /api/health`: Service health and model status check.
  - `GET /api/model-info`: Active model metadata, parameter counts, and architecture description.
  - `GET /api/metrics`: Evaluation metrics for active and comparative models.
  - `GET /api/dataset-stats`: Genuine dataset statistics (sample counts, vocabulary size, class balances).
  - `GET /api/model-comparison`: Comparative benchmark table across all 6 models.

### C. Frontend Architecture (`Phase 2 & Phase 4`)
- Create a modern, responsive Single Page Application (SPA) powered by Tailwind CSS, Chart.js, and Lucide Icons, served seamlessly by Flask:
  - **Page 1: Landing Page** (Hero section, live analysis preview, feature grid, architecture overview).
  - **Page 2: Detection Studio** (Interactive text area, sample news prefill buttons, confidence meter, credibility badge, influential tokens highlight, AI disclaimer).
  - **Page 3: Analytics Dashboard** (Real dataset distribution, class balance charts, vocabulary statistics, genuine evaluation metrics).
  - **Page 4: Model Performance** (Interactive multi-model comparison bar chart, confusion matrices, precision/recall/F1 radar/table).
  - **Page 5: About Project** (Problem statement, literature review, proposed vs. existing system, academic methodology, tech stack, future scope).
  - **Page 6: How It Works** (Interactive step-by-step pipeline diagram from raw text to prediction).

