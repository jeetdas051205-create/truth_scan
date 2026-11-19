# TRUTHSCAN AI: COMPREHENSIVE ACADEMIC PROJECT REPORT
**Project Title:** AI-Powered Fake News Detection System  
**Tagline:** *Separate Facts From Fiction.*  
**Academic Degree:** Bachelor of Technology in Computer Science & Engineering (AI & ML)  
**Course Code / Module:** CS3244 Machine Learning Capstone  
**Date of Evaluation:** November 2026  

---

## 1. ABSTRACT

In an interconnected global information ecosystem, the rapid dissemination of unverified, biased, and fabricated digital news stories poses existential threats to civic discourse, economic stability, and public institutions. Traditional manual verification is unable to scale with the velocity of web publication. 

This project presents **TruthScan AI**, an industrial-grade end-to-end Machine Learning and Natural Language Processing (NLP) system engineered to classify news articles as **REAL** or **FAKE** based strictly on textual features, stylistic composition, and lexical attribution patterns. The system evaluates five classical machine learning algorithms—**Support Vector Machines (LinearSVC)**, **Passive Aggressive Classifier**, **Logistic Regression**, **Random Forest**, and **Multinomial Naive Bayes**—alongside a pre-trained **Bidirectional Long Short-Term Memory (Bi-LSTM)** deep sequence neural network. Benchmarked across **34,103** articles from the authoritative ISOT dataset, our top models (LinearSVC and Bi-LSTM) achieved **98.81% accuracy** and a **98.81% F1-score** with sub-25ms inference latency and calibrated probabilistic confidence intervals. An interactive, responsive SaaS web application provides live detection, an analytics dashboard, and linguistic attribution explainability.

---

## 2. PROBLEM STATEMENT & OBJECTIVES

### 2.1 Problem Statement
Modern misinformation campaigns rely on sophisticated sensationalist vocabulary, emotional priming, and fabricated headlines designed to manipulate human cognitive biases. Existing approaches frequently suffer from:
1. **Black-box opacity:** Producing binary labels without explaining which linguistic markers triggered the decision.
2. **Uncalibrated confidence:** Reporting arbitrary heuristic percentages rather than true statistical probabilities.
3. **Severe data leakage:** Fitting text vectorizers across train and test sets simultaneously, artificially inflating test metrics.
4. **Poor usability:** Confining ML code to isolated Jupyter notebooks without interactive, reproducible web deployment.

### 2.2 Project Objectives
1. **Corpus Consolidation:** Curate, deduplicate, and clean standard academic benchmarks (ISOT and FakeNewsNet) totaling over 57,000 articles.
2. **Leakage-Free Preprocessing:** Implement strict stratified holdout splits (80% train / 20% test) with isolated vectorizer fitting.
3. **Multi-Algorithm Benchmarking:** Train and cross-evaluate 5 traditional ML models and 1 Deep Recurrent Neural Network.
4. **Calibrated Explainability:** Provide calibrated classification probabilities and extract influential n-grams for transparency.
5. **Production Deployment:** Deliver a responsive, modern web application and REST API adhering to SaaS interface standards.

---

## 3. DATASET CHARACTERISTICS

### 3.1 ISOT Fake News Dataset (Primary Training Benchmark)
- **Institution:** University of Victoria Information Security and Object Technology (ISOT) Research Lab.
- **Total Valid Records:** 34,103 deduplicated news articles.
- **Class Balance:** Perfectly balanced:
  - **Real Articles (Class 0):** 17,056 samples (50.01%) — scraped from verified Reuters wire journalism.
  - **Fake Articles (Class 1):** 17,047 samples (49.99%) — collected from identified propaganda websites and fact-checking reviews (PolitiFact).
- **Split Configuration:** 80% Training (27,282 articles) | 20% Test (6,821 articles), stratified with random seed `42`.

### 3.2 FakeNewsNet Dataset (Secondary Benchmark)
- **PolitiFact Subset:** 1,056 political statements verified by fact-checkers.
- **GossipCop Subset:** 22,140 entertainment articles and celebrity rumors.
- **Benchmark Observation:** Demonstrates domain shift degradation (71.48% accuracy on PolitiFact), highlighting that vocabulary distributions vary substantially between political wires and entertainment social feeds.

---

## 4. MATHEMATICAL & ALGORITHMIC FORMULATIONS

### 4.1 Feature Representation: TF-IDF Vectorization
Textual documents are mapped into high-dimensional vector spaces using Term Frequency-Inverse Document Frequency (TF-IDF):

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

To prevent dominating outlier document lengths, sublinear term-frequency scaling is applied:
$$\text{Sublinear TF}(t, d) = 1 + \log(\text{TF}(t, d)) \quad \text{for } \text{TF} > 0$$

- **N-gram Range:** $(1, 2)$ — Captures both individual keywords (`senate`, `unredacted`) and collocations (`white house`, `breaking bombshell`).
- **Feature Capacity:** Capped at the top 10,000 most informative n-grams based on corpus frequency.

### 4.2 Linear Support Vector Classifier (LinearSVC)
Seeks an optimal separating hyperplane $\mathbf{w}^T \mathbf{x} + b = 0$ that maximizes the geometric margin $\frac{2}{\|\mathbf{w}\|}$ while penalizing margin violations using the hinge loss:

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

To transform uncalibrated decision distances into legitimate confidence probabilities, **Platt Sigmoid Calibration** (`CalibratedClassifierCV`) is applied via 3-fold cross-validation:
$$P(Y=1 \mid \mathbf{x}) = \frac{1}{1 + \exp(A \cdot f(\mathbf{x}) + B)}$$

### 4.3 Passive Aggressive Classifier (PAC)
An online learning algorithm that updates weights passively if a sample is correctly classified with margin $\ge 1$, but aggressively updates if an error occurs:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \text{sign}(y_t - \hat{y}_t) \min\left(C, \frac{\ell_t}{\|\mathbf{x}_t\|^2}\right) \mathbf{x}_t$$

### 4.4 Logistic Regression
Models the posterior probability of the fake news class via the logistic sigmoid link function with $L_2$ Ridge regularization:

$$P(Y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

### 4.5 Bidirectional Long Short-Term Memory (Bi-LSTM)
A recurrent neural network containing forward ($\vec{h}_t$) and backward ($\overleftarrow{h}_t$) hidden states:
$$h_t = [\vec{h}_t \,\|\, \overleftarrow{h}_t]$$
- **Embedding Layer:** Maps token indices to dense vectors ($d=128$, input length=100).
- **Recurrent Layer:** 128 hidden bidirectional memory cells capturing long-distance syntactic dependencies.
- **Dense Classifier:** 32-unit ReLU layer with 20% Dropout followed by a 1-unit Sigmoid output neuron. Total trainable parameters: **4,148,933**.

---

## 5. EXPERIMENTAL RESULTS & COMPARATIVE BENCHMARK

All models were evaluated on the held-out test split of 6,821 previously unseen articles:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (LinearSVC)** | **98.81%** | **98.97%** | 98.65% | **98.81%** | **~21 ms** |
| **Bidirectional LSTM (Deep Learning)** | **98.81%** | 98.75% | **98.87%** | **98.81%** | ~45 ms |
| **Passive Aggressive Classifier** | 98.70% | **99.11%** | 98.27% | 98.69% | **~19 ms** |
| **Logistic Regression** | 98.48% | 98.96% | 97.98% | 98.47% | ~22 ms |
| **Random Forest Classifier (100 Trees)** | 96.48% | 98.68% | 94.22% | 96.40% | ~85 ms |
| **Multinomial Naive Bayes ($\alpha=0.1$)** | 95.18% | 95.19% | 95.16% | 95.18% | **~18 ms** |

### Detailed Confusion Matrix for LinearSVC (Best Model, N=6,821):
- **True Negatives (Correctly Identified Real News):** 3,376 (98.97% specificity)
- **False Positives (Real News Mislabeled as Fake):** 35 (Type I error: 0.51%)
- **False Negatives (Fake News Missed):** 46 (Type II error: 0.67%)
- **True Positives (Correctly Identified Fake News):** 3,364 (98.65% sensitivity)
- **Overall Error Rate:** $1.19\%$

---

## 6. EXPLAINABILITY & LINGUISTIC ATTRIBUTION

By analyzing the highest magnitude weights $\mathbf{w}$ from the linear kernel, TruthScan AI isolates the specific n-grams distinguishing deceptive text from verified journalism:

- **Top Deceptive Indicators (Positive Coefficients):**
  `breaking`, `bombshell`, `unredacted`, `secret fbi`, `shocking`, `clandestine`, `whistleblower`, `cabal`, `alien`, `conspiracy`.
- **Top Credibility Indicators (Negative Coefficients):**
  `tuesday`, `spokesman said`, `reuters`, `department`, `legislation`, `committee`, `statement`, `senior official`, `bipartisan`, `passed`.

Authentic journalism is characterized by temporal grounding (`on Tuesday`), institutional attribution (`spokesman said`), and legislative terminology, whereas disinformation relies on hyperbolic urgency and ungrounded speculation.

---

## 7. SYSTEM ARCHITECTURE & IMPLEMENTATION

The complete application is structured into four cohesive layers:
1. **Model Persistence Layer:** Serialized models (`best_model.joblib`, `fake_news_bilstm.keras`, `tfidf_vectorizer.joblib`) cached in memory during server startup.
2. **REST API Service Layer:** Flask 3.1.3 endpoints serving prediction requests, health checks, dataset telemetry, and comparative metrics in standard JSON.
3. **Single Page Application (SPA):** Modern frontend built with Tailwind CSS, Chart.js, and Lucide Icons, providing dark/light theming, instant reactive navigation, and zero Node.js build overhead.
4. **Diagnostic & Testing Suite:** 11 comprehensive automated unit and integration tests executing real-time verification across all application paths.

---

## 8. CONCLUSION & ACADEMIC DEFENSE SUMMARY

TruthScan AI successfully elevates a traditional academic machine learning benchmark into a complete, reproducible, and explainable software product. By combining rigorous leakage-free preprocessing, calibrated probabilistic scoring, and dual ML/DL model architectures, the system achieves an industry-competitive 98.81% F1-score while preserving sub-25ms inference speeds suitable for live news production pipelines.

