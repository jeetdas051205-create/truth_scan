# TRUTHSCAN AI — MODEL TRAINING & RETRAINING GUIDE

This technical guide instructs researchers, examiners, and developers on how to reproduce the training process, tune hyperparameters, or train on custom misinformation datasets.

---

## 1. Prerequisites & Environment Setup

Verify that the Python virtual environment has the required packages installed:

```powershell
# Navigate to the workspace root
cd "c:\Users\asus\Downloads\CS3244-Fake-News-Detection-master (1)"

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Verify packages
python -c "import sklearn, tensorflow, keras, pandas, nltk; print('All ML modules loaded successfully.')"
```

---

## 2. Dataset Preparation

The training pipeline expects the ISOT dataset in `CS3244-Fake-News-Detection-master/data/raw/isot/`:
- `Fake.csv`: Must contain `text` column with unverified / fake articles.
- `True.csv`: Must contain `text` column with verified / factual news articles.

If datasets are absent, run the automated acquisition command:

```powershell
python -c "
import urllib.request, os
os.makedirs('CS3244-Fake-News-Detection-master/data/raw/isot', exist_ok=True)
base = 'https://huggingface.co/datasets/Phoenyx83/ISOT-Fake-News-Dataset-FineTuned-2022/resolve/main/raw/'
urllib.request.urlretrieve(base + 'Fake%20ok.csv', 'CS3244-Fake-News-Detection-master/data/raw/isot/Fake.csv')
urllib.request.urlretrieve(base + 'True%20ok%20v2.csv', 'CS3244-Fake-News-Detection-master/data/raw/isot/True.csv')
print('Datasets downloaded successfully.')
"
```

---

## 3. Running the Training Pipeline

Execute `train_model.py`:

```powershell
python ".\CS3244-Fake-News-Detection-master\train_model.py"
```

### What `train_model.py` performs automatically:
1. **Deduplication:** Merges and removes duplicate articles (34,103 clean samples).
2. **Preprocessing:** Applies lowercasing, HTML tag removal, URL removal, and Reuters wire metadata stripping.
3. **Stratified Split:** Partitions the dataset into 80% Train (27,282 articles) and 20% Test (6,821 articles) using `random_state=42`.
4. **TF-IDF Fitting:** Fits unigram + bigram vectorizer on the training split only (`max_features=10000`).
5. **Multi-Model Training:**
   - **Logistic Regression** (`C=1.0, class_weight='balanced'`)
   - **Multinomial Naive Bayes** (`alpha=0.1`)
   - **Passive Aggressive Classifier** (`C=0.5, max_iter=1000`)
   - **Support Vector Machine** (`LinearSVC` calibrated with `CalibratedClassifierCV`)
   - **Random Forest Classifier** (`100 trees, max_depth=25, n_jobs=-1`)
6. **Report Generation:** Calculates test metrics and writes:
   - `model/best_model.joblib`
   - `model/all_models.joblib`
   - `model/tfidf_vectorizer.joblib`
   - `reports/metrics.json`
   - `reports/model_comparison.json`
   - `reports/feature_importance.json`
   - `reports/dataset_stats.json`

---

## 4. Evaluating Models & Generating Reports

Run `evaluate_model.py` to display the formatted academic evaluation table:

```powershell
python ".\CS3244-Fake-News-Detection-master\evaluate_model.py"
```

---

## 5. Hyperparameter Tuning Reference

To modify hyperparameter settings, edit lines in `CS3244-Fake-News-Detection-master/train_model.py`:

| Parameter | Default | Recommended Range | Description |
| :--- | :---: | :---: | :--- |
| `MAX_FEATURES` | `10000` | `5000` – `30000` | Size of TF-IDF vocabulary matrix. |
| `ngram_range` | `(1, 2)` | `(1, 1)` – `(1, 3)` | Range of token n-grams to extract. |
| `LinearSVC(C)` | `0.5` | `0.1` – `2.0` | Regularization penalty for margin slack. |
| `RandomForest(n_estimators)`| `100` | `50` – `200` | Number of bagged decision trees. |
| `MultinomialNB(alpha)` | `0.1` | `0.01` – `1.0` | Laplace smoothing parameter. |

