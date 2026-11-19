# TRUTHSCAN AI — REST API SPECIFICATION
**API Version:** 2.0.0  
**Base URL:** `http://localhost:5000`  
**Content-Type:** `application/json`  

---

## 1. POST /api/predict
Classifies the credibility of a submitted news article.

### Request Headers
```http
Content-Type: application/json
```

### Request Body
```json
{
  "text": "WASHINGTON (Reuters) - The U.S. Senate passed sweeping bipartisan legislation on Tuesday to reform federal research grants, sending the comprehensive measure to the President's desk after a final vote in the upper chamber.",
  "model": "Support Vector Machine (LinearSVC)"
}
```

*Note: The `model` field is optional. If omitted, the system defaults to the top-performing `Support Vector Machine (LinearSVC)`. Other valid options include:*
- `Support Vector Machine (LinearSVC)`
- `Bidirectional LSTM (Deep Learning)`
- `Passive Aggressive Classifier`
- `Logistic Regression`
- `Random Forest Classifier`
- `Multinomial Naive Bayes`

### Response (200 OK)
```json
{
  "success": true,
  "prediction": "REAL",
  "confidence": 96.74,
  "probability_fake": 0.0326,
  "probability_real": 0.9674,
  "model_used": "Support Vector Machine (LinearSVC)",
  "latency_ms": 19.8,
  "word_count": 34,
  "explanation": "The neutral tone, attribution cadence, and contextual vocabulary conform to verified, factual journalism standards.",
  "influential_words": [
    {
      "word": "tuesday",
      "weight": 2.4182,
      "raw_weight": -2.4182,
      "type": "Credibility Signal"
    },
    {
      "word": "senate",
      "weight": 1.9421,
      "raw_weight": -1.9421,
      "type": "Credibility Signal"
    },
    {
      "word": "legislation",
      "weight": 1.6318,
      "raw_weight": -1.6318,
      "type": "Credibility Signal"
    }
  ],
  "disclaimer": "This AI system provides an automated linguistic classification based on its training data. Always verify important news via official, primary, and reputable news outlets."
}
```

### Error Response (400 Bad Request)
```json
{
  "success": false,
  "error": "The news text cannot be empty. Please enter or paste an article."
}
```

---

## 2. GET /api/health
Returns service health, model readiness, and available algorithm identifiers.

### Response (200 OK)
```json
{
  "status": "healthy",
  "service": "TruthScan AI Engine",
  "models_ready": true,
  "available_algorithms": [
    "Logistic Regression",
    "Multinomial Naive Bayes",
    "Passive Aggressive Classifier",
    "Support Vector Machine (LinearSVC)",
    "Random Forest Classifier",
    "Bidirectional LSTM (Deep Learning)"
  ],
  "timestamp": 1794066120.12
}
```

---

## 3. GET /api/model-info
Retrieves detailed metadata about the active ML/DL pipeline architectures.

### Response (200 OK)
```json
{
  "system_name": "TRUTHSCAN AI",
  "version": "2.0.0",
  "tagline": "Separate Facts From Fiction",
  "active_default_model": "Support Vector Machine (LinearSVC)",
  "available_models": [
    "Logistic Regression",
    "Multinomial Naive Bayes",
    "Passive Aggressive Classifier",
    "Support Vector Machine (LinearSVC)",
    "Random Forest Classifier",
    "Bidirectional LSTM (Deep Learning)"
  ],
  "vectorizer": "TF-IDF (10,000 features, n-gram 1-2, sublinear TF)",
  "deep_learning_architecture": "Bi-LSTM (Embedding 128d -> BiLSTM 128 -> Dense 32 -> Dense 1, Sigmoid)",
  "primary_dataset": "ISOT Fake News Corpus (34,103 verified articles)",
  "benchmark_dataset": "FakeNewsNet (PolitiFact + GossipCop, 23,196 articles)"
}
```

---

## 4. GET /api/metrics
Returns exact accuracy, precision, recall, F1, and confusion matrix records for all models.

### Response (200 OK)
```json
{
  "Support Vector Machine (LinearSVC)": {
    "name": "Support Vector Machine (LinearSVC)",
    "accuracy": 0.9881,
    "precision": 0.9897,
    "recall": 0.9865,
    "f1_score": 0.9881,
    "train_time_sec": 1.25,
    "predict_time_sec": 0.021,
    "confusion_matrix": {
      "true_negative": 3376,
      "false_positive": 35,
      "false_negative": 46,
      "true_positive": 3364
    }
  }
}
```

---

## 5. GET /api/dataset-stats
Returns true dataset metrics, class balance distribution, and split dimensions.

### Response (200 OK)
```json
{
  "dataset_name": "ISOT Fake News Dataset (University of Victoria)",
  "total_records": 34103,
  "real_count": 17056,
  "fake_count": 17047,
  "train_records": 27282,
  "test_records": 6821,
  "vocabulary_size": 10000,
  "classes": ["Real News (0)", "Fake News (1)"],
  "class_balance_percentage": {
    "real": 50.01,
    "fake": 49.99
  }
}
```

---

## 6. GET /api/model-comparison
Returns pre-formatted tabular benchmark arrays for rendering in comparison components.

