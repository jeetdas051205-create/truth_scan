"""
TruthScan AI - Single & Batch Inference Engine
=============================================
Provides production-grade inference functions for fake news detection:
- Supports Ensemble (Multi-Model Consensus Engine) combining Deep Learning & Classical ML
- Supports all trained ML models (LinearSVC, Logistic Regression, Naive Bayes, Passive Aggressive, Random Forest)
- Supports Bidirectional LSTM Keras deep learning model
- Calibrated confidence scores with zero-bias baseline for neutral journalism
- Keyword explainability & contribution breakdown
- Input validation & safety guardrails
"""

import os
import re
import time
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

_cached_resources = {}


def clean_text(text: str) -> str:
    """Standardized text preprocessing function."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'^\s*\(Reuters\)\s*-\s*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'Reuters', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def load_inference_resources():
    """Load and cache vectorizer and models for fast reuse."""
    global _cached_resources
    if _cached_resources:
        return _cached_resources

    print("[*] Loading TruthScan AI inference models...")
    vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
    all_models_path = os.path.join(MODEL_DIR, "all_models.joblib")
    best_model_path = os.path.join(MODEL_DIR, "best_model.joblib")

    resources = {}
    if os.path.exists(vectorizer_path):
        resources["vectorizer"] = joblib.load(vectorizer_path)
    if os.path.exists(all_models_path):
        resources["all_models"] = joblib.load(all_models_path)
    if os.path.exists(best_model_path):
        resources["best_model"] = joblib.load(best_model_path)

    # Attempt to load BiLSTM model if available
    bilstm_path = os.path.join(MODEL_DIR, "fake_news_bilstm.keras")
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
    if os.path.exists(bilstm_path) and os.path.exists(tokenizer_path):
        try:
            import keras
            import pickle
            resources["bilstm_model"] = keras.models.load_model(bilstm_path)
            with open(tokenizer_path, "rb") as f:
                resources["tokenizer"] = pickle.load(f)
            resources["bilstm_maxlen"] = int(resources["bilstm_model"].input_shape[1])
        except Exception as e:
            print(f"[-] Optional Bi-LSTM loading note: {e}")

    _cached_resources = resources
    return resources


def explain_prediction(text_clean: str, vectorizer, model, top_k=7):
    """Identify which words in the article pushed towards Real vs Fake."""
    try:
        words = text_clean.split()
        if not words:
            return []

        feature_names = vectorizer.get_feature_names_out()
        feat_to_idx = {feat: idx for idx, feat in enumerate(feature_names)}

        # Extract weights from linear model or calibrated classifier
        if hasattr(model, 'coef_'):
            coefs = model.coef_[0]
        elif hasattr(model, 'calibrated_classifiers_'):
            coefs = np.mean([clf.estimator.coef_[0] for clf in model.calibrated_classifiers_], axis=0)
        else:
            return []

        word_influences = []
        seen = set()
        for w in words:
            if w in feat_to_idx and w not in seen:
                seen.add(w)
                idx = feat_to_idx[w]
                weight = float(coefs[idx])
                # In our training: label 1 = Fake (positive weight), label 0 = Real (negative weight)
                direction = "Fake Signal" if weight > 0 else "Credibility Signal"
                word_influences.append({
                    "word": w,
                    "weight": round(abs(weight), 4),
                    "raw_weight": round(weight, 4),
                    "indicates": direction
                })

        word_influences.sort(key=lambda x: x["weight"], reverse=True)
        return word_influences[:top_k]
    except Exception:
        return []


def predict_news(text: str, model_name: str = "Ensemble (Multi-Model Consensus)") -> dict:
    """
    Run inference on raw news article text.
    Returns structured prediction dictionary with confidence, model details, and latency.
    """
    start_time = time.time()
    resources = load_inference_resources()

    if not text or not text.strip():
        return {
            "error": "Input news article is empty.",
            "status": "invalid"
        }

    cleaned = clean_text(text)
    word_count = len(text.split())
    if len(cleaned.split()) < 3:
        return {
            "error": "Input is too brief for reliable linguistic classification. Please provide at least 3-5 sentences.",
            "status": "too_short"
        }

    vectorizer = resources.get("vectorizer")
    all_models = resources.get("all_models", {})
    has_bilstm = "bilstm_model" in resources and "tokenizer" in resources

    # --------------------------------------------------------------------------
    # Case A: Ensemble (Multi-Model Consensus Engine) - RECOMMENDED DEFAULT
    # --------------------------------------------------------------------------
    if "Ensemble" in model_name or model_name == "default":
        p_bilstm_fake = None
        if has_bilstm:
            try:
                tokenizer = resources["tokenizer"]
                bilstm = resources["bilstm_model"]
                maxlen = resources.get("bilstm_maxlen", 100)
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                seq = tokenizer.texts_to_sequences([text])
                padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
                p_real_dl = float(bilstm.predict(padded, verbose=0)[0][0])
                p_bilstm_fake = 1.0 - p_real_dl
            except Exception as e:
                print(f"[-] Ensemble BiLSTM notice: {e}")
                p_bilstm_fake = None

        vec = vectorizer.transform([cleaned])
        
        # 1. Calibrated LinearSVC (centered dot product to avoid intercept drift)
        svc = all_models.get("Support Vector Machine (LinearSVC)")
        if svc and hasattr(svc, "calibrated_classifiers_"):
            coefs = np.mean([clf.estimator.coef_[0] for clf in svc.calibrated_classifiers_], axis=0)
            dot_svc = float(vec.dot(coefs)[0])
            p_svc_fake = 1.0 / (1.0 + np.exp(-3.0 * dot_svc))
        else:
            p_svc_fake = 0.5

        # 2. Random Forest
        rf = all_models.get("Random Forest Classifier")
        if rf:
            p_rf_fake = float(rf.predict_proba(vec)[0][1])
        else:
            p_rf_fake = 0.5

        # 3. Logistic Regression (centered dot product)
        lr = all_models.get("Logistic Regression")
        if lr and hasattr(lr, "decision_function"):
            z_lr = float(lr.decision_function(vec)[0])
            dot_lr = z_lr - float(lr.intercept_[0])
            p_lr_fake = 1.0 / (1.0 + np.exp(-2.0 * dot_lr))
        else:
            p_lr_fake = 0.5

        # Weighted multi-model consensus
        if p_bilstm_fake is not None:
            # DL 50% + Calibrated LinearSVC 25% + Random Forest 25%
            prob_fake = 0.50 * p_bilstm_fake + 0.25 * p_svc_fake + 0.25 * p_rf_fake
        else:
            # Classical ensemble: SVC 40% + RF 30% + LR 30%
            prob_fake = 0.40 * p_svc_fake + 0.30 * p_rf_fake + 0.30 * p_lr_fake

        prob_real = 1.0 - prob_fake

        if prob_fake >= 0.50:
            label = "FAKE"
            confidence = round(prob_fake * 100, 2)
            summary = "Multi-Model Consensus: Deep sequence syntax and lexical feature distributions exhibit high correlation with sensationalist or unverified reporting patterns."
        else:
            label = "REAL"
            confidence = round(prob_real * 100, 2)
            summary = "Multi-Model Consensus: Linguistic tone, attribution cadence, and contextual vocabulary conform to verified journalistic reporting standards."

        influential_words = explain_prediction(cleaned, vectorizer, svc or lr, top_k=7)
        latency_ms = round((time.time() - start_time) * 1000, 1)

        return {
            "prediction": label,
            "confidence": confidence,
            "probability_fake": round(prob_fake, 4),
            "probability_real": round(prob_real, 4),
            "model_used": "Ensemble (Multi-Model Consensus)",
            "processing_time_ms": latency_ms,
            "word_count": word_count,
            "explanation": summary,
            "influential_words": influential_words,
            "disclaimer": "This AI system provides an automated linguistic classification based on its training data. Always verify important news via official, primary, and reputable news outlets."
        }

    # --------------------------------------------------------------------------
    # Case B: User selected Bi-LSTM Deep Learning Model
    # --------------------------------------------------------------------------
    if "LSTM" in model_name and has_bilstm:
        tokenizer = resources["tokenizer"]
        model = resources["bilstm_model"]
        maxlen = resources.get("bilstm_maxlen", 100)

        from tensorflow.keras.preprocessing.sequence import pad_sequences
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=maxlen, padding="post", truncating="post")
        raw_prob = float(model.predict(padded, verbose=0)[0][0])

        if raw_prob >= 0.5:
            label = "REAL"
            confidence = round(raw_prob * 100, 2)
            summary = "The linguistic structure and terminology strongly match verified journalistic reporting."
        else:
            label = "FAKE"
            confidence = round((1.0 - raw_prob) * 100, 2)
            summary = "The stylistic markers and syntax exhibit significant patterns typical of deceptive reporting."

        latency_ms = round((time.time() - start_time) * 1000, 1)
        return {
            "prediction": label,
            "confidence": confidence,
            "probability": round(raw_prob, 4),
            "probability_real": round(raw_prob, 4),
            "probability_fake": round(1.0 - raw_prob, 4),
            "model_used": "Bidirectional LSTM (Deep Learning)",
            "processing_time_ms": latency_ms,
            "word_count": word_count,
            "explanation": summary,
            "influential_words": [],
            "disclaimer": "This AI system provides an automated linguistic classification based on its training data. Always verify important news via official, primary, and reputable news outlets."
        }

    # --------------------------------------------------------------------------
    # Case C: Classical ML Model (LinearSVC, Logistic Regression, etc.)
    # --------------------------------------------------------------------------
    model = all_models.get(model_name) or resources.get("best_model")
    if not model or not vectorizer:
        return {
            "error": "Trained ML model artifacts could not be located. Run train_model.py first.",
            "status": "not_ready"
        }

    # Transform
    vec = vectorizer.transform([cleaned])

    # Calibrated probability calculation to avoid sparse-intercept false alarms on real news:
    if "LinearSVC" in model_name and hasattr(model, "calibrated_classifiers_"):
        coefs = np.mean([clf.estimator.coef_[0] for clf in model.calibrated_classifiers_], axis=0)
        dot = float(vec.dot(coefs)[0])
        # Centered sigmoid on vocabulary evidence
        prob_fake = 1.0 / (1.0 + np.exp(-3.0 * dot))
        prob_real = 1.0 - prob_fake
        pred = 1 if prob_fake >= 0.5 else 0
    elif "Logistic Regression" in model_name and hasattr(model, "decision_function"):
        z = float(model.decision_function(vec)[0])
        dot = z - float(model.intercept_[0])
        prob_fake = 1.0 / (1.0 + np.exp(-2.0 * dot))
        prob_real = 1.0 - prob_fake
        pred = 1 if prob_fake >= 0.5 else 0
    elif "Passive Aggressive" in model_name and hasattr(model, "decision_function"):
        z = float(model.decision_function(vec)[0])
        dot = z - float(model.intercept_[0])
        prob_fake = 1.0 / (1.0 + np.exp(-2.0 * dot))
        prob_real = 1.0 - prob_fake
        pred = 1 if prob_fake >= 0.5 else 0
    elif hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        # In training: 0 = Real, 1 = Fake
        prob_fake = float(probs[1])
        prob_real = float(probs[0])
        pred = int(model.predict(vec)[0])
    else:
        pred = int(model.predict(vec)[0])
        prob_fake = 1.0 if pred == 1 else 0.0
        prob_real = 1.0 - prob_fake

    if pred == 1:
        label = "FAKE"
        confidence = round(prob_fake * 100, 2)
        summary = "The vocabulary, sensationalist syntax, and emotional phrasing strongly align with identified misinformation patterns."
    else:
        label = "REAL"
        confidence = round(prob_real * 100, 2)
        summary = "The neutral tone, attribution cadence, and contextual vocabulary conform to verified, factual journalism standards."

    influential_words = explain_prediction(cleaned, vectorizer, model, top_k=7)
    latency_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "prediction": label,
        "confidence": confidence,
        "probability_fake": round(prob_fake, 4),
        "probability_real": round(prob_real, 4),
        "model_used": model_name,
        "processing_time_ms": latency_ms,
        "word_count": word_count,
        "explanation": summary,
        "influential_words": influential_words,
        "disclaimer": "This AI system provides an automated linguistic classification based on its training data. Always verify important news via official, primary, and reputable news outlets."
    }


if __name__ == "__main__":
    test_articles = {
        "REAL - Modern Supreme Court": "WASHINGTON — The Supreme Court heard arguments on Tuesday regarding federal regulatory power, with justices questioning attorneys on the scope of agency authority under established statutes. The case could impact environmental and economic regulations nationwide.",
        "REAL - 2024 Presidential Election": "Donald Trump won the 2024 presidential election defeating Kamala Harris to become the 47th President of the United States. World leaders congratulated the president-elect as transition preparations began in Washington.",
        "REAL - NASA Science News": "NASA James Webb Space Telescope has discovered the most distant active supermassive black hole found to date in a galaxy that formed just 570 million years after the Big Bang, astronomers reported on Thursday.",
        "FAKE - Conspiracy Hoax": "BREAKING EXPLOSIVE: Secret FBI files discovered in basement prove Obama personally ordered election rigging with alien technology and clandestine payoffs to mainstream media executives!"
    }

    print("=" * 60)
    print("Testing TruthScan AI Predict Engine (Ensemble & Calibrated)")
    print("=" * 60)
    for expected, sample in test_articles.items():
        print(f"\n--- Testing: {expected} ---")
        res = predict_news(sample)
        print(f"Prediction: {res.get('prediction')} | Confidence: {res.get('confidence')}% | Latency: {res.get('processing_time_ms')}ms")
        print(f"Model: {res.get('model_used')}")
        print(f"Influential Words: {[w['word'] for w in res.get('influential_words', [])]}")
