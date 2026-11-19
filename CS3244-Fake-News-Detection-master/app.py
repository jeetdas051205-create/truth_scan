"""
TruthScan AI - Production Flask Application & REST API
=====================================================
AI-Powered Fake News Detection & Real-World Verification System
"Separate Facts From Fiction."

Backend endpoints:
- POST /api/predict            : Real-time text classification + AI-content analysis
- POST /api/scrape-and-predict : Live URL scraping and automatic verification
- POST /api/check-ai-content   : Stylometric AI-generated content detector
- GET  /api/health             : System health & loaded model status
- GET  /api/model-info         : Metadata on active algorithms & architectures
- GET  /api/metrics            : Authentic model evaluation metrics
- GET  /api/dataset-stats      : Authentic dataset records & class balance
- GET  /api/model-comparison   : Comparative benchmarks table
- GET  /                       : TruthScan AI Web Application
"""

import os
import re
import json
import time
import pickle
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

from url_scraper import scrape_article_from_url
from ai_detector import analyze_ai_content

# Initialize Flask app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR)
app.config['JSON_SORT_KEYS'] = False

# Global state for cached models and data
MODELS_STATE = {
    "vectorizer": None,
    "best_model": None,
    "all_models": {},
    "bilstm_model": None,
    "tokenizer": None,
    "bilstm_maxlen": 100,
    "metrics": {},
    "model_comparison": [],
    "dataset_stats": {},
    "feature_importance": {},
    "active_default_model": "Ensemble (Multi-Model Consensus)",
    "status": "initializing"
}

_PREDICTION_CACHE = {}


def clean_text(text: str) -> str:
    """Preprocess text consistent with training pipeline."""
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


def load_all_artifacts():
    """Load all ML and DL models, vectorizers, and reports at startup."""
    print("[*] TruthScan AI: Initializing backend models & reports...")

    # Load Reports / Metrics
    try:
        metrics_file = os.path.join(REPORTS_DIR, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                MODELS_STATE["metrics"] = json.load(f)

        comp_file = os.path.join(REPORTS_DIR, "model_comparison.json")
        if os.path.exists(comp_file):
            with open(comp_file, "r") as f:
                MODELS_STATE["model_comparison"] = json.load(f)

        stats_file = os.path.join(REPORTS_DIR, "dataset_stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, "r") as f:
                MODELS_STATE["dataset_stats"] = json.load(f)

        feat_file = os.path.join(REPORTS_DIR, "feature_importance.json")
        if os.path.exists(feat_file):
            with open(feat_file, "r") as f:
                MODELS_STATE["feature_importance"] = json.load(f)
    except Exception as e:
        print(f"[-] Error loading reports: {e}")

    # Load Classical Models & Vectorizer
    try:
        vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
        if os.path.exists(vec_path):
            MODELS_STATE["vectorizer"] = joblib.load(vec_path)
            print("[+] Loaded TF-IDF Vectorizer.")

        best_path = os.path.join(MODEL_DIR, "best_model.joblib")
        if os.path.exists(best_path):
            MODELS_STATE["best_model"] = joblib.load(best_path)
            print("[+] Loaded Best Model.")

        all_path = os.path.join(MODEL_DIR, "all_models.joblib")
        if os.path.exists(all_path):
            MODELS_STATE["all_models"] = joblib.load(all_path)
            print(f"[+] Loaded {len(MODELS_STATE['all_models'])} classical ML models.")
    except Exception as e:
        print(f"[-] Error loading classical models: {e}")

    # Load Deep Learning Bi-LSTM
    try:
        bilstm_path = os.path.join(MODEL_DIR, "fake_news_bilstm.keras")
        tok_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
        if os.path.exists(bilstm_path) and os.path.exists(tok_path):
            import keras
            MODELS_STATE["bilstm_model"] = keras.models.load_model(bilstm_path)
            with open(tok_path, "rb") as f:
                MODELS_STATE["tokenizer"] = pickle.load(f)
            MODELS_STATE["bilstm_maxlen"] = int(MODELS_STATE["bilstm_model"].input_shape[1])
            print("[+] Loaded Bidirectional LSTM Deep Learning Model.")
    except Exception as e:
        print(f"[-] Bi-LSTM loading notice: {e}")

    MODELS_STATE["status"] = "ready"
    print("[+] TruthScan AI Engine ready.")


# Load models once on startup
load_all_artifacts()


def get_available_models_list():
    """Returns the ordered list of user-selectable models."""
    model_list = ["Ensemble (Multi-Model Consensus)"]
    if MODELS_STATE["bilstm_model"] is not None:
        model_list.append("Bidirectional LSTM (Deep Learning)")
    model_list.extend(list(MODELS_STATE["all_models"].keys()))
    return model_list


def explain_prediction(text_clean: str, model_name: str, top_k=7):
    """Identify influential keywords that influenced the prediction."""
    vectorizer = MODELS_STATE.get("vectorizer")
    model = MODELS_STATE["all_models"].get(model_name) or MODELS_STATE.get("best_model")
    if not vectorizer or not model:
        model = MODELS_STATE["all_models"].get("Support Vector Machine (LinearSVC)") or MODELS_STATE["all_models"].get("Logistic Regression")
    if not vectorizer or not model:
        return []

    try:
        words = text_clean.split()
        if not words:
            return []

        feature_names = vectorizer.get_feature_names_out()
        feat_to_idx = {feat: idx for idx, feat in enumerate(feature_names)}

        if hasattr(model, 'coef_'):
            coefs = model.coef_[0]
        elif hasattr(model, 'calibrated_classifiers_'):
            coefs = np.mean([clf.estimator.coef_[0] for clf in model.calibrated_classifiers_], axis=0)
        else:
            return []

        influences = []
        seen = set()
        for w in words:
            if w in feat_to_idx and w not in seen:
                seen.add(w)
                idx = feat_to_idx[w]
                raw_weight = float(coefs[idx])
                nature = "Fake Signal" if raw_weight > 0 else "Credibility Signal"
                influences.append({
                    "word": w,
                    "weight": round(abs(raw_weight), 4),
                    "raw_weight": round(raw_weight, 4),
                    "type": nature
                })

        influences.sort(key=lambda x: x["weight"], reverse=True)
        return influences[:top_k]
    except Exception:
        return []


def run_prediction_core(text: str, model_name: str = None) -> dict:
    """Internal core inference helper returning structured results dictionary."""
    start_time = time.time()
    if not model_name:
        model_name = MODELS_STATE["active_default_model"]

    cleaned = clean_text(text)
    word_count = len(text.split())
    if len(cleaned.split()) < 3:
        return {
            "success": False,
            "error": "The article is too brief for linguistic verification. Please provide at least 15-20 words."
        }

    # Fast cache lookup for repeated predictions
    cache_key = (hash(text.strip()), model_name)
    if cache_key in _PREDICTION_CACHE:
        cached_result = dict(_PREDICTION_CACHE[cache_key])
        cached_result["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        return cached_result

    # AI Content & Authenticity Analysis
    ai_analysis = analyze_ai_content(text)

    vectorizer = MODELS_STATE.get("vectorizer")
    all_models = MODELS_STATE.get("all_models", {})
    has_bilstm = MODELS_STATE["bilstm_model"] is not None and MODELS_STATE["tokenizer"] is not None

    # 1. Ensemble Consensus Engine
    if "Ensemble" in model_name or model_name == "default":
        p_bilstm_fake = None
        if has_bilstm:
            try:
                tokenizer = MODELS_STATE["tokenizer"]
                bilstm = MODELS_STATE["bilstm_model"]
                maxlen = MODELS_STATE["bilstm_maxlen"]
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                seq = tokenizer.texts_to_sequences([text])
                padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
                p_real_dl = float(bilstm.predict(padded, verbose=0)[0][0])
                p_bilstm_fake = 1.0 - p_real_dl
            except Exception as e:
                p_bilstm_fake = None

        vec = vectorizer.transform([cleaned])

        # Centered LinearSVC dot product
        svc = all_models.get("Support Vector Machine (LinearSVC)")
        if svc and hasattr(svc, "calibrated_classifiers_"):
            coefs = np.mean([clf.estimator.coef_[0] for clf in svc.calibrated_classifiers_], axis=0)
            dot_svc = float(vec.dot(coefs)[0])
            p_svc_fake = 1.0 / (1.0 + np.exp(-3.0 * dot_svc))
        else:
            p_svc_fake = 0.5

        # Random Forest
        rf = all_models.get("Random Forest Classifier")
        if rf:
            p_rf_fake = float(rf.predict_proba(vec)[0][1])
        else:
            p_rf_fake = 0.5

        # Logistic Regression
        lr = all_models.get("Logistic Regression")
        if lr and hasattr(lr, "decision_function"):
            z_lr = float(lr.decision_function(vec)[0])
            dot_lr = z_lr - float(lr.intercept_[0])
            p_lr_fake = 1.0 / (1.0 + np.exp(-2.0 * dot_lr))
        else:
            p_lr_fake = 0.5

        # Weighted Multi-Model Consensus
        if p_bilstm_fake is not None:
            prob_fake = 0.50 * p_bilstm_fake + 0.25 * p_svc_fake + 0.25 * p_rf_fake
        else:
            prob_fake = 0.40 * p_svc_fake + 0.30 * p_rf_fake + 0.30 * p_lr_fake

        prob_real = 1.0 - prob_fake

        if prob_fake >= 0.50:
            pred_label = "FAKE"
            confidence = round(prob_fake * 100, 2)
            exp = "Multi-Model Consensus: Deep sequence syntax and lexical feature distributions exhibit high correlation with sensationalist or unverified reporting patterns."
        else:
            pred_label = "REAL"
            confidence = round(prob_real * 100, 2)
            exp = "Multi-Model Consensus: Linguistic tone, attribution cadence, and contextual vocabulary conform to verified journalistic reporting standards."

        influential = explain_prediction(cleaned, "Support Vector Machine (LinearSVC)", top_k=7)
        latency = round((time.time() - start_time) * 1000, 2)

        res_dict = {
            "success": True,
            "prediction": pred_label,
            "confidence": confidence,
            "probability_fake": round(prob_fake, 4),
            "probability_real": round(prob_real, 4),
            "model_used": "Ensemble (Multi-Model Consensus)",
            "latency_ms": latency,
            "word_count": word_count,
            "explanation": exp,
            "influential_words": influential,
            "ai_analysis": ai_analysis,
            "disclaimer": "This AI system provides an automated linguistic classification based on its training data. Always verify important news via official, primary, and reputable news outlets."
        }
        _PREDICTION_CACHE[cache_key] = res_dict
        return res_dict

    # 2. DL Bi-LSTM Branch
    if "LSTM" in model_name and has_bilstm:
        try:
            tokenizer = MODELS_STATE["tokenizer"]
            bilstm = MODELS_STATE["bilstm_model"]
            maxlen = MODELS_STATE["bilstm_maxlen"]
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq = tokenizer.texts_to_sequences([text])
            padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
            raw_prob = float(bilstm.predict(padded, verbose=0)[0][0])

            if raw_prob >= 0.5:
                pred_label = "REAL"
                confidence = round(raw_prob * 100, 2)
                exp = "Linguistic syntax, attribution flow, and lexical patterns strongly correlate with verified journalistic standards."
            else:
                pred_label = "FAKE"
                confidence = round((1.0 - raw_prob) * 100, 2)
                exp = "Syntax displays elevated sensationalism, rhetorical framing, and stylistic variance typical of unverified reporting."

            latency = round((time.time() - start_time) * 1000, 2)
            res_dict = {
                "success": True,
                "prediction": pred_label,
                "confidence": confidence,
                "probability_raw": round(raw_prob, 4),
                "probability_real": round(raw_prob, 4),
                "probability_fake": round(1.0 - raw_prob, 4),
                "model_used": "Bidirectional LSTM (Deep Learning)",
                "latency_ms": latency,
                "word_count": word_count,
                "explanation": exp,
                "influential_words": [],
                "ai_analysis": ai_analysis,
                "disclaimer": "This AI system provides an automated linguistic classification based on its training data. Always verify important news via official, primary, and reputable news outlets."
            }
            _PREDICTION_CACHE[cache_key] = res_dict
            return res_dict
        except Exception as e:
            print(f"[-] BiLSTM inference error: {e}")

    # 3. Classical ML Branch
    model = all_models.get(model_name) or MODELS_STATE.get("best_model")
    if not vectorizer or not model:
        return {
            "success": False,
            "error": "Model artifacts unavailable. Please ensure models are trained."
        }

    vec = vectorizer.transform([cleaned])

    if "LinearSVC" in model_name and hasattr(model, "calibrated_classifiers_"):
        coefs = np.mean([clf.estimator.coef_[0] for clf in model.calibrated_classifiers_], axis=0)
        dot = float(vec.dot(coefs)[0])
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
        prob_fake = float(probs[1])
        prob_real = float(probs[0])
        pred = int(model.predict(vec)[0])
    else:
        pred = int(model.predict(vec)[0])
        prob_fake = 1.0 if pred == 1 else 0.0
        prob_real = 1.0 - prob_fake

    if pred == 1:
        pred_label = "FAKE"
        confidence = round(prob_fake * 100, 2)
        exp = "The vocabulary, sensationalist syntax, and emotional phrasing strongly align with identified misinformation patterns."
    else:
        pred_label = "REAL"
        confidence = round(prob_real * 100, 2)
        exp = "The neutral tone, attribution cadence, and contextual vocabulary conform to verified, factual journalism standards."

    influential = explain_prediction(cleaned, model_name, top_k=7)
    latency = round((time.time() - start_time) * 1000, 2)

    res_dict = {
        "success": True,
        "prediction": pred_label,
        "confidence": confidence,
        "probability_fake": round(prob_fake, 4),
        "probability_real": round(prob_real, 4),
        "model_used": model_name,
        "latency_ms": latency,
        "word_count": word_count,
        "explanation": exp,
        "influential_words": influential,
        "ai_analysis": ai_analysis,
        "disclaimer": "This AI system provides an automated linguistic classification based on its training data. Always verify important news via official, primary, and reputable news outlets."
    }
    _PREDICTION_CACHE[cache_key] = res_dict
    return res_dict


# ==============================================================================
#                               API ENDPOINTS
# ==============================================================================

@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "TruthScan AI Engine",
        "models_ready": MODELS_STATE["status"] == "ready",
        "features": [
            "Real-Time URL Scraping & Verification",
            "Stylometric AI vs Human Detection",
            "Multi-Model Ensemble Consensus",
            "Interactive 3D Three.js Visualization"
        ],
        "available_algorithms": get_available_models_list(),
        "timestamp": time.time()
    })


@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    """Metadata regarding active algorithms and architecture."""
    return jsonify({
        "system_name": "TRUTHSCAN AI",
        "version": "3.0.0 (3D & Real-World Web Edition)",
        "tagline": "Separate Facts From Fiction",
        "active_default_model": MODELS_STATE["active_default_model"],
        "available_models": get_available_models_list(),
        "features": [
            "Live Web Article Ingestion & URL Scraping",
            "Stylometric AI-Generated Content Checker",
            "Multi-Model Consensus Soft Voting",
            "Centered Vocabulary Evidence Calibration"
        ],
        "vectorizer": "TF-IDF (10,000 features, n-gram 1-2, sublinear TF)",
        "deep_learning_architecture": "Bi-LSTM (Embedding 128d -> BiLSTM 128 -> Dense 32 -> Dense 1, Sigmoid)",
        "primary_dataset": "ISOT Fake News Corpus (34,103 verified articles)",
        "benchmark_dataset": "FakeNewsNet (PolitiFact + GossipCop, 23,196 articles)"
    })


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Evaluation metrics for all models."""
    return jsonify(MODELS_STATE["metrics"])


@app.route("/api/dataset-stats", methods=["GET"])
def api_dataset_stats():
    """Authentic dataset statistics."""
    return jsonify(MODELS_STATE["dataset_stats"])


@app.route("/api/model-comparison", methods=["GET"])
def api_model_comparison():
    """Comparative benchmarking table."""
    return jsonify(MODELS_STATE["model_comparison"])


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict whether a submitted news article is Real or Fake."""
    data = request.get_json(silent=True) or request.form
    text = data.get("text", "") or data.get("news", "")
    model_name = data.get("model", MODELS_STATE["active_default_model"])

    if not text or not text.strip():
        return jsonify({
            "success": False,
            "error": "The news text cannot be empty. Please enter or paste an article."
        }), 400

    result = run_prediction_core(text, model_name)
    status_code = 200 if result.get("success", False) else 400
    return jsonify(result), status_code


@app.route("/api/scrape-and-predict", methods=["POST"])
def api_scrape_and_predict():
    """
    Scrapes a live news URL, extracts article body, and runs AI verification.
    """
    data = request.get_json(silent=True) or request.form
    url = data.get("url", "")
    model_name = data.get("model", MODELS_STATE["active_default_model"])

    if not url or not url.strip():
        return jsonify({
            "success": False,
            "error": "Please enter a valid website URL."
        }), 400

    # 1. Scrape Article
    scrape_res = scrape_article_from_url(url)
    if not scrape_res.get("success", False):
        return jsonify({
            "success": False,
            "error": scrape_res.get("error", "Unable to extract article text from the URL.")
        }), 400

    # 2. Run Verification on Extracted Content
    text_to_verify = f"{scrape_res['title']}\n\n{scrape_res['inference_text']}"
    pred_res = run_prediction_core(text_to_verify, model_name)

    if not pred_res.get("success", False):
        return jsonify({
            "success": False,
            "error": pred_res.get("error", "Error analyzing extracted content.")
        }), 400

    # Combine Scraped Details + Prediction
    combined_response = {
        "success": True,
        "scraped_details": {
            "url": scrape_res["url"],
            "domain": scrape_res["domain"],
            "site_name": scrape_res["site_name"],
            "title": scrape_res["title"],
            "word_count": scrape_res["word_count"],
            "read_time_min": scrape_res["read_time_min"],
            "excerpt": scrape_res["excerpt"]
        },
        "prediction": pred_res["prediction"],
        "confidence": pred_res["confidence"],
        "probability_fake": pred_res.get("probability_fake", 0.0),
        "probability_real": pred_res.get("probability_real", 0.0),
        "model_used": pred_res["model_used"],
        "latency_ms": pred_res["latency_ms"],
        "explanation": pred_res["explanation"],
        "influential_words": pred_res.get("influential_words", []),
        "ai_analysis": pred_res.get("ai_analysis", {}),
        "disclaimer": pred_res["disclaimer"]
    }
    return jsonify(combined_response)


@app.route("/api/check-ai-content", methods=["POST"])
def api_check_ai_content():
    """Standalone endpoint to evaluate whether text is AI-generated."""
    data = request.get_json(silent=True) or request.form
    text = data.get("text", "")

    if not text or not text.strip():
        return jsonify({
            "success": False,
            "error": "Please provide text to evaluate for AI markers."
        }), 400

    res = analyze_ai_content(text)
    return jsonify({
        "success": True,
        "text_length": len(text.split()),
        "ai_analysis": res
    })


# Legacy POST /verify route backward compatibility
@app.route("/verify", methods=["POST"])
def legacy_verify():
    """Backward compatibility for legacy form POST submissions."""
    news = request.form.get("news", "")
    res = api_predict()
    if isinstance(res, tuple):
        return render_template("result.html", result="Invalid input", confidence=0)
    data = res.get_json()
    return render_template(
        "result.html",
        result=data.get("prediction", "UNKNOWN"),
        confidence=data.get("confidence", 0)
    )


# ==============================================================================
#                               FRONTEND ROUTE
# ==============================================================================

@app.route("/")
def index():
    """Main application landing page and SPA container."""
    return render_template(
        "index.html",
        models=get_available_models_list(),
        default_model=MODELS_STATE["active_default_model"],
        dataset_stats=MODELS_STATE["dataset_stats"],
        model_comparison=MODELS_STATE["model_comparison"]
    )


if __name__ == "__main__":
    print("[*] Starting TruthScan AI Server at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
