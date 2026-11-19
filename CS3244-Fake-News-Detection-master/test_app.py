"""
TruthScan AI - Automated Validation & Integration Test Suite
===========================================================
Tests:
1. Model loading & health endpoint
2. Metadata & Model-info endpoint
3. Dataset stats endpoint
4. Metrics endpoint
5. Model comparison endpoint
6. Real news inference via POST /api/predict (Ensemble Consensus & LinearSVC)
7. Contemporary 2024 real news verification (no false alarms on real news)
8. Fake news inference via POST /api/predict
9. Empty input validation & rejection (400)
10. Short input rejection (400)
11. Bi-LSTM deep learning inference
12. HTML rendering of index.html & 3D Three.js container
13. AI-Generated content detection endpoint (/api/check-ai-content)
14. URL Scraper input validation & error handling (/api/scrape-and-predict)
"""

import sys
import os
import json

# Add current folder to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app import app

client = app.test_client()

print("=" * 70)
print("     TRUTHSCAN AI - SYSTEM VALIDATION & ENDPOINT VERIFICATION")
print("=" * 70)

# 1. Health Endpoint
print("\n[TEST 1] GET /api/health")
res = client.get("/api/health")
print(f"Status Code: {res.status_code}")
data = res.get_json()
assert res.status_code == 200
assert data["status"] == "healthy"
assert data["models_ready"] is True
print(f"PASS: Service healthy, {len(data['available_algorithms'])} algorithms online.")

# 2. Model-Info Endpoint
print("\n[TEST 2] GET /api/model-info")
res = client.get("/api/model-info")
print(f"Status Code: {res.status_code}")
data = res.get_json()
assert res.status_code == 200
assert data["system_name"] == "TRUTHSCAN AI"
assert "Ensemble (Multi-Model Consensus)" in data["available_models"]
print(f"PASS: System name: {data['system_name']}, Default Model: {data['active_default_model']}")

# 3. Dataset-Stats Endpoint
print("\n[TEST 3] GET /api/dataset-stats")
res = client.get("/api/dataset-stats")
print(f"Status Code: {res.status_code}")
data = res.get_json()
assert res.status_code == 200
assert data["total_records"] == 34103
assert data["real_count"] == 17056
assert data["fake_count"] == 17047
print(f"PASS: Verified 34,103 dataset records ({data['real_count']} Real, {data['fake_count']} Fake).")

# 4. Metrics Endpoint
print("\n[TEST 4] GET /api/metrics")
res = client.get("/api/metrics")
print(f"Status Code: {res.status_code}")
data = res.get_json()
assert res.status_code == 200
assert "Support Vector Machine (LinearSVC)" in data
assert "Logistic Regression" in data
print(f"PASS: Retrieved evaluation metrics for {len(data)} models.")

# 5. Model Comparison Endpoint
print("\n[TEST 5] GET /api/model-comparison")
res = client.get("/api/model-comparison")
print(f"Status Code: {res.status_code}")
data = res.get_json()
assert res.status_code == 200
assert len(data) >= 5
print(f"PASS: Model comparison table contains {len(data)} benchmarked models.")

# 6. Real News Prediction (Ensemble Default)
print("\n[TEST 6] POST /api/predict (DEFAULT ENSEMBLE CONSENSUS - REAL NEWS)")
real_payload = {
    "text": "WASHINGTON (Reuters) - The U.S. Senate passed sweeping bipartisan legislation on Tuesday to reform federal research grants, sending the comprehensive measure to the President's desk after a final vote in the upper chamber.",
    "model": "Ensemble (Multi-Model Consensus)"
}
res = client.post("/api/predict", json=real_payload)
print(f"Status Code: {res.status_code}")
data = res.get_json()
assert res.status_code == 200
assert data["success"] is True
assert data["prediction"] == "REAL"
assert data["confidence"] > 70.0
assert "ai_analysis" in data
print(f"PASS: Classified as {data['prediction']} ({data['confidence']}% confidence in {data['latency_ms']}ms).")
print(f"AI Authenticity: {data['ai_analysis']['verdict']} (Human: {data['ai_analysis']['human_probability']}%, AI: {data['ai_analysis']['ai_probability']}%)")

# 7. Contemporary Real News (2024 Election & Supreme Court - Zero False Alarm Check)
print("\n[TEST 7] POST /api/predict (CONTEMPORARY REAL NEWS VERIFICATION)")
contemporary_samples = [
    ("2024 Presidential Election", "Donald Trump won the 2024 presidential election defeating Kamala Harris to become the 47th President of the United States. World leaders congratulated the president-elect as transition preparations began in Washington."),
    ("Supreme Court Ruling", "WASHINGTON — The Supreme Court heard arguments on Tuesday regarding federal regulatory power, with justices questioning attorneys on the scope of agency authority under established statutes. The case could impact environmental and economic regulations nationwide.")
]
for title, text in contemporary_samples:
    res = client.post("/api/predict", json={"text": text, "model": "Ensemble (Multi-Model Consensus)"})
    data = res.get_json()
    assert res.status_code == 200
    assert data["prediction"] == "REAL", f"False positive detected on real news: {title}"
    print(f"PASS [{title}]: Correctly classified as REAL with {data['confidence']}% confidence.")

# 8. Fake News Prediction
print("\n[TEST 8] POST /api/predict (FAKE NEWS SAMPLE)")
fake_payload = {
    "text": "BREAKING SHOCKING BOMBSHELL: Unredacted secret FBI files confirm top politicians used alien radio technology to rig the national presidential election in underground basement laboratories!",
    "model": "Ensemble (Multi-Model Consensus)"
}
res = client.post("/api/predict", json=fake_payload)
print(f"Status Code: {res.status_code}")
data = res.get_json()
assert res.status_code == 200
assert data["success"] is True
assert data["prediction"] == "FAKE"
assert data["confidence"] > 70.0
print(f"PASS: Classified as {data['prediction']} ({data['confidence']}% confidence in {data['latency_ms']}ms).")

# 9. Empty Input Validation
print("\n[TEST 9] POST /api/predict (EMPTY INPUT REJECTION)")
res = client.post("/api/predict", json={"text": ""})
print(f"Status Code: {res.status_code}")
data = res.get_json()
assert res.status_code == 400
assert data["success"] is False
print("PASS: Empty input rejected with 400 Bad Request.")

# 10. Short Input Validation
print("\n[TEST 10] POST /api/predict (SHORT INPUT REJECTION)")
res = client.post("/api/predict", json={"text": "hello world"})
print(f"Status Code: {res.status_code}")
data = res.get_json()
assert res.status_code == 400
assert data["success"] is False
print("PASS: Too-short input rejected with 400 Bad Request.")

# 11. Deep Learning Bi-LSTM Prediction
print("\n[TEST 11] POST /api/predict (DEEP LEARNING BI-LSTM MODEL)")
bilstm_payload = {
    "text": "WASHINGTON (Reuters) - The Federal Reserve held benchmark interest rates steady following a two-day policy meeting on monetary stability and inflationary pressures.",
    "model": "Bidirectional LSTM (Deep Learning)"
}
res = client.post("/api/predict", json=bilstm_payload)
print(f"Status Code: {res.status_code}")
data = res.get_json()
assert res.status_code == 200
assert data["success"] is True
assert data["prediction"] == "REAL"
print(f"PASS: Bi-LSTM predicted: {data['prediction']} ({data['confidence']}% confidence in {data['latency_ms']}ms).")

# 12. HTML Web Page Container & 3D Canvas
print("\n[TEST 12] GET / (WEB UI CONTAINER & 3D CANVASES)")
res = client.get("/")
print(f"Status Code: {res.status_code}")
assert res.status_code == 200
assert b"TRUTHSCAN AI" in res.data
assert b"three-canvas" in res.data
assert b"Live URL Scanner" in res.data
print("PASS: Main single-page web UI loaded with 3D canvas and URL Scanner tabs.")

# 13. AI-Generated Content Detection Endpoint
print("\n[TEST 13] POST /api/check-ai-content (AI vs HUMAN STYLOMETRIC DETECTOR)")
human_sample = "WASHINGTON — The Supreme Court heard arguments on Tuesday regarding federal regulatory power, with justices questioning attorneys on the scope of agency authority. Court officials confirmed a decision by June."
ai_sample = "In the ever-evolving landscape of modern governance, judicial rulings play a crucial role. Furthermore, it is important to note that navigating the complexities of regulatory frameworks serves as a reminder of our collective democratic tapestry. In conclusion, the multifaceted approach remains pivotal."

res_human = client.post("/api/check-ai-content", json={"text": human_sample})
data_human = res_human.get_json()
assert res_human.status_code == 200
assert data_human["ai_analysis"]["human_probability"] >= 60.0
print(f"PASS Human Text: Human Prob = {data_human['ai_analysis']['human_probability']}%, Verdict = {data_human['ai_analysis']['verdict']}")

res_ai = client.post("/api/check-ai-content", json={"text": ai_sample})
data_ai = res_ai.get_json()
assert res_ai.status_code == 200
assert data_ai["ai_analysis"]["ai_probability"] >= 60.0
print(f"PASS AI Text: AI Prob = {data_ai['ai_analysis']['ai_probability']}%, Verdict = {data_ai['ai_analysis']['verdict']}")

# 14. URL Scraper Input Validation
print("\n[TEST 14] POST /api/scrape-and-predict (URL INPUT VALIDATION)")
res_empty_url = client.post("/api/scrape-and-predict", json={"url": ""})
assert res_empty_url.status_code == 400
print("PASS: Empty URL rejected with 400 Bad Request.")

print("\n" + "=" * 70)
print("     ALL 14 INTEGRATION, 3D, URL & AI-DETECTION TESTS PASSED!")
print("=" * 70)
