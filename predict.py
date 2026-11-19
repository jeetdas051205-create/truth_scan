"""
Root Entrypoint for TruthScan AI Predict Engine
==============================================
Provides command-line & library inference access.
"""

import sys
import os
import importlib.util

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBDIR = os.path.join(CURRENT_DIR, "CS3244-Fake-News-Detection-master")
if SUBDIR not in sys.path:
    sys.path.insert(0, SUBDIR)

predict_module_path = os.path.join(SUBDIR, "predict.py")
spec = importlib.util.spec_from_file_location("sub_predict", predict_module_path)
sub_predict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub_predict)

predict_news = sub_predict.predict_news
clean_text = sub_predict.clean_text
load_inference_resources = sub_predict.load_inference_resources

if __name__ == "__main__":
    test_articles = {
        "REAL - Modern Supreme Court": "WASHINGTON — The Supreme Court heard arguments on Tuesday regarding federal regulatory power, with justices questioning attorneys on the scope of agency authority under established statutes. The case could impact environmental and economic regulations nationwide.",
        "REAL - 2024 Presidential Election": "Donald Trump won the 2024 presidential election defeating Kamala Harris to become the 47th President of the United States. World leaders congratulated the president-elect as transition preparations began in Washington.",
        "REAL - NASA Science News": "NASA James Webb Space Telescope has discovered the most distant active supermassive black hole found to date in a galaxy that formed just 570 million years after the Big Bang, astronomers reported on Thursday.",
        "FAKE - Conspiracy Hoax": "BREAKING EXPLOSIVE: Secret FBI files discovered in basement prove Obama personally ordered election rigging with alien technology and clandestine payoffs to mainstream media executives!"
    }

    print("=" * 60)
    print("Testing TruthScan AI Predict Engine (Root Wrapper)")
    print("=" * 60)
    for expected, sample in test_articles.items():
        print(f"\n--- Testing: {expected} ---")
        res = predict_news(sample)
        print(f"Prediction: {res.get('prediction')} | Confidence: {res.get('confidence')}% | Latency: {res.get('processing_time_ms')}ms")
        print(f"Model: {res.get('model_used')}")
        print(f"Influential Words: {[w['word'] for w in res.get('influential_words', [])]}")
