"""
Fake News Detector - Try It Yourself
--------------------------------------
Loads the trained model and lets you type in any headline/article
to see the prediction.
"""

import joblib

MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

print("Loading trained model...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
print("Ready!\n")


def predict(text: str):
    """Returns (label, confidence, probability_dict) for the given text."""
    X = vectorizer.transform([text])
    label = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    confidence = max(proba)
    prob_dict = {cls: round(p, 4) for cls, p in zip(classes, proba)}
    return label, confidence, prob_dict


if __name__ == "__main__":
    print("=" * 60)
    print("FAKE NEWS DETECTOR")
    print("Type a headline or article text below. Type 'quit' to exit.")
    print("=" * 60)

    while True:
        text = input("\nEnter text: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not text:
            print("(empty input, try again)")
            continue

        label, confidence, probs = predict(text)
        print(f"\n  Prediction : {label}")
        print(f"  Confidence : {confidence:.2%}")
        print(f"  Probabilities: {probs}")
