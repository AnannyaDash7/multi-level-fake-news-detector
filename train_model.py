"""
Fake News Detector - Model Training Script
--------------------------------------------
Trains a TF-IDF + Logistic Regression classifier on labeled news
articles (REAL vs FAKE) and saves the trained model + vectorizer
to disk for reuse.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "data.csv"
MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

# 1. Load data
print("Step 1: Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df)} articles")
print(f"  Label distribution:\n{df['label'].value_counts()}\n")

# 2. Combine title + text into a single feature
#    (titles in fake news are often very telling - clickbait style etc.)
print("Step 2: Preparing text...")
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

X = df["content"]
y = df["label"]

# 3. Split into train/test sets (80/20), stratified to keep label balance
print("Step 3: Splitting into train/test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}\n")

# 4. Convert text to TF-IDF features
print("Step 4: Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=10000,     # keep top 10k most informative words/phrases
    stop_words="english",   # remove common words like "the", "is", "and"
    ngram_range=(1, 2),     # consider single words AND two-word phrases
    max_df=0.7,             # ignore words that appear in >70% of articles (too common)
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}\n")

# 5. Train the classifier
print("Step 5: Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, C=10)
model.fit(X_train_tfidf, y_train)
print("  Training complete.\n")

# 6. Evaluate
print("Step 6: Evaluating on test set...")
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"  Accuracy: {acc:.4f}\n")
print("  Classification Report:")
print(classification_report(y_test, y_pred))
print("  Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Save model + vectorizer for reuse
print("\nStep 7: Saving model and vectorizer to disk...")
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
print(f"  Saved: {MODEL_PATH}")
print(f"  Saved: {VECTORIZER_PATH}")
print("\nDone! Run predict.py to test it on your own text.")
