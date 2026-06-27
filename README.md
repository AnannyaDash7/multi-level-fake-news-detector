# Multi-Level Fake News & Rumor Detector

A system to assess how trustworthy a news claim/headline is, combining:
1. A **style-based ML classifier** (TF-IDF + Logistic Regression)
2. A **live, multi-level corroboration checker** — verifies whether a claim
   is being independently reported at the **Global**, **National (India)**,
   and **Local/Regional** levels, using live news search.

## Why this approach

Most fake news detection projects classify a single article based on its
writing style/vocabulary. This project adds a different, complementary
signal: **does independent reporting actually corroborate this claim, and
at what geographic scope?** A claim reported by multiple independent
outlets across global, national, and local levels is far more likely to
be genuine than one with no independent corroboration anywhere.

## Components

### 1. ML Classifier (`train_model.py`, `predict.py`)
- TF-IDF vectorization (unigrams + bigrams, 10,000 features) + Logistic Regression
- Trained on the "Fake or Real News" dataset (George McIntire), ~6,300 articles
- **Test accuracy: 93.84%**
- Best on full-length articles; less reliable on short headlines (documented limitation)

### 2. Multi-Level Corroboration Checker (`multi_level_detector.py`)
- Searches live news via the NewsData.io API
- **Global**: worldwide search, no country filter
- **National**: India-wide search (`country=in`)
- **Local**: India search + matching against a curated list of known
  regional/state-level Indian outlets
- Each level produces a confidence score (0-100%) based on how many
  distinct independent sources corroborate the claim
- The three scores combine (equal weight) into one overall verdict:
  LIKELY REAL / UNVERIFIED / LIKELY FAKE

### 3. Web Interface (`app.py`)
- Gradio-based UI: paste a claim, optionally specify a state/region,
  see all three confidence scores plus the overall verdict and matching
  sources

## How to run

```bash
pip install scikit-learn pandas joblib requests gradio

# ML model demo (offline, already trained)
python predict.py

# Multi-level corroboration checker (needs free NewsData.io API key)
# Get a key at https://newsdata.io/register
export NEWSDATA_KEY="your_key_here"      # Windows PowerShell: $env:NEWSDATA_KEY="..."
python multi_level_detector.py

# Web interface
python app.py
# then open http://127.0.0.1:7860 in your browser
```

## Known limitations (honest, by design)

- **ML model**: trained on 2016-era US political news; performs best on
  similar topics/style and full-length articles, less reliably on short
  headlines or unrelated domains.
- **Corroboration checker**: depends on NewsData.io's free tier, which has
  a ~12 hour indexing delay and a 1-month historical window. Very recent
  (same-day) news may not yet be indexed. Regional outlet coverage is
  currently a curated starter list (Odisha-focused) and not yet
  comprehensive across all states/UTs.
- **The ML model and corroboration checker are currently separate signals**,
  not yet combined into a single unified score — a planned next step.

## Project status / roadmap

- [x] ML baseline classifier (93.8% accuracy)
- [x] Live corroboration checking (global/national/local)
- [x] Web interface (Gradio)
- [ ] Combine ML + corroboration into one unified hybrid score
- [ ] Expand regional outlet coverage to all Indian states/UTs
- [ ] Add known fact-checker cross-reference (Alt News, PIB Fact Check, Boom)
- [ ] Evaluate corroboration system accuracy on a labeled test set of
      known real/fake claims
- [ ] Explainability (highlight which words/sources drove each score)

## Files

| File | Purpose |
|---|---|
| `train_model.py` | Trains the ML classifier from scratch |
| `predict.py` | Interactive CLI to test the ML model |
| `model.joblib`, `vectorizer.joblib` | Saved trained ML model |
| `multi_level_detector.py` | Core corroboration logic (global/national/local) |
| `app.py` | Gradio web interface |
| `data.csv` | Training dataset for the ML model |

## Author

[Your Name] — Final Year Project
