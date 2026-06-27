"""
Multi-Level Fake News & Rumor Detector
------------------------------------------------------------
Takes a news claim/headline and checks how corroborated it is at
THREE levels:
    - GLOBAL    : worldwide news sources (no country filter)
    - NATIONAL  : Indian national news sources (country=in)
    - LOCAL     : regional/state-level Indian sources (matched by
                  known regional outlet names + state name in query)

Each level gets its own confidence score (0-100%) based on how many
distinct sources are reporting something similar. The three scores
are combined (equal weight) into one overall verdict.

This is the "real-time" corroboration signal - it checks claims
against LIVE, CURRENT news, unlike a standalone ML model trained on
old/offline data.

Usage:
    Set NEWSDATA_KEY as an environment variable, then run:
    python multi_level_detector.py
"""

import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- CONFIG ----
NEWSDATA_KEY = os.environ.get("NEWSDATA_KEY", "PASTE_YOUR_KEY_HERE")
NEWSDATA_URL = "https://newsdata.io/api/1/latest"

SIMILARITY_THRESHOLD = 0.20

# How many articles to consider per search/level.
# NOTE: NewsData.io's free tier 'latest' endpoint returns a MAXIMUM of 10
# articles per request regardless of this number - this setting only
# controls how many of those returned articles we actually use/display.
# (Paid plans support a higher per-request article count.)
ARTICLES_PER_SEARCH = 10

# Maximum number of words from the claim/headline to actually send as the
# search query. Long, full-sentence queries tend to match poorly on
# NewsData.io/NewsAPI (too many "noise" words dilute the match) - short,
# keyword-style queries work much better. Change this number to control
# how aggressively long claims get trimmed before searching.
# NOTE: NewsData.io enforces a hard 100-CHARACTER limit on the free tier
# (not a word limit) - so this is set high, but MAX_QUERY_CHARS below is
# the real safety cap that prevents errors regardless of word count.
MAX_QUERY_WORDS = 20

# NewsData.io hard limit: query text cannot exceed 100 characters.
# This is enforced in addition to MAX_QUERY_WORDS, since some words
# (e.g. "Chhattisgarh-based") are long enough that even a small number
# of words can exceed the character limit.
MAX_QUERY_CHARS = 100


def trim_query(text: str, max_words: int = MAX_QUERY_WORDS, max_chars: int = MAX_QUERY_CHARS) -> str:
    """
    Keeps only the first `max_words` words of the input, then further
    truncates to `max_chars` characters if still too long (NewsData.io
    rejects queries over 100 characters with a 422 error).
    """
    words = text.split()
    trimmed = " ".join(words[:max_words])
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars].rsplit(" ", 1)[0]  # cut at last full word
    return trimmed

# A small list of known Indian regional/state-level outlets, used to
# identify which corroborating sources count as "local" coverage.
# (Not exhaustive - can be extended over time.)
REGIONAL_OUTLETS = {
    "odishabytes", "otv", "odisha tv", "kalinga tv", "kanak news",
    "dharitri", "samaja", "the new indian express", "sambad",
    "etv bharat", "lokmat", "eenadu", "sakshi", "malayala manorama",
    "the hindu tamil", "dinamalar", "anandabazar patrika", "prajavani",
    "deccan herald", "deccan chronicle",
}

# Major national/international outlets, used just for reference/labeling
NATIONAL_OUTLETS_HINT = {
    "the times of india", "ndtv", "hindustan times", "the indian express",
    "india today", "news18", "zee news", "abp news", "the hindu",
    "outlook india", "india.com", "republic world",
}


def search_news(query: str, country: str = None, page_size: int = ARTICLES_PER_SEARCH):
    """Search NewsData.io 'latest' endpoint. Returns normalized article list."""
    if NEWSDATA_KEY == "PASTE_YOUR_KEY_HERE" or not NEWSDATA_KEY:
        raise ValueError("No NewsData.io key set. Set env var NEWSDATA_KEY.")

    params = {"apikey": NEWSDATA_KEY, "q": query, "language": "en"}
    if country:
        params["country"] = country

    response = requests.get(NEWSDATA_URL, params=params, timeout=15)
    if response.status_code != 200:
        raise RuntimeError(f"NewsData.io error (HTTP {response.status_code}): {response.text[:300]}")

    data = response.json()
    if data.get("status") != "success":
        raise RuntimeError(f"NewsData.io error: {data.get('results', data)}")

    articles = data.get("results", [])
    normalized = []
    for a in articles[:page_size]:
        normalized.append({
            "title": a.get("title", "") or "",
            "description": a.get("description", "") or "",
            "url": a.get("link", ""),
            "source": (a.get("source_name") or a.get("source_id") or "Unknown"),
            "publishedAt": a.get("pubDate", ""),
        })
    return normalized


def compute_similarity(claim: str, articles: list):
    """TF-IDF + cosine similarity between claim and each article's title+description."""
    if not articles:
        return []
    texts = [claim] + [a["title"] + " " + a["description"] for a in articles]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    scored = list(zip(articles, sims))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def score_to_confidence(num_sources: int) -> float:
    """
    Converts a count of distinct corroborating sources into a 0-100
    confidence score. Diminishing returns after a few sources.
    """
    if num_sources == 0:
        return 5.0   # not zero - claim might still be true but unreported online
    if num_sources == 1:
        return 40.0
    if num_sources == 2:
        return 65.0
    if num_sources == 3:
        return 80.0
    return min(95.0, 80.0 + (num_sources - 3) * 5)


def get_corroborating_sources(claim: str, articles: list):
    """Filters articles by similarity threshold, dedupes by source name."""
    scored = compute_similarity(claim, articles)
    corroborating = [(a, s) for a, s in scored if s >= SIMILARITY_THRESHOLD]
    distinct = {}
    for article, sim in corroborating:
        name = article["source"]
        if name not in distinct or sim > distinct[name][1]:
            distinct[name] = (article, sim)
    return distinct, scored


def check_global(claim: str):
    """No country filter - worldwide search."""
    articles = search_news(trim_query(claim), country=None)
    distinct, scored = get_corroborating_sources(claim, articles)
    confidence = score_to_confidence(len(distinct))
    return {
        "level": "GLOBAL",
        "confidence": confidence,
        "num_sources": len(distinct),
        "sources": distinct,
        "raw_count": len(articles),
        "debug_top_scores": scored[:5],
    }


def check_national(claim: str):
    """India-wide search using country=in filter."""
    articles = search_news(trim_query(claim), country="in")
    distinct, scored = get_corroborating_sources(claim, articles)
    confidence = score_to_confidence(len(distinct))
    return {
        "level": "NATIONAL",
        "confidence": confidence,
        "num_sources": len(distinct),
        "sources": distinct,
        "raw_count": len(articles),
        "debug_top_scores": scored[:5],
    }


def check_local(claim: str, state_hint: str = None):
    """
    Regional/local search. Since the free tier doesn't support a direct
    state/region filter, we:
      1. Optionally append the state name to the query for better targeting
      2. Search within India (country=in)
      3. Identify which corroborating sources are known REGIONAL outlets
    """
    if state_hint:
        # Trim claim a bit more to leave room for the state name, then
        # re-trim the combined string to guarantee we stay under the limit
        query = trim_query(f"{trim_query(claim, max_words=MAX_QUERY_WORDS)} {state_hint}")
    else:
        query = trim_query(claim)
    articles = search_news(query, country="in")
    distinct, scored = get_corroborating_sources(claim, articles)

    # Split into regional vs national-looking sources
    regional_matches = {
        name: val for name, val in distinct.items()
        if name.strip().lower() in REGIONAL_OUTLETS
    }

    # If we found regional-specific matches, prioritize those for the score;
    # otherwise fall back to all distinct matches (better than nothing)
    effective = regional_matches if regional_matches else distinct
    confidence = score_to_confidence(len(effective))

    return {
        "level": "LOCAL",
        "confidence": confidence,
        "num_sources": len(effective),
        "sources": effective,
        "all_distinct_sources": distinct,
        "raw_count": len(articles),
        "debug_top_scores": scored[:5],
    }


def overall_verdict(global_score, national_score, local_score):
    """Equal-weighted average of the three level confidences -> verdict label."""
    combined = (global_score + national_score + local_score) / 3.0
    if combined >= 70:
        verdict = "LIKELY REAL"
    elif combined >= 40:
        verdict = "UNVERIFIED / MIXED EVIDENCE"
    else:
        verdict = "LIKELY FAKE / UNCORROBORATED"
    return combined, verdict


def analyze_claim(claim: str, state_hint: str = None):
    """Runs all three checks and prints a full report."""
    print(f"\nAnalyzing: '{claim}'")
    if state_hint:
        print(f"(Local search hint: {state_hint})")
    print("-" * 60)

    print("Checking GLOBAL coverage...")
    g = check_global(claim)

    print("Checking NATIONAL (India) coverage...")
    n = check_national(claim)

    print("Checking LOCAL/REGIONAL coverage...")
    l = check_local(claim, state_hint)

    combined, verdict = overall_verdict(g["confidence"], n["confidence"], l["confidence"])

    print("\n" + "=" * 60)
    print(f"RESULTS for: '{claim}'")
    print("=" * 60)
    print(f"  GLOBAL   confidence: {g['confidence']:.0f}%  ({g['num_sources']} sources)")
    print(f"  NATIONAL confidence: {n['confidence']:.0f}%  ({n['num_sources']} sources)")
    print(f"  LOCAL    confidence: {l['confidence']:.0f}%  ({l['num_sources']} sources)")
    print("-" * 60)
    print(f"  OVERALL CONFIDENCE: {combined:.0f}%")
    print(f"  VERDICT: {verdict}")
    print("=" * 60)

    for level_result in (g, n, l):
        if level_result["sources"]:
            print(f"\n  [{level_result['level']}] Matching sources:")
            for name, (article, sim) in level_result["sources"].items():
                print(f"    - {name}: {article['title'][:60]} (sim={sim:.2f})")

    return {
        "claim": claim,
        "global": g,
        "national": n,
        "local": l,
        "overall_confidence": combined,
        "verdict": verdict,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-LEVEL FAKE NEWS & RUMOR DETECTOR")
    print("Checks claims against global, national, and local news")
    print("=" * 60)

    while True:
        claim = input("\nEnter a news claim/headline (or 'quit'): ").strip()
        if claim.lower() in ("quit", "exit", "q"):
            break
        if not claim:
            continue

        state = input("Optional: state/region name for local search (press Enter to skip): ").strip()
        state = state if state else None

        try:
            analyze_claim(claim, state_hint=state)
        except Exception as e:
            print(f"Error: {e}")
