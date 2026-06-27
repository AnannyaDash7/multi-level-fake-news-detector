"""
Multi-Level Fake News & Rumor Detector - Web Interface
------------------------------------------------------------
A Gradio web app wrapping the multi_level_detector logic.
Paste a claim/headline, optionally add a state/region for local
search, and see GLOBAL / NATIONAL / LOCAL confidence scores plus
an overall verdict.

Usage:
    pip install gradio
    Set NEWSDATA_KEY as an environment variable
    python app.py
    (opens automatically in your browser, usually at http://127.0.0.1:7860)
"""

import gradio as gr
from multi_level_detector import check_global, check_national, check_local, overall_verdict


def format_sources(level_result):
    """Turns a level's matching sources into a readable markdown bullet list."""
    sources = level_result["sources"]
    if not sources:
        return "_No corroborating sources found._"
    lines = []
    for name, (article, sim) in sources.items():
        title = article["title"][:80]
        url = article.get("url", "")
        if url:
            lines.append(f"- **{name}**: [{title}]({url}) (similarity: {sim:.2f})")
        else:
            lines.append(f"- **{name}**: {title} (similarity: {sim:.2f})")
    return "\n".join(lines)


def run_detector(claim: str, state_hint: str):
    if not claim or not claim.strip():
        return (
            "Please enter a claim or headline.",
            0, 0, 0, 0,
            "", "", ""
        )

    state_hint = state_hint.strip() if state_hint else None

    try:
        g = check_global(claim)
        n = check_national(claim)
        l = check_local(claim, state_hint)
    except Exception as e:
        error_msg = f"⚠️ Error: {e}"
        return (error_msg, 0, 0, 0, 0, "", "", "")

    combined, verdict = overall_verdict(g["confidence"], n["confidence"], l["confidence"])

    verdict_emoji = {
        "LIKELY REAL": "✅",
        "UNVERIFIED / MIXED EVIDENCE": "⚠️",
        "LIKELY FAKE / UNCORROBORATED": "❌",
    }.get(verdict, "")

    verdict_text = f"## {verdict_emoji} {verdict}\n\n**Overall Confidence: {combined:.0f}%**"

    return (
        verdict_text,
        g["confidence"],
        n["confidence"],
        l["confidence"],
        combined,
        format_sources(g),
        format_sources(n),
        format_sources(l),
    )


with gr.Blocks(title="Fake News & Rumor Detector") as demo:
    gr.Markdown(
        """
        # 🔍 Multi-Level Fake News & Rumor Detector
        Paste a news claim or headline below. The system checks **live news**
        at three levels — Global, National (India), and Local/Regional — to see
        how widely it's being corroborated by real sources right now.
        """
    )

    with gr.Row():
        claim_input = gr.Textbox(
            label="News claim / headline",
            placeholder="e.g. Odisha ammonia gas leak deaths rise",
            lines=2,
        )
    with gr.Row():
        state_input = gr.Textbox(
            label="State/Region (optional, improves local search)",
            placeholder="e.g. Odisha",
        )

    check_btn = gr.Button("Check Claim", variant="primary")

    verdict_output = gr.Markdown()

    with gr.Row():
        global_score = gr.Number(label="🌍 Global Confidence (%)", interactive=False)
        national_score = gr.Number(label="🇮🇳 National Confidence (%)", interactive=False)
        local_score = gr.Number(label="📍 Local Confidence (%)", interactive=False)
        overall_score = gr.Number(label="📊 Overall Confidence (%)", interactive=False)

    with gr.Accordion("🌍 Global sources found", open=False):
        global_sources_output = gr.Markdown()
    with gr.Accordion("🇮🇳 National sources found", open=False):
        national_sources_output = gr.Markdown()
    with gr.Accordion("📍 Local sources found", open=False):
        local_sources_output = gr.Markdown()

    gr.Markdown(
        """
        ---
        **Note:** This tool checks corroboration against live news (via NewsData.io),
        not a fact-checking database. A low score means the claim isn't widely
        reported yet — it could be very recent, hyperlocal, or unverified.
        Free-tier news APIs have coverage and freshness limits (articles may be
        delayed ~12 hours, and regional coverage varies by outlet).
        """
    )

    check_btn.click(
        fn=run_detector,
        inputs=[claim_input, state_input],
        outputs=[
            verdict_output,
            global_score, national_score, local_score, overall_score,
            global_sources_output, national_sources_output, local_sources_output,
        ],
    )

if __name__ == "__main__":
    demo.launch()
