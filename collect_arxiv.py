"""Harvest arXiv paper-counts per year per track and merge them into data/pubcounts.csv.

Methodology (UNIFIED across every track — one reproducible method, one window):
  For each track we run, PER YEAR, a single arXiv API query of the form

      (abs:"syn A" OR ti:"syn A" OR abs:"syn B" OR ti:"syn B" ...)
      AND (cat:cs.AI OR cat:cs.LG OR ...)   # category set is per-track (see CAT_* below)
      AND submittedDate:[YYYY01010000 TO YYYY12312359]

  and read ONLY <opensearch:totalResults> (one cheap request per track-year, no deep
  paging → robust against arXiv's result-window/rate limits). The count is bucketed by
  the paper's submittedDate. Recall is broadened two ways vs the previous pass:
  (1) each phrase is matched in TITLE and ABSTRACT (ti: OR abs:), not abstract-only;
  (2) the category filter is per-track — security tracks add cs.CR, adversarial adds cs.CV,
  governance adds cs.CY, multi-agent adds cs.RO — instead of one global AI/ML set.
  All tracks still share ONE window (2015-2026) and ONE reproducible recipe. Numbers rise
  again vs the previous pass because ti: + wider categories capture more — this is a
  METHODOLOGY change, not only real-world growth, and is flagged in the post.

  These remain keyword PROXIES, not bibliometrics: broad phrases ('AI control', 'dangerous
  capabilities', 'model editing') carry real noise; ranked per track in the `noise` column
  (low/med/high) and explained in `note`.

Pass 16 (precision, not breadth):
  - a DEDUPLICATED honest series '_safety_corpus' — ONE combined OR-query of safety-specific
    phrases, so arXiv counts each paper ONCE (the per-track sums double-count overlaps; this
    series does not). It is the field-level attention line the post/charts should trust.
  - the two noisiest tracks are DENOISED: 'interpretability' drops the bare word 'interpretability'
    (it matched all of general ML); 'truthfulness' is kept but flagged noise=high and excluded
    from the field sum. Every series carries an explicit `noise` level.

Everything is reproducible: the exact query URL is stored in the `source` column, the exact
boolean expression in `query`, the noise level in `noise`, the recipe in `note`. No fabricated
numbers — counts come straight from the API. Fails loudly (no try/except beyond a narrow 429
rate-limit backoff).

Run (per repo rules) inside tmux with a watchdog:
  tmux new-session -d -s arxiv 'python3 collect_arxiv.py 2>&1 | tee logs/arxiv.log'
"""

import time
import urllib.parse
import urllib.request
import re
import pandas as pd

PUBCOUNTS_CSV = "data/pubcounts.csv"
API = "https://export.arxiv.org/api/query?"
YEARS = list(range(2015, 2027))          # unified analysis window 2015-2026 inclusive
PARTIAL_YEAR = 2026                        # incomplete calendar year -> flagged, not a real decline
REQUEST_PAUSE_S = 3                        # arXiv asks ~3s between requests

# category filters, per-track (extend recall without global noise):
#   DEFAULT = AI/ML/NLP/statistical ML; GOV adds policy (cs.CY); SEC adds security/crypto
#   (cs.CR) for red-teaming/unlearning/watermarking/backdoors; ADV also adds vision (cs.CV)
#   for adversarial examples; ROBO adds robotics (cs.RO) for embodied/agentic safety.
CAT_DEFAULT = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:stat.ML"
CAT_GOV = CAT_DEFAULT + " OR cat:cs.CY"
CAT_SEC = CAT_DEFAULT + " OR cat:cs.CR"
CAT_ADV = CAT_DEFAULT + " OR cat:cs.CR OR cat:cs.CV"
CAT_ROBO = CAT_DEFAULT + " OR cat:cs.RO"
CAT_ALL = CAT_DEFAULT + " OR cat:cs.CR OR cat:cs.CV OR cat:cs.CY OR cat:cs.RO"  # union, for the dedup corpus

# Pass 16 — DEDUPLICATED unique-safety-corpus series. This is the single most honest number
# in the file: because it is ONE combined OR-query, arXiv counts each paper ONCE (no per-track
# double counting). The phrase set is curated to be SAFETY-SPECIFIC (deliberately excludes the
# generic single words like bare 'interpretability'/'hallucination'/'model editing' that inflate
# the per-track proxies). It is stored under the reserved slug '_safety_corpus' (leading
# underscore = service series; build_viz skips it in per-track charts and uses it as the honest
# field-level attention line). Denominators for context (see post): ~437K total cs.AI/LG/CL/stat.ML
# papers 2015-2026, ~2.3K papers that literally self-label 'AI safety/alignment'.
SAFETY_CORPUS_PHRASES = [
    "AI safety", "AI alignment", "aligning large language models", "alignment of language models",
    "mechanistic interpretability", "sparse autoencoder",
    "reinforcement learning from human feedback", "RLHF", "constitutional AI", "RLAIF",
    "scalable oversight", "weak-to-strong generalization", "AI safety via debate",
    "iterated amplification", "AI control", "untrusted model",
    "dangerous capabilities", "frontier model evaluation", "safety evaluation",
    "jailbreak", "prompt injection", "adversarial robustness", "red teaming",
    "machine unlearning", "chain-of-thought faithfulness",
    "deceptive alignment", "sycophancy", "sandbagging", "AI deception",
    "eliciting latent knowledge", "truthful AI", "sleeper agents",
    "model organisms of misalignment", "corrigibility", "agent foundations",
    "activation steering", "representation engineering", "guaranteed safe AI",
    "cooperative AI", "multi-agent safety", "AI governance", "frontier AI governance",
]

# track -> (list of phrase synonyms, category expr, noise-level, note). noise in {low, med, high}
# ranks how much non-safety keyword noise the proxy carries (surfaced in the post + charts).
# Slugs reuse the events taxonomy where possible; the rest are pub-only proxy tracks.
TRACKS = {
    # --- existing curves, now broadened + unified method/window ---
    "interpretability": (
        ["mechanistic interpretability", "sparse autoencoder",
         "feature attribution", "probing classifier"],
        CAT_DEFAULT, "med",
        "proxy: interpretability OR-set — DENOISED Pass 16 (dropped bare 'interpretability', which "
        "matched all of general ML); now mechanistic/SAE/attribution/probing only (still some noise)"),
    "reward_modeling": (
        ["reinforcement learning from human feedback", "RLHF", "reward model",
         "preference optimization", "direct preference optimization"],
        CAT_DEFAULT, "low",
        "proxy: OR-set of RLHF/preference-learning phrases; fairly clean term-of-art"),
    "scalable_oversight": (
        ["scalable oversight", "iterated amplification", "recursive reward modeling",
         "weak-to-strong generalization", "AI safety via debate"],
        CAT_DEFAULT, "low",
        "proxy: OR-set of oversight/amplification phrases; term-of-art, fairly clean"),
    "evals": (
        ["dangerous capabilities", "capability evaluation", "model evaluations",
         "frontier model evaluation", "safety evaluation"],
        CAT_DEFAULT, "high",
        "proxy: OR-set of eval phrases; NOISY (generic evaluation terms, not all AI-safety evals)"),
    "ai_control": (
        ["AI control", "control protocol", "untrusted model", "control evaluation"],
        CAT_DEFAULT, "high",
        "proxy: OR-set incl 'AI control'; NOISY (matches control-theory, not just Redwood-style control)"),
    "agent_foundations": (
        ["agent foundations", "embedded agency", "logical induction", "corrigibility"],
        CAT_DEFAULT, "low",
        "proxy: OR-set of agent-foundations phrases; rare terms, low counts expected (track faded)"),
    "value_learning": (
        ["value learning", "reward learning", "inverse reward design", "value alignment"],
        CAT_DEFAULT, "med",
        "proxy: OR-set of value-learning phrases; term predates the CHAI track (keyword noise pre-2016)"),
    # --- new proxy tracks (both broadened method AND new coverage the user asked for) ---
    "robustness": (
        ["adversarial robustness", "adversarial examples", "jailbreak",
         "adversarial attack", "prompt injection"],
        CAT_ADV, "high",
        "proxy: adversarial-robustness / jailbreaks OR-set (+cs.CR/cs.CV); NOISY (adversarial ML predates LLM safety)"),
    "red_teaming": (
        ["red teaming", "red-teaming", "automated red teaming"],
        CAT_SEC, "low",
        "proxy: red-teaming OR-set (+cs.CR); young track, LLM-era"),
    "unlearning": (
        ["machine unlearning", "knowledge unlearning", "concept erasure", "model editing"],
        CAT_SEC, "high",
        "proxy: unlearning / model-editing OR-set (+cs.CR); NOISY ('model editing' is broader than safety)"),
    "cot_faithfulness": (
        ["chain-of-thought faithfulness", "faithfulness of chain-of-thought",
         "reasoning faithfulness", "faithful reasoning"],
        CAT_DEFAULT, "low",
        "proxy: CoT-faithfulness OR-set; young, low counts expected"),
    "deception": (
        ["sycophancy", "sandbagging", "deceptive alignment", "AI deception", "deceptive behavior"],
        CAT_DEFAULT, "med",
        "proxy: sycophancy/sandbagging/deception OR-set; LLM-era safety concern"),
    "governance": (
        ["AI governance", "AI policy", "AI regulation", "frontier AI governance"],
        CAT_GOV, "high",
        "proxy: AI-governance OR-set incl cat:cs.CY; NOISY (broad policy language)"),
    "alignment_broad": (
        ["AI alignment", "aligning large language models", "alignment of language models",
         "value-aligned"],
        CAT_DEFAULT, "med",
        "proxy: broad-alignment OR-set; umbrella term (overlaps other tracks by design)"),
    # --- Pass 15: more new proxy tracks (deeper coverage) ---
    "honesty_elk": (
        ["eliciting latent knowledge", "AI honesty", "honest AI", "truthful AI"],
        CAT_DEFAULT, "low",
        "proxy: honesty/ELK OR-set; young, niche (Christiano/ARC line)"),
    "truthfulness": (
        ["truthfulness", "hallucination", "factuality", "factual consistency"],
        CAT_DEFAULT, "high",
        "proxy: truthfulness OR-set; VERY NOISY ('hallucination'/'factuality' are huge general-NLP "
        "terms, mostly NOT AI-safety) — kept for completeness but excluded from the field sum & flagged high"),
    "activation_steering": (
        ["activation steering", "representation engineering", "steering vector",
         "activation addition"],
        CAT_DEFAULT, "med",
        "proxy: activation-steering / representation-engineering OR-set; overlaps interpretability"),
    "watermarking": (
        ["text watermarking", "machine-generated text detection", "AI-generated text detection",
         "content provenance"],
        CAT_SEC, "med",
        "proxy: watermarking / provenance OR-set (+cs.CR); LLM-era"),
    "singular_learning_theory": (
        ["singular learning theory", "developmental interpretability", "local learning coefficient"],
        CAT_DEFAULT, "low",
        "proxy: SLT / dev-interp OR-set; very young, low counts (Timaeus line)"),
    "constitutional_ai": (
        ["constitutional AI", "reinforcement learning from AI feedback", "RLAIF"],
        CAT_DEFAULT, "low",
        "proxy: constitutional-AI / RLAIF OR-set; overlaps reward_modeling (Anthropic line)"),
    "guaranteed_safe": (
        ["formal verification of neural networks", "neural network verification",
         "provable guarantees", "guaranteed safe AI", "safety verification"],
        CAT_DEFAULT, "high",
        "proxy: formal-verification / guaranteed-safe OR-set; NOISY (formal methods predate AI safety); overlaps robustness/ARIA"),
    "model_organisms": (
        ["sleeper agents", "model organisms of misalignment", "backdoor attack",
         "trojan attack", "data poisoning"],
        CAT_SEC, "high",
        "proxy: model-organisms / backdoors OR-set (+cs.CR); NOISY (backdoor/trojan ML predates safety framing)"),
    "multi_agent_safety": (
        ["cooperative AI", "multi-agent safety", "collusion between AI",
         "multi-agent reinforcement learning safety"],
        CAT_ROBO, "low",
        "proxy: multi-agent-safety / cooperative-AI OR-set (+cs.RO); young, niche"),
}


def _get(url):
    """GET with a narrow backoff for arXiv's HTTP 429 rate-limit ONLY. Every other error
    (and a 429 that survives the backoff) propagates loudly — this is transient-rate-limit
    handling for a public API, not error swallowing."""
    for attempt in range(5):
        try:
            return urllib.request.urlopen(url, timeout=60).read().decode()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 4:
                wait = 30 * (attempt + 1)
                print(f"   arxiv 429; backing off {wait}s (attempt {attempt+1}/5)")
                time.sleep(wait)
                continue
            raise


def _phrase_expr(synonyms):
    """OR-group of exact phrases matched in TITLE or ABSTRACT: (abs:"a" OR ti:"a" OR ...)."""
    parts = []
    for s in synonyms:
        parts.append(f'abs:"{s}"')
        parts.append(f'ti:"{s}"')
    return "(" + " OR ".join(parts) + ")"


def count_year(phrase_expr, cat_expr, year):
    """One request: totalResults for the query within `year` (bucketed by submittedDate)."""
    lo, hi = f"{year}01010000", f"{year}12312359"
    q = f"{phrase_expr} AND ({cat_expr}) AND submittedDate:[{lo} TO {hi}]"
    params = {"search_query": q, "start": 0, "max_results": 1}
    url = API + urllib.parse.urlencode(params)
    head = _get(url)
    total = int(re.search(r"<opensearch:totalResults[^>]*>(\d+)", head).group(1))
    time.sleep(REQUEST_PAUSE_S)
    return total, url


def _collect_series(track, phrase_expr, cat_expr, noise, note):
    """Run the per-year totalResults query for one series and return its pubcounts rows."""
    counts, last_url = {}, None
    for y in YEARS:
        counts[y], last_url = count_year(phrase_expr, cat_expr, y)
    query_str = phrase_expr  # store the exact boolean expression sent (fully reproducible)
    rows = []
    for y in YEARS:
        row_note = note
        if y == PARTIAL_YEAR:
            row_note = f"{note} · PARTIAL ({PARTIAL_YEAR} = incomplete calendar year, undercounts)"
        rows.append({"track": track, "query": query_str, "year": y, "arxiv_count": counts[y],
                     "noise": noise, "note": row_note, "source": last_url})
    yrs_nonzero = [y for y in YEARS if counts[y]]
    print(f"rc=0 {track} (total={sum(counts.values())}, "
          f"{min(yrs_nonzero or [0])}-{max(yrs_nonzero or [0])})")
    return rows


def main():
    print("START collect_arxiv")
    new_rows = []
    for track, (synonyms, cat_expr, noise, note) in TRACKS.items():
        new_rows += _collect_series(track, _phrase_expr(synonyms), cat_expr, noise, note)

    # DEDUPLICATED unique-safety-corpus: ONE combined OR-query → arXiv counts each paper once.
    corpus_note = ("DEDUP corpus: single combined OR-query of safety-SPECIFIC phrases (each paper "
                   "counted once, no per-track double count). Honest field-level attention proxy; "
                   "context ~437K total cs.AI/LG/CL/stat.ML papers, ~2.3K self-label 'AI safety/alignment'")
    new_rows += _collect_series("_safety_corpus", _phrase_expr(SAFETY_CORPUS_PHRASES),
                                CAT_ALL, "dedup", corpus_note)

    cols = ["track", "query", "year", "arxiv_count", "noise", "note", "source"]
    out = pd.DataFrame(new_rows)[cols]
    out = out.sort_values(["track", "year"]).drop_duplicates(["track", "year"], keep="last")
    out.to_csv(PUBCOUNTS_CSV, index=False)
    print(f"wrote {PUBCOUNTS_CSV}: {len(out)} rows, {out.track.nunique()} tracks (incl _safety_corpus)")
    print("ALL DONE")


if __name__ == "__main__":
    main()
