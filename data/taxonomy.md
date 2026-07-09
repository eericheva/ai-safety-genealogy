# Taxonomy of directions (tracks) — grows as collection proceeds

The `direction` value in events.csv. We split finely: "focus declared" / "first publication" / "grant to the track" as separate events.

## Large tracks
- `agent_foundations` — decision theory, logical induction, corrigibility (early MIRI)
- `value_learning` — IRL, CIRL, assistance games (CHAI/Russell)
- `reward_modeling` — RLHF, deep RL from human prefs, RLAIF (the Christiano line)
- `scalable_oversight` — debate, IDA, recursive reward modeling, weak-to-strong, CAIS-services
- `interpretability` — circuits, superposition, SAE, attribution [expected from the 2020+ side]
- `evals` — dangerous capabilities, red-teaming, model evals [2022+]
- `ai_control` — the Redwood track [2023+]
- `governance` — policy, compute governance, safety institutes, principles
- `macrostrategy` — existential risk strategy, field forecasting (FHI/CSER/GCRI)
- `forecasting` — timelines, takeoff speeds, AI progress (AI Impacts)

## Smaller / service tracks
- `robustness` — adversarial, safety gridworlds
- `multi_agent_safety` — ARCHES, cooperative AI
- `disclosure_norms` — staged release, publication norms (GPT-2, OpenAI charter, MIRI nondisclosure)
- `model_organisms` — sleeper agents, artificial "organisms" of misalignment (Anthropic 2024+)
- `field_building` — funds, fellowships, forums, newsletters, prizes, courses
- `capabilities` — pure capabilities events that shape focus (GPT-3, DeepMind acquisition)

## Aggregate (funding analytics, not research tracks)
- `mixed_technical` — the field as a whole / a funder's overall annual total (OpenPhil/LTFF overall)
- `technical_safety` — a funder's "technical AI safety research" bucket (OpenPhil/LTFF "technical-research" subtotal)

## arXiv-proxy tracks (attention curves only in pubcounts.csv, not money tracks)
A single collection method (OR-synonyms in title+abstract `ti:`/`abs:` + a per-track category filter,
window 2015-2026, see collect_arxiv.py). Some reuse the money slugs (`robustness`,
`governance`, `model_organisms`, `multi_agent_safety` — these also get a pub curve);
the rest are proxy curves only:
- `red_teaming` — red-teaming, automated red teaming (LLM era; +cs.CR)
- `unlearning` — machine unlearning, concept erasure, model editing (+cs.CR)
- `cot_faithfulness` — chain-of-thought faithfulness, faithful reasoning
- `deception` — sycophancy, sandbagging, deceptive alignment
- `alignment_broad` — the umbrella "AI alignment" (overlaps other tracks)
- `honesty_elk` — eliciting latent knowledge, honest/truthful AI (the ARC line)
- `truthfulness` — truthfulness, hallucination, factuality (NOISY: general NLP)
- `activation_steering` — activation steering, representation engineering (⊂ interpretability)
- `watermarking` — text watermarking, machine-generated-text detection, provenance (+cs.CR)
- `singular_learning_theory` — SLT, developmental interpretability (Timaeus; ⊂ interpretability)
- `constitutional_ai` — constitutional AI, RLAIF (⊂ reward_modeling; the Anthropic line)
- `guaranteed_safe` — formal verification, provable guarantees (NOISY; ⊂ robustness/ARIA)

### The `noise` column (Pass 16) — proxy honesty
Every proxy track in pubcounts.csv carries a `noise` level (low/med/high): how much the curve is
contaminated by non-safety papers. `high` = the term shows up massively in general ML/NLP and badly inflates
the count (`evals`, `ai_control`, `robustness`, `unlearning`, `governance`, `truthfulness`,
`guaranteed_safe`, `model_organisms`); `low` = a stable term-of-art (`reward_modeling`,
`scalable_oversight`, `red_teaming`, `cot_faithfulness`, ...). The post and charts rank trust by
this flag.

### Denoising the two worst tracks (Pass 16)
- `interpretability`: the bare word `interpretability` was REMOVED from the OR-set (it caught almost all of general ML);
  what remains is mechanistic interpretability / sparse autoencoder / feature attribution / probing classifier.
  The count drops sharply — this is MORE HONEST, not "undercounting".
- `truthfulness`: kept for completeness, but flagged `noise=high` and excluded from the field sum.

### The service dedup series `_safety_corpus` (Pass 16, the MAIN honest artifact)
ONE combined OR-query of safety-specific phrases (mechanistic interpretability, RLHF,
scalable oversight, AI control, jailbreak, red teaming, deceptive alignment, sleeper agents, ELK,
constitutional AI, dangerous capabilities, ...), `ti:`+`abs:`, unified categories. Because the query is
SINGLE — arXiv counts each paper EXACTLY ONCE (auto-deduplication), so this is an honest series of
"unique safety papers/year". The leading-underscore slug (`_`) = service: build_viz shows
it as the honest field-level attention line in `money_vs_attention` and does NOT draw it as a regular track on
the per-track charts (the `startswith('_')` filter). Denominators for context: ~437K total
cs.AI/LG/CL/stat.ML papers for 2015-2026; ~2.3K papers that literally self-label as "AI safety/alignment".

Umbrella/subset proxies (`alignment_broad`, `constitutional_ai`, `activation_steering`,
`guaranteed_safe`, `singular_learning_theory`, `model_organisms`, `honesty_elk`, `truthfulness`)
are excluded from the FIELD attention sum (build_viz `FIELD_SUM_EXCLUDE`), so as not to count the same
papers twice; on the per-track charts they are shown in full. The per-track sum is an inflated
keyword proxy; the honest number is given by `_safety_corpus`.

## Money lenses (event_type)
- `funding_total` — a funder's annual total (OpenPhil/LTFF), the "how much the funder gave" lens.
- `grant_out` — a specific itemized grant (the "which track the money went to" lens).
- `investment` (Pass 16) — VC/equity into a safety startup (Goodfire, Gray Swan, Lakera, HiddenLayer,
  Protect AI). A SEPARATE lens: this is NOT a charitable grant; it is never summed with
  grants/totals and never enters the grant $-charts (build_viz `MONEY_TYPES`).
- `statement` — stated/unconfirmed amounts, national programs, acquisitions, coalition money —
  outside the $-charts to avoid double counting.

## Flow-merging rule
When a track "fades", it often doesn't die but merges into another (reward_modeling -> mainstream RLHF).
We encode such transitions with a `pivot` event plus a justifying link — this is exactly the material for the alluvial flows.
