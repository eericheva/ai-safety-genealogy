"""Phase 5 — build every AI-safety-genealogy visualisation + report from data/*.csv.

Data-first: nothing is hand-drawn. Every chart and every number is derived from
data/events.csv (the single source of truth) and data/pubcounts.csv. No fabricated
numbers: what the data cannot support is flagged in the dashboard, not invented.

Rerun after any data edit:  python3 build_viz.py

Data volume is tiny (~200 rows), so aggregations are trivial pandas groupbys — no
GPU is needed or used here. Code fails loudly (asserts, no try/except) by design.

All tunable parameters live in the PARAMS block below and flow downward into the
chart functions; nothing is redefined ad-hoc inside the functions.
"""

import math
import os
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------- PARAMS ----
EVENTS_CSV = "data/events.csv"
PUBCOUNTS_CSV = "data/pubcounts.csv"
TAXONOMY_MD = "data/taxonomy.md"
VIZ_DIR = "viz"
IMG_DIR = "viz/img"          # static PNG exports live here (viz/img/<name>.png)
IMG_SCALE = 2                # PNG resolution multiplier (fig width/height x this)
REPORT_MD = "report.md"

# dark theme (matches the mrs_wallbreaker site the charts are embedded in)
DARK_BG = "#0f1117"
DARK_PANEL = "#171a23"
DARK_LINE = "#2a2f3c"
DARK_MUTED = "#93a0b5"
DARK_ACCENT = "#5b8def"
DARK_FG = "#e6e9f0"

# the latest calendar year is incomplete (data collected mid-year) -> shown but flagged
# everywhere as partial so the H1-only undercount is never read as a real decline
PARTIAL_YEAR = 2026

# shared x-axis range for the year-based charts (data 2015-2025 + room for end labels)
YEAR_AXIS_RANGE = [2014.5, 2026.5]

# continuous colorscale sampled to colour research tracks by their year of first appearance,
# so the timeline colour itself encodes a track's "age" (early -> late along the spectrum)
TRACK_COLORSCALE = "Turbo"

REQUIRED_COLS = [
    "year", "date", "actor", "actor_type", "direction",
    "event_type", "scale", "detail", "source_url", "source_type", "confidence",
]

# event types that carry money (actor = funder, not the organisation that "did" the work).
# 'investment' = VC/equity INTO a safety startup — a SEPARATE lens that is never summed with
# grant/donor money; listed here so the non-money charts (lifespan/alluvial/births) exclude it
# via ~event_type.isin(MONEY_TYPES). The $-charts filter explicitly on grant_out/funding_total,
# so equity never leaks into grant totals either.
MONEY_TYPES = ["grant_out", "grant_in", "funding_total", "investment"]

# aggregate/funding-analytic directions (not research tracks) — excluded where we
# want per-research-track signal, to avoid double counting donor totals
AGGREGATE_TRACKS = ["mixed_technical", "technical_safety"]

# narrative eras (label, first_year, last_year) — used to bin the alluvial + shade the timeline
ERAS = [
    ("2005-2012 · pre-history", 2005, 2012),
    ("2013-2016 · founding", 2013, 2016),
    ("2017-2019 · institutionalization", 2017, 2019),
    ("2020-2021 · prosaic turn", 2020, 2021),
    ("2022-2023 · scaling & ChatGPT", 2022, 2023),
    ("2024-2026 · consolidation", 2024, 2026),
]

# Shared x-range for every year-axis chart below the "cross-cutting patterns" heading, so they
# line up with the timeline/track_lifespan span above (2005..2026) instead of each auto-scaling
# to its own data. Derived from ERAS (first era start … last era end), never hardcoded.
YEAR_AXIS_RANGE = [ERAS[0][1] - 0.5, ERAS[-1][2] + 0.5]

# --- readable groupings for the compact alluvial variants (era → actor-type → track-family) ---
# The raw alluvial has ~115 organisation nodes and 17 raw track slugs → unreadable spaghetti.
# These maps collapse it: organisations into 7 actor-type buckets, tracks into 4 named families.
# Labels are English (charts are reader-facing but must stay English-only).
ERA_RU = {
    "2005-2012 · pre-history": "2005–2012 · pre-history",
    "2013-2016 · founding": "2013–2016 · founding",
    "2017-2019 · institutionalization": "2017–2019 · institutionalization",
    "2020-2021 · prosaic turn": "2020–2021 · prosaic turn",
    "2022-2023 · scaling & ChatGPT": "2022–2023 · scaling & ChatGPT",
    "2024-2026 · consolidation": "2024–2026 · consolidation",
}

ACTOR_TYPE_RU = {
    "research": "Researchers",
    "funder": "Funders",
    "governance": "Policy/institutes",
    "training": "Talent training",
    "industry": "Industry",
    "meta": "Meta/community",
    "government": "Government",
}

# qualitative palette for the 7 actor types (order matches ACTOR_TYPE_RU)
ACTOR_TYPE_COLORS = {
    "Researchers": "#4C78A8",
    "Funders": "#F58518",
    "Policy/institutes": "#54A24B",
    "Talent training": "#B279A2",
    "Industry": "#E45756",
    "Meta/community": "#72B7B2",
    "Government": "#EECA3B",
}

# every research-track slug maps to exactly one family (asserted at build time — fail loudly)
TRACK_FAMILY = {
    "macrostrategy": "Foundations & strategy",
    "agent_foundations": "Foundations & strategy",
    "value_learning": "Foundations & strategy",
    "forecasting": "Foundations & strategy",
    "interpretability": "Technical safety",
    "evals": "Technical safety",
    "ai_control": "Technical safety",
    "reward_modeling": "Technical safety",
    "scalable_oversight": "Technical safety",
    "model_organisms": "Technical safety",
    "robustness": "Technical safety",
    "multi_agent_safety": "Technical safety",
    "red_teaming": "Technical safety",
    "unlearning": "Technical safety",
    "cot_faithfulness": "Technical safety",
    "deception": "Technical safety",
    "alignment_broad": "Technical safety",
    "honesty_elk": "Technical safety",
    "truthfulness": "Technical safety",
    "activation_steering": "Technical safety",
    "watermarking": "Technical safety",
    "singular_learning_theory": "Technical safety",
    "constitutional_ai": "Technical safety",
    "guaranteed_safe": "Technical safety",
    "mixed_technical": "Technical safety",
    "governance": "Governance & infrastructure",
    "field_building": "Governance & infrastructure",
    "disclosure_norms": "Governance & infrastructure",
    "capabilities": "Capabilities",
}

FAMILY_ORDER = [
    "Foundations & strategy",
    "Technical safety",
    "Governance & infrastructure",
    "Capabilities",
]

FAMILY_COLORS = {
    "Foundations & strategy": "#4C78A8",
    "Technical safety": "#F58518",
    "Governance & infrastructure": "#54A24B",
    "Capabilities": "#B279A2",
}

# Umbrella / subset / noisy-overlap arXiv-proxy tracks EXCLUDED from the FIELD-WIDE attention
# sum only, to avoid double-counting the same papers (each overlaps a retained core track by
# design): alignment_broad is an umbrella over everything; constitutional_ai ⊂ reward_modeling
# (RLAIF); activation_steering & singular_learning_theory ⊂ interpretability; guaranteed_safe ⊂
# robustness (formal methods); model_organisms ⊂ robustness/deception (backdoors); honesty_elk &
# truthfulness overlap deception/factuality. They STILL get their own per-track curves — only the
# single field-level SUM in chart_money_vs_attention uses the disjoint core.
FIELD_SUM_EXCLUDE = {"alignment_broad", "constitutional_ai", "activation_steering",
                     "guaranteed_safe", "singular_learning_theory", "model_organisms",
                     "honesty_elk", "truthfulness"}

# --- cross-era shift chart (era on the X axis; how the field migrates chapter to chapter) ---
# narrative chapters (id, short_tick, first_year, last_year) — cover 2005–2026 gap-free.
# These are the reader-facing chapters (Part N); distinct from ERAS (the 6 fine bins used
# in the overview alluvial/timeline shading).
NARRATIVE_ERAS = [
    (1, "2005–13 · Pt1", 2005, 2013),
    (2, "2014–19 · Pt2", 2014, 2019),
    (3, "2020–21 · Pt3", 2020, 2021),
    (4, "2022–23 · Pt4", 2022, 2023),
    (6, "2024–26 · Pt6", 2024, 2026),
]

# fixed funder set for the money panel (exact actor strings in data → short display labels).
ERA_FUNDERS = ["Open Philanthropy", "SFF", "FTX Future Fund", "FLI", "LTFF", "Jaan Tallinn"]
ERA_FUNDER_LABELS = {
    "Open Philanthropy": "OpenPhil", "SFF": "SFF", "FTX Future Fund": "FTX",
    "FLI": "FLI", "LTFF": "LTFF", "Jaan Tallinn": "Tallinn",
}
ERA_FUNDER_COLORS = {
    "OpenPhil": "#F58518", "SFF": "#4C78A8", "FTX": "#E45756",
    "FLI": "#54A24B", "LTFF": "#B279A2", "Tallinn": "#72B7B2",
}
# every grant_out actor NOT in ERA_FUNDERS (government/international institutes, new
# philanthropy) is rolled into ONE aggregated band so the era money panel totals ALL
# itemized grant $, not just the 6 long-standing philanthropic funders. Neutral grey
# (same as the untracked band in chart_funding_absolute) so it steals no funder colour.
ERA_OTHER_LABEL = "other / gov &amp; new"
ERA_OTHER_COLOR = "#BAB0AC"

# the eight "big" research tracks we require year-coverage for (Phase-3/5 coverage control)
BIG_TRACKS = [
    "agent_foundations", "value_learning", "reward_modeling", "scalable_oversight",
    "interpretability", "evals", "ai_control", "governance", "macrostrategy", "forecasting",
]

# milestone publications to annotate on the timeline (year, actor, short label)
MILESTONES = [
    (2014, "Bostrom", "Superintelligence"),
    (2016, "Amodei,Olah,Steinhardt,Christiano,Schulman,Mane", "Concrete Problems"),
    (2017, "Christiano,Leike et al", "Deep RL from Human Prefs (RLHF)"),
    (2020, "OpenAI", "Circuits (mech interp)"),
    (2022, "Anthropic", "Toy Models of Superposition"),
    (2023, "Anthropic", "Towards Monosemanticity (SAE)"),
    (2024, "Anthropic", "Sleeper Agents"),
]

# the tracks for which we currently have verified arXiv attention curves
PUB_TRACKS_VERIFIED = ["interpretability", "reward_modeling"]

# ---------------------------------------------------------------- LOAD ------

def era_of(year):
    for label, lo, hi in ERAS:
        if lo <= year <= hi:
            return label
    raise ValueError(f"year {year} falls outside every era in ERAS")


def taxonomy_tracks():
    text = open(TAXONOMY_MD, encoding="utf-8").read()
    return set(re.findall(r"^-\s+`([a-z_]+)`", text, flags=re.MULTILINE))


def validate(df):
    assert list(df.columns) == REQUIRED_COLS, f"unexpected columns: {list(df.columns)}"
    assert pd.api.types.is_numeric_dtype(df["scale"]), "scale must be numeric"
    assert pd.api.types.is_integer_dtype(df["year"]), "year must be integer"
    tax = taxonomy_tracks()
    unknown = set(df["direction"]) - tax
    assert not unknown, f"directions not in taxonomy.md: {sorted(unknown)}"
    money = df[df.event_type.isin(["grant_out", "funding_total"])]
    assert money["scale"].notna().all(), "every grant_out/funding_total row must carry a scale"
    for y in df["year"].unique():
        era_of(int(y))  # fail loudly if any year is not covered by an era


def load():
    df = pd.read_csv(EVENTS_CSV)
    validate(df)
    pubs = pd.read_csv(PUBCOUNTS_CSV)
    return df, pubs


# ---------------------------------------------------------------- CHARTS ----

def save(fig, name):
    """Write both the interactive HTML and a static PNG (kaleido) for one figure.

    PNG size = the figure's own layout width/height x IMG_SCALE. Fails loudly if the
    image backend is missing — no silent skip.
    """
    fig.update_layout(template="plotly_dark", paper_bgcolor=DARK_BG,
                      plot_bgcolor=DARK_BG, font_color=DARK_FG)
    fig.write_image(f"{IMG_DIR}/{name}.png", scale=IMG_SCALE)   # keep fixed px size for PNG
    if name == "alluvial_org":                                  # left as-is per request
        fig.write_html(f"{VIZ_DIR}/{name}.html", include_plotlyjs="cdn")
    else:
        fig.update_layout(width=None, autosize=True)            # width follows the container
        fig.write_html(f"{VIZ_DIR}/{name}.html", include_plotlyjs="cdn",
                       default_width="100%", config={"responsive": True})


def _track_birth_order(df):
    """Tracks ordered by year of first appearance (tie-break: more events first), and a
    chronological colour map sampled from TRACK_COLORSCALE so colour encodes a track's age."""
    birth = df.groupby("direction")["year"].min()
    size = df.groupby("direction").size()
    track_order = sorted(birth.index, key=lambda t: (birth[t], -size[t]))
    n = len(track_order)
    colors = px.colors.sample_colorscale(TRACK_COLORSCALE, [i / (n - 1) for i in range(n)])
    return track_order, dict(zip(track_order, colors))


def _with_pub_colors(cmap, pubs):
    """Extend the birth-order colour map with distinct colours for arXiv-proxy tracks that
    have NO events row (pub-only tracks like red_teaming/unlearning) — so charts that colour
    attention lines by track never KeyError. Money-track colours are left untouched."""
    extra = [t for t in pubs["track"].unique() if t not in cmap]
    pal = px.colors.qualitative.Dark24
    return {**cmap, **{t: pal[i % len(pal)] for i, t in enumerate(sorted(extra))}}


def chart_timeline(df):
    """Two panels sharing the year axis:
      Row 1 — every event by organisation (y, ordered by first year → bottom-left→top-right
              diagonal), coloured by track. Colour is a chronological gradient (tracks ordered
              by birth year), so the field's colour drift over time reads at a glance.
      Row 2 — stacked area of events per track per year: the same colours, showing the
              activity mix shift (early macrostrategy/agent_foundations → late interp/evals/control).

    Era shading + milestone stars are kept on the main panel. Row 2 is event COUNTS, which is
    collection density, not real-world activity — flagged in-chart (caveat #2).
    """
    track_order, cmap = _track_birth_order(df)
    # actor y-order: first event year asc, tie-break by last event year → clean diagonal
    ao = df.groupby("actor")["year"].agg(["min", "max"]).sort_values(["min", "max"])
    actor_order = ao.index.tolist()
    n_actors = len(actor_order)
    years = list(range(int(df.year.min()), int(df.year.max()) + 1))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2],
                        vertical_spacing=0.04,
                        subplot_titles=("events by organisation & track (colour = track age)",
                                        "events recorded per track per year "
                                        "(CAVEAT: collection density, not real-world activity)"))
    for t in track_order:
        sub = df[df.direction == t]
        cd = sub[["date", "event_type", "detail", "scale", "confidence"]].values
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub["actor"], mode="markers", name=t, legendgroup=t,
            marker=dict(color=cmap[t], size=9, line=dict(width=0.5, color="white")),
            customdata=cd,
            hovertemplate=("<b>%{y}</b> · %{x} · " + t +
                           "<br>%{customdata[1]} (%{customdata[0]})"
                           "<br>%{customdata[2]}"
                           "<br>$%{customdata[3]} · conf=%{customdata[4]}<extra></extra>")),
            row=1, col=1)
        counts = sub.groupby("year").size().reindex(years, fill_value=0)
        fig.add_trace(go.Scatter(
            x=years, y=counts.values, mode="lines", name=t, legendgroup=t, showlegend=False,
            stackgroup="events", line=dict(width=0.5, color=cmap[t]), fillcolor=cmap[t],
            hovertemplate=t + " %{x}: %{y} events<extra></extra>"),
            row=2, col=1)

    fig.update_yaxes(categoryorder="array", categoryarray=actor_order,
                     range=[-2.5, n_actors + 0.5], row=1, col=1)
    fig.update_yaxes(title_text="events/yr", row=2, col=1)
    fig.update_xaxes(dtick=1, gridcolor="#eee", range=[2004, 2027], row=2, col=1)
    fig.update_xaxes(dtick=1, gridcolor="#eee", range=[2004, 2027], row=1, col=1)

    for i, (label, lo, hi) in enumerate(ERAS):
        fig.add_vrect(x0=lo - 0.5, x1=hi + 0.5, fillcolor="LightSalmon",
                      opacity=0.05 + 0.02 * (i % 2), line_width=0, row="all", col=1)
        fig.add_annotation(x=(lo + hi) / 2, y=n_actors - 0.3 if i % 2 == 0 else -1.4,
                           text=label.split(" · ")[1], showarrow=False,
                           font=dict(size=10, color="#8a3b1e"),
                           bgcolor="rgba(255,255,255,0.6)", row=1, col=1)
    for i, (year, actor, label) in enumerate(MILESTONES):
        fig.add_annotation(x=year, y=actor, text="★ " + label, showarrow=True,
                           arrowhead=2, ax=14, ay=-26 - 20 * (i % 3), font=dict(size=9),
                           row=1, col=1)
    maxy = int(df.year.max())
    fig.add_annotation(x=maxy, y=1.0, yref="paper", text=f"{maxy} = partial (H1)", showarrow=False,
                       font=dict(size=10, color="grey"), xanchor="center", row=1, col=1)

    fig.update_layout(height=1380, width=1520, plot_bgcolor="white",
                      legend_title="track (ordered by first appearance)",
                      legend=dict(font=dict(size=10)), margin=dict(t=110, b=50),
                      title="AI-safety genealogy 2005-2026 — organisations & tracks over time "
                            "(colour = track age; bottom panel = activity-mix shift)")
    save(fig, "timeline")


def chart_track_lifespan(df, pubs):
    """Per-track lifespan dumbbell: birth (first event) → last recorded event, ordered by
    birth year. A hollow diamond marks the last year the track still has arXiv publications
    (independent proxy) — so "old track, still alive" reads at a glance even when discrete
    events stop early (e.g. reward_modeling: last event 2017, but arXiv still 2025).

    HONESTY: the bar end is the last RECORDED event = collection density (caveat #2), NOT a
    death date. The arXiv diamond is a separate, independent signal (7 tracks only)."""
    track_order, cmap = _track_birth_order(df)
    cmap = _with_pub_colors(cmap, pubs)
    exclude = set(AGGREGATE_TRACKS) | {"capabilities"}
    tracks = [t for t in track_order if t not in exclude]
    maxy = int(df.year.max())
    fig = go.Figure()
    for t in tracks:
        sub = df[df.direction == t]
        y0, y1 = int(sub.year.min()), int(sub.year.max())
        fig.add_trace(go.Scatter(x=[y0, y1], y=[t, t], mode="lines",
                                 line=dict(color=cmap[t], width=4), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[y0, y1], y=[t, t], mode="markers",
            marker=dict(color=cmap[t], size=[9, 14], line=dict(width=0.5, color="white")),
            showlegend=False,
            customdata=[["born", y0], ["last recorded event", y1]],
            hovertemplate="%{y}: %{customdata[0]} %{customdata[1]}<extra></extra>"))
        if t in set(pubs.track):
            # "last FULL year" -> exclude the partial year so the diamond reflects a complete count
            active = pubs[(pubs.track == t) & (pubs.arxiv_count > 0) & (pubs.year < PARTIAL_YEAR)]
            if len(active):
                last_pub = int(active.year.max())
                fig.add_trace(go.Scatter(
                    x=[last_pub], y=[t], mode="markers",
                    marker=dict(symbol="diamond", size=12, color="white",
                                line=dict(color=cmap[t], width=2)),
                    showlegend=False,
                    hovertemplate=t + ": arXiv still publishing in " + str(last_pub) +
                                  " (independent proxy)<extra></extra>"))
    fig.add_vline(x=maxy, line=dict(color="grey", dash="dot", width=1))
    fig.add_annotation(x=maxy, y=1.02, yref="paper", text=f"latest data year ({maxy} = partial H1)",
                       showarrow=False, font=dict(size=10, color="grey"), xanchor="right")
    fig.add_annotation(
        x=0, y=-0.11, xref="paper", yref="paper", showarrow=False, align="left",
        font=dict(size=10, color=DARK_MUTED),
        text="● line = birth → last recorded event (collection density, caveat) · "
             f"◆ = last FULL year with arXiv publications (independent proxy; {PARTIAL_YEAR} excluded — partial year) · "
             "absence of late events ≠ the track died")
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(tracks)))
    fig.update_xaxes(dtick=1, gridcolor="#eee", range=[2004, 2027])
    fig.update_layout(height=720, width=1250, plot_bgcolor="white", margin=dict(b=110),
                      title="Track lifespans — birth → last recorded event, ordered by birth year "
                            "(◆ = still publishing on arXiv; old ≠ dead)")
    save(fig, "track_lifespan")


def _write_zoom_html(fig, name):
    """Standalone page with in-page zoom (+/- , reset, Ctrl+wheel) for a non-zoomable
    trace like parcats: resize via Plotly.relayout (keeps hover correct) inside a scroller."""
    W, H = int(fig.layout.width), int(fig.layout.height)
    inner = fig.to_html(full_html=False, include_plotlyjs="cdn")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<style>
 html,body{{margin:0;background:{DARK_BG};color:{DARK_FG};font-family:-apple-system,Segoe UI,Roboto,sans-serif;}}
 #zoomwrap{{position:fixed;inset:0;overflow:auto;}}
 #zoombar{{position:fixed;top:12px;right:16px;z-index:10;display:flex;gap:6px;}}
 #zoombar button{{width:36px;height:36px;font-size:18px;cursor:pointer;background:{DARK_PANEL};
   color:{DARK_FG};border:1px solid {DARK_LINE};border-radius:8px;}}
 #zoomhint{{position:fixed;bottom:12px;left:16px;z-index:10;color:{DARK_MUTED};font-size:12px;}}
</style></head><body>
<div id="zoombar"><button id="zout">\u2212</button><button id="zrst">\u25a1</button><button id="zin">+</button></div>
<div id="zoomhint">zoom: +/\u2212 buttons or Ctrl + scroll \u00b7 drag to pan</div>
<div id="zoomwrap">{inner}</div>
<script>
addEventListener('load', () => {{
  const gd = document.querySelector('.plotly-graph-div'), wrap = document.getElementById('zoomwrap');
  const W = {W}, H = {H}; let s;
  const apply = () => Plotly.relayout(gd, {{width: W*s, height: H*s}});
  const fit = () => {{ s = Math.min(1, wrap.clientWidth / W); apply(); }};
  fit();
  document.getElementById('zin').onclick  = () => {{ s = Math.min(s*1.25, 3); apply(); }};
  document.getElementById('zout').onclick = () => {{ s = Math.max(s/1.25, 0.1); apply(); }};
  document.getElementById('zrst').onclick = fit;
  wrap.addEventListener('wheel', e => {{ if(!e.ctrlKey) return; e.preventDefault();
    s = Math.min(Math.max(s*(e.deltaY<0?1.1:0.9), 0.1), 3); apply(); }}, {{passive:false}});
  let px, py, sxp, syp, pan = false;
  wrap.style.cursor = 'grab';
  wrap.addEventListener('mousedown', e => {{ pan = true; wrap.style.cursor = 'grabbing';
    sxp = e.clientX; syp = e.clientY; px = wrap.scrollLeft; py = wrap.scrollTop; e.preventDefault(); }});
  addEventListener('mousemove', e => {{ if (!pan) return;
    wrap.scrollLeft = px - (e.clientX - sxp); wrap.scrollTop = py - (e.clientY - syp); }});
  addEventListener('mouseup', () => {{ pan = false; wrap.style.cursor = 'grab'; }});
}});
</script></body></html>"""
    open(f"{VIZ_DIR}/{name}.html", "w", encoding="utf-8").write(html)


def chart_alluvial(df):
    """Organisational parallel-sets: era -> organisation -> track (non-money events only).

    Money rows (grant_out/funding_total) are excluded because their actor is the FUNDER,
    not the organisation doing the work — including them would mislabel funders as orgs.
    """
    d = df[~df.event_type.isin(MONEY_TYPES)].copy()
    d["era"] = d["year"].map(era_of)
    era_labels = [label for label, _, _ in ERAS if label in set(d["era"])]
    actor_order = d.groupby("actor")["year"].min().sort_values().index.tolist()
    d["track_code"] = pd.factorize(d["direction"])[0]
    dims = [
        go.parcats.Dimension(values=d["era"], label="era",
                             categoryorder="array", categoryarray=era_labels),
        go.parcats.Dimension(values=d["actor"], label="organisation",
                             categoryorder="array", categoryarray=actor_order),
        go.parcats.Dimension(values=d["direction"], label="track"),
    ]
    fig = go.Figure(go.Parcats(
        dimensions=dims,
        line={"color": d["track_code"], "colorscale": "Turbo", "shape": "hspline"},
        hoveron="color", hoverinfo="count+probability", arrangement="freeform",
        labelfont=dict(size=15), tickfont=dict(size=11),
    ))
    # ~90 organisation nodes stack in the middle column, so their labels need vertical room;
    # scale BOTH axes together (keep proportions) so text stops overlapping and the ribbons
    # widen with the canvas instead of turning into a thin tall strip.
    fig.update_layout(
        title="Organisational alluvial — era → organisation → track "
              "(who worked on what, when; excludes money rows)",
        height=4400, width=6400, margin=dict(l=210, r=150, t=100, b=40),
    )
    save(fig, "alluvial_org")
    _write_zoom_html(fig, "alluvial_org")   # overwrite standalone page with in-page zoom controls
    # compact gallery preview: same parcats, autosized to fit its box; the full-size
    # alluvial_org.html stays big & scrollable for the "open full chart" link
    fig.update_layout(width=None, height=720, autosize=True, title_text="")
    fig.write_html(f"{VIZ_DIR}/alluvial_org_preview.html", include_plotlyjs="cdn",
                   default_width="100%", config={"responsive": True})


# --- compact/readable alluvial variants (candidates to replace the spaghetti one) -----------

def _hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _rgb_to_rgba(rgb_color, alpha):
    """Add an alpha to a Plotly colorscale sample ('rgb(r, g, b)' string, as from
    sample_colorscale on TRACK_COLORSCALE). Fails loudly on any non-'rgb(...)' input."""
    r, g, b = (int(float(x)) for x in rgb_color[rgb_color.index("(") + 1:rgb_color.index(")")].split(","))
    return f"rgba({r},{g},{b},{alpha})"


def _family_colorscale():
    """Stepped continuous colorscale so a factorised family index (0..n-1) maps to its EXACT
    family colour. parcats needs a continuous scale, so we fake discrete bands centred on each
    normalised code position."""
    colors = [FAMILY_COLORS[f] for f in FAMILY_ORDER]
    n = len(colors)
    pos = [k / (n - 1) for k in range(n)]
    edges = [0.0] + [(pos[i] + pos[i + 1]) / 2 for i in range(n - 1)] + [1.0]
    scale = []
    for i, c in enumerate(colors):
        scale += [[edges[i], c], [edges[i + 1], c]]
    return scale


def _grouped_events(df):
    """Non-money events with readable era / actor-type / track-family columns.
    Fails loudly if any slug is missing from the RU / family maps (no silent drop)."""
    d = df[~df.event_type.isin(MONEY_TYPES)].copy()
    missing_t = sorted(set(d["direction"]) - set(TRACK_FAMILY))
    assert not missing_t, f"tracks missing from TRACK_FAMILY: {missing_t}"
    missing_a = sorted(set(d["actor_type"]) - set(ACTOR_TYPE_RU))
    assert not missing_a, f"actor_types missing from ACTOR_TYPE_RU: {missing_a}"
    d["era_ru"] = d["year"].map(era_of).map(ERA_RU)
    d["actor_ru"] = d["actor_type"].map(ACTOR_TYPE_RU)
    d["family"] = d["direction"].map(TRACK_FAMILY)
    return d


def chart_alluvial_grouped(df):
    """Variant A — compact parcats: era → actor-type → track-family. Both spaghetti axes are
    collapsed into buckets, so ribbons become wide and legible. Colour = track family."""
    d = _grouped_events(df)
    era_order = [ERA_RU[l] for l, _, _ in ERAS if ERA_RU[l] in set(d["era_ru"])]
    at_order = d["actor_ru"].value_counts().index.tolist()
    fam_order = [f for f in FAMILY_ORDER if f in set(d["family"])]
    fam_code = d["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)})
    dims = [
        go.parcats.Dimension(values=d["era_ru"], label="era",
                             categoryorder="array", categoryarray=era_order),
        go.parcats.Dimension(values=d["actor_ru"], label="actor type",
                             categoryorder="array", categoryarray=at_order),
        go.parcats.Dimension(values=d["family"], label="track family",
                             categoryorder="array", categoryarray=fam_order),
    ]
    fig = go.Figure(go.Parcats(
        dimensions=dims,
        line={"color": fam_code, "colorscale": _family_colorscale(), "shape": "hspline",
              "cmin": 0, "cmax": len(FAMILY_ORDER) - 1},
        hoveron="color", hoverinfo="count+probability", arrangement="freeform",
    ))
    fig.update_layout(
        title="Variant A — era → actor type → track family<br>"
              "<sub>non-money events: ~115 orgs collapsed to 7 actor types, 17 tracks to 4 families</sub>",
        height=650, width=1150, margin=dict(l=150, r=170, t=95, b=40),
    )
    save(fig, "alluvial_grouped")


def chart_alluvial_by_era(df):
    """Variant B — 6 mini-Sankeys (one per era): actor-type → track-family inside each era.
    Splitting by era keeps every panel tiny and readable. Non-money events; ribbon = event count."""
    d = _grouped_events(df)
    eras = [l for l, _, _ in ERAS if ERA_RU[l] in set(d["era_ru"])]
    fig = go.Figure()
    for i, era_label in enumerate(eras):
        sub = d[d["era_ru"] == ERA_RU[era_label]]
        at_nodes = sub["actor_ru"].value_counts().index.tolist()
        fam_nodes = [f for f in FAMILY_ORDER if f in set(sub["family"])]
        labels = at_nodes + fam_nodes
        node_colors = ["#9aa0a6"] * len(at_nodes) + [FAMILY_COLORS[f] for f in fam_nodes]
        idx = {("A", n): k for k, n in enumerate(at_nodes)}
        idx.update({("F", n): len(at_nodes) + k for k, n in enumerate(fam_nodes)})
        g = sub.groupby(["actor_ru", "family"]).size().reset_index(name="n")
        src = [idx[("A", r.actor_ru)] for r in g.itertuples()]
        tgt = [idx[("F", r.family)] for r in g.itertuples()]
        val = [r.n for r in g.itertuples()]
        lcol = [_hex_to_rgba(FAMILY_COLORS[r.family], 0.55) for r in g.itertuples()]
        row, col = i // 3, i % 3
        x0, x1 = col / 3 + 0.012, (col + 1) / 3 - 0.012
        y0 = 1 - (row + 1) / 2 + 0.06
        y1 = 1 - row / 2 - 0.03
        fig.add_trace(go.Sankey(
            domain=dict(x=[x0, x1], y=[y0, y1]),
            node=dict(label=labels, color=node_colors, pad=9, thickness=11, line=dict(width=0)),
            link=dict(source=src, target=tgt, value=val, color=lcol,
                      hovertemplate="%{source.label} → %{target.label}: %{value}<extra></extra>"),
        ))
        fig.add_annotation(x=(x0 + x1) / 2, y=y1 + 0.025, xref="paper", yref="paper",
                           text=f"<b>{ERA_RU[era_label]}</b> · n={len(sub)}",
                           showarrow=False, font=dict(size=11))
    fig.update_layout(
        title="Variant B — by era: actor type → track family<br>"
              "<sub>one panel per era; non-money events; ribbon width = event count</sub>",
        height=820, width=1250, margin=dict(l=20, r=20, t=95, b=20),
    )
    save(fig, "alluvial_by_era")


def chart_alluvial_pairs(df):
    """Variant C — two clean 2-level Sankeys side by side:
       left  = era → track-family (how the field's focus shifts over time);
       right = actor-type → track-family (who works on what). Non-money events."""
    d = _grouped_events(df)

    def flow(left_col, left_order):
        left_nodes = [x for x in left_order if x in set(d[left_col])]
        fam_nodes = [f for f in FAMILY_ORDER if f in set(d["family"])]
        labels = left_nodes + fam_nodes
        node_colors = ["#9aa0a6"] * len(left_nodes) + [FAMILY_COLORS[f] for f in fam_nodes]
        idx = {("L", n): k for k, n in enumerate(left_nodes)}
        idx.update({("F", n): len(left_nodes) + k for k, n in enumerate(fam_nodes)})
        g = d.groupby([left_col, "family"]).size().reset_index(name="n")
        src = [idx[("L", getattr(r, left_col))] for r in g.itertuples()]
        tgt = [idx[("F", r.family)] for r in g.itertuples()]
        val = [r.n for r in g.itertuples()]
        lcol = [_hex_to_rgba(FAMILY_COLORS[r.family], 0.55) for r in g.itertuples()]
        return labels, node_colors, src, tgt, val, lcol

    era_order = [ERA_RU[l] for l, _, _ in ERAS]
    at_order = d["actor_ru"].value_counts().index.tolist()
    fig = go.Figure()
    for title, lc, lo, dom in [
        ("era → family", "era_ru", era_order, [0.0, 0.46]),
        ("actor type → family", "actor_ru", at_order, [0.54, 1.0]),
    ]:
        labels, node_colors, src, tgt, val, lcol = flow(lc, lo)
        fig.add_trace(go.Sankey(
            domain=dict(x=dom, y=[0.0, 0.88]),
            node=dict(label=labels, color=node_colors, pad=16, thickness=15, line=dict(width=0)),
            link=dict(source=src, target=tgt, value=val, color=lcol,
                      hovertemplate="%{source.label} → %{target.label}: %{value}<extra></extra>"),
        ))
        fig.add_annotation(x=(dom[0] + dom[1]) / 2, y=0.99, xref="paper", yref="paper",
                           text=f"<b>{title}</b>", showarrow=False, font=dict(size=13))
    fig.update_layout(
        title="Variant C — two cuts: era→family and actor type→family<br>"
              "<sub>non-money events; ribbon width = event count</sub>",
        height=640, width=1250, margin=dict(l=20, r=20, t=115, b=20),
    )
    save(fig, "alluvial_pairs")


# --- cross-era shift: one comparison figure, era on the X axis ------------------------------

def chart_era_shift(df):
    """How the field migrates chapter to chapter, era on the X axis (5 NARRATIVE_ERAS). One 2x2:
      1) direction families — 100%-stacked (mix shift; n=events per era annotated on top);
      2) actor types — 100%-stacked (who is active);
      3) money by funder — absolute stacked $ (the money explosion; early eras honestly a sliver);
      4) org dynamics — diverging bars (founded up, closed+pivot down).
    Panels 1,2,4 = non-money events (collection density, caveat #2); panel 3 = grant_out only.
    Never empty: every era is a bar, so the sparse early era reads as 'start', not a blank panel."""
    d = df[~df.event_type.isin(MONEY_TYPES)].copy()
    missing_t = sorted(set(d["direction"]) - set(TRACK_FAMILY))
    assert not missing_t, f"tracks missing from TRACK_FAMILY: {missing_t}"
    missing_a = sorted(set(d["actor_type"]) - set(ACTOR_TYPE_RU))
    assert not missing_a, f"actor_types missing from ACTOR_TYPE_RU: {missing_a}"
    d["family"] = d["direction"].map(TRACK_FAMILY)
    d["actor_ru"] = d["actor_type"].map(ACTOR_TYPE_RU)
    g = df[(df.event_type == "grant_out") & df.scale.notna()].copy()
    g["scale"] = g["scale"].astype(float)
    g["funder"] = g["actor"].map(ERA_FUNDER_LABELS)

    ticks = [t for _cid, t, _lo, _hi in NARRATIVE_ERAS]
    ranges = [(t, lo, hi) for _cid, t, lo, hi in NARRATIVE_ERAS]
    era_events = {t: len(d[(d.year >= lo) & (d.year <= hi)]) for t, lo, hi in ranges}
    actor_order = list(ACTOR_TYPE_RU.values())

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Track families (share of era events)",
                        "Who is active — actor types (share of era events)",
                        "Era money by funder, $ (all itemized grants; grey = gov &amp; new)",
                        "Org dynamics: births (up) vs closures/pivots (down)"),
        vertical_spacing=0.16, horizontal_spacing=0.11,
    )

    def era_share(col_series, key):
        n = era_events[key]
        return (col_series / n * 100.0) if n else 0.0

    # panel 1: families, 100%-stacked
    for fam in FAMILY_ORDER:
        ys = []
        for t, lo, hi in ranges:
            e = d[(d.year >= lo) & (d.year <= hi)]
            ys.append(era_share((e["family"] == fam).sum(), t))
        fig.add_trace(go.Bar(
            x=ticks, y=ys, name=fam, legendgroup="fam", legendgrouptitle_text="Families",
            marker_color=FAMILY_COLORS[fam],
            text=[f"{v:.0f}%" if v >= 9 else "" for v in ys], textposition="inside",
            insidetextfont=dict(color="white", size=11),
            hovertemplate=f"{fam}<br>%{{x}}: %{{y:.0f}}%<extra></extra>",
        ), row=1, col=1)

    # panel 2: actor types, 100%-stacked
    for actor in actor_order:
        ys = []
        for t, lo, hi in ranges:
            e = d[(d.year >= lo) & (d.year <= hi)]
            ys.append(era_share((e["actor_ru"] == actor).sum(), t))
        fig.add_trace(go.Bar(
            x=ticks, y=ys, name=actor, legendgroup="actor", legendgrouptitle_text="Actors",
            marker_color=ACTOR_TYPE_COLORS[actor],
            text=[f"{v:.0f}%" if v >= 10 else "" for v in ys], textposition="inside",
            insidetextfont=dict(color="white", size=11),
            hovertemplate=f"{actor}<br>%{{x}}: %{{y:.0f}}%<extra></extra>",
        ), row=1, col=2)

    # panel 3: money by funder, absolute stacked $ — the 6 named funders keep fixed colours,
    # everything else (govt/international institutes + new philanthropy) rolls into ONE grey band
    # plotted last (on top) so the stack totals ALL itemized grant $ for the era.
    for funder in ERA_FUNDER_LABELS.values():
        ys = []
        for t, lo, hi in ranges:
            ge = g[(g.year >= lo) & (g.year <= hi)]
            ys.append(float(ge.loc[ge["funder"] == funder, "scale"].sum()))
        fig.add_trace(go.Bar(
            x=ticks, y=ys, name=funder, legendgroup="fund", legendgrouptitle_text="Funders",
            marker_color=ERA_FUNDER_COLORS[funder],
            hovertemplate=f"{funder}<br>%{{x}}: $%{{y:,.0f}}<extra></extra>",
        ), row=2, col=1)
    other_ys = []
    for t, lo, hi in ranges:
        ge = g[(g.year >= lo) & (g.year <= hi)]
        other_ys.append(float(ge.loc[ge["funder"].isna(), "scale"].sum()))
    fig.add_trace(go.Bar(
        x=ticks, y=other_ys, name=ERA_OTHER_LABEL, legendgroup="fund",
        marker_color=ERA_OTHER_COLOR,
        hovertemplate=f"{ERA_OTHER_LABEL}<br>%{{x}}: $%{{y:,.0f}}<extra></extra>",
    ), row=2, col=1)

    # panel 4: org dynamics, diverging (founded up, closed+pivot down)
    founded = [int(((d.event_type == "founded") & (d.year >= lo) & (d.year <= hi)).sum())
               for _t, lo, hi in ranges]
    closed = [int((d.event_type.isin(["closed", "pivot"]) & (d.year >= lo) & (d.year <= hi)).sum())
              for _t, lo, hi in ranges]
    fig.add_trace(go.Bar(
        x=ticks, y=founded, name="new orgs", legendgroup="org", legendgrouptitle_text="Orgs",
        marker_color="#4C78A8", hovertemplate="new: %{y}<extra></extra>"), row=2, col=2)
    fig.add_trace(go.Bar(
        x=ticks, y=[-c for c in closed], name="closures/pivots", legendgroup="org",
        marker_color="#E45756", customdata=closed,
        hovertemplate="closures/pivots: %{customdata}<extra></extra>"), row=2, col=2)
    # explicit value labels (Plotly drops textposition='outside' for stacked bars)
    for t, f, c in zip(ticks, founded, closed):
        fig.add_annotation(x=t, y=f, xref="x4", yref="y4", yshift=8, text=str(f),
                           showarrow=False, font=dict(size=11, color="#4C78A8"))
        if c:
            fig.add_annotation(x=t, y=-c, xref="x4", yref="y4", yshift=-8, text=str(c),
                               showarrow=False, font=dict(size=11, color="#E45756"))

    # families/actors stack to 100%; money stacks absolute; orgs diverge around 0
    # (relative == stack for positive-only panels, but pushes negatives below zero)
    fig.update_layout(barmode="relative")
    fig.update_traces(row=2, col=2, width=0.55)

    # n=events per era above the family panel (normalisation hides the ~10x growth otherwise)
    for t in ticks:
        fig.add_annotation(x=t, y=106, xref="x1", yref="y1", yanchor="bottom",
                           text=f"n={era_events[t]}", showarrow=False, font=dict(size=10, color=DARK_MUTED))

    fig.update_yaxes(range=[0, 116], tickvals=list(range(0, 101, 20)), ticksuffix="%", row=1, col=1)
    fig.update_yaxes(range=[0, 100], tickvals=list(range(0, 101, 20)), ticksuffix="%", row=1, col=2)
    fig.update_yaxes(title_text="$ individual grants", row=2, col=1)
    fig.update_yaxes(title_text="orgs", range=[-max(closed) - 4, max(founded) + 6],
                     row=2, col=2, zeroline=True, zerolinecolor="#888")
    fig.update_layout(
        title="How the field shifted across eras — families, actors, money, org dynamics<br>"
              "<sub>era on X; panels 1–2 normalized to era events (n annotated on top); "
              "money = ALL itemized grants (6 named funders + grey gov/new band; donor totals excluded); "
              "events ≠ real-world activity (caveat 2)</sub>",
        height=850, width=1300, plot_bgcolor="white", bargap=0.28,
        legend=dict(groupclick="toggleitem", tracegroupgap=14, font=dict(size=11)),
        margin=dict(l=70, r=210, t=110, b=50),
    )
    save(fig, "era_shift")


def chart_funding_sankey(df):
    """2-level Sankey: funder -> track, on known individual grants (grant_out)."""
    g = df[(df.event_type == "grant_out") & df.scale.notna()].copy()
    g["scale"] = g["scale"].astype(float)
    nodes = pd.unique(g[["actor", "direction"]].values.ravel()).tolist()
    idx = {n: i for i, n in enumerate(nodes)}
    _, cmap = _track_birth_order(df)
    node_colors = [cmap[n] if n in cmap else "#9aa0a6" for n in nodes]
    link_colors = [_rgb_to_rgba(cmap[d], 0.5) for d in g["direction"]]
    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, pad=18, thickness=16, color=node_colors),
        link=dict(source=g["actor"].map(idx), target=g["direction"].map(idx),
                  value=g["scale"], customdata=g["detail"], color=link_colors,
                  hovertemplate="%{customdata}<br>$%{value:,.0f}<extra></extra>"),
    ))
    fig.update_layout(title="Funding flows — funder → track (known individual grants)",
                      height=750, width=1250)
    save(fig, "funding_sankey")


def chart_funding_sankey_era(df):
    """3-level Sankey: era -> funder -> track, on known individual grants.

    We route through era (not recipient-organisation) because the recipient org is only
    present as free text in `detail`, not a structured column — parsing it would risk
    fabricated attributions. Era is a clean structural third level.
    """
    g = df[(df.event_type == "grant_out") & df.scale.notna()].copy()
    g["scale"] = g["scale"].astype(float)
    g["era"] = g["year"].map(era_of)
    eras = [label for label, _, _ in ERAS if label in set(g["era"])]
    funders = sorted(g["actor"].unique().tolist())
    tracks = sorted(g["direction"].unique().tolist())
    nodes = eras + funders + tracks
    idx = {n: i for i, n in enumerate(nodes)}

    ef = g.groupby(["era", "actor"], as_index=False)["scale"].sum()
    ft = g.groupby(["actor", "direction"], as_index=False)["scale"].sum()
    src = ef["era"].map(idx).tolist() + ft["actor"].map(idx).tolist()
    tgt = ef["actor"].map(idx).tolist() + ft["direction"].map(idx).tolist()
    val = ef["scale"].tolist() + ft["scale"].tolist()

    _, cmap = _track_birth_order(df)
    node_colors = [cmap[n] if n in cmap else "#9aa0a6" for n in nodes]
    link_colors = (["rgba(154,160,166,0.35)"] * len(ef)
                   + [_rgb_to_rgba(cmap[d], 0.5) for d in ft["direction"]])

    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, pad=16, thickness=15, color=node_colors),
        link=dict(source=src, target=tgt, value=val, color=link_colors,
                  hovertemplate="$%{value:,.0f}<extra></extra>"),
    ))
    fig.update_layout(title="Funding routing — era → funder → track (known individual grants)",
                      height=850, width=1300)
    save(fig, "funding_sankey_era")


def chart_track_emergence(df):
    """Area of raw event counts per track per year — with the incomplete final (partial)
    year explicitly shaded so its H1-only undercount is not read as a real contraction.

    2024-2025 were re-collected from live sources (org-founding wave, new governance
    institutions, 2025 milestones), so the earlier 2024-25 dip is gone. Only PARTIAL_YEAR
    remains a genuine artefact (mid-year, H1 only). Money and publication curves stay the
    honest activity signals (event count = collection density, caveat #2).
    """
    _, cmap = _track_birth_order(df)
    c = df.groupby(["year", "direction"]).size().reset_index(name="n")
    fig = px.area(c, x="year", y="n", color="direction", color_discrete_map=cmap,
                  title="Track activity by event count/yr — event count = COLLECTION DENSITY, not real activity")
    ymax = c.groupby("year")["n"].sum().max()
    fig.add_vrect(x0=PARTIAL_YEAR - 0.5, x1=PARTIAL_YEAR + 0.5, fillcolor="gray", opacity=0.18,
                  line_width=0,
                  annotation_text=f"{PARTIAL_YEAR} = partial year (H1 only)",
                  annotation_position="top left", annotation_font_size=11)
    fig.update_layout(height=700, width=1300, xaxis=dict(dtick=1, range=YEAR_AXIS_RANGE),
                      plot_bgcolor="white", yaxis=dict(range=[0, ymax * 1.05]))
    save(fig, "track_emergence")


def chart_money_vs_attention(df, pubs):
    """Field-level dual axis: known yearly money vs arXiv attention proxy.

    Money = sum of funding_total[mixed_technical] per year (OpenPhil + LTFF overall totals;
    the only clean non-double-counted yearly money series). Attention has TWO honest curves:
      * `_safety_corpus` — the DEDUPLICATED unique-safety-paper series (one combined arXiv OR-query
        → each paper counted once). This is the primary, honest field line.
      * the disjoint-core KEYWORD-PROXY SUM (per-track counts minus FIELD_SUM_EXCLUDE overlaps and
        minus service tracks) — shown as a lighter dashed line to expose how much the per-track
        proxies INFLATE the count vs the dedup corpus.
    Per-*track* dual axis is impossible: the tracks with pub curves have no track-level
    funding_total, and vice versa.
    """
    # canonical annual donor total per (actor, year): the mixed_technical umbrella row where it
    # exists, else that funder-year's full funding_total sum (same de-dup as cumulative_funding,
    # so sub-category rows are never added on top of the umbrella). This extends the series past
    # 2023: OpenPhil 2024 has only a technical_safety row ($28M), so it now shows as a 2024 bar.
    ft = df[(df.event_type == "funding_total") & df.scale.notna()].copy()
    mt = ft[ft.direction == "mixed_technical"].groupby(["actor", "year"])["scale"].sum()
    anyft = ft.groupby(["actor", "year"])["scale"].sum()
    money = mt.reindex(anyft.index).fillna(anyft).groupby("year").sum()
    corpus = pubs[pubs.track == "_safety_corpus"].groupby("year")["arxiv_count"].sum()
    # keyword-proxy sum: drop overlap-excluded tracks AND every service track (leading underscore)
    core = pubs[~pubs.track.isin(FIELD_SUM_EXCLUDE) & ~pubs.track.str.startswith("_")]
    attn = core.groupby("year")["arxiv_count"].sum()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=money.index, y=money.values, name="money ($, OpenPhil+LTFF totals)",
                         marker_color="#4C78A8", opacity=0.7), secondary_y=False)
    fig.add_trace(go.Scatter(x=corpus.index, y=corpus.values,
                             name="attention — dedup safety corpus (unique papers/yr)",
                             mode="lines+markers", line=dict(color="#E45756", width=3)),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=attn.index, y=attn.values,
                             name="keyword-proxy sum (inflated, double-counts overlaps)",
                             mode="lines", line=dict(color="#E45756", width=1.5, dash="dot"),
                             opacity=0.6), secondary_y=True)
    fig.update_layout(title="Money vs scientific attention (field level; both are proxies)<br>"
                            "<sub>bars = OpenPhil+LTFF published donor totals -> end at 2024 (2025+ totals not "
                            "published yet; itemized grants shown on other charts)<br>"
                            "solid = dedup safety corpus, dotted = inflated keyword sum, to "
                            f"{PARTIAL_YEAR}; arXiv axis = LOG</sub>",
                      height=690, width=1150, plot_bgcolor="white", margin=dict(t=130),
                      legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                                  bgcolor="rgba(255,255,255,0.6)"),
                      xaxis=dict(dtick=1, range=YEAR_AXIS_RANGE))
    fig.update_yaxes(title_text="USD / year", secondary_y=False)
    fig.update_yaxes(title_text="arXiv papers / year (log)", type="log", secondary_y=True)
    if PARTIAL_YEAR in set(corpus.index):
        fig.add_vrect(x0=PARTIAL_YEAR - 0.5, x1=PARTIAL_YEAR + 0.5, fillcolor="gray",
                      opacity=0.12, line_width=0,
                      annotation_text=f"{PARTIAL_YEAR} partial (H1)",
                      annotation_position="top left", annotation_font_size=10)
    save(fig, "money_vs_attention")


# ------------------------------------------------------ NEW: money & lifecycle

# aggregate donor-total buckets + the single non-safety grant, excluded from per-TRACK money
NONTRACK_MONEY = AGGREGATE_TRACKS + ["capabilities"]


def _track_money(df):
    """Track-classified money from known individual grants (grant_out) — one row per grant.

    Excludes aggregate donor-total buckets (mixed_technical / technical_safety) and the lone
    non-safety 'capabilities' grant. This is a DIFFERENT lens from funding_total donor totals
    (chart_cumulative_funding's OpenPhil/LTFF lines); the two are never summed together, so
    nothing double-counts.
    """
    g = df[(df.event_type == "grant_out") & df.scale.notna()
           & ~df.direction.isin(NONTRACK_MONEY)].copy()
    g["scale"] = g["scale"].astype(float)
    return g


def chart_org_lifespan(df):
    """Gantt of organisation lifespans: founding → closure/pivot (or ongoing to latest year).

    Start = first 'founded' event; end = last 'closed'/'pivot' event if any, else latest data
    year (marked 'ongoing'). Any org that ever closed/pivoted MUST appear: if it has no
    'founded' event we fall back to its first-seen year, so no closure/pivot org can silently
    disappear (this is why births_deaths and org_lifespan reconcile).
    """
    founded = df[df.event_type == "founded"].groupby("actor")["year"].min()
    ends = (df[df.event_type.isin(["closed", "pivot"])].sort_values("year")
            .groupby("actor").agg(end_year=("year", "max"), end_type=("event_type", "last"),
                                  end_detail=("detail", "last")))
    # start year per org: real founding where known, else first-seen year for closure/pivot orgs
    first_seen = df.groupby("actor")["year"].min()
    starts = {actor: int(fy) for actor, fy in founded.items()}
    fallback = set()
    for actor in ends.index:
        if actor not in starts:
            starts[actor] = int(first_seen[actor])
            fallback.add(actor)
    maxyear = int(df.year.max())
    rows = []
    for actor, fy in starts.items():
        atype = df[df.actor == actor]["actor_type"].iloc[0]
        if actor in ends.index and int(ends.loc[actor, "end_year"]) >= int(fy):
            ey, et, det = int(ends.loc[actor, "end_year"]), ends.loc[actor, "end_type"], ends.loc[actor, "end_detail"]
        else:
            ey, et, det = maxyear, "ongoing", "no closure/pivot event in data"
        start_note = " [start = first-seen year; no founding event in data]" if actor in fallback else ""
        rows.append(dict(actor=actor, start=pd.Timestamp(int(fy), 1, 1),
                         end=pd.Timestamp(ey if ey > fy else fy, 12, 31),
                         actor_type=atype, end_type=et, detail=det + start_note))
    L = pd.DataFrame(rows).sort_values("start")
    fig = px.timeline(L, x_start="start", x_end="end", y="actor", color="actor_type",
                      hover_data=["end_type", "detail"],
                      title="Organisation lifespans — founding → closure/pivot (or ongoing); from events.csv")
    fig.update_yaxes(autorange="reversed")
    marks = L[L.end_type.isin(["closed", "pivot"])]
    fig.add_trace(go.Scatter(x=marks["end"], y=marks["actor"], mode="markers+text",
                             marker=dict(symbol="x", size=10, color=DARK_FG),
                             text=marks["end_type"], textposition="middle right",
                             showlegend=False, hoverinfo="skip"))
    # one row per org (71+); give each ~26px so EVERY label renders (Plotly silently thins tick
    # labels when rows are cramped — that hid orgs like FHI even though their bars were drawn).
    fig.update_yaxes(tickfont=dict(size=10))
    fig.update_layout(height=max(1050, 26 * len(L)), width=1450, plot_bgcolor="white")
    save(fig, "org_lifespan")


def chart_births_deaths(df):
    """Org births (founded) up vs closures+pivots down, per year."""
    births = df[df.event_type == "founded"].groupby("year").size()
    deaths = df[df.event_type.isin(["closed", "pivot"])].groupby("year").size()
    years = list(range(int(df.year.min()), int(df.year.max()) + 1))
    fig = go.Figure()
    fig.add_bar(x=years, y=[int(births.get(y, 0)) for y in years], name="births (founded)", marker_color="#4C78A8")
    fig.add_bar(x=years, y=[-int(deaths.get(y, 0)) for y in years], name="closures + pivots", marker_color="#E45756")
    fig.update_layout(barmode="relative", height=600, width=1250,
                      xaxis=dict(dtick=1, range=YEAR_AXIS_RANGE),
                      yaxis_title="# orgs  (down = closed / pivoted)", plot_bgcolor="white",
                      title="Organisational births vs closures/pivots per year (from events.csv)")
    save(fig, "births_deaths")


def _spread_labels(true_ys, lo, hi, gap):
    """Two-pass greedy label de-collision: keep each label near its true y but at least `gap`
    apart, clamped to [lo, hi]. Returns adjusted y in the SAME order as `true_ys`."""
    order = sorted(range(len(true_ys)), key=lambda i: true_ys[i])
    ys = [None] * len(true_ys)
    prev = lo - gap
    for i in order:                       # bottom-up: push up when too close to the one below
        y = max(true_ys[i], prev + gap)
        ys[i] = y
        prev = y
    nxt = hi + gap
    for i in reversed(order):             # top-down: pull back down if the stack overran `hi`
        y = min(ys[i], nxt - gap)
        ys[i] = y
        nxt = y
    return ys


# --- funder-type colours + canonical per-funder money (shared by the 3 grant charts) ----------
# One colour per funder TYPE (not per funder), so the three grant views read consistently. VC is
# its own colour because it is a SEPARATE lens (equity, never summed with grants).
TYPE_COLOR = {"philanthropy": "#4C78A8", "government": "#F58518",
              "corporate": "#E45756", "VC/equity": "#54A24B"}
ATYPE_TO_TYPE = {"funder": "philanthropy", "government": "government", "industry": "corporate"}


def _canonical_per_funder(df):
    """One canonical money series per funder, de-duplicated exactly as before: OpenPhil/LTFF use
    their donor annual total (funding_total mixed_technical, falling back to that year's full
    funding_total sum), everyone else uses their itemized grant_out. Returns (per, totals):
    per = rows of (actor, year, scale, cum, type); totals = per-funder (actor, type, total) sorted
    ascending. Type comes from the MONEY rows only — a global .first() mis-maps actors like FLI
    whose non-money rows carry a different actor_type."""
    money = df[df.event_type.isin(["grant_out", "funding_total"]) & df.scale.notna()].copy()
    money["scale"] = money["scale"].astype(float)
    has_total = set(money[money.event_type == "funding_total"].actor)
    ft = money[money.event_type == "funding_total"]
    mt = ft[ft.direction == "mixed_technical"].groupby(["actor", "year"])["scale"].sum()
    anyft = ft.groupby(["actor", "year"])["scale"].sum()
    annual = mt.reindex(anyft.index).fillna(anyft).reset_index(name="scale")
    grants = (money[(~money.actor.isin(has_total)) & (money.event_type == "grant_out")]
              .groupby(["actor", "year"], as_index=False)["scale"].sum())
    per = pd.concat([annual, grants], ignore_index=True).sort_values(["actor", "year"])
    per["cum"] = per.groupby("actor")["scale"].cumsum()
    atype = money.groupby("actor")["actor_type"].first()
    per["type"] = per["actor"].map(atype).map(ATYPE_TO_TYPE)
    assert not per["type"].isna().any(), \
        f"unmapped actor_type: {per[per.type.isna()].actor.unique()}"
    totals = per.groupby(["actor", "type"])["cum"].max().reset_index(name="total").sort_values("total")
    return per, totals


def _vc_rounds(df):
    inv = df[df.event_type == "investment"].copy()
    assert not inv.empty, "no investment rows found — expected VC/equity events"
    inv["scale"] = inv["scale"].astype(float)
    return inv


def chart_funding_ranked(df):
    """Ranked horizontal bar: one bar per funder, sorted by canonical total, log x, coloured by
    funder TYPE. VC/equity into startups is appended as a 4th colour group — still a SEPARATE
    lens (equity), never summed into the grant total, just shown on the same magnitude axis."""
    _, totals = _canonical_per_funder(df)
    inv = _vc_rounds(df)
    vc = inv.groupby("actor", as_index=False)["scale"].sum().rename(columns={"scale": "total"})
    vc["type"] = "VC/equity"
    both = pd.concat([totals, vc[["actor", "type", "total"]]], ignore_index=True).sort_values("total")
    fig = go.Figure()
    for typ in ["philanthropy", "government", "corporate", "VC/equity"]:
        s = both[both.type == typ]
        if s.empty:
            continue
        fig.add_trace(go.Bar(
            y=s["actor"], x=s["total"], orientation="h", name=typ, marker_color=TYPE_COLOR[typ],
            text=[f"${v / 1e6:,.1f}M" for v in s["total"]], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>"))
    fig.update_xaxes(type="log", title_text="total $ (log)", range=[5, 8.9])
    fig.update_layout(
        title="Money by funder, ranked — grants by type + VC/equity as a 4th group (log)<br>"
              "<sub>one bar per funder; canonical total (OpenPhil/LTFF = donor yearly totals, rest = "
              "itemized grants). VC/equity is equity, a separate lens — never summed with grants</sub>",
        height=820, width=1150, plot_bgcolor="white", barmode="stack",
        margin=dict(l=210, r=90, t=100), legend=dict(orientation="h", y=1.04))
    save(fig, "funding_ranked")


def chart_cumulative_funding(df):
    """Cumulative money over time by funder TYPE: grants stacked (philanthropy / government /
    corporate) = the money explosion and the government takeover from 2023; VC/equity as a
    dotted separate-lens line on the same axis (equity, never summed into the grant stack)."""
    per, _ = _canonical_per_funder(df)
    inv = _vc_rounds(df)
    yr = list(range(int(YEAR_AXIS_RANGE[0] + 0.5), int(YEAR_AXIS_RANGE[1] + 0.5) + 1))
    fig = go.Figure()
    for typ in ["philanthropy", "government", "corporate"]:
        s = per[per.type == typ].groupby("year")["scale"].sum().reindex(yr, fill_value=0).cumsum()
        fig.add_trace(go.Scatter(
            x=yr, y=s.values, mode="lines", name=typ, stackgroup="grants",
            line=dict(width=0.5, color=TYPE_COLOR[typ]), fillcolor=TYPE_COLOR[typ],
            hovertemplate=typ + " %{x}: $%{y:,.0f} cumulative<extra></extra>"))
    vc = inv.groupby("year")["scale"].sum().reindex(yr, fill_value=0).cumsum()
    fig.add_trace(go.Scatter(
        x=yr, y=vc.values, mode="lines", name="VC/equity (separate lens)",
        line=dict(width=2.5, color=TYPE_COLOR["VC/equity"], dash="dot"),
        hovertemplate="VC/equity %{x}: $%{y:,.0f} cumulative<extra></extra>"))
    fig.update_xaxes(dtick=1, title_text="year", range=YEAR_AXIS_RANGE)
    fig.update_yaxes(title_text="cumulative $ (grants stacked)")
    fig.update_layout(
        title="Cumulative money over time, by funder type<br>"
              "<sub>grants stacked (philanthropy / government / corporate) — the money explosion &amp; "
              "the government takeover from 2023; VC/equity as a dotted separate-lens line, never summed</sub>",
        height=720, width=1250, plot_bgcolor="white",
        margin=dict(t=100), legend=dict(x=0.01, y=0.99))
    save(fig, "cumulative_funding")


def chart_funding_dotstrip(df):
    """Dot-strip: x = year, y = funder (sorted by canonical total), dot AREA ~ that year's $,
    colour = funder type. Shows WHEN each funder acted and how big each move was; VC/equity
    rounds are diamonds (separate lens)."""
    per, totals = _canonical_per_funder(df)
    inv = _vc_rounds(df)
    inv2 = inv.copy()
    inv2["type"] = "VC/equity"
    order = totals.sort_values("total")["actor"].tolist()
    vc_order = inv.groupby("actor")["scale"].sum().sort_values().index.tolist()
    yorder = vc_order + order
    sizeref = 2.0 * per["scale"].max() / (44.0 ** 2)
    fig = go.Figure()
    for typ in ["philanthropy", "government", "corporate"]:
        s = per[per.type == typ]
        fig.add_trace(go.Scatter(
            x=s["year"], y=s["actor"], mode="markers", name=typ,
            marker=dict(color=TYPE_COLOR[typ], size=s["scale"], sizemode="area",
                        sizeref=sizeref, sizemin=4, line=dict(width=0.5, color="white")),
            hovertemplate="%{y} (%{x})<br>$%{marker.size:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=inv2["year"], y=inv2["actor"], mode="markers", name="VC/equity",
        marker=dict(color=TYPE_COLOR["VC/equity"], size=inv2["scale"], sizemode="area",
                    sizeref=sizeref, sizemin=4, symbol="diamond", line=dict(width=0.5, color="white")),
        hovertemplate="%{y} (%{x})<br>$%{marker.size:,.0f}<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=yorder)
    fig.update_xaxes(dtick=1, title_text="year", range=YEAR_AXIS_RANGE, gridcolor="#eee")
    fig.update_layout(
        title="Dot-strip: when each funder acted &amp; how big (dot area = $)<br>"
              "<sub>x = year, y = funder (sorted by total), colour = type; VC/equity rounds as diamonds "
              "(separate lens)</sub>",
        height=820, width=1150, plot_bgcolor="white",
        margin=dict(l=210, t=100), legend=dict(orientation="h", y=1.04))
    save(fig, "funding_dotstrip")


def chart_event_composition(df):
    """100%-stacked event-type mix per era — how the field's activity shifted from founding
    orgs → publishing → funding."""
    d = df.copy()
    d["era"] = d["year"].map(era_of)
    eras = [l for l, _, _ in ERAS if l in set(d["era"])]
    c = d.groupby(["era", "event_type"]).size().reset_index(name="n")
    fig = px.bar(c, x="era", y="n", color="event_type",
                 category_orders={"era": eras},
                 title="Event-type composition by era (100%-stacked) — shift of the field's activity mix")
    fig.update_layout(height=650, width=1250, barmode="stack", barnorm="percent",
                      yaxis_title="% of era's events", plot_bgcolor="white")
    save(fig, "event_composition")


def chart_funding_absolute(df, pubs):
    """SMALL MULTIPLES: one mini-panel per track — grant-$ bars (left axis) + that track's arXiv
    attention line (right axis, log, dotted). Replaces the old single dual-axis chart, which piled
    ~20 attention lines over the stacked bars into unreadable spaghetti. Here each track is legible
    on its own and its own money-vs-attention divergence is directly visible.

    Money is ALL itemized grant_out (not just the track-classified $680M): the aggregate donor
    buckets + lone non-safety grant (NONTRACK_MONEY) have no real track, so they get one honest
    "other (untracked/aggregate)" panel (money only, no arXiv line) — the grid still totals the
    full $763.6M, nothing dropped. Both money and arXiv are proxies (caveats 1-2)."""
    OTHER_BAND = "other (untracked/aggregate)"
    g = df[(df.event_type == "grant_out") & df.scale.notna()].copy()
    g["scale"] = g["scale"].astype(float)
    g["direction"] = g["direction"].where(~g["direction"].isin(NONTRACK_MONEY), OTHER_BAND)
    years = list(range(2015, int(df.year.max()) + 1))
    _, cmap = _track_birth_order(df)
    cmap = _with_pub_colors(cmap, pubs)
    cmap[OTHER_BAND] = "#BAB0AC"  # neutral grey; not a real track, so it steals no track colour
    # one panel per money track, ordered by total $ desc; the untracked band goes last.
    tot = g.groupby("direction")["scale"].sum().sort_values(ascending=False)
    panels = [t for t in tot.index if t != OTHER_BAND] + ([OTHER_BAND] if OTHER_BAND in tot.index else [])
    pub_track_list = [t for t in pubs["track"].unique() if not t.startswith("_")]

    cols = 3
    rows = math.ceil(len(panels) / cols)
    specs = [[{"secondary_y": True} for _ in range(cols)] for _ in range(rows)]
    fig = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=panels,
                        vertical_spacing=0.07, horizontal_spacing=0.075)
    for i, t in enumerate(panels):
        r, col = i // cols + 1, i % cols + 1
        m = g[g.direction == t].groupby("year")["scale"].sum().reindex(years, fill_value=0)
        fig.add_trace(go.Bar(x=years, y=m.values, marker_color=cmap[t], showlegend=False,
                             hovertemplate=t + " %{x}: $%{y:,.0f}<extra></extra>"),
                      row=r, col=col, secondary_y=False)
        if t in pub_track_list:  # the untracked band has no arXiv proxy -> money-only panel
            a = pubs[pubs.track == t].groupby("year")["arxiv_count"].sum().reindex(years, fill_value=0)
            fig.add_trace(go.Scatter(x=years, y=a.values, mode="lines", showlegend=False,
                                     line=dict(color="#444", width=1.6, dash="dot"),
                                     hovertemplate=t + " %{x}: %{y} arXiv/yr<extra></extra>"),
                          row=r, col=col, secondary_y=True)
            fig.update_yaxes(type="log", showgrid=False, tickfont=dict(size=7),
                             row=r, col=col, secondary_y=True)
        fig.update_yaxes(tickfont=dict(size=7), row=r, col=col, secondary_y=False)
        fig.update_xaxes(tickfont=dict(size=7), dtick=2, row=r, col=col)
    for ann in fig.layout.annotations:      # per-panel titles: keep them small & readable
        ann.font.size = 11
    fig.update_layout(height=250 * rows, width=1300, plot_bgcolor="white", bargap=0.15,
                      margin=dict(t=100, l=55, r=45, b=45),
                      title="Grant money vs scientific attention, per track (small multiples)<br>"
                            "<sub>one panel per track: bars = grant money/yr USD (left), dotted line = arXiv "
                            "papers/yr (right, LOG). All itemized grants (763.6M USD) incl. an 'other/untracked' "
                            "panel; both are proxies (caveats 1-2)</sub>")
    save(fig, "funding_absolute")


def chart_track_radar(df):
    """Radar of total known grant $ per track. CAVEAT: single money axis only — this is a
    footprint on ONE metric, not a multi-metric comparison (those mislead on mixed axes)."""
    g = _track_money(df)
    tot = g.groupby("direction")["scale"].sum().sort_values(ascending=False)
    cats = tot.index.tolist() + [tot.index[0]]
    vals = tot.values.tolist() + [tot.values[0]]
    fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill="toself", name="grant $"))
    fig.update_layout(height=720, width=920, polar=dict(radialaxis=dict(type="log", title="USD (log)")),
                      margin=dict(t=110),
                      title="Track money footprint — total known grant $ per track<br>"
                            "<sub>CAVEAT: single log money axis; itemized grants only; NOT multi-metric</sub>")
    save(fig, "track_radar")


# fourth money lens: collected $ that is NOT a disbursed grant — pledges, annual budgets and
# field-wide estimates. Kept OFF every $-chart above (mixing them would conflate a pledge with a
# real grant or double-count overlapping estimates) and shown here as labelled DOTS only, never
# summed. Sub-type is derived from the row itself (event_type + whether the actor is the whole
# "field"), so no new data column is needed.
PLEDGE_EVENT_TYPES = ["statement", "program_launch", "founded"]


def _pledge_subtype(row):
    if row["event_type"] == "founded":
        return "seed at founding"
    if row["event_type"] == "program_launch":
        return "launch pledge"
    if row["actor"] == "field":
        return "field-wide estimate (overlapping)"
    return "org annual budget / fund pledge"


PLEDGE_COLORS = {
    "seed at founding": "#54A24B",
    "launch pledge": "#E45756",
    "field-wide estimate (overlapping)": "#B279A2",
    "org annual budget / fund pledge": "#4C78A8",
}


def _pledges(df):
    """The 4th-lens rows (pledges / budgets / estimates with a $ figure), tagged with sub-type and
    a unique bar label. Some actors have two rows in one year (e.g. FTX 2022: a launch pledge AND
    a budget), so labels are disambiguated by amount to keep bars from colliding on a shared y."""
    p = df[df.event_type.isin(PLEDGE_EVENT_TYPES) & df.scale.notna()].copy()
    assert not p.empty, "no pledge/budget/estimate rows with a $ figure found"
    p["scale"] = p["scale"].astype(float)
    p["subtype"] = p.apply(_pledge_subtype, axis=1)
    missing = set(p["subtype"]) - set(PLEDGE_COLORS)
    assert not missing, f"pledge sub-types missing a colour: {sorted(missing)}"
    p["label"] = p["actor"] + " (" + p["year"].astype(str) + ")"
    dup = p["label"].duplicated(keep=False)
    p.loc[dup, "label"] = (p.loc[dup, "label"] + " · $"
                           + (p.loc[dup, "scale"] / 1e6).round(2).astype(str) + "M")
    assert not p["label"].duplicated().any(), "pledge labels still collide"
    return p


def chart_pledges_budgets(df):
    """Fourth lens — money that is NOT a disbursed grant: launch pledges, annual org budgets and
    overlapping field-wide estimates. Ranked HORIZONTAL bars grouped by sub-type on a log axis;
    magnitude is the message. Heterogeneous and NON-additive (the 'field' estimates already
    contain the org budgets, and a pledge is not a disbursement) — read each bar alone, never a
    total, and never mixed with grants/donor totals/VC."""
    p = _pledges(df)
    order = ["launch pledge", "field-wide estimate (overlapping)",
             "org annual budget / fund pledge", "seed at founding"]
    p["ord"] = p["subtype"].map({s: i for i, s in enumerate(order)})
    p = p.sort_values(["ord", "scale"], ascending=[False, True])
    fig = go.Figure()
    for sub in order:
        s = p[p.subtype == sub]
        if s.empty:
            continue
        fig.add_trace(go.Bar(
            y=s["label"], x=s["scale"], orientation="h", name=sub, marker_color=PLEDGE_COLORS[sub],
            text=[f"${v / 1e6:,.2f}M" for v in s["scale"]], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>"))
    fig.update_xaxes(type="log", title_text="disclosed figure $ (log; NOT a grant, NEVER summed)",
                     range=[3.5, 8.6])
    fig.update_layout(
        # NOTE: never put two '$' in one title string — Plotly reads $...$ as MathJax/LaTeX and
        # garbles the text. Use 'USD' in the title prose.
        title="Pledges, annual budgets &amp; field estimates — a 4th lens, ranked (log)<br>"
              "<sub>collected money that is NOT a disbursed grant: launch pledges (FTX ≈160M USD "
              "pledged vs 18.7M USD disbursed), org budgets, overlapping field estimates.<br>"
              "Heterogeneous &amp; non-additive — read each bar alone, never a total, never mixed "
              "with grants/donor totals/VC</sub>",
        height=840, width=1200, plot_bgcolor="white", barmode="overlay",
        margin=dict(l=230, r=90, t=110, b=90),
        legend=dict(orientation="h", yanchor="top", y=-0.08, x=0.5, xanchor="center"))
    save(fig, "pledges_budgets")


def chart_pledges_ftx(df):
    """The 4th lens's headline, made a picture: a pledge is NOT a disbursement. FTX Future Fund
    pledged ≈$160M at launch but actually disbursed only $18.7M (dumbbell); everything else that
    is not a grant — org budgets, overlapping field estimates, seeds — ranked below on a log axis.
    Never summed with grants/donor totals/VC."""
    ftx_pledge = 160e6
    g = df[(df.event_type == "grant_out") & df.scale.notna()].copy()
    g["scale"] = g["scale"].astype(float)
    ftx_disbursed = float(g.loc[g.actor == "FTX Future Fund", "scale"].sum())
    assert ftx_disbursed > 0, "expected FTX Future Fund grant_out rows for the disbursed total"
    p = _pledges(df)
    rest = p[~((p.actor == "FTX Future Fund") & (p.event_type == "program_launch"))].sort_values("scale")
    fig = make_subplots(rows=2, cols=1, row_heights=[0.22, 0.78], vertical_spacing=0.16,
                        subplot_titles=("FTX Future Fund: pledged vs actually disbursed",
                                        "All other pledges / budgets / estimates (ranked, log)"))
    fig.add_trace(go.Scatter(
        x=[ftx_disbursed, ftx_pledge], y=["FTX Future Fund", "FTX Future Fund"], mode="lines",
        line=dict(color="#999", width=3), showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[ftx_pledge], y=["FTX Future Fund"], mode="markers+text", name="pledged",
        marker=dict(color="#E45756", size=18), text=[f"pledged ${ftx_pledge / 1e6:.0f}M"],
        textposition="top center", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[ftx_disbursed], y=["FTX Future Fund"], mode="markers+text", name="disbursed",
        marker=dict(color="#4C78A8", size=18), text=[f"disbursed ${ftx_disbursed / 1e6:.1f}M"],
        textposition="bottom center", hoverinfo="skip"), row=1, col=1)
    fig.update_xaxes(type="log", range=[7, 8.3], row=1, col=1)
    for sub in PLEDGE_COLORS:
        s = rest[rest.subtype == sub]
        if s.empty:
            continue
        fig.add_trace(go.Bar(
            y=s["label"], x=s["scale"], orientation="h", name=sub, marker_color=PLEDGE_COLORS[sub],
            text=[f"${v / 1e6:,.2f}M" for v in s["scale"]], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>"), row=2, col=1)
    fig.update_xaxes(type="log", title_text="$ (log)", range=[3.5, 8.3], row=2, col=1)
    fig.update_layout(
        # NOTE: never put two '$' in one title string (MathJax trap). Use 'USD' in the prose.
        title="A pledge is not a disbursement — FTX 160M USD pledged vs 18.7M USD disbursed<br>"
              "<sub>the dumbbell is the headline; the rest of the 4th lens (budgets, overlapping field "
              "estimates, seeds) ranked below — never summed with grants/donor totals/VC</sub>",
        height=880, width=1200, plot_bgcolor="white", barmode="overlay",
        margin=dict(l=230, r=90, t=110, b=120),
        legend=dict(orientation="h", yanchor="top", y=-0.09, x=0.5, xanchor="center"))
    save(fig, "pledges_ftx")


def _money_attention_totals(df, pubs):
    """Per-track final cumulative grant $ and arXiv papers, for tracks that have BOTH.
    Shared by the diverging-gap and dumbbell charts so they agree exactly."""
    g = _track_money(df)
    pub_tracks = [t for t in pubs["track"].unique() if not t.startswith("_")]
    tracks = sorted(set(g.direction.unique()) & set(pub_tracks))
    money = {t: float(g[g.direction == t]["scale"].sum()) for t in tracks}
    attn = {t: float(pubs[pubs.track == t]["arxiv_count"].sum()) for t in tracks}
    tracks = [t for t in tracks if money[t] > 0 and attn[t] > 0]
    assert tracks, "no track has both grant money and a pub curve"
    return tracks, money, attn


def chart_money_vs_attention_tracks(df, pubs):
    """DIVERGING GAP bars: for each track, how far it sits from the field-average money->attention
    rate. Replaces the old log-log trajectory tangle. k = total_attention / total_money is the
    field-average papers-per-dollar; a track's gap = log2(actual attention / (k * its money)):
      > 0  attention leads money (science ran ahead of funding) — blue,
      < 0  money leads attention                               — red.
    Only tracks with BOTH a money series and a pub proxy. Both signals are proxies (caveats 1-2)."""
    tracks, money, attn = _money_attention_totals(df, pubs)
    k = sum(attn.values()) / sum(money.values())
    gap = {t: math.log2(attn[t] / (k * money[t])) for t in tracks}
    order = sorted(tracks, key=lambda t: gap[t])          # most money-ahead at bottom
    vals = [gap[t] for t in order]
    colors = ["#4C78A8" if v >= 0 else "#E45756" for v in vals]
    labels = [f"{'attention' if v >= 0 else 'money'} ×{2 ** abs(v):.1f}" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=order, orientation="h", marker_color=colors,
        text=labels, textposition="outside", textfont=dict(size=10),
        customdata=[[money[t], attn[t]] for t in order],
        hovertemplate="%{y}<br>grant $%{customdata[0]:,.0f} · %{customdata[1]:,.0f} arXiv papers<extra></extra>"))
    fig.add_vline(x=0, line_color="#888", line_width=1)
    m = max(abs(min(vals)), abs(max(vals))) * 1.4
    fig.update_layout(
        height=560, width=1050, plot_bgcolor="white", margin=dict(t=110, l=150, r=60, b=60),
        xaxis=dict(range=[-m, m], zeroline=False,
                   title="← money leads   ·   log2(attention / field-average)   ·   attention leads →"),
        title="Who leads per track — attention vs money (gap from the field average)<br>"
              "<sub>gap from field-average money→attention rate; blue = attention ahead of money, "
              "red = money ahead; ×N = the factor. Proxies (caveats 1-2)</sub>")
    save(fig, "money_vs_attention_tracks")


def chart_money_attention_dumbbell(df, pubs):
    """DUMBBELL: each track's RANK by grant money vs its RANK by arXiv attention, connected.
    A long connector = the two ranks disagree strongly (attention far ahead of money = blue, or
    money far ahead = red); overlapping dots = money and attention rank the track the same.
    Companion to the diverging-gap chart (ranks show the ordering; the gap chart shows magnitude).
    Only tracks with BOTH series; both are proxies (caveats 1-2)."""
    tracks, money, attn = _money_attention_totals(df, pubs)
    n = len(tracks)
    money_rank = {t: i + 1 for i, t in enumerate(sorted(tracks, key=lambda t: money[t]))}
    attn_rank = {t: i + 1 for i, t in enumerate(sorted(tracks, key=lambda t: attn[t]))}
    order = sorted(tracks, key=lambda t: attn_rank[t] + money_rank[t])
    fig = go.Figure()
    for t in order:                                       # connector coloured by who leads
        lead_attn = attn_rank[t] > money_rank[t]
        fig.add_trace(go.Scatter(
            x=[money_rank[t], attn_rank[t]], y=[t, t], mode="lines",
            line=dict(color="#4C78A8" if lead_attn else "#E45756", width=3),
            showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[money_rank[t] for t in order], y=order, mode="markers", name="money rank",
        marker=dict(size=13, color="#F58518", line=dict(width=1, color="white")),
        hovertemplate="%{y}: money rank %{x}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[attn_rank[t] for t in order], y=order, mode="markers", name="attention rank",
        marker=dict(size=13, color="#54A24B", line=dict(width=1, color="white")),
        hovertemplate="%{y}: attention rank %{x}<extra></extra>"))
    fig.update_layout(
        height=560, width=1050, plot_bgcolor="white", margin=dict(t=110, l=150, r=60, b=60),
        xaxis=dict(title=f"rank (1 = smallest … {n} = largest)", dtick=1),
        legend=dict(x=0.5, y=1.03, xanchor="center", orientation="h"),
        title="Money rank vs attention rank per track (dumbbell)<br>"
              "<sub>orange = money rank, green = attention rank; long blue bar = attention far ahead "
              "of money, red = money far ahead; overlap = agree. Companion to the gap chart</sub>")
    save(fig, "money_attention_dumbbell")


# ---------------------------------------------------------------- REPORT ----

def md_table(frame):
    cols = list(frame.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in frame.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def build_report(df, pubs):
    lines = ["# Auto-report — AI-safety genealogy (generated by build_viz.py)", ""]
    lines += [f"Rows: **{len(df)}** · actors: **{df.actor.nunique()}** · tracks: "
              f"**{df.direction.nunique()}** · years: **{df.year.min()}–{df.year.max()}**", ""]

    lines += ["## Events by type", ""]
    et = df.event_type.value_counts().rename_axis("event_type").reset_index(name="n")
    lines += [md_table(et), ""]

    lines += ["## Known individual grants ($) by funder (grant_out)", ""]
    g = df[(df.event_type == "grant_out") & df.scale.notna()]
    by_funder = (g.groupby("actor")["scale"].agg(["sum", "count"])
                 .sort_values("sum", ascending=False).reset_index())
    by_funder["sum"] = by_funder["sum"].map(lambda v: f"${v:,.0f}")
    by_funder.columns = ["funder", "total_$", "n_grants"]
    lines += [md_table(by_funder), ""]
    lines += [f"Total individual grant $ mapped: **${g.scale.sum():,.0f}** across {len(g)} grants.", ""]

    lines += ["## Donor yearly totals — funding_total[mixed_technical] (OpenPhil + LTFF)", ""]
    mt = (df[(df.event_type == "funding_total") & (df.direction == "mixed_technical")]
          .pivot_table(index="year", columns="actor", values="scale", aggfunc="sum")
          .fillna(0).astype(int).reset_index())
    lines += [md_table(mt), ""]

    lines += ["## Funding by subtrack × year (funding_total, excl. mixed_technical)", ""]
    st = (df[(df.event_type == "funding_total") & (df.direction != "mixed_technical")]
          .pivot_table(index="year", columns="direction", values="scale", aggfunc="sum")
          .fillna(0).astype(int).reset_index())
    lines += [md_table(st), ""]

    lines += ["## Known grant $ by track (grant_out; excl. aggregate & non-safety buckets)", ""]
    tm = _track_money(df)
    by_track = (tm.groupby("direction")["scale"].agg(["sum", "count"])
                .sort_values("sum", ascending=False).reset_index())
    by_track["sum"] = by_track["sum"].map(lambda v: f"${v:,.0f}")
    by_track.columns = ["track", "grant_$", "n_grants"]
    lines += [md_table(by_track), ""]

    lines += ["## Actor registry (from events)", ""]
    reg = (df.groupby("actor")
           .agg(first_year=("year", "min"), last_year=("year", "max"), n_events=("year", "size"))
           .sort_values("first_year").reset_index())
    lines += [md_table(reg), ""]

    lines += ["## Coverage control — big tracks × year (event count; 0 = potential gap)", ""]
    years = list(range(int(df.year.min()), int(df.year.max()) + 1))
    cov = pd.DataFrame(0, index=BIG_TRACKS, columns=years)
    counts = df[df.direction.isin(BIG_TRACKS)].groupby(["direction", "year"]).size()
    for (trk, yr), n in counts.items():
        cov.loc[trk, yr] = n
    cov = cov.reset_index().rename(columns={"index": "track"})
    lines += [md_table(cov), ""]

    lines += ["## arXiv attention (per-track proxies + `_safety_corpus` dedup line; `noise` = keyword-noise level)", ""]
    pub_cols = ["track", "year", "arxiv_count"] + (["noise"] if "noise" in pubs.columns else [])
    lines += [md_table(pubs[pub_cols]), ""]

    lines += ["## Caveats (must be stated in the post)", "",
              "1. arXiv keyword counts are a proxy, not bibliometrics — the term lags the real track start.",
              "2. Event count per track = collection density, not real-world activity (use money + publications).",
              "3. Causal 'why' claims are interpretation over verified events, not sourced facts.",
              "4. Recipient organisation is not a structured column (only free text in `detail`), so funding "
              "Sankeys route funder→track / era→funder→track, never funder→org.", ""]

    lines += ["## Resolved this pass (with primary sources)", "",
              "- arXiv attention curves added for scalable_oversight, evals, ai_control, agent_foundations, value_learning (raw exact-phrase proxies, noise flagged; `source` = query URL).",
              "- OpenPhil 2024 technical-safety total (~$28M, LW analysis of grants DB; flagged as different methodology than vipulnaik); RFP $40M re-dated to its true 2025.",
              "- MATS founding date fixed to Dec 2021 (primary LW post) + a duplicate 2021 launch row de-duplicated into the real 2022 cohort.",
              "- CAIS revenue verified via ProPublica 990 (EIN 88-1751310): FY22 $6.66M / FY23 $16.09M / FY24 $10.24M.",
              "- FTX '$32M' re-sourced off the Fortune paywall to the EAF/LW overview; changed grant_out→statement to fix a double-count with the itemized FTX regrants.",
              "- 2024-25 coverage events added (Scaling Monosemanticity, GPT-4 SAEs, circuit tracing, Inspect, Alignment Faking, Ctrl-Z) so the coverage matrix no longer contradicts the pub curves.", ""]

    lines += ["## Remaining honest gaps (NOT fabricated — flagged)", "",
              "- arXiv curves for the 5 newer tracks are keyword proxies (noise flagged per row), not bibliometrics.",
              "- 2026 track-classified money is not yet machine-readable — left as a gap, not a zero or a guess.",
              "- Recipient-organisation is free text in `detail`, not a structured column, so funder→org flows are not built.", ""]

    open(REPORT_MD, "w", encoding="utf-8").write("\n".join(lines))


# ---------------------------------------------------------------- DASHBOARD -

CHARTS = [
    ("timeline.html", "Timeline", "Every event by organisation & year; track colour is a chronological gradient (ordered by first appearance). Bottom panel: stacked-area of events/track/year showing the activity-mix shift.", "A diagonal from bottom-left to top-right — early macrostrategy/agent-foundations give way to interpretability, evals and AI control; the bottom panel shows the same activity-mix shift."),
    ("track_lifespan.html", "Track lifespans (dumbbell)", "Birth → last recorded event per track (ordered by birth). Hollow diamond = last year still publishing on arXiv (independent proxy). Old ≠ dead; absence of late events = collection density, not death.", "Old is not dead: reward modeling's last event is 2017 but RLHF publishes to 2025; agent foundations and value learning go quiet as events yet stay alive on arXiv — tracks get absorbed or move to the background."),
    ("alluvial_org.html", "Organisational alluvial", "era → organisation → track. Ribbons show who worked on what, when (money rows excluded).", "The full route era to organisation to track. At first glance it is spaghetti (~115 orgs) — the next three variants roll it up to stay readable."),
    ("alluvial_grouped.html", "Alluvial compact (era → actor-type → family)", "Variant A — ~115 orgs collapsed to 7 actor types, 17 tracks to 4 families; wide legible ribbons.", "Rolled up to 7 actor types and 4 families: blue 'foundations & strategy' is dense early/left, orange 'technical safety' gains mass on the right."),
    ("alluvial_by_era.html", "Alluvial by era (small multiples)", "Variant B — one mini-Sankey per era: actor-type → track-family.", "One panel per era — 'foundations & strategy' rules 2005-2016, then the panels flood orange with 'technical safety' from the 2020s."),
    ("alluvial_pairs.html", "Alluvial pairs (2-level Sankeys)", "Variant C — era→family and actor-type→family, two clean 2-level flows.", "Two clean flows: the field's focus shifts to technical safety over time (left); researchers/industry pull technical safety while government and institutes pull governance (right)."),
    ("era_shift.html", "How the field shifted across eras", "Cross-era comparison (era on X): family mix (100%), actor mix (100%), money by funder ($), org births/closures.", "Across the eras: 'foundations & strategy' falls 67% to 6% while 'technical safety' rises to ~60%; government goes from nothing to 24% and industry to 26%; grant money explodes from ~$0 to ~$335M with a dominant grey government band by 2024-26."),
    ("funding_sankey.html", "Funding Sankey (funder → track)", "Known individual grants routed from funder to research track.", "OpenPhil and SFF spread across many tracks; government institutes enter with precision — mostly field-building, evals and robustness. Routes all itemized grants (~$764M)."),
    ("funding_sankey_era.html", "Funding routing (era → funder → track)", "Same grants, 3 structural levels. Recipient-org is not a clean column, so it is not a level.", "The same grants with an era level — the mass of money shifts into 2024-2026, and the new government institutes feed evals/field-building/robustness."),
    ("funding_absolute.html", "Funding vs attention by track (small multiples)", "One mini-panel per track: grant money/yr (bars, left axis) + arXiv attention (dotted line, right axis = log). All itemized grants ($763.6M) incl. a grey 'other (untracked/aggregate)' panel. Both are proxies.", "One panel per track: interpretability is a steep attention line over tiny money bars (attention far ahead of money), evals is a wall of 2024-25 bars, AI control has money before its attention lifts."),
    ("funding_ranked.html", "Money by funder, ranked", "One horizontal bar per funder, sorted by canonical total, log axis, coloured by type (philanthropy/government/corporate) + VC/equity as a 4th group. Equity is a SEPARATE lens — never summed with grants.", "One bar per funder by type — philanthropy (OpenPhil $304.5M, SFF $144M...), government (UK AISI ~$159M, ARIA $74M...), corporate, and a separate VC/equity group (~$268M) never summed with the $763.6M of grants."),
    ("cumulative_funding.html", "Cumulative money by funder type", "Cumulative $ over time, grants stacked by type (philanthropy/government/corporate) — the money explosion & government takeover from 2023; VC/equity as a dotted separate-lens line, never summed.", "Through 2020 it is almost entirely one blue philanthropy artery; from 2023 the orange government band explodes as national institutes arrive. The dotted green VC line (~$268M) is a separate lens, never summed."),
    ("funding_dotstrip.html", "Funding dot-strip (when & how big)", "x=year, y=funder (sorted by total), dot area ~ that year's $, colour=type; VC/equity rounds as diamonds. Shows when each funder acted and the size of each move.", "OpenPhil is active almost every year; the government cluster is unmistakable in 2023-2025 and VC rounds appear as green diamonds from 2023 — philanthropy carried the field alone until the state and market arrived together."),
    ("track_radar.html", "Track money footprint (radar)", "Total grant $ per track on ONE log axis. Caveat: single-metric, not multi-axis.", "Tracks' money footprint on a single log axis — deliberately one metric (money), since multi-metric radars mislead across scales."),
    ("money_vs_attention.html", "Money vs attention (field)", "Field-level: OpenPhil+LTFF yearly donor totals (bars, end at 2024 \u2014 2025+ not published yet) vs arXiv attention (solid = dedup corpus, dotted = inflated keyword sum; right axis = log).", "Donor totals (OpenPhil+LTFF, bars) end at 2024 because 2025+ totals are not published yet; the solid red line is the trustworthy dedup corpus, the dotted line the inflated keyword sum."),
    ("money_vs_attention_tracks.html", "Who leads per track (gap from field average)", "Diverging bars: per track log2(attention / field-average money-to-attention rate); blue = attention runs ahead of money (interpretability ×36.9), red = money ahead (scalable_oversight, agent_foundations, governance). Both are proxies.", "Distance from the field-average money-to-attention rate: blue = attention ahead of money (interpretability ~x36.9), red = money ahead (scalable oversight, agent foundations, governance)."),
    ("money_attention_dumbbell.html", "Money rank vs attention rank (dumbbell)", "Companion to the gap chart: each track's rank by grant money (orange) vs rank by arXiv attention (green); long blue bar = attention far ahead of money, red = money far ahead, overlap = the two agree.", "The same tracks by rank — money rank vs attention rank joined by a connector; interpretability's dots pull wide apart, well-matched tracks keep them close."),
    ("org_lifespan.html", "Organisation lifespans (Gantt)", "Founding → closure/pivot (or ongoing). FHI/Superalignment closed 2024; institutes renamed 2025.", "One bar per organisation, founding to last closure/pivot; 14 closure/pivot events collapse to 10 distinct orgs. Real closures (FHI, OpenAI Superalignment) land in 2024; FTX's 2022 collapse is the earlier exception."),
    ("births_deaths.html", "Births vs closures/pivots", "New orgs (up) vs closures + pivots (down) per year.", "New orgs up, closures/pivots down. Counts events, not orgs — 14 closure/pivot events (3 closures + 11 pivots), since one org can pivot several times (MIRI in 2013/2018/2024)."),
    ("event_composition.html", "Event-type mix by era", "100%-stacked: how the activity mix shifted (founding → publishing → funding).", "Prehistory (2005-2012) is 100% org foundings; publications, grants and statements enter from 2013, and by the last era foundings are only ~28% of a far larger mix."),
    ("track_emergence.html", "Track emergence (event count)", "CAVEAT: collection density, not real activity; recent-years window shaded as incomplete.", "Almost empty until 2012-13, explodes 2015-17 with the first technical tracks, then grows 2021-24 with evals, AI control and model organisms. Height is collection density, not real activity."),
    ("pledges_budgets.html", "Pledges, budgets & estimates (4th lens)", "Collected money that is NOT a disbursed grant: launch pledges (FTX ~$160M pledged vs $18.7M disbursed), org annual budgets, overlapping field-wide estimates. Ranked bars grouped by sub-type on a log axis, NEVER summed with grants/donor totals/VC.", "The 4th lens — money that is not a disbursed grant: launch pledges (FTX ~$160M), org budgets, overlapping field estimates, seeds. Heterogeneous and non-additive, so shown but never summed."),
    ("pledges_ftx.html", "FTX pledged vs disbursed (4th lens)", "The 4th lens's headline as a dumbbell: FTX pledged ~$160M at launch but disbursed only $18.7M; the rest of the non-grant money (budgets, field estimates, seeds) ranked below. Never summed with grants/donor totals/VC.", "The sharpest gap: FTX pledged ~$160M at launch but disbursed only $18.7M before its Nov-2022 collapse — announced intent vs delivered money."),
]

FINDINGS = [
    "OpenPhil is the dominant artery; peak 2021 ($81.7M); 2017 inflated by a single $30M OpenAI grant.",
    "Governance funding exploded 2018→2023 ($0.4M→$18.4M) — by 2023 the biggest OpenPhil subtrack after technical_safety ($24.6M).",
    "SFF grew explosively: 2020 $5.4M → 2023 $42.3M → 2025 $34.9M.",
    "FTX collapse (Nov 2022) was a funding shock (Redwood $6.6M, Ought $5M and other regrants cut).",
    "interpretability publications exploded (denoised 69→657, 2021–25); RLHF kept rising (25→1777, 2021–25) = absorption into mainstream.",
    "interpretability got a 2024-25 second wind (Scaling Monosemanticity 2024; circuit-tracing 2025); ai_control born 2023 (Redwood) → agentic control evals by 2025 (Ctrl-Z).",
    "Ending arc: in 2025 both institutes dropped 'Safety' (UK → AI Security Institute, US → CAISI); FHI closed (2024); OpenAI Superalignment dissolved (2024).",
]

CAVEATS = [
    "arXiv keyword counts are a proxy, not bibliometrics (term lags the real track start); 'AI control'/'dangerous capabilities'/'value learning' are especially noisy (flagged per row).",
    "Event count per track = collection density, not real-world activity — use money + publications; recent years are under-sampled (shaded on the emergence chart).",
    "Two money lenses that never mix: funding_total = donor yearly totals; grant_out = itemized grants. Per-track money views use grant_out only.",
    "OpenPhil 2024 total ($28M) is a different methodology than vipulnaik (org self-reports ~$50M); 2026 money is not yet available (honest gap, not zero).",
    "Radar uses ONE money axis only — multi-metric radars mislead on mixed scales, so we don't build them.",
    "Causal 'why' claims are interpretation over verified events, not sourced facts.",
]

DEBTS = [
    "arXiv curves for the 5 newer tracks are raw exact-phrase proxies (noise flagged); a citation-graph count would be cleaner but is out of scope.",
    "2026 track-classified money is not yet machine-readable — left as an honest gap, not fabricated.",
    "Recipient-organisation is still free text in `detail` (not a structured column), so funder→org flows remain out of reach.",
]


def build_dashboard(df, pubs, public):
    """Return the gallery HTML. public=True builds the site-embedded variant (links home,
    no internal report/post links, no per-chart PNG links); False builds the local one."""
    def ul(items):
        return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"

    blocks = []
    for fname, title, desc, note in CHARTS:
        png = "img/" + fname.replace(".html", ".png")
        pnglink = "" if public else f' &nbsp;·&nbsp; <a href="{png}" target="_blank">PNG →</a>'
        src = "alluvial_org_preview.html" if fname == "alluvial_org.html" else fname  # embed the compact preview
        blocks.append(
            f'<section><h2>{title}</h2><p class="note">{note}</p><p class="desc">{desc}</p>'
            f'<iframe src="{src}" class="fit" loading="lazy"></iframe>'
            f'<p class="open"><a href="{fname}" target="_blank">open full chart →</a>{pnglink}</p></section>'
        )

    stat = (f"{len(df)} events · {df.actor.nunique()} actors · {df.direction.nunique()} tracks · "
            f"{df.year.min()}–{df.year.max()} · "
            f"${df[df.event_type=='grant_out'].scale.sum():,.0f} individual grants mapped")

    nav = ('<a href="/">← Mrs Wallbreaker</a>' if public else
           '<a href="../report.md">report.md</a> · <a href="../POST_en.md">POST_en.md</a>')

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI-safety genealogy — visual dashboard</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; color: {DARK_FG}; background: {DARK_BG}; }}
  header {{ background: {DARK_PANEL}; border-bottom: 1px solid {DARK_LINE}; padding: 28px 40px; }}
  header h1 {{ margin: 0 0 6px; font-size: 26px; }}
  header .stat {{ color: {DARK_MUTED}; font-size: 14px; }}
  header .nav {{ margin-top: 10px; font-size: 13px; }}
  a {{ color: {DARK_ACCENT}; text-decoration: none; }}
  main {{ max-width: 1560px; margin: 0 auto; padding: 24px 20px 80px; }}
  .cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin: 8px 0 32px; }}
  .card {{ background: {DARK_PANEL}; border: 1px solid {DARK_LINE}; border-radius: 10px; padding: 16px 18px; }}
  .card h3 {{ margin: 0 0 8px; font-size: 15px; }}
  .card.findings {{ border-left: 4px solid {DARK_ACCENT}; }}
  .card.caveats {{ border-left: 4px solid #e5736f; }}
  .card.debts {{ border-left: 4px solid #F58518; }}
  ul {{ margin: 6px 0 0; padding-left: 20px; font-size: 13px; line-height: 1.5; color: {DARK_MUTED}; }}
  section {{ background: {DARK_PANEL}; border: 1px solid {DARK_LINE}; border-radius: 10px; margin: 0 0 26px; padding: 18px 20px; }}
  section h2 {{ margin: 0 0 4px; font-size: 19px; }}
  .note {{ color: {DARK_FG}; font-size: 14px; line-height: 1.5; margin: 0 0 10px; border-left: 2px solid {DARK_ACCENT}; padding-left: 10px; }}
  .desc {{ color: {DARK_MUTED}; font-size: 13px; margin: 0 0 12px; }}
  iframe {{ width: 100%; height: 720px; border: 0; border-radius: 6px; background: {DARK_BG}; }}
  .open a {{ font-size: 13px; }}
</style></head>
<body>
<header>
  <h1>AI-safety / alignment genealogy — visual dashboard</h1>
  <div class="stat">{stat}. <a href="https://github.com/eericheva/ai-safety-genealogy/tree/main/data" target="_blank">raw data</a></div>
  <div class="nav">{nav}</div>
</header>
<main>
  <div class="cards">
    <div class="card findings"><h3>Key findings</h3>{ul(FINDINGS)}</div>
    <div class="card caveats"><h3>Caveats</h3>{ul(CAVEATS)}</div>
    <div class="card debts"><h3>Open collection debts (flagged, not invented)</h3>{ul(DEBTS)}</div>
  </div>
  {"".join(blocks)}
</main>
<script>
const fit = f => {{ f.style.height = f.contentWindow.document.documentElement.scrollHeight + 'px'; }};
document.querySelectorAll('iframe.fit').forEach(f => f.addEventListener('load', () => fit(f)));
addEventListener('resize', () => document.querySelectorAll('iframe.fit').forEach(fit));
</script>
</body></html>"""


# ---------------------------------------------------------------- MAIN ------

def main():
    print("START build_viz")
    os.makedirs(IMG_DIR, exist_ok=True)
    df, pubs = load()
    print(f"rc=0 load ({len(df)} events, {df.actor.nunique()} actors, {df.direction.nunique()} tracks)")

    chart_timeline(df); print("rc=0 timeline")
    chart_track_lifespan(df, pubs); print("rc=0 track_lifespan")
    chart_alluvial(df); print("rc=0 alluvial_org")
    chart_alluvial_grouped(df); print("rc=0 alluvial_grouped")
    chart_alluvial_by_era(df); print("rc=0 alluvial_by_era")
    chart_alluvial_pairs(df); print("rc=0 alluvial_pairs")
    chart_era_shift(df); print("rc=0 era_shift")
    chart_funding_sankey(df); print("rc=0 funding_sankey")
    chart_funding_sankey_era(df); print("rc=0 funding_sankey_era")
    chart_track_emergence(df); print("rc=0 track_emergence")
    chart_money_vs_attention(df, pubs); print("rc=0 money_vs_attention")
    chart_org_lifespan(df); print("rc=0 org_lifespan")
    chart_births_deaths(df); print("rc=0 births_deaths")
    chart_funding_ranked(df); print("rc=0 funding_ranked")
    chart_cumulative_funding(df); print("rc=0 cumulative_funding")
    chart_funding_dotstrip(df); print("rc=0 funding_dotstrip")
    chart_event_composition(df); print("rc=0 event_composition")
    chart_funding_absolute(df, pubs); print("rc=0 funding_absolute")
    chart_track_radar(df); print("rc=0 track_radar")
    chart_money_vs_attention_tracks(df, pubs); print("rc=0 money_vs_attention_tracks")
    chart_money_attention_dumbbell(df, pubs); print("rc=0 money_attention_dumbbell")
    chart_pledges_budgets(df); print("rc=0 pledges_budgets")
    chart_pledges_ftx(df); print("rc=0 pledges_ftx")

    build_report(df, pubs); print("rc=0 report.md")
    open(f"{VIZ_DIR}/index.html", "w", encoding="utf-8").write(build_dashboard(df, pubs, False))
    print("rc=0 viz/index.html")

    g = df[(df.event_type == "grant_out") & df.scale.notna()]
    print(f"grant_out $ mapped: {g.scale.sum():,.0f} | funding_total rows: "
          f"{(df.event_type=='funding_total').sum()}")
    print("ALL DONE")


if __name__ == "__main__":
    main()
