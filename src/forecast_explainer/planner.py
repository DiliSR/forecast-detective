"""The planning layer: "which model should I use, and where?"

The explanation agent answers "why was THIS hour off?". The planner answers the question that comes next: given what we
know about both models, which one should I rely on - overall, on holidays, in the evening, in a given week?

    accuracy slices (both models, per hour)          hours_in_scope()   choose the hours the question is about
        -> day-level bootstrap of the error gap      compare_models()   who is better, and can we tell?
        -> per-slice verdicts + a switching policy   slice_rows(), plan_policy()
        -> each model's worst hour + its top drivers worst_hours()      (from the cached explanations)
        -> labelled evidence blocks [E:...]          evidence_blocks()
        -> (RAG sources for background)              retrieve.search()
        -> writer LLM: the REASONING only            recommend()
        -> checks.check_text + one retry             same checker as the explanations

Design rule, the same as everywhere in this project: CODE DECIDES, THE LLM EXPLAINS. Which model wins, how sure we are, and
where to switch are computed here, with a statistical test, before any LLM is called. The writer receives the decision as a
fixed line and may only give the reasons for it; it cannot turn a weak lean into a recommendation, and the app shows the
code-made headline above its text. An LLM asked to "pick the best model" from a table would happily call a one-day
difference decisive.

What "can we tell?" means: errors of neighbouring hours are strongly related (a bad day is bad all day), so treating
4,416 hours as 4,416 independent trials would be far too confident. compare_models resamples whole DAYS instead.
"""

import hashlib
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

from forecast_explainer import PROJECT_ROOT, accuracy as ac, checks, evidence as ev, llm, registry, store
from forecast_explainer.narrate import retry_message
from forecast_explainer.rag import MAX_DISTANCE
from forecast_explainer.rag.queries import SPECIAL_DAYS
from forecast_explainer.rag.retrieve import format_context, label_hits, search

PROMPT_VERSION = "p3"  # bump when the planner's prompt or analysis changes, so cached recommendations are not reused
RECOMMENDATIONS_DIR = PROJECT_ROOT / "outputs" / "recommendations"

# --- statistical settings. Each is a judgement call; the values and the reasons are written down so they can be argued with.
MIN_DAYS = 5          # fewer distinct days than this cannot be separated from luck (a bootstrap over 1-4 days is meaningless)
MIN_EFFECT = 3.0      # percent. The leaning model's RMSE must be at least this much lower (relative); a smaller gap is not worth switching for
BOOTSTRAP_DRAWS = 2000
SEED = 0              # fixed, so the same question always gives the same interval

# The slice labels come from accuracy._slice_labels; they are listed here in reading order (and double as the `focus` choices).
DAY_TYPES = ["major holiday", "holiday", "weekend", "weekday"]
PARTS_OF_DAY = ["night 00-05", "morning 06-11", "afternoon 12-17", "evening 18-23"]
FOCUS_CHOICES = DAY_TYPES + PARTS_OF_DAY


# ----------------------------------------------------------------------------- 1. which hours is the question about?

def hours_in_scope(start: str | None = None, end: str | None = None, focus: str | None = None) -> pd.DataFrame:
    """The test-period hours a question is about, with each model's error per hour.

    start, end: first and last DAY to include (inclusive, "YYYY-MM-DD"); default = the whole test period.
    focus:      keep only one kind of hour: a day type ("major holiday", "holiday", "weekend", "weekday") or a part of
                the day ("night 00-05", "morning 06-11", "afternoon 12-17", "evening 18-23").

    Columns: y_true, one forecast column per model, calendar columns, `day_type`, `part_of_day`, `day`, and for each
    model `err_<m>` (signed, % of actual: positive = over-forecast, used for bias), `ae_<m>` (size of the miss, kWh)
    and `se_<m>` (the miss squared, kWh^2: the quantity RMSE averages).
    The requested period is kept in `frame.attrs["period"]` for the description of the scope.

    Raises ValueError (turned into a readable tool error for the agent) for dates outside the models' test period,
    start after end, an unknown focus, or a selection with no hours.
    """
    frame = ac.prediction_frame()
    first, last = frame.index.min().normalize(), frame.index.max().normalize()
    lo = pd.Timestamp(start).normalize() if start else first
    hi = pd.Timestamp(end).normalize() if end else last
    if lo > hi:
        raise ValueError(f"start {lo:%Y-%m-%d} is after end {hi:%Y-%m-%d}.")
    if lo < first or hi > last:
        raise ValueError(f"Only days between {first:%Y-%m-%d} and {last:%Y-%m-%d} can be compared (the models' test period).")
    frame = frame[(frame.index >= lo) & (frame.index < hi + pd.Timedelta(days=1))].copy()

    frame["day_type"] = ac._slice_labels(frame, "day_type")
    frame["part_of_day"] = ac._slice_labels(frame, "part_of_day")
    frame["day"] = frame.index.normalize()
    if focus:
        if focus in DAY_TYPES:
            frame = frame[frame["day_type"] == focus]
        elif focus in PARTS_OF_DAY:
            frame = frame[frame["part_of_day"] == focus]
        else:
            raise ValueError(f"unknown focus {focus!r}; choose from {FOCUS_CHOICES}.")
    if frame.empty:
        raise ValueError("No hours match that period and focus (for example, no public holiday falls in it).")

    for m in ev.ALL_MODELS:
        frame[f"err_{m}"] = 100 * (frame[m] - frame["y_true"]) / frame["y_true"]
        frame[f"ae_{m}"] = (frame[m] - frame["y_true"]).abs()
        frame[f"se_{m}"] = (frame[m] - frame["y_true"]) ** 2
    frame.attrs["period"] = (lo, hi)
    return frame


# ----------------------------------------------------------------------------- 2. who is better, and can we tell?

def compare_models(frame: pd.DataFrame) -> dict:
    """Compare the two models on a set of hours. Returns the numbers AND a verdict.

    The headline number is RMSE (root mean squared error, in kWh): square every hour's miss, average, take the root.
    Squaring makes one big miss count far more than several small ones, which suits a grid operator (a 25,000 kWh miss
    is much worse than five of 5,000), and it keeps the unit of the load. The question is whether the gap is real.
    Method:
      1. The "leaning" model is the one with the lower RMSE. Its gain = how much lower its RMSE is than the other's, in
         percent of the other's: 100 * (1 - RMSE_leaning / RMSE_other). Positive by construction; the question is how
         reliable it is.
      2. Resample whole DAYS with replacement 2,000 times and recompute the gain each time (RMSE of a resample =
         root of the summed squared misses over its hours). The 2.5th and 97.5th percentiles give a 95% interval.
         Days, not hours, because the hours of one day share the same causes (a holiday, a cold snap), so hour-level
         resampling would make the interval far too narrow.
      3. Verdict:
           clear                the whole interval is above 0 AND the gain is at least MIN_EFFECT percent;
           too_few_days         fewer than MIN_DAYS distinct days: no interval is computed, the numbers are only descriptive;
           no_clear_difference  otherwise: the gap could be luck.

    Returns: n_hours, n_days, rmse (kWh) / bias_pct / wins_pct per model, leaning, gain_pct, gain_ci ([low, high] or None),
    confidence, winner (the leaning model if confidence == "clear", else None).
    bias_pct is the mean signed error (positive = the model runs high, % of actual); wins_pct is the share of hours where
    that model had the strictly smaller miss in kWh.
    """
    models = ev.ALL_MODELS
    rmse = {m: float(np.sqrt(frame[f"se_{m}"].mean())) for m in models}
    bias = {m: float(frame[f"err_{m}"].mean()) for m in models}
    wins = {m: float(100 * (frame[f"ae_{m}"] < frame[f"ae_{[o for o in models if o != m][0]}"]).mean()) for m in models}
    leaning = min(rmse, key=rmse.get)
    other = next(m for m in models if m != leaning)

    gain_pct = 100 * (1 - rmse[leaning] / rmse[other])
    per_day = frame.groupby("day")[[f"se_{leaning}", f"se_{other}"]].sum()     # squared misses added up per day
    n_days = len(per_day)

    ci = None
    if n_days >= MIN_DAYS:
        rng = np.random.default_rng(SEED)
        picks = rng.integers(0, n_days, size=(BOOTSTRAP_DRAWS, n_days))          # which days each resample contains
        lean_sum = per_day[f"se_{leaning}"].to_numpy()[picks].sum(axis=1)
        other_sum = per_day[f"se_{other}"].to_numpy()[picks].sum(axis=1)
        resampled = 100 * (1 - np.sqrt(lean_sum / other_sum))                    # the gain of each resample (hour counts cancel)
        ci = [float(x) for x in np.percentile(resampled, [2.5, 97.5])]

    if n_days < MIN_DAYS:
        confidence = "too_few_days"
    elif ci[0] > 0 and gain_pct >= MIN_EFFECT:
        confidence = "clear"
    else:
        confidence = "no_clear_difference"
    return {"n_hours": int(len(frame)), "n_days": int(n_days), "rmse": rmse, "bias_pct": bias, "wins_pct": wins,
            "leaning": leaning, "gain_pct": float(gain_pct), "gain_ci": ci, "confidence": confidence,
            "winner": leaning if confidence == "clear" else None}


def slice_rows(frame: pd.DataFrame) -> list[dict]:
    """compare_models for each day type and each part of the day that has hours in the frame.

    Where the models differ is the planner's real output: a model that is better overall can still be worse on
    holidays or in the evening. Each row carries `by` ("day_type" or "part_of_day") and `label`.
    """
    rows = []
    for by, labels in (("day_type", DAY_TYPES), ("part_of_day", PARTS_OF_DAY)):
        for label in labels:
            part = frame[frame[by] == label]
            if len(part):
                rows.append({"by": by, "label": label, **compare_models(part)})
    return rows


def plan_policy(overall: dict, rows: list[dict]) -> dict:
    """A switching policy: a default model, and the slices where another model is better.

    default:          the overall leaning model, with how sure we are (`default_confidence`).
    switch_to:        slices where the OTHER model is clearly better. These are reliable.
    possible_switch:  slices where the other model had the lower error but there are too few days to rely on it
                      (for example 25 December: one day). They are listed so they are not forgotten, labelled as unreliable.
    Rules apply per slice and are not combined: an evening on a weekday matches two slices, and the policy does not say
    which wins. When the scope itself covers fewer than MIN_DAYS days there is no meaningful default, so no switches are listed.
    """
    default = overall["leaning"]
    broad = overall["n_days"] >= MIN_DAYS   # a one-week scope has a default worth switching away from; one day does not

    def brief(r):
        return {"by": r["by"], "label": r["label"], "model": r["leaning"], "n_days": r["n_days"], "n_hours": r["n_hours"],
                "rmse": r["rmse"], "gain_pct": r["gain_pct"]}

    return {"default": default, "default_confidence": overall["confidence"],
            "switch_to": [brief(r) for r in rows if broad and r["confidence"] == "clear" and r["winner"] != default],
            "possible_switch": [brief(r) for r in rows if broad and r["confidence"] == "too_few_days" and r["leaning"] != default]}


# ----------------------------------------------------------------------------- 3. what went wrong at each model's worst hour?

def worst_hours(frame: pd.DataFrame, with_explanations: bool = True) -> dict:
    """Each model's single worst hour in the scope, and (optionally) the two inputs it leaned on most there.

    This is how the EXPLANATIONS enter the planning: a model that is wrong on a holiday because it relied on last week's
    load is a different risk from one that was wrong for no visible reason. The drivers come from SHAP (LightGBM) and
    Integrated Gradients (LSTM) via the explanation cache; a hour not yet explained is computed once (about 6 s).
    with_explanations=False skips that (fast, used in tests). If an explanation cannot be produced, the drivers are
    left out and the reason is recorded; the planner still answers.
    """
    out = {}
    for m in ev.ALL_MODELS:
        ts = frame[f"ae_{m}"].idxmax()            # the largest miss in kWh, the same unit RMSE is built from
        row = frame.loc[ts]
        entry = {"timestamp": str(ts), "actual_kwh": float(row["y_true"]), "forecast_kwh": float(row[m]),
                 "error_kwh": float(row[m] - row["y_true"]), "error_pct": float(row[f"err_{m}"]), "drivers": None}
        if with_explanations:
            try:
                explanation = store.get_explanation(ts, verbose=False)
                records = explanation["lightgbm"]["shap"] if m == "lightgbm" else explanation["lstm"]["ig_by_feature"]
                top = sorted(records, key=lambda r: abs(r["attribution_kwh"]), reverse=True)[:2]
                entry["drivers"] = [{"feature": r["feature"], "kwh": float(r["attribution_kwh"])}
                                    for r in top if round(r["attribution_kwh"]) != 0]
            except Exception as exc:   # a failure here must not take the whole recommendation down
                entry["drivers_error"] = f"{type(exc).__name__}: {exc}"
        out[m] = entry
    return out


# ----------------------------------------------------------------------------- 4. the decision (made by code)

def describe_scope(frame: pd.DataFrame, focus: str | None) -> str:
    """The scope in words: "all hours, 01 Jul to 31 Dec 2014" or "major holiday hours, 01 Jul to 31 Dec 2014"."""
    lo, hi = frame.attrs["period"]
    period = f"{lo:%d %b %Y}" if lo == hi else f"{lo:%d %b} to {hi:%d %b %Y}"
    return f"{focus} hours, {period}" if focus else f"all hours, {period}"


def make_decision(scope: str, overall: dict, policy: dict) -> dict:
    """The one-line decision. Written by code from the verdict, so the LLM can neither change nor soften it.

    Returns: model (the winner, or None when there is none), leaning (the lower-RMSE model), confidence, headline, and
    `switches`: one sentence per slice in the policy ("Switch to the LSTM for ...", or "Possibly ... not reliable").
    """
    lean, other = overall["leaning"], next(m for m in ev.ALL_MODELS if m != overall["leaning"])
    a, b = overall["rmse"][lean], overall["rmse"][other]
    lean_name = ev.LABELS[lean]
    if overall["confidence"] == "clear":
        headline = f"Use {lean_name} for {scope}: its error is lower (RMSE {a:,.0f} vs {b:,.0f} kWh)."
    elif overall["confidence"] == "too_few_days":
        n = overall["n_days"]
        headline = (f"Too few days to separate the models for {scope} ({_days(n)}): {lean_name} had the "
                    f"lower error (RMSE {a:,.0f} vs {b:,.0f} kWh), which is not enough to rely on.")
    else:
        headline = (f"No clear winner for {scope}: {lean_name} looks slightly better (RMSE {a:,.0f} vs {b:,.0f} kWh), "
                    f"but the difference is within the noise.")

    switches = []
    for s in policy["switch_to"]:
        o = next(m for m in ev.ALL_MODELS if m != s["model"])
        switches.append(f"Switch to {ev.LABELS[s['model']]} for {s['label']} hours (RMSE {s['rmse'][s['model']]:,.0f} vs "
                        f"{s['rmse'][o]:,.0f} kWh, {_days(s['n_days'])}).")
    for s in policy["possible_switch"]:
        o = next(m for m in ev.ALL_MODELS if m != s["model"])
        switches.append(f"Possibly {ev.LABELS[s['model']]} for {s['label']} hours (RMSE {s['rmse'][s['model']]:,.0f} vs "
                        f"{s['rmse'][o]:,.0f} kWh), but only {_days(s['n_days'])}: not reliable.")
    return {"model": overall["winner"], "leaning": lean, "confidence": overall["confidence"], "headline": headline,
            "switches": switches}


def analyse(start: str | None = None, end: str | None = None, focus: str | None = None, with_explanations: bool = True) -> dict:
    """The complete deterministic analysis for one question. No LLM, no network; everything is JSON-serialisable.

    Returns: scope {start, end, focus, description, n_hours, n_days}, overall, slices, policy, worst, decision.
    """
    frame = hours_in_scope(start, end, focus)
    overall = compare_models(frame)
    rows = slice_rows(frame)
    policy = plan_policy(overall, rows)
    lo, hi = frame.attrs["period"]
    description = describe_scope(frame, focus)
    return {"scope": {"start": f"{lo:%Y-%m-%d}", "end": f"{hi:%Y-%m-%d}", "focus": focus, "description": description,
                      "n_hours": overall["n_hours"], "n_days": overall["n_days"]},
            "overall": overall, "slices": rows, "policy": policy,
            "worst": worst_hours(frame, with_explanations), "decision": make_decision(description, overall, policy)}


# ----------------------------------------------------------------------------- 5. evidence the writer may use

def _days(n: int) -> str:
    """"1 day", "5 days": the plural matters when the whole point is that a group has too few of them."""
    return f"{n} day{'s' if n != 1 else ''}"


def _verdict_text(r: dict) -> str:
    """One slice's verdict in words, with the numbers the writer is allowed to quote."""
    lean, name = r["leaning"], ev.LABELS[r["leaning"]]
    if r["confidence"] == "clear":
        return f"{name} is clearly better (RMSE lower by {r['gain_pct']:.1f}%)"
    if r["confidence"] == "too_few_days":
        return f"too few days to tell ({_days(r['n_days'])}); {name} had the lower error but this is not reliable"
    return f"no clear difference ({name} slightly lower by {r['gain_pct']:.1f}%, within the noise)"


def evidence_blocks(analysis: dict) -> dict[str, str]:
    """The analysis as labelled text blocks the writer may cite: E:scope, E:overall, E:slices, E:policy, E:worst, E:limits.

    Same idea as evidence.build_evidence: every number the writer may state is computed and formatted here, so the
    checker can verify the text against these blocks and the writer never does arithmetic.
    """
    s, o, pol = analysis["scope"], analysis["overall"], analysis["policy"]
    lean = o["leaning"]
    L = ev.LABELS

    scope = (f"Question scope: {s['description']}. {s['n_hours']:,} hours on {_days(s['n_days'])} of the test period "
             f"(both models were trained on earlier data).")

    lines = [f"RMSE (typical size of the miss, big misses weigh more): " + ", ".join(f"{L[m]} {o['rmse'][m]:,.0f} kWh" for m in ev.ALL_MODELS) + ".",
             "Bias (positive = the model runs high): " + ", ".join(f"{L[m]} {o['bias_pct'][m]:+.1f}%" for m in ev.ALL_MODELS) + ".",
             "Share of hours with the smaller miss: " + ", ".join(f"{L[m]} {o['wins_pct'][m]:.0f}%" for m in ev.ALL_MODELS) + "."]
    if o["gain_ci"]:
        lines.append(f"{L[lean]}'s RMSE is lower by {o['gain_pct']:.1f}% "
                     f"(95% interval {o['gain_ci'][0]:.1f}% to {o['gain_ci'][1]:.1f}%, resampling {o['n_days']} days).")
    else:
        lines.append(f"{L[lean]}'s RMSE is lower by {o['gain_pct']:.1f}%, but only "
                     f"{_days(o['n_days'])} {'is' if o['n_days'] == 1 else 'are'} covered, so no interval can be given.")
    lines.append("Verdict: " + (f"{L[o['winner']]} is clearly better." if o["winner"] else _verdict_text(o) + "."))
    overall = "\n".join(lines)

    slice_lines = [f"- {r['label']} ({r['n_hours']:,} hours, {_days(r['n_days'])}): "
                   + ", ".join(f"{L[m]} {r['rmse'][m]:,.0f}" for m in ev.ALL_MODELS) + " kWh RMSE; " + _verdict_text(r) + "."
                   for r in analysis["slices"]]
    slices = "\n".join(["RMSE by kind of hour:", *slice_lines])

    if s["n_days"] < MIN_DAYS:
        policy = [f"No switching policy: the scope covers {_days(s['n_days'])}, fewer than the {MIN_DAYS} needed to compare groups."]
    else:
        policy = ["Switching policy (rules apply per slice and are not combined):",
                  f"- Default: {L[pol['default']]} ({'clear' if pol['default_confidence'] == 'clear' else 'not a clear difference' if pol['default_confidence'] == 'no_clear_difference' else 'too few days'})."]
        policy += [f"- {x}" for x in analysis["decision"]["switches"]] or ["- No slice has a different clear winner."]

    worst = ["Each model's worst hour in this scope:"]
    for m in ev.ALL_MODELS:
        w = analysis["worst"][m]
        line = (f"- {L[m]}: {pd.Timestamp(w['timestamp']):%a %Y-%m-%d %H:%M}, actual {w['actual_kwh']:,.0f} kWh, forecast "
                f"{w['forecast_kwh']:,.0f} kWh (error {w['error_kwh']:+,.0f} kWh, {w['error_pct']:+.1f}%).")
        if w.get("drivers"):
            line += " Inputs it relied on most: " + ", ".join(f"{d['feature']} ({d['kwh']:+,.0f} kWh)" for d in w["drivers"]) + "."
        worst.append(line)

    limits = (f"The comparison covers one test period (July to December 2014, one Christmas). Intervals come from resampling whole "
              f"days {BOOTSTRAP_DRAWS:,} times. A group needs at least {MIN_DAYS} days to be compared, and an RMSE gap smaller "
              f"than {MIN_EFFECT:g}% counts as no difference. The models have no weather input.")
    return {"E:scope": scope, "E:overall": overall, "E:slices": slices, "E:policy": "\n".join(policy),
            "E:worst": "\n".join(worst), "E:limits": limits}


def planner_queries(analysis: dict) -> list[dict]:
    """Queries for the knowledge base, rule-based like rag.queries (no LLM): model comparison always, calendar background when relevant.

    Special days (24/25/31 December, 1 January) are searched only when the scope is short (14 days or less); in a
    half-year scope they would add noise to every question.
    """
    queries = [{"query": "LightGBM and LSTM accuracy comparison where the two models differ", "why": "comparing the two models"},
               {"query": "which inputs each model relies on: recent load versus holiday calendar features", "why": "how the models use their inputs"}]
    s = analysis["scope"]
    if s["n_days"] <= 14:
        for day in pd.date_range(s["start"], s["end"]):
            if (day.month, day.day) in SPECIAL_DAYS:
                text, why = SPECIAL_DAYS[(day.month, day.day)]
                queries.append({"query": text, "why": why})
    if s["focus"] in ("major holiday", "holiday"):
        queries.append({"query": "ordinary Portuguese public holiday effect on electricity load", "why": "focus is holiday hours"})
    return queries


# ----------------------------------------------------------------------------- 6. the writer

# Words a writer uses to JUDGE a sample ("a substantial dataset", "a reasonable sample size"). Nothing in the evidence supports such a
# judgement (the thresholds are in [E:limits]), the checker cannot see it, and the writer kept doing it despite the prompt rule, so
# it is caught here and sent back once. This is a partial guard, not a proof: a writer can use a synonym that is not listed, and
# overclaims such as "consistently lower across hour types" are invisible to code. The manual review sheet is for those.
# "insufficient" is allowed: it is only used when the verdict itself says too few days.
JUDGING = re.compile(r"\b(substantial|reasonable|robust|sufficient|adequate|ample|solid|sizeable|decent|healthy|large enough)\b", re.I)

MAX_WORDS = 110       # the prompt asks for 80; a text over this is sent back once, because the writer ignores soft limits

SYSTEM = """You advise an energy analyst which of two day-ahead electricity-load forecasting models to use: LightGBM (gradient-boosted \
trees) or an LSTM (a neural sequence model).

The choice has ALREADY been made by code and is given to you as DECISION. Your job is to explain the reasoning, not to decide. You are \
given EVIDENCE blocks labelled [E:...] and SOURCES with background labelled [S1], [S2], ...

Rules:
1. Use only the EVIDENCE and SOURCES. Do not use outside knowledge. The models have no weather input.
2. Never contradict DECISION. If it says there is no clear winner, or too few days, say so plainly. Never turn a weak lean into a \
recommendation, and never call a group with too few days reliable.
3. Cite every sentence that states a fact, with the ID in square brackets right after it, one ID per bracket, for example: \
"... [E:overall] [E:slices]". Never cite an ID that was not provided.
4. Numbers: copy them exactly as written in the EVIDENCE or SOURCES, in the same unit. You may round the way a person would \
(2.46 -> 2.5). Never calculate a new number (no sums, differences or ratios).
5. Make statements of fact, not claims of cause. Do not use the words because, due to, owing to, caused, reason. Say "the evidence \
covers only 1 day", not "unreliable due to the short period". A real-world reason (why a model does better on holidays) may only be \
given if a SOURCE states it, cited as "according to [S#]". Inputs listed in [E:worst] describe how the MODEL behaved, not what \
happened in the world. Do not judge whether a sample is large enough or "reasonable"; quote [E:limits] instead.
6. Do not repeat the DECISION or its switching advice: the reader sees them above your text.

Format: Markdown, at most 80 words in total, no preamble.
**Why** - 2 or 3 short bullets with the strongest evidence, each with a citation.
**Caution** - at most 2 short bullets: sample size, the length of the test period, anything in [E:limits]."""


def build_prompt(question: str | None, decision: dict, blocks: dict[str, str], hits: list[dict]) -> str:
    """The user message: the question, the fixed decision, the evidence blocks and the sources."""
    sources = format_context(hits) if hits else "(no sources provided; do not cite any [S...] ID)"
    return "\n\n".join([
        f"QUESTION: {question or 'Which model should I use?'}",
        "DECISION (made by code; do not change or repeat it):\n" + " ".join([decision["headline"], *decision["switches"]]),
        "EVIDENCE:\n" + ev.render_evidence(blocks),
        "SOURCES (background; cite as [S#]):\n" + sources,
        "Write the reasoning now.",
    ])


def recommendation_path(key: dict):
    """Cache file for one question: the hash covers everything that changes the answer (scope, wording, LLM, prompt, models)."""
    digest = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return RECOMMENDATIONS_DIR / f"recommendation_{digest}.json"


def recommend(start: str | None = None, end: str | None = None, focus: str | None = None, question: str | None = None,
              model: str | None = None, use_rag: bool = True, with_explanations: bool = True, refresh: bool = False,
              max_retries: int = 1, llm_client=None, temperature: float = 0.0) -> dict:
    """Recommend a model for a period and/or kind of hour, with cited reasoning that has been checked.

    start, end:        first and last day to compare (inclusive); default the whole test period.
    focus:             one of FOCUS_CHOICES, to compare on one kind of hour only.
    question:          the user's own words, so the writer can answer what was asked.
    model:             LLM for the writer (default LLM_MODEL). The decision does not depend on it.
    use_rag:           False leaves out the retrieved sources (the "no RAG" arm of a comparison).
    with_explanations: include each model's worst hour with its top drivers (needs the explanation cache; ~6 s per new hour).
    refresh:           True ignores a cached recommendation.
    max_retries:       how many times the writer may try again after failing the checks.
    llm_client:        a fake client for tests.

    Returns the analysis plus: text, checks, attempts, evidence, sources, queries, usage, model, source ("llm" or "cache").
    """
    model = model or llm.default_model()
    key = {"start": start, "end": end, "focus": focus, "question": question, "model": model, "rag": use_rag,
           "explanations": with_explanations, "prompt": PROMPT_VERSION, "fingerprint": registry.fingerprint("total")}
    path = recommendation_path(key)
    if path.exists() and not refresh:
        return {**json.loads(path.read_text()), "source": "cache"}

    analysis = analyse(start, end, focus, with_explanations)
    blocks = evidence_blocks(analysis)
    queries = planner_queries(analysis) if use_rag else []
    hits = label_hits(search([q["query"] for q in queries], k=3, max_total=6, max_distance=MAX_DISTANCE)) if use_rag else []
    prompt = build_prompt(question, analysis["decision"], blocks, hits)

    attempts, user, tokens = [], prompt, [0, 0]
    for _ in range(max_retries + 1):
        result = llm.complete(SYSTEM, user, model=model, temperature=temperature, client=llm_client)
        verdict = checks.check_text(result.text, blocks, hits)
        words = len(result.text.split())
        verdict["words"] = words                       # length is reported, and triggers a retry, but is not part of `passed`
        attempts.append({"text": result.text, "checks": verdict})
        tokens[0] += result.prompt_tokens or 0
        tokens[1] += result.completion_tokens or 0
        too_long = words > MAX_WORDS
        judging = sorted({w.lower() for w in JUDGING.findall(result.text)})
        verdict["judging_words"] = judging
        if verdict["passed"] and not too_long and not judging:
            break
        feedback = retry_message(verdict) if not verdict["passed"] else ""
        if judging:
            feedback += (f"\nDo not judge sample sizes with words such as {', '.join(judging)}. State the number of days and, if needed, "
                         "quote [E:limits]; do not say whether it is enough.")
        if too_long:
            feedback += (f"\nYour text has {words} words; the limit is 80. Remove the least important bullet or shorten the wording, "
                         "and keep every citation.")
        user = prompt + "\n\n" + f"PREVIOUS ANSWER:\n{result.text}\n\n" + feedback

    final = attempts[-1]
    output = {**analysis, "kind": "recommendation", "question": question, "text": final["text"], "checks": final["checks"],
              "n_attempts": len(attempts), "attempts": attempts, "evidence": blocks,
              "sources": [{k: h[k] for k in ("sid", "citation", "page", "section", "distance", "text")} for h in hits],
              "queries": queries, "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1]},
              "model": result.model, "requested_model": model, "prompt_version": PROMPT_VERSION, "use_rag": use_rag,
              "created_at": datetime.now().isoformat(timespec="seconds"), "model_fingerprint": key["fingerprint"]}
    RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, default=str))
    return {**output, "source": "llm"}
