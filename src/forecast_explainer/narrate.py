"""Write the natural-language explanation of one forecast, with citations, and check it.

Pipeline for one target hour:

    store.get_explanation          numbers + disagreement flags   (cached; seconds the first time)
        -> evidence.build_evidence      compact labelled blocks [E:forecast] [E:shap] ...
        -> rag.retrieve_for_explanation cited background [S1] [S2] ...           (optional: use_rag=False for the ablation)
        -> llm.complete                 the explanation
        -> checks.check_text            citation + number check; ONE retry if it fails, with the problems fed back
        -> save to outputs/narratives/  so the same question later is instant and identical

The LLM is the only non-deterministic step, so everything before it is cached or deterministic, and everything
after it is a deterministic check. The result dict keeps all inputs (evidence, sources, queries) next to the text:
the judge and the app need to see exactly what the writer saw.
"""

import hashlib
import json
import re
from datetime import datetime

import pandas as pd

from forecast_explainer import PROJECT_ROOT, checks, evidence as ev, llm, registry, store
from forecast_explainer.rag.retrieve import format_context, retrieve_for_explanation

NARRATIVES_DIR = PROJECT_ROOT / "outputs" / "narratives"
PROMPT_VERSION = "v6"  # bump when the system prompt changes, so cached narratives from the old prompt are not reused

MODEL_INTRO = {
    "lightgbm": "LightGBM (explained with SHAP and LIME)",
    "lstm": "an LSTM (explained with Integrated Gradients and TimeSHAP)",
}


def system_prompt(models="both") -> str:
    """The writer's standing rules, adapted to the model(s) being explained.

    Only the first paragraph and the "each model" wording depend on `models`; the rules are identical, so a
    single-model explanation is held to exactly the same standard as a two-model one.
    """
    models = ev.normalize_models(models)
    if len(models) == 2:
        intro = f"Two models forecast the hourly load: {MODEL_INTRO['lightgbm']} and {MODEL_INTRO['lstm']}."
        drivers = "at most 2 bullets for each model"
    else:
        intro = f"You explain the forecast of one model: {MODEL_INTRO[models[0]]}. Do not discuss any other model."
        drivers = "at most 3 bullets"
    # Wording rule that only makes sense when LightGBM's SHAP and LIME blocks are in the evidence.
    tree_rule = ('\n8. In SHAP lines, "input value" is what the feature was (a load level, a flag); "contribution" is how much it moved the '
                 'forecast. Only the contribution says how much a feature mattered. Never call an input value a contribution. LIME numbers '
                 'are weights (slopes of a local line), not contributions: call them weights, and prefer SHAP for the drivers.'
                 if "lightgbm" in models else "")
    return _SYSTEM_TEMPLATE.replace("{INTRO}", intro).replace("{DRIVERS}", drivers).replace("{TREE_RULE}", tree_rule)


_SYSTEM_TEMPLATE = """You explain a day-ahead electricity-load forecast to an energy analyst. {INTRO}

You are given EVIDENCE blocks about this one forecast, labelled [E:...], and SOURCES with background, labelled [S1], [S2], ...

Rules:
1. Use only the EVIDENCE and SOURCES. Do not use outside knowledge, and do not invent causes. The models have no weather \
input. If the material does not answer something, say that it does not.
2. Cite every sentence that states a fact, with the ID in square brackets right after it, one ID per bracket, \
for example: "... [E:shap] [S2]". Never cite an ID that was not provided.
3. Numbers: copy them exactly as written in the EVIDENCE or SOURCES, in the same unit (kWh, %). You may round the way a \
person would (14.6% -> 15%). Never convert units, and never calculate a new number (no sums, differences or ratios).
4. Mention every disagreement listed in [E:flags] whose severity is HIGH or WARNING, in one short line each saying what it \
means for how far the reader can trust the explanation, and cite [E:flags]. Use SOURCES to explain why methods can disagree.
5. Attributions describe how the MODEL used its inputs, not what happened in the real world. Say "the model relied on", \
not "the load fell because". A real-world reason may only be given if a SOURCE states it, and then cited and phrased as \
"according to [S#]".
6. Different methods use different reference points (see the evidence). Do not compare their numbers as if they were \
the same quantity.
7. Put the citation for the forecast numbers ([E:forecast]) in the same sentence or paragraph that quotes them.{TREE_RULE}

Format: very short, Markdown, at most 90 words in total, no preamble. The reader already sees the forecast numbers in a chart, so \
do not list them again unless they are the point.
**Summary** - one or two sentences: how far off each forecast was and the main reason the model gave.
**Drivers** - {DRIVERS}, one short line each: the input, its contribution in kWh, its citation. Do not nest bullets.
**Caution** - at most 2 bullets, merging related flags from [E:flags]; leave this section out if there are no HIGH or WARNING flags.
If the user asks a question, answer it in the Summary."""

SYSTEM = system_prompt("both")  # kept for notebooks that print the standard prompt


def build_prompt(blocks: dict[str, str], hits: list[dict], question: str | None) -> str:
    """The user message: the question, the evidence blocks, the sources.

    blocks: {"E:forecast": text, ...} from evidence.build_evidence.
    hits:   labelled retrieved chunks (empty when RAG is switched off).
    """
    sources = format_context(hits) if hits else "(no sources provided; do not cite any [S...] ID)"
    return "\n\n".join([
        f"QUESTION: {question or 'Explain this forecast.'}",
        "EVIDENCE (facts about this forecast):\n" + ev.render_evidence(blocks),
        "SOURCES (background; cite as [S#]):\n" + sources,
        "Write the explanation now.",
    ])


def retry_message(result: dict) -> str:
    """What to tell the writer when the checker rejected its answer: the specific problems, nothing else."""
    problems = []
    if result["unknown_citations"]:
        problems.append(f"You cited IDs that do not exist: {', '.join(result['unknown_citations'])}. Use only provided IDs.")
    for item in result["unsupported_numbers"][:6]:
        if item.get("found_in"):   # right number, wrong (or missing) citation: say so, or the writer cannot know what to fix
            where = ", ".join(f"[{i}]" for i in item["found_in"][:3])
            problems.append(f"The number {item['number']} is correct but not in the block(s) cited for it. It appears in {where}: "
                            f"cite that in the same sentence (sentence: \"{item['sentence']}\").")
        else:
            problems.append(f"The number {item['number']} is not in the EVIDENCE or SOURCES (sentence: \"{item['sentence']}\"). "
                            "Remove it or use a number as written there.")
    if result["flags_not_cited"]:
        problems.append("[E:flags] lists HIGH/WARNING disagreements but your text never cites [E:flags].")
    return ("Your previous answer had these problems:\n- " + "\n- ".join(problems)
            + "\nRewrite the explanation and fix ONLY these problems. Keep everything else, in particular do not leave out "
              "a driver or caveat just to avoid a number: if a number comes from a feature name such as load_lag_168h, "
              "write the feature name instead.")


def narrative_path(timestamp, client: str, model: str, use_rag: bool, question: str | None, models="both"):
    """Cache file for one (hour, LLM, prompt version, rag on/off, question, explained model(s)) combination."""
    which = "-".join(ev.normalize_models(models)) if ev.normalize_models(models) != ev.ALL_MODELS else "both"
    q = f"_q{hashlib.sha1(question.encode()).hexdigest()[:6]}" if question else ""
    name = f"{client.lower()}_{pd.Timestamp(timestamp):%Y-%m-%d_%H%M}_{re.sub(r'[^A-Za-z0-9.-]+', '-', model)}_{which}_{PROMPT_VERSION}_{'rag' if use_rag else 'norag'}{q}.json"
    return NARRATIVES_DIR / name


def narrate(timestamp, client: str = "total", question: str | None = None, model: str | None = None,
            models="both", use_rag: bool = True, refresh: bool = False, max_retries: int = 1, llm_client=None,
            top_n: int = 6, temperature: float = 0.0) -> dict:
    """Explain the forecast for one target hour in words, with citations, and verify it.

    timestamp:   the target hour ("2014-12-24 18:00"); must lie in the period the models can explain.
    client:      "total" or a client id.
    question:    optional question from the user ("why was it so high?"); the writer answers it first.
    model:       LLM to use (default: LLM_MODEL from .env).
    models:      which forecasting model(s) to explain: "both" (default), "lightgbm" or "lstm".
    use_rag:     False leaves out the retrieved SOURCES - the "no RAG" arm of an ablation.
    refresh:     True ignores a cached narrative and calls the LLM again.
    max_retries: how many times to ask again, with the checker's complaints, if the answer fails the checks.
    llm_client:  inject a fake client in tests.
    top_n:       rows per explainer table in the evidence pack.
    temperature: passed to the LLM; 0 for repeatable output.
    """
    model = model or llm.default_model()
    models = ev.normalize_models(models)
    path = narrative_path(timestamp, client, model, use_rag, question, models)
    if path.exists() and not refresh:
        return {**json.loads(path.read_text()), "source": "cache"}

    explanation = store.get_explanation(timestamp, client, verbose=False)
    blocks = ev.build_evidence(explanation, top_n, models)
    hits, queries = retrieve_for_explanation(explanation, models=models) if use_rag else ([], [])
    prompt = build_prompt(blocks, hits, question)

    attempts, messages_user, tokens = [], prompt, [0, 0]
    for attempt in range(max_retries + 1):
        result = llm.complete(system_prompt(models), messages_user, model=model, temperature=temperature, client=llm_client)
        verdict = checks.check_text(result.text, blocks, hits)
        attempts.append({"text": result.text, "checks": verdict})
        tokens[0] += result.prompt_tokens or 0
        tokens[1] += result.completion_tokens or 0
        if verdict["passed"]:
            break
        messages_user = prompt + "\n\n" + f"PREVIOUS ANSWER:\n{result.text}\n\n" + retry_message(verdict)

    final = attempts[-1]
    output = {
        "timestamp": str(pd.Timestamp(timestamp)), "client": client, "question": question, "models": list(models),
        "model": result.model, "requested_model": model, "prompt_version": PROMPT_VERSION, "use_rag": use_rag,
        "text": final["text"], "checks": final["checks"], "n_attempts": len(attempts), "attempts": attempts,
        "evidence": blocks,
        "sources": [{k: h[k] for k in ("sid", "citation", "page", "section", "distance", "text")} for h in hits],
        "queries": queries,
        "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1]},
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_fingerprint": registry.fingerprint(client),
    }
    NARRATIVES_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, default=str))
    return {**output, "source": "llm"}
