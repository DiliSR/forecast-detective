"""Build retrieval queries from an explanation's evidence - no LLM involved.

Why not just let the LLM write queries? Rule-based queries are deterministic (same explanation -> same
queries -> same sources), cheap, easy to test, and every query carries a `why`, so we can see exactly
what in the evidence triggered which search. The LLM step later receives the retrieved chunks, not the
queries.

Input is the explanation dict from store.get_explanation (including its "disagreements" flags).
Output is a list of {"query": text, "why": reason}.
"""

import pandas as pd

from forecast_explainer.evidence import normalize_models, restrict_to_models

# Special days whose load pattern the models do not know. Keys are (month, day). The query text repeats
# the vocabulary of knowledge/notes/calendar_effects.md so the embedding lands near that note.
SPECIAL_DAYS = {
    (12, 24): ("Christmas Eve 24 December evening load drops although it is not a public holiday",
               "target hour is on 24 December"),
    (12, 25): ("Christmas Day 25 December major holiday whole-day load falls by about 40 percent",
               "target hour is on 25 December"),
    (12, 31): ("New Year's Eve 31 December late evening load drop",
               "target hour is on 31 December"),
    (1, 1): ("New Year's Day 1 January holiday load lower than a normal day",
             "target hour is on 1 January"),
}


def _is_dst_change_night(ts: pd.Timestamp) -> bool:
    """True for the small hours of the last Sunday of March or October (the clock changes)."""
    return ts.month in (3, 10) and ts.dayofweek == 6 and ts.day > 24 and ts.hour <= 4


def _flag_not_ok(flags: list[dict], type_: str) -> bool:
    """Did the disagreement check `type_` raise anything (severity other than "ok")?"""
    return any(f["type"] == type_ and f["severity"] != "ok" for f in flags)


def queries_from_evidence(explanation: dict, models="both") -> list[dict]:
    """Queries to run against the knowledge base for this explanation.

    models: which model(s) are being explained. Flags about the other model are ignored, and the standing
            "how to read" query is only added for explainers that are actually shown.

    Three groups, in order:
      1. the date: special days and clock changes (domain notes);
      2. the disagreement flags that fired: how to interpret / why methods differ (papers + explainer guide);
      3. always: how to read SHAP and the LSTM baseline, because every explanation shows those tables.
    """
    models = normalize_models(models)
    ts = pd.Timestamp(explanation["timestamp"])
    flags = restrict_to_models(explanation, models).get("disagreements", [])
    queries = []

    # 1. the date
    if (ts.month, ts.day) in SPECIAL_DAYS:
        text, why = SPECIAL_DAYS[(ts.month, ts.day)]
        queries.append({"query": text, "why": why})
    elif any(r["feature"] == "is_holiday" and r["value"] == 1 for r in explanation["lightgbm"]["shap"]):
        queries.append({"query": "ordinary Portuguese public holiday effect on electricity load",
                        "why": "target hour is a public holiday"})
    if _is_dst_change_night(ts):
        queries.append({"query": "daylight saving clock change repair of the missing or duplicated hour in the data",
                        "why": "target hour is on a clock-change night"})

    # 2. the flags that fired
    if _flag_not_ok(flags, "shared_error"):
        queries.append({"query": "both models forecast too high or too low because of missing calendar information",
                        "why": "flag shared_error: both models miss in the same direction"})
    if _flag_not_ok(flags, "model_gap"):
        queries.append({"query": "LightGBM and LSTM accuracy comparison where the two models differ",
                        "why": "flag model_gap: the models forecast different values"})
    if _flag_not_ok(flags, "shap_vs_lime"):
        queries.append({"query": "why SHAP and LIME give different feature attributions: different reference points and quantities",
                        "why": "flag shap_vs_lime"})
    if _flag_not_ok(flags, "lime_fit") or _flag_not_ok(flags, "lime_instability"):
        queries.append({"query": "LIME local linear surrogate fidelity low R-squared and instability from random sampling",
                        "why": "flag lime_fit / lime_instability"})
    if _flag_not_ok(flags, "ig_vs_timeshap") or _flag_not_ok(flags, "ig_convergence"):
        queries.append({"query": "Integrated Gradients and TimeSHAP baseline input, completeness and convergence",
                        "why": "flag ig_vs_timeshap / ig_convergence"})
    if _flag_not_ok(flags, "cross_model_driver"):
        queries.append({"query": "which inputs each model relies on: recent load versus holiday calendar features",
                        "why": "flag cross_model_driver"})

    # 3. always
    if "lightgbm" in models:
        queries.append({"query": "SHAP values add up to the forecast minus the expected value, the average model output",
                        "why": "always: how to read the SHAP table"})
    if "lstm" in models:
        queries.append({"query": "baseline for the LSTM explanations: a typical non-holiday weekday at the same hour and season",
                        "why": "always: how to read the IG / TimeSHAP tables"})
    return queries
