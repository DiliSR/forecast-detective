"""The evidence pack: everything the LLM may state about one forecast, as short labelled text blocks.

Why not give the LLM the raw explanation JSON? It is ~370 KB (mostly the 168x8 IG table), most of it irrelevant to
a reader, and an LLM handed raw tables tends to pick numbers at random or do its own arithmetic. The pack instead:

  - keeps only the top rows of each explainer;
  - pre-computes every derived number the explanation needs (errors in kWh and %, the gap between models), so the
    LLM never has to calculate - and any number in its answer that is NOT in the pack is a red flag (checks.py);
  - gives each block an ID the LLM must cite: [E:forecast], [E:shap], [E:lime], [E:ig], [E:timeshap],
    [E:flags], [E:accuracy]. A cited ID points to exactly one block, so a claim can be traced and verified.

Number format: kWh as whole numbers with thousands separators ("242,680 kWh"), percentages with one decimal,
signed where direction matters. checks.py accepts a quoted number if it matches a pack number at the precision quoted.
"""

import pandas as pd

from forecast_explainer import accuracy as ac
from forecast_explainer.features import calendar_features

SEVERITY_WORDS = {"high": "HIGH", "warning": "WARNING", "info": "info", "ok": "ok"}

ALL_MODELS = ("lightgbm", "lstm")
LABELS = {"lightgbm": "LightGBM", "lstm": "LSTM"}
# Which disagreement checks concern which model. Checks in neither set compare the two models with each
# other (model_gap, shared_error, cross_model_driver) and only make sense when both are being explained.
FLAGS_OF = {
    "lightgbm": {"shap_vs_lime", "lime_fit", "lime_instability"},
    "lstm": {"ig_vs_timeshap", "ig_vs_timeshap_timestep", "ig_convergence"},
}


def normalize_models(models) -> tuple[str, ...]:
    """Turn a model choice into a tuple in canonical order: ("lightgbm",), ("lstm",) or both.

    Accepts "both" / None (= both), a single key ("lstm"), or a list/tuple of keys. Anything else raises, so a typo
    from the user or the agent becomes a clear error instead of silently explaining the wrong model.
    """
    if models is None or models == "both":
        return ALL_MODELS
    wanted = [models] if isinstance(models, str) else list(models)
    unknown = [m for m in wanted if m not in ALL_MODELS]
    if unknown or not wanted:
        raise ValueError(f"unknown model(s) {unknown or wanted}; choose from {list(ALL_MODELS)} or 'both'")
    return tuple(m for m in ALL_MODELS if m in wanted)


def restrict_to_models(explanation: dict, models) -> dict:
    """A copy of the explanation whose disagreement flags concern only the chosen models.

    When explaining one model, flags about the other model (or about the two models together) would only
    confuse the reader, and the writer is required to mention every HIGH/WARNING flag it is given.
    """
    models = normalize_models(models)
    if models == ALL_MODELS:
        return explanation
    keep = set().union(*(FLAGS_OF[m] for m in models))
    return {**explanation, "disagreements": [f for f in explanation.get("disagreements", []) if f["type"] in keep]}


def _kwh(x: float, signed: bool = False) -> str:
    """Format energy: 242680.4 -> "242,680 kWh" ("+30,917 kWh" when signed).

    A value that rounds to zero prints as "0 kWh", never "-0 kWh" (a tiny negative number would otherwise
    show a meaningless minus sign).
    """
    if round(x) == 0:
        return "0 kWh"
    return f"{x:+,.0f} kWh" if signed else f"{x:,.0f} kWh"


def _pct(x: float, signed: bool = True) -> str:
    return f"{x:+.1f}%" if signed else f"{x:.1f}%"


def _value(feature: str, value: float) -> str:
    """A feature's value for display: load-like features in kWh, flags and calendar fields as plain numbers."""
    if feature == "load" or feature.startswith("load_"):    # load_lag_*, load_mean_*, load_std_*, load_same_hour_*
        return f"{value:,.0f} kWh"
    return f"{value:g}"


def _top(records: list[dict], n: int, key: str = "attribution_kwh") -> list[dict]:
    """The n records with the largest absolute attribution (largest first)."""
    return sorted(records, key=lambda r: abs(r[key]), reverse=True)[:n]


def _material(records: list[dict], n: int) -> tuple[list[dict], str]:
    """Top-n records that round to at least 1 kWh, plus a one-line note when some were left out.

    The LSTM has 8 input features and most contribute exactly ~0 to a given forecast; listing them
    as "+0 kWh" only adds noise and invites the LLM to say something about them.
    """
    top = [r for r in _top(records, n) if round(r["attribution_kwh"]) != 0]
    left_out = len(records) - len(top)
    return top, (f"- all other inputs ({left_out}): about 0 kWh" if left_out else "")


# ----------------------------------------------------------------------------- the blocks

def forecast_block(e: dict, models=ALL_MODELS) -> str:
    """[E:forecast] the target hour, the actual load, each chosen model's forecast and error, and (for two models) their gap."""
    ts = pd.Timestamp(e["timestamp"])
    actual = e["actual_kwh"]
    lines = [f"Target hour: {ts:%A %Y-%m-%d %H:%M}. Actual load: {_kwh(actual)}."]
    for m in models:
        p = e[m]["prediction_kwh"]
        lines.append(f"{LABELS[m]} forecast: {_kwh(p)} (error {_kwh(p - actual, True)}, {_pct(100 * (p - actual) / actual)}).")
    if len(models) == 2:
        lg, ls = e["lightgbm"]["prediction_kwh"], e["lstm"]["prediction_kwh"]
        lines.append(f"The two forecasts differ by {_kwh(abs(lg - ls))} ({_pct(100 * abs(lg - ls) / ((lg + ls) / 2), False)} of their average).")
    return "\n".join(lines)


def shap_block(e: dict, top_n: int) -> str:
    """[E:shap] SHAP contributions for LightGBM. Reference = average forecast over all training hours."""
    lg = e["lightgbm"]
    rows = [f"- {r['feature']}: input value {_value(r['feature'], r['value'])}, contribution {_kwh(r['attribution_kwh'], True)}"
            for r in _top(lg["shap"], top_n)]
    return "\n".join([
        f"SHAP for LightGBM. Reference (average forecast over all training hours): {_kwh(lg['expected_value_kwh'])}. "
        f"Contributions add up to the forecast {_kwh(lg['prediction_kwh'])}. Largest {top_n} contributions:", *rows])


def lime_block(e: dict, top_n: int) -> str:
    """[E:lime] LIME weights for LightGBM, with their spread across seeds and the surrogate fit."""
    lg = e["lightgbm"]
    r2 = lg["lime_fit"]["surrogate_r2_per_seed"]
    rows = [f"- {r['feature']} ({r['condition']}): {_kwh(r['attribution_kwh'], True)} "
            f"(std {r['attribution_std']:,.0f} kWh over seeds, rank {int(r['rank_min'])}-{int(r['rank_max'])})"
            for r in _top(lg["lime"], top_n)]
    return "\n".join([
        f"LIME for LightGBM (mean of {len(r2)} random seeds). Surrogate R2 per seed: "
        f"{', '.join(f'{v:.2f}' for v in r2)}; mean {sum(r2) / len(r2):.2f}. LIME weights are slopes of a local linear model, "
        f"not a split of the forecast. Largest {top_n} weights:", *rows])


def ig_block(e: dict, top_n: int) -> str:
    """[E:ig] Integrated Gradients for the LSTM: per feature and the most influential past hours."""
    ls = e["lstm"]
    top, rest = _material(ls["ig_by_feature"], top_n)
    feats = [f"- {r['feature']}: {_kwh(r['attribution_kwh'], True)}" for r in top] + ([rest] if rest else [])
    hours = [f"- t-{int(r['hours_before_target'])}h ({r['timestamp'][:16].replace('T', ' ')}): {_kwh(r['attribution_kwh'], True)}"
             for r in _top(ls["ig_by_timestep"], 3)]
    return "\n".join([
        f"Integrated Gradients for the LSTM. Baseline (typical non-holiday weekday, same hour and season) forecast: "
        f"{_kwh(ls['baseline_prediction_kwh'])}; attributions add up to the LSTM forecast {_kwh(ls['prediction_kwh'])} "
        f"minus that baseline. By feature:", *feats, "Most influential past hours (own features only):", *hours])


def timeshap_block(e: dict, top_n: int) -> str:
    """[E:timeshap] TimeSHAP for the LSTM at feature and event level (same baseline as IG)."""
    ls = e["lstm"]
    top, rest = _material(ls["timeshap_features"], top_n)
    feats = [f"- {r['feature']}: {_kwh(r['attribution_kwh'], True)}" for r in top] + ([rest] if rest else [])
    events = [f"- {r['event']}: {_kwh(r['attribution_kwh'], True)}" for r in _top(ls["timeshap_events"], 4)]
    return "\n".join([
        f"TimeSHAP for the LSTM (same baseline as Integrated Gradients; {ls['timeshap_events_kept']} recent hours kept, "
        f"older hours pruned). By feature:", *feats,
        "By past hour (an hour's total includes part of the effect of the repeated target-day flags):", *events])


def flags_block(e: dict) -> str:
    """[E:flags] disagreement flags: the ones that fired in full, the ones that passed as a single line."""
    flags = e.get("disagreements", [])
    fired = [f for f in flags if f["severity"] != "ok"]
    ok = [f["type"] for f in flags if f["severity"] == "ok"]
    lines = [f"- [{SEVERITY_WORDS[f['severity']]}] {f['type']}: {f['message']}" for f in fired] or ["- none fired"]
    return "\n".join(["Disagreement checks that fired:", *lines, f"Checks with no disagreement: {', '.join(ok) or 'none'}."])


def accuracy_block(e: dict, models=ALL_MODELS) -> str:
    """[E:accuracy] how accurate each model is on this kind of day and time of day (test set, Jul-Dec 2014).

    Tells the reader whether this hour is a known weak spot (e.g. major holidays) or ordinary.
    """
    ts = pd.Timestamp(e["timestamp"])
    cal = calendar_features(pd.DatetimeIndex([ts]))
    frame = cal.assign(hour=cal["hour"])
    day_type = ac._slice_labels(frame, "day_type").iloc[0]
    part = ac._slice_labels(frame, "part_of_day").iloc[0]
    lines = []
    for by, label in (("overall", "all hours"), ("day_type", day_type), ("part_of_day", part)):
        table = ac.accuracy_by(by)
        rows = table[table["slice"] == label].set_index("model")
        scores = ", ".join(f"{LABELS[m]} RMSE {rows.loc[m, 'rmse']:,.0f} kWh" for m in models)
        lines.append(f"- {by.replace('_', ' ')} '{label}' ({int(rows['n'].iloc[0])} test hours): {scores}")
    return "\n".join(["Test-set accuracy (July-December 2014) for this kind of hour; RMSE is the typical size of the miss in kWh (big misses weigh more):", *lines])


# ----------------------------------------------------------------------------- the pack

def build_evidence(explanation: dict, top_n: int = 6, models="both") -> dict[str, str]:
    """All evidence blocks for one explanation: {"E:forecast": text, "E:shap": text, ...}, in reading order.

    explanation: the dict from store.get_explanation (needs its "disagreements").
    top_n:       how many rows to keep per explainer table. 6 covers the drivers without burying the reader.
    models:      "both" (default), "lightgbm" or "lstm". A single model gets only its own explainer blocks
                 (SHAP+LIME or IG+TimeSHAP), its own forecast, its own flags and its own accuracy.
    """
    models = normalize_models(models)
    e = restrict_to_models(explanation, models)
    blocks = {"E:forecast": forecast_block(e, models)}
    if "lightgbm" in models:
        blocks["E:shap"] = shap_block(e, top_n)
        blocks["E:lime"] = lime_block(e, top_n)
    if "lstm" in models:
        blocks["E:ig"] = ig_block(e, top_n)
        blocks["E:timeshap"] = timeshap_block(e, top_n)
    blocks["E:flags"] = flags_block(e)
    blocks["E:accuracy"] = accuracy_block(e, models)
    return blocks


def render_evidence(evidence: dict[str, str]) -> str:
    """The pack as one text block for the prompt: "[E:shap]\\n<text>" separated by blank lines."""
    return "\n\n".join(f"[{key}]\n{text}" for key, text in evidence.items())
