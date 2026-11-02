"""Detect where the models and the explainers disagree about ONE forecast.

Why this exists: four explainers and two models will not always tell the same story, and an
explanation that hides that is misleading. Each check below turns one kind of disagreement into a
flag the LLM must mention and the judge can verify:

    model_gap            LightGBM and LSTM forecast different numbers
    shared_error         both models miss the actual in the same direction (a missing input?)
    shap_vs_lime         the two LightGBM explainers rank/sign the features differently
    lime_fit             LIME's local linear model is a poor fit, so its weights are unreliable
    lime_instability     LIME's top features change rank between random seeds
    ig_vs_timeshap       the two LSTM explainers rank/sign the features differently
    ig_convergence       IG's numerical integral is inaccurate
    cross_model_driver   the two models lean on different kinds of information

Input: the explanation in its dict form - what `ForecastExplanation.to_dict()` returns and what
store.py reads from outputs/explanations/*.json. Tables are lists of records. Working on the dict
(not on live DataFrames) means these checks run on cached files without importing torch or shap.

Output: a list of flags, each {"type", "severity", "message", "numbers"}.
    severity "high"    the explanation should not be trusted/quoted without this caveat
             "warning" worth mentioning
             "info"    detail for a curious reader
             "ok"      the check ran and found agreement (kept so the LLM can say "the models agree")
`numbers` holds the figures behind the message so the judge can verify the text against them.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

SEVERITY_ORDER = {"high": 0, "warning": 1, "info": 2, "ok": 3}


@dataclass(frozen=True)
class Thresholds:
    """All tunable cut-offs in one place, so they are visible and can be changed without editing logic.

    The defaults are starting points chosen from the saved examples, not statistical truths;
    notebook 03 shows how they behave on several hours.

    model_gap_warn_pct / model_gap_high_pct:
        |LightGBM - LSTM| as % of their mean. Both models score ~3% MAPE, so a 3% gap between them
        is already as large as a typical error; 6% means they genuinely disagree.
    shared_error_pct:
        a model counts as "off" when its forecast is this far (%) from the actual.
    top_k:
        how many top features count as "the drivers" when comparing two explainers.
    rank_corr_warn:
        Spearman correlation of the |attribution| ranks below this means the explainers order the
        features differently. (1 = same order, 0 = unrelated.)
    min_overlap:
        the two explainers' top-k lists must share at least this many features.
    min_effect_share:
        a sign flip only counts if both explainers give the feature at least this share of the
        forecast (e.g. 0.02 = 2%). Otherwise tiny near-zero attributions would flip sign by noise.
    lime_r2_warn:
        LIME's surrogate R^2 below this => the local linear model explains <30% of the variation
        around the forecast, so its weights are weak evidence.
    rank_spread_warn:
        a top-k feature whose LIME rank moves by this many places across seeds is unstable.
    ig_delta_share:
        IG convergence error above this share of the explained difference (forecast - baseline).
    timestep_tolerance_h:
        IG's and TimeSHAP's most influential past hours may differ by this many hours before we
        call it a disagreement.
    """
    model_gap_warn_pct: float = 3.0
    model_gap_high_pct: float = 6.0
    shared_error_pct: float = 10.0
    top_k: int = 5
    rank_corr_warn: float = 0.5
    min_overlap: int = 3
    min_effect_share: float = 0.02
    lime_r2_warn: float = 0.3
    rank_spread_warn: int = 3
    ig_delta_share: float = 0.05
    timestep_tolerance_h: int = 2


# Which kind of information each input feature carries. Used to compare models that have different
# feature sets: "do both rely mainly on recent load, or on the holiday calendar?"
LGBM_FAMILIES = {
    "recent load": ["load_lag_", "load_mean_", "load_std_", "load_same_hour_mean_"],
    "holiday calendar": ["is_holiday", "is_major_holiday", "is_bridge_day", "lag_24h_was_", "lag_168h_was_"],
    "time of day/week": ["hour", "day_of_week", "month", "is_weekend"],
}
LSTM_FAMILIES = {
    "recent load": ["load"],
    "holiday calendar": ["is_holiday", "is_major_holiday", "target_is_holiday", "target_is_major_holiday"],
    "time of day/week": ["hour_sin", "hour_cos", "target_is_weekend"],
}


# ----------------------------------------------------------------------------- small helpers

def _flag(type_: str, severity: str, message: str, **numbers) -> dict:
    """Build one flag. Floats are rounded so the JSON stays short and the LLM quotes clean numbers."""
    clean = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in numbers.items()}
    return {"type": type_, "severity": severity, "message": message, "numbers": clean}


def _frame(records: list[dict]) -> pd.DataFrame:
    """A saved table (list of records) back into a DataFrame."""
    return pd.DataFrame(records)


def _by_feature(records: list[dict]) -> pd.Series:
    """attribution_kwh indexed by feature name, from a saved table."""
    df = _frame(records)
    return df.set_index("feature")["attribution_kwh"].astype(float)


def _rank_corr(a: pd.Series, b: pd.Series, min_abs: float = 0.0, min_features: int = 4) -> float:
    """Spearman correlation between the *importance ranks* (|attribution|) of two explainers.

    Spearman = Pearson correlation of the ranks. We rank |attribution| (size, not direction) so the
    question is purely "do they agree on WHICH features matter most?"; direction is checked
    separately with sign flips. Only features present in both are compared.

    min_abs (kWh): only features where at least one explainer gives |attribution| >= min_abs take
        part. Most of the 21 features contribute ~0 to any single forecast; their relative order is
        pure noise and would drag the correlation down even when the explainers agree on every
        feature that matters.
    min_features: with fewer material features than this a rank correlation is meaningless, so we
        return NaN ("not applicable") and rely on the top-feature and sign checks instead.
    """
    common = a.index.intersection(b.index)
    a, b = a[common], b[common]
    material = (a.abs() >= min_abs) | (b.abs() >= min_abs)
    if material.sum() < min_features:
        return float("nan")
    return float(a[material].abs().rank().corr(b[material].abs().rank()))


def _fmt_corr(corr: float) -> str:
    """Rank correlation for a message; "n/a" when too few features mattered to compute it."""
    return "n/a" if np.isnan(corr) else f"{corr:.2f}"


def _top(series: pd.Series, k: int) -> list[str]:
    """Names of the k features with the largest |attribution|."""
    return series.abs().nlargest(k).index.tolist()


def _sign_flips(a: pd.Series, b: pd.Series, k: int, min_effect: float) -> list[str]:
    """Features among either explainer's top-k whose direction (push up vs down) differs.

    min_effect (kWh): both attributions must be at least this big, otherwise a value like +3 kWh vs
    -2 kWh would count as a "flip" although both just mean "no effect".
    """
    candidates = set(_top(a, k)) | set(_top(b, k))
    return sorted(f for f in candidates
                  if f in a.index and f in b.index
                  and np.sign(a[f]) != np.sign(b[f])
                  and abs(a[f]) >= min_effect and abs(b[f]) >= min_effect)


def _family_totals(attr: pd.Series, families: dict[str, list[str]]) -> pd.Series:
    """Sum |attribution| per feature family. A feature belongs to the family whose prefix it starts with."""
    totals = dict.fromkeys(families, 0.0)
    for feature, value in attr.abs().items():
        for family, prefixes in families.items():
            if any(feature == p or feature.startswith(p) for p in prefixes):
                totals[family] += value
                break
    return pd.Series(totals)


# ----------------------------------------------------------------------------- the checks
# Each check takes the explanation dict `e` and the Thresholds `t` and returns a list of flags.

def check_model_gap(e: dict, t: Thresholds) -> list[dict]:
    """Do LightGBM and the LSTM forecast roughly the same number?

    The gap needs no ground truth, so it also works for future hours. A big gap means at least one
    model is extrapolating, and any explanation of "the forecast" is really two different stories.
    """
    lg, ls = e["lightgbm"]["prediction_kwh"], e["lstm"]["prediction_kwh"]
    gap = 100 * abs(lg - ls) / ((lg + ls) / 2)
    numbers = dict(lightgbm_kwh=lg, lstm_kwh=ls, gap_pct=gap)
    if gap >= t.model_gap_high_pct:
        return [_flag("model_gap", "high", f"The models disagree: LightGBM {lg:,.0f} kWh vs LSTM {ls:,.0f} kWh ({gap:.1f}% apart).", **numbers)]
    if gap >= t.model_gap_warn_pct:
        return [_flag("model_gap", "warning", f"The models differ noticeably: {gap:.1f}% apart ({lg:,.0f} vs {ls:,.0f} kWh).", **numbers)]
    return [_flag("model_gap", "ok", f"The models agree: {gap:.1f}% apart ({lg:,.0f} vs {ls:,.0f} kWh).", **numbers)]


def check_shared_error(e: dict, t: Thresholds) -> list[dict]:
    """Do the models agree with each other but not with reality?

    When both over- (or under-) forecast by a lot in the same direction, the cause is rarely model
    noise - more likely something neither model has as an input (e.g. Christmas Eve is not an
    official holiday, so no holiday flag fires). That is the most useful thing to tell a reader.
    """
    actual = e["actual_kwh"]
    err = {m: 100 * (e[m]["prediction_kwh"] - actual) / actual for m in ("lightgbm", "lstm")}
    numbers = dict(actual_kwh=actual, lightgbm_error_pct=err["lightgbm"], lstm_error_pct=err["lstm"])
    off = {m: abs(v) >= t.shared_error_pct for m, v in err.items()}
    same_sign = np.sign(err["lightgbm"]) == np.sign(err["lstm"])
    direction = "over" if err["lightgbm"] > 0 else "under"
    if all(off.values()) and same_sign:
        return [_flag("shared_error", "high",
                      f"Both models {direction}-forecast the actual {actual:,.0f} kWh "
                      f"(LightGBM {err['lightgbm']:+.1f}%, LSTM {err['lstm']:+.1f}%) - likely information neither model has.",
                      **numbers)]
    if any(off.values()):
        worse = max(err, key=lambda m: abs(err[m]))
        return [_flag("shared_error", "warning",
                      f"Only {worse} is far from the actual ({err[worse]:+.1f}%); the other is within {t.shared_error_pct:.0f}%.",
                      **numbers)]
    return [_flag("shared_error", "ok",
                  f"Both models are within {t.shared_error_pct:.0f}% of the actual (LightGBM {err['lightgbm']:+.1f}%, LSTM {err['lstm']:+.1f}%).",
                  **numbers)]


def check_shap_vs_lime(e: dict, t: Thresholds) -> list[dict]:
    """Do SHAP and LIME (both explaining LightGBM) name the same drivers in the same direction?

    Caveat that makes some disagreement expected: SHAP splits the forecast relative to the average
    forecast over training hours, while LIME is the slope of a local linear model. Their numbers are
    not identical quantities, so we compare *rankings and signs*, not magnitudes.
    """
    lg = e["lightgbm"]
    shap, lime = _by_feature(lg["shap"]), _by_feature(lg["lime"])
    min_effect = t.min_effect_share * lg["prediction_kwh"]
    corr = _rank_corr(shap, lime, min_effect)
    top_shap, top_lime = _top(shap, t.top_k), _top(lime, t.top_k)
    overlap = len(set(top_shap) & set(top_lime))
    flips = _sign_flips(shap, lime, t.top_k, min_effect)
    numbers = dict(rank_correlation=corr, top_k=t.top_k, overlap=overlap, top_shap=top_shap,
                   top_lime=top_lime, sign_flips=flips)

    problems = []
    if corr < t.rank_corr_warn:  # NaN compares False: "not applicable" is not a problem
        problems.append(f"their feature rankings correlate only {corr:.2f}")
    if overlap < t.min_overlap:
        problems.append(f"their top-{t.top_k} lists share {overlap} feature(s)")
    if flips:
        problems.append(f"they disagree on direction for {', '.join(flips)}")
    if problems:
        severity = "high" if len(problems) >= 2 else "warning"
        return [_flag("shap_vs_lime", severity, "SHAP and LIME disagree: " + "; ".join(problems) + ".", **numbers)]
    return [_flag("shap_vs_lime", "ok",
                  f"SHAP and LIME agree: rank correlation {_fmt_corr(corr)}, {overlap}/{t.top_k} top features in common, no direction flips.",
                  **numbers)]


def check_lime_reliability(e: dict, t: Thresholds) -> list[dict]:
    """Is LIME trustworthy for this forecast? Two separate questions: fit and stability.

    Fit: LIME fits a weighted linear model on perturbed samples around the instance; its R^2 says
    how much of the model's local behaviour that line captures. Low R^2 => the weights describe
    a line that doesn't match the model (typical for tree models near sharp splits such as holidays).
    Stability: LIME is random. We ran several seeds; if a top feature's rank moves a lot between
    seeds, the ranking is partly noise.
    """
    lg = e["lightgbm"]
    flags = []
    r2 = lg["lime_fit"]["surrogate_r2_per_seed"]
    mean_r2 = float(np.mean(r2))
    numbers = dict(mean_r2=mean_r2, min_r2=float(np.min(r2)), n_seeds=len(r2))
    if mean_r2 < t.lime_r2_warn:
        flags.append(_flag("lime_fit", "warning",
                           f"LIME's local linear model fits poorly (R2 {mean_r2:.2f}), so its weights are weak evidence.", **numbers))
    else:
        flags.append(_flag("lime_fit", "ok", f"LIME's local linear model fits reasonably (R2 {mean_r2:.2f}).", **numbers))

    lime = _frame(lg["lime"]).set_index("feature")
    top = lime["attribution_kwh"].abs().nlargest(t.top_k).index
    spread = (lime.loc[top, "rank_max"] - lime.loc[top, "rank_min"]).astype(int)
    unstable = spread[spread >= t.rank_spread_warn]
    if len(unstable):
        detail = ", ".join(f"{f} (rank {int(lime.loc[f, 'rank_min'])}-{int(lime.loc[f, 'rank_max'])})" for f in unstable.index)
        flags.append(_flag("lime_instability", "warning",
                           f"LIME's ranking changes between random seeds for: {detail}.",
                           unstable_features=unstable.index.tolist(), max_rank_spread=int(spread.max())))
    else:
        flags.append(_flag("lime_instability", "ok", f"LIME's top-{t.top_k} ranking is stable across {len(r2)} seeds.",
                           max_rank_spread=int(spread.max())))
    return flags


def _ig_and_timeshap_by_feature(l: dict) -> tuple[pd.Series, pd.Series]:
    """IG and TimeSHAP attributions per input feature, restricted to the features both report.

    TimeSHAP also reports pseudo-features such as "(older hours, pruned)" - those start with "("
    and have no IG counterpart, so they are dropped. (TimeSHAP's own per-feature totals leave the
    pruned old hours out of each feature; their combined effect is below the pruning tolerance.)
    """
    ig = _by_feature(l["ig_by_feature"])
    ts_all = _frame(l["timeshap_features"])
    ts = ts_all[~ts_all["feature"].str.startswith("(")].set_index("feature")["attribution_kwh"].astype(float)
    common = ig.index.intersection(ts.index)
    return ig[common], ts[common]


def check_ig_vs_timeshap(e: dict, t: Thresholds) -> list[dict]:
    """Do IG and TimeSHAP (both explaining the LSTM, same baseline) agree?

    Both explain "forecast minus the baseline forecast", so unlike SHAP vs LIME their magnitudes are
    directly comparable: we also report the top feature of each. Second question: which past HOUR
    mattered most. IG's per-hour totals leave out the repeated target-hour flags, TimeSHAP's event
    totals include them, so a difference there is reported as "info", not as a warning.
    """
    l = e["lstm"]
    ig, ts = _ig_and_timeshap_by_feature(l)
    min_effect = t.min_effect_share * l["prediction_kwh"]
    corr = _rank_corr(ig, ts, min_effect)
    top_ig, top_ts = _top(ig, 3), _top(ts, 3)
    flips = _sign_flips(ig, ts, 3, min_effect)
    numbers = dict(rank_correlation=corr, top_ig=top_ig, top_timeshap=top_ts, sign_flips=flips,
                   ig_top_kwh=float(ig[top_ig[0]]), timeshap_top_kwh=float(ts[top_ts[0]]))

    problems = []
    if top_ig[0] != top_ts[0]:
        problems.append(f"IG's top feature is {top_ig[0]} ({ig[top_ig[0]]:+,.0f} kWh) but TimeSHAP's is {top_ts[0]} ({ts[top_ts[0]]:+,.0f} kWh)")
    if corr < t.rank_corr_warn:
        problems.append(f"feature rankings correlate only {corr:.2f}")
    if flips:
        problems.append(f"they disagree on direction for {', '.join(flips)}")
    if problems:
        severity = "high" if len(problems) >= 2 else "warning"
        flags = [_flag("ig_vs_timeshap", severity, "IG and TimeSHAP disagree: " + "; ".join(problems) + ".", **numbers)]
    else:
        flags = [_flag("ig_vs_timeshap", "ok", f"IG and TimeSHAP agree on the main driver ({top_ig[0]}); rank correlation {_fmt_corr(corr)}.", **numbers)]

    # Which past hour matters most? Compare IG's top timestep with TimeSHAP's top single event.
    steps = _frame(l["ig_by_timestep"])
    events = _frame(l["timeshap_events"])
    events = events[events["hours_before_target"].notna()]  # drop the "older hours (pruned)" group
    if len(steps) and len(events):
        ig_hour = int(steps.loc[steps["attribution_kwh"].abs().idxmax(), "hours_before_target"])
        ts_hour = int(events.loc[events["attribution_kwh"].abs().idxmax(), "hours_before_target"])
        n = dict(ig_top_hours_before=ig_hour, timeshap_top_hours_before=ts_hour)
        if abs(ig_hour - ts_hour) > t.timestep_tolerance_h:
            flags.append(_flag("ig_vs_timeshap_timestep", "info",
                               f"IG's most influential past hour is t-{ig_hour}h, TimeSHAP's is t-{ts_hour}h.", **n))
        else:
            flags.append(_flag("ig_vs_timeshap_timestep", "ok",
                               f"Both point to about t-{ig_hour}h as the most influential past hour.", **n))
    return flags


def check_ig_convergence(e: dict, t: Thresholds) -> list[dict]:
    """Is the IG numerical integral accurate?

    IG's attributions should sum to (forecast - baseline forecast) - the 'completeness' property.
    The convergence delta is how far they are from that; if it is a sizeable share of the explained
    difference, the IG numbers are only approximate (raise n_steps).
    """
    l = e["lstm"]
    explained = l["prediction_kwh"] - l["baseline_prediction_kwh"]
    delta = abs(l["ig_convergence_delta_kwh"])
    share = delta / abs(explained) if explained else float("inf")
    numbers = dict(delta_kwh=float(delta), explained_kwh=float(explained), share=float(share))
    if share > t.ig_delta_share:
        return [_flag("ig_convergence", "warning",
                      f"IG is only approximate here: its error ({delta:,.0f} kWh) is {100 * share:.0f}% of the explained difference.", **numbers)]
    return [_flag("ig_convergence", "ok", f"IG sums to the forecast difference within {delta:,.0f} kWh.", **numbers)]


def check_cross_model_driver(e: dict, t: Thresholds) -> list[dict]:
    """Do the two models lean on the same *kind* of information?

    Their features differ, so we group them into families (recent load / holiday calendar /
    time of day-week) and ask which family carries the most |attribution| for each model. Models
    that agree on the forecast but rely on different families are explaining it differently,
    which is worth telling the reader.
    """
    shap = _by_feature(e["lightgbm"]["shap"])
    ig, ts = _ig_and_timeshap_by_feature(e["lstm"])
    lgbm_top = _family_totals(shap, LGBM_FAMILIES)
    # TimeSHAP is the more complete LSTM explainer (it separates the target-hour flags properly).
    lstm_top = _family_totals(ts, LSTM_FAMILIES)
    a, b = lgbm_top.idxmax(), lstm_top.idxmax()
    numbers = dict(lightgbm_top_family=a, lstm_top_family=b,
                   lightgbm_families={k: round(float(v)) for k, v in lgbm_top.items()},
                   lstm_families={k: round(float(v)) for k, v in lstm_top.items()})
    if a != b:
        return [_flag("cross_model_driver", "warning",
                      f"LightGBM leans mostly on {a}, the LSTM mostly on {b}.", **numbers)]
    return [_flag("cross_model_driver", "ok", f"Both models lean mostly on {a}.", **numbers)]


CHECKS = [check_model_gap, check_shared_error, check_shap_vs_lime, check_lime_reliability,
          check_ig_vs_timeshap, check_ig_convergence, check_cross_model_driver]


# ----------------------------------------------------------------------------- public API

def find_disagreements(explanation: dict, thresholds: Thresholds = Thresholds()) -> list[dict]:
    """Run every check on one explanation and return the flags, most severe first.

    explanation: the dict form of a ForecastExplanation (see module docstring).
    thresholds:  cut-offs; pass a modified Thresholds() to make the checks stricter or looser.
    Sorting is stable, so flags of equal severity keep the order of CHECKS.
    """
    flags = [flag for check in CHECKS for flag in check(explanation, thresholds)]
    return sorted(flags, key=lambda f: SEVERITY_ORDER[f["severity"]])


def summarize(flags: list[dict]) -> dict:
    """Counts per severity, e.g. {"high": 2, "warning": 1, "info": 0, "ok": 4}. For headlines and tests."""
    counts = dict.fromkeys(SEVERITY_ORDER, 0)
    for f in flags:
        counts[f["severity"]] += 1
    return counts


def thresholds_dict(thresholds: Thresholds = Thresholds()) -> dict:
    """The thresholds as a plain dict, stored next to the flags so a result can be reproduced."""
    return asdict(thresholds)
