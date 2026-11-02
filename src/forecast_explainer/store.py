"""Load-or-compute cache for explanations: outputs/explanations/{client}_{YYYY-MM-DD_HHMM}.json.

Why cache? Computing an explanation (SHAP, LIME x5 seeds, IG, TimeSHAP) takes several seconds, and
LIME/TimeSHAP are random, so recomputing gives slightly different numbers each time. The agents will
ask about the same hour repeatedly ("why was Christmas Eve high?", then "and which model was better
there?"); the second question must quote exactly the same numbers as the first (the judge checks the
text against them), and a planning agent that looks at dozens of hours should not recompute each one.

The cache can go stale in one way: the models are retrained, so the saved explanation describes a
model that no longer exists. Each file therefore stores `model_fingerprint` (registry.fingerprint),
a hash of the model files, and is recomputed when it no longer matches.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from forecast_explainer import PROJECT_ROOT
from forecast_explainer import registry
from forecast_explainer.disagreement import Thresholds, find_disagreements, summarize, thresholds_dict

EXPLANATIONS_DIR = PROJECT_ROOT / "outputs" / "explanations"


def explanation_path(timestamp, client: str = "total", directory: Path = EXPLANATIONS_DIR) -> Path:
    """Where the explanation for one target hour lives. Same naming as ForecastExplanation.save()."""
    return directory / f"{client.lower()}_{pd.Timestamp(timestamp):%Y-%m-%d_%H%M}.json"


def is_fresh(path: Path, client: str = "total") -> bool:
    """Was the explanation in `path` computed with the current models?

    New files carry a fingerprint, compared exactly. Files written by scripts/explain.py or by the
    earlier notebook have no fingerprint; for those we accept the file if it is newer than every
    model file (it cannot have been computed from older models then).
    """
    if not path.exists():
        return False
    stored = json.loads(path.read_text()).get("model_fingerprint")
    if stored is not None:
        return stored == registry.fingerprint(client)
    return path.stat().st_mtime >= registry.newest_model_mtime(client)


def get_explanation(timestamp, client: str = "total", recompute: bool = False,
                    thresholds: Thresholds = Thresholds(), verbose: bool = True, **explain_kwargs) -> dict:
    """The explanation for one target hour, from cache if it is current, otherwise computed and saved.

    timestamp:       the hour being forecast, anything pd.Timestamp accepts ("2014-12-24 18:00").
    client:          "total" or a client id; selects which trained models are explained.
    recompute:       True forces a fresh computation even if a current file exists.
    thresholds:      cut-offs for the disagreement flags (see disagreement.Thresholds). Flags are
                     recomputed on every call (they take milliseconds) and written back, so
                     changing thresholds never requires recomputing the slow explainers.
    verbose:         print progress while computing (TimeSHAP is slow; silence is confusing).
    **explain_kwargs: passed to explain_forecast, e.g. lime_repeats=10, timeshap_nsamples=4000.

    Returns the explanation dict (tables as lists of records, same layout as the JSON file) with
    extra keys: "model_fingerprint", "computed_at", "disagreements", "disagreement_summary",
    "disagreement_thresholds", and "source" ("cache" or "computed"; not saved to the file).
    """
    timestamp = pd.Timestamp(timestamp)
    path = explanation_path(timestamp, client)

    if not recompute and is_fresh(path, client):
        explanation, source = json.loads(path.read_text()), "cache"
        explanation.setdefault("model_fingerprint", registry.fingerprint(client))
    else:
        # Imported here, not at the top: explain.py pulls in torch, shap, lime, captum and
        # timeshap (several seconds). Reading a cached explanation should not pay for that.
        from forecast_explainer.explain import explain_forecast

        computed = explain_forecast(timestamp, client, verbose=verbose, **explain_kwargs)
        explanation, source = computed.to_dict(), "computed"
        explanation["model_fingerprint"] = registry.fingerprint(client)
        explanation["computed_at"] = datetime.now().isoformat(timespec="seconds")

    flags = find_disagreements(explanation, thresholds)
    explanation["disagreements"] = flags
    explanation["disagreement_summary"] = summarize(flags)
    explanation["disagreement_thresholds"] = thresholds_dict(thresholds)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(explanation, indent=2, default=str))
    explanation["source"] = source
    return explanation
