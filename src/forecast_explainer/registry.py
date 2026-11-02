"""Model registry: the one place that knows which forecasting models exist and where their files live.

Why a registry? The later agents (explanation agent, planning agent) must work for "any model" and
must not hard-code `lgbm_total.txt` or `lstm_total.pt`. They ask the registry:
    - "what models do we have?"                      -> MODELS
    - "give me every model's test-set forecasts"     -> all_predictions()
    - "have the model files changed since I cached?" -> fingerprint()
Adding a third model later (e.g. XGBoost) means adding one ModelSpec here; nothing else in the
accuracy / disagreement / cache code needs to change.

File naming convention (set by scripts/train_*.py):
    models/{prefix}_{client}.{txt|pt}                 the trained weights
    models/{prefix}_{client}.json                     metadata (features, split, metrics)
    models/{prefix}_{client}_predictions.parquet      test-set forecasts: columns y_true, y_pred (+ features)
"""

import hashlib
from dataclasses import dataclass

import pandas as pd

from forecast_explainer import MODELS_DIR


@dataclass(frozen=True)
class ModelSpec:
    """Static description of one forecasting model.

    key:            short machine name used as a column / dict key everywhere ("lightgbm").
    label:          human name for charts and the LLM's text ("LightGBM").
    prefix:         start of the file names in models/ ("lgbm" -> lgbm_total.txt).
    weights_suffix: extension of the weights file (".txt" for a LightGBM booster, ".pt" for torch).
    explainers:     which explanation methods apply to this model (documentation + used by agents
                    to know what evidence exists).
    """
    key: str
    label: str
    prefix: str
    weights_suffix: str
    explainers: tuple[str, ...]

    def stem(self, client: str) -> str:
        """File-name stem for one client, e.g. "lgbm_total". Lower-cased to match the train scripts."""
        return f"{self.prefix}_{client.lower()}"

    def files(self, client: str) -> list:
        """The files that define this model. Used to fingerprint it (not the predictions file:
        that is an output of the model, not part of it)."""
        stem = self.stem(client)
        return [MODELS_DIR / f"{stem}{self.weights_suffix}", MODELS_DIR / f"{stem}.json"]

    def predictions(self, client: str = "total") -> pd.DataFrame:
        """The saved test-set forecasts: index = target hour, columns y_true and y_pred (kWh).

        `columns=` skips the feature columns LightGBM's file also stores; we only need these two.
        """
        path = MODELS_DIR / f"{self.stem(client)}_predictions.parquet"
        return pd.read_parquet(path, columns=["y_true", "y_pred"])


MODELS: dict[str, ModelSpec] = {
    "lightgbm": ModelSpec("lightgbm", "LightGBM", "lgbm", ".txt", ("shap", "lime")),
    "lstm": ModelSpec("lstm", "LSTM", "lstm", ".pt", ("integrated_gradients", "timeshap")),
}


def all_predictions(client: str = "total", models: list[str] | None = None) -> pd.DataFrame:
    """Every model's test-set forecast side by side: columns y_true, <model key>, <model key>, ...

    models: restrict to some model keys (default: all in MODELS).

    Models are aligned on their target hours with an inner join. The two models start on slightly
    different dates (the LSTM needs a full 168 h window, LightGBM needs 4 weeks of lags), so we
    only keep hours that every model forecast; otherwise accuracy would compare different hours.
    The `y_true` of each model must agree, otherwise they were trained on different data and a
    comparison would be meaningless - we fail loudly rather than silently mixing them.
    """
    keys = models or list(MODELS)
    frames = {k: MODELS[k].predictions(client) for k in keys}
    out = pd.concat({k: f["y_pred"] for k, f in frames.items()}, axis=1, join="inner")
    truth = pd.concat({k: f["y_true"] for k, f in frames.items()}, axis=1, join="inner")
    spread = (truth.max(axis=1) - truth.min(axis=1)).abs().max()
    if spread > 1e-6 * truth.abs().max().max():
        raise ValueError("models were trained on different data: their y_true columns disagree")
    out.insert(0, "y_true", truth.iloc[:, 0])
    return out


def fingerprint(client: str = "total") -> str:
    """Short hash of all model files. Changes whenever any model is retrained.

    Saved explanations store this value; if it no longer matches, the explanation was computed for
    an older model and must be recomputed (see store.py). We hash file *contents*, not timestamps,
    so copying the project to another machine (e.g. into a Docker image) keeps caches valid.
    """
    digest = hashlib.sha1()
    for spec in MODELS.values():
        for path in spec.files(client):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def newest_model_mtime(client: str = "total") -> float:
    """Modification time of the most recently written model file (seconds since epoch).

    Only used for explanation files created before fingerprints existed: if such a file is newer
    than every model file, it was computed with the current models.
    """
    return max(p.stat().st_mtime for spec in MODELS.values() for p in spec.files(client))
