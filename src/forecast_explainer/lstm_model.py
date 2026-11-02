"""Small LSTM day-ahead load forecaster: the sequence model explained with Captum IG and TimeSHAP.

Each sample is the week of hourly history that is known a day ahead: the 168 hours ending at t-24.
Every timestep is a real past hour (its load and calendar). The target hour's calendar flags are
repeated on every timestep, the usual way to give TimeSHAP a static covariate.
"""

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from forecast_explainer import MODELS_DIR
from forecast_explainer.features import MIN_LAG, calendar_features
from forecast_explainer.lgbm_model import Split

SEQ_LEN = 168  # one week of hourly history
STEP_FEATURES = ["load", "hour_sin", "hour_cos", "is_holiday", "is_major_holiday"]
TARGET_FEATURES = ["target_is_weekend", "target_is_holiday", "target_is_major_holiday"]
FEATURES = STEP_FEATURES + TARGET_FEATURES


@dataclass
class LstmConfig:
    hidden_size: int = 64
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 40
    patience: int = 5
    seed: int = 42


class LoadLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


@dataclass
class Scaler:
    """Standardises load; the model works in these units, explanations are converted back to kWh."""
    mean: float
    std: float

    def scale(self, x):
        return (x - self.mean) / self.std

    def unscale(self, z):
        return z * self.std + self.mean


def hourly_table(load: pd.Series, scaler: Scaler) -> pd.DataFrame:
    """Per-hour step features and target features, before windowing."""
    cal = calendar_features(load.index)
    table = pd.DataFrame(index=load.index)
    table["load"] = scaler.scale(load)
    table["hour_sin"] = np.sin(2 * np.pi * cal["hour"] / 24)
    table["hour_cos"] = np.cos(2 * np.pi * cal["hour"] / 24)
    table["is_holiday"] = cal["is_holiday"]
    table["is_major_holiday"] = cal["is_major_holiday"]
    table["target_is_weekend"] = cal["is_weekend"]
    table["target_is_holiday"] = cal["is_holiday"]
    table["target_is_major_holiday"] = cal["is_major_holiday"]
    return table.astype("float32")


def build_sequences(load: pd.Series, scaler: Scaler) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """All (sequence, scaled target) pairs. X has shape (n, SEQ_LEN, len(FEATURES))."""
    table = hourly_table(load, scaler)
    steps = table[STEP_FEATURES].to_numpy()
    targets = table[TARGET_FEATURES].to_numpy()
    # windows[i] covers hours i .. i+SEQ_LEN-1; its target hour is MIN_LAG after the last one.
    windows = np.lib.stride_tricks.sliding_window_view(steps, SEQ_LEN, axis=0).transpose(0, 2, 1)
    first_target = SEQ_LEN - 1 + MIN_LAG
    windows = windows[: len(table) - first_target]
    target_feats = np.repeat(targets[first_target:, None, :], SEQ_LEN, axis=1)
    X = np.concatenate([windows, target_feats], axis=2).astype("float32")
    y = table["load"].to_numpy()[first_target:]
    return X, y, table.index[first_target:]


def sequence_for(load: pd.Series, scaler: Scaler, timestamp) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """The single input sequence for one target hour, plus the timestamps of its 168 steps."""
    timestamp = pd.Timestamp(timestamp)
    end = timestamp - pd.Timedelta(hours=MIN_LAG)
    start = end - pd.Timedelta(hours=SEQ_LEN - 1)
    X, _, index = build_sequences(load.loc[start:timestamp], scaler)
    assert index[-1] == timestamp, "not enough history before this timestamp"
    return X[-1:], pd.date_range(start, end, freq="h")


def split_masks(index: pd.DatetimeIndex, split: Split) -> dict[str, np.ndarray]:
    return {
        "train": index <= split.train_end,
        "valid": (index > split.train_end) & (index <= split.valid_end),
        "test": index > split.valid_end,
    }


def train(X_train, y_train, X_valid, y_valid, config: LstmConfig = LstmConfig()) -> LoadLSTM:
    torch.manual_seed(config.seed)
    model = LoadLSTM(X_train.shape[2], config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()
    X_t, y_t = torch.from_numpy(X_train), torch.from_numpy(y_train).float().unsqueeze(1)

    best_loss, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(config.max_epochs):
        model.train()
        for idx in torch.randperm(len(X_t)).split(config.batch_size):
            optimizer.zero_grad()
            loss_fn(model(X_t[idx]), y_t[idx]).backward()
            optimizer.step()
        valid_loss = float(np.mean((predict_scaled(model, X_valid) - y_valid) ** 2))
        print(f"epoch {epoch + 1:2d}  valid MSE {valid_loss:.4f}")
        if valid_loss < best_loss:
            best_loss, best_state, bad_epochs = valid_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


def predict_scaled(model: LoadLSTM, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """Model output in scaled units, shape (n,). Accepts numpy so TimeSHAP can call it directly."""
    model.eval()
    with torch.no_grad():
        out = [model(torch.from_numpy(np.asarray(b, dtype="float32"))) for b in np.array_split(X, max(1, len(X) // batch_size))]
    return torch.cat(out).squeeze(1).numpy()


def save(model: LoadLSTM, name: str, scaler: Scaler, config: LstmConfig, meta: dict, predictions: pd.DataFrame) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / f"{name}.pt")
    meta = meta | {"scaler": asdict(scaler), "config": asdict(config), "features": FEATURES, "seq_len": SEQ_LEN}
    (MODELS_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str))
    predictions.to_parquet(MODELS_DIR / f"{name}_predictions.parquet")


def load(name: str) -> tuple[LoadLSTM, Scaler, dict]:
    meta = json.loads((MODELS_DIR / f"{name}.json").read_text())
    model = LoadLSTM(len(meta["features"]), meta["config"]["hidden_size"])
    model.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt"))
    return model.eval(), Scaler(**meta["scaler"]), meta
