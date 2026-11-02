"""Train the LSTM day-ahead forecaster (same target and split as the LightGBM model).

    python scripts/train_lstm.py                  # total load of full-coverage clients
    python scripts/train_lstm.py --client MT_320  # a single client
"""

import argparse

import pandas as pd

from forecast_explainer import lgbm_model as lm
from forecast_explainer import lstm_model as sm
from forecast_explainer.data import load_series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", default="total", help="'total' or a client id such as MT_320")
    args = parser.parse_args()

    load = load_series(args.client)
    split = lm.Split()
    train_load = load.loc[: split.train_end]
    scaler = sm.Scaler(mean=float(train_load.mean()), std=float(train_load.std()))

    X, y, index = sm.build_sequences(load, scaler)
    masks = sm.split_masks(index, split)
    for name, mask in masks.items():
        print(f"{name:5s} {index[mask].min()} -> {index[mask].max()}  ({mask.sum():,} sequences)")

    config = sm.LstmConfig()
    model = sm.train(X[masks["train"]], y[masks["train"]], X[masks["valid"]], y[masks["valid"]], config)

    test = masks["test"]
    y_test = pd.Series(scaler.unscale(y[test]), index=index[test])
    y_pred = scaler.unscale(sm.predict_scaled(model, X[test]))
    scores = {"lstm": lm.metrics(y_test, y_pred)}
    lgbm_meta_path = f"lgbm_{args.client.lower()}"
    try:
        scores["lightgbm"] = lm.load(lgbm_meta_path)[1]["test_metrics"]["lightgbm"]
    except FileNotFoundError:
        pass
    print(f"\ntest metrics ({y_test.index.min():%Y-%m-%d} -> {y_test.index.max():%Y-%m-%d})")
    print(pd.DataFrame(scores).T.round(2).to_string())

    name = f"lstm_{args.client.lower()}"
    meta = {
        "client": args.client,
        "target": "hourly load (kWh)",
        "horizon": "day-ahead (input window ends 24h before the target hour)",
        "split": split.__dict__,
        "test_metrics": scores,
    }
    sm.save(model, name, scaler, config, meta, pd.DataFrame({"y_true": y_test, "y_pred": y_pred}))
    print(f"\nsaved models/{name}.pt, .json and _predictions.parquet")


if __name__ == "__main__":
    main()
