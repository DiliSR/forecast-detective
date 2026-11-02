"""Train the LightGBM day-ahead forecaster.

    python scripts/train_lgbm.py                  # total load of full-coverage clients
    python scripts/train_lgbm.py --client MT_320  # a single client
"""

import argparse
from dataclasses import asdict

import pandas as pd

from forecast_explainer import lgbm_model as lm
from forecast_explainer.data import full_coverage_clients, load_hourly, load_series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", default="total", help="'total' or a client id such as MT_320")
    args = parser.parse_args()

    load = load_series(args.client)
    X, y = lm.build_dataset(load)
    split = lm.Split()
    parts = lm.split_frame(X, y, split)
    for name, (Xp, _) in parts.items():
        print(f"{name:5s} {Xp.index.min()} -> {Xp.index.max()}  ({len(Xp):,} rows)")

    params = lm.LgbmParams()
    booster = lm.train(*parts["train"], *parts["valid"], params=params)
    print(f"best iteration: {booster.best_iteration}")

    X_test, y_test = parts["test"]
    y_pred = booster.predict(X_test, num_iteration=booster.best_iteration)

    scores = {"lightgbm": lm.metrics(y_test, y_pred)}
    scores |= {name: lm.metrics(y_test, pred) for name, pred in lm.baselines(X_test).items()}
    print(f"\ntest metrics ({X_test.index.min():%Y-%m-%d} -> {X_test.index.max():%Y-%m-%d})")
    print(pd.DataFrame(scores).T.round(2).to_string())

    importance = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
    print("\ntop features by gain\n", (importance / importance.sum()).sort_values(ascending=False).head(10).round(3).to_string())

    name = f"lgbm_{args.client.lower()}"
    clients = full_coverage_clients(load_hourly()) if args.client == "total" else [args.client]
    meta = {
        "client": args.client,
        "n_clients": len(clients),
        "clients": clients,
        "target": "hourly load (kWh)",
        "horizon": "day-ahead (all load features lagged >= 24h)",
        "features": booster.feature_name(),
        "split": asdict(split),
        "params": params.to_lgb(),
        "best_iteration": booster.best_iteration,
        "test_metrics": scores,
    }
    predictions = X_test.assign(y_true=y_test, y_pred=y_pred)
    lm.save(booster, name, meta, predictions)
    print(f"\nsaved models/{name}.txt, .json and _predictions.parquet")


if __name__ == "__main__":
    main()
