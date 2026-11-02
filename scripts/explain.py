"""Compute local explanations for one forecast hour and save them as JSON.

    python scripts/explain.py "2014-12-25 18:00"
    python scripts/explain.py "2014-10-16 12:00" --client total --lime-repeats 10

Needs the trained models: scripts/train_lgbm.py and scripts/train_lstm.py.
"""

import argparse
import warnings

import pandas as pd

from forecast_explainer.explain import explain_forecast

warnings.filterwarnings("ignore")  # LIME/TimeSHAP emit sklearn/pandas deprecation noise


def top(df: pd.DataFrame, n: int, columns: list[str]) -> str:
    return df.head(n)[columns].round(1).to_string(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("timestamp", help='target hour, e.g. "2014-12-25 18:00"')
    parser.add_argument("--client", default="total")
    parser.add_argument("--lime-repeats", type=int, default=5)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    e = explain_forecast(args.timestamp, args.client, lime_repeats=args.lime_repeats)
    lg, ls = e.lgbm, e.lstm
    print(f"\n=== {e.timestamp:%A %Y-%m-%d %H:%M}  ({e.client}) ===")
    print(f"actual {e.actual_kwh:,.0f} kWh | LightGBM {lg['prediction_kwh']:,.0f} | LSTM {ls['prediction_kwh']:,.0f}")

    print(f"\n--- LightGBM · SHAP (average forecast {lg['expected_value_kwh']:,.0f} kWh + contributions) ---")
    print(top(lg["shap"], args.top, ["feature", "value", "attribution_kwh"]))

    r2 = lg["lime_fit"]["surrogate_r2_per_seed"]
    print(f"\n--- LightGBM · LIME (mean of {len(r2)} seeds; surrogate R² {min(r2):.2f}-{max(r2):.2f}) ---")
    print(top(lg["lime"], args.top, ["feature", "condition", "attribution_kwh", "attribution_std", "rank_min", "rank_max"]))

    print(f"\n--- LSTM · Integrated Gradients (baseline forecast {ls['baseline_prediction_kwh']:,.0f} kWh, "
          f"convergence delta {ls['ig_convergence_delta_kwh']:,.1f} kWh) ---")
    print("by feature:\n" + ls["ig_by_feature"].sort_values("attribution_kwh", key=abs, ascending=False).round(0).to_string(index=False))
    by_step = ls["ig_by_timestep"].sort_values("attribution_kwh", key=abs, ascending=False)
    print("most influential past hours (own features only):\n" + by_step.head(5).round(0).to_string(index=False))

    print(f"\n--- LSTM · TimeSHAP ({ls['timeshap_events_kept']} recent hours kept, older ones pruned) ---")
    print("events:\n" + top(ls["timeshap_events"], 6, ["event", "timestamp", "attribution_kwh"]))
    print("features:\n" + top(ls["timeshap_features"], 6, ["feature", "attribution_kwh"]))
    print("cells:\n" + top(ls["timeshap_cells"], 6, ["event", "feature", "attribution_kwh"]))

    print(f"\nsaved {e.save()}")


if __name__ == "__main__":
    main()
