"""Pure data logic behind the app's forecast chart (no Streamlit here, so plain pytest can test it).

The app shows actual load against both forecasts, circles the largest misses, and lets the user click a point to choose
an hour. These functions prepare that data and read the click.
"""

import pandas as pd

from forecast_explainer import accuracy as ac

SERIES = {"y_true": "Actual", "lightgbm": "LightGBM", "lstm": "LSTM"}


def forecast_table() -> pd.DataFrame:
    """One row per test hour: time, idx, actual (y_true) and both forecasts in kWh, and each model's error in %.

    idx is the hour's position (0, 1, 2, ...). A clicked chart point is identified by this integer rather than by its
    timestamp: an integer reads back identically in every browser time zone, whereas a timestamp could come back
    shifted by the viewer's UTC offset and select the wrong hour.
    """
    frame = ac.error_frame().reset_index(names="time")
    frame["idx"] = range(len(frame))
    return frame


def long_form(table: pd.DataFrame) -> pd.DataFrame:
    """The table as (time, idx, series, kwh, mwh) rows, one per hour and series - the shape a layered line chart wants."""
    long = table.melt(id_vars=["time", "idx"], value_vars=list(SERIES), var_name="series", value_name="kwh")
    long["series"] = long["series"].map(SERIES)
    long["mwh"] = long["kwh"] / 1000
    return long


def largest_misses(table: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The n hours with the biggest forecast error of either model, at most one per calendar day.

    One per day because the worst hours cluster: Christmas Eve 19:00, 20:00 and 21:00 are one story, and listing all
    three would hide the other days. The column `miss` is the signed error (kWh) of whichever model was further off: kWh, like RMSE, so the biggest
    absolute misses rank first, not the biggest percentages on low-load hours.
    """
    t = table.copy()
    t["lightgbm_err"], t["lstm_err"] = t["lightgbm"] - t["y_true"], t["lstm"] - t["y_true"]
    t["miss"] = t[["lightgbm_err", "lstm_err"]].apply(lambda r: r.loc[r.abs().idxmax()], axis=1)
    best_per_day = t.loc[t["miss"].abs().groupby(t["time"].dt.normalize()).idxmax()]
    return best_per_day.reindex(best_per_day["miss"].abs().sort_values(ascending=False).index)[:n]


def hour_from_selection(selection, table: pd.DataFrame) -> pd.Timestamp | None:
    """The hour behind a click on the chart, or None.

    selection: what Streamlit returns for the chart's "pick" selection, e.g. {"pick": [{"idx": 4140}]}. Anything that
               is missing, empty, malformed or points outside the table gives None, so a odd event can never crash the page.
    """
    try:
        points = selection.get("pick") or []
        if not points:
            return None
        return table.set_index("idx").loc[int(points[0]["idx"]), "time"]
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return None
