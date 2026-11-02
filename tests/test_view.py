"""Plain tests for the chart's data logic (no Streamlit)."""

import pandas as pd
import pytest

from forecast_explainer import view


@pytest.fixture(scope="module")
def table():
    return view.forecast_table()


def test_table_covers_the_test_period_with_a_running_index(table):
    assert table["time"].min() == pd.Timestamp("2014-07-01 00:00") and table["time"].max() == pd.Timestamp("2014-12-31 23:00")
    assert table["idx"].tolist() == list(range(len(table)))
    assert {"y_true", "lightgbm", "lstm", "lightgbm_err_pct", "lstm_err_pct"} <= set(table.columns)


def test_long_form_has_three_series_per_hour(table):
    long = view.long_form(table)
    assert len(long) == 3 * len(table) and set(long["series"]) == {"Actual", "LightGBM", "LSTM"}
    assert long["mwh"].iloc[0] == pytest.approx(long["kwh"].iloc[0] / 1000)


def test_largest_misses_are_sorted_one_per_day_and_include_christmas(table):
    misses = view.largest_misses(table, 5)
    assert len(misses) == 5 and misses["time"].dt.normalize().is_unique
    assert misses["miss"].abs().is_monotonic_decreasing
    days = set(misses["time"].dt.strftime("%m-%d"))
    assert {"12-24", "12-25"} <= days                  # the days this whole project is about


def test_miss_is_the_signed_error_of_the_worse_model(table):
    row = view.largest_misses(table, 1).iloc[0]
    worse = max(abs(row["lightgbm"] - row["y_true"]), abs(row["lstm"] - row["y_true"]))
    assert abs(row["miss"]) == pytest.approx(worse)


def test_a_click_is_turned_into_its_hour(table):
    idx = int(table.loc[table["time"] == pd.Timestamp("2014-12-24 18:00"), "idx"].iloc[0])
    assert view.hour_from_selection({"pick": [{"idx": idx}]}, table) == pd.Timestamp("2014-12-24 18:00")


@pytest.mark.parametrize("selection", [None, {}, {"pick": []}, {"pick": [{"idx": 10**9}]}, {"pick": [{"x": 1}]},
                                       {"pick": ["junk"]}, "not a dict", {"pick": [{"idx": "abc"}]}])
def test_odd_selections_never_crash_and_give_none(table, selection):
    assert view.hour_from_selection(selection, table) is None
