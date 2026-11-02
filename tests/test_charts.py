"""Tests for the chart builders (no Streamlit, no browser)."""

import json

import altair as alt
import pandas as pd
import pytest

from forecast_explainer import charts, view


@pytest.fixture(scope="module")
def data():
    table = view.forecast_table()
    return table, view.long_form(table), view.largest_misses(table, 5)


def spec(chart):
    alt.data_transformers.disable_max_rows()
    return json.loads(chart.to_json())


def layer_data(s, i):
    """Rows of the dataset used by layer i of a layered spec."""
    return s["datasets"][s["layer"][i]["data"]["name"]]


def test_click_layer_is_on_top_so_rings_cannot_swallow_clicks(data):
    s = spec(charts.forecast_chart(*data, pd.Timestamp("2014-12-24 18:00"), "Christmas week"))
    (param,) = s["params"]                                                    # the only selection in the whole chart
    assert param["name"] == "pick" and param["select"]["nearest"] is True and param["select"]["fields"] == ["idx"]
    assert param["views"][0].endswith(f"_{len(s['layer']) - 1}")              # attached to the LAST (top) layer
    assert s["layer"][-1]["mark"]["opacity"] == 0                             # invisible: it only receives hover and clicks


def test_the_data_is_cut_to_the_window_so_the_y_axis_fits(data):
    table, long, spots = data
    week = spec(charts.forecast_chart(table, long, spots, pd.Timestamp("2014-12-24 18:00"), "Christmas week"))
    whole = spec(charts.forecast_chart(table, long, spots, pd.Timestamp("2014-12-24 18:00"), "Whole period"))
    assert len(layer_data(week, 0)) == 3 * 12 * 24 and len(layer_data(whole, 0)) == 3 * len(table)   # series x hours
    assert len(layer_data(week, -1)) == 12 * 24                               # the click layer is cut the same way
    assert len(layer_data(week, 1)) < len(layer_data(whole, 1)) == 5          # only the circles inside the window


def test_a_custom_range_zooms_to_the_chosen_days_and_falls_back_while_incomplete(data):
    table, long, spots = data
    pick = (pd.Timestamp("2014-10-25").date(), pd.Timestamp("2014-10-27").date())    # the clock-change weekend
    assert charts.window_bounds(charts.CUSTOM, pick) == (pd.Timestamp("2014-10-25"), pd.Timestamp("2014-10-28"))
    assert charts.window_bounds(charts.CUSTOM, (pick[0],)) == charts.window_bounds("Christmas week")   # one date picked so far
    assert charts.window_bounds(charts.CUSTOM, None) == charts.window_bounds("Christmas week")
    zoomed = spec(charts.forecast_chart(table, long, spots, pd.Timestamp("2014-10-26 12:00"), charts.CUSTOM, pick))
    assert len(layer_data(zoomed, 0)) == 3 * 3 * 24 and len(layer_data(zoomed, -1)) == 3 * 24     # three days, all in the window


def test_a_selected_hour_outside_the_window_draws_no_marker(data):
    s = spec(charts.forecast_chart(*data, pd.Timestamp("2014-08-04 10:00"), "Christmas week"))
    rule = next(i for i, l in enumerate(s["layer"]) if l["mark"].get("type") == "rule")
    assert layer_data(s, rule) == []


def test_tooltip_on_the_click_layer_shows_all_three_values_and_a_miss_note(data):
    s = spec(charts.forecast_chart(*data, pd.Timestamp("2014-12-24 18:00"), "Whole period"))
    assert [t["title"] for t in s["layer"][-1]["encoding"]["tooltip"]] == ["Hour", "Actual MWh", "LightGBM MWh", "LSTM MWh", "Note"]
    assert sum(row["note"].startswith("largest miss") for row in layer_data(s, -1)) == 5


def test_drivers_chart_orders_by_size_drops_zero_bars_and_keeps_a_fixed_row_height():
    records = [{"feature": "a", "attribution_kwh": 100.0}, {"feature": "b", "attribution_kwh": -900.0},
               {"feature": "zero", "attribution_kwh": 0.2}, {"feature": "c", "attribution_kwh": 50.0}]
    s = spec(charts.drivers_chart(records, "T"))
    assert s["encoding"]["y"]["sort"] == ["b", "a", "c"]                      # biggest first, 'zero' dropped
    assert s["height"] == {"step": 30}                                       # a one-bar chart is not squashed


def test_style_is_set_in_the_chart_not_left_to_the_page_theme(data):
    s = spec(charts.forecast_chart(*data, pd.Timestamp("2014-12-24 18:00"), "Christmas week"))
    assert s["config"]["background"] == "transparent" and s["config"]["axis"]["gridColor"] == charts.GRID
