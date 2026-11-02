"""The app's Altair charts, built without Streamlit so they can be tested and rendered to an image on their own.

Two charts:
  forecast_chart   actual load against both forecasts over time, largest misses circled, selected hour marked,
                   and an invisible top layer that turns a click into an hour index.
  drivers_chart    the inputs that moved one forecast most, as horizontal bars (blue up, orange down).

Style is set here (`style`) and the app passes theme=None to Streamlit, so the charts look identical on the first
draw and after every update. With Streamlit's own theme the first draw was styled and later redraws fell back to plain
Vega defaults (white panel, different fonts), which made the page look inconsistent.
"""

import altair as alt
import pandas as pd

from forecast_explainer.view import SERIES

SERIES_LABELS = {"lightgbm": "LightGBM", "lstm": "LSTM"}   # model key -> name shown on charts

BLUE, ORANGE, INK, MUTED, YELLOW, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#eda100", "#e6e5e0"

# Time window shown on the forecast chart: (first day, day after the last), or None for the whole test period.
# "Custom range" has no fixed dates: the user picks them (see window_bounds), so its entry is the marker CUSTOM.
CUSTOM = "Custom range"
VIEWS = {"Christmas week": ((2014, 12, 20), (2015, 1, 1)), "December": ((2014, 12, 1), (2015, 1, 1)), "Whole period": None, CUSTOM: "custom"}


def style(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Apply the app's look: transparent background, light grid, muted axis text, compact legend and title."""
    return (chart.configure(background="transparent")
            .configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor=GRID, tickColor=GRID,
                            labelFontSize=11, titleFontSize=12, titleFontWeight="normal")
            .configure_legend(labelColor=INK, labelFontSize=12, symbolStrokeWidth=3)
            .configure_title(color=INK, fontSize=14, fontWeight=600, anchor="start"))


def window_bounds(view: str, custom: tuple | None = None) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """First and last timestamp of a view, or None for the whole period.

    custom: (first day, last day) the user chose, both included, used only when view == "Custom range". The end
            becomes midnight after the last day so that day's 24 hours are inside the window, like the fixed views.
            Without a complete choice (a date range widget returns one date while the user is still clicking) the
            Christmas week is shown instead of failing.
    """
    window = VIEWS[view]
    if window == "custom":
        if not custom or len(custom) != 2:
            window = VIEWS["Christmas week"]
        else:
            return pd.Timestamp(custom[0]).normalize(), pd.Timestamp(custom[1]).normalize() + pd.Timedelta(days=1)
    return None if window is None else tuple(pd.Timestamp(*w) for w in window)


def forecast_chart(table: pd.DataFrame, long: pd.DataFrame, spots: pd.DataFrame, selected: pd.Timestamp, view: str,
                   custom: tuple | None = None) -> alt.LayerChart:
    """Actual load and both forecasts over time, largest misses circled, the selected hour marked.

    table:    view.forecast_table() (one row per hour, wide).
    long:     view.long_form(table) (one row per hour and series).
    spots:    view.largest_misses(table), the circled hours.
    selected: the hour marked with a dotted line and a dot.
    view:     a key of VIEWS.

    Layers, bottom to top: (1) the three lines; (2) rings on the largest misses; (3) a dotted rule and a dot on the
    selected hour; (4) invisible points that receive hover and clicks. The click layer must be on TOP: in Vega a layer
    above it would swallow the click (an earlier version put the rings above it, so clicking a circled miss - the most
    natural click - did nothing). The selection is on `idx` and `nearest=True` picks the closest hour wherever you
    click. Because the top layer receives the hover, its tooltip shows all three values at that hour.

    The DATA is cut to the window, not just the x-axis: Vega fits the y-axis to every point in a layer, so leaving the
    other months in would stretch the axis to their peaks and squash the days being looked at.
    """
    bounds = window_bounds(view, custom)
    in_window = (lambda df: df[(df["time"] >= bounds[0]) & (df["time"] < bounds[1])]) if bounds else (lambda df: df)

    x_scale = alt.Scale(domain=[alt.DateTime(year=t.year, month=t.month, date=t.day) for t in bounds]) if bounds else alt.Scale()
    x = alt.X("time:T", title=None, scale=x_scale, axis=alt.Axis(format="%d %b", labelOverlap=True, grid=False))
    # The y-axis title angle is pinned: left to Vega it came out rotated on one draw and horizontal (on top of the
    # legend) on the next. The click layer below repeats the same axis so the two layers agree.
    y_axis = alt.Axis(tickCount=6, titleAngle=270, titleAlign="center", titlePadding=8)
    y = alt.Y("mwh:Q", title="MWh per hour", scale=alt.Scale(zero=False, nice=True), axis=y_axis)
    colour = alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                       scale=alt.Scale(domain=list(SERIES.values()), range=[INK, BLUE, ORANGE]))
    lines = alt.Chart(in_window(long)).mark_line(clip=True).encode(
        x=x, y=y, color=colour, strokeWidth=alt.condition(alt.datum.series == "Actual", alt.value(2.2), alt.value(1.4)))

    ring_data = in_window(spots.assign(mwh=spots["y_true"] / 1000))
    rings = alt.Chart(ring_data).mark_point(shape="circle", filled=False, size=170, strokeWidth=2, color=YELLOW, clip=True).encode(x=x, y=y)

    marker = pd.DataFrame({"time": [selected], "mwh": [table.set_index("time").loc[selected, "y_true"] / 1000]})
    if bounds and not (bounds[0] <= selected < bounds[1]):
        marker = marker.iloc[0:0]   # selected hour is outside this window: draw nothing (it would stretch the y-axis)
    rule = alt.Chart(marker).mark_rule(strokeDash=[4, 4], color=MUTED).encode(x=x)
    dot = alt.Chart(marker).mark_point(filled=True, size=110, color=INK, clip=True).encode(x=x, y=y)

    wide = table[["time", "idx"]].assign(actual=table["y_true"] / 1000, lightgbm=table["lightgbm"] / 1000, lstm=table["lstm"] / 1000)
    notes = {int(r["idx"]): f"largest miss ({r['miss']:+,.0f} kWh)" for _, r in spots.iterrows()}
    wide["note"] = wide["idx"].map(notes).fillna("")
    pick = alt.selection_point(name="pick", fields=["idx"], nearest=True, on="click")
    click_targets = alt.Chart(in_window(wide)).mark_point(opacity=0, size=220, clip=True).encode(
        x=x, y=alt.Y("actual:Q", title="MWh per hour", scale=alt.Scale(zero=False, nice=True), axis=y_axis),
        tooltip=[alt.Tooltip("time:T", title="Hour", format="%a %d %b %H:%M"), alt.Tooltip("actual:Q", title="Actual MWh", format=",.1f"),
                 alt.Tooltip("lightgbm:Q", title="LightGBM MWh", format=",.1f"), alt.Tooltip("lstm:Q", title="LSTM MWh", format=",.1f"),
                 alt.Tooltip("note:N", title="Note")]).add_params(pick)
    return style(alt.layer(lines, rings, rule, dot, click_targets).properties(height=330))


def drivers_chart(records: list[dict], title: str, n: int = 6) -> alt.Chart:
    """Horizontal bars of the n largest attributions (kWh): blue pushed the forecast up, orange pushed it down.

    records: attribution rows with `feature` and `attribution_kwh`.
    Rows that round to 0 kWh are dropped (a zero bar says nothing). Row height is a fixed step, so a chart with one bar
    is as readable as one with six; an explicit total height collapsed the one-bar LSTM chart to nothing.
    """
    df = pd.DataFrame(records)[["feature", "attribution_kwh"]].rename(columns={"attribution_kwh": "kwh"})
    df = df.reindex(df["kwh"].abs().sort_values(ascending=False).index)[:n]
    df = df[df["kwh"].round() != 0]
    chart = alt.Chart(df).mark_bar(size=18).encode(
        y=alt.Y("feature:N", sort=df["feature"].tolist(), title=None, axis=alt.Axis(labelLimit=240, labelOverlap=False, grid=False)),
        x=alt.X("kwh:Q", title="kWh vs reference"),
        color=alt.condition(alt.datum.kwh > 0, alt.value(BLUE), alt.value(ORANGE)),
        tooltip=[alt.Tooltip("feature:N", title="Input"), alt.Tooltip("kwh:Q", title="kWh", format="+,.0f")],
    ).properties(title=title, height=alt.Step(30))
    return style(chart)


def slice_chart(rows: list[dict], title: str = "RMSE by kind of hour") -> alt.Chart:
    """Grouped horizontal bars: both models' RMSE (kWh) for each kind of hour, with the planner's verdict in the row label.

    rows: planner.analyse()["slices"], each with `label`, `rmse` (per model), `confidence`, `winner`, `n_days`.
    The verdict is part of the label ("evening 18-23 · no clear difference") because a bar chart alone invites reading
    a small gap as a win; the label says whether the gap is real, and a group with one day says so.
    Same colours as the forecast chart (blue = LightGBM, orange = LSTM), so a model keeps its colour across the page.
    """
    def verdict(r: dict) -> str:
        if r["confidence"] == "clear":
            return f"{SERIES_LABELS[r['winner']]} clearly"
        if r["confidence"] == "too_few_days":
            return f"{r['n_days']} day{'s' if r['n_days'] != 1 else ''}, not reliable"
        return "no clear difference"

    records = [{"slice": f"{r['label']} · {verdict(r)}", "model": SERIES_LABELS[m], "rmse": r["rmse"][m]} for r in rows for m in r["rmse"]]
    order = [f"{r['label']} · {verdict(r)}" for r in rows]
    chart = alt.Chart(pd.DataFrame(records)).mark_bar(size=11).encode(
        y=alt.Y("slice:N", sort=order, title=None, axis=alt.Axis(labelLimit=300, labelOverlap=False, grid=False)),
        yOffset=alt.YOffset("model:N", sort=list(SERIES_LABELS.values())),
        x=alt.X("rmse:Q", title="RMSE (kWh)"),
        color=alt.Color("model:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=list(SERIES_LABELS.values()), range=[BLUE, ORANGE])),
        tooltip=[alt.Tooltip("slice:N", title="Kind of hour"), alt.Tooltip("model:N", title="Model"),
                 alt.Tooltip("rmse:Q", title="RMSE kWh", format=",.0f")],
    ).properties(title=title, height=alt.Step(30))
    return style(chart)
