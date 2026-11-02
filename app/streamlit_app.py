"""Forecast Detective: look at the forecast, pick an hour, ask why.

    streamlit run app/streamlit_app.py        (from the project root, so .streamlit/config.toml is found)

The page starts with the forecast itself: actual load against both models' forecasts as a time series, with the
largest misses circled, so you can see where something is worth asking about. Then you can
  - click a point on the chart, press a "largest miss" button, or choose a day and hour;
  - press "Explain this hour" (choose LightGBM, LSTM or both), or
  - type a question in plain English and let the agent work out the hour and model, or
  - ask which model to use: for the chart window ("Compare models here") or in words ("Which model for holidays?").
The answer is short: three numbers, the main drivers as a chart, and a few cited lines. Sources, disagreement checks
and the exact evidence the writer saw are one click away under "Details".

All logic lives in the forecast_explainer package; this file only draws things.
"""

import os
from datetime import date

import pandas as pd
import streamlit as st

from forecast_explainer import agent, agent_tools, charts, checks, evidence as ev, llm, narrate, planner, store, view as fv

st.set_page_config(page_title="Forecast Detective", page_icon=":material/search:", layout="wide")

MODEL_CHOICES = {"Both": "both", "LightGBM": "lightgbm", "LSTM": "lstm"}
EXAMPLES = ["Why was Christmas Eve evening so far off?", "Explain the LSTM forecast for 16 October at noon",
            "Which model should I use overall?", "Which model is better in the evening?"]
SEVERITY_ICON = {"high": "🔴 high", "warning": "🟡 warning", "info": "⚪ info", "ok": "🟢 ok"}
RANGE_START, RANGE_END = date(2014, 7, 1), date(2014, 12, 31)


# ----------------------------------------------------------------------------- data (cached, read-only)

@st.cache_data(ttl="1h", max_entries=2, show_spinner=False)
def forecast_table() -> pd.DataFrame:
    """The test-period table (see view.forecast_table), loaded once and shared by all sessions."""
    return fv.forecast_table()


@st.cache_data(ttl="1h", max_entries=2, show_spinner=False)
def forecast_long() -> pd.DataFrame:
    """The same data in long form for the line chart."""
    return fv.long_form(forecast_table())


@st.cache_data(ttl="1h", max_entries=2, show_spinner=False)
def largest_misses() -> pd.DataFrame:
    """The five largest misses, one per day (see view.largest_misses)."""
    return fv.largest_misses(forecast_table(), 5)


@st.cache_data(ttl="1h", max_entries=32, show_spinner=False)
def explanation_for(timestamp: str) -> dict:
    """The saved explanation for one hour (it exists once the hour has been explained, so this is a file read)."""
    return store.get_explanation(timestamp, verbose=False)


# ----------------------------------------------------------------------------- state and callbacks

S = st.session_state
S.setdefault("pick_date", date(2014, 12, 24))
S.setdefault("pick_hour", 18)
S.setdefault("model_choice", "Both")
S.setdefault("view", "Christmas week")
S.setdefault("results", [])       # explanations currently shown (narrate() results)
S.setdefault("recs", [])          # model recommendations currently shown (planner.recommend() results)
S.setdefault("trace", [])         # the agent's tool calls for the last question, if any
S.setdefault("ask_history", [])   # conversation with the agent, so follow-ups like "now only the LSTM" work
S.setdefault("clarify", None)     # a question the agent put to the user
S.setdefault("last_pick", None)   # last chart click handled, so the same click is not handled again after a rerun

# An agent answer moves the pickers to the hour it explained. Widget values can only be changed before the widget is
# drawn in a run, so the answer leaves a "pending" note and it is applied here, at the top of the next run.
if "pending" in S:
    when, models = S.pop("pending")
    chosen = ev.normalize_models(models)
    S.pick_date, S.pick_hour = when.date(), when.hour
    S.model_choice = "Both" if len(chosen) == 2 else ev.LABELS[chosen[0]]


def clear_results() -> None:
    """Hide the shown explanation when the user changes the hour or the model: it would no longer match the pickers.

    A model recommendation (S.recs) is about a window or a kind of hour, not about the picked hour, so it stays.
    """
    S.results, S.trace = [], []


def set_hour(when: pd.Timestamp) -> None:
    """Callback for a 'largest miss' button: move the pickers to that hour."""
    S.pick_date, S.pick_hour = when.date(), when.hour
    clear_results()


def queue_example() -> None:
    """Callback for the example pills: turn the pick into a question to ask, and reset the pills."""
    S.queued_ask, S.example = S.example, None


# ----------------------------------------------------------------------------- the page: forecast first

has_key = bool(os.getenv("OPENAI_API_KEY"))
spots = largest_misses()

with st.sidebar:
    st.header("Settings")
    llm_model = st.text_input("LLM model", value=llm.default_model(), help="Used to drive the agent and to write explanations.")
    use_rag = st.toggle("Use the knowledge base", value=True, help="Off: the writer gets no retrieved sources, for comparison.")
    start, end = agent_tools.valid_range()
    st.caption(f"Explainable hours: {start:%d %b} to {end:%d %b %Y}, the models' test period.")
    if not has_key:
        st.warning("No OPENAI_API_KEY found, so explanations are off. Add it to `.env` and restart.", icon=":material/key_off:")

st.title("Forecast Detective", anchor=False)
st.caption("Hourly electricity load in Portugal, day-ahead forecasts from LightGBM and an LSTM. Pick an hour, then ask why the forecast missed.")

with st.container(border=True):
    top = st.container(horizontal=True, vertical_alignment="center")
    top.markdown("**Actual load vs forecasts**")
    view = top.segmented_control("Window", list(charts.VIEWS), key="view", label_visibility="collapsed")
    custom = None
    if view == charts.CUSTOM:        # zoom to any stretch of the test period; the chart and "Compare models here" both use it
        custom = st.date_input("Range", value=(date(2014, 12, 20), date(2014, 12, 31)), min_value=RANGE_START, max_value=RANGE_END,
                               key="custom_range", help="First and last day to show. Pick both dates.")
    compare_clicked = top.button("Compare models here", icon=":material/compare_arrows:", disabled=not has_key,
                                 help="Which model is more accurate in the window shown, and where to switch.")
    hour = pd.Timestamp(S.pick_date) + pd.Timedelta(hours=int(S.pick_hour))
    chart_event = st.altair_chart(
        charts.forecast_chart(forecast_table(), forecast_long(), spots, hour, view or "Christmas week", custom),
        theme=None, on_select="rerun", selection_mode=["pick"], key="forecast_chart",
        alt="Hourly load in MWh: actual against LightGBM and LSTM forecasts. Yellow circles mark the largest misses; "
            "the dotted line is the selected hour.")
    st.caption("Yellow circles are the largest misses. Click the chart, use a button below, or ask in words.")

clicked = fv.hour_from_selection(chart_event.selection, forecast_table()) if chart_event else None
if clicked is not None and clicked != S.last_pick:      # a new click: move the pickers there and redraw
    S.last_pick = clicked
    S.pick_date, S.pick_hour = clicked.date(), clicked.hour
    clear_results()
    st.rerun()

with st.container(horizontal=True, vertical_alignment="center"):
    st.caption("Largest misses")
    for _, r in spots.iterrows():
        st.button(f"{r['time']:%d %b %H:%M} · {r['miss']:+,.0f} kWh", key=f"spot{int(r['idx'])}", on_click=set_hour,
                  args=(r["time"],), icon=":material/my_location:")

with st.container(border=True):
    pickers = st.container(horizontal=True, vertical_alignment="bottom")
    pickers.date_input("Day", min_value=RANGE_START, max_value=RANGE_END, key="pick_date", on_change=clear_results)
    pickers.select_slider("Hour", options=list(range(24)), format_func=lambda h: f"{h:02d}:00", key="pick_hour",
                          on_change=clear_results, width=260)
    pickers.segmented_control("Explain", list(MODEL_CHOICES), key="model_choice", on_change=clear_results)
    explain_clicked = pickers.button("Explain this hour", type="primary", icon=":material/search_insights:", disabled=not has_key)

    row = forecast_table().set_index("time").loc[hour]
    cards = st.columns(3)
    cards[0].metric("Actual", f"{row['y_true']:,.0f} kWh", border=True, height="stretch")
    cards[1].metric("LightGBM", f"{row['lightgbm']:,.0f} kWh", f"{row['lightgbm_err_pct']:+.1f}% vs actual", delta_color="off",
                    border=True, height="stretch")
    cards[2].metric("LSTM", f"{row['lstm']:,.0f} kWh", f"{row['lstm_err_pct']:+.1f}% vs actual", delta_color="off",
                    border=True, height="stretch")

with st.container(border=True):
    if S.clarify:
        st.info(S.clarify, icon=":material/help:")
    with st.form("ask", border=False, clear_on_submit=True):
        cols = st.columns([6, 1], vertical_alignment="bottom")
        question = cols[0].text_input("Or ask in words", placeholder="e.g. Why was Christmas Eve evening so far off? Only the LSTM, please.")
        asked = cols[1].form_submit_button("Ask", icon=":material/send:", width="stretch", disabled=not has_key)
    st.pills("Try", EXAMPLES, key="example", on_change=queue_example, label_visibility="collapsed")

result_box = st.container()   # explanations appear here; the slow work below fills it last


# ----------------------------------------------------------------------------- explanations

def render_result(n: dict) -> None:
    """One explanation: headline, check badges, drivers chart beside the short text, details tucked away."""
    when, models = pd.Timestamp(n["timestamp"]), n["models"]
    explanation = explanation_for(str(when))
    verdict = n["checks"]
    with st.container(border=True):
        head = st.container(horizontal=True, vertical_alignment="center")
        head.markdown(f"**{when:%a %d %b %Y, %H:%M}** · {' + '.join(ev.LABELS[m] for m in models)}")
        if verdict["passed"]:
            head.badge("Checks passed", icon=":material/verified:", color="green",
                       help="Every citation exists and every number is in the evidence it cites.")
        else:
            head.badge("Checks failed", icon=":material/error:", color="red", help="See Details for what could not be verified.")
        if verdict.get("causal_without_source"):
            head.badge("Cause without source", icon=":material/help:", color="orange",
                       help="A sentence states a cause with no cited source. Model attributions describe the model, not the real world.")

        left, right = st.columns([3, 2])
        with left:
            if "lightgbm" in models:
                st.altair_chart(charts.drivers_chart(explanation["lightgbm"]["shap"], "LightGBM · SHAP"), theme=None, width="stretch",
                                alt="Bar chart of the inputs that moved the LightGBM forecast most, in kWh")
            if "lstm" in models:
                st.altair_chart(charts.drivers_chart(explanation["lstm"]["ig_by_feature"], "LSTM · Integrated Gradients"), theme=None, width="stretch",
                                alt="Bar chart of the inputs that moved the LSTM forecast most, in kWh")
        with right:
            st.markdown(n["text"])

        with st.expander("Details", icon=":material/lab_profile:"):
            if not verdict["passed"]:
                problems = []
                if verdict["unsupported_numbers"]:
                    problems.append("numbers not found in the cited evidence: " + ", ".join(x["number"] for x in verdict["unsupported_numbers"]))
                if verdict["unknown_citations"]:
                    problems.append("unknown sources cited: " + ", ".join(verdict["unknown_citations"]))
                if verdict["flags_not_cited"]:
                    problems.append("disagreements exist but are not cited")
                st.warning("This text could not be fully verified: " + "; ".join(problems) + ".", icon=":material/error:")
            for sentence in verdict.get("causal_without_source", []):
                st.caption(f"States a cause without a source: {sentence}")

            tab_names = ["Sources", "Disagreements", "Evidence", "More charts"] + (["Agent steps"] if S.trace else [])
            tabs = dict(zip(tab_names, st.tabs(tab_names)))
            with tabs["Sources"]:
                cited = set(checks.citations(n["text"]))
                for s in sorted(n["sources"], key=lambda s: s["sid"] not in cited):
                    where = f", p.{s['page']}" if s["page"] else f" › {s['section']}"
                    with st.expander(f"{'Cited' if s['sid'] in cited else 'Retrieved'} · [{s['sid']}] {s['citation']}{where}"):
                        st.write(s["text"])
                if not n["sources"]:
                    st.caption("No sources: the knowledge base was off.")
            with tabs["Disagreements"]:
                flags = pd.DataFrame([{"severity": SEVERITY_ICON[f["severity"]], "check": f["type"], "finding": f["message"]}
                                      for f in ev.restrict_to_models(explanation, models)["disagreements"]])
                st.dataframe(flags, hide_index=True, alt="Checks comparing the models and explainers, with severity")
            with tabs["Evidence"]:
                st.caption("Exactly what the writer was given. It may use nothing else.")
                for block_id, text in n["evidence"].items():
                    with st.expander(f"[{block_id}]"):
                        st.text(text)
            with tabs["More charts"]:
                if "lightgbm" in models:
                    st.altair_chart(charts.drivers_chart(explanation["lightgbm"]["lime"], "LightGBM · LIME"), theme=None, width="stretch",
                                    alt="Bar chart of LIME weights for the LightGBM forecast, in kWh")
                if "lstm" in models:
                    tsh = [r for r in explanation["lstm"]["timeshap_features"] if not r["feature"].startswith("(")]
                    st.altair_chart(charts.drivers_chart(tsh, "LSTM · TimeSHAP"), theme=None, width="stretch",
                                    alt="Bar chart of TimeSHAP values for the LSTM forecast, in kWh")
                st.caption("SHAP is measured against the average forecast over all training hours; Integrated Gradients and TimeSHAP against a "
                           "typical non-holiday weekday at the same hour and season; LIME has no reference. Compare rankings and directions, not sizes.")
            if "Agent steps" in tabs:
                with tabs["Agent steps"]:
                    for step in S.trace:
                        st.markdown(f"**{step['tool']}** · {step['seconds']} s  \n`{step['arguments']}`")
            st.caption(f"Written by {n['model']} in {n['n_attempts']} attempt(s) · {n['usage']['prompt_tokens']:,} + "
                       f"{n['usage']['completion_tokens']:,} tokens · knowledge base {'on' if n['use_rag'] else 'off'}")


def render_recommendation(r: dict) -> None:
    """One model recommendation: the code-made decision first, then the evidence as a chart and a few cited lines."""
    d, verdict = r["decision"], r["checks"]
    with st.container(border=True):
        head = st.container(horizontal=True, vertical_alignment="center")
        head.markdown("**Which model?**")
        if verdict["passed"]:
            head.badge("Checks passed", icon=":material/verified:", color="green",
                       help="Every citation exists and every number is in the evidence it cites.")
        else:
            head.badge("Checks failed", icon=":material/error:", color="red", help="See Details for what could not be verified.")
        # The headline is written by code from a statistical test, not by the LLM; its colour says how sure the test is.
        shout = {"clear": st.success, "no_clear_difference": st.info, "too_few_days": st.warning}[d["confidence"]]
        shout(d["headline"], icon=":material/balance:")
        for line in d["switches"]:
            st.caption(line)

        left, right = st.columns([3, 2])
        with left:
            st.altair_chart(charts.slice_chart(r["slices"]), theme=None, width="stretch",
                            alt="Grouped bars of LightGBM and LSTM error for each kind of hour, with whether the difference is reliable")
        with right:
            st.markdown(r["text"])

        with st.expander("Details", icon=":material/lab_profile:"):
            tabs = dict(zip(["Numbers", "Evidence", "Sources"], st.tabs(["Numbers", "Evidence", "Sources"])))
            with tabs["Numbers"]:
                o = r["overall"]
                table = pd.DataFrame([{"kind of hour": x["label"], "hours": x["n_hours"], "days": x["n_days"],
                                       "LightGBM RMSE (kWh)": round(x["rmse"]["lightgbm"]), "LSTM RMSE (kWh)": round(x["rmse"]["lstm"]),
                                       "verdict": {"clear": f"{ev.LABELS[x['leaning']]} clearly", "too_few_days": "too few days",
                                                   "no_clear_difference": "no clear difference"}[x["confidence"]]}
                                      for x in [{"label": "all hours in scope", **o}, *r["slices"]]])
                st.dataframe(table, hide_index=True, alt="Accuracy of both models by kind of hour, with a verdict for each")
                st.caption("A verdict is only 'clear' when a 95% interval, from resampling whole days, excludes zero and the gap is at "
                           f"least {planner.MIN_EFFECT:g}% lower in RMSE. Groups with fewer than {planner.MIN_DAYS} days are not compared.")
            with tabs["Evidence"]:
                st.caption("Exactly what the writer was given. It may use nothing else.")
                for block_id, text in r["evidence"].items():
                    with st.expander(f"[{block_id}]"):
                        st.text(text)
            with tabs["Sources"]:
                cited = set(checks.citations(r["text"]))
                for src in sorted(r["sources"], key=lambda x: x["sid"] not in cited):
                    where = f", p.{src['page']}" if src["page"] else f" › {src['section']}"
                    with st.expander(f"{'Cited' if src['sid'] in cited else 'Retrieved'} · [{src['sid']}] {src['citation']}{where}"):
                        st.write(src["text"])
                if not r["sources"]:
                    st.caption("No sources: the knowledge base was off.")
            st.caption(f"Written by {r['model']} in {r['n_attempts']} attempt(s) · {r['usage']['prompt_tokens']:,} + "
                       f"{r['usage']['completion_tokens']:,} tokens · knowledge base {'on' if r['use_rag'] else 'off'}")


ask_text = S.pop("queued_ask", None) or (question.strip() if asked and question.strip() else None)

if explain_clicked:
    with result_box:
        status = st.status("Explaining this hour…", expanded=True)
        try:
            status.write("Running the explainers (cached after the first time), retrieving sources, writing…")
            S.results = [narrate.narrate(hour, models=MODEL_CHOICES[S.model_choice or "Both"], model=llm_model, use_rag=use_rag)]
            S.trace, S.recs = [], []
            status.update(label="Done", state="complete", expanded=False)
        except Exception as exc:  # API, network or key problems: say so and keep the page alive
            status.update(label="Failed", state="error")
            st.error(f"Something went wrong: {type(exc).__name__}: {exc}")

elif compare_clicked:
    with result_box:
        status = st.status("Comparing the two models…", expanded=True)
        try:
            bounds = charts.window_bounds(view or "Christmas week", custom)       # None = the whole test period
            first, last = (None, None) if bounds is None else (f"{bounds[0]:%Y-%m-%d}", f"{min(bounds[1] - pd.Timedelta(days=1), pd.Timestamp(RANGE_END)):%Y-%m-%d}")
            status.write("Testing the accuracy gap on whole days, checking each kind of hour, writing the reasoning…")
            S.recs = [planner.recommend(first, last, model=llm_model, use_rag=use_rag)]
            S.results, S.trace = [], []
            status.update(label="Done", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Failed", state="error")
            st.error(f"Something went wrong: {type(exc).__name__}: {exc}")

elif ask_text:
    with result_box:
        status = st.status("Working out what you mean…", expanded=True)
        outcome = None
        try:
            outcome = agent.run_agent(ask_text, S.ask_history, model=llm_model, use_rag=use_rag,
                                      on_step=lambda step: status.write(f"{step['tool']} · {step['seconds']} s"))
            S.ask_history += [{"role": "user", "content": ask_text}, {"role": "assistant", "content": outcome["history_text"]}]
            S.clarify = outcome["clarification"] if outcome["stopped"] == "clarify" else None
            S.recs = outcome.get("recommendations", [])
            if outcome["narratives"]:
                S.results, S.trace = outcome["narratives"], outcome["trace"]
                S.pending = (pd.Timestamp(outcome["narratives"][0]["timestamp"]), outcome["narratives"][0]["models"])
            elif S.recs:
                S.results, S.trace = [], []
            elif outcome["message"] and not S.clarify:
                S.results, S.trace = [], []
                st.info(outcome["message"])
            status.update(label="Done", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Failed", state="error")
            st.error(f"Something went wrong: {type(exc).__name__}: {exc}")
    if outcome and (outcome["narratives"] or S.clarify):
        st.rerun()   # redraw so the chart and pickers show the hour the agent chose

with result_box:
    for r in S.recs:
        render_recommendation(r)
    for n in S.results:
        render_result(n)
    if not S.results and not S.recs and not explain_clicked and not compare_clicked and not ask_text:
        st.caption("Choose an hour and press Explain, compare the models for the window above, or ask a question.")
