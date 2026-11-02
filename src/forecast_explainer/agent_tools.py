"""The tools the explanation agent may call. Each is a plain Python function with validation; the agent's LLM only
chooses which to call and with what arguments - it never computes or recalls forecast numbers itself.

    get_forecasts(date)                     the 24 hours of one day: actual, both forecasts, errors  (lets the agent pick an hour)
    explain_hour(timestamp, models, ...)    runs the explainers + RAG + writer + checker for one hour
    recommend_model(start, end, focus)      which model to use for a period or kind of hour (the planner)
    ask_user(question)                      stops the run and puts one question to the user

Errors are returned as {"error": "..."} (never raised) so the LLM can read them and correct itself, e.g. after
choosing a date outside the data range.
"""

import json

import pandas as pd

from forecast_explainer import accuracy as ac, evidence as ev, narrate, planner, registry
from forecast_explainer.features import calendar_features

MAX_EXPLANATIONS_PER_REQUEST = 3  # each costs seconds of compute and an LLM call; bounds a runaway agent
MAX_RECOMMENDATIONS_PER_REQUEST = 2

TOOLS = [
    {"type": "function", "function": {
        "name": "get_forecasts",
        "description": "The 24 hourly rows of one day: actual load, LightGBM and LSTM forecasts (kWh) and their errors (%). "
                       "Use it to choose an hour when the user gives only a day or a vague time such as 'evening'.",
        "parameters": {"type": "object", "properties": {
            "date": {"type": "string", "description": "Day as YYYY-MM-DD."}}, "required": ["date"]}}},
    {"type": "function", "function": {
        "name": "explain_hour",
        "description": "Explain the forecast of ONE hour in natural language, with citations. Runs the explainers, retrieves "
                       "background and writes a checked explanation, which is shown to the user automatically.",
        "parameters": {"type": "object", "properties": {
            "timestamp": {"type": "string", "description": "Target hour as 'YYYY-MM-DD HH:00'."},
            "models": {"type": "string", "enum": ["both", "lightgbm", "lstm"], "description": "Which forecasting model(s) to explain. Default both."},
            "question": {"type": "string", "description": "Optional: the user's own question about this forecast, to be answered first."}},
            "required": ["timestamp"]}}},
    {"type": "function", "function": {
        "name": "recommend_model",
        "description": "Which forecasting model (LightGBM or LSTM) to use, overall or for a period or kind of hour. Compares their "
                       "accuracy with a statistical test, says how sure it is, and where to switch. The decision is computed by code "
                       "and the reasoning is shown to the user automatically.",
        "parameters": {"type": "object", "properties": {
            "start_date": {"type": "string", "description": "First day to compare, YYYY-MM-DD. Omit for the whole test period."},
            "end_date": {"type": "string", "description": "Last day to compare, YYYY-MM-DD. Omit for the whole test period."},
            "focus": {"type": "string", "enum": planner.FOCUS_CHOICES,
                      "description": "Compare only this kind of hour. Omit to compare all hours."},
            "question": {"type": "string", "description": "The user's own words, so the answer addresses what was asked."}}}}},
    {"type": "function", "function": {
        "name": "ask_user",
        "description": "Ask the user ONE short clarifying question when the request is genuinely ambiguous. Ends this run.",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}},
]


def valid_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last hour both models forecast (the test period). Only these hours can be explained:
    the models were trained on earlier data, so explaining those would describe in-sample fits."""
    index = registry.all_predictions().index
    return index.min(), index.max()


class ToolBox:
    """The tools plus the state of one agent run (the explanations it produced, a pending question).

    model:      LLM used by the writer inside explain_hour (default LLM_MODEL).
    use_rag:    False runs the writer without retrieved sources (ablation).
    llm_client: injected fake client for tests.
    """

    def __init__(self, model: str | None = None, use_rag: bool = True, llm_client=None):
        self.model, self.use_rag, self.llm_client = model, use_rag, llm_client
        self.narratives: list[dict] = []
        self.recommendations: list[dict] = []   # planner results, shown to the user like the narratives
        self.clarification: str | None = None
        self._memo: dict[str, dict] = {}        # identical (tool, arguments) calls within one run are answered once
        self._explained: set[pd.Timestamp] = set()
        self.start, self.end = valid_range()

    def call(self, name: str, args: dict) -> dict:
        """Run a tool by name; unknown tools and bad arguments become an {"error": ...} result for the LLM to read."""
        handlers = {"get_forecasts": self.get_forecasts, "explain_hour": self.explain_hour,
                    "recommend_model": self.recommend_model, "ask_user": self.ask_user}
        if name not in handlers:
            return {"error": f"unknown tool {name!r}; available: {list(handlers)}"}
        # Models sometimes emit the same call twice in one turn. Re-running would duplicate work (and, for
        # explain_hour, the output), so a repeated identical call just gets the earlier answer.
        key = name + json.dumps(args, sort_keys=True, default=str)
        if key in self._memo:
            return self._memo[key]
        try:
            result = handlers[name](**args)
        except (ValueError, TypeError) as exc:
            result = {"error": str(exc)}
        self._memo[key] = result
        return result

    # ------------------------------------------------------------------ tools

    def _range_error(self) -> str:
        return f"Only hours between {self.start:%Y-%m-%d %H:%M} and {self.end:%Y-%m-%d %H:%M} can be explained."

    def get_forecasts(self, date: str) -> dict:
        """24 rows for one day plus the hours with the largest error for each model.

        date: "YYYY-MM-DD". Returns an error dict if the day lies outside the explainable range.
        """
        day = pd.Timestamp(date).normalize()
        if day < self.start.normalize() or day > self.end.normalize():
            return {"error": self._range_error()}
        frame = ac.error_frame().loc[day:day + pd.Timedelta(hours=23)]
        rows = [{"hour": f"{ts:%H:%M}", "actual_kwh": round(r.y_true), "lightgbm_kwh": round(r.lightgbm), "lstm_kwh": round(r.lstm),
                 "lightgbm_error_pct": round(r.lightgbm_err_pct, 1), "lstm_error_pct": round(r.lstm_err_pct, 1)}
                for ts, r in frame.iterrows()]
        cal = calendar_features(pd.DatetimeIndex([day])).iloc[0]
        label = ("major public holiday" if cal.is_major_holiday else "public holiday" if cal.is_holiday
                 else "weekend" if cal.is_weekend else "ordinary weekday")
        worst = {m: f"{frame[f'{m}_err_pct'].abs().idxmax():%H:%M}" for m in ev.ALL_MODELS}
        return {"date": f"{day:%Y-%m-%d}", "weekday": f"{day:%A}", "day_type": label,
                "largest_error_hour": worst, "hours": rows}

    def explain_hour(self, timestamp: str, models: str = "both", question: str | None = None) -> dict:
        """Explain one hour. Validates the hour and the model choice, then runs narrate.narrate.

        Returns only a short status for the LLM; the full explanation (text, checks, sources, evidence) is kept in
        self.narratives and shown to the user by the front end, so the agent cannot alter or mis-quote it.
        """
        if len(self.narratives) >= MAX_EXPLANATIONS_PER_REQUEST:
            return {"error": f"At most {MAX_EXPLANATIONS_PER_REQUEST} hours can be explained per request."}
        ts = pd.Timestamp(timestamp).floor("h")
        if ts < self.start or ts > self.end:
            return {"error": self._range_error()}
        if ts in self._explained:
            return {"error": f"{ts:%Y-%m-%d %H:%M} was already explained in this request. models='both' explains both "
                             "models together; do not explain the same hour again."}
        chosen = ev.normalize_models(models)  # raises ValueError for a typo -> returned as an error
        result = narrate.narrate(ts, models=chosen, question=question, model=self.model,
                                 use_rag=self.use_rag, llm_client=self.llm_client)
        self.narratives.append(result)
        self._explained.add(ts)
        c = result["checks"]
        return {"status": "done", "timestamp": f"{ts:%Y-%m-%d %H:%M}", "models": list(chosen),
                "checks_passed": c["passed"], "n_citations": c["n_citations"], "n_sources": len(result["sources"]),
                "note": "The explanation is shown to the user automatically. Do not repeat or summarise it."}

    def recommend_model(self, start_date: str | None = None, end_date: str | None = None, focus: str | None = None,
                        question: str | None = None) -> dict:
        """Run the planner for a period and/or kind of hour (see planner.recommend).

        Dates and focus are validated by the planner (ValueError -> an error dict the agent can read and fix). Like
        explain_hour, the full result is kept in self.recommendations for the front end; the agent gets only a status,
        so it can neither alter nor misquote the decision.
        """
        if len(self.recommendations) >= MAX_RECOMMENDATIONS_PER_REQUEST:
            return {"error": f"At most {MAX_RECOMMENDATIONS_PER_REQUEST} model recommendations can be made per request."}
        result = planner.recommend(start_date, end_date, focus, question=question, model=self.model, use_rag=self.use_rag,
                                   llm_client=self.llm_client)
        self.recommendations.append(result)
        d = result["decision"]
        return {"status": "done", "scope": result["scope"]["description"], "better_model": d["model"], "confidence": d["confidence"],
                "checks_passed": result["checks"]["passed"],
                "note": "The recommendation is shown to the user automatically. Do not repeat or summarise it."}

    def ask_user(self, question: str) -> dict:
        """Record a clarifying question; the agent loop stops and the front end shows it."""
        self.clarification = question
        return {"status": "question will be shown to the user"}
