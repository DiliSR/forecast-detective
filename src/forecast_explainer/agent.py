"""The explanation agent: a plain-English request in, cited explanations out.

    "Why was the forecast so off on Christmas Eve evening? Only the LSTM, please."
        |
        v   LLM turn 1: understands the request, calls get_forecasts("2014-12-24") to see the hours
        v   LLM turn 2: chooses 18:00, calls explain_hour("2014-12-24 18:00", models="lstm")
        |                 -> store.get_explanation -> evidence -> RAG -> writer LLM -> checker   (all deterministic code,
        |                    cached; the writer is the only other LLM call)
        v   LLM turn 3: one sentence stating what it chose ("Explained 24 Dec 18:00, LSTM only")
    result: the agent's note + the checked explanation(s) + a trace of every step

Design rules:
  - The agent LLM decides WHAT to run (which hour, which model, whether to ask); tools do all the computing.
  - It never writes the explanation. The explanation comes from explain_hour and is returned untouched, so a
    chat message cannot add an unverified number to a checked text.
  - It can also call recommend_model (the planner) for "which model should I use?". Which model wins is computed
    by planner.py with a statistical test; the agent only chooses the scope.
  - Bounded: a step limit, at most 3 explanations and 2 recommendations per request, and ask_user ends the run.
"""

import json
import time

from forecast_explainer import llm, planner
from forecast_explainer.agent_tools import MAX_EXPLANATIONS_PER_REQUEST, TOOLS, ToolBox

MAX_STEPS = 8  # LLM turns per request; a normal request needs 2-4

_SYSTEM_TEMPLATE = """You are the assistant of a day-ahead electricity-load forecast explainer for Portugal. Users ask, in plain \
English, why a forecast for some date or hour came out the way it did. Two forecasting models exist: LightGBM (gradient-boosted \
trees) and an LSTM (a neural sequence model). Users also ask which model to use. You use tools; you never state forecast facts, \
numbers or judgements about which model is better yourself.

Data: you can only explain hours between {start} and {end}. If the user gives no year, it is 2014. There is no "today": resolve \
relative phrases such as "last Tuesday" against {end}, and ask if that is not clear.

How to work:
1. Work out the target hour and the model(s). Use "both" unless the user clearly names one ("the tree model"/"LightGBM" -> lightgbm; \
"the neural network"/"LSTM" -> lstm).
2. If the user gives an exact hour ("at noon" = 12:00, "midnight" = 00:00, "6pm" = 18:00, "18:00"), use it and do NOT call \
get_forecasts. If they give only a day or a vague time, call get_forecasts(date) once and choose: "evening" = around 18:00-21:00, "morning" = 06:00-11:00, "worst"/"most wrong"/"biggest error" = the hour with the largest absolute \
error for the model(s) in question. If nothing is specified, choose the hour with the largest error and say so.
2b. A question about a whole day or period ("which factors affect the 25th?", "what drives Christmas Day?", "why is Dec 25 hard to \
forecast?") IS a request you can serve: call get_forecasts(date) once, then explain up to {max_hours} representative hours of that day: \
the hour with the largest error, plus one or two contrasting hours (for example a night hour and a midday or evening hour). Pass the \
user's wording as `question`. Say in your note that the day is described through these hours.
2c. A question about WHICH MODEL is better, best, safer or more accurate ("which model should I use?", "which is better on holidays?", \
"is the LSTM better in the evening?", "which model for Christmas week?") is answered by calling recommend_model once. Pass start_date and \
end_date only if the user names a period (otherwise the whole test period is compared), `focus` only if the user names a kind of hour \
({focuses}), and the user's wording as `question`. Which model wins is computed by code; never judge it yourself. If the user also asks \
why a particular hour went wrong, call explain_hour as well.
3. If the DATE is genuinely ambiguous (for example "Christmas" can mean the 24th or the 25th, or "the holiday"), call ask_user with \
ONE short question. Do not ask when only one reading is possible, and do not ask about the hour: choose and say so.
4. Call explain_hour exactly once per hour (models="both" already covers both models; never explain the same hour again with \
another model setting), at most {max_hours} hours per request. If the user asks a specific question about the forecast, \
pass it as `question`.
5. After explain_hour or recommend_model succeeds, reply with at most two sentences: which hour(s) and model(s) you explained, or which \
scope you compared, and any assumption you made. Do NOT repeat or summarise the explanation; it is shown to the user automatically.
6. If a tool returns an error, read it and correct your call, or tell the user plainly. If the request is neither about explaining the \
forecast of a specific day, period or hour nor about choosing between the two models (general questions, other topics), say that you \
can only explain forecasts for specific days and hours and compare the two models for now. A question that names any date, or asks \
which model to use, is never in this category. Never invent numbers or reasons."""


def agent_system_prompt(box: ToolBox) -> str:
    """The agent's instructions, with the explainable date range filled in from the actual data."""
    return _SYSTEM_TEMPLATE.format(start=f"{box.start:%Y-%m-%d %H:%M}", end=f"{box.end:%Y-%m-%d %H:%M}",
                                   max_hours=MAX_EXPLANATIONS_PER_REQUEST, focuses=", ".join(f'"{f}"' for f in planner.FOCUS_CHOICES))


def run_agent(message: str, history: list[dict] | None = None, model: str | None = None, writer_model: str | None = None,
              use_rag: bool = True, llm_client=None, max_steps: int = MAX_STEPS, on_step=None) -> dict:
    """Handle one user request.

    message:      the user's text.
    history:      earlier turns as [{"role": "user"|"assistant", "content": str}, ...], so follow-ups work
                  ("now the same for the LSTM only"). Use result["history_text"] for the assistant side.
    model:        LLM that drives the agent (tool choice). Default LLM_MODEL.
    writer_model: LLM that writes the explanations. Default = same as `model`.
    use_rag:      False runs the writer without the knowledge base (ablation).
    llm_client:   injected fake client for tests (used for both the agent and the writer).
    max_steps:    cap on agent LLM turns.
    on_step:      optional callback called with each trace entry right after a tool finishes - lets a front end show
                  progress ("explain_hour ... 10 s") during the seconds the explainers take.

    Returns a dict:
      message        the agent's note (or the clarifying question)
      stopped        "done" | "clarify" | "max_steps"
      clarification  the question, when stopped == "clarify"
      narratives     the full narrate() results, one per explained hour (text, checks, sources, evidence, ...)
      recommendations  the full planner.recommend() results, one per model comparison
      trace          one entry per tool call: tool, arguments, result summary, seconds
      usage          tokens spent by the agent LLM itself (the writer's tokens are inside each narrative)
      history_text   what to append to `history` as the assistant's turn
    """
    box = ToolBox(model=writer_model or model, use_rag=use_rag, llm_client=llm_client)
    messages = [{"role": "system", "content": agent_system_prompt(box)}, *(history or []), {"role": "user", "content": message}]
    trace, tokens, final, stopped = [], [0, 0], "", "max_steps"

    for _ in range(max_steps):
        turn = llm.chat(messages, tools=TOOLS, model=model, client=llm_client)
        tokens[0] += turn.prompt_tokens
        tokens[1] += turn.completion_tokens
        messages.append(turn.message)
        if not turn.tool_calls:                       # no tool requested: the model is giving its final answer
            final, stopped = turn.content, "done"
            break
        for call in turn.tool_calls:
            started = time.time()
            try:
                args = json.loads(call.arguments or "{}")
                result = box.call(call.name, args)
            except json.JSONDecodeError as exc:       # the model produced malformed arguments: tell it, let it retry
                args, result = {"raw": call.arguments}, {"error": f"arguments are not valid JSON: {exc}"}
            trace.append({"tool": call.name, "arguments": args, "result": result, "seconds": round(time.time() - started, 1)})
            if on_step:
                on_step(trace[-1])
            # Tool results go back to the LLM as JSON text; the full tables stay out of the trace shown to users.
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)})
        if box.clarification:
            final, stopped = box.clarification, "clarify"
            break

    explained = "; ".join(f"{n['timestamp'][:16]} ({'+'.join(n['models'])})" for n in box.narratives)
    compared = "; ".join(r["scope"]["description"] for r in box.recommendations)
    history_text = final + (f" [explained: {explained}]" if explained else "") + (f" [compared models: {compared}]" if compared else "")
    return {"message": final, "stopped": stopped, "clarification": box.clarification, "narratives": box.narratives,
            "recommendations": box.recommendations,
            "trace": trace, "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1]}, "history_text": history_text}
