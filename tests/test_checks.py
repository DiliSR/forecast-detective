"""Tests for the deterministic citation / number checker."""

from forecast_explainer import checks

EVIDENCE = {
    "E:forecast": "Actual load: 211,763 kWh. LightGBM forecast: 242,680 kWh (error +30,917 kWh, +14.6%).",
    "E:shap": "- load_lag_168h = 235,522 kWh: +27,176 kWh",
    "E:flags": "Disagreement checks that fired:\n- [HIGH] shared_error: both models over-forecast.",
}
HITS = [{"sid": "S1", "text": "At 20:00 the load is 29% to 43% below a normal day."}]


def run(text, evidence=EVIDENCE, hits=HITS):
    return checks.check_text(text, evidence, hits)


def test_clean_text_passes():
    out = run("The actual load was 211,763 kWh [E:forecast]. The model over-forecast by +14.6% [E:forecast] [E:flags].")
    assert out["passed"] and not out["unsupported_numbers"] and not out["unknown_citations"]


def test_invented_number_is_reported():
    out = run("The actual load was 215,000 kWh [E:forecast] [E:flags].")
    assert not out["passed"]
    assert out["unsupported_numbers"][0]["number"] == "215,000"


def test_derived_number_is_not_accepted():
    # 242,680 - 211,763 = 30,917 is in the evidence, but a rounded difference the LLM computed itself is not.
    assert not run("It was off by about 31,000 kWh [E:forecast] [E:flags].")["passed"]
    assert run("It was off by 30,917 kWh [E:forecast] [E:flags].")["passed"]


def test_human_rounding_is_accepted_but_not_a_different_number():
    assert run("About 15% too high [E:forecast] [E:flags].")["passed"]            # 14.6 rounds to 15
    assert not run("About 16% too high [E:forecast] [E:flags].")["passed"]
    assert not run("It was 14.7% too high [E:forecast] [E:flags].")["passed"]     # same precision, different value


def test_numbers_from_sources_are_allowed():
    assert run("The load falls 29% to 43% at 20:00 [S1] [E:flags].")["passed"]


def test_unknown_citation_is_reported_even_with_underscore():
    out = run("Something [E:made_up] and [S9] and [E:shap] [E:flags].")
    assert out["unknown_citations"] == ["E:made_up", "S9"] and not out["passed"]


def test_multiple_ids_in_one_bracket():
    assert checks.citations("x [E:shap, S1] y") == ["E:shap", "S1"]


def test_dates_times_features_lags_and_counts_are_not_claims():
    text = ("On 24 December 2014 at 18:00 the feature `load_lag_168h` and load_lag_24h at t-24h mattered; "
            "the top 5 features and 3 seeds agree [E:shap] [E:flags].")
    assert run(text)["unsupported_numbers"] == []


def test_flags_must_be_cited_when_something_fired():
    assert run("The actual load was 211,763 kWh [E:forecast].")["flags_not_cited"]
    quiet = {**EVIDENCE, "E:flags": "Disagreement checks that fired:\n- none fired"}
    assert not run("The actual load was 211,763 kWh [E:forecast].", quiet)["flags_not_cited"]


def test_sentence_with_number_but_no_citation_is_listed():
    out = run("The actual load was 211,763 kWh. It was high [E:forecast] [E:flags].")
    assert out["uncited_numbers"] == ["The actual load was 211,763 kWh."]


def test_number_must_come_from_a_block_cited_in_the_same_paragraph():
    # 30,917 exists in E:forecast, but the sentence cites E:shap (and E:flags), so it is not supported there.
    assert not run("The error was 30,917 kWh [E:shap] [E:flags].")["passed"]
    assert run("The error was 30,917 kWh [E:forecast] [E:flags].")["passed"]


def test_number_found_elsewhere_by_coincidence_is_not_accepted():
    ev = {**EVIDENCE, "E:flags": "Disagreement checks that fired:\n- [HIGH] x: LIME rank 3-17."}
    # Same paragraph cites both blocks, but the sentence with 17% cites only E:forecast, which has no 17.
    out = run("It was 17% too high [E:forecast]. Disagreements fired [E:flags].", ev)
    assert [n["number"] for n in out["unsupported_numbers"]] == ["17"]


def test_citation_at_the_end_of_the_paragraph_covers_earlier_sentences():
    text = "Actual load was 211,763 kWh. LightGBM forecast 242,680 kWh [E:forecast] [E:flags]."
    assert run(text)["passed"]


def test_hours_in_a_feature_name_may_be_written_out():
    assert run("The load 168 hours earlier gave +27,176 kWh [E:shap] [E:flags].")["passed"]


def test_paragraph_without_any_citation_falls_back_to_all_evidence_and_is_listed_uncited():
    out = run("Actual load was 211,763 kWh.\n\nDisagreements fired [E:flags].")
    assert out["unsupported_numbers"] == [] and out["uncited_numbers"] == ["Actual load was 211,763 kWh."]


def test_cause_without_a_source_is_cautioned_but_does_not_fail():
    out = run("The main reason for the miss is missing weather data [E:forecast] [E:flags].")
    assert out["passed"] and out["causal_without_source"] == ["The main reason for the miss is missing weather data [E:forecast] [E:flags]."]


def test_cause_with_a_source_is_fine():
    assert run("The load fell because of Christmas Eve habits, according to [S1] [E:flags].")["causal_without_source"] == []


def test_unsupported_number_reports_where_it_does_appear():
    out = run("Both models were off by 30,917 kWh [E:shap] [E:flags].")
    assert out["unsupported_numbers"][0]["found_in"] == ["E:forecast"]
    nowhere = run("Both models were off by 99,999 kWh [E:forecast] [E:flags].")
    assert nowhere["unsupported_numbers"][0]["found_in"] == []
