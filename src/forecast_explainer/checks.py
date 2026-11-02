"""Deterministic checks on an LLM explanation: citations and numbers. No LLM involved.

Why check in code first? Some failures are mechanical and an LLM judge would be a slow, costly and unreliable way to
find them: citing a source that does not exist, or quoting a number that is not in the evidence. Code finds those
exactly, for free, every time. (Whether a cited source actually SUPPORTS the sentence is a meaning question - that is
what the LLM judge is for.)

The checks:
  unknown_citations   an [E:...] or [S..] ID that was not provided
  unsupported_numbers a number that is not in the evidence blocks / sources it cites (see check_text for the scope)
  uncited_numbers     a sentence with a number but no citation (a claim nobody can trace)
  flags_not_cited     disagreement flags fired but the text never cites [E:flags]
  causal_without_source  a sentence that states a cause ("because", "due to", "the main reason") with no [S#] source -
                      the model's attributions show how the MODEL used its inputs, never why the real load changed, so
                      a real-world cause needs a source (reported as a caution; it does not fail the text)

`passed` is True when the first two are empty and the flags were cited. Uncited numbers are reported but do not
fail the text on their own, because a number in a sentence that is cited in the next sentence is a style issue.
"""

import re

# One or more IDs inside square brackets. IDs are matched loosely (E:\w+) on purpose: a made-up ID such as
# [E:made_up] must be recognised as a citation attempt so it can be reported as unknown, not silently ignored.
CITATION = re.compile(r"\[\s*((?:E:\w+|S\d+)(?:\s*[,;]\s*(?:E:\w+|S\d+))*)\s*\]")
NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?")

MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*"

# Things that look like numbers but are not claims about the forecast. They are removed before number extraction.
NOT_CLAIMS = [
    re.compile(r"`[^`]*`"),                              # `load_lag_168h` - code spans
    re.compile(r"\b[A-Za-z]+(?:_[A-Za-z0-9]+)+\b"),      # load_lag_168h, lag_24h_was_holiday - feature names
    re.compile(r"\bt-\d+h\b"),                           # t-24h - hours before the target
    re.compile(r"\b\d{1,2}:\d{2}\b"),                    # 18:00 - clock times
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}\b"),   # 24 December
    re.compile(rf"\b{MONTHS}\s+\d{{1,2}}\b"),            # December 24
    re.compile(r"\b(?:19|20)\d{2}(?:-\d{2}-\d{2})?\b"),  # 2014, 2014-12-24
    re.compile(r"\bR\s*(?:²|2)\b"),                      # R2 / R² - the name of a statistic, not a value
    re.compile(r"\b\d+(?:st|nd|rd|th)\b"),               # 1st, 2nd
]
# Words that assert a cause. Matched case-insensitively on whole words.
CAUSAL = re.compile(r"\b(because|due to|owing to|caused|causes|the (?:main |primary |likely )?reasons?|resulting from|results? from|explained by|as a result of)\b", re.I)
SMALL_COUNT = 10  # integers up to this are treated as counts or ranks ("top 5", "3 seeds") unless followed by % or kWh


def citations(text: str) -> list[str]:
    """Every cited ID in the text, in order, e.g. "[E:shap, S2]" gives ["E:shap", "S2"]."""
    ids = []
    for group in CITATION.findall(text):
        ids += [part.strip() for part in re.split(r"[,;]", group)]
    return ids


def _to_float(token: str) -> float:
    return float(token.replace(",", ""))


# Numbers inside feature names: load_lag_168h -> 168, load_mean_7d -> 7, load_same_hour_mean_4w -> 4. A writer may
# legitimately say "load from 168 hours earlier" when reading the feature name, so these count as present in the evidence.
IDENTIFIER_NUMBER = re.compile(r"_(\d+)[a-z]?\b")


def all_numbers(text: str) -> list[float]:
    """Every number a text contains, unfiltered: plain numbers plus the numbers embedded in feature names.

    Used to collect what the evidence blocks and the retrieved sources contain.
    """
    return [_to_float(t) for t in NUMBER.findall(text)] + [float(t) for t in IDENTIFIER_NUMBER.findall(text)]


def claim_numbers(text: str) -> list[tuple[str, float]]:
    """(as written, value) for each number in the text that makes a claim.

    Citations and the NOT_CLAIMS patterns are blanked first. Small bare integers are skipped (counts and ranks).
    Signs are ignored: "+14.6%" and "14.6%" are the same claim.
    """
    text = CITATION.sub(" ", text)
    for pattern in NOT_CLAIMS:
        text = pattern.sub(" ", text)
    found = []
    for match in NUMBER.finditer(text):
        token = match.group()
        value = _to_float(token)
        tail = text[match.end():match.end() + 6].lstrip()
        has_unit = tail.startswith(("%", "kWh", "MWh", "kW"))
        if "." not in token and "," not in token and value <= SMALL_COUNT and not has_unit:
            continue
        found.append((token, value))
    return found


def is_supported(token: str, value: float, allowed: list[float]) -> bool:
    """Is the quoted number one of the allowed numbers, rounded to the precision it was written with?

    "15" (no decimals) is supported by 14.6 or 15.1 because both round to 15; "14.6" needs a value that
    rounds to 14.6. So the LLM may round a number the way a human would, but not change it.
    """
    decimals = len(token.split(".")[1]) if "." in token else 0
    return any(abs(round(a, decimals) - value) < 1e-9 for a in allowed)


def split_sentences(text: str) -> list[str]:
    """Sentences, roughly: split after . ! ? or a line break. Good enough to find uncited claims."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[*])|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def check_text(text: str, evidence: dict[str, str], hits: list[dict]) -> dict:
    """Run all checks on one explanation.

    text:     the LLM's explanation.
    evidence: {"E:forecast": block text, ...} that the LLM was given.
    hits:     the labelled retrieved chunks (with "sid" and "text") that the LLM was given.

    Number rule - a claim number must appear in the blocks/sources the text cites for it, using the tightest scope
    available:
      1. the citations in the same sentence;
      2. if the sentence has none, the citations anywhere in its paragraph (text between blank lines);
      3. if the paragraph has none, any evidence at all (and the sentence is listed under uncited_numbers).
    Why not simply "anywhere in the evidence"? A pooled check lets a wrong number through whenever it happens to occur
    somewhere else (a wrong "17%" would pass because a LIME rank range "3-17" is in the flags block). Why not
    sentence-only? Writers often cite once at the end of several sentences.
    """
    by_id = {**evidence, **{h["sid"]: h["text"] for h in hits}}
    cited = citations(text)
    unknown = sorted({c for c in cited if c not in by_id})
    everything = [x for block in by_id.values() for x in all_numbers(block)]

    unsupported, uncited = [], []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph_ids = [c for c in citations(paragraph) if c in by_id]
        for sentence in split_sentences(paragraph):
            ids = [c for c in citations(sentence) if c in by_id] or paragraph_ids
            allowed = [x for i in set(ids) for x in all_numbers(by_id[i])] if ids else everything
            claims = claim_numbers(sentence)
            for tok, val in claims:
                if not is_supported(tok, val, allowed):
                    # Where the number DOES appear: lets the retry say "you quoted it correctly but cited the wrong block".
                    found = [i for i, block in by_id.items() if is_supported(tok, val, all_numbers(block))]
                    unsupported.append({"number": tok, "sentence": sentence, "found_in": found})
            if claims and not citations(sentence):
                uncited.append(sentence)

    causal = [s for s in split_sentences(text)
              if CAUSAL.search(s) and not any(c.startswith("S") for c in citations(s))]

    fired = "[HIGH]" in evidence.get("E:flags", "") or "[WARNING]" in evidence.get("E:flags", "")
    flags_missing = fired and "E:flags" not in cited

    return {
        "passed": not unknown and not unsupported and not flags_missing,
        "n_citations": len(cited),
        "unknown_citations": unknown,
        "unsupported_numbers": unsupported,
        "uncited_numbers": uncited,
        "flags_not_cited": bool(flags_missing),
        "causal_without_source": causal,
    }
