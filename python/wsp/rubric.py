"""Probe rubric for a weight edit.

It checks three things the reference runs actually broke: the concept never
appears, a held-out fact (Paris) disappears, or the answer collapses into a
repeated word.
"""

import re


def score_rows(rows, words=None):
    words = list(words or [])
    if not rows:
        return {"conceptHit": None, "factOk": None, "collapse": True, "fluent": False}
    hits = 0
    facts = 0
    facts_ok = 0
    collapse = False
    fluent = True
    for row in rows:
        text = str(row.get("edited") or "")
        toks = _tokens(text)
        echo = text.strip().lower() == str(row.get("prompt") or "").strip().lower()
        row_collapse = echo or len(toks) < 8 or _repetition_ratio(toks) >= 0.34 or _dominates(toks, words)
        if row_collapse:
            collapse = True
            fluent = False
        if words and any(str(word).lower() in text.lower() for word in words):
            hits += 1
        if re.search(r"\bfrance\b|\bcapital of\b", row.get("prompt") or "", re.I | re.ASCII):
            facts += 1
            if re.search(r"\bparis\b", text, re.I | re.ASCII):
                facts_ok += 1
    return {
        "conceptHit": (hits / len(rows)) if words else None,
        "factOk": (facts_ok == facts) if facts else None,
        "collapse": collapse,
        "fluent": fluent,
    }


def score_summary(score):
    concept = "concept n/a" if score["conceptHit"] is None else f"concept {round(score['conceptHit'] * 100)}%"
    if score["factOk"] is None:
        fact = "no fact probe"
    else:
        fact = "Paris held" if score["factOk"] else "Paris lost"
    shape = "collapsed" if score["collapse"] else "not collapsed"
    fluent = "fluent" if score["fluent"] else "not fluent"
    return f"{concept} · {fact} · {shape} · {fluent}"


def _tokens(text):
    return [re.sub(r"[^a-z0-9]+", "", tok) for tok in text.lower().split() if re.sub(r"[^a-z0-9]+", "", tok)]


def _repetition_ratio(toks):
    if len(toks) < 8:
        return 0
    counts = {}
    for tok in toks:
        counts[tok] = counts.get(tok, 0) + 1
    return max(counts.values()) / len(toks)


def _dominates(toks, words):
    if not words or len(toks) < 8:
        return False
    want = {str(word).lower() for word in words}
    n = sum(1 for tok in toks if tok in want)
    return n / len(toks) > 0.4
