"""PARSELY-DIP: Parsing And RegEx Syntactic Engine with Linguistic Yield - Deterministic Intent Parser"""

__version__ = "0.0.11"

import os
from pathlib import Path
from parsely_dip.engine.regex import load_patterns, check_regex
from parsely_dip.engine.nlp import load_nlp_patterns, check_nlp
from parsely_dip.engine.registry import dispatch

# Auto-import intents to trigger @intent decorator registration
from parsely_dip.intents import time  # noqa: F401
from parsely_dip.intents import weather  # noqa: F401
from parsely_dip.intents import scrum  # noqa: F401
from parsely_dip.intents import day  # noqa: F401

# Pattern search path: project/env layers sit ON TOP of the package's vendored
# defaults. check_regex returns the FIRST match, so dirs are ordered most-specific
# first (env > project > vendor) — a project can override a built-in intent or just
# add its own, and the shipped defaults are always the fallback. (#1151)
_VENDOR_PATTERNS_DIR = Path(__file__).parent / "patterns"
_regex_patterns = None
_nlp_patterns = None
_patterns_mtime = 0


def _pattern_dirs():
    """Ordered, most-specific first: env override, <project-root>/patterns, vendor defaults."""
    dirs = []
    env = os.environ.get("PARSELY_PATTERNS_DIR")
    if env:
        dirs.append(Path(env))
    dirs.append(Path.cwd() / "patterns")    # the consuming app's own patterns (run from its root)
    dirs.append(_VENDOR_PATTERNS_DIR)        # shipped defaults — always the fallback
    seen, out = set(), []
    for d in dirs:
        rd = d.resolve()
        if rd not in seen and d.is_dir():
            seen.add(rd)
            out.append(d)
    return out


def _collect(d):
    """Load every *.patterns (regex) and *_nlp.json (NLP) file in one dir."""
    rx, nlp = [], []
    for f in sorted(d.glob("*.patterns")):
        rx += load_patterns(f)
    for f in sorted(d.glob("*_nlp.json")):
        nlp += load_nlp_patterns(f)
    return rx, nlp


def _load_default_patterns():
    """Merge patterns across the search path, reloading if any file changed on disk."""
    global _regex_patterns, _nlp_patterns, _patterns_mtime

    dirs = _pattern_dirs()
    files = []
    for d in dirs:
        files += list(d.glob("*.patterns")) + list(d.glob("*_nlp.json"))
    mt = max((f.stat().st_mtime for f in files), default=0)

    if _regex_patterns is None or mt != _patterns_mtime:
        rx, nlp = [], []
        for d in dirs:
            r, n = _collect(d)
            rx += r
            nlp += n
        _regex_patterns, _nlp_patterns, _patterns_mtime = rx, nlp, mt


# Smart quotes/apostrophes -> straight ASCII, so patterns written with ' and " still match
# input where the keyboard/browser auto-curled them ("what's" with U+2019, etc.).
_SMART_QUOTES = {0x2018: 0x27, 0x2019: 0x27, 0x201C: 0x22, 0x201D: 0x22, 0x2032: 0x27}


def _normalize(prompt):
    return prompt.translate(_SMART_QUOTES) if prompt else prompt


def parse(prompt, regex_patterns=None, nlp_patterns=None):
    """Parse a prompt through the DIP pipeline.

    RegEx first, then NLP, then returns None (pass to LLM).

    Args:
        prompt: raw user input string
        regex_patterns: optional override (list from load_patterns)
        nlp_patterns: optional override (list from load_nlp_patterns)

    Returns:
        str response if matched, None if no match
    """
    prompt = _normalize(prompt)

    if regex_patterns is None or nlp_patterns is None:
        _load_default_patterns()

    rx = regex_patterns if regex_patterns is not None else _regex_patterns
    nlp = nlp_patterns if nlp_patterns is not None else _nlp_patterns

    # Layer 1: Regex
    intent_name, context = check_regex(prompt, rx)

    # Layer 2: NLP (only if regex missed)
    if not intent_name and nlp:
        intent_name, context = check_nlp(prompt, nlp)

    # Dispatch to handler with context
    if intent_name:
        return dispatch(intent_name, context)

    return None
