"""PARSELY-DIP: Parsing And RegEx Syntactic Engine with Linguistic Yield - Deterministic Intent Parser"""

__version__ = "0.0.2"

from pathlib import Path
from parsely_dip.engine.regex import load_patterns, check_regex
from parsely_dip.engine.nlp import load_nlp_patterns, check_nlp
from parsely_dip.engine.registry import dispatch

# Auto-import intents to trigger @intent decorator registration
from parsely_dip.intents import time  # noqa: F401
from parsely_dip.intents import weather  # noqa: F401
from parsely_dip.intents import scrum  # noqa: F401

# Default pattern paths (package-relative)
_PATTERNS_DIR = Path(__file__).parent / "patterns"
_regex_patterns = None
_nlp_patterns = None


def _load_default_patterns():
    """Load default patterns on first call."""
    global _regex_patterns, _nlp_patterns

    if _regex_patterns is None:
        base_patterns = _PATTERNS_DIR / "base.patterns"
        _regex_patterns = load_patterns(base_patterns)

    if _nlp_patterns is None:
        base_nlp = _PATTERNS_DIR / "base_nlp.json"
        _nlp_patterns = load_nlp_patterns(base_nlp)


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
    if regex_patterns is None or nlp_patterns is None:
        _load_default_patterns()

    rx = regex_patterns if regex_patterns is not None else _regex_patterns
    nlp = nlp_patterns if nlp_patterns is not None else _nlp_patterns

    # Layer 1: Regex
    intent_name = check_regex(prompt, rx)

    # Layer 2: NLP (only if regex missed)
    if not intent_name and nlp:
        intent_name = check_nlp(prompt, nlp)

    # Dispatch to handler
    if intent_name:
        return dispatch(intent_name)

    return None
