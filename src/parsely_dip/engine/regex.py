"""Regex pattern matching engine.

Loads .patterns files and matches user input against them.
Pattern file format: one pattern per line
    (regex) => intent_name
    # comments and blank lines ignored
"""

import re
from pathlib import Path


def load_patterns(pattern_file):
    """Load regex patterns from a .patterns file.

    Args:
        pattern_file: Path to .patterns file

    Returns:
        list of (compiled_regex, intent_name) tuples
    """
    patterns = []
    path = Path(pattern_file)

    if not path.exists():
        return patterns

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split('=>')
        if len(parts) != 2:
            continue

        regex_str = parts[0].strip()
        intent_name = parts[1].strip()

        try:
            compiled = re.compile(regex_str, re.IGNORECASE)
            patterns.append((compiled, intent_name))
        except re.error as e:
            print(f"PARSELY regex error in '{regex_str}': {e}")

    return patterns


def check_regex(user_input, patterns):
    """Match user input against loaded regex patterns.

    Args:
        user_input: raw user text
        patterns: list of (compiled_regex, intent_name) from load_patterns()

    Returns:
        tuple: (intent_name, match_data) or (None, None)
        match_data is a dict with 'groups', 'full_match', and 'raw_input'
    """
    for compiled, intent_name in patterns:
        match = compiled.match(user_input.strip())
        if match:
            match_data = {
                'full_match': match.group(0),
                'groups': match.groups(),
                'raw_input': user_input.strip(),
                'source': 'regex'
            }
            return intent_name, match_data
    return None, None
