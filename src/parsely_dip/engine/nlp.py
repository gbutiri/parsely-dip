"""NLP pattern matching via Stanza service.

Sends user input to Stanza for constituency + dependency parsing,
then matches against NLP pattern specs (JSON).
"""

import json
import os
import requests
from pathlib import Path


STANZA_URL = "http://127.0.0.1:5013/process_syntactic_parsing"


def load_nlp_patterns(pattern_file):
    """Load NLP patterns from a JSON file.

    Args:
        pattern_file: Path to JSON file containing NLP pattern array

    Returns:
        list of {'intent': str, 'nlp': dict} entries
    """
    path = Path(pattern_file)
    if not path.exists():
        return []

    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError) as e:
        print(f"PARSELY NLP pattern load error: {e}")
        return []


def match_nlp_pattern(sentence_data, pattern_spec):
    """Match parsed sentence data against an NLP pattern spec.

    Args:
        sentence_data: dict with 'words' list and 'constituency' tree (from Stanza)
        pattern_spec: dict with 'sentence_type' and 'words' list

    Returns:
        True if pattern matches, False otherwise
    """
    constituency = sentence_data.get('constituency', {})
    words = sentence_data.get('words', [])

    # Check sentence type
    sentence_type = pattern_spec.get('sentence_type')
    if sentence_type:
        types_to_check = [sentence_type] if isinstance(sentence_type, str) else sentence_type

        if 'ROOT' in constituency:
            root_phrase = constituency.get("ROOT", [])[0] if constituency.get("ROOT") else None
            if not root_phrase or not any(t in root_phrase for t in types_to_check):
                return False
        else:
            if not any(t in constituency for t in types_to_check):
                return False

    # Check required words
    for word_pattern in pattern_spec.get('words', []):
        if not word_pattern.get('required', True):
            continue

        found = False
        for sent_word in words:
            if word_pattern.get('word'):
                pattern_words = word_pattern['word']
                if not isinstance(pattern_words, list):
                    pattern_words = [pattern_words]
                if sent_word.get('text', '').lower() not in [w.lower() for w in pattern_words]:
                    continue

            if word_pattern.get('lemma'):
                pattern_lemmas = word_pattern['lemma']
                if not isinstance(pattern_lemmas, list):
                    pattern_lemmas = [pattern_lemmas]
                if sent_word.get('lemma', '').lower() not in [l.lower() for l in pattern_lemmas]:
                    continue

            if word_pattern.get('pos'):
                pattern_pos = word_pattern['pos']
                if not isinstance(pattern_pos, list):
                    pattern_pos = [pattern_pos]
                if sent_word.get('pos') not in pattern_pos:
                    continue

            if word_pattern.get('dep'):
                pattern_deps = word_pattern['dep']
                if not isinstance(pattern_deps, list):
                    pattern_deps = [pattern_deps]
                if sent_word.get('dep') not in pattern_deps:
                    continue

            if word_pattern.get('head_lemma') is not None:
                pattern_head_lemmas = word_pattern['head_lemma']
                if not isinstance(pattern_head_lemmas, list):
                    pattern_head_lemmas = [pattern_head_lemmas]
                if sent_word.get('head_lemma') not in pattern_head_lemmas:
                    continue

            found = True
            break

        if not found:
            return False

    return True


def check_nlp(user_input, patterns, stanza_url=None):
    """Parse input via Stanza and match against NLP patterns.

    Args:
        user_input: raw user text
        patterns: list from load_nlp_patterns()
        stanza_url: optional override for Stanza service URL

    Returns:
        intent_name string or None
    """
    url = stanza_url or STANZA_URL

    headers = {}
    token = os.getenv("STANZA_API_TOKEN")
    if token:
        headers["X-Stanza-Token"] = token

    try:
        response = requests.post(
            url,
            json={"user_input": user_input},
            headers=headers,
            timeout=5
        )
        response.raise_for_status()
        parse_data = response.json()

        for sentence in parse_data.get('sentences', []):
            for pattern in patterns:
                nlp_spec = pattern.get('nlp', pattern)
                if match_nlp_pattern(sentence, nlp_spec):
                    match_data = {
                        'words': sentence.get('words', []),
                        'constituency': sentence.get('constituency', {}),
                        'raw_input': user_input.strip(),
                        'source': 'nlp'
                    }
                    return pattern.get('intent'), match_data

    except requests.exceptions.RequestException:
        pass

    return None, None
