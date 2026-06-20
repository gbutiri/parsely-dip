"""PARSELY-DIP Test Runner

Loads .tests files and runs each test case against the pattern engine.
Writes results to parsely_tests table in db/parsely.db.

Usage:
    py tests/test_runner.py tests/tell_time.tests
    py tests/test_runner.py tests/tell_time.tests --type regex
    py tests/test_runner.py tests/tell_time.tests --type nlp
"""

import sys
import os
import sqlite3
import argparse
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from parsely_dip.engine.regex import load_patterns, check_regex
from parsely_dip.engine.nlp import load_nlp_patterns, check_nlp


DB_PATH = Path(__file__).parent.parent / 'db' / 'parsely.db'
PATTERNS_DIR = Path(__file__).parent.parent / 'src' / 'parsely_dip' / 'patterns'


def parse_test_file(filepath):
    """Parse a .tests file into test cases.

    Format: sentence => intent | type | expected
    Returns list of dicts.
    """
    tests = []
    path = Path(filepath)
    if not path.exists():
        print(f"Test file not found: {filepath}")
        return tests

    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split('=>')
        if len(parts) != 2:
            continue

        sentence = parts[0].strip()
        meta = parts[1].strip()
        fields = [f.strip() for f in meta.split('|')]

        if len(fields) != 3:
            print(f"Skipping malformed line: {line}")
            continue

        intent, test_type, expected = fields
        tests.append({
            'input': sentence,
            'intent': intent,
            'type': test_type,
            'expected': int(expected)
        })

    return tests


def run_tests(tests, type_filter=None, source_file=None):
    """Run test cases against the engine.

    Args:
        tests: list of test dicts from parse_test_file
        type_filter: 'regex', 'nlp', or None (both)
        source_file: name of the .tests file
    """
    # Load patterns
    regex_patterns = load_patterns(PATTERNS_DIR / 'base.patterns')
    nlp_patterns = load_nlp_patterns(PATTERNS_DIR / 'base_nlp.json')

    run_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    db = sqlite3.connect(str(DB_PATH))

    passed = 0
    failed = 0
    skipped = 0
    errors = []

    # Check if stanza is available for NLP tests
    stanza_available = False
    nlp_needed = any(t['type'] == 'nlp' for t in tests)
    if nlp_needed and type_filter != 'regex':
        try:
            import requests
            r = requests.get('http://127.0.0.1:5013/', timeout=2)
            stanza_available = True
        except Exception:
            print("  Stanza service not running — NLP tests will be skipped")

    for t in tests:
        if type_filter and t['type'] != type_filter:
            skipped += 1
            continue

        if t['type'] == 'nlp' and not stanza_available:
            skipped += 1
            continue

        # Run the test
        if t['type'] == 'regex':
            intent, _ = check_regex(t['input'], regex_patterns)
            matched = 1 if intent == t['intent'] else 0
        elif t['type'] == 'nlp':
            intent, _ = check_nlp(t['input'], nlp_patterns)
            matched = 1 if intent == t['intent'] else 0
        else:
            skipped += 1
            continue

        # Compare
        test_passed = (matched == t['expected'])
        if test_passed:
            passed += 1
            status = 'PASS'
        else:
            failed += 1
            status = 'FAIL'
            errors.append({
                'input': t['input'],
                'intent': t['intent'],
                'type': t['type'],
                'expected': t['expected'],
                'got': matched
            })

        # Write to DB
        db.execute('''
            INSERT INTO parsely_tests (pt_input, pt_intent, pt_type, pt_source, pt_expected, pt_result, pt_run)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (t['input'], t['intent'], t['type'], source_file, t['expected'], matched, run_timestamp))

    db.commit()
    db.close()

    # Print results
    total = passed + failed
    print()
    print(f"  {'='*50}")
    print(f"  PARSELY-DIP Test Results")
    print(f"  {'='*50}")
    print(f"  Run:     {run_timestamp}")
    print(f"  Source:  {source_file}")
    print(f"  {'─'*50}")
    print(f"  Total:   {total}  |  Passed: {passed}  |  Failed: {failed}  |  Skipped: {skipped}")

    if total > 0:
        pct = (passed / total) * 100
        bar_len = 30
        filled = int(bar_len * passed / total)
        bar = '#' * filled + '-' * (bar_len - filled)
        print(f"  Score:   [{bar}] {pct:.0f}%")

    if errors:
        print(f"  {'─'*50}")
        print(f"  FAILURES:")
        for e in errors:
            exp_label = "MATCH" if e['expected'] == 1 else "NO MATCH"
            got_label = "MATCH" if e['got'] == 1 else "NO MATCH"
            print(f"    [{e['type'].upper():5s}] \"{e['input']}\"")
            print(f"           Intent: {e['intent']}  Expected: {exp_label}  Got: {got_label}")

    print(f"  {'='*50}")
    print()

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description='PARSELY-DIP Test Runner')
    parser.add_argument('testfile', help='Path to .tests file')
    parser.add_argument('--type', choices=['regex', 'nlp'], default=None,
                        help='Only run tests of this type')
    args = parser.parse_args()

    source_file = Path(args.testfile).name

    print()
    print(f"  Loading tests from {source_file}...")
    tests = parse_test_file(args.testfile)
    if not tests:
        print("  No tests found.")
        return

    print(f"  Found {len(tests)} test cases")

    success = run_tests(tests, type_filter=args.type, source_file=source_file)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
