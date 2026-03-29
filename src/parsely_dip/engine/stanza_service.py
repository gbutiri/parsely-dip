"""
Stanza NLP syntactic parsing service for PARSELY-DIP

Standalone Flask service running on port 5013 that provides constituency
parsing using Stanza. Stateless service that parses user input and returns
linguistic features for intent matching.

Ported from: unibot/services/stanza_nlp.py

DEPENDENCIES:
    - stanza: NLP pipeline (tokenize, pos, lemma, depparse, constituency)
    - flask: Service endpoint at localhost:5013
    - dotenv: Environment variable management

SECURITY:
    - IP restriction: Only accepts from 127.0.0.1
    - Token authentication: X-Stanza-Token header required
    - Environment variable: STANZA_API_TOKEN

NLP PIPELINE:
    - Language: English
    - Processors: tokenize, pos, lemma, depparse, constituency
    - GPU: Enabled (set to False if unavailable)

ARCHITECTURE:
    - Stateless service (no database dependency)
    - Start once, stays running — parsely-dip calls via HTTP per query
    - Portable and deployable independently

CALLED BY:
    - parsely_dip.engine.nlp.check_nlp() via HTTP request (port 5013)

SERVICE ENDPOINT:
    POST /process_syntactic_parsing
    Body: {"user_input": "What time is it?"}
    Returns: {"sentences": [{"text": ..., "words": [...], "constituency": {...}}]}

KEY FUNCTIONS:
    - process_syntactic_parsing(): Main endpoint for parsing requests
    - tree_to_json_with_all_info(): Constituency tree to JSON converter
    - debug_parse(): Debug endpoint for raw parse data
"""

import logging
logging.getLogger('stanza').setLevel(logging.ERROR)

from dotenv import load_dotenv
from flask import Flask, jsonify, request
import stanza
import os
import json
import sys

load_dotenv()
STANZA_API_TOKEN = os.getenv("STANZA_API_TOKEN")

# Try loading Stanza: default_accurate first, then default, then error
import torch
from pathlib import Path

nlp = None
package_loaded = None

# Check if stanza models are cached — if so, skip downloads
stanza_cache = Path.home() / 'AppData' / 'Local' / 'StanfordNLP' / 'stanza' / 'Cache'
if not stanza_cache.exists():
    # Also check Unix path
    stanza_cache = Path.home() / 'stanza_resources'

stanza_cached = any(stanza_cache.rglob('en/default*')) if stanza_cache.exists() else False
download_method = None if stanza_cached else stanza.DownloadMethod.REUSE_RESOURCES

# Try accurate model first — always allow downloads for transformer weights
try:
    print("Loading Stanza NLP pipeline (package: default_accurate)...")
    nlp = stanza.Pipeline(
        'en',
        package='default_accurate',
        processors='tokenize,pos,lemma,depparse,constituency',
        use_gpu=True,
        download_method=stanza.DownloadMethod.REUSE_RESOURCES
    )
    package_loaded = 'default_accurate'
except ImportError as e:
    print(f"\n  [PARSELY-DIP] 'default_accurate' requires additional packages.")
    print(f"  This model uses PEFT fine-tuned transformers for better accuracy.")
    print(f"  The biggest improvement is in constituency parsing — the core of NLP intent matching.")
    print(f"  Requires ~1-2GB extra VRAM on a dedicated GPU.\n")
    print(f"  Install required packages: py -m pip install transformers sentencepiece")
    print(f"  For GPU acceleration:     py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130\n")
    print(f"  Error: {e}\n")
    print(f"  Options:")
    print(f"    1. Exit and install the packages above, then restart (recommended for GPU users)")
    print(f"    2. Continue with the standard CharLM model (solid accuracy, no extra dependencies)\n")
    choice = input("  Enter 1 or 2: ").strip()
    if choice != '2':
        sys.exit(1)
except Exception as e:
    import traceback
    print(f"\n  [PARSELY-DIP] Failed to load 'default_accurate':\n")
    traceback.print_exc()
    print(f"\n  Options:")
    print(f"    1. Exit and fix the issue above")
    print(f"    2. Continue with the standard model\n")
    choice = input("  Enter 1 or 2: ").strip()
    if choice != '2':
        sys.exit(1)

# If accurate failed, try default
if nlp is None:
    try:
        print("Loading Stanza NLP pipeline (package: default)...")
        nlp = stanza.Pipeline(
            'en',
            package='default',
            processors='tokenize,pos,lemma,depparse,constituency',
            use_gpu=True,
            download_method=download_method
        )
        package_loaded = 'default'
    except Exception as e:
        print(f"  Failed to load 'default': {e}")

if nlp:
    gpu_in_use = torch.cuda.is_available()
    print(f"Stanza NLP pipeline loaded successfully (package: {package_loaded}).")
    if gpu_in_use:
        print(f"Using GPU: Yes ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using GPU: No (CPU mode)")
        print(f"  For GPU acceleration: py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130")
else:
    print("\n[PARSELY-DIP] No Stanza English model found.")
    print("The NLP layer requires a Stanza English model for constituency and dependency parsing.\n")
    print("Download the default model (~526MB):")
    print("    py -c \"import stanza; stanza.download('en')\"\n")
    print("Or download the 'default_accurate' model (uses transformers, more accurate):")
    print("    py -c \"import stanza; stanza.download('en', package='default_accurate')\"")
    print("    - Better constituency parsing accuracy (biggest improvement area)")
    print("    - Requires ~1-2GB extra VRAM on GPU")
    print("    - Minimal per-query overhead once loaded — a few extra seconds at startup")
    print("    - Recommended if you have a dedicated GPU (RTX 3060+, etc.)")
    print("    - The default CharLM-based model is fine for most use cases\n")
    print("Note: There is no lightweight option. You need at least the default model.")
    print()
    sys.exit(1)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")


@app.before_request
def restrict_access():
    """
    Security middleware to restrict API access

    Validates all incoming requests before processing:
    1. Source IP must be 127.0.0.1 (localhost only)
    2. X-Stanza-Token header must match STANZA_API_TOKEN

    Returns:
        None: Allows request to continue if valid
        tuple: (error_json, 403) if validation fails
    """
    # Check source IP
    if request.remote_addr != '127.0.0.1':
        return jsonify({"error": "Access denied: Invalid source IP"}), 403

    # Verify token
    token = request.headers.get('X-Stanza-Token')
    if token != STANZA_API_TOKEN:
        return jsonify({"error": "Access denied: Invalid token"}), 403


def tree_to_json_with_all_info(tree, sentence, word_index=None):
    """
    Convert Stanza constituency tree to JSON with full linguistic annotations

    Copied from uni-app/services/stanza_nlp.py

    Recursively traverses constituency parse tree, enriching each leaf node
    with linguistic features: lemma, POS tag, dependency relation, head word,
    and before/after context words.

    Args:
        tree: Stanza constituency Tree object
        sentence: Stanza Sentence object with .words list
        word_index (list[int]): Mutable counter for current word position

    Returns:
        dict: Nested dictionary mirroring tree structure with enriched leaf nodes
    """
    if word_index is None:
        word_index = [0]

    if not tree.children:  # Leaf node
        if word_index[0] < len(sentence.words):
            word = sentence.words[word_index[0]]

            # Find the head word info (for readability)
            head_text = None
            head_lemma = None
            if word.head > 0:  # head=0 means root
                head_word = sentence.words[word.head - 1]  # head is 1-indexed
                head_text = head_word.text
                head_lemma = head_word.lemma

            # Get before/after word info
            before_text = None
            before_lemma = None
            after_text = None
            after_lemma = None

            current_idx = word_index[0]

            if current_idx > 0:
                before_word = sentence.words[current_idx - 1]
                before_text = before_word.text
                before_lemma = before_word.lemma

            if current_idx < len(sentence.words) - 1:
                after_word = sentence.words[current_idx + 1]
                after_text = after_word.text
                after_lemma = after_word.lemma

            result = {
                "label": tree.label,
                "text": word.text,
                "lemma": word.lemma,
                "pos": word.upos,
                "dep": word.deprel,
                "head": word.head,
                "head_text": head_text,
                "head_lemma": head_lemma,
                "before_text": before_text,
                "before_lemma": before_lemma,
                "after_text": after_text,
                "after_lemma": after_lemma
            }
            word_index[0] += 1
            return result
        else:
            return {tree.label: tree.label}
    else:
        return {
            tree.label: [tree_to_json_with_all_info(child, sentence, word_index) for child in tree.children]
        }


@app.route('/process_syntactic_parsing', methods=['POST'])
def process_syntactic_parsing():
    """
    Parse user input and return linguistic data

    Pure NLP parsing service - receives text, returns parsed linguistic features.
    No pattern matching or intent detection occurs here.

    PROCESSING FLOW:
        1. Extract user_input from POST JSON
        2. Run Stanza NLP pipeline (tokenize, POS, lemma, dependency, constituency)
        3. Return parsed sentence data

    REQUEST:
        POST /process_syntactic_parsing
        Body: {"user_input": "What time is it?"}
        Headers: X-Stanza-Token: <token>

    RESPONSE:
        {
            "sentences": [
                {
                    "text": "What time is it?",
                    "words": [
                        {"text": "What", "lemma": "what", "pos": "DET", "dep": "det"},
                        {"text": "time", "lemma": "time", "pos": "NOUN", "dep": "root"},
                        ...
                    ]
                }
            ]
        }

    Returns:
        tuple: (jsonify(dict), status_code)
    """
    data = request.get_json()
    user_input = data.get('user_input', '')

    if not user_input:
        return jsonify({"sentences": []}), 200

    # Process with Stanza
    doc = nlp(user_input)

    sentences = []
    for sentence in doc.sentences:
        # Extract word-level features
        words = []
        for word in sentence.words:
            # Calculate head_lemma (lemma of the word that this word depends on)
            head_lemma = None
            if word.head > 0:  # head=0 means root
                head_word = sentence.words[word.head - 1]  # head is 1-indexed
                head_lemma = head_word.lemma

            words.append({
                "text": word.text,
                "lemma": word.lemma,
                "pos": word.upos,
                "dep": word.deprel,
                "head": word.head,
                "head_lemma": head_lemma
            })

        # Extract constituency tree
        constituency = tree_to_json_with_all_info(sentence.constituency, sentence)

        sentences.append({
            "text": sentence.text,
            "words": words,
            "constituency": constituency
        })

    return jsonify({"sentences": sentences}), 200


@app.route('/debug_parse', methods=['POST'])
def debug_parse():
    """
    Debug endpoint - returns raw parse data (POS tags, constituency tree)

    Returns raw linguistic data without capability matching.
    For debugging and understanding sentence structure.

    REQUEST:
        POST /debug_parse
        Body: {"user_input": "Can you tell time?"}
        Headers: X-Stanza-Token: <token>

    RESPONSE:
        {
            "sentence": "Can you tell time?",
            "words": [
                {"text": "Can", "pos": "MD", "lemma": "can", "dep": "aux"},
                ...
            ],
            "constituency_tree": "(ROOT (SQ (MD Can) ...))"
        }
    """
    data = request.get_json()
    user_input = data.get('user_input', '')

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Process with Stanza
    doc = nlp(user_input)

    results = []
    for sentence in doc.sentences:
        words = []
        for word in sentence.words:
            head_text = sentence.words[word.head - 1].text if word.head > 0 else "ROOT"
            words.append({
                "text": word.text,
                "pos": word.upos,
                "lemma": word.lemma,
                "dep": word.deprel,
                "head": head_text
            })

        # Get enriched constituency tree with word details
        enriched_tree = tree_to_json_with_all_info(sentence.constituency, sentence)

        results.append({
            "sentence": sentence.text,
            "words": words,
            "constituency_tree": str(sentence.constituency),
            "visual_tree": format_tree_visual(sentence.constituency),
            "enriched_tree": enriched_tree
        })

    return jsonify({"results": results}), 200


def format_tree_visual(tree, prefix="", is_last=True):
    """Render a Stanza constituency tree as an indented visual tree with branch lines."""
    connector = "└── " if is_last else "├── "
    label = tree.label

    if not tree.children:
        return prefix + connector + label + "\n"

    result = prefix + connector + label + "\n"
    child_prefix = prefix + ("    " if is_last else "|   ")
    for i, child in enumerate(tree.children):
        result += format_tree_visual(child, child_prefix, i == len(tree.children) - 1)
    return result


def interactive_chat():
    """Interactive mode to explore sentence structure."""
    print("\n=== Stanza NLP Structure Explorer ===")
    print("Type a sentence to see its parse structure.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input(">>> ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input:
                continue

            doc = nlp(user_input)
            for sentence in doc.sentences:
                print(f"\n--- Constituency Tree (inline) ---")
                print(sentence.constituency)

                print(f"\n--- Constituency Tree (visual) ---")
                print(format_tree_visual(sentence.constituency))

                print(f"\n--- Words (POS + Dependency) ---")
                for word in sentence.words:
                    head_text = sentence.words[word.head - 1].text if word.head > 0 else "ROOT"
                    print(f"  {word.text:15} POS={word.upos:6} DEP={word.deprel:10} HEAD={head_text}")

                print()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("Goodbye!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ['--chat', '-c']:
        interactive_chat()
    else:
        print("Starting PARSELY-DIP Stanza NLP service on port 5013...")
        app.run(debug=True, host='127.0.0.1', port=5013)
