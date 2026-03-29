# PARSELY-DIP

**Parsing And RegEx Syntactic Engine with Linguistic Yield — Deterministic Intent Parser**

*Parsely dip for silicon chips.*

A deterministic intent recognition engine that processes natural language through a cascading pipeline — RegEx first, then constituency and dependency parsing via Stanza, then LLM fallback. Each layer only fires if the one above didn't match. The cheapest, fastest layer runs first. The LLM is the last resort, not the default.

Your LLM is expensive, slow, and unpredictable. When a user says "what time is it" or "move the card to done," there is zero ambiguity. A regex handles it in microseconds. An LLM spends tokens guessing what you already know. PARSELY-DIP intercepts deterministic commands before they reach the LLM, executes them directly, and returns the result.

## What It Does

```python
from parsely_dip import parse

result = parse("what time is it")
# result = "14:32"

result = parse("what is the weather like")
# result = "It's 36°F and broken clouds in Cleveland."

result = parse("tell me about quantum physics")
# result = None  (no match — pass to LLM)
```

One call. One input. Response string or None.

## Install

```bash
pip install parsely-dip
```

From source:

```bash
git clone https://github.com/gbutiri/parsely-dip.git
cd parsely-dip
pip install -e .
```

### NLP Layer Setup (Optional)

The RegEx layer works out of the box. The NLP layer requires Stanza and a running parse service.

**1. Download the Stanza English model (~526MB):**

```bash
python -c "import stanza; stanza.download('en')"
```

**2. (Recommended) Download the accurate model with transformer support:**

```bash
python -c "import stanza; stanza.download('en', package='default_accurate')"
pip install transformers sentencepiece
```

The `default_accurate` model uses PEFT fine-tuned transformers (Google Electra Large). The biggest accuracy improvement is in constituency parsing — the core of NLP intent matching. Requires ~1-2GB extra VRAM on a dedicated GPU.

**3. (Recommended) Install PyTorch with GPU support:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Without this, Stanza runs on CPU. With a dedicated GPU (RTX 3060+), parsing is near-instant.

**4. Start the NLP service:**

```bash
python -m parsely_dip.engine.stanza_service
```

The service loads once and stays running. PARSELY-DIP calls it via HTTP on port 5013 for each query that passes the RegEx layer. The service auto-detects the best available model (`default_accurate` > `default`) and reports GPU status on startup.

---

## Three-Tier Pipeline

```
User Input
    |
    v
[RegEx Layer]  — Pattern matching, microseconds, zero dependencies
    |  match? --> handler executes, returns response
    |  no match? --> continue
    v
[NLP Layer]    — Stanza constituency + dependency parsing via HTTP service
    |  match? --> handler executes, returns response
    |  no match? --> continue
    v
[LLM Fallback] — parse() returns None, caller decides what to do
```

### Layer 1: RegEx

Patterns stored in flat `.patterns` text files. One pattern per line. No JSON escaping nightmares.

```
# Format: (regex) => intent_name
# intents/time.py
(what('s|\s+is)\s+the\s+time|what\s+time\s+is\s+it)\?? => tell_time

# intents/weather.py
((what|how)('s|\s+is)\s+the\s+weather(\s+like)?)\?? => tell_weather

# intents/scrum.py
show(\s+me|\s+us)?\s+the\s+(current|active)(\s+scrum)?\s+cards?[.!]? => show_current_card
```

**Pattern convention:** `\s+` goes BEFORE the word it separates, not after.

```
CORRECT: (what('s|\s+is)\s+the\s+time)
WRONG:   (what('s|is\s+)the\s+time\s+)
```

The space belongs to the approach of the next word, not trailing from the previous.

Each pattern is a named capture group mapped to an intent. When a pattern matches, the associated handler fires immediately and the pipeline stops — no NLP service call, no model inference, no latency. Regex handles the majority of real-world intents because most user commands fall into a small set of stable, predictable surface forms. When someone types "what time is it" or "show me the current card," there is exactly one thing they could mean. A regex resolves it in microseconds.

When regex cannot match — polite variations, embedded clauses, unpredictable word order — the pipeline falls through to the NLP layer.

### Layer 2: NLP

Patterns stored in `.json` files. Each pattern defines a grammatical structure using sentence type, POS tags, dependency relations, and head words. Matches on linguistic features, not exact strings — so "what time is it, please?" and "hey, what's the time right now?" both match without needing separate regex patterns.

```json
[
  {
    "intent": "tell_time",
    "nlp": {
      "sentence_type": ["SBARQ", "SQ", "WHNP"],
      "words": [
        {"word": "what", "pos": "DET", "dep": "det", "required": true},
        {"lemma": "time", "pos": "NOUN", "required": true},
        {"lemma": "be", "pos": "AUX", "dep": "cop", "required": true},
        {"word": "it", "pos": "PRON", "dep": "nsubj", "required": true}
      ]
    }
  }
]
```

The NLP layer requires the Stanza service running on port 5013. If the service is not running, the NLP layer is silently skipped and the pipeline falls through to LLM.

### Why NLP Over RegEx for Intent Detection

RegEx matches exact strings. If someone says "what time is it" your pattern fires. But when they say "what's the time, please?" — different contraction, added article, trailing politeness — your regex misses. You write another pattern. Then "could you tell me the time?" needs a third. Every variation is a new regex. It does not scale.

NLP matches grammatical structure. Compare these two parses:

**"What time is it?"**
```
  What            POS=DET    DEP=det        HEAD=time
  time            POS=NOUN   DEP=root       HEAD=ROOT
  is              POS=AUX    DEP=cop        HEAD=time
  it              POS=PRON   DEP=nsubj      HEAD=time
```

**"What's the time, please?"**
```
  What            POS=PRON   DEP=root       HEAD=ROOT
  's              POS=AUX    DEP=cop        HEAD=What
  the             POS=DET    DEP=det        HEAD=time
  time            POS=NOUN   DEP=nsubj      HEAD=What
  ,               POS=PUNCT  DEP=punct      HEAD=please
  please          POS=INTJ   DEP=discourse  HEAD=time
```

Different words, different structure, same core features: a NOUN "time", an AUX copula "be" (lemmatized from "'s" and "is"), and a question sentence type (SBARQ). One NLP pattern catches both. The extra words — "the", "please", punctuation — are ignored because they are not marked `required` in the pattern. The pattern matches on the grammatical skeleton, not the surface text.

### Same Meaning, Different Trees

Two sentences can have completely different constituency trees and still express the same intent. The trees above prove it — "What time is it?" has `time` as the root with `What` as its determiner. "What's the time, please?" flips it — `What` becomes the root and `time` becomes the subject. The tree structure changed. The dependency roles shifted. But the meaning is identical: the user wants to know the time.

This is the key insight. As sentences grow more complex — "hey, do you think you could possibly tell me what time it is right now?" — the tree gets deeper, more clauses nest inside each other, and the surface text looks nothing like the original. But buried inside that tree, the same core features exist: a NOUN "time", a question structure, and a copula linking them. The NLP pattern finds those features regardless of how many layers of politeness, hedging, or subordination surround them.

RegEx sees characters. NLP sees grammar. Grammar is stable across paraphrases. Characters are not.

### Why Structure Matters More Than Keywords

A regex pattern like `(time|weather|apples)` will match the keyword anywhere — in a question, a statement, a song lyric. It has no concept of what role that word plays in the sentence. NLP does. Consider this sentence that has nothing to do with asking about time or weather:

**"I went to the store and bought some apples."**
```
--- Constituency Tree (visual) ---
└── ROOT
    └── S
        ├── NP
        |   └── PRP
        |       └── I
        ├── VP
        |   ├── VP
        |   |   ├── VBD
        |   |   |   └── went
        |   |   └── PP
        |   |       ├── IN
        |   |       |   └── to
        |   |       └── NP
        |   |           ├── DT
        |   |           |   └── the
        |   |           └── NN
        |   |               └── store
        |   ├── CC
        |   |   └── and
        |   └── VP
        |       ├── VBD
        |       |   └── bought
        |       └── NP
        |           ├── DT
        |           |   └── some
        |           └── NNS
        |               └── apples
        └── .
            └── .

--- Words (POS + Dependency) ---
  I               POS=PRON   DEP=nsubj      HEAD=went
  went            POS=VERB   DEP=root       HEAD=ROOT
  to              POS=ADP    DEP=case       HEAD=store
  the             POS=DET    DEP=det        HEAD=store
  store           POS=NOUN   DEP=obl        HEAD=went
  and             POS=CCONJ  DEP=cc         HEAD=bought
  bought          POS=VERB   DEP=conj       HEAD=went
  some            POS=DET    DEP=det        HEAD=apples
  apples          POS=NOUN   DEP=obj        HEAD=bought
```

This is a declarative sentence (S), not a question (SBARQ). The root is a VERB "went", not a NOUN "time". There is no AUX copula, no question pronoun, no interrogative structure at all. A regex with a loose wildcard — say `.*time.*` or `.*store.*` — could false-positive on "I don't have time to go to the store." The regex sees the word "time" and fires. But the NLP layer sees that "time" in that sentence is an object of "have", not the root of a question, and the sentence type is S (declarative), not SBARQ (question). The pattern does not match.

This is the tradeoff. NLP uses more resources than regex — it requires a running Stanza service, a loaded model, and a round-trip HTTP call. Regex runs in microseconds with zero dependencies. But regex can only match character sequences, and character sequences lie. The word "time" appears in thousands of sentences that have nothing to do with asking the time. A wildcard regex that catches all the ways someone might ask "what time is it" will inevitably also catch sentences where "time" is used as a verb ("time the race"), an adjective modifier ("time machine"), or an object of a completely unrelated verb ("I wasted time"). Every wildcard you add to cover more phrasings also opens the door to more false positives.

NLP eliminates this entire class of errors by matching on grammatical role, not surface text. The word "time" must be a NOUN, it must be in a question structure, and it must have a copula linking it. If any of those structural requirements are missing, the pattern does not fire — no matter how many times the word "time" appears in the sentence. The cost is higher per query (milliseconds instead of microseconds), but the accuracy is categorically better. For deterministic intent matching, accuracy is the only thing that matters. A false positive that triggers the wrong handler is worse than no match at all, because no match falls through to the LLM which can handle ambiguity. A false positive executes the wrong action with full confidence.

### Real-World Scenarios: Commands vs Thinking

In practice, different environments produce different kinds of input. A workspace command line sees short, imperative commands: "move the file", "show the card", "deploy to staging." A conversational assistant sees open-ended input with detail, politeness, and embedded clauses. The regex and NLP layers each excel in one of these scenarios.

#### Scenario 1: Imperative Commands with Detail

Consider a developer telling their assistant to reorganize a file:

**"Move the README.md file to the done folder."**
```
--- Constituency Tree (visual) ---
└── ROOT
    └── S
        ├── VP
        |   ├── VB
        |   |   └── Move
        |   ├── NP
        |   |   ├── DT
        |   |   |   └── the
        |   |   ├── NN
        |   |   |   └── README
        |   |   ├── NN
        |   |   |   └── .md
        |   |   └── NN
        |   |       └── file
        |   └── PP
        |       ├── IN
        |       |   └── to
        |       └── NP
        |           ├── DT
        |           |   └── the
        |           ├── JJ
        |           |   └── done
        |           └── NN
        |               └── folder
        └── .
            └── .

--- Words (POS + Dependency) ---
  Move            POS=VERB   DEP=root       HEAD=ROOT
  the             POS=DET    DEP=det        HEAD=file
  README          POS=NOUN   DEP=compound   HEAD=file
  .md             POS=NOUN   DEP=compound   HEAD=file
  file            POS=NOUN   DEP=obj        HEAD=Move
  to              POS=ADP    DEP=case       HEAD=folder
  the             POS=DET    DEP=det        HEAD=folder
  done            POS=ADJ    DEP=amod       HEAD=folder
  folder          POS=NOUN   DEP=obl        HEAD=Move
```

The parse tree breaks this sentence into its operational components: a VERB root ("Move"), an object NP ("the README.md file"), and a destination PP ("to the done folder"). A regex could handle this exact phrasing — `move\s+the\s+.*\s+to\s+the\s+.*\s+folder` — but what happens when the user says "Move the README.md file to the done folder, please"? Or "Could you move the README.md file to the done folder?" The regex either misses or you add more patterns. The NLP layer does not care about the "please" or the "could you" — those words are not required in the pattern. The structural core remains: a VERB "move", an object NOUN, a prepositional destination. The pattern fires regardless of how the user wraps the command.

More importantly, the NLP layer can extract the operands. The object of "Move" is "file" (with compounds "README" and ".md"). The oblique destination is "folder" (with modifier "done"). These are not just matched — they are parsed into named grammatical roles that a handler can read. A regex gives you capture groups of character sequences. NLP gives you a grammatical decomposition of what is being moved, and where.

#### Scenario 2: Possession and Slot-Based Matching

Not every intent requires specific words. Some patterns are structural — they match any sentence that fits a grammatical template, regardless of the nouns involved.

**"I have a cat."**
```
--- Constituency Tree (visual) ---
└── ROOT
    └── S
        ├── NP
        |   └── PRP
        |       └── I
        ├── VP
        |   ├── VBP
        |   |   └── have
        |   └── NP
        |       ├── DT
        |       |   └── a
        |       └── NN
        |           └── cat
        └── .
            └── .

--- Words (POS + Dependency) ---
  I               POS=PRON   DEP=nsubj      HEAD=have
  have            POS=VERB   DEP=root       HEAD=ROOT
  a               POS=DET    DEP=det        HEAD=cat
  cat             POS=NOUN   DEP=obj        HEAD=have
```

This is a simple possession statement: subject PRON ("I"), VERB root ("have"), object NOUN ("cat"). The key insight is that the NOUN in the object position is a slot — it could be "cat", "dog", "computer", "headache", or anything else. The grammatical structure is identical in every case: `PRON(nsubj) → VERB(have/root) → NOUN(obj)`.

An NLP pattern for detecting possession does not need to know what the user possesses. It only needs to verify:
- The root VERB is "have" (lemma match)
- There is a PRON subject (the possessor)
- There is a NOUN object (the possessed thing)

```json
{
  "intent": "detect_possession",
  "nlp": {
    "sentence_type": "S",
    "words": [
      {"pos": "PRON", "dep": "nsubj", "required": true},
      {"lemma": "have", "pos": "VERB", "dep": "root", "required": true},
      {"pos": "NOUN", "dep": "obj", "required": true}
    ]
  }
}
```

Notice the third word has no `word` or `lemma` field — just `pos` and `dep`. This is a slot. It matches any NOUN that serves as the object of "have." The handler can then read what that NOUN actually is and act accordingly.

Try doing this with regex. You would need a pattern like `I\s+have\s+a\s+(\w+)` — but that only catches "I have a [single word]." It misses "I have two cats", "I have a big red car", "I've got a cat." To cover those, you start adding alternations and optional groups, and eventually you are building a regex that approximates a grammar parser — badly. Or you build a category lexicon — a list of all possible nouns that could appear in that position — and check against it. That lexicon needs constant maintenance as new words appear.

NLP skips all of that. The POS tagger already knows "cat" is a NOUN. The dependency parser already knows it is the object of "have." The pattern matches on those structural facts. No lexicon needed. No word list to maintain. Any NOUN the language can produce in that grammatical position will match the slot.

This is where NLP patterns fundamentally differ from regex: they can define intent by grammatical shape rather than by vocabulary. A "possession" pattern works for every possessable noun in the English language without listing a single one.

### Layer 3: LLM Fallback

`parse()` returns `None`. The caller decides what to do — send to an LLM, show an error, or ignore. PARSELY-DIP does not call any LLM itself.

---

## Intent Handlers

Self-registering via the `@intent` decorator. Import the module, the decorator registers the handler. No config files, no setup step.

```python
from parsely_dip.engine.registry import intent

@intent('tell_time')
def tell_time():
    from datetime import datetime
    now = datetime.now()
    return f"{now.hour:02d}:{now.minute:02d}"
```

### Built-in Intents

| Intent | File | What It Does |
|--------|------|-------------|
| `tell_time` | `intents/time.py` | Returns current time in 24-hour format |
| `tell_weather` | `intents/weather.py` | Returns weather via OpenWeatherMap API (requires `WEATHER_API_KEY` in `.env`) |
| `show_current_card` | `intents/scrum.py` | Shows active scrum cards from SQLite database |
| `read_current_card` | `intents/scrum.py` | Same data as show, but intended for LLM to summarize |

### Adding New Intents

1. Create a new file in `intents/` (e.g., `intents/greeting.py`)
2. Write a handler function with the `@intent` decorator
3. Add regex patterns to `patterns/base.patterns`
4. (Optional) Add NLP patterns to `patterns/base_nlp.json`
5. Import the module in `__init__.py`

---

## Project Structure

```
parsely-dip/
  pyproject.toml           — Package config, dependencies
  README.md                — This file
  env_parselydip/          — Virtual environment
  db/                      — Database files (if needed by intents)
  logs/                    — Log files
  tests/                   — Test suite
  src/parsely_dip/
    __init__.py            — parse(prompt) single entry point
    engine/
      registry.py          — @intent decorator, handler registry, dispatch()
      regex.py             — load_patterns(), check_regex()
      nlp.py               — load_nlp_patterns(), check_nlp(), match_nlp_pattern()
      splitter.py          — Sentence splitting (future expansion)
      stanza_service.py    — Stanza NLP Flask service (port 5013)
    intents/
      __init__.py           — Auto-imports all intent modules
      time.py               — tell_time handler
      weather.py            — tell_weather handler (OpenWeatherMap API)
      scrum.py              — show_current_card, read_current_card handlers
    patterns/
      base.patterns         — RegEx patterns (flat text, one per line)
      base_nlp.json         — NLP patterns (structured JSON)
    cli/
      __init__.py           — CLI entry point (future)
```

---

## Hook Integration

PARSELY-DIP is designed to run as a Claude Code `UserPromptSubmit` hook. The hook intercepts the user's message, runs it through the pipeline, and either handles it deterministically or lets the LLM process it.

### Hook Script

```bash
#!/bin/bash
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
VENV_PY="$PROJECT_DIR/env_bibliotech/Scripts/python.exe"
[ ! -f "$VENV_PY" ] && exit 0

"$VENV_PY" -c "
import sys, json
from parsely_dip import parse
data = json.load(sys.stdin)
prompt = data.get('prompt', '')
if prompt:
    r = parse(prompt)
    if r:
        print('=== PARSELY-DIP ===')
        print('Relay this to the user EXACTLY as written, nothing else:')
        print(r)
        print('=== END PARSELY-DIP ===')
" 2>/dev/null
exit 0
```

### How It Works

1. Hook reads the user's prompt from stdin (JSON with `prompt` field)
2. Calls `parsely_dip.parse(prompt)`
3. If result: prints it to stdout (shown to LLM as context, LLM relays verbatim)
4. If None: no output, LLM processes the prompt normally

### Known Limitation

Claude Code's `UserPromptSubmit` hooks cannot display text directly to the user without the LLM firing. The documented `decision: "block"` + `reason` field blocks the prompt but does not render the reason in the VS Code extension (confirmed bug). The current approach uses plain text stdout with exit 0 — the LLM sees the result and relays it.

---

## Stanza NLP Service

The NLP service is a Flask app that wraps Stanford's Stanza NLP library. It runs as a background service on port 5013, loads the model once at startup, and handles parse requests via HTTP.

### Starting the Service

```bash
python -m parsely_dip.engine.stanza_service
```

### What Happens at Startup

1. Tries to load `default_accurate` (transformer-based, best accuracy)
2. If that fails (missing packages), prompts the user to install or continue with standard
3. Falls back to `default` (CharLM-based, solid accuracy)
4. If no model found, prints install instructions and exits
5. Reports GPU status (name of GPU if available, install command if not)

### Service Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/process_syntactic_parsing` | POST | Parse text, return words with POS/dependency/constituency |
| `/debug_parse` | POST | Raw parse data for debugging sentence structure |

### Interactive Mode

```bash
python -m parsely_dip.engine.stanza_service --chat
```

Opens an interactive prompt where you can type sentences and see their full parse structure — constituency trees (inline and visual), POS tags, and dependency relations. Useful for building new NLP patterns.

```
>>> What's your name?

--- Constituency Tree (inline) ---
(ROOT (SBARQ (WHNP (WP What)) (SQ (SQ (VBZ 's) (NP (PRP$ your) (NN name)))) (. ?)))

--- Constituency Tree (visual) ---
└── ROOT
    └── SBARQ
        ├── WHNP
        |   └── WP
        |       └── What
        ├── SQ
        |   └── SQ
        |       ├── VBZ
        |       |   └── 's
        |       └── NP
        |           ├── PRP$
        |           |   └── your
        |           └── NN
        |               └── name
        └── .
            └── ?

--- Words (POS + Dependency) ---
  What            POS=PRON   DEP=root       HEAD=ROOT
  's              POS=AUX    DEP=cop        HEAD=What
  your            POS=PRON   DEP=nmod:poss  HEAD=name
  name            POS=NOUN   DEP=nsubj      HEAD=What
  ?
```

### Security

- Localhost only (127.0.0.1) — rejects non-local requests
- Optional token auth via `STANZA_API_TOKEN` environment variable — enforced if set, skipped if not

---

## NLP Pattern Specification

NLP patterns define grammatical structures that map to intents. Unlike regex (exact string matching), NLP patterns match on linguistic features extracted by Stanza.

### Pattern Structure

```json
{
  "intent": "intent_name",
  "nlp": {
    "sentence_type": "SBARQ",
    "words": [
      {
        "word": "exact_word",
        "lemma": "base_form",
        "pos": "NOUN",
        "dep": "nsubj",
        "head_lemma": "parent_word",
        "required": true
      }
    ]
  }
}
```

### Matching Modes

- **Exact Word Match** — `word` specified: match that exact word in that grammatical position
- **Structural Match (Slot)** — `word` empty: match ANY word with specified POS + dependency features
- **Optional Words** — `required: false`: pattern matches with or without this word

### Supported Values

**Sentence Types:** S, SBARQ, SQ, SINV, FRAG (+ 20 more constituency labels)

**POS Tags (17 Universal):** NOUN, VERB, AUX, ADJ, ADV, PRON, DET, ADP, NUM, PART, CCONJ, SCONJ, INTJ, PROPN, PUNCT, SYM, X

**Dependency Relations (37+):** nsubj, obj, root, det, cop, aux, mark, case, advmod, amod, compound, conj, cc, xcomp, ccomp, advcl, acl, nmod, obl, nummod, appos, dep, fixed, flat, list, parataxis, orphan, goeswith, reparandum, punct, clf, discourse, dislocated, expl, iobj, vocative, csubj

### Specificity Rule

**A loose pattern that matches incorrectly is WORSE than no pattern (LLM fallback).**

Every NLP pattern must be maximally specific. Include all words that disambiguate the intent — articles, pronouns, structural words. If removing a word would cause false positives, that word is required.

---

## Configuration

### .env

```
WEATHER_API_KEY=your_openweathermap_key
STANZA_API_TOKEN=optional_security_token
```

### pyproject.toml Dependencies

```toml
dependencies = [
    "stanza>=1.5",
    "requests>=2.28",
    "python-dotenv>=1.0",
    "flask>=3.0",
]
```

Optional (for `default_accurate` model):
```
pip install transformers sentencepiece
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

---

## Requirements

- Python 3.9+
- Stanza 1.5+ (for NLP layer)
- Flask 3.0+ (for NLP service)
- A dedicated GPU is recommended but not required (RTX 3060+ for transformer models)
- The RegEx layer works with zero dependencies beyond the base package

## Why Deterministic Matters

An LLM interprets. PARSELY-DIP executes. The difference matters when ambiguity has consequences.

### The Pipeline

```
User Input
     |
     v
[Loaded Skill File]          <- domain-specific patterns
     |
     v
[RegEx Match] ────────────── match found ──> [Handler/Protocol] ──> Response
     |                                         (3-10 lines of code)
     | no match
     v
[NLP Match] ─────────────── match found ──> [Handler/Protocol] ──> Response
     |                                       (structural match)
     | no match
     v
[LLM Fallback]              <- only fires when nothing matched
     |
     v
Caller decides what to do
```

Every matched intent executes a handler — a Python function that does exactly one thing. The `tell_time` handler is three lines:

```python
@intent('tell_time')
def tell_time():
    from datetime import datetime
    now = datetime.now()
    return f"{now.hour:02d}:{now.minute:02d}"
```

No token cost. No latency. No hallucination. No "I think it might be around 3pm." It is 04:07. Done.

An LLM asked the same question will spend tokens reasoning about timezone preferences, 12-hour vs 24-hour format, whether you meant wall clock or elapsed time, and may still get it wrong. The handler calls `datetime.now()` and returns the answer. The LLM never sees the question.

### Domain-Specific Skill Files

The patterns loaded into PARSELY-DIP define the domain. The same engine serves completely different environments by swapping which `.patterns` and `_nlp.json` files are loaded.

A surgical suite loads `surgical.patterns`:

```
(scalpel)\s*[.!]? => hand_instrument
(clamp)\s*[.!]? => hand_instrument
(suction)\s*[.!]? => activate_suction
(close)\s*[.!]? => begin_closure
```

A surgeon says "scalpel." That single word means: identify the scalpel on the instrument tray, actuate the robotic arm to retrieve it, position it for handoff, confirm grip transfer. The handler knows all of this. The regex matched in microseconds. There is no LLM in the loop deciding whether the surgeon really needs the scalpel or perhaps meant something else.

A military operations center loads `tactical.patterns` and `tactical_nlp.json`:

```
(medevac)\s*[.!]? => request_extraction
(extract(ion)?)\s*[.!]? => request_extraction
(out\s+of\s+ammo)\s*[.!]? => resupply_request
(winchester)\s*[.!]? => resupply_request
```

"Medevac" and "we need extraction" are two different commands that both mean people need to be pulled out of a dangerous situation — but "medevac" additionally signals wounded personnel, which changes the response protocol (medical team on the receiving helicopter, triage preparation at the landing zone). Two patterns, two intents, or the same intent with a metadata flag. The skill file defines it. The handler executes it.

"Out of ammo" on a battlefield triggers a resupply protocol. "Out of ammo" in a business context means nothing. The loaded skill file determines which interpretation wins. There is no LLM weighing probabilities. The pattern matched. The protocol runs.

### Context Is Not Ambiguity

An LLM treats every input as a reasoning problem. It considers context, weighs alternatives, generates a probabilistic response. That is powerful for open-ended conversation. It is dangerous for commands where the meaning is already known.

"Crush them" in a military briefing means engage the enemy with overwhelming force. "Crush them" in a business meeting means outperform the competition. "Crush them" in a kitchen means pulverize the garlic cloves. An LLM with no domain context will guess. A PARSELY-DIP skill file loaded for a military operations center does not guess — it maps "crush them" to the correct tactical protocol because that is the only interpretation that exists in the loaded pattern set.

The skill file is not just a vocabulary list. It is a commitment: these are the commands this system understands, these are the actions those commands trigger, and nothing else happens. If the input does not match a loaded pattern, the system explicitly says "I don't know what that means" — or passes it to an LLM for open-ended handling. There is no middle ground where a deterministic command gets probabilistically misinterpreted.

### The Handler Is the Proof

Every handler in PARSELY-DIP is a small, testable, deterministic function. It does not reason. It does not infer. It reads the matched intent, executes the protocol, and returns the result.

The `tell_time` handler is 3 lines. A weather handler is 10 lines (API call, format response). A scrum card handler is 15 lines (database query, format output). A surgical instrument handler would be whatever the robotic arm API requires — but the decision to pick up the scalpel was made in microseconds by a regex, not in seconds by an LLM.

The size of the handler is the point. When the intent is known, the action is small. The complexity belongs in the matching layer (did the user really mean this?) not in the execution layer (what do I do about it?). PARSELY-DIP puts all the intelligence in the matching — regex for surface forms, NLP for grammatical structure — so the handler can be as simple as the action requires.

The LLM is still there. It handles everything the patterns do not cover — open-ended questions, creative requests, ambiguous input. But for the commands that matter, the commands where getting it wrong has real consequences, the LLM never touches them.

## Target Audience

Linguists and NLP researchers who understand constituency trees, dependency relations, and POS tags. You can run commands and follow instructions, but you should not have to debug import errors or port conflicts. PARSELY-DIP tells you what's wrong and how to fix it.

## Status

v0.0.2 — Visual constituency tree display in interactive mode. Expanded documentation with NLP vs RegEx tradeoff analysis, parse tree examples, slot-based matching, and domain-specific skill file architecture. Proprietary license aligned with python-tapestry. GitHub repository live.

v0.0.1 — Core engine built. RegEx pipeline working with time, weather, and scrum card intents. NLP layer ported from Uni with Stanza service (default_accurate with Electra Large transformer, GPU accelerated). Hook integration tested with Claude Code. CLI available via `parsely` command.

## License

[Proprietary](LICENSE) — Source-available, not open source.

**Free for:** personal use, development, testing, research, academic work, non-commercial projects. Study it, fork it, learn from it.

**Requires a commercial license for:** hosted services, revenue-generating products, organizational/business use. Contact george@iseestudios.com.

Patent-protected. See LICENSE for full terms.

## Author

George Butiri — [george@iseestudios.com](mailto:george@iseestudios.com)
