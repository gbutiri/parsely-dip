"""PARSELY-DIP CLI entry point."""

import sys
from parsely_dip import parse


def start():
    """Start the Stanza NLP service (foreground)."""
    from parsely_dip.engine.stanza_service import app, nlp  # noqa: F401 — triggers model load
    print("Starting PARSELY-DIP Stanza NLP service on port 5013...")
    app.run(debug=True, host='127.0.0.1', port=5013)


def test():
    """Interactive parse tree explorer — calls the running Stanza service via HTTP."""
    import requests

    url = "http://127.0.0.1:5013/debug_parse"
    print("\n=== PARSELY-DIP Structure Explorer ===")
    print("Type a sentence to see its parse structure.")
    print("Requires: parsely start (in another terminal)")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input(">>> ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input:
                continue

            try:
                resp = requests.post(url, json={"user_input": user_input}, timeout=30)
            except requests.ConnectionError:
                print("  Service not running. Start it with: parsely start\n")
                continue

            data = resp.json()
            for result in data.get("results", []):
                print(f"\n--- Constituency Tree (inline) ---")
                print(result["constituency_tree"])

                print(f"\n--- Constituency Tree (visual) ---")
                print(result.get("visual_tree", ""))

                print(f"--- Words (POS + Dependency) ---")
                for word in result["words"]:
                    print(f"  {word['text']:15} POS={word['pos']:6} DEP={word['dep']:10} HEAD={word.get('head', '')}")

                print()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"  Error: {e}")

    print("Goodbye!")


def chat():
    """Interactive mode — runs prompts through the full parse() pipeline."""
    print("\n=== PARSELY-DIP Chat Mode ===")
    print("Type a prompt to run through the pipeline (RegEx -> NLP -> LLM fallback).")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input(">>> ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input:
                continue

            result = parse(user_input)
            if result:
                print(f"  {result}")
            else:
                print("  [No match — would fall through to LLM]")
            print()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"  Error: {e}")

    print("Goodbye!")


def cli():
    if len(sys.argv) < 2:
        print("Usage: parsely \"your prompt here\"")
        print("       parsely start")
        print("       parsely --chat")
        print("       parsely --test")
        sys.exit(1)

    if sys.argv[1] == 'start':
        start()
        return

    if sys.argv[1] in ['--chat', '-c']:
        chat()
        return

    if sys.argv[1] in ['--test', '-t']:
        test()
        return

    prompt = " ".join(sys.argv[1:])
    result = parse(prompt)

    if result:
        print(result)
    else:
        print("No match.")
        sys.exit(1)


if __name__ == "__main__":
    cli()
