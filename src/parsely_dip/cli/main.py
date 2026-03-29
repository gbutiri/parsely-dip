"""PARSELY-DIP CLI entry point."""

import sys
from parsely_dip import parse


def cli():
    if len(sys.argv) < 2:
        print("Usage: parsely \"your prompt here\"")
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])
    result = parse(prompt)

    if result:
        print(result)
    else:
        print("No match.")
        sys.exit(1)


if __name__ == "__main__":
    cli()
