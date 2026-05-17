"""Intent: Scrum Card Operations (demo)

Demo intent showing how to register a parsely-dip intent and return a
formatted string. Returns hard-coded mock output so the example works without
any database or environment setup.

Replace this module in your own project to query your real data — this file
is just the registration pattern.
"""

from parsely_dip.engine.registry import intent


MOCK_CARD_OUTPUT = """#42 [3pts] Example: Demo card title
  Board: Demo Board
  This is a mocked response from the parsely-dip scrum demo intent.
  Replace this module to query your own card data.
  Tasks: 2/3
    [x] First demo task
    [x] Second demo task
    [ ] Third demo task"""


@intent('show_current_card')
def show_current_card():
    """Show a (mocked) current card. Demo only — see module docstring."""
    return MOCK_CARD_OUTPUT


@intent('read_current_card')
def read_current_card():
    """Same data as show_current_card; kept for symmetry."""
    return show_current_card()
