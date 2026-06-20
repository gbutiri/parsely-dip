"""Intent: Tell Time

Returns current time in natural language.
"""

import random
from datetime import datetime
from parsely_dip.engine.registry import intent, INTENT_REGISTRY


# Phrasings the answer is randomly drawn from. {t} = the 12-hour time (e.g. "8:49 PM").
TIME_VARIATIONS = [
    "It's {t}.",
    "It's now {t}.",
    "The time is {t}.",
    "Right now it's {t}.",
    "It's currently {t}.",
    "{t} right now.",
]


def _human_time(now=None):
    """12-hour clock with AM/PM, cross-platform (no %-I)."""
    now = now or datetime.now()
    hour12 = now.hour % 12 or 12
    ampm = "AM" if now.hour < 12 else "PM"
    return f"{hour12}:{now.minute:02d} {ampm}"


def format_natural_time():
    """Current time in a randomly-chosen human-readable phrasing."""
    return random.choice(TIME_VARIATIONS).format(t=_human_time())


@intent('tell_time')
def tell_time():
    """Returns current time."""
    return format_natural_time()


@intent('check_ability_time')
def check_ability_time():
    """Responds to 'can you tell me the time?' — an ability check, not a time request."""
    handler = INTENT_REGISTRY.get('tell_time')
    if handler:
        try:
            result = handler()
            if result:
                return "Yes, just ask 'what time is it?' or 'what's the time?'"
        except Exception:
            pass
    return "Sorry, I can't do that right now."
