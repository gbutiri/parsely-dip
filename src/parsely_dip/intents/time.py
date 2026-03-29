"""Intent: Tell Time

Returns current time in natural language.
"""

from datetime import datetime
from parsely_dip.engine.registry import intent, INTENT_REGISTRY


def format_natural_time():
    """Format current time in military (24-hour) format."""
    now = datetime.now()
    return f"{now.hour:02d}:{now.minute:02d}"


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
