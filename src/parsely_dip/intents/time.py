"""Intent: Tell Time

Returns current time in natural language.
"""

from datetime import datetime
from parsely_dip.engine.registry import intent


def format_natural_time():
    """Format current time in military (24-hour) format."""
    now = datetime.now()
    return f"{now.hour:02d}:{now.minute:02d}"


@intent('tell_time')
def tell_time():
    """Returns current time."""
    return format_natural_time()
