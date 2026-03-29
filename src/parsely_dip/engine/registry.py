"""Intent registry — maps intent names to handler functions.

Handlers self-register via the @intent decorator.
"""

INTENT_REGISTRY = {}


def intent(intent_name):
    """Decorator to register an intent handler.

    Usage:
        @intent('tell_time')
        def tell_time():
            return "It's 2:30 PM"
    """
    def decorator(func):
        INTENT_REGISTRY[intent_name] = func
        func._intent_name = intent_name
        return func
    return decorator


def dispatch(intent_name):
    """Dispatch to registered handler. Returns response string or None."""
    handler = INTENT_REGISTRY.get(intent_name)
    if handler:
        return handler()
    return None


def get_registry():
    """Return all registered intents."""
    return INTENT_REGISTRY
