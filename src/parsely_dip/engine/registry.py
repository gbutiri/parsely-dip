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


def dispatch(intent_name, context=None):
    """Dispatch to registered handler with optional context data.

    Args:
        intent_name: registered intent name
        context: dict from regex match or NLP parse (passed to handler if it accepts it)

    Returns:
        response string or None
    """
    handler = INTENT_REGISTRY.get(intent_name)
    if handler:
        import inspect
        sig = inspect.signature(handler)
        if sig.parameters:
            return handler(context)
        return handler()
    return None


def get_registry():
    """Return all registered intents."""
    return INTENT_REGISTRY
