"""Repeatable video focus stacking."""
__all__ = ["Config", "run_stack"]


def __getattr__(name):
    # Keep maintenance commands independent of OpenCV/pipeline startup and avoid
    # preloading storage when it is invoked with python -m macrostitch.storage.
    if name in __all__:
        from .pipeline import Config, run_stack
        globals().update(Config=Config, run_stack=run_stack)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
