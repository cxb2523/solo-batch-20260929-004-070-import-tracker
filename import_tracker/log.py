"""
This module holds the shared logging strategy used by the other modules. The
strategy is injectable: callers may replace the active logger with
:func:`set_log` and restore the default with :func:`reset_log`. The
module-level ``log`` handle is a stable proxy that always forwards to the
currently active strategy, so existing ``from import_tracker.log import log``
callers keep working unchanged.

The log handles have additional higher-order logging functions defined that
align with the levels for alog (https://github.com/IBM/alchemy-logging).
"""

# Standard
from typing import Optional, Union
import logging
import threading

# Add higher-order levels as constants to logging
setattr(logging, "DEBUG1", logging.DEBUG - 1)
setattr(logging, "DEBUG2", logging.DEBUG - 2)
setattr(logging, "DEBUG3", logging.DEBUG - 3)
setattr(logging, "DEBUG4", logging.DEBUG - 4)


def _add_higher_order_levels(logger: logging.Logger) -> logging.Logger:
    """Attach the alog-style higher-order debug functions to a logger"""
    for level in range(1, 5):
        level_name = f"debug{level}"
        level_value = getattr(logging, level_name.upper())
        if not hasattr(logger, level_name):
            setattr(
                logger,
                level_name,
                (lambda lvl: (lambda *args, **kwargs: logger.log(lvl, *args, **kwargs)))(
                    level_value
                ),
            )
    return logger


def _make_default_log() -> logging.Logger:
    """Build the default shared logger"""
    return _add_higher_order_levels(logging.getLogger("IMPRT"))


# The default logging strategy and a lock guarding strategy swaps
_default_log = _make_default_log()
_strategy_lock = threading.Lock()
_active_log = _default_log


def get_log() -> logging.Logger:
    """Get the currently active logging strategy"""
    return _active_log


def set_log(logger: Optional[logging.Logger]) -> logging.Logger:
    """Inject a custom logging strategy. Passing None restores the default.

    Returns the previously active strategy so callers can restore it.
    """
    global _active_log
    with _strategy_lock:
        previous = _active_log
        _active_log = _default_log if logger is None else _add_higher_order_levels(logger)
    return previous


def reset_log() -> None:
    """Restore the default logging strategy"""
    set_log(None)


def configure_logging(level: Union[int, str, None] = None) -> logging.Logger:
    """Assemble the logging strategy for standalone use.

    This wires up the root logging config and applies the given level to the
    active strategy. It is used by the ``__main__`` entrypoint and may be used
    by library assembly code (e.g. setup_tools) as well.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), None) or int(level)
    logging.basicConfig(level=level)
    if level is not None:
        _active_log.setLevel(level)
    return _active_log


class _LogProxy:
    """Stable handle that forwards all attribute access to the active logging
    strategy so that injected strategies take effect for every caller, including
    those that imported ``log`` before the injection.
    """

    def __getattr__(self, name: str):
        return getattr(get_log(), name)

    def __setattr__(self, name: str, value) -> None:
        setattr(get_log(), name, value)


# Shared, stable log handle used across the package
log = _LogProxy()
