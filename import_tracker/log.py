"""
This module holds the shared logging strategy used by the other modules in
this package. The strategy is injectable: ``set_log`` installs a custom
``logging.Logger`` (or any compatible object) and ``get_log`` resolves the
currently active strategy. The module-level ``log`` handle is a lightweight
proxy which always delegates to the currently injected strategy, so existing
callers that do ``from import_tracker.log import log`` keep working unchanged
and automatically follow whatever strategy has been injected.

Entrypoints (``__main__.py`` and callers of ``setup_tools``) assemble the
strategy via ``configure_logging`` / ``set_log``.

The log handle has additional higher-order logging functions defined that align
with the levels for alog (https://github.com/IBM/alchemy-logging).
"""

# Standard
from typing import Optional
import logging
import threading

## Public ######################################################################

# The name of the default shared logger
DEFAULT_LOG_NAME = "IMPRT"

# Add higher-order levels as constants to logging
setattr(logging, "DEBUG1", logging.DEBUG - 1)
setattr(logging, "DEBUG2", logging.DEBUG - 2)
setattr(logging, "DEBUG3", logging.DEBUG - 3)
setattr(logging, "DEBUG4", logging.DEBUG - 4)


def get_log() -> logging.Logger:
    """Resolve the currently injected logging strategy"""
    return _current_log


def set_log(logger: Optional[logging.Logger] = None) -> logging.Logger:
    """Inject a custom logging strategy. Passing None resets to the default
    shared logger.

    Args:
        logger:  Optional[logging.Logger]
            The logger (or compatible object) to use as the logging strategy.
            If None, the default shared logger is restored.

    Returns:
        logger:  logging.Logger
            The now-active logging strategy
    """
    global _current_log
    with _lock:
        _current_log = (
            _add_higher_order_levels(logger) if logger is not None else _default_log
        )
        return _current_log


def reset_log():
    """Reset the logging strategy to the shared default logger"""
    set_log(None)


def configure_logging(
    level=None,
    logger: Optional[logging.Logger] = None,
) -> logging.Logger:
    """Assembly helper used by entrypoints to configure the logging backend
    and (optionally) inject a custom logging strategy in one call.

    Args:
        level:  Optional[Union[int, str]]
            If given, passed through to logging.basicConfig
        logger:  Optional[logging.Logger]
            If given, injected as the active logging strategy

    Returns:
        logger:  logging.Logger
            The now-active logging strategy
    """
    if logger is not None:
        set_log(logger)
    if level is not None:
        logging.basicConfig(level=level)
    return get_log()


## Implementation Details ######################################################

_lock = threading.RLock()


def _add_higher_order_levels(logger: logging.Logger) -> logging.Logger:
    """Attach the alog-style higher-order debug levels to the given logger"""
    for level_name in ["DEBUG1", "DEBUG2", "DEBUG3", "DEBUG4"]:
        level_value = getattr(logging, level_name)
        setattr(
            logger,
            level_name.lower(),
            lambda *args, _level=level_value, **kwargs: logger.log(
                _level, *args, **kwargs
            ),
        )
    return logger


def _make_default_logger() -> logging.Logger:
    """Create the default shared logger with the higher-order levels"""
    return _add_higher_order_levels(logging.getLogger(DEFAULT_LOG_NAME))


class _LogStrategyProxy:
    """Attribute proxy that always resolves against the currently injected
    logging strategy. This keeps ``from import_tracker.log import log``
    callers working while allowing the strategy to be swapped at runtime.
    """

    def __getattr__(self, name: str):
        return getattr(get_log(), name)

    def __repr__(self) -> str:
        return repr(get_log())


# The default shared logger and the currently active strategy
_default_log = _make_default_logger()
_current_log = _default_log

# The shared, injectable log handle. Existing callers import this name.
log = _LogStrategyProxy()
