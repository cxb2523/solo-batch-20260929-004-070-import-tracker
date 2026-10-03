"""
Tests for the explicit, thread-safe hook installation strategy
(MetaPathHookRegistry) and the injectable logging strategy.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
import logging
import sys
import threading

# Third Party
import pytest

# Local
from import_tracker.import_tracker import (
    MetaPathHookRegistry,
    get_hook_registry,
    reset_hook_registry,
    set_hook_registry,
)
from import_tracker.lazy_import_errors import _LazyErrorMetaFinder
from import_tracker.log import configure_logging, get_log, log, reset_log, set_log
import import_tracker

## Helpers #####################################################################


def _lazy_finders():
    """Snapshot the lazy error finders currently in sys.meta_path"""
    return [f for f in sys.meta_path if isinstance(f, _LazyErrorMetaFinder)]


@pytest.fixture(autouse=True)
def clean_hooks():
    """Ensure no residual hooks or injected strategies leak between tests"""
    yield
    sys.meta_path[:] = [
        f for f in sys.meta_path if not isinstance(f, _LazyErrorMetaFinder)
    ]
    get_hook_registry().clear()
    reset_hook_registry()
    reset_log()


## Concurrent Installation Tests ###############################################


def test_concurrent_context_install_single_hook():
    """Multiple threads entering lazy_import_errors simultaneously must result
    in exactly one registered finder, and a clean meta_path after all exit.
    """
    num_threads = 8
    start_barrier = threading.Barrier(num_threads)
    entered_barrier = threading.Barrier(num_threads)

    def worker():
        start_barrier.wait()
        with import_tracker.lazy_import_errors():
            # Wait until every thread is inside its context, then check that
            # only a single finder was registered for all of them
            entered_barrier.wait()
            assert len(_lazy_finders()) == 1

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(worker) for _ in range(num_threads)]
        for future in futures:
            future.result()

    assert not _lazy_finders()


def test_concurrent_direct_install_idempotent():
    """Multiple threads concurrently calling lazy_import_errors as a function
    (no context manager) must register exactly one finder.
    """
    num_threads = 8
    start_barrier = threading.Barrier(num_threads)

    def worker():
        start_barrier.wait()
        import_tracker.lazy_import_errors()

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(worker) for _ in range(num_threads)]
        for future in futures:
            future.result()

    assert len(_lazy_finders()) == 1


def test_concurrent_first_import_no_duplicate_registration():
    """Multiple threads performing the first import of a missing module at the
    same time must neither double-register the finder nor miss registration,
    and all threads must get the same lazy module which raises on use.
    """
    num_threads = 8
    start_barrier = threading.Barrier(num_threads)

    with import_tracker.lazy_import_errors():

        def worker():
            start_barrier.wait()
            # Third Party (missing)
            # Third Party
            import concurrent_first_missing_mod

            return concurrent_first_missing_mod

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            modules = list(pool.map(lambda _: worker(), range(num_threads)))

        # No duplicate registration
        assert len(_lazy_finders()) == 1

        # All threads resolved the same module object
        assert all(mod is modules[0] for mod in modules)

    # The lazy error still fires with the full message after the context exits
    for mod in modules:
        with pytest.raises(
            ModuleNotFoundError,
            match="No module named 'concurrent_first_missing_mod'",
        ):
            mod.anything()


## Uninstall / Reinstall Tests #################################################


def test_uninstall_then_reinstall():
    """After a context exits, no residual hooks may remain, and a subsequent
    context must install a working finder again.
    """
    with import_tracker.lazy_import_errors():
        assert len(_lazy_finders()) == 1
    assert not _lazy_finders()

    with import_tracker.lazy_import_errors():
        assert len(_lazy_finders()) == 1
        # Third Party (missing)
        # Third Party
        import reinstall_missing_mod
    assert not _lazy_finders()

    with pytest.raises(
        ModuleNotFoundError, match="No module named 'reinstall_missing_mod'"
    ):
        reinstall_missing_mod.foo()


def test_residual_hook_instances_purged_on_uninstall():
    """If duplicate instances of the finder somehow end up in sys.meta_path,
    uninstalling must purge all of them.
    """
    ctx = import_tracker.lazy_import_errors()
    finder = ctx.finder
    assert finder is not None

    # Artificially inject residual duplicates
    sys.meta_path.append(finder)
    sys.meta_path.insert(0, finder)
    assert sys.meta_path.count(finder) == 3

    ctx.__exit__()
    assert finder not in sys.meta_path
    assert not _lazy_finders()

    # Uninstalling again is a safe no-op
    ctx.__exit__()
    assert not _lazy_finders()


def test_nested_contexts_share_single_hook():
    """Nested contexts with the same configuration share one finder and the
    finder survives until the outermost context exits.
    """
    with import_tracker.lazy_import_errors():
        with import_tracker.lazy_import_errors():
            assert len(_lazy_finders()) == 1
        assert len(_lazy_finders()) == 1
        # Third Party (missing)
        # Third Party
        import nested_shared_missing_mod
    assert not _lazy_finders()

    with pytest.raises(ModuleNotFoundError):
        nested_shared_missing_mod.foo()


def test_repeated_direct_calls_single_finder():
    """Repeated direct (non-context) calls are idempotent"""
    import_tracker.lazy_import_errors()
    import_tracker.lazy_import_errors()
    import_tracker.lazy_import_errors()
    assert len(_lazy_finders()) == 1


## Repeated Import Tests #######################################################


def test_repeated_import_same_missing_module():
    """Re-importing the same missing module (after eviction from sys.modules)
    must yield fresh lazy modules that both raise the full error message.
    """
    with import_tracker.lazy_import_errors():
        # Third Party (missing)
        # Third Party
        import repeat_missing_mod

        first = repeat_missing_mod
        sys.modules.pop("repeat_missing_mod")

        # Third Party (missing)
        # Third Party
        import repeat_missing_mod as second

    assert first is not second
    for mod in (first, second):
        with pytest.raises(
            ModuleNotFoundError, match="No module named 'repeat_missing_mod'"
        ):
            mod.trigger()


def test_repeated_import_error_message_not_degraded():
    """The lazy error message must keep the full module name across repeated
    installs and imports.
    """
    for _ in range(3):
        with import_tracker.lazy_import_errors():
            # Third Party (missing)
            # Third Party
            import repeat_msg_missing_mod

            sys.modules.pop("repeat_msg_missing_mod")
        with pytest.raises(
            ModuleNotFoundError, match="No module named 'repeat_msg_missing_mod'"
        ):
            repeat_msg_missing_mod.use()


## Registry Unit Tests #########################################################


class _DummyFinder:
    """Minimal stand-in meta-path hook"""

    def find_spec(self, *_, **__):
        return None


def test_registry_refcounting_and_idempotency():
    """The registry installs a hook once per key, refcounts repeated installs,
    and purges the hook only when the last reference is released.
    """
    registry = MetaPathHookRegistry()
    hook_a = registry.install("key", _DummyFinder)
    hook_b = registry.install("key", _DummyFinder)
    assert hook_a is hook_b
    assert sys.meta_path.count(hook_a) == 1
    assert registry.is_installed("key")

    # First release: still held by the second reference
    assert registry.uninstall("key")
    assert hook_a in sys.meta_path

    # Final release: purged
    assert registry.uninstall("key")
    assert hook_a not in sys.meta_path
    assert not registry.is_installed("key")

    # Further uninstalls are safe no-ops
    assert not registry.uninstall("key")


def test_registry_reinstall_after_external_removal():
    """If a hook is removed from sys.meta_path externally, the next install
    registers a fresh hook exactly once.
    """
    registry = MetaPathHookRegistry()
    hook_a = registry.install("key", _DummyFinder)
    sys.meta_path.remove(hook_a)

    hook_b = registry.install("key", _DummyFinder)
    assert hook_b is not hook_a
    assert sys.meta_path.count(hook_b) == 1

    # The stale holder cannot unregister the fresh hook
    assert not registry.uninstall("key", hook_a)
    assert hook_b in sys.meta_path
    registry.uninstall("key", hook_b)
    assert hook_b not in sys.meta_path


def test_registry_thread_safety():
    """Concurrent install/uninstall cycles on one registry leave meta_path
    clean and consistent.
    """
    registry = MetaPathHookRegistry()
    num_threads = 8
    start_barrier = threading.Barrier(num_threads)

    def worker():
        start_barrier.wait()
        hook = registry.install("key", _DummyFinder)
        assert sys.meta_path.count(hook) >= 1
        registry.uninstall("key", hook)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(worker) for _ in range(num_threads)]
        for future in futures:
            future.result()

    assert not any(isinstance(f, _DummyFinder) for f in sys.meta_path)


def test_explicit_hook_registry_injection():
    """lazy_import_errors accepts an explicitly injected registry"""
    registry = MetaPathHookRegistry()
    with import_tracker.lazy_import_errors(hook_registry=registry):
        assert len(_lazy_finders()) == 1
        # Third Party (missing)
        # Third Party
        import explicit_registry_missing_mod
    assert not _lazy_finders()

    with pytest.raises(ModuleNotFoundError):
        explicit_registry_missing_mod.foo()


def test_global_hook_registry_injection():
    """The shared hook installation strategy can be swapped out"""
    custom_registry = MetaPathHookRegistry()
    set_hook_registry(custom_registry)
    assert get_hook_registry() is custom_registry
    with import_tracker.lazy_import_errors():
        assert len(_lazy_finders()) == 1
    assert not _lazy_finders()
    reset_hook_registry()
    assert get_hook_registry() is not custom_registry


## Logging Strategy Tests ######################################################


def test_logging_strategy_injection():
    """A custom logging strategy can be injected and the shared handle follows"""
    custom_logger = logging.getLogger("IMPRT_TEST_CUSTOM")
    set_log(custom_logger)
    try:
        assert get_log() is custom_logger
        # The module-level proxy delegates to the injected strategy
        assert log.name == custom_logger.name
        # Higher-order alog-style levels are available on the custom logger
        for level_name in ["debug1", "debug2", "debug3", "debug4"]:
            assert hasattr(log, level_name)
    finally:
        reset_log()
    assert get_log() is logging.getLogger("IMPRT")
    assert log.name == "IMPRT"


def test_logging_strategy_reset_default():
    """Passing None to set_log restores the default strategy"""
    set_log(logging.getLogger("IMPRT_TEST_OTHER"))
    assert log.name == "IMPRT_TEST_OTHER"
    set_log(None)
    assert log.name == "IMPRT"


def test_configure_logging_injects_logger():
    """configure_logging assembles a custom strategy in one call"""
    custom_logger = logging.getLogger("IMPRT_TEST_CONFIGURE")
    configure_logging(logger=custom_logger)
    try:
        assert get_log() is custom_logger
        assert repr(log) == repr(custom_logger)
    finally:
        reset_log()
    assert log.name == "IMPRT"


def test_unhashable_error_message_factory():
    """An unhashable make_error_message callable still installs exactly one
    finder and produces the custom message.
    """

    class UnhashableMessage:
        __hash__ = None

        def __call__(self, name):
            return f"custom message for {name}"

    with import_tracker.lazy_import_errors(make_error_message=UnhashableMessage()):
        assert len(_lazy_finders()) == 1
        # Third Party (missing)
        # Third Party
        import unhashable_msg_missing_mod
    assert not _lazy_finders()

    with pytest.raises(
        ModuleNotFoundError, match="custom message for unhashable_msg_missing_mod"
    ):
        unhashable_msg_missing_mod.foo()
