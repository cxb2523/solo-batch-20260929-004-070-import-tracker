"""
Tests for the thread-safe, idempotent hook installation strategy and the
injectable logging strategy
"""

# Standard
import logging
import sys
import threading

# Third Party
import pytest

# Local
from import_tracker import lazy_import_errors
from import_tracker.import_tracker import (
    HookContext,
    MetaPathHookInstaller,
    get_hook_installer,
    set_hook_installer,
)
from import_tracker.lazy_import_errors import _LazyErrorMetaFinder
from import_tracker.log import get_log, reset_log, set_log
import import_tracker


@pytest.fixture
def clean_meta_path():
    """Snapshot sys.meta_path and restore it after the test"""
    before = list(sys.meta_path)
    yield
    sys.meta_path[:] = before


@pytest.fixture
def restore_log_strategy():
    """Restore the default logging strategy after the test"""
    yield
    reset_log()


def _lazy_finders():
    """Get all lazy error finders currently on sys.meta_path"""
    return [f for f in sys.meta_path if isinstance(f, _LazyErrorMetaFinder)]


## Idempotent installation #####################################################


def test_install_same_context_twice_registers_once(clean_meta_path):
    """Installing the same context repeatedly only takes effect once"""
    ctx = lazy_import_errors()
    try:
        assert len(_lazy_finders()) == 1
        assert ctx.install() is False
        assert ctx.install() is False
        assert len(_lazy_finders()) == 1
    finally:
        ctx.uninstall()
    assert not _lazy_finders()


def test_uninstall_cleans_residual_hooks(clean_meta_path):
    """Uninstalling removes residual hook instances from sys.meta_path"""
    installer = MetaPathHookInstaller()
    ctx = lazy_import_errors(hook_installer=installer)
    # Simulate a leftover duplicate of the same finder
    sys.meta_path.append(ctx.finder)
    assert _lazy_finders().count(ctx.finder) == 2
    ctx.uninstall()
    assert not _lazy_finders()


def test_uninstall_without_install_is_noop(clean_meta_path):
    """Uninstalling a context that is not installed is a safe no-op"""
    installer = MetaPathHookInstaller()
    ctx = lazy_import_errors(hook_installer=installer)
    ctx.uninstall()
    before = list(sys.meta_path)
    ctx.uninstall()
    ctx.uninstall()
    assert list(sys.meta_path) == before


def test_uninstall_reinstall(clean_meta_path):
    """After uninstalling, reinstalling restores full lazy error behavior"""
    ctx = lazy_import_errors()
    ctx.uninstall()
    assert not _lazy_finders()

    # Without the hook, the import fails eagerly
    with pytest.raises(ModuleNotFoundError):
        import reinstall_missing_mod  # noqa: F401

    # Reinstall and check the lazy behavior is back with the full message
    ctx.install()
    try:
        import reinstall_missing_mod

        with pytest.raises(ModuleNotFoundError, match="reinstall_missing_mod"):
            reinstall_missing_mod.foo()
    finally:
        ctx.uninstall()
    assert not _lazy_finders()


def test_shared_context_concurrent_install(clean_meta_path):
    """Concurrent installs of the same context from many threads register the
    hook exactly once
    """
    installer = MetaPathHookInstaller()
    ctx = lazy_import_errors(hook_installer=installer)
    ctx.uninstall()
    assert not _lazy_finders()

    barrier = threading.Barrier(16)
    errors = []

    def worker():
        try:
            barrier.wait(timeout=10)
            ctx.install()
        except Exception as err:  # pragma: no cover
            errors.append(err)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert len(_lazy_finders()) == 1

    # Concurrent uninstalls from many threads leave no residuals
    barrier = threading.Barrier(16)

    def unworker():
        try:
            barrier.wait(timeout=10)
            ctx.uninstall()
        except Exception as err:  # pragma: no cover
            errors.append(err)

    threads = [threading.Thread(target=unworker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert not _lazy_finders()


## Concurrent first imports ####################################################


def test_concurrent_first_import_no_double_registration(clean_meta_path):
    """Many threads importing a missing module for the first time under a
    shared lazy error context must neither double-register nor miss the hook
    """
    ctx = lazy_import_errors()
    barrier = threading.Barrier(8)
    results = []
    errors = []

    def worker():
        try:
            barrier.wait(timeout=10)
            import concurrent_missing_mod

            results.append(concurrent_missing_mod)
        except Exception as err:  # pragma: no cover
            errors.append(err)

    try:
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not errors
        # All threads got the same single lazily-imported module
        assert len(results) == 8
        assert len({id(mod) for mod in results}) == 1
        # The hook is still registered exactly once
        assert len(_lazy_finders()) == 1
        # The lazy error still raises with the full message
        with pytest.raises(ModuleNotFoundError, match="concurrent_missing_mod"):
            results[0].foo()
    finally:
        ctx.uninstall()
    assert not _lazy_finders()


def test_concurrent_nested_contexts(clean_meta_path):
    """Threads entering and exiting their own lazy error contexts concurrently
    never corrupt each other's hooks
    """
    barrier = threading.Barrier(8)
    errors = []

    def worker(idx):
        try:
            barrier.wait(timeout=10)
            with lazy_import_errors():
                __import__(f"nested_missing_mod_{idx}")
        except Exception as err:  # pragma: no cover
            errors.append(err)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert not _lazy_finders()


## Duplicate imports ###########################################################


def test_duplicate_import_same_missing_module(clean_meta_path):
    """Importing the same missing module twice yields one cached lazy module
    and the error message is not degraded
    """
    with lazy_import_errors():
        import dup_missing_mod

        import dup_missing_mod as dup_missing_mod_again

        assert dup_missing_mod is dup_missing_mod_again
        assert sys.modules["dup_missing_mod"] is dup_missing_mod
        with pytest.raises(ModuleNotFoundError) as excinfo:
            dup_missing_mod.foo()
        assert "No module named 'dup_missing_mod'" in str(excinfo.value)


def test_reimport_after_uninstall_raises_eagerly(clean_meta_path):
    """After uninstalling and clearing the cached module, re-importing raises
    the eager error again
    """
    ctx = lazy_import_errors()
    import cached_missing_mod  # noqa: F401

    ctx.uninstall()
    sys.modules.pop("cached_missing_mod", None)
    with pytest.raises(ModuleNotFoundError):
        import cached_missing_mod  # noqa: F401


## Injectable strategies #######################################################


def test_hook_installer_injection(clean_meta_path):
    """A custom hook installer can be injected and the default restored"""
    custom = MetaPathHookInstaller()
    previous = set_hook_installer(custom)
    try:
        assert get_hook_installer() is custom
        ctx = lazy_import_errors()
        assert ctx._installer is custom
        ctx.uninstall()
    finally:
        set_hook_installer(previous)
    assert get_hook_installer() is previous


def test_log_strategy_injection(restore_log_strategy):
    """A custom logging strategy can be injected and is used through the
    shared log handle
    """
    custom = logging.getLogger("IMPRT_TEST_INJECTED")
    set_log(custom)
    assert get_log() is custom
    # The higher-order alog levels are available on the injected strategy
    assert callable(get_log().debug1)
    assert callable(get_log().debug4)
    # The shared proxy handle forwards to the injected strategy
    from import_tracker.log import log

    assert log.name == custom.name
    reset_log()
    assert get_log() is not custom
    assert get_log().name == "IMPRT"


def test_log_strategy_used_by_track_module(restore_log_strategy, caplog):
    """The tracking code logs through the injected strategy"""
    custom = logging.getLogger("IMPRT_TEST_TRACK")
    set_log(custom)
    with caplog.at_level(logging.DEBUG, logger="IMPRT_TEST_TRACK"):
        import_tracker.track_module("sample_lib.submod1")
    assert any(
        "sample_lib" in record.getMessage() and "submod1" in record.getMessage()
        for record in caplog.records
    )
