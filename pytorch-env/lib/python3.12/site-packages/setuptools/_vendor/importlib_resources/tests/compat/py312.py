import contextlib

from .py39 import import_helper


@contextlib.contextmanager
def isolated_modules():
    """
    Save modules on entry and cleanup on exit.
    """
    (saved,) = import_helper.modules_setup()
    try:
        yield
    finally:
        import_helper.modules_cleanup(saved)


vars(import_helper).setdefault('isolated_modules', isolated_modules)
