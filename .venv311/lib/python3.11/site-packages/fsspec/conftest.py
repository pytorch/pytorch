import os
import shutil
import subprocess
import sys
import time
from collections import deque
from collections.abc import Generator, Sequence

import pytest

import fsspec


@pytest.fixture()
def m():
    """
    Fixture providing a memory filesystem.
    """
    m = fsspec.filesystem("memory")
    m.store.clear()
    m.pseudo_dirs.clear()
    m.pseudo_dirs.append("")
    try:
        yield m
    finally:
        m.store.clear()
        m.pseudo_dirs.clear()
        m.pseudo_dirs.append("")


class InstanceCacheInspector:
    """
    Helper class to inspect instance caches of filesystem classes in tests.
    """

    def clear(self) -> None:
        """
        Clear instance caches of all currently imported filesystem classes.
        """
        classes = deque([fsspec.spec.AbstractFileSystem])
        while classes:
            cls = classes.popleft()
            cls.clear_instance_cache()
            classes.extend(cls.__subclasses__())

    def gather_counts(self, *, omit_zero: bool = True) -> dict[str, int]:
        """
        Gather counts of filesystem instances in the instance caches
        of all currently imported filesystem classes.

        Parameters
        ----------
        omit_zero:
            Whether to omit instance types with no cached instances.
        """
        out: dict[str, int] = {}
        classes = deque([fsspec.spec.AbstractFileSystem])
        while classes:
            cls = classes.popleft()
            count = len(cls._cache)  # there is no public interface for the cache
            # note: skip intermediate AbstractFileSystem subclasses
            #   if they proxy the protocol attribute via a property.
            if isinstance(cls.protocol, (Sequence, str)):
                key = cls.protocol if isinstance(cls.protocol, str) else cls.protocol[0]
                if count or not omit_zero:
                    out[key] = count
            classes.extend(cls.__subclasses__())
        return out


@pytest.fixture(scope="function", autouse=True)
def instance_caches() -> Generator[InstanceCacheInspector, None, None]:
    """
    Fixture to ensure empty filesystem instance caches before and after a test.

    Used by default for all tests.
    Clears caches of all imported filesystem classes.
    Can be used to write test assertions about instance caches.

    Usage:

        def test_something(instance_caches):
            # Test code here
            fsspec.open("file://abc")
            fsspec.open("memory://foo/bar")

            # Test assertion
            assert instance_caches.gather_counts() == {"file": 1, "memory": 1}

    Returns
    -------
    instance_caches: An instance cache inspector for clearing and inspecting caches.
    """
    ic = InstanceCacheInspector()

    ic.clear()
    try:
        yield ic
    finally:
        ic.clear()


@pytest.fixture(scope="function")
def ftp_writable(tmpdir):
    """
    Fixture providing a writable FTP filesystem.
    """
    pytest.importorskip("pyftpdlib")

    d = str(tmpdir)
    with open(os.path.join(d, "out"), "wb") as f:
        f.write(b"hello" * 10000)
    P = subprocess.Popen(
        [sys.executable, "-m", "pyftpdlib", "-d", d, "-u", "user", "-P", "pass", "-w"]
    )
    try:
        time.sleep(1)
        yield "localhost", 2121, "user", "pass"
    finally:
        P.terminate()
        P.wait()
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
