# Owner(s): ["module: inductor"]
import json
import os
import tempfile
from dataclasses import dataclass
from unittest import mock

from torch._inductor.remote_cache import (
    create_cache,
    RemoteCache,
    RemoteCacheBackend,
    RemoteCachePassthroughSerde,
)
from torch.testing._internal.common_utils import TestCase


class FailingBackend(RemoteCacheBackend):
    def _get(self, key):
        raise AssertionError("testget")

    def _put(self, key, data):
        raise AssertionError("testput")


class NoopBackend(RemoteCacheBackend):
    def _get(self, key):
        return None

    def _put(self, key, data):
        return None


@dataclass
class TestSample:
    fail: str = None


class FakeCache(RemoteCache):
    def __init__(self):
        super().__init__(FailingBackend(), RemoteCachePassthroughSerde())

    def _create_sample(self):
        return TestSample()

    def _log_sample(self, sample):
        self.sample = sample


class TestRemoteCache(TestCase):
    def test_normal_logging(
        self,
    ) -> None:
        c = RemoteCache(NoopBackend(), RemoteCachePassthroughSerde())
        c.put("test", "value")
        c.get("test")

    def test_failure_no_sample(
        self,
    ) -> None:
        c = RemoteCache(FailingBackend(), RemoteCachePassthroughSerde())
        with self.assertRaises(AssertionError):
            c.put("test", "value")
        with self.assertRaises(AssertionError):
            c.get("test")

    def test_failure_logging(
        self,
    ) -> None:
        c = FakeCache()
        with self.assertRaises(AssertionError):
            c.put("test", "value")
        self.assertEqual(c.sample.fail_reason, "testput")
        with self.assertRaises(AssertionError):
            c.get("test")
        self.assertEqual(c.sample.fail_reason, "testget")

    def test_create_local_autotune_cache(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            key = os.path.join(tmpdir, "cache", "entry.best_config")
            c = create_cache("local-autotune", local_cache_cls="LocalAutotuneCache")
            if c is None:
                self.fail("Expected local autotune cache")

            expected = {"value": 1}
            c.put(key, expected)

            self.assertTrue(os.path.exists(key))
            self.assertEqual(c.get(key), expected)

    def test_local_autotune_cache_corrupt_json_is_miss(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            key = os.path.join(tmpdir, "cache", "entry.best_config")
            os.makedirs(os.path.dirname(key), exist_ok=True)
            with open(key, "wb") as fd:
                fd.write(b'{"value": 1}{"value": 2}')

            c = create_cache("local-autotune", local_cache_cls="LocalAutotuneCache")
            if c is None:
                self.fail("Expected local autotune cache")

            with self.assertLogs("torch._inductor.remote_cache", level="WARNING") as cm:
                self.assertIsNone(c.get(key))
            self.assertIn("Ignoring corrupt local cache entry", cm.output[0])
            self.assertIn("JSONDecodeError", cm.output[0])
            self.assertIn("Extra data", cm.output[0])
            self.assertNotIn("Traceback", cm.output[0])

    def test_local_autotune_cache_invalid_utf8_is_miss(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            key = os.path.join(tmpdir, "cache", "entry.best_config")
            os.makedirs(os.path.dirname(key), exist_ok=True)
            with open(key, "wb") as fd:
                fd.write(b"\xff")

            c = create_cache("local-autotune", local_cache_cls="LocalAutotuneCache")
            if c is None:
                self.fail("Expected local autotune cache")

            with self.assertLogs("torch._inductor.remote_cache", level="WARNING") as cm:
                self.assertIsNone(c.get(key))
            self.assertIn("Ignoring corrupt local cache entry", cm.output[0])
            self.assertIn("UnicodeDecodeError", cm.output[0])
            self.assertIn("invalid start byte", cm.output[0])
            self.assertNotIn("Traceback", cm.output[0])

    def test_local_autotune_cache_put_uses_atomic_write(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            key = os.path.join(tmpdir, "cache", "entry.best_config")
            c = create_cache("local-autotune", local_cache_cls="LocalAutotuneCache")
            if c is None:
                self.fail("Expected local autotune cache")

            expected = {"value": 1}
            with mock.patch("torch._inductor.codecache.write_atomic") as write_atomic:
                c.put(key, expected)

            write_atomic.assert_called_once()
            args, _kwargs = write_atomic.call_args
            self.assertEqual(args[0], key)
            self.assertEqual(json.loads(args[1]), expected)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
