from dataclasses import dataclass
from unittest import TestCase

from torch._inductor.remote_cache import (
        RemoteCachePassthroughSerde,
        RemoteCacheBackend,
        RemoteCache
)

has_aiplatform = True


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
