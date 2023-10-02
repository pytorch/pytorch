# Owner(s): ["module: inductor"]
import functools
import pickle
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._inductor import config
from torch._inductor.codecache import (
    AsyncCompile,
    FxGraphCache,
    FxGraphCachePickler,
    FxGraphHashDetails,
    TensorMetadataHolder,
    TypedStorageMetadataHolder,
)
from torch.testing._internal.inductor_utils import HAS_CUDA

requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, "requires cuda")


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 10)

    def forward(self, inp):
        return self.fc1(inp)


def _run_codecache_test(start_method):
    torch._inductor.config.worker_start_method = start_method
    torch._inductor.config.compile_threads = 16
    AsyncCompile.warm_pool()

    model = MyModel().cuda()
    model = torch.compile(model)
    inp = torch.rand(10, 10).cuda()
    model(inp).sum().backward()


@requires_cuda()
def test_codecache_spawn():
    _run_codecache_test("spawn")


@requires_cuda()
def test_codecache_fork():
    _run_codecache_test("fork")


class TestFxGraphCache(TestCase):
    @requires_cuda()
    @config.patch({"fx_graph_cache": True})
    @patch("torch._inductor.codecache.cache_dir")
    def test_cache_load(self, mock_cache_dir):
        """
        Verify that we can populate and load CompiledFxGraphs from the cache.
        """

        def fn(x, y):
            z = x + y
            return (z @ z,)

        inps = [
            torch.rand([5, 5]).cuda(),
            torch.rand([5, 5]).cuda(),
        ]

        # Reroute all disk activity to a clean temporary directory to
        # ensure isolation (and an initial cache miss).
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_cache_dir.return_value = tmpdir

            # Spy on these methods so we can verify they are called.
            mock_save_graph = Mock(wraps=FxGraphCache.save_graph)
            mock_load_graph = Mock(wraps=FxGraphCache.load_graph)
            FxGraphCache.save_graph = mock_save_graph
            FxGraphCache.load_graph = mock_load_graph

            # A first call shold miss in the cache.
            compiled_fn = torch.compile(fn)
            compiled_fn(*inps)
            self.assertEqual(fn(*inps), compiled_fn(*inps))

            mock_save_graph.assert_called_once()
            mock_load_graph.assert_not_called()

            # A second call should hit. (First reset so in-memory guards
            # don't prevent compilation).
            torch._dynamo.reset()

            compiled_fn = torch.compile(fn)
            compiled_fn(*inps)
            self.assertEqual(fn(*inps), compiled_fn(*inps))

            mock_save_graph.assert_called_once()
            mock_load_graph.assert_called_once()

    def test_hash_tensors(self):
        """
        Test hashing (pickling) FakeTensors with various characteristics.
        """
        with torch._subclasses.FakeTensorMode():
            # Verify that FakeTensors get pickled into a TensorMetadataHolder:
            data = FxGraphCachePickler.dumps(torch.randn(1))
            self.assertIsInstance(pickle.loads(data), TensorMetadataHolder)

            # Compare hashing of tensors with various characteristics:
            self.assertEqual(
                FxGraphCachePickler.dumps(torch.randn(3)),
                FxGraphCachePickler.dumps(torch.randn(3)),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(3)),
                FxGraphCachePickler.dumps(torch.randn(4)),
            )

            self.assertEqual(
                FxGraphCachePickler.dumps(torch.randn(3, 3)),
                FxGraphCachePickler.dumps(torch.randn(3, 3)),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(3, 3)),
                FxGraphCachePickler.dumps(torch.randn(3, 4)),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(3, 3)),
                FxGraphCachePickler.dumps(torch.randn(4, 3)),
            )

            self.assertEqual(
                FxGraphCachePickler.dumps(torch.randn(3, dtype=torch.float32)),
                FxGraphCachePickler.dumps(torch.randn(3, dtype=torch.float32)),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(3, dtype=torch.float32)),
                FxGraphCachePickler.dumps(torch.randn(3, dtype=torch.float64)),
            )

            self.assertEqual(
                FxGraphCachePickler.dumps(torch.randn(3, requires_grad=True)),
                FxGraphCachePickler.dumps(torch.randn(3, requires_grad=True)),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(3, requires_grad=True)),
                FxGraphCachePickler.dumps(torch.randn(3, requires_grad=False)),
            )

            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(1, 2, 3, 4)),
                FxGraphCachePickler.dumps(
                    torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last)
                ),
            )

            self.assertEqual(
                FxGraphCachePickler.dumps(torch.randn(3, device="meta")),
                FxGraphCachePickler.dumps(torch.randn(3, device="meta")),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(3, device="meta")),
                FxGraphCachePickler.dumps(torch.randn(3, device="cpu")),
            )

            if HAS_CUDA and torch.cuda.device_count() >= 2:
                self.assertEqual(
                    FxGraphCachePickler.dumps(torch.randn(3, device="cuda:1")),
                    FxGraphCachePickler.dumps(torch.randn(3, device="cuda:1")),
                )
                self.assertNotEqual(
                    FxGraphCachePickler.dumps(torch.randn(3, device="cuda:0")),
                    FxGraphCachePickler.dumps(torch.randn(3, device="cuda:1")),
                )

    def test_hash_kwargs(self):
        """
        Test the special handling of the kwargs when hashing, i.e.,
        ordering of the kwargs dict and any set arguments.
        """
        # Dict order of the kwargs should not affect hashes.
        details1 = FxGraphHashDetails([], {"a": 0, "z": 1})
        details2 = FxGraphHashDetails([], {"z": 1, "a": 0})
        self.assertEqual(
            FxGraphCachePickler.dumps(details1),
            FxGraphCachePickler.dumps(details2),
        )

        # Different kwarg values should affect hashes.
        details1 = FxGraphHashDetails([], {"a": 0})
        details2 = FxGraphHashDetails([], {"a": 1})
        self.assertNotEqual(
            FxGraphCachePickler.dumps(details1),
            FxGraphCachePickler.dumps(details2),
        )

        # Set order should not affect hashes. Sets are unordered, but
        # sorting and creating a new set seems to change the order.
        set1 = {"a", "b", "c", "d", "e", "f", "g"}
        set2 = set(sorted(set1))  # noqa: C414
        details1 = FxGraphHashDetails([], {"a": set1})
        details2 = FxGraphHashDetails([], {"a": set2})
        self.assertEqual(
            FxGraphCachePickler.dumps(details1),
            FxGraphCachePickler.dumps(details2),
        )

        # But different set contents should affect hashes.
        details1 = FxGraphHashDetails([], {"a": {1, 2, 3}})
        details2 = FxGraphHashDetails([], {"a": {1, 2}})
        self.assertNotEqual(
            FxGraphCachePickler.dumps(details1),
            FxGraphCachePickler.dumps(details2),
        )

    def test_hash_config_changes(self):
        """
        Test that different config settings affect hashes.
        """
        with config.patch({"max_autotune": False}):
            details1 = FxGraphHashDetails([], {})
            details2 = FxGraphHashDetails([], {})

        with config.patch({"max_autotune": True}):
            details3 = FxGraphHashDetails([], {})

        self.assertEqual(
            FxGraphCachePickler.dumps(details1),
            FxGraphCachePickler.dumps(details2),
        )
        self.assertNotEqual(
            FxGraphCachePickler.dumps(details1),
            FxGraphCachePickler.dumps(details3),
        )

    def test_hash_typed_storage(self):
        """
        Test hashing (pickling) TypedStorage objects.
        """
        # Verify that TypedStorage objects get pickled into TypedStorageMetadataHolders:
        data = FxGraphCachePickler.dumps(torch.TypedStorage([0]))
        self.assertIsInstance(pickle.loads(data), TypedStorageMetadataHolder)

        self.assertEqual(
            FxGraphCachePickler.dumps(torch.TypedStorage([0])),
            FxGraphCachePickler.dumps(torch.TypedStorage([0])),
        )
        self.assertNotEqual(
            FxGraphCachePickler.dumps(torch.TypedStorage([0])),
            FxGraphCachePickler.dumps(torch.TypedStorage([1])),
        )


if __name__ == "__main__":
    run_tests()
