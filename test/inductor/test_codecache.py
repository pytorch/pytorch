# Owner(s): ["module: inductor"]
import functools
import pickle
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codecache import (
    AsyncCompile,
    FxGraphCachePickler,
    FxGraphHashDetails,
    TensorMetadata,
    TensorMetadataAndValues,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
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


@instantiate_parametrized_tests
class TestFxGraphCache(TestCase):
    @classmethod
    def setUpClass(cls):
        # Reroute all cache disk activity to a clean temporary directory to
        # ensure isolation (and initial cache misses). Deliberately create the
        # temp dir in setUpClass, however, so that individual test runs reuse
        # the same location. We don't expect different tests to reuse cache
        # entries, so preserving the temp dir provides that additional testing.
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.cache_dir_patch = patch("torch._inductor.codecache.cache_dir")
        cls.cache_dir_patch.start().return_value = cls.tmpdir.name

    @classmethod
    def tearDownClass(cls):
        cls.cache_dir_patch.stop()
        cls.tmpdir.cleanup()

    def setUp(self):
        counters.clear()

    @config.patch({"fx_graph_cache": True})
    @parametrize("device", ("cuda", "cpu"))
    @parametrize("dtype", (torch.float, torch.bfloat16))
    def test_cache_load_function(self, device, dtype):
        """
        Verify that we can populate and load functions from the cache.
        """
        if device == "cuda" and not HAS_CUDA:
            raise unittest.SkipTest("requires CUDA")

        def fn(x, y):
            return (x * 2, y @ y)

        a = torch.rand(25, dtype=dtype, device=device)
        b = torch.rand(5, 5, dtype=dtype, device=device)
        c = a.view(5, 5)

        compiled_fn = torch.compile(fn, dynamic=False)

        # A first call shold miss in the cache.
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        # A second call should hit. (First reset so in-memory guards
        # don't prevent compilation).
        torch._dynamo.reset()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

        # But we expect different code if the tensors are aliased.
        torch._dynamo.reset()
        self.assertEqual(fn(a, c), compiled_fn(a, c))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

    @config.patch({"fx_graph_cache": True})
    @parametrize("device", ("cuda", "cpu"))
    @parametrize("dtype", (torch.float, torch.bfloat16))
    def test_cache_load_model(self, device, dtype):
        """
        Verify that we can populate and load models from the cache.
        """
        if device == "cuda" and not HAS_CUDA:
            raise unittest.SkipTest("requires CUDA")

        model = MyModel().to(device)
        if dtype == torch.float:
            model = model.float()
        elif dtype == torch.bfloat16:
            model = model.bfloat16()

        a = torch.rand(10, 10, dtype=dtype, device=device)

        compiled_model = torch.compile(model, dynamic=False)

        # A first call shold miss in the cache.
        self.assertEqual(model(a), compiled_model(a))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        # A second call should hit. (First reset so in-memory guards
        # don't prevent compilation).
        torch._dynamo.reset()
        self.assertEqual(model(a), compiled_model(a))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)


class TestFxGraphCacheHashing(TestCase):
    def test_tensor_constants(self):
        """
        Test the handling of small vs. large tensor constants.
        """
        data = FxGraphCachePickler.dumps(torch.tensor(list(range(9))))
        self.assertIsInstance(pickle.loads(data), TensorMetadata)

        data = FxGraphCachePickler.dumps(torch.tensor(list(range(8))))
        self.assertIsInstance(pickle.loads(data), TensorMetadataAndValues)

    def test_hash_fake_tensors(self):
        """
        Test hashing (pickling) FakeTensors with various characteristics.
        """
        with torch._subclasses.FakeTensorMode():
            # Verify that FakeTensors get pickled into a TensorMetadata:
            data = FxGraphCachePickler.dumps(torch.randn(1))
            self.assertIsInstance(pickle.loads(data), TensorMetadata)

            # Different shapes:
            self.assertEqual(
                FxGraphCachePickler.dumps(torch.randn(3)),
                FxGraphCachePickler.dumps(torch.randn(3)),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(3)),
                FxGraphCachePickler.dumps(torch.randn(4)),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(3)),
                FxGraphCachePickler.dumps(torch.randn(3, 3)),
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

            # Different strides:
            self.assertEqual(
                FxGraphCachePickler.dumps(torch.randn(3, 3)),
                FxGraphCachePickler.dumps(
                    torch.randn(3, 3).transpose(0, 1).transpose(0, 1)
                ),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(3, 3)),
                FxGraphCachePickler.dumps(torch.randn(3, 3).transpose(0, 1)),
            )

            # Different storage offsets:
            self.assertEqual(
                FxGraphCachePickler.dumps(torch.randn(3)[1:]),
                FxGraphCachePickler.dumps(torch.randn(3)[1:]),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(3)[1:]),
                FxGraphCachePickler.dumps(torch.randn(2)),
            )

            # Different dtypes:
            self.assertEqual(
                FxGraphCachePickler.dumps(torch.randn(3, dtype=torch.float32)),
                FxGraphCachePickler.dumps(torch.randn(3, dtype=torch.float32)),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(3, dtype=torch.float32)),
                FxGraphCachePickler.dumps(torch.randn(3, dtype=torch.float64)),
            )

            # Different 'requires_grad':
            self.assertEqual(
                FxGraphCachePickler.dumps(torch.randn(3, requires_grad=True)),
                FxGraphCachePickler.dumps(torch.randn(3, requires_grad=True)),
            )
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(3, requires_grad=True)),
                FxGraphCachePickler.dumps(torch.randn(3, requires_grad=False)),
            )

            # Different memory formats:
            self.assertNotEqual(
                FxGraphCachePickler.dumps(torch.randn(1, 2, 3, 4)),
                FxGraphCachePickler.dumps(
                    torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last)
                ),
            )

            # Different devices:
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


if __name__ == "__main__":
    run_tests()
