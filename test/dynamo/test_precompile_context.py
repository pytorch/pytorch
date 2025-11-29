# Owner(s): ["module: dynamo"]
import sys
import unittest

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._functorch
import torch.distributed as dist
import torch.nn as nn
from torch._dynamo.precompile_context import BackendCacheArtifact, PrecompileContext
from torch._functorch import config as functorch_config
from torch._functorch._aot_autograd.autograd_cache import (
    BundledAOTAutogradCacheArtifact,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_triton


@functorch_config.patch({"enable_autograd_cache": True})
@torch._dynamo.config.patch(
    {"caching_precompile": True}
)  # Requires bundledaotautograd cache for now
class PrecompileContextTests(InductorTestCase):
    def setUp(self):
        """
        Reset all counters and caches before each unit test
        """
        super().setUp()
        # Clear PrecompileContext cache artifacts
        PrecompileContext.clear()

    @requires_triton()
    def test_basic(self):
        """
        Test that after torch.compile, PrecompileContext._new_cache_artifacts length is 1
        """

        def simple_function(x):
            return x.sin() + x.cos()

        compiled_fn = torch.compile(simple_function)

        # Run the compiled function
        x = torch.randn(10, device=GPU_TYPE, requires_grad=True)
        result = compiled_fn(x)
        result.sum().backward()
        self.assertEqual(len(PrecompileContext._dynamo_cache_entries), 1)
        self.assertEqual(len(PrecompileContext._backend_artifacts_by_key), 1)
        cache_entries, _ = PrecompileContext.create_cache_entries()
        self.assertEqual(len(cache_entries), 1)

    @requires_triton()
    def test_serialize_by_key(self):
        def simple_function(x):
            return x.sin() + x.cos()

        compiled_fn = torch.compile(simple_function)

        # Run the compiled function
        x = torch.randn(10, device=GPU_TYPE, requires_grad=True)
        result = compiled_fn(x)
        result.sum().backward()
        self.assertEqual(len(PrecompileContext._dynamo_cache_entries), 1)
        self.assertEqual(len(PrecompileContext._backend_artifacts_by_key), 1)
        for key in PrecompileContext._backend_artifacts_by_key.keys():
            result = PrecompileContext.serialize_artifact_by_key(key)
            assert isinstance(result, BackendCacheArtifact)
            self.assertEqual(result.key, key)

        # This should still work
        result, _ = PrecompileContext.create_cache_entries()
        assert len(result) == 1

    @requires_triton()
    def test_editable(self):
        """
        Test that after torch.compile, PrecompileContext._new_cache_artifacts length is 1
        """

        def simple_function(x):
            return x.sin() + x.cos()

        compiled_fn = torch.compile(simple_function)

        # Run the compiled function
        x = torch.randn(10, device=GPU_TYPE, requires_grad=True)
        result = compiled_fn(x)
        result.sum().backward()
        self.assertEqual(len(PrecompileContext._dynamo_cache_entries), 1)
        self.assertEqual(len(PrecompileContext._backend_artifacts_by_key), 1)
        # Find the key for the artifact of type "precompile_aot_autograd"
        key = next(iter(PrecompileContext._backend_artifacts_by_key))

        def edit_fn(x):
            x._my_private_field = 42
            return x

        PrecompileContext.edit_artifact(key, edit_fn)

        result = PrecompileContext.serialize_artifact_by_key(key)
        assert isinstance(result, BundledAOTAutogradCacheArtifact)
        self.assertEqual(result.key, key)

        result, _ = PrecompileContext.create_cache_entries()
        assert len(result) == 1
        aot_autograd_artifacts = next(iter(result.values())).backends
        assert len(aot_autograd_artifacts) == 1
        entry = next(iter(aot_autograd_artifacts.values())).content
        self.assertEqual(entry._my_private_field, 42)

    @requires_triton()
    @unittest.skipIf(not dist.is_available(), "Distributed not available")
    def test_deepcopy_with_device_mesh(self):
        """
        Test that deepcopy in record_artifact works correctly with device mesh
        and DTensor. This reproduces the issue where deepcopy would
        fail when trying to meta storages. See the following PR for more info:
        https://github.com/pytorch/pytorch/pull/169242
        """
        store = FakeStore()
        dist.init_process_group("fake", store=store, rank=0, world_size=4)

        try:
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.tensor import DTensor, Replicate

            mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))

            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(32, 32)

                def forward(self, x, d_x, mesh):
                    x = self.linear(x)
                    y = d_x.redistribute(mesh, placements=(Replicate(), Replicate()))
                    return x, y

            model = SimpleModel().cuda()
            input_tensor = torch.randn(32, 32, device="cuda")
            placements = (Replicate(), Replicate())
            d_input_tensor = DTensor.from_local(input_tensor, mesh, placements)

            compiled_fn = torch.compile(model, fullgraph=True)

            # This should not raise an error about device mismatch during deepcopy
            result = compiled_fn(input_tensor, d_input_tensor, mesh)

            self.assertGreater(len(PrecompileContext._backend_artifacts_by_key), 0)
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
