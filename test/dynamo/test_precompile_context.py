# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._functorch
from torch._dynamo.precompile_context import BackendCacheArtifact, PrecompileContext
from torch._functorch import config as functorch_config
from torch._functorch._aot_autograd.autograd_cache import (
    BundledAOTAutogradCacheArtifact,
)
from torch._inductor.test_case import TestCase as InductorTestCase
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
        for key in PrecompileContext._backend_artifacts_by_key:
            result = PrecompileContext.serialize_artifact_by_key(key)
            if not isinstance(result, BackendCacheArtifact):
                raise AssertionError(
                    f"Expected BackendCacheArtifact, got {type(result)}"
                )
            self.assertEqual(result.key, key)

        # This should still work
        result, _ = PrecompileContext.create_cache_entries()
        if len(result) != 1:
            raise AssertionError(f"Expected len(result) == 1, got {len(result)}")

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
        if not isinstance(result, BundledAOTAutogradCacheArtifact):
            raise AssertionError(
                f"Expected BundledAOTAutogradCacheArtifact, got {type(result)}"
            )
        self.assertEqual(result.key, key)

        result, _ = PrecompileContext.create_cache_entries()
        if len(result) != 1:
            raise AssertionError(f"Expected len(result) == 1, got {len(result)}")
        aot_autograd_artifacts = next(iter(result.values())).backends
        if len(aot_autograd_artifacts) != 1:
            raise AssertionError(
                f"Expected len(aot_autograd_artifacts) == 1, got {len(aot_autograd_artifacts)}"
            )
        entry = next(iter(aot_autograd_artifacts.values())).content
        self.assertEqual(entry._my_private_field, 42)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
