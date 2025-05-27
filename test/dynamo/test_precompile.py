# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._functorch
from torch._dynamo.precompile import PrecompileCacheArtifact, PrecompileContext
from torch._functorch import config as functorch_config
from torch._inductor.test_case import TestCase as InductorTestCase


@functorch_config.patch({"enable_autograd_cache": True})
@functorch_config.patch({"bundled_autograd_cache": True}) # Requires bundledaotautograd cache for now
class PrecompileContextTests(InductorTestCase):
    def setUp(self):
        """
        Reset all counters and caches before each unit test
        """
        super().setUp()
        # Clear PrecompileContext cache artifacts
        PrecompileContext.clear()

    def test_basic(self):
        """
        Test that after torch.compile, PrecompileContext._new_cache_artifacts length is 1
        """
        def simple_function(x):
            return x.sin() + x.cos()

        compiled_fn = torch.compile(simple_function)

        # Run the compiled function
        x = torch.randn(10, device="cuda", requires_grad=True)
        result = compiled_fn(x)
        result.sum().backward()
        # Check that PrecompileContext._new_cache_artifacts has length 1
        self.assertEqual(len(PrecompileContext._new_cache_artifacts), 1)
        result = PrecompileContext.serialize()
        assert result is not None
        serialized, cache_info = result
        self.assertEqual(len(cache_info.precompile_aot_autograd_artifacts), 1)

        #
        artifacts = PrecompileContext.deserialize_into_callable(serialized)
        assert artifacts is not None
        deserialized = artifacts["precompile_aot_autograd"]
        assert len(deserialized) == 1
        entry = deserialized[0]
        assert isinstance(entry, torch._functorch._aot_autograd.autograd_cache.BundledAOTAutogradCacheEntry)

        # Now that we've serialized, there should be no new cache artifacts
        self.assertEqual(len(PrecompileContext._new_cache_artifacts["precompile_aot_autograd"]), 0)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    run_tests()
