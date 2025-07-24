# Owner(s): ["module: dynamo"]

import pickle

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._functorch
from torch._dynamo.precompile_context import (
    EditablePrecompileCacheArtifact,
    PrecompileCacheArtifact,
    PrecompileContext,
)
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
        self.assertEqual(len(PrecompileContext._new_cache_artifacts_by_key), 2)
        self.assertEqual(len(PrecompileContext._new_cache_artifacts), 0)

        result = PrecompileContext.serialize()
        assert result is not None
        serialized, cache_info = result
        self.assertEqual(len(cache_info.precompile_aot_autograd_artifacts), 1)

        artifacts = PrecompileContext.deserialize(serialized)
        assert artifacts is not None
        deserialized = artifacts["precompile_aot_autograd"]
        assert len(deserialized) == 1
        entry = deserialized[0]
        assert isinstance(entry, BundledAOTAutogradCacheArtifact)
        entry = entry.after_deserialization()
        # Now that we've serialized, there should be no new cache artifacts
        self.assertEqual(
            len(PrecompileContext._new_cache_artifacts["precompile_aot_autograd"]), 0
        )

    @requires_triton()
    def test_serialize_by_key(self):
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
        self.assertEqual(len(PrecompileContext._new_cache_artifacts_by_key), 2)
        for key in PrecompileContext._new_cache_artifacts_by_key.keys():
            result = PrecompileContext.serialize_artifact_by_key(key)
            assert isinstance(result, PrecompileCacheArtifact)
            self.assertEqual(result.key, key)

        self.assertEqual(len(PrecompileContext._new_cache_artifacts), 0)
        result = PrecompileContext.serialize()
        assert result is not None
        _, cache_info = result
        self.assertEqual(len(cache_info.precompile_aot_autograd_artifacts), 1)

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
        self.assertEqual(len(PrecompileContext._new_cache_artifacts_by_key), 2)
        # Find the key for the artifact of type "precompile_aot_autograd"
        key = next(
            k
            for k, v in PrecompileContext._new_cache_artifacts_by_key.items()
            if isinstance(v, EditablePrecompileCacheArtifact)
        )

        def edit_fn(x):
            x._my_private_field = 42
            return x

        PrecompileContext.edit_artifact(key, edit_fn)

        result = PrecompileContext.serialize_artifact_by_key(key)
        assert isinstance(result, BundledAOTAutogradCacheArtifact)
        self.assertEqual(result.key, key)

        self.assertEqual(len(PrecompileContext._new_cache_artifacts), 0)
        result = PrecompileContext.serialize()
        assert result is not None
        artifacts, cache_info = result
        self.assertEqual(len(cache_info.precompile_aot_autograd_artifacts), 1)

        deserialized = PrecompileContext.deserialize(artifacts)
        assert deserialized is not None
        aot_autograd_artifacts = deserialized["precompile_aot_autograd"]
        assert len(aot_autograd_artifacts) == 1
        entry = aot_autograd_artifacts[0]
        assert isinstance(entry, BundledAOTAutogradCacheArtifact)
        raw_entry = pickle.loads(entry.content)
        self.assertEqual(raw_entry._my_private_field, 42)
        # Now that we've serialized, there should be no new cache artifacts
        self.assertEqual(
            len(PrecompileContext._new_cache_artifacts["precompile_aot_autograd"]), 0
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
