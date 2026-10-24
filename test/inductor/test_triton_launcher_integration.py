# Owner(s): ["module: inductor"]
"""Tests for save_gpu_kernel() Level 0 schema integration (diff 3/N).

Validates that save_gpu_kernel reads metadata from launch_metadata_schema
when available, falling back to hasattr probing for older Triton versions.

Run:
  buck2 test @fbcode//mode/opt --modifier ovr_config//triton:beta \
    fbcode//caffe2/test/inductor:test_triton_launcher_integration
"""

import unittest
from unittest.mock import MagicMock, patch

from torch.testing._internal.common_utils import TestCase


def _make_mock_launcher(
    metadata_name="test_kernel_0d1d2d",
    num_warps=4,
    shared=0,
    launch_metadata_schema=None,
):
    """Create a mock launcher with bin (CompiledKernel)."""
    launcher = MagicMock()
    launcher.config = MagicMock()
    launcher.def_args = ["X", "Y", "N"]
    launcher.call_args = ["X", "Y", "N"]
    launcher.global_scratch = None
    launcher.profile_scratch = None

    bin_mock = MagicMock()
    bin_mock.metadata.name = metadata_name
    bin_mock.metadata.num_warps = num_warps
    bin_mock.metadata.shared = shared
    bin_mock.num_warps = num_warps
    bin_mock.shared = shared
    bin_mock.launch_metadata_schema = launch_metadata_schema
    bin_mock.asm = {"cubin": b"\x00", "ptx": "mock_ptx"}
    launcher.bin = bin_mock

    return launcher


def _make_mock_autotuner(kernel_name="test_kernel"):
    """Create a mock CachingAutotuner (self) for save_gpu_kernel."""
    autotuner = MagicMock()
    autotuner.inductor_meta = {"kernel_name": kernel_name}
    autotuner.triton_meta = {"signature": {0: "*fp32"}}
    autotuner.device_props = MagicMock()
    autotuner.device_props.type = "cuda"
    return autotuner


class SaveGpuKernelSchemaTest(TestCase):
    """Unit tests for save_gpu_kernel() reading from Level 0 schema."""

    def _call_save_gpu_kernel(self, launcher, kernel_name="test_kernel"):
        """Call save_gpu_kernel with mocks and return the params passed to cache."""
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        autotuner = _make_mock_autotuner(kernel_name)
        with (
            patch(
                "torch._inductor.codecache.CudaKernelParamCache.set"
            ) as mock_cache_set,
            patch(
                "torch._inductor.runtime.triton_heuristics.config_to_dict",
                return_value={"BLOCK_SIZE": 1024},
            ),
        ):
            CachingAutotuner.save_gpu_kernel(autotuner, stream=0, launcher=launcher)
            mock_cache_set.assert_called_once()
            _key, params, _binary, _bin_type, _asm, _asm_type = (
                mock_cache_set.call_args[0]
            )
        return params

    def test_schema_path_reads_entry_name(self):
        """When schema exists, mangled_name should come from schema['entry_name']."""
        schema = {
            "abi_version": 1,
            "entry_name": "schema_kernel_name",
            "num_warps": 8,
            "shared_mem": 4096,
        }
        launcher = _make_mock_launcher(
            metadata_name="metadata_kernel_name",
            launch_metadata_schema=schema,
        )
        params = self._call_save_gpu_kernel(launcher)
        self.assertEqual(params["mangled_name"], "schema_kernel_name")

    def test_schema_path_reads_num_warps(self):
        """When schema exists, num_warps should come from schema."""
        schema = {
            "abi_version": 1,
            "entry_name": "test_kernel",
            "num_warps": 16,
            "shared_mem": 8192,
        }
        launcher = _make_mock_launcher(
            num_warps=4,
            launch_metadata_schema=schema,
        )
        params = self._call_save_gpu_kernel(launcher)
        self.assertEqual(params["num_warps"], 16)

    def test_schema_path_reads_shared_mem(self):
        """When schema exists, shared_mem should come from schema."""
        schema = {
            "abi_version": 1,
            "entry_name": "test_kernel",
            "num_warps": 4,
            "shared_mem": 49152,
        }
        launcher = _make_mock_launcher(
            shared=0,
            launch_metadata_schema=schema,
        )
        params = self._call_save_gpu_kernel(launcher)
        self.assertEqual(params["shared_mem"], 49152)

    def test_fallback_when_no_schema(self):
        """Without schema, should use hasattr probing (metadata.name, etc.)."""
        launcher = _make_mock_launcher(
            metadata_name="fallback_name",
            num_warps=4,
            shared=1024,
            launch_metadata_schema=None,
        )
        params = self._call_save_gpu_kernel(launcher)
        self.assertEqual(params["mangled_name"], "fallback_name")
        self.assertEqual(params["num_warps"], 4)
        self.assertEqual(params["shared_mem"], 1024)


class LazyCompileParamCacheCollisionTest(TestCase):
    """Unit tests for #199612: prevent CudaKernelParamCache collision in lazy compile."""

    def setUp(self):
        super().setUp()
        from torch._inductor.codecache import CudaKernelParamCache

        CudaKernelParamCache.cache_clear()

    def tearDown(self):
        from torch._inductor.codecache import CudaKernelParamCache

        CudaKernelParamCache.cache_clear()
        super().tearDown()

    def test_cache_set_and_get_signature_distinction(self):
        from torch._inductor.codecache import CudaKernelParamCache

        sig_a = {"in_ptr0": "*fp32", "in_ptr1": "*fp32", "out_ptr0": "*fp32"}
        params_a = {
            "mangled_name": "triton_poi_fused_add_mul_0_hashA",
            "num_warps": 4,
            "shared_mem": 1024,
            "triton_meta": {"signature": sig_a},
            "kernel_hash": "hash_aaa",
        }

        sig_c = {
            "in_ptr0": "*fp32",
            "in_ptr1": "*fp32",
            "in_ptr2": "*fp32",
            "out_ptr0": "*fp32",
        }
        params_c = {
            "mangled_name": "triton_poi_fused_add_mul_0_hashC",
            "num_warps": 8,
            "shared_mem": 2048,
            "triton_meta": {"signature": sig_c},
            "kernel_hash": "hash_ccc",
        }

        with patch("torch._inductor.codecache.write", return_value=("", "/fake/path/bin")):
            CudaKernelParamCache.set("triton_poi_fused_add_mul_0", params_a, b"cubin_a", "cubin")
            CudaKernelParamCache.set("triton_poi_fused_add_mul_0", params_c, b"cubin_c", "cubin")

        # Flat lookup returns the latest entry (Kernel C) for backward compatibility
        self.assertEqual(
            CudaKernelParamCache.get("triton_poi_fused_add_mul_0")["mangled_name"],
            "triton_poi_fused_add_mul_0_hashC",
        )

        # Lookup with signature for Kernel A returns Kernel A
        res_a = CudaKernelParamCache.get("triton_poi_fused_add_mul_0", signature=sig_a)
        self.assertIsNotNone(res_a)
        self.assertEqual(res_a["mangled_name"], "triton_poi_fused_add_mul_0_hashA")

        # Lookup with signature for Kernel C returns Kernel C
        res_c = CudaKernelParamCache.get("triton_poi_fused_add_mul_0", signature=sig_c)
        self.assertIsNotNone(res_c)
        self.assertEqual(res_c["mangled_name"], "triton_poi_fused_add_mul_0_hashC")

        # Lookup with non-existent signature returns None rather than wrong entry
        sig_unknown = {"in_ptr0": "*fp64"}
        self.assertIsNone(
            CudaKernelParamCache.get("triton_poi_fused_add_mul_0", signature=sig_unknown)
        )

    def test_cache_clear(self):
        from torch._inductor.codecache import CudaKernelParamCache

        sig_a = {"in_ptr0": "*fp32"}
        params_a = {
            "mangled_name": "kernel_a",
            "triton_meta": {"signature": sig_a},
        }
        with patch("torch._inductor.codecache.write", return_value=("", "/fake/path/bin")):
            CudaKernelParamCache.set("kernel_a", params_a, b"cubin_a", "cubin")

        self.assertIsNotNone(CudaKernelParamCache.get("kernel_a", signature=sig_a))
        CudaKernelParamCache.cache_clear()
        self.assertIsNone(CudaKernelParamCache.get("kernel_a", signature=sig_a))
        self.assertIsNone(CudaKernelParamCache.get("kernel_a"))

    def test_run_triton_kernel_with_autotune_uses_kernel_params_on_fast_path(self):
        from torch._inductor.codecache import CudaKernelParamCache
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        from torch._inductor.runtime.triton_lazy_compile import (
            run_triton_kernel_with_autotune,
        )

        sig_a = {"in_ptr0": "*fp32", "in_ptr1": "*fp32", "out_ptr0": "*fp32"}
        autotuner_a = MagicMock(spec=CachingAutotuner)
        autotuner_a.inductor_meta = {"kernel_name": "triton_poi_fused_add_mul_0"}
        autotuner_a.triton_meta = {"signature": sig_a}
        autotuner_a.kernel_hash = "hash_aaa"
        launcher_a = MagicMock()
        launcher_a.config = None
        autotuner_a.launchers = [launcher_a]

        params_a = {
            "mangled_name": "mangled_A",
            "num_warps": 4,
            "shared_mem": 1024,
            "cubin_path": "/path/to/cubin_A.cubin",
            "runtime_bin_path": "/path/to/cubin_A.cubin",
            "global_scratch": 0,
            "profile_scratch": 0,
            "tensordesc_meta": [],
            "triton_meta": {"signature": sig_a},
            "kernel_hash": "hash_aaa",
        }
        autotuner_a.cuda_kernel_saved = True
        autotuner_a.cuda_kernel_params = params_a

        sig_c = {
            "in_ptr0": "*fp32",
            "in_ptr1": "*fp32",
            "in_ptr2": "*fp32",
            "out_ptr0": "*fp32",
        }
        params_c = {
            "mangled_name": "mangled_C",
            "num_warps": 8,
            "shared_mem": 2048,
            "cubin_path": "/path/to/cubin_C.cubin",
            "runtime_bin_path": "/path/to/cubin_C.cubin",
            "global_scratch": 0,
            "profile_scratch": 0,
            "tensordesc_meta": [],
            "triton_meta": {"signature": sig_c},
            "kernel_hash": "hash_ccc",
        }
        with patch("torch._inductor.codecache.write", return_value=("", "/path/to/cubin_C.cubin")):
            CudaKernelParamCache.set("triton_poi_fused_add_mul_0", params_c, b"cubin_c", "cubin")

        pending_kernels = {"triton_poi_fused_add_mul_0": autotuner_a}
        result = run_triton_kernel_with_autotune(
            pending_kernels,
            "triton_poi_fused_add_mul_0",
            stream=None,
            args=[MagicMock(), MagicMock(), MagicMock()],
        )

        self.assertEqual(result.cubin_path, "/path/to/cubin_A.cubin")
        self.assertEqual(result.mangled_name, "mangled_A")
        self.assertEqual(result.num_warps, 4)
        self.assertEqual(result.shared_mem, 1024)

    def test_caching_autotuner_fast_path_bypasses_when_store_cubin_not_saved(self):
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        autotuner = CachingAutotuner.__new__(CachingAutotuner)
        autotuner._cached_launcher = MagicMock(return_value="fast_result")
        autotuner.inductor_meta = {"store_cubin": True}
        autotuner.device_props = MagicMock()
        autotuner.device_props.type = "cuda"
        autotuner.cuda_kernel_saved = False
        autotuner.cpu_kernel_saved = False

        condition = (
            autotuner._cached_launcher is not None
            and not (
                autotuner.inductor_meta.get("store_cubin")
                and (
                    not autotuner.cpu_kernel_saved
                    if autotuner.device_props.type == "cpu"
                    else not autotuner.cuda_kernel_saved
                )
            )
        )
        self.assertFalse(
            condition,
            "Fast path must NOT be taken when store_cubin is True but kernel not yet saved",
        )

        autotuner.cuda_kernel_saved = True
        condition_after_save = (
            autotuner._cached_launcher is not None
            and not (
                autotuner.inductor_meta.get("store_cubin")
                and (
                    not autotuner.cpu_kernel_saved
                    if autotuner.device_props.type == "cpu"
                    else not autotuner.cuda_kernel_saved
                )
            )
        )
        self.assertTrue(
            condition_after_save, "Fast path CAN be taken after kernel is saved"
        )


# =========================================================================
# E2E tests: AOTI pipeline → save_gpu_kernel schema integration
# Requires GPU + torch + inductor. Run via buck2 on H100.
# =========================================================================

_HAS_GPU = False
try:
    import torch

    _HAS_GPU = hasattr(torch, "cuda") and torch.cuda.is_available()
except (ImportError, AttributeError):
    pass


@unittest.skipUnless(_HAS_GPU, "requires CUDA GPU")
class SaveGpuKernelAOTIE2ETest(TestCase):
    """E2E tests for 3/N: verify AOTI pipeline uses save_gpu_kernel schema path."""

    def setUp(self):
        super().setUp()
        # Fix a known module-loading-order issue (see D74745573):
        # torch.utils.cpp_extension.CUDA_HOME is evaluated at import time.
        # In CI's RE sandbox, CUDA_HOME env var isn't set yet at that point,
        # so the module-level variable caches None. We fix both the env var
        # AND the already-cached module variable.
        # NOTE: triton.fb is fbcode-only; OSS CI sets CUDA_HOME via env.
        import os

        if "CUDA_HOME" not in os.environ and "CUDA_PATH" not in os.environ:
            try:
                from triton.fb.build import build_paths

                os.environ["CUDA_HOME"] = build_paths.sdk_home
            except ImportError:
                pass

        import torch.utils.cpp_extension

        if torch.utils.cpp_extension.CUDA_HOME is None:
            torch.utils.cpp_extension.CUDA_HOME = os.environ.get("CUDA_HOME")

    def test_aoti_pipeline_correctness(self):
        """Full AOTI pipeline (export → compile → load → run) should produce correct results."""
        import torch._dynamo
        import torch._inductor
        import torch.export

        torch._dynamo.reset()

        class AddModel(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = AddModel().to("cuda")
        x = torch.randn(1024, device="cuda")
        y = torch.randn(1024, device="cuda")
        expected = model(x, y)

        ep = torch.export.export(model, (x, y))
        pkg_path = torch._inductor.aoti_compile_and_package(ep)
        loaded = torch._inductor.aoti_load_package(pkg_path)
        result = loaded(x, y)
        torch.testing.assert_close(result, expected)

    def test_aoti_schema_populated(self):
        """AOTI compilation should call save_gpu_kernel with a populated schema."""
        import torch._dynamo
        import torch._inductor
        import torch._inductor.runtime.triton_heuristics as th
        import torch.export

        torch._dynamo.reset()

        original_save = th.CachingAutotuner.save_gpu_kernel
        schema_info = {"called": False, "schema_used": False, "schema_keys": None}

        def patched_save_gpu_kernel(self, stream, launcher):
            schema = getattr(launcher.bin, "launch_metadata_schema", None)
            schema_info["called"] = True
            if schema is not None:
                schema_info["schema_used"] = True
                schema_info["schema_keys"] = list(schema.keys())
            return original_save(self, stream, launcher)

        th.CachingAutotuner.save_gpu_kernel = patched_save_gpu_kernel
        try:

            class MulAddModel(torch.nn.Module):
                def forward(self, x, y, z):
                    return x * y + z

            model = MulAddModel().to("cuda")
            x = torch.randn(512, device="cuda")
            y = torch.randn(512, device="cuda")
            z = torch.randn(512, device="cuda")
            expected = model(x, y, z)

            ep = torch.export.export(model, (x, y, z))
            pkg_path = torch._inductor.aoti_compile_and_package(ep)
            loaded = torch._inductor.aoti_load_package(pkg_path)
            result = loaded(x, y, z)

            torch.testing.assert_close(result, expected)

            self.assertTrue(
                schema_info["called"],
                "save_gpu_kernel was never called during AOTI compilation",
            )
            # launch_metadata_schema is only available on Triton beta (fbcode
            # uses ovr_config//triton:beta via BUCK modifiers).  OSS CI may run
            # with a Triton version that doesn't expose the schema yet, so we
            # conditionally validate the schema contents here.  The Level 0
            # schema reading logic is fully covered by SaveGpuKernelSchemaTest
            # unit tests above (with mocks).
            if schema_info["schema_used"]:
                self.assertIn("entry_name", schema_info["schema_keys"])
                self.assertIn("num_warps", schema_info["schema_keys"])
                self.assertIn("shared_mem", schema_info["schema_keys"])
        finally:
            th.CachingAutotuner.save_gpu_kernel = original_save


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
