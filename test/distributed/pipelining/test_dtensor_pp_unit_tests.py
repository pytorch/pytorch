# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
"""
Unit tests for DTensor support in Pipeline Parallelism.

Tests the _utils.py metadata infrastructure and microbatch DTensor splitting:
1. Metadata classes: _TensorMeta, _DTensorMeta, _MeshCache
2. InferenceMode decision logic
3. Validation functions: validate_metadata, validate_tensors_metadata,
   validate_static_dtensor_grad_correspondence
4. Helper functions: extract_tensor_metas, to_local_if_dtensor,
   validate_and_normalize_to_tuple
5. Microbatch DTensor splitting: _split_tensor placement change warnings
"""

import contextlib
import functools
import warnings
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining._utils import (
    _DTensorMeta,
    _MeshCache,
    _StageMeta,
    _TensorMeta,
    extract_tensor_metas,
    InferenceMode,
    PipeliningMetadataError,
    to_local_if_dtensor,
    validate_and_normalize_to_tuple,
    validate_metadata,
    validate_static_dtensor_grad_correspondence,
    validate_tensors_metadata,
)
from torch.distributed.pipelining.microbatch import _split_tensor, TensorChunkSpec
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_accelerator_dist_backend,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_MULTIACCELERATOR,
)


# =============================================================================
# Test Constants
# =============================================================================

d_hid = 256
batch_size = 64
n_microbatches = 4
microbatch_size = batch_size // n_microbatches

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
backend = dist.get_default_backend_for_device(device_type)


def _requires_multi_gpu(fn):
    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 4+ GPUs"
    )
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


# =============================================================================
# All DTensor PP Unit Tests (single class → single PG initialization)
# =============================================================================


class TestDTensorPPUnitTests(MultiProcContinuousTest):
    """Unit tests for DTensor PP metadata infrastructure.

    All tests live in a single class so the process group is initialized once
    rather than once per test category.
    """

    @classmethod
    def backend_str(cls) -> str:
        return backend

    @classmethod
    def device_type(cls) -> str:
        return device_type

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def init_pg(self):
        if device_type == "cuda":
            torch.cuda.set_device(self.device)

    # -----------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------

    def _make_mesh(self, dim_names=("tp",)):
        """Create a 1D device mesh spanning all ranks."""
        return init_device_mesh(
            device_type, (self.world_size,), mesh_dim_names=dim_names
        )

    def _make_dtensor(self, mesh, placements, shape=(8, 16), requires_grad=False):
        """Create a DTensor with given properties."""
        t = torch.randn(*shape, device=self.device, requires_grad=requires_grad)
        dt = distribute_tensor(t, mesh, placements)
        if requires_grad:
            dt.retain_grad()
        return dt

    def _make_tensor_meta(
        self,
        shape: torch.Size = torch.Size([4, 8]),
        stride: tuple[int, ...] = (8, 1),
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = True,
    ) -> _TensorMeta:
        """Create a _TensorMeta with sensible defaults."""
        return _TensorMeta(
            shape=shape, stride=stride, dtype=dtype, requires_grad=requires_grad
        )

    def _make_dtensor_meta(self) -> _DTensorMeta:
        """Create a _DTensorMeta with sensible defaults."""
        return _DTensorMeta(
            shape=torch.Size([4, 8]),
            stride=(8, 1),
            dtype=torch.float32,
            requires_grad=True,
            global_shape=torch.Size([8, 8]),
            global_stride=(8, 1),
            placements=(Shard(0),),
            mesh_dim_names=("tp",),
            mesh_layout=None,
        )

    def _assert_split_warns_on_placement_change(
        self,
        fake_get_placements,
        expected_warning_substring,
        use_debug_mask=False,
    ):
        """Shared helper: verify _split_tensor warns when placements change."""
        mesh = self._make_mesh()
        dt = self._make_dtensor(mesh, [Shard(0)], shape=(8, 16))

        maybe_mask = (
            patch(
                "torch.distributed.pipelining.microbatch._debug_mask_minibatches", True
            )
            if use_debug_mask
            else contextlib.nullcontext()
        )

        with patch(
            "torch.distributed.pipelining.microbatch._get_dtensor_placements",
            side_effect=fake_get_placements,
        ):
            with maybe_mask:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    _ = _split_tensor(dt, TensorChunkSpec(0), num_chunks=2)
                    self.assertTrue(
                        any(
                            "DTensor placements changed during microbatch split"
                            in str(x.message)
                            for x in w
                        )
                    )
                    self.assertTrue(
                        any(expected_warning_substring in str(x.message) for x in w)
                    )

    # -----------------------------------------------------------------
    # Category 1: Metadata Classes (_TensorMeta, _DTensorMeta, _MeshCache)
    # -----------------------------------------------------------------

    @_requires_multi_gpu
    def test_tensor_meta_extraction_roundtrip_and_errors(self):
        """Test _TensorMeta: extraction, roundtrip, DTensor rejection."""
        self.init_pg()
        mesh = self._make_mesh()

        # Extraction and roundtrip for various shapes/dtypes/requires_grad
        for shape in [(4, 8), (2, 3, 4)]:
            for dtype in [torch.float32, torch.float16]:
                for requires_grad in [True, False]:
                    t = torch.randn(*shape, dtype=dtype, requires_grad=requires_grad)
                    meta = _TensorMeta.from_tensor(t)
                    self.assertEqual(meta.shape, t.shape)
                    self.assertEqual(meta.stride, t.stride())
                    self.assertEqual(meta.dtype, dtype)
                    self.assertEqual(meta.requires_grad, requires_grad)

                    # Roundtrip via meta tensor
                    reconstructed = meta.to_meta_tensor()
                    reconstructed_meta = _TensorMeta(
                        shape=reconstructed.shape,
                        stride=reconstructed.stride(),
                        dtype=reconstructed.dtype,
                        requires_grad=reconstructed.requires_grad,
                    )
                    self.assertEqual(meta.get_diff(reconstructed_meta), [])

        # DTensor rejection
        dt = self._make_dtensor(mesh, [Shard(0)])
        with self.assertRaises(PipeliningMetadataError) as ctx:
            _TensorMeta.from_tensor(dt)
        self.assertIn("DTensor", str(ctx.exception))

    @_requires_multi_gpu
    def test_dtensor_meta_extraction_and_roundtrip(self):
        """Test _DTensorMeta: extraction and roundtrip for Shard/Replicate."""
        self.init_pg()
        mesh = self._make_mesh()

        for placements in [[Shard(0)], [Replicate()]]:
            dt = self._make_dtensor(mesh, placements)
            meta = _DTensorMeta.from_dtensor(dt)

            self.assertEqual(meta.global_shape, dt.shape)
            self.assertEqual(meta.placements, tuple(placements))
            self.assertEqual(meta.mesh_dim_names, ("tp",))
            self.assertEqual(meta.mesh_cache_key, (("tp",), mesh._layout))

            # Roundtrip
            reconstructed = meta.to_meta_dtensor(mesh)
            reconstructed_meta = _DTensorMeta.from_dtensor(reconstructed)
            self.assertEqual(meta.get_diff(reconstructed_meta), [])

    @_requires_multi_gpu
    def test_get_diff_detects_mismatches(self):
        """Test get_diff() detects shape/dtype/placement/cross-type differences."""
        self.init_pg()
        mesh = self._make_mesh()

        # _TensorMeta: shape
        m1 = self._make_tensor_meta(shape=torch.Size([4, 8]), stride=(8, 1))
        m2 = self._make_tensor_meta(shape=torch.Size([4, 16]), stride=(16, 1))
        self.assertTrue(any("shape" in d for d in m1.get_diff(m2)))

        # _TensorMeta: dtype
        m3 = self._make_tensor_meta(dtype=torch.float16)
        self.assertTrue(any("dtype" in d for d in m1.get_diff(m3)))

        # _DTensorMeta: placement
        dt_shard = self._make_dtensor(mesh, [Shard(0)])
        dt_rep = self._make_dtensor(mesh, [Replicate()])
        dm1 = _DTensorMeta.from_dtensor(dt_shard)
        dm2 = _DTensorMeta.from_dtensor(dt_rep)
        self.assertTrue(any("placements" in d for d in dm1.get_diff(dm2)))

        # Cross-type: _DTensorMeta vs _TensorMeta
        diffs = dm1.get_diff(m1)
        self.assertTrue(any("type" in d.lower() or "_TensorMeta" in d for d in diffs))

    @_requires_multi_gpu
    def test_mesh_cache_operations(self):
        """Test _MeshCache: get/put/get_or_create/update_from_tensors."""
        self.init_pg()
        mesh = self._make_mesh()
        key = (("tp",), mesh._layout)

        # --- get/put/contains ---
        cache = _MeshCache()
        self.assertIsNone(cache.get(key))
        self.assertFalse(key in cache)

        cache.put(key, mesh)
        self.assertTrue(key in cache)
        self.assertIs(cache.get(key), mesh)

        # --- get_or_create: hit (callback NOT called) ---
        callback_called = [False]

        def callback(_names, _layout):
            callback_called[0] = True
            return mesh

        self.assertIs(cache.get_or_create(key, callback), mesh)
        self.assertFalse(callback_called[0])

        # --- get_or_create: miss (callback called) ---
        cache2 = _MeshCache()
        self.assertIs(cache2.get_or_create(key, callback), mesh)
        self.assertTrue(callback_called[0])

        # --- get_or_create: miss without callback raises ---
        cache3 = _MeshCache()
        with self.assertRaises(PipeliningMetadataError) as ctx:
            cache3.get_or_create(key, None)
        self.assertIn("get_mesh", str(ctx.exception))

        # --- update_from_tensors: extracts mesh from DTensors ---
        cache4 = _MeshCache()
        dt = self._make_dtensor(mesh, [Shard(0)], shape=(4, 4))
        plain = torch.randn(4, 4, device=self.device)
        cache4.update_from_tensors((dt, plain, None))
        self.assertTrue(key in cache4)
        self.assertIs(cache4.get(key), mesh)
        self.assertEqual(len(cache4), 1)

    # -----------------------------------------------------------------
    # Category 2: InferenceMode Decision Logic
    # -----------------------------------------------------------------

    @_requires_multi_gpu
    def test_needs_dynamic_all_cases(self):
        """Test all 8 cases of the needs_dynamic() decision matrix."""
        self.init_pg()
        tm = self._make_tensor_meta()
        dm = self._make_dtensor_meta()

        cases = [
            # (meta, has_backward, expected_needs_dynamic)
            # Case 1: No forward metadata
            (_StageMeta(inputs=None, outputs=None), True, True),
            # Case 2: Partial forward (inputs only)
            (_StageMeta(inputs=(tm,), outputs=None), True, True),
            # Case 3: Plain tensors, complete forward, backward → STATIC
            (_StageMeta(inputs=(tm,), outputs=(tm,)), True, False),
            # Case 4: DTensors, no backward → STATIC
            (_StageMeta(inputs=(dm,), outputs=(dm,)), False, False),
            # Case 5: DTensors, backward, no grads → DYNAMIC
            (_StageMeta(inputs=(dm,), outputs=(dm,)), True, True),
            # Case 6: DTensors, backward, only input_grads → DYNAMIC
            (
                _StageMeta(
                    inputs=(dm,), outputs=(dm,), input_grads=(dm,), output_grads=None
                ),
                True,
                True,
            ),
            # Case 7: DTensors, backward, only output_grads → DYNAMIC
            (
                _StageMeta(
                    inputs=(dm,), outputs=(dm,), input_grads=None, output_grads=(dm,)
                ),
                True,
                True,
            ),
            # Case 8: DTensors, backward, complete grads → STATIC
            (
                _StageMeta(
                    inputs=(dm,),
                    outputs=(dm,),
                    input_grads=(dm,),
                    output_grads=(dm,),
                ),
                True,
                False,
            ),
        ]
        for i, (meta, has_bwd, expected) in enumerate(cases, 1):
            with self.subTest(case=i):
                self.assertEqual(
                    InferenceMode.needs_dynamic(meta, has_bwd),
                    expected,
                    f"Case {i} failed",
                )

    # -----------------------------------------------------------------
    # Category 3: Validation Functions
    # -----------------------------------------------------------------

    @_requires_multi_gpu
    def test_validate_metadata(self):
        """Test validate_metadata: match, raise, warn, type mismatch."""
        self.init_pg()
        mesh = self._make_mesh()

        # Match → no diffs
        dt = self._make_dtensor(mesh, [Shard(0)])
        meta = _DTensorMeta.from_dtensor(dt)
        self.assertEqual(validate_metadata("test", meta, dt), [])

        # Mismatch with raise_on_mismatch
        other = self._make_dtensor(mesh, [Replicate()])
        with self.assertRaises(PipeliningMetadataError):
            validate_metadata("test", meta, other, raise_on_mismatch=True)

        # Mismatch with warn_on_mismatch
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            diffs = validate_metadata("test", meta, other, warn_on_mismatch=True)
            self.assertTrue(len(diffs) > 0)
            self.assertTrue(any("Metadata mismatch" in str(x.message) for x in w))

        # Type mismatch: _DTensorMeta vs plain tensor
        plain = torch.randn(8, 16, device=self.device)
        diffs = validate_metadata("test", meta, plain)
        self.assertTrue(any("type" in d for d in diffs))

    @_requires_multi_gpu
    def test_validate_tensors_metadata(self):
        """Test validate_tensors_metadata: batch validation, length mismatch, None handling."""
        self.init_pg()
        mesh = self._make_mesh()

        dt = self._make_dtensor(mesh, [Shard(0)])
        meta = _DTensorMeta.from_dtensor(dt)

        # Match
        validate_tensors_metadata("test", (meta,), (dt,))

        # Length mismatch raises
        with self.assertRaises(PipeliningMetadataError):
            validate_tensors_metadata("test", (meta, meta), (dt,))

        # Length mismatch warns
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tensors_metadata(
                "test", (meta, meta), (dt,), warn_on_mismatch=True
            )
            self.assertTrue(len(w) > 0)

        # None-None pair passes
        validate_tensors_metadata("test", (None,), (None,))

        # skip_none_actuals=True skips None actuals
        validate_tensors_metadata("test", (meta,), (None,), skip_none_actuals=True)

        # skip_none_actuals=False raises on None actual
        with self.assertRaises(PipeliningMetadataError):
            validate_tensors_metadata("test", (meta,), (None,))

    @_requires_multi_gpu
    def test_validate_static_dtensor_grad_correspondence(self):
        """Test DTensor↔grad correspondence: all valid/invalid combos."""
        self.init_pg()
        mesh = self._make_mesh()

        dt_grad = self._make_dtensor(mesh, [Shard(0)], requires_grad=True)
        dt_nograd = self._make_dtensor(mesh, [Shard(0)], requires_grad=False)
        dt_grad2 = self._make_dtensor(mesh, [Replicate()], requires_grad=True)
        plain = torch.randn(8, 16, device=self.device)

        # DTensor(requires_grad=True) + DTensor grad → OK
        validate_static_dtensor_grad_correspondence(
            0, (dt_grad,), (dt_grad2,), is_input=True
        )

        # DTensor(requires_grad=False) + None grad → OK
        validate_static_dtensor_grad_correspondence(
            0, (dt_nograd,), (None,), is_input=True
        )

        # Plain tensor + None grad → OK
        validate_static_dtensor_grad_correspondence(0, (plain,), (None,), is_input=True)

        # DTensor(requires_grad=True) + None grad → OK (boundary with no gradient)
        validate_static_dtensor_grad_correspondence(
            0, (dt_grad,), (None,), is_input=True
        )

        # DTensor(requires_grad=True) + plain tensor grad → ERROR
        with self.assertRaises(PipeliningMetadataError):
            validate_static_dtensor_grad_correspondence(
                0, (dt_grad,), (plain,), is_input=True
            )

        # Length mismatch → ERROR
        with self.assertRaises(PipeliningMetadataError):
            validate_static_dtensor_grad_correspondence(
                0, (dt_grad, dt_grad), (dt_grad2,), is_input=False
            )

    # -----------------------------------------------------------------
    # Category 4: Helper Functions
    # -----------------------------------------------------------------

    @_requires_multi_gpu
    def test_extract_tensor_metas(self):
        """Test extract_tensor_metas: plain, DTensor, None handling."""
        self.init_pg()
        mesh = self._make_mesh()

        dt = self._make_dtensor(mesh, [Shard(0)])
        plain = torch.randn(4, 4, device=self.device)

        # None input → None output
        self.assertIsNone(extract_tensor_metas(None))

        # Mixed DTensor + plain → correct meta types
        metas = extract_tensor_metas((dt, plain))
        self.assertIsNotNone(metas)
        self.assertIsInstance(metas[0], _DTensorMeta)  # type: ignore[index]
        self.assertIsInstance(metas[1], _TensorMeta)  # type: ignore[index]
        self.assertNotIsInstance(metas[1], _DTensorMeta)  # type: ignore[index]

        # allow_none=True preserves None
        metas_with_none = extract_tensor_metas((dt, None), allow_none=True)
        self.assertIsNotNone(metas_with_none)
        self.assertIsInstance(metas_with_none[0], _DTensorMeta)  # type: ignore[index]
        self.assertIsNone(metas_with_none[1])  # type: ignore[index]

        # allow_none=False (default) rejects None
        with self.assertRaises(PipeliningMetadataError):
            extract_tensor_metas((dt, None))  # type: ignore[arg-type]

    @_requires_multi_gpu
    def test_to_local_if_dtensor(self):
        """Test to_local_if_dtensor: DTensor→local, plain passthrough, detach."""
        self.init_pg()
        mesh = self._make_mesh()

        dt = self._make_dtensor(mesh, [Shard(0)], requires_grad=True)
        plain = torch.randn(4, 4, device=self.device, requires_grad=True)

        # DTensor → local tensor (not DTensor)
        local = to_local_if_dtensor(dt)
        self.assertNotIsInstance(local, DTensor)
        self.assertEqual(local.shape, dt._local_tensor.shape)

        # Plain tensor → same tensor
        result = to_local_if_dtensor(plain)
        self.assertIs(result, plain)

        # detach=True → result is detached
        local_detached = to_local_if_dtensor(dt, detach=True)
        self.assertNotIsInstance(local_detached, DTensor)
        self.assertFalse(local_detached.requires_grad)

    @_requires_multi_gpu
    def test_validate_and_normalize_to_tuple(self):
        """Test validate_and_normalize_to_tuple: single, tuple, list, None, errors."""
        self.init_pg()
        t = torch.randn(4, 4, device=self.device)

        # None → None
        self.assertIsNone(validate_and_normalize_to_tuple(None))

        # Single tensor → 1-tuple
        result = validate_and_normalize_to_tuple(t)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)  # type: ignore[arg-type]
        self.assertIs(result[0], t)  # type: ignore[index]

        # List → tuple
        result = validate_and_normalize_to_tuple([t, t])
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)  # type: ignore[arg-type]

        # Non-tensor element → error
        with self.assertRaises(PipeliningMetadataError):
            validate_and_normalize_to_tuple((t, "not_a_tensor"))  # type: ignore[arg-type]

        # None in tuple without allow_none → error
        with self.assertRaises(PipeliningMetadataError):
            validate_and_normalize_to_tuple((t, None))  # type: ignore[arg-type]

        # None in tuple with allow_none=True → OK
        result_with_none = validate_and_normalize_to_tuple((t, None), allow_none=True)
        self.assertIsNotNone(result_with_none)
        self.assertEqual(len(result_with_none), 2)  # type: ignore[arg-type]
        self.assertIsNone(result_with_none[1])  # type: ignore[index]

        # Wrong type entirely → error
        with self.assertRaises(PipeliningMetadataError):
            validate_and_normalize_to_tuple(42)  # type: ignore[arg-type]

    # -----------------------------------------------------------------
    # Category 5: Microbatch DTensor Splitting
    # -----------------------------------------------------------------

    @_requires_multi_gpu
    def test_split_tensor_warns_on_dtensor_chunk_placement_change(self):
        self.init_pg()
        call_counter = {"count": 0}

        def fake_get_dtensor_placements(_tensor):
            call_counter["count"] += 1
            if call_counter["count"] == 1:
                return (Shard(0),)
            return (Replicate(),)

        self._assert_split_warns_on_placement_change(
            fake_get_dtensor_placements, "new placements"
        )

    @_requires_multi_gpu
    def test_split_tensor_warns_on_dtensor_expanded_chunk_placement_change(self):
        self.init_pg()
        call_counter = {"count": 0}

        def fake_get_dtensor_placements(_tensor):
            call_counter["count"] += 1
            if call_counter["count"] == 1:
                return (Shard(0),)
            if call_counter["count"] <= 3:
                return (Shard(0),)
            return (Replicate(),)

        self._assert_split_warns_on_placement_change(
            fake_get_dtensor_placements, "expanded chunk", use_debug_mask=True
        )


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    run_tests()
