# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
"""
Unit tests for DTensor support in Pipeline Parallelism.

Tests the _utils.py metadata infrastructure and microbatch DTensor operations:
1. Metadata classes: _TensorMeta, _DTensorMeta, _MeshCache
2. InferenceMode decision logic
3. Validation functions: validate_metadata, validate_tensors_metadata,
   validate_static_arg_grad_correspondence
4. Helper functions: extract_tensor_metas, to_local_if_dtensor,
   validate_and_normalize_to_tuple
5. Microbatch DTensor operations: _split_tensor, merge_chunks
"""

import functools
import warnings

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining._utils import (
    _derive_grad_metas,
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
    validate_static_arg_grad_correspondence,
    validate_tensors_metadata,
)
from torch.distributed.pipelining.microbatch import (
    _split_tensor,
    merge_chunks,
    TensorChunkSpec,
)
from torch.distributed.pipelining.stage import PipelineStage
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import (
    HardwareClassification,
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


def _requires_multi_gpu(fn):
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "Test requires 4+ GPUs")
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


# =============================================================================
# All DTensor PP Unit Tests (single class → single PG initialization)
# =============================================================================


class DTensorPPTestBase(MultiProcContinuousTest):
    # instantiate_device_type_tests copies non-test members using getattr(). A
    # classmethod defined on generic_test_class is copied already bound to that
    # class and cannot see the generated class's device_type. Defining backend_str
    # on this base leaves it inherited, so cls refers to the generated
    # device-specific test class when called.
    @classmethod
    def backend_str(cls) -> str:
        return dist.get_default_backend_for_device(cls.device_type)

    def _rank_device(self, device: str) -> torch.device:
        device_type = torch.device(device).type
        return torch.device(device_type, self.rank)


class TestDTensorPPUnitTests(DTensorPPTestBase):
    """Unit tests for DTensor PP metadata infrastructure.

    All tests live in a single class so the process group is initialized once
    rather than once per test category.
    """

    hw_classification = HardwareClassification.ACCELERATOR

    def init_pg(self, device: torch.device):
        if torch.accelerator.is_available():
            torch.accelerator.set_device_index(device)

    # -----------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------

    def _make_mesh(self, dim_names=("tp",)):
        """Create a 1D device mesh spanning all ranks."""
        return init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=dim_names
        )

    def _make_dtensor(
        self, mesh, placements, shape=(8, 16), requires_grad=False, *, device
    ):
        """Create a DTensor with given properties."""
        t = torch.randn(*shape, device=device, requires_grad=requires_grad)
        return distribute_tensor(t, mesh, placements)

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

    # -----------------------------------------------------------------
    # Category 1: Metadata Classes (_TensorMeta, _DTensorMeta, _MeshCache)
    # -----------------------------------------------------------------

    @_requires_multi_gpu
    def test_tensor_meta_extraction_roundtrip_and_errors(self, device):
        """Test _TensorMeta: extraction, roundtrip, DTensor rejection."""
        rank_device = self._rank_device(device)
        self.init_pg(rank_device)
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

                    # Roundtrip via tensor on device
                    reconstructed = meta.to_tensor(rank_device)
                    reconstructed_meta = _TensorMeta(
                        shape=reconstructed.shape,
                        stride=reconstructed.stride(),
                        dtype=reconstructed.dtype,
                        requires_grad=reconstructed.requires_grad,
                    )
                    self.assertEqual(meta.get_diff(reconstructed_meta), [])

        # DTensor rejection
        dt = self._make_dtensor(mesh, [Shard(0)], device=rank_device)
        with self.assertRaises(PipeliningMetadataError) as ctx:
            _TensorMeta.from_tensor(dt)
        self.assertIn("DTensor", str(ctx.exception))

    @_requires_multi_gpu
    def test_tensor_meta_non_float_requires_grad_guard(self, device):
        """Test that to_tensor/to_dtensor don't set requires_grad on non-float dtypes."""
        rank_device = self._rank_device(device)
        self.init_pg(rank_device)
        mesh = self._make_mesh()

        # _TensorMeta: non-float with requires_grad=True in metadata should not crash
        for dtype in [torch.long, torch.int32, torch.bool]:
            meta = _TensorMeta(
                shape=torch.Size([4, 8]),
                stride=(8, 1),
                dtype=dtype,
                requires_grad=True,  # would crash without the guard
            )
            t = meta.to_tensor(rank_device)
            self.assertEqual(t.dtype, dtype)
            self.assertFalse(t.requires_grad)

        # _TensorMeta: float with requires_grad=True should still work
        float_meta = _TensorMeta(
            shape=torch.Size([4, 8]),
            stride=(8, 1),
            dtype=torch.float32,
            requires_grad=True,
        )
        t = float_meta.to_tensor(rank_device)
        self.assertTrue(t.requires_grad)

        # _DTensorMeta: non-float with requires_grad=True should not crash
        for dtype in [torch.long, torch.int32]:
            dt_meta = _DTensorMeta(
                shape=torch.Size([4, 8]),
                stride=(8, 1),
                dtype=dtype,
                requires_grad=True,
                global_shape=torch.Size([8, 8]),
                global_stride=(8, 1),
                placements=(Shard(0),),
                mesh_dim_names=("tp",),
                mesh_layout=mesh._layout,
            )
            dt = dt_meta.to_dtensor(rank_device, mesh)
            self.assertEqual(dt.dtype, dtype)
            self.assertFalse(dt.requires_grad)

    @_requires_multi_gpu
    def test_dtensor_meta_extraction_and_roundtrip(self, device):
        """Test _DTensorMeta: extraction and roundtrip for Shard/Replicate."""
        rank_device = self._rank_device(device)
        self.init_pg(rank_device)
        mesh = self._make_mesh()

        for placements in [[Shard(0)], [Replicate()]]:
            dt = self._make_dtensor(mesh, placements, device=rank_device)
            meta = _DTensorMeta.from_dtensor(dt)

            self.assertEqual(meta.global_shape, dt.shape)
            self.assertEqual(meta.placements, tuple(placements))
            self.assertEqual(meta.mesh_dim_names, ("tp",))
            self.assertEqual(meta.mesh_cache_key, (("tp",), mesh._layout))

            # Roundtrip
            reconstructed = meta.to_dtensor(rank_device, mesh)
            reconstructed_meta = _DTensorMeta.from_dtensor(reconstructed)
            self.assertEqual(meta.get_diff(reconstructed_meta), [])

    @_requires_multi_gpu
    def test_get_diff_detects_mismatches(self, device):
        """Test get_diff() detects shape/dtype/placement/cross-type differences."""
        rank_device = self._rank_device(device)
        self.init_pg(rank_device)
        mesh = self._make_mesh()

        # _TensorMeta: shape
        m1 = self._make_tensor_meta(shape=torch.Size([4, 8]), stride=(8, 1))
        m2 = self._make_tensor_meta(shape=torch.Size([4, 16]), stride=(16, 1))
        self.assertTrue(any("shape" in d for d in m1.get_diff(m2)))

        # _TensorMeta: dtype
        m3 = self._make_tensor_meta(dtype=torch.float16)
        self.assertTrue(any("dtype" in d for d in m1.get_diff(m3)))

        # _DTensorMeta: placement
        dt_shard = self._make_dtensor(mesh, [Shard(0)], device=rank_device)
        dt_rep = self._make_dtensor(mesh, [Replicate()], device=rank_device)
        dm1 = _DTensorMeta.from_dtensor(dt_shard)
        dm2 = _DTensorMeta.from_dtensor(dt_rep)
        self.assertTrue(any("placements" in d for d in dm1.get_diff(dm2)))

        # Cross-type: _DTensorMeta vs _TensorMeta
        diffs = dm1.get_diff(m1)
        self.assertTrue(any("type" in d.lower() or "_TensorMeta" in d for d in diffs))

    @_requires_multi_gpu
    def test_mesh_cache_operations(self, device):
        """Test _MeshCache: __contains__/put/get_mesh/callback/update_from_tensors."""
        rank_device = self._rank_device(device)
        self.init_pg(rank_device)
        mesh = self._make_mesh()
        key = (("tp",), mesh._layout)

        # --- __contains__ + put + get_mesh: basic cache hit ---
        cache = _MeshCache()
        self.assertFalse(key in cache)
        cache.put(key, mesh)
        self.assertTrue(key in cache)
        self.assertIs(cache.get_mesh(key), mesh)

        # --- get_mesh_cb: callback NOT called on cache hit ---
        cb_called = [False]

        def cb(mesh_dim_names, mesh_layout):
            cb_called[0] = True
            return mesh

        cache_with_cb = _MeshCache(get_mesh_cb=cb)
        cache_with_cb.put(key, mesh)
        self.assertIs(cache_with_cb.get_mesh(key), mesh)
        self.assertFalse(cb_called[0])

        # --- get_mesh_cb: callback IS called on cache miss ---
        cb2_called = [False]

        def cb2(mesh_dim_names, mesh_layout):
            cb2_called[0] = True
            return mesh

        cache_miss = _MeshCache(get_mesh_cb=cb2)
        self.assertIs(cache_miss.get_mesh(key), mesh)
        self.assertTrue(cb2_called[0])

        # --- no callback: get_mesh on miss raises ---
        cache_no_cb = _MeshCache()
        with self.assertRaises(PipeliningMetadataError) as ctx:
            cache_no_cb.get_mesh(key)
        self.assertIn("get_mesh", str(ctx.exception))

        # --- update_from_tensors: extracts mesh from DTensors ---
        cache_upd = _MeshCache()
        dt = self._make_dtensor(mesh, [Shard(0)], shape=(4, 4), device=rank_device)
        plain = torch.randn(4, 4, device=rank_device)
        cache_upd.update_from_tensors((dt, plain, None))
        self.assertTrue(key in cache_upd)
        self.assertIs(cache_upd.get_mesh(key), mesh)
        self.assertEqual(len(cache_upd), 1)

    # -----------------------------------------------------------------
    # Category 2: InferenceMode Decision Logic
    # -----------------------------------------------------------------

    @_requires_multi_gpu
    def test_needs_dynamic_all_cases(self, device):
        """Test all 8 cases of the needs_dynamic() decision matrix."""
        rank_device = self._rank_device(device)
        self.init_pg(rank_device)
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
                    lambda msg: f"{msg}\nCase {i} failed",
                )

    @_requires_multi_gpu
    def test_NoneGrad_dtensor_grad_metadata_not_derived(self, device):
        rank_device = self._rank_device(device)
        self.init_pg(rank_device)
        mesh = self._make_mesh()

        plain_meta = self._make_tensor_meta()
        dtensor_meta = _DTensorMeta.from_dtensor(
            self._make_dtensor(
                mesh, [Replicate()], requires_grad=True, device=rank_device
            )
        )

        plain_grad_meta, dtensor_grad_meta = _derive_grad_metas(
            (plain_meta, dtensor_meta)
        )
        self.assertIsNotNone(plain_grad_meta)
        self.assertIsNone(dtensor_grad_meta)

    @_requires_multi_gpu
    def test_NoneGrad_dtensor_runtime_send_contract(self, device):
        rank_device = self._rank_device(device)
        self.init_pg(rank_device)
        mesh = self._make_mesh()
        dtensor_grad = self._make_dtensor(
            mesh,
            [Replicate()],
            shape=(4, 8),
            requires_grad=False,
            device=rank_device,
        )
        dtensor_meta = _DTensorMeta.from_dtensor(dtensor_grad)

        stage = PipelineStage(
            torch.nn.Identity(),
            1,
            self.world_size,
            rank_device,
        )
        stage.has_backward = True
        stage.chunks = 1
        stage.grad_send_info = [0]

        stage._stage_meta = _StageMeta(input_grads=(None,))
        stage.bwd_cache[0] = (dtensor_grad,)
        with self.assertRaisesRegex(
            PipeliningMetadataError,
            "DTensor grad metadata cannot be derived",
        ):
            stage.get_bwd_send_ops(0)

        stage._stage_meta = _StageMeta(input_grads=(dtensor_meta,))
        stage.bwd_cache[0] = (None,)
        ops = stage.get_bwd_send_ops(0)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].tensor, torch.zeros_like(dtensor_grad.to_local()))

    # -----------------------------------------------------------------
    # Category 3: Validation Functions
    # -----------------------------------------------------------------

    @_requires_multi_gpu
    def test_validate_metadata(self, device):
        """Test validate_metadata: match, raise, warn, type mismatch."""
        rank_device = self._rank_device(device)
        self.init_pg(rank_device)
        mesh = self._make_mesh()

        # Match → no diffs
        dt = self._make_dtensor(mesh, [Shard(0)], device=rank_device)
        meta = _DTensorMeta.from_dtensor(dt)
        self.assertEqual(validate_metadata("test", meta, dt), [])

        # Mismatch with raise_on_mismatch
        other = self._make_dtensor(mesh, [Replicate()], device=rank_device)
        with self.assertRaises(PipeliningMetadataError):
            validate_metadata("test", meta, other, raise_on_mismatch=True)

        # Mismatch with warn_on_mismatch
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            diffs = validate_metadata("test", meta, other, warn_on_mismatch=True)
            self.assertTrue(len(diffs) > 0)
            self.assertTrue(any("Metadata mismatch" in str(x.message) for x in w))

        # Type mismatch: _DTensorMeta vs plain tensor
        plain = torch.randn(8, 16, device=rank_device)
        diffs = validate_metadata("test", meta, plain)
        self.assertTrue(any("type" in d for d in diffs))

    @_requires_multi_gpu
    def test_validate_tensors_metadata(self, device):
        """Test validate_tensors_metadata: batch validation, length mismatch, None handling."""
        rank_device = self._rank_device(device)
        self.init_pg(rank_device)
        mesh = self._make_mesh()

        dt = self._make_dtensor(mesh, [Shard(0)], device=rank_device)
        meta = _DTensorMeta.from_dtensor(dt)

        # Match
        validate_tensors_metadata("test", (meta,), (dt,))

        # Length mismatch raises
        with self.assertRaises(PipeliningMetadataError):
            validate_tensors_metadata("test", (meta, meta), (dt,))

        # Length mismatch warns (raise disabled)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tensors_metadata(
                "test",
                (meta, meta),
                (dt,),
                raise_on_mismatch=False,
                warn_on_mismatch=True,
            )
            self.assertTrue(len(w) > 0)

        # None-None pair passes
        validate_tensors_metadata("test", (None,), (None,))

        # One-None mismatch raises (expected has metadata but actual is None)
        with self.assertRaises(PipeliningMetadataError):
            validate_tensors_metadata("test", (meta,), (None,))

    @_requires_multi_gpu
    def test_validate_static_arg_grad_correspondence(self, device):
        """Test DTensor↔grad correspondence: all valid/invalid combos."""
        rank_device = self._rank_device(device)
        self.init_pg(rank_device)
        mesh = self._make_mesh()

        dt_grad = self._make_dtensor(
            mesh, [Shard(0)], requires_grad=True, device=rank_device
        )
        dt_nograd = self._make_dtensor(
            mesh, [Shard(0)], requires_grad=False, device=rank_device
        )
        dt_grad2 = self._make_dtensor(
            mesh, [Replicate()], requires_grad=True, device=rank_device
        )
        plain = torch.randn(8, 16, device=rank_device)

        # DTensor(requires_grad=True) + DTensor grad → OK
        validate_static_arg_grad_correspondence(
            0, (dt_grad,), (dt_grad2,), is_input=True
        )

        # DTensor(requires_grad=False) + None grad → OK
        validate_static_arg_grad_correspondence(0, (dt_nograd,), (None,), is_input=True)

        # Plain tensor + None grad → OK
        validate_static_arg_grad_correspondence(0, (plain,), (None,), is_input=True)

        # DTensor(requires_grad=True) + None grad → OK (boundary with no gradient)
        validate_static_arg_grad_correspondence(0, (dt_grad,), (None,), is_input=True)

        # DTensor(requires_grad=True) + plain tensor grad → ERROR
        with self.assertRaises(PipeliningMetadataError):
            validate_static_arg_grad_correspondence(
                0, (dt_grad,), (plain,), is_input=True
            )

        # Length mismatch → ERROR
        with self.assertRaises(PipeliningMetadataError):
            validate_static_arg_grad_correspondence(
                0, (dt_grad, dt_grad), (dt_grad2,), is_input=False
            )

    # -----------------------------------------------------------------
    # Category 4: Helper Functions
    # -----------------------------------------------------------------

    @_requires_multi_gpu
    def test_extract_tensor_metas(self, device):
        """Test extract_tensor_metas: plain, DTensor, None handling."""
        rank_device = self._rank_device(device)
        self.init_pg(rank_device)
        mesh = self._make_mesh()

        dt = self._make_dtensor(mesh, [Shard(0)], device=rank_device)
        plain = torch.randn(4, 4, device=rank_device)

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
    def test_to_local_if_dtensor(self, device):
        """Test to_local_if_dtensor: DTensor→local, plain passthrough, detach."""
        rank_device = self._rank_device(device)
        self.init_pg(rank_device)
        mesh = self._make_mesh()

        dt = self._make_dtensor(
            mesh, [Shard(0)], requires_grad=True, device=rank_device
        )
        plain = torch.randn(4, 4, device=rank_device, requires_grad=True)

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
    def test_validate_and_normalize_to_tuple(self, device):
        """Test validate_and_normalize_to_tuple: single, tuple, list, None, errors."""
        rank_device = self._rank_device(device)
        self.init_pg(rank_device)
        t = torch.randn(4, 4, device=rank_device)

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
    # Category 5: Microbatch DTensor Split & Merge
    # -----------------------------------------------------------------

    @_requires_multi_gpu
    def test_split_tensor_dtensor_preserves_placements(self, device):
        """Test _split_tensor splits a DTensor into chunks that are still DTensors
        with the same placements, correct local shapes, and matching data."""
        rank_device = self._rank_device(device)
        self.init_pg(rank_device)
        mesh = self._make_mesh()
        num_chunks = 2
        split_dim = 0
        shape = (8, 16)

        for placements in [[Shard(0)], [Replicate()]]:
            dt = self._make_dtensor(mesh, placements, shape=shape, device=rank_device)
            spec = TensorChunkSpec(split_dim)
            chunks = _split_tensor(dt, spec, num_chunks)

            self.assertEqual(len(chunks), num_chunks)
            for i, chunk in enumerate(chunks):
                # Each chunk must be a DTensor with preserved placements
                self.assertIsInstance(
                    chunk, DTensor, lambda msg: f"{msg}\nchunk {i} is not a DTensor"
                )
                self.assertEqual(
                    chunk.placements,
                    tuple(placements),
                    lambda msg: f"{msg}\nchunk {i} placements differ",
                )
                # Each chunk must have the same device mesh
                self.assertIs(chunk.device_mesh, mesh)

            # Local shard sizes along split_dim must sum to the original
            local_split_sizes = [c._local_tensor.size(split_dim) for c in chunks]
            self.assertEqual(
                sum(local_split_sizes),
                dt._local_tensor.size(split_dim),
            )

            # Full tensor data must be preserved: cat the chunks and compare
            cat_fn = torch.cat([c._local_tensor for c in chunks], dim=split_dim)
            self.assertTrue(
                torch.equal(cat_fn, dt._local_tensor),
                "Split chunks do not reconstruct the original local tensor",
            )

    @_requires_multi_gpu
    def test_merge_chunks_dtensor_roundtrip(self, device):
        """Test that split → merge is a roundtrip: merge_chunks reconstructs
        the original DTensor data and preserves placements."""
        rank_device = self._rank_device(device)
        self.init_pg(rank_device)
        mesh = self._make_mesh()
        num_chunks = 2
        split_dim = 0
        shape = (8, 16)

        for placements in [[Shard(0)], [Replicate()]]:
            dt = self._make_dtensor(mesh, placements, shape=shape, device=rank_device)
            chunk_spec = TensorChunkSpec(split_dim)

            # Split into chunks
            chunks = list(_split_tensor(dt, chunk_spec, num_chunks))

            # Merge chunks back
            # merge_chunks expects a list of "outputs", each output is a tuple.
            # For a single-tensor case, wrap each chunk as a 1-tuple.
            merged = merge_chunks(
                [(chunk,) for chunk in chunks],
                (chunk_spec,),
            )
            # merged is a tuple of tensors (mirrors the tuple structure)
            merged_dt = merged[0]

            self.assertIsInstance(merged_dt, DTensor)
            self.assertEqual(merged_dt.placements, tuple(placements))
            self.assertIs(merged_dt.device_mesh, mesh)

            # Data must match the original
            self.assertTrue(
                torch.equal(merged_dt._local_tensor, dt._local_tensor),
                "Merged DTensor local data differs from original",
            )
            self.assertEqual(merged_dt.shape, dt.shape)

    @_requires_multi_gpu
    def test_merge_chunks_dtensor_placement_mismatch_raises(self, device):
        """Test that merge_chunks raises when chunk placements don't match."""
        rank_device = self._rank_device(device)
        self.init_pg(rank_device)
        mesh = self._make_mesh()
        shape = (8, 16)

        dt_shard = self._make_dtensor(mesh, [Shard(0)], shape=shape, device=rank_device)
        dt_rep = self._make_dtensor(
            mesh, [Replicate()], shape=shape, device=rank_device
        )

        chunk_spec = TensorChunkSpec(0)
        with self.assertRaises(AssertionError) as ctx:
            merge_chunks(
                [(dt_shard,), (dt_rep,)],
                (chunk_spec,),
            )
        self.assertIn("placement mismatch", str(ctx.exception))

    @_requires_multi_gpu
    def test_merge_chunks_dtensor_mixed_types_raises(self, device):
        """Test that merge_chunks raises when mixing DTensors and plain tensors."""
        rank_device = self._rank_device(device)
        self.init_pg(rank_device)
        mesh = self._make_mesh()
        shape = (4, 8)

        dt = self._make_dtensor(mesh, [Shard(0)], shape=shape, device=rank_device)
        plain = torch.randn(*shape, device=rank_device)

        chunk_spec = TensorChunkSpec(0)
        with self.assertRaises(AssertionError) as ctx:
            merge_chunks(
                [(dt,), (plain,)],
                (chunk_spec,),
            )
        self.assertIn("mix", str(ctx.exception))

    @_requires_multi_gpu
    def test_split_target_preserves_dtensor_placements(self, device):
        """Regression test: schedules' target-split path must split DTensor
        targets via _split_tensor (not torch.tensor_split) so that Shard
        placements survive (no implicit all-gather to Replicate) and so that
        uneven splits agree across ranks. If the target-split path is reverted
        to torch.tensor_split, Shard(0) placements degrade to Replicate() and
        this test fails."""
        from torch.distributed.pipelining.schedules import _TARGET_CHUNK_SPEC

        rank_device = self._rank_device(device)
        self.init_pg(rank_device)
        mesh = self._make_mesh()

        # Even-split and uneven-split cases; Shard(0) and Replicate() placements.
        for n_microbatches, batch in [(2, 8), (3, 7)]:
            for placements in ([Shard(0)], [Replicate()]):
                dt = self._make_dtensor(
                    mesh, placements, shape=(batch,), device=rank_device
                )
                chunks = list(_split_tensor(dt, _TARGET_CHUNK_SPEC, n_microbatches))

                self.assertEqual(len(chunks), n_microbatches)
                total = 0
                for i, chunk in enumerate(chunks):
                    self.assertIsInstance(
                        chunk, DTensor, lambda msg: f"{msg}\nchunk {i} not DTensor"
                    )
                    self.assertEqual(
                        chunk.placements,
                        tuple(placements),
                        lambda msg: f"{msg}\nchunk {i} placements changed from {placements} "
                        f"to {chunk.placements}",
                    )
                    total += chunk.shape[0]
                self.assertEqual(total, batch)


# =============================================================================
# Run Tests
# =============================================================================

instantiate_device_type_tests(
    TestDTensorPPUnitTests,
    globals(),
    except_for=["cpu"],
    allow_xpu=True,
)


if __name__ == "__main__":
    run_tests()
