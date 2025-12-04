# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from collections.abc import Callable

import torch
from torch.distributed.tensor import (
    create_distributed_block_mask,
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Placement,
    Replicate,
    Shard,
)
from torch.nn.attention.flex_attention import AuxRequest, BlockMask, flex_attention
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


# Type alias for mask modification functions
MaskModFunc = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]


def _causal_mask(
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    """Causal mask that prevents attention to future tokens."""
    return q_idx >= kv_idx


DOC_LEN: int = 30


def _generate_block_causal_batch(
    global_bs: int, global_query_tokens: int, device_type: str
) -> torch.Tensor:
    total_elements = global_bs * global_query_tokens
    batch = torch.arange(total_elements, device=device_type) % DOC_LEN
    batch = batch.reshape(global_bs, global_query_tokens)
    return batch


def _create_block_causal_mask_mod(
    global_bs: int,
    global_query_tokens: int,
    device_type: str,
) -> MaskModFunc:
    """Helper to create a block_causal_mask function for tests.

    Creates a mask function for packed sequences where each has DOC_LEN
    tokens and the last token is the EOS token (DOC_LEN - 1).

    Args:
        global_bs: Global batch size
        global_query_tokens: Global query sequence length
        device_type: Device type (e.g., 'cuda', 'cpu')

    Returns:
        block_causal_mask: Mask function for block-causal attention
    """
    batch = _generate_block_causal_batch(global_bs, global_query_tokens, device_type)
    # Pre-compute seq_idx for the entire batch
    mask = batch == (DOC_LEN - 1)
    mask[:, -1] = True
    acc_mask = torch.cumsum(torch.where(mask, 1, 0), dim=1)
    seq_idx = torch.zeros_like(acc_mask, dtype=torch.int32)
    seq_idx[:, 1:] = acc_mask[:, :-1]

    def block_causal_mask(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        return (seq_idx[b, q_idx] == seq_idx[b, kv_idx]) & (q_idx >= kv_idx)

    return block_causal_mask


def _create_distributed_block_causal_mask_mod(
    global_bs: int,
    global_query_tokens: int,
    device_type: str,
    device_mesh: DeviceMesh,
    batch_placement: list[Placement],
) -> MaskModFunc:
    """Helper to create a block_causal_mask function for tests.

    Creates a mask function for packed sequences where each has DOC_LEN
    tokens and the last token is the EOS token (DOC_LEN - 1).

    Args:
        global_bs: Global batch size
        global_query_tokens: Global query sequence length
        device_type: Device type (e.g., 'cuda', 'cpu')
        device_mesh: DeviceMesh for distributed execution.
        batch_placement: Placement for the batch tensor

    Returns:
        block_causal_mask: Mask function for block-causal attention
    """
    batch = _generate_block_causal_batch(global_bs, global_query_tokens, device_type)
    batch = distribute_tensor(batch, device_mesh, batch_placement)

    # Pre-compute seq_idx for the entire batch
    mask = batch._local_tensor == (DOC_LEN - 1)
    mask[:, -1] = True
    acc_mask = torch.cumsum(torch.where(mask, 1, 0), dim=1)
    seq_idx = torch.zeros_like(acc_mask, dtype=torch.int32)
    seq_idx[:, 1:] = acc_mask[:, :-1]

    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    _, global_offset = compute_local_shape_and_global_offset(
        batch.shape,
        device_mesh,
        batch_placement,
    )
    b_offset_int, *_ = global_offset

    def block_causal_mask(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        b_offset = torch.full_like(b, b_offset_int)
        local_b = b - b_offset
        return (seq_idx[local_b, q_idx] == seq_idx[local_b, kv_idx]) & (q_idx >= kv_idx)

    return block_causal_mask


class MyTestClass(DTensorTestBase):
    # Global dimension constants for all tests
    GLOBAL_BS: int = 16  # Batch size (divisible by world_size=4)
    GLOBAL_NHEADS: int = 8  # Number of attention heads (divisible by world_size=4)
    GLOBAL_QUERY_TOKENS: int = 1024  # Query sequence length (divisible by world_size=4)
    GLOBAL_CONTEXT_TOKENS: int = 1024  # KV sequence length (divisible by world_size=4)
    DIM: int = 32  # Head dimension

    def setUp(self) -> None:
        super().setUp()
        torch.use_deterministic_algorithms(True)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(42)

    @property
    def device_mesh(self) -> DeviceMesh:
        """Helper function to create a DeviceMesh for testing."""
        if getattr(self, "my_device_mesh", None) is None:
            self.my_device_mesh = self.build_device_mesh()
        return self.my_device_mesh

    def _create_block_causal_mask_mod(
        self,
        device_mesh: DeviceMesh | None = None,
        batch_placement: list[Placement] | None = None,
    ) -> MaskModFunc:
        """Helper to create a mask function for block_causal_mask tests.

        Returns the mask function.
        """
        if device_mesh is None:
            return _create_block_causal_mask_mod(
                self.GLOBAL_BS,
                self.GLOBAL_QUERY_TOKENS,
                self.device_type,
            )
        else:
            return _create_distributed_block_causal_mask_mod(
                self.GLOBAL_BS,
                self.GLOBAL_QUERY_TOKENS,
                self.device_type,
                device_mesh,
                batch_placement,
            )


class DistFlexAttentionTest(MyTestClass):
    def _test_flex_attention_with_placement(
        self,
        device_mesh: DeviceMesh,
        placement: list[Placement],
        mask_fn: MaskModFunc | None = None,
        full_mask_fn: MaskModFunc | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Helper function to test FlexAttention with different placements.

        Args:
            placement: List of Placement objects (e.g., [Shard(0)], [Replicate()], [Shard(2)])
            mask_fn: Mask function to use (defaults to _causal_mask if None)
            dtype: Data type for tensors (default: torch.bfloat16)
        """
        # Use default causal mask if none provided
        if mask_fn is None:
            mask_fn = _causal_mask
        if full_mask_fn is None:
            full_mask_fn = _causal_mask

        # Create input tensors with gradients enabled
        q = torch.rand(
            (self.GLOBAL_BS, self.GLOBAL_NHEADS, self.GLOBAL_QUERY_TOKENS, self.DIM),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )
        k = torch.rand(
            (self.GLOBAL_BS, self.GLOBAL_NHEADS, self.GLOBAL_CONTEXT_TOKENS, self.DIM),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )
        v = torch.rand(
            (self.GLOBAL_BS, self.GLOBAL_NHEADS, self.GLOBAL_CONTEXT_TOKENS, self.DIM),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )

        # Create distributed block mask
        local_mask = create_distributed_block_mask(
            mask_fn,
            self.GLOBAL_BS,
            self.GLOBAL_NHEADS,
            self.GLOBAL_QUERY_TOKENS,
            self.GLOBAL_CONTEXT_TOKENS,
            device_mesh,
            placement,
        )

        # Distribute tensors with the specified placement
        q_dt = distribute_tensor(q, device_mesh, placement)
        k_dt = distribute_tensor(k, device_mesh, placement)
        v_dt = distribute_tensor(v, device_mesh, placement)

        # Compile flex_attention for better performance
        # TODO: we cannot turn on torch.compile yet.
        # Error message:
        #   Expected metadata: (DTensorSpec(...)), expected type: <class 'torch.distributed.tensor.DTensor'>
        #   Runtime metadata: None, runtime type: <class 'torch.Tensor'>
        #   shape: torch.Size([16, 8, 1024])
        #   To fix this, your tensor subclass must implement the dunder method __force_to_same_metadata__.
        # This is very likely to be the case where grad_logsumexp is a plain tensor.
        flex_attention_compiled = flex_attention

        # Run distributed FlexAttention
        out_dt, aux_dt = flex_attention_compiled(
            q_dt,
            k_dt,
            v_dt,
            block_mask=local_mask,
            return_aux=AuxRequest(lse=True, max_scores=True),
        )
        local_out = out_dt
        local_logsumexp = aux_dt.lse
        local_max_scores = aux_dt.max_scores

        # Verify outputs are DTensors
        self.assertIsInstance(local_out, DTensor)
        self.assertIsInstance(local_logsumexp, DTensor)
        self.assertIsInstance(local_max_scores, DTensor)

        # Verify placements based on the input placement
        if isinstance(placement[0], Shard):
            shard_dim = placement[0].dim
            self.assertTrue(local_out.placements[0].is_shard(dim=shard_dim))
            self.assertTrue(local_logsumexp.placements[0].is_shard(dim=shard_dim))
            self.assertTrue(local_max_scores.placements[0].is_shard(dim=shard_dim))
        elif isinstance(placement[0], Replicate):
            self.assertTrue(local_out.placements[0].is_replicate())
            self.assertTrue(local_logsumexp.placements[0].is_replicate())
            self.assertTrue(local_max_scores.placements[0].is_replicate())

        # Create regular (non-DTensor) block mask for reference computation
        # Use the cached compiled version from create_distributed_block_mask
        full_mask = create_distributed_block_mask.create_block_mask(
            full_mask_fn,
            self.GLOBAL_BS,
            self.GLOBAL_NHEADS,
            self.GLOBAL_QUERY_TOKENS,
            self.GLOBAL_CONTEXT_TOKENS,
        )

        # Get full tensors and run reference computation with regular tensors
        full_q = q_dt.full_tensor().detach()
        full_k = k_dt.full_tensor().detach()
        full_v = v_dt.full_tensor().detach()
        full_q.requires_grad = True
        full_k.requires_grad = True
        full_v.requires_grad = True
        full_out, full_aux = flex_attention_compiled(
            full_q,
            full_k,
            full_v,
            block_mask=full_mask,
            return_aux=AuxRequest(lse=True, max_scores=True),
        )
        full_logsumexp = full_aux.lse
        full_max_scores = full_aux.max_scores

        # Verify correctness by comparing DTensor results with regular tensor results
        torch.testing.assert_close(local_out.full_tensor(), full_out)
        torch.testing.assert_close(local_logsumexp.full_tensor(), full_logsumexp)
        torch.testing.assert_close(local_max_scores.full_tensor(), full_max_scores)

        # Test backward pass
        local_out.sum().backward()
        full_out.sum().backward()

        # Compare gradients
        torch.testing.assert_close(q_dt.grad.full_tensor(), full_q.grad)
        torch.testing.assert_close(k_dt.grad.full_tensor(), full_k.grad)
        torch.testing.assert_close(v_dt.grad.full_tensor(), full_v.grad)

    @with_comms
    def test_flex_attention_replicate(self) -> None:
        """Test FlexAttention with replicated placement."""
        device_mesh = self.build_device_mesh()
        self._test_flex_attention_with_placement(
            device_mesh, [Replicate()], dtype=torch.bfloat16
        )

    @with_comms
    def test_flex_attention_shard0(self) -> None:
        """Test FlexAttention with batch dimension sharding (Shard(0))."""
        device_mesh = self.build_device_mesh()
        self._test_flex_attention_with_placement(
            device_mesh, [Shard(0)], dtype=torch.bfloat16
        )

    @with_comms
    def test_flex_attention_shard2(self) -> None:
        """Test FlexAttention with sequence dimension sharding (Shard(2))."""
        device_mesh = self.build_device_mesh()
        self._test_flex_attention_with_placement(
            device_mesh, [Shard(2)], dtype=torch.float32
        )

    @with_comms
    def test_flex_attention_block_causal_replicate(self) -> None:
        """Test FlexAttention with block_causal_mask and Replicate
        placement.
        """
        device_mesh = self.build_device_mesh()
        mask_fn = self._create_block_causal_mask_mod(device_mesh, [Replicate()])
        full_mask_fn = self._create_block_causal_mask_mod()
        self._test_flex_attention_with_placement(
            device_mesh,
            [Replicate()],
            mask_fn=mask_fn,
            full_mask_fn=full_mask_fn,
            dtype=torch.bfloat16,
        )

    @with_comms
    def test_flex_attention_block_causal_shard0(self) -> None:
        """Test FlexAttention with block_causal_mask and Shard(0)
        placement.
        """
        device_mesh = self.build_device_mesh()
        mask_fn = self._create_block_causal_mask_mod(device_mesh, [Shard(0)])
        full_mask_fn = self._create_block_causal_mask_mod()
        self._test_flex_attention_with_placement(
            device_mesh,
            [Shard(0)],
            mask_fn=mask_fn,
            full_mask_fn=full_mask_fn,
            dtype=torch.bfloat16,
        )

    @with_comms
    def test_flex_attention_block_causal_shard2(self) -> None:
        """Test FlexAttention with block_causal_mask and Shard(2)
        placement.
        """
        device_mesh = self.build_device_mesh()
        # For CP/Shard2, the batch is actually sharded on dim 1, because that's
        # the sequence dimension. The input batch doesn't have the head dimension.
        mask_fn = self._create_block_causal_mask_mod(device_mesh, [Replicate()])
        full_mask_fn = self._create_block_causal_mask_mod()
        self._test_flex_attention_with_placement(
            device_mesh,
            [Shard(2)],
            mask_fn=mask_fn,
            full_mask_fn=full_mask_fn,
            dtype=torch.bfloat16,
        )


class DistBlockMaskTest(MyTestClass):
    """Test suite for create_distributed_block_mask functionality."""

    def _test_create_distributed_block_mask(
        self,
        placement: list[Placement],
        mask_fn: MaskModFunc | None = None,
        full_mask_fn: MaskModFunc | None = None,
        expected_kv_placement: Placement | None = None,
        expected_q_num_blocks_placement: Placement | None = None,
        expected_q_indices_placement: Placement | None = None,
        verify_values: bool = True,
    ) -> None:
        """Helper function to test create_distributed_block_mask with different placements.

        Args:
            placement: List of Placement objects (e.g., [Shard(0)], [Replicate()], [Shard(2)])
            mask_fn: Mask function to use (defaults to _causal_mask if None)
            expected_kv_placement: Expected placement for kv fields. If None, uses placement[0]
            expected_q_num_blocks_placement: Expected placement for q_num_blocks. If None, uses expected_kv_placement
            expected_q_indices_placement: Expected placement for q_indices. If None, uses expected_kv_placement
            verify_values: Whether to verify tensor values by comparing with full_block_mask (default: True)
        """
        device_mesh = self.build_device_mesh()

        # Use default causal mask if none provided
        if mask_fn is None:
            assert full_mask_fn is None
            mask_fn = _causal_mask
            full_mask_fn = _causal_mask

        # Create distributed block mask
        block_mask = create_distributed_block_mask(
            mask_fn,
            self.GLOBAL_BS,
            self.GLOBAL_NHEADS,
            self.GLOBAL_QUERY_TOKENS,
            self.GLOBAL_CONTEXT_TOKENS,
            device_mesh,
            placement,
        )

        # Create reference full block mask if needed for value verification
        full_block_mask = None
        if verify_values:
            full_block_mask = create_distributed_block_mask.create_block_mask(
                full_mask_fn,
                B=self.GLOBAL_BS,
                H=self.GLOBAL_NHEADS,
                Q_LEN=self.GLOBAL_QUERY_TOKENS,
                KV_LEN=self.GLOBAL_CONTEXT_TOKENS,
                device=device_mesh.device_type,
            )

        # Default expected placements
        if expected_kv_placement is None:
            expected_kv_placement = placement[0]

        # Verify placements (and optionally values)
        self._verify_block_mask_placements(
            block_mask,
            expected_kv_placement,
            expected_q_num_blocks_placement=expected_q_num_blocks_placement,
            expected_q_indices_placement=expected_q_indices_placement,
            full_block_mask=full_block_mask,
        )

    def _verify_block_mask_placements(
        self,
        block_mask: BlockMask,
        expected_kv_placement: Placement,
        expected_q_num_blocks_placement: Placement | None = None,
        expected_q_indices_placement: Placement | None = None,
        full_block_mask: BlockMask | None = None,
    ) -> None:
        """Verify all block mask components have expected placements.

        Args:
            block_mask: BlockMask to verify
            expected_kv_placement: Expected placement for kv_num_blocks, kv_indices,
                                  full_kv_num_blocks, full_kv_indices
            expected_q_num_blocks_placement: Expected placement for q_num_blocks, full_q_num_blocks.
                                            If None, uses expected_kv_placement
            expected_q_indices_placement: Expected placement for q_indices, full_q_indices.
                                         If None, uses expected_kv_placement
            full_block_mask: Optional reference BlockMask (non-distributed) to compare
                           tensor values against using full_tensor()
        """
        if expected_q_num_blocks_placement is None:
            expected_q_num_blocks_placement = expected_kv_placement
        if expected_q_indices_placement is None:
            expected_q_indices_placement = expected_kv_placement

        # Verify kv_num_blocks
        if block_mask.kv_num_blocks is not None:
            self.assertIsInstance(block_mask.kv_num_blocks, DTensor)
            self.assertEqual(
                block_mask.kv_num_blocks.placements[0], expected_kv_placement
            )
            if (
                full_block_mask is not None
                and full_block_mask.kv_num_blocks is not None
            ):
                gathered_tensor = block_mask.kv_num_blocks.full_tensor()
                self.assertTrue(
                    torch.equal(gathered_tensor, full_block_mask.kv_num_blocks),
                    "kv_num_blocks values don't match after gathering",
                )

        # Verify kv_indices
        if block_mask.kv_indices is not None:
            self.assertIsInstance(block_mask.kv_indices, DTensor)
            self.assertEqual(block_mask.kv_indices.placements[0], expected_kv_placement)
            if full_block_mask is not None and full_block_mask.kv_indices is not None:
                gathered_tensor = block_mask.kv_indices.full_tensor()
                self.assertTrue(
                    torch.equal(gathered_tensor, full_block_mask.kv_indices),
                    "kv_indices values don't match after gathering",
                )

        # Verify full_kv_num_blocks
        if block_mask.full_kv_num_blocks is not None:
            self.assertIsInstance(block_mask.full_kv_num_blocks, DTensor)
            self.assertEqual(
                block_mask.full_kv_num_blocks.placements[0], expected_kv_placement
            )
            if (
                full_block_mask is not None
                and full_block_mask.full_kv_num_blocks is not None
            ):
                gathered_tensor = block_mask.full_kv_num_blocks.full_tensor()
                self.assertTrue(
                    torch.equal(gathered_tensor, full_block_mask.full_kv_num_blocks),
                    "full_kv_num_blocks values don't match after gathering",
                )

        # Verify full_kv_indices
        if block_mask.full_kv_indices is not None:
            self.assertIsInstance(block_mask.full_kv_indices, DTensor)
            self.assertEqual(
                block_mask.full_kv_indices.placements[0], expected_kv_placement
            )
            if (
                full_block_mask is not None
                and full_block_mask.full_kv_indices is not None
            ):
                gathered_tensor = block_mask.full_kv_indices.full_tensor()
                self.assertTrue(
                    torch.equal(gathered_tensor, full_block_mask.full_kv_indices),
                    "full_kv_indices values don't match after gathering",
                )

        # Verify q_num_blocks
        if block_mask.q_num_blocks is not None:
            self.assertIsInstance(block_mask.q_num_blocks, DTensor)
            self.assertEqual(
                block_mask.q_num_blocks.placements[0], expected_q_num_blocks_placement
            )
            if full_block_mask is not None and full_block_mask.q_num_blocks is not None:
                gathered_tensor = block_mask.q_num_blocks.full_tensor()
                self.assertTrue(
                    torch.equal(gathered_tensor, full_block_mask.q_num_blocks),
                    "q_num_blocks values don't match after gathering",
                )

        # Verify q_indices
        if block_mask.q_indices is not None:
            self.assertIsInstance(block_mask.q_indices, DTensor)
            self.assertEqual(
                block_mask.q_indices.placements[0], expected_q_indices_placement
            )
            if full_block_mask is not None and full_block_mask.q_indices is not None:
                gathered_tensor = block_mask.q_indices.full_tensor()
                self.assertTrue(
                    torch.equal(gathered_tensor, full_block_mask.q_indices),
                    "q_indices values don't match after gathering",
                )

        # Verify full_q_num_blocks
        if block_mask.full_q_num_blocks is not None:
            self.assertIsInstance(block_mask.full_q_num_blocks, DTensor)
            self.assertEqual(
                block_mask.full_q_num_blocks.placements[0],
                expected_q_num_blocks_placement,
            )
            if (
                full_block_mask is not None
                and full_block_mask.full_q_num_blocks is not None
            ):
                gathered_tensor = block_mask.full_q_num_blocks.full_tensor()
                self.assertTrue(
                    torch.equal(gathered_tensor, full_block_mask.full_q_num_blocks),
                    "full_q_num_blocks values don't match after gathering",
                )

        # Verify full_q_indices
        if block_mask.full_q_indices is not None:
            self.assertIsInstance(block_mask.full_q_indices, DTensor)
            self.assertEqual(
                block_mask.full_q_indices.placements[0], expected_q_indices_placement
            )
            if (
                full_block_mask is not None
                and full_block_mask.full_q_indices is not None
            ):
                gathered_tensor = block_mask.full_q_indices.full_tensor()
                self.assertTrue(
                    torch.equal(gathered_tensor, full_block_mask.full_q_indices),
                    "full_q_indices values don't match after gathering",
                )

    @with_comms
    def test_create_distributed_block_mask_replicate(self) -> None:
        """Test create_distributed_block_mask with Replicate placement."""
        self._test_create_distributed_block_mask([Replicate()])

    @with_comms
    def test_create_distributed_block_mask_shard0(self) -> None:
        """Test create_distributed_block_mask with Shard(0) placement."""
        self._test_create_distributed_block_mask([Shard(0)])

    @with_comms
    def test_create_distributed_block_mask_shard2(self) -> None:
        """Test create_distributed_block_mask with Shard(2) placement (context parallel)."""
        # Shard(2) uses a sharded mask_mod with local dimensions, so tensor values
        # will differ from a reference block mask created with unsharded mask_mod
        self._test_create_distributed_block_mask(
            [Shard(2)],
            expected_kv_placement=Shard(2),
            expected_q_num_blocks_placement=Replicate(),
            expected_q_indices_placement=Shard(3),
            verify_values=False,
        )

    @with_comms
    def test_create_distributed_block_mask_block_causal_replicate(self) -> None:
        """Test create_distributed_block_mask with block_causal_mask
        and Replicate placement.
        """
        mask_fn = self._create_block_causal_mask_mod(
            self.build_device_mesh(), [Replicate()]
        )
        self._test_create_distributed_block_mask(
            [Replicate()], mask_fn=mask_fn, full_mask_fn=mask_fn
        )

    @with_comms
    def test_create_distributed_block_mask_block_causal_shard0(self) -> None:
        """Test create_distributed_block_mask with block_causal_mask
        and Shard(0) placement.
        """
        mask_fn = self._create_block_causal_mask_mod(
            self.build_device_mesh(), [Shard(0)]
        )
        full_mask_fn = self._create_block_causal_mask_mod()
        self._test_create_distributed_block_mask(
            [Shard(0)], mask_fn=mask_fn, full_mask_fn=full_mask_fn
        )

    @with_comms
    def test_create_distributed_block_mask_block_causal_shard2(self) -> None:
        """Test create_distributed_block_mask with block_causal_mask
        and Shard(2) placement.
        """
        mask_fn = self._create_block_causal_mask_mod(
            self.build_device_mesh(), [Replicate()]
        )
        # Shard(2) uses a sharded mask_mod with local dimensions, so tensor values
        # will differ from a reference block mask created with unsharded mask_mod
        self._test_create_distributed_block_mask(
            [Shard(2)],
            mask_fn=mask_fn,
            full_mask_fn=mask_fn,
            expected_kv_placement=Shard(2),
            expected_q_num_blocks_placement=Replicate(),
            expected_q_indices_placement=Shard(3),
            verify_values=False,
        )

    @with_comms
    def test_create_distributed_block_mask_invalid_sharding(self) -> None:
        """Test error handling for unsupported sharding dimensions."""
        device_mesh = self.build_device_mesh()

        # Test sharding on head dimension (dim=1) - should raise error
        with self.assertRaisesRegex(
            ValueError, "Sharding on head dimension is not supported"
        ):
            create_distributed_block_mask(
                _causal_mask,
                self.GLOBAL_BS,
                self.GLOBAL_NHEADS,
                self.GLOBAL_QUERY_TOKENS,
                self.GLOBAL_CONTEXT_TOKENS,
                device_mesh,
                [Shard(1)],
            )

        # Test sharding on KV sequence dimension (dim=3) - should raise error
        with self.assertRaisesRegex(
            ValueError, "Sharding on KV sequence dimension is not supported"
        ):
            create_distributed_block_mask(
                _causal_mask,
                self.GLOBAL_BS,
                self.GLOBAL_NHEADS,
                self.GLOBAL_QUERY_TOKENS,
                self.GLOBAL_CONTEXT_TOKENS,
                device_mesh,
                [Shard(3)],
            )

        # Test sharding on None dimension (broadcasting) - should raise error
        with self.assertRaisesRegex(ValueError, "value is set to None"):
            create_distributed_block_mask(
                _causal_mask,
                None,  # B is None but we're trying to shard on it
                self.GLOBAL_NHEADS,
                self.GLOBAL_QUERY_TOKENS,
                self.GLOBAL_CONTEXT_TOKENS,
                device_mesh,
                [Shard(0)],
            )


if __name__ == "__main__":
    run_tests()
