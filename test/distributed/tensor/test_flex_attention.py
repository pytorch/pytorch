# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed.tensor import (
    create_distributed_block_mask,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.nn.attention.flex_attention import AuxRequest, flex_attention
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


def _causal_mask(
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    """Causal mask that prevents attention to future tokens."""
    return q_idx >= kv_idx


class DistFlexAttentionTest(DTensorTestBase):
    # Global dimension constants for all tests
    GLOBAL_BS = 16  # Batch size (divisible by world_size=4)
    GLOBAL_NHEADS = 8  # Number of attention heads (divisible by world_size=4)
    GLOBAL_QUERY_TOKENS = 512  # Query sequence length (divisible by world_size=4)
    GLOBAL_CONTEXT_TOKENS = 1024  # KV sequence length (divisible by world_size=4)
    DIM = 32  # Head dimension

    @with_comms
    def test_flex_attention_shard0(self):
        """Test FlexAttention with batch dimension sharding (Shard(0))."""
        device_mesh = self.build_device_mesh()
        dtype = torch.bfloat16

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

        # Create block mask for distributed execution
        local_mask = create_distributed_block_mask(
            _causal_mask,
            self.GLOBAL_BS,
            self.GLOBAL_NHEADS,
            self.GLOBAL_QUERY_TOKENS,
            self.GLOBAL_CONTEXT_TOKENS,
            device_mesh,
            [Shard(0)],
        )

        # Distribute tensors with Shard(0)
        placement = [Shard(0)]
        q_dt = distribute_tensor(q, device_mesh, placement)
        k_dt = distribute_tensor(k, device_mesh, placement)
        v_dt = distribute_tensor(v, device_mesh, placement)

        # Compile flex_attention for better performance

        # TODO: we cannot turn on torch.compile yet.
        # Error message:
        #   Expected metadata: (DTensorSpec(...)), expected type: <class 'torch.distributed.tensor.DTensor'>
        #   Runtime metadata: None, runtime type: <class 'torch.Tensor'>
        #   shape: torch.Size([16, 8, 512])
        #   To fix this, your tensor subclass must implement the dunder method __force_to_same_metadata__.

        # This is very likely to be the case where grad_logsumexp is a plain tensor.
        flex_attention_compiled = torch.compile(flex_attention)
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

        # Verify outputs are DTensors with correct placements
        self.assertIsInstance(local_out, DTensor)
        self.assertIsInstance(local_logsumexp, DTensor)
        self.assertIsInstance(local_max_scores, DTensor)
        self.assertTrue(local_out.placements[0].is_shard(dim=0))
        self.assertTrue(local_logsumexp.placements[0].is_shard(dim=0))
        self.assertTrue(local_max_scores.placements[0].is_shard(dim=0))

        # Create regular (non-DTensor) block mask for reference computation
        # Use the cached compiled version from create_distributed_block_mask
        full_mask = create_distributed_block_mask.create_block_mask(
            _causal_mask,
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
        self.assertTrue(
            torch.allclose(local_out.full_tensor(), full_out, atol=1e-3, rtol=1e-3)
        )
        self.assertTrue(
            torch.allclose(
                local_logsumexp.full_tensor(), full_logsumexp, atol=1e-3, rtol=1e-3
            )
        )
        self.assertTrue(
            torch.allclose(
                local_max_scores.full_tensor(), full_max_scores, atol=1e-3, rtol=1e-3
            )
        )

        # Test backward pass
        local_out.sum().backward()
        full_out.sum().backward()

        # Compare gradients
        torch.testing.assert_close(
            q_dt.grad.full_tensor(), full_q.grad, atol=2e-06, rtol=1e-05
        )
        torch.testing.assert_close(
            k_dt.grad.full_tensor(), full_k.grad, atol=2e-06, rtol=1e-05
        )
        torch.testing.assert_close(
            v_dt.grad.full_tensor(), full_v.grad, atol=2e-06, rtol=1e-05
        )

    @with_comms
    def test_flex_attention_replicate(self):
        """Test FlexAttention with replicated placement."""
        device_mesh = self.build_device_mesh()
        dtype = torch.bfloat16

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
        mask = create_distributed_block_mask(
            _causal_mask,
            self.GLOBAL_BS,
            self.GLOBAL_NHEADS,
            self.GLOBAL_QUERY_TOKENS,
            self.GLOBAL_CONTEXT_TOKENS,
            device_mesh,
            [Replicate()],
        )

        # Distribute tensors with Replicate
        placement = [Replicate()]
        q_dt = distribute_tensor(q, device_mesh, placement)
        k_dt = distribute_tensor(k, device_mesh, placement)
        v_dt = distribute_tensor(v, device_mesh, placement)

        # Compile flex_attention for better performance
        flex_attention_compiled = torch.compile(flex_attention)
        flex_attention_compiled = flex_attention

        # Run FlexAttention with DTensors
        out_dt, aux_dt = flex_attention_compiled(
            q_dt,
            k_dt,
            v_dt,
            block_mask=mask,
            return_aux=AuxRequest(lse=True, max_scores=True),
        )
        local_out = out_dt
        local_logsumexp = aux_dt.lse
        local_max_scores = aux_dt.max_scores

        # Verify outputs are DTensors with replicate placements
        self.assertIsInstance(local_out, DTensor)
        self.assertIsInstance(local_logsumexp, DTensor)
        self.assertIsInstance(local_max_scores, DTensor)
        self.assertTrue(local_out.placements[0].is_replicate())
        self.assertTrue(local_logsumexp.placements[0].is_replicate())
        self.assertTrue(local_max_scores.placements[0].is_replicate())

        # Verify shapes
        self.assertEqual(
            local_out.shape,
            (self.GLOBAL_BS, self.GLOBAL_NHEADS, self.GLOBAL_QUERY_TOKENS, self.DIM),
        )

        # Create regular (non-DTensor) block mask for reference computation
        # Use the cached compiled version from create_distributed_block_mask
        full_mask = create_distributed_block_mask.create_block_mask(
            _causal_mask,
            self.GLOBAL_BS,
            self.GLOBAL_NHEADS,
            self.GLOBAL_QUERY_TOKENS,
            self.GLOBAL_CONTEXT_TOKENS,
        )

        # Get full tensors and run reference computation
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
        self.assertTrue(
            torch.allclose(local_out.full_tensor(), full_out, atol=1e-3, rtol=1e-3)
        )
        self.assertTrue(
            torch.allclose(
                local_logsumexp.full_tensor(), full_logsumexp, atol=1e-3, rtol=1e-3
            )
        )
        self.assertTrue(
            torch.allclose(
                local_max_scores.full_tensor(), full_max_scores, atol=1e-3, rtol=1e-3
            )
        )

        # Test backward pass
        local_out.sum().to_local().backward()
        full_out.sum().backward()

        # Compare gradients
        torch.testing.assert_close(
            q_dt.grad.full_tensor(), full_q.grad, atol=2e-06, rtol=1e-05
        )
        torch.testing.assert_close(
            k_dt.grad.full_tensor(), full_k.grad, atol=2e-06, rtol=1e-05
        )
        torch.testing.assert_close(
            v_dt.grad.full_tensor(), full_v.grad, atol=2e-06, rtol=1e-05
        )

    @with_comms
    def test_flex_attention_shard2(self):
        """Test FlexAttention with sequence dimension sharding (Shard(2))."""
        device_mesh = self.build_device_mesh()
        dtype = torch.bfloat16

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

        # Create block mask for distributed execution with Shard(2)
        local_mask = create_distributed_block_mask(
            _causal_mask,
            self.GLOBAL_BS,
            self.GLOBAL_NHEADS,
            self.GLOBAL_QUERY_TOKENS,
            self.GLOBAL_CONTEXT_TOKENS,
            device_mesh,
            [Shard(2)],
        )

        # For context parallel: Q, K, V are all sharded on sequence dim.
        # DTensor will perform the allgather for KV.
        q_dt = distribute_tensor(q, device_mesh, [Shard(2)])
        k_dt = distribute_tensor(k, device_mesh, [Shard(2)])
        v_dt = distribute_tensor(v, device_mesh, [Shard(2)])

        # Compile flex_attention for better performance
        # TODO: we cannot turn on torch.compile yet.
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

        # Verify outputs are DTensors with correct placements
        self.assertIsInstance(local_out, DTensor)
        self.assertIsInstance(local_logsumexp, DTensor)
        self.assertIsInstance(local_max_scores, DTensor)
        self.assertTrue(local_out.placements[0].is_shard(dim=2))
        self.assertTrue(local_logsumexp.placements[0].is_shard(dim=2))
        self.assertTrue(local_max_scores.placements[0].is_shard(dim=2))

        # Create regular (non-DTensor) block mask for reference computation
        # Use the cached compiled version from create_distributed_block_mask
        full_mask = create_distributed_block_mask.create_block_mask(
            _causal_mask,
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
        self.assertTrue(
            torch.allclose(local_out.full_tensor(), full_out, atol=1e-3, rtol=1e-3)
        )
        self.assertTrue(
            torch.allclose(
                local_logsumexp.full_tensor(), full_logsumexp, atol=1e-3, rtol=1e-3
            )
        )
        self.assertTrue(
            torch.allclose(
                local_max_scores.full_tensor(), full_max_scores, atol=1e-3, rtol=1e-3
            )
        )

        # Test backward pass
        local_out.sum().to_local().backward()
        full_out.sum().backward()

        # Compare gradients
        assert False, (q_dt.grad._spec, k_dt.grad._spec)
        """
        torch.testing.assert_close(
            q_dt.grad.full_tensor(), full_q.grad, atol=2e-02, rtol=1e-02
        )
        torch.testing.assert_close(
            k_dt.grad.full_tensor(), full_k.grad, atol=2e-02, rtol=1e-02
        )
        torch.testing.assert_close(
            v_dt.grad.full_tensor(), full_v.grad, atol=2e-02, rtol=1e-02
        )
        """


class DistBlockMaskTest(DTensorTestBase):
    """Test suite for create_distributed_block_mask functionality.

    All dimension values used in these tests (batch size, num_heads, query_tokens,
    context_tokens) represent **global** dimensions across all ranks, not local/sharded
    dimensions. The values are chosen to be divisible by the world_size (typically 4)
    to ensure clean sharding without remainder.
    """

    # Global dimension constants for all tests
    GLOBAL_BS = 16  # Batch size (divisible by world_size=4)
    GLOBAL_NHEADS = 8  # Number of attention heads (divisible by world_size=4)
    GLOBAL_QUERY_TOKENS = 512  # Query sequence length (divisible by world_size=4)
    GLOBAL_CONTEXT_TOKENS = 1024  # KV sequence length (divisible by world_size=4)

    def _verify_block_mask_placements(
        self,
        block_mask,
        expected_kv_placement,
        expected_q_num_blocks_placement=None,
        expected_q_indices_placement=None,
    ):
        """Verify all block mask components have expected placements.

        Args:
            block_mask: BlockMask to verify
            expected_kv_placement: Expected placement for kv_num_blocks, kv_indices,
                                  full_kv_num_blocks, full_kv_indices
            expected_q_num_blocks_placement: Expected placement for q_num_blocks, full_q_num_blocks.
                                            If None, uses expected_kv_placement
            expected_q_indices_placement: Expected placement for q_indices, full_q_indices.
                                         If None, uses expected_kv_placement
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

        # Verify kv_indices
        if block_mask.kv_indices is not None:
            self.assertIsInstance(block_mask.kv_indices, DTensor)
            self.assertEqual(block_mask.kv_indices.placements[0], expected_kv_placement)

        # Verify full_kv_num_blocks
        if block_mask.full_kv_num_blocks is not None:
            self.assertIsInstance(block_mask.full_kv_num_blocks, DTensor)
            self.assertEqual(
                block_mask.full_kv_num_blocks.placements[0], expected_kv_placement
            )

        # Verify full_kv_indices
        if block_mask.full_kv_indices is not None:
            self.assertIsInstance(block_mask.full_kv_indices, DTensor)
            self.assertEqual(
                block_mask.full_kv_indices.placements[0], expected_kv_placement
            )

        # Verify q_num_blocks
        if block_mask.q_num_blocks is not None:
            self.assertIsInstance(block_mask.q_num_blocks, DTensor)
            self.assertEqual(
                block_mask.q_num_blocks.placements[0], expected_q_num_blocks_placement
            )

        # Verify q_indices
        if block_mask.q_indices is not None:
            self.assertIsInstance(block_mask.q_indices, DTensor)
            self.assertEqual(
                block_mask.q_indices.placements[0], expected_q_indices_placement
            )

        # Verify full_q_num_blocks
        if block_mask.full_q_num_blocks is not None:
            self.assertIsInstance(block_mask.full_q_num_blocks, DTensor)
            self.assertEqual(
                block_mask.full_q_num_blocks.placements[0],
                expected_q_num_blocks_placement,
            )

        # Verify full_q_indices
        if block_mask.full_q_indices is not None:
            self.assertIsInstance(block_mask.full_q_indices, DTensor)
            self.assertEqual(
                block_mask.full_q_indices.placements[0], expected_q_indices_placement
            )

    @with_comms
    def test_create_distributed_block_mask_shard0(self):
        """Test create_distributed_block_mask with Shard(0) placement."""
        device_mesh = self.build_device_mesh()

        # Create distributed block mask with Shard(0)
        block_mask = create_distributed_block_mask(
            _causal_mask,
            self.GLOBAL_BS,
            self.GLOBAL_NHEADS,
            self.GLOBAL_QUERY_TOKENS,
            self.GLOBAL_CONTEXT_TOKENS,
            device_mesh,
            [Shard(0)],
        )

        # Verify all mask components are DTensors with Shard(0) placement
        # All 8 fields should be Shard(0) when sharding on batch dimension
        self._verify_block_mask_placements(block_mask, Shard(0))

    @with_comms
    def test_create_distributed_block_mask_replicate(self):
        """Test create_distributed_block_mask with Replicate placement."""
        device_mesh = self.build_device_mesh()

        # Create distributed block mask with Replicate
        block_mask = create_distributed_block_mask(
            _causal_mask,
            self.GLOBAL_BS,
            self.GLOBAL_NHEADS,
            self.GLOBAL_QUERY_TOKENS,
            self.GLOBAL_CONTEXT_TOKENS,
            device_mesh,
            [Replicate()],
        )

        # Verify all mask components are DTensors with Replicate placement
        # All 8 fields should be Replicate()
        self._verify_block_mask_placements(block_mask, Replicate())

    @with_comms
    def test_create_distributed_block_mask_shard2(self):
        """Test create_distributed_block_mask with Shard(2) placement (context parallel)."""
        device_mesh = self.build_device_mesh()

        # Create distributed block mask with Shard(2)
        block_mask = create_distributed_block_mask(
            _causal_mask,
            self.GLOBAL_BS,
            self.GLOBAL_NHEADS,
            self.GLOBAL_QUERY_TOKENS,
            self.GLOBAL_CONTEXT_TOKENS,
            device_mesh,
            [Shard(2)],
        )

        # Verify placements:
        # - kv fields (kv_num_blocks, kv_indices, full_kv_*): Shard(2)
        # - q_num_blocks, full_q_num_blocks: Replicate() (no Q_LEN dimension)
        # - q_indices, full_q_indices: Shard(3) (Q_LEN swapped to last dimension)
        self._verify_block_mask_placements(
            block_mask,
            expected_kv_placement=Shard(2),
            expected_q_num_blocks_placement=Replicate(),
            expected_q_indices_placement=Shard(3),
        )

    @with_comms
    def test_create_distributed_block_mask_invalid_sharding(self):
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
