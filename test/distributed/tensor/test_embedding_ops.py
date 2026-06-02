# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch._dynamo.testing
from torch.distributed.tensor import (
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorTestBase,
    with_comms,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


funcol = torch.ops.c10d_functional


class TestEmbeddingOp(DTensorTestBase):
    def _apply_sharding(self, embedding_mod, shard_dim, device_mesh):
        def shard_embedding_fn(name, module, device_mesh):
            for name, param in module.named_parameters():
                dist_param = torch.nn.Parameter(
                    distribute_tensor(param, device_mesh, [Shard(shard_dim)])
                )
                module.register_parameter(name, dist_param)

        sharded_embedding = distribute_module(
            embedding_mod, device_mesh, shard_embedding_fn
        )
        return sharded_embedding

    def _run_embedding_op_test(
        self,
        device_mesh,
        shard_dim,
        input_size,
        num_embeddings,
        embedding_dim,
        **kwargs,
    ):
        # Use same seed.
        torch.manual_seed(0)
        local_embedding = torch.nn.Embedding(
            num_embeddings,
            embedding_dim,
            device=self.device_type,
            **kwargs,
        )
        sharded_embedding = torch.nn.Embedding(
            num_embeddings,
            embedding_dim,
            device=self.device_type,
            **kwargs,
        )

        # Shard the parameter of local embedding and set it to sharded embedding.
        sharded_embedding.weight = torch.nn.Parameter(
            local_embedding.weight.detach().clone()
        )

        sharded_embedding = self._apply_sharding(
            sharded_embedding, shard_dim, device_mesh
        )

        # Run sharded computation
        torch.manual_seed(10)
        inp = torch.randint(
            0, num_embeddings, tuple(input_size), device=self.device_type
        )
        target = torch.empty(
            *inp.size(), embedding_dim, dtype=torch.float, device=self.device_type
        ).random_(0, 1)
        dist_inp = distribute_tensor(inp, device_mesh, [Replicate()])

        # fwd computation, ensure no comm happened
        with CommDebugMode() as fwd_mode:
            dist_output = sharded_embedding(dist_inp)
            self.assertEqual(fwd_mode.get_total_counts(), 0)

        output = dist_output.full_tensor()
        # Run local computation
        local_output = local_embedding(inp)

        # Verify
        self.assertEqual(local_output, output)

        # Use a sample cross entry loss to verify backward and grad computation.
        loss = torch.nn.CrossEntropyLoss()
        emb_loss = loss(
            output,
            target,
        )
        emb_dup_loss = loss(
            local_output,
            target,
        )

        # local embedding backward
        emb_dup_loss.backward()

        # sharded embedding bwd computation, ensure no comm happened
        with CommDebugMode() as bwd_mode:
            emb_loss.backward()
            self.assertEqual(bwd_mode.get_total_counts(), 0)

        gradient = sharded_embedding.weight.grad.full_tensor()

        local_grad = local_embedding.weight.grad

        # Verify gradient.
        self.assertEqual(gradient, local_grad)

        # Validate for torch.nn.functional.embedding version.
        local_output = torch.nn.functional.embedding(
            inp,
            local_embedding.weight,
            **kwargs,
        )
        sharded_output = torch.nn.functional.embedding(
            DTensor.from_local(inp, device_mesh, [Replicate()], run_check=False),
            sharded_embedding.weight,
            **kwargs,
        )
        self.assertEqual(local_output, sharded_output.full_tensor())

    @with_comms
    def test_sharded_embedding_colwise(self):
        mesh = self.build_device_mesh()
        self._run_embedding_op_test(mesh, 1, [5, 4], 17, 12)
        self._run_embedding_op_test(mesh, 1, [6, 7, 6], 21, 11)
        self._run_embedding_op_test(mesh, 1, [8, 6, 5, 4], 23, 13)
        self._run_embedding_op_test(mesh, 1, [8, 6, 5, 4, 7], 23, 16)
        self._run_embedding_op_test(mesh, 1, [4], 15, 14)
        self._run_embedding_op_test(mesh, 1, [34], 15, 14, padding_idx=10)
        self._run_embedding_op_test(mesh, 1, [8, 6, 5, 4], 23, 13, padding_idx=12)

    @with_comms
    def test_sharded_embedding_colwise_max_norm_errors(self):
        mesh = self.build_device_mesh()
        with self.assertRaisesRegex(
            NotImplementedError,
            "aten.embedding_renorm_.default does not have a sharding strategy registered.",
        ):
            self._run_embedding_op_test(
                mesh, 1, [8, 6, 5, 4], 23, 13, padding_idx=12, max_norm=2.0
            )

    @with_comms
    def test_sharded_embedding_rowwise(self):
        mesh = self.build_device_mesh()
        # test correctness
        self._run_embedding_op_test(mesh, 0, [5, 12], 16, 22)
        self._run_embedding_op_test(mesh, 0, [6, 7, 6], 13, 22)
        self._run_embedding_op_test(mesh, 0, [34], 15, 14, padding_idx=10)

        from torch.distributed.tensor.placement_types import _MaskPartial

        # test collectives
        embedding_mod = torch.nn.Embedding(10, 20, device=self.device_type)
        sharded_embedding = self._apply_sharding(embedding_mod, 0, mesh)
        inp = torch.randint(0, 10, (8, 8), device=self.device_type)
        replicated_inp = DTensor.from_local(inp, mesh, [Replicate()], run_check=False)
        output = sharded_embedding(replicated_inp)
        self.assertIsInstance(output.placements[0], _MaskPartial)

        comm_mode = CommDebugMode()

        with comm_mode:
            output.full_tensor()
            self.assertEqual(comm_mode.get_total_counts(), 1)
            self.assertEqual(comm_mode.get_comm_counts()[funcol.all_reduce], 1)

    @with_comms
    def test_sharded_embedding_rowwise_compile(self):
        if self.is_local_tensor_enabled:
            self.skipTest("torch.compile DTensor subclass output test")
        mesh = self.build_device_mesh()
        torch.manual_seed(0)
        local_embedding = torch.nn.Embedding(10, 20, device=self.device_type)
        sharded_embedding = torch.nn.Embedding(10, 20, device=self.device_type)
        sharded_embedding.weight = torch.nn.Parameter(
            local_embedding.weight.detach().clone()
        )
        sharded_embedding = self._apply_sharding(sharded_embedding, 0, mesh)
        compiled_embedding = torch.compile(sharded_embedding)

        torch.manual_seed(10)
        inp = torch.randint(0, 10, (8, 8), device=self.device_type)
        replicated_inp = DTensor.from_local(inp, mesh, [Replicate()], run_check=False)
        output = compiled_embedding(replicated_inp)

        from torch.distributed.tensor.placement_types import _MaskPartial

        self.assertIsInstance(output.placements[0], _MaskPartial)
        self.assertIsNotNone(output.placements[0].mask_buffer.data)
        self.assertEqual(output.full_tensor(), local_embedding(inp))

    @with_comms
    def test_pending_mask_partial_compile_input(self):
        if self.is_local_tensor_enabled:
            self.skipTest("torch.compile DTensor subclass input test")
        mesh = self.build_device_mesh()
        torch.manual_seed(0)
        local_embedding = torch.nn.Embedding(10, 20, device=self.device_type)
        sharded_embedding = torch.nn.Embedding(10, 20, device=self.device_type)
        sharded_embedding.weight = torch.nn.Parameter(
            local_embedding.weight.detach().clone()
        )
        sharded_embedding = self._apply_sharding(sharded_embedding, 0, mesh)

        torch.manual_seed(10)
        inp = torch.randint(0, 10, (8, 8), device=self.device_type)
        replicated_inp = DTensor.from_local(inp, mesh, [Replicate()], run_check=False)
        pending_output = sharded_embedding(replicated_inp)
        compiled_output = torch.compile(lambda x: x)(pending_output)

        self.assertEqual(compiled_output.full_tensor(), local_embedding(inp))

    @with_comms
    def test_pending_mask_partial_compile_input_guard_reuses_graph(self):
        if self.is_local_tensor_enabled:
            self.skipTest("torch.compile DTensor subclass input test")
        mesh = self.build_device_mesh()
        torch.manual_seed(0)
        local_embedding = torch.nn.Embedding(10, 20, device=self.device_type)
        sharded_embedding = torch.nn.Embedding(10, 20, device=self.device_type)
        sharded_embedding.weight = torch.nn.Parameter(
            local_embedding.weight.detach().clone()
        )
        sharded_embedding = self._apply_sharding(sharded_embedding, 0, mesh)

        inp1 = torch.zeros((4, 4), dtype=torch.long, device=self.device_type)
        inp2 = torch.full((4, 4), 9, dtype=torch.long, device=self.device_type)
        replicated_inp1 = DTensor.from_local(inp1, mesh, [Replicate()], run_check=False)
        replicated_inp2 = DTensor.from_local(inp2, mesh, [Replicate()], run_check=False)
        pending_output1 = sharded_embedding(replicated_inp1)
        pending_output2 = sharded_embedding(replicated_inp2)

        counter = torch._dynamo.testing.CompileCounter()
        compiled_identity = torch.compile(
            lambda x: x, backend=counter, fullgraph=True, dynamic=False
        )

        compiled_output1 = compiled_identity(pending_output1)
        compiled_output2 = compiled_identity(pending_output2)

        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(compiled_output1.full_tensor(), local_embedding(inp1))
        self.assertEqual(compiled_output2.full_tensor(), local_embedding(inp2))

    @with_comms
    def test_compiled_embedding_repeated_pending_outputs(self):
        if self.is_local_tensor_enabled:
            self.skipTest("torch.compile DTensor subclass output test")
        mesh = self.build_device_mesh()
        torch.manual_seed(0)
        local_embedding = torch.nn.Embedding(16, 8, device=self.device_type)
        sharded_embedding = torch.nn.Embedding(16, 8, device=self.device_type)
        sharded_embedding.weight = torch.nn.Parameter(
            local_embedding.weight.detach().clone()
        )
        sharded_embedding = self._apply_sharding(sharded_embedding, 0, mesh)
        compiled_embedding = torch.compile(sharded_embedding)

        inp1 = torch.zeros((4, 4), dtype=torch.long, device=self.device_type)
        inp2 = torch.full((4, 4), 15, dtype=torch.long, device=self.device_type)
        replicated_inp1 = DTensor.from_local(inp1, mesh, [Replicate()], run_check=False)
        replicated_inp2 = DTensor.from_local(inp2, mesh, [Replicate()], run_check=False)

        output1 = compiled_embedding(replicated_inp1)
        output2 = compiled_embedding(replicated_inp2)

        self.assertIsNot(
            output1.placements[0].mask_buffer, output2.placements[0].mask_buffer
        )
        self.assertEqual(output1.full_tensor(), local_embedding(inp1))
        self.assertEqual(output2.full_tensor(), local_embedding(inp2))

    @with_comms
    def test_multiple_compiled_embeddings_rowwise_different_masks(self):
        if self.is_local_tensor_enabled:
            self.skipTest("torch.compile DTensor subclass output test")
        mesh = self.build_device_mesh()
        torch.manual_seed(0)
        local_embedding1 = torch.nn.Embedding(16, 8, device=self.device_type)
        local_embedding2 = torch.nn.Embedding(16, 8, device=self.device_type)
        sharded_embedding1 = torch.nn.Embedding(16, 8, device=self.device_type)
        sharded_embedding2 = torch.nn.Embedding(16, 8, device=self.device_type)
        sharded_embedding1.weight = torch.nn.Parameter(
            local_embedding1.weight.detach().clone()
        )
        sharded_embedding2.weight = torch.nn.Parameter(
            local_embedding2.weight.detach().clone()
        )
        sharded_embedding1 = self._apply_sharding(sharded_embedding1, 0, mesh)
        sharded_embedding2 = self._apply_sharding(sharded_embedding2, 0, mesh)

        class TwoEmbeddings(torch.nn.Module):
            def __init__(self, embedding1, embedding2):
                super().__init__()
                self.embedding1 = embedding1
                self.embedding2 = embedding2

            def forward(self, inp1, inp2):
                return self.embedding1(inp1), self.embedding2(inp2)

        inp1 = torch.zeros((4, 4), dtype=torch.long, device=self.device_type)
        inp2 = torch.full((4, 4), 15, dtype=torch.long, device=self.device_type)
        replicated_inp1 = DTensor.from_local(inp1, mesh, [Replicate()], run_check=False)
        replicated_inp2 = DTensor.from_local(inp2, mesh, [Replicate()], run_check=False)

        output1, output2 = torch.compile(
            TwoEmbeddings(sharded_embedding1, sharded_embedding2)
        )(replicated_inp1, replicated_inp2)

        self.assertIsNot(output1.placements[0], output2.placements[0])
        self.assertIsNot(
            output1.placements[0].mask_buffer, output2.placements[0].mask_buffer
        )
        self.assertEqual(output1.full_tensor(), local_embedding1(inp1))
        self.assertEqual(output2.full_tensor(), local_embedding2(inp2))

    @with_comms
    def test_pending_mask_partial_redistribute_preserves_original(self):
        mesh = self.build_device_mesh()
        torch.manual_seed(0)
        local_embedding = torch.nn.Embedding(10, 20, device=self.device_type)
        sharded_embedding = torch.nn.Embedding(10, 20, device=self.device_type)
        sharded_embedding.weight = torch.nn.Parameter(
            local_embedding.weight.detach().clone()
        )
        sharded_embedding = self._apply_sharding(sharded_embedding, 0, mesh)

        torch.manual_seed(10)
        inp = torch.randint(0, 10, (8, 8), device=self.device_type)
        replicated_inp = DTensor.from_local(inp, mesh, [Replicate()], run_check=False)
        pending_output = sharded_embedding(replicated_inp)
        copied_output = pending_output.redistribute(
            placements=pending_output.placements
        )

        self.assertIsNot(
            pending_output.placements[0].mask_buffer,
            copied_output.placements[0].mask_buffer,
        )
        self.assertEqual(pending_output.full_tensor(), local_embedding(inp))
        self.assertEqual(copied_output.full_tensor(), local_embedding(inp))

    @with_comms
    def test_multiple_embeddings_rowwise(self):
        mesh = self.build_device_mesh()

        inp = torch.randint(0, 10, (4, 4), device=self.device_type)
        replicated_inp = DTensor.from_local(inp, mesh, [Replicate()], run_check=False)

        from torch.distributed.tensor.placement_types import _MaskPartial

        # case 1: two embeddings with the same shape get separate output
        # _MaskPartial instances so their live mask buffers do not alias.

        emb1 = torch.nn.Embedding(10, 23, device=self.device_type)
        sharded_emb1 = self._apply_sharding(emb1, 0, mesh)
        output1 = sharded_emb1(replicated_inp)

        emb2 = torch.nn.Embedding(10, 23, device=self.device_type)
        sharded_emb2 = self._apply_sharding(emb2, 0, mesh)
        output2 = sharded_emb2(replicated_inp)

        partial_placement1 = output1.placements[0]
        self.assertIsInstance(partial_placement1, _MaskPartial)
        output1.full_tensor()

        partial_placement2 = output2.placements[0]
        self.assertIsInstance(partial_placement2, _MaskPartial)
        output2.full_tensor()

        self.assertIsNot(partial_placement1, partial_placement2)
        self.assertIsNot(partial_placement1.mask_buffer, partial_placement2.mask_buffer)
        self.assertEqual(
            partial_placement1.offset_shape, partial_placement2.offset_shape
        )
        self.assertEqual(partial_placement1.offset_dim, partial_placement2.offset_dim)

        # case 2: two embeddings with the same logical_dim_size, but different logical_shape
        # thus they will have different _MaskPartial placements (with no cache hit)

        emb3 = torch.nn.Embedding(10, 29, device=self.device_type)
        sharded_emb3 = self._apply_sharding(emb3, 0, mesh)
        output3 = sharded_emb3(replicated_inp)
        partial_placement3 = output3.placements[0]
        self.assertIsInstance(partial_placement3, _MaskPartial)
        output3.full_tensor()

        # not equal because of different logical_shape, despite of same logical_dim_size
        self.assertNotEqual(partial_placement1, partial_placement3)

    @with_comms
    def test_embedding_backward_different_num_embeddings(self):
        # Regression test: embedding_dense_backward op strategy must include
        # num_weights in its cache key. Without this, multiple embeddings with
        # different num_embeddings share a cached strategy, producing gradients
        # with the wrong shape.
        mesh = self.build_device_mesh()
        torch.manual_seed(0)

        emb_small = torch.nn.Embedding(16, 12, device=self.device_type)
        emb_large = torch.nn.Embedding(32, 12, device=self.device_type)
        sharded_emb_small = self._apply_sharding(emb_small, 1, mesh)
        sharded_emb_large = self._apply_sharding(emb_large, 1, mesh)

        inp_small = torch.randint(0, 16, (4, 4), device=self.device_type)
        inp_large = torch.randint(0, 32, (4, 4), device=self.device_type)
        dist_inp_small = DTensor.from_local(
            inp_small, mesh, [Replicate()], run_check=False
        )
        dist_inp_large = DTensor.from_local(
            inp_large, mesh, [Replicate()], run_check=False
        )

        out_large = sharded_emb_large(dist_inp_large)
        out_small = sharded_emb_small(dist_inp_small)
        loss = out_large.sum() + out_small.sum()
        loss.backward()

        self.assertEqual(sharded_emb_small.weight.grad.full_tensor().shape, (16, 12))
        self.assertEqual(sharded_emb_large.weight.grad.full_tensor().shape, (32, 12))


TestEmbeddingOpWithLocalTensor = create_local_tensor_test_class(
    TestEmbeddingOp,
)

if __name__ == "__main__":
    run_tests()
