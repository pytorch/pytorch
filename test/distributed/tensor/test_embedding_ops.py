# Owner(s): ["oncall: distributed"]

import sys

import torch
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

        from torch.distributed.tensor.placement_types import MaskPartial

        # test collectives
        embedding_mod = torch.nn.Embedding(10, 20, device=self.device_type)
        sharded_embedding = self._apply_sharding(embedding_mod, 0, mesh)
        inp = torch.randint(0, 10, (8, 8), device=self.device_type)
        replicated_inp = DTensor.from_local(inp, mesh, [Replicate()], run_check=False)
        output = sharded_embedding(replicated_inp)
        self.assertIsInstance(output.placements[0], MaskPartial)

        comm_mode = CommDebugMode()

        with comm_mode:
            output.full_tensor()
            self.assertEqual(comm_mode.get_total_counts(), 1)
            self.assertEqual(comm_mode.get_comm_counts()[funcol.all_reduce], 1)

    @with_comms
    def test_multiple_embeddings_rowwise(self):
        mesh = self.build_device_mesh()

        inp = torch.randint(0, 10, (4, 4), device=self.device_type)
        replicated_inp = DTensor.from_local(inp, mesh, [Replicate()], run_check=False)

        from torch.distributed.tensor.placement_types import MaskPartial

        # case 1: two embeddings with the same shape, thus sharing the underlying MaskPartial
        # and MaskBuffer, because of cache hit from sharding propagation

        emb1 = torch.nn.Embedding(10, 23, device=self.device_type)
        sharded_emb1 = self._apply_sharding(emb1, 0, mesh)
        output1 = sharded_emb1(replicated_inp)

        emb2 = torch.nn.Embedding(10, 29, device=self.device_type)
        sharded_emb2 = self._apply_sharding(emb2, 0, mesh)
        output2 = sharded_emb2(replicated_inp)

        partial_placement1 = output1.placements[0]
        self.assertIsInstance(partial_placement1, MaskPartial)
        output1.full_tensor()

        partial_placement2 = output2.placements[0]
        self.assertIsInstance(partial_placement2, MaskPartial)
        output2.full_tensor()

        self.assertTrue(id(partial_placement1), id(partial_placement2))

        # case 2: two embeddings with the same logical_dim_size, but different logical_shape
        # thus they will have different MaskPartial placements (with no cache hit)

        emb3 = torch.nn.Embedding(10, 29, device=self.device_type)
        sharded_emb3 = self._apply_sharding(emb3, 0, mesh)
        output3 = sharded_emb3(replicated_inp)
        partial_placement3 = output3.placements[0]
        self.assertIsInstance(partial_placement3, MaskPartial)
        output2.full_tensor()

        # not equal because of different logical_shape, despite of same logical_dim_size
        self.assertNotEqual(partial_placement1, partial_placement3)


TestEmbeddingOpWithLocalTensor = create_local_tensor_test_class(
    TestEmbeddingOp,
)

if __name__ == "__main__":
    run_tests()
