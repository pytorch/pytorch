# Owner(s): ["oncall: distributed"]

import sys

import torch
from torch.distributed._tensor import DTensor
from torch.distributed._tensor.placement_types import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestEmbeddingOp(DTensorTestBase):
    def _run_embedding_op_test(
        self,
        shard_dim,
        input_size,
        num_embeddings,
        embedding_dim,
        **kwargs,
    ):
        # Use same seed.
        device_mesh = self.build_device_mesh()
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
            local_embedding.weight.clone().detach()
        )
        parallelize_module(
            module=sharded_embedding,
            device_mesh=device_mesh,
            parallelize_plan=ColwiseParallel(output_layouts=Replicate())
            if shard_dim == 1
            else RowwiseParallel(),
        )

        # Run sharded computation
        torch.manual_seed(10)
        inp = torch.randint(
            0, num_embeddings, tuple(input_size), device=self.device_type
        )
        target = torch.empty(
            *inp.size(), embedding_dim, dtype=torch.float, device=self.device_type
        ).random_(0, 1)
        output = sharded_embedding(inp)

        # Run local computation
        local_output = local_embedding(inp)

        # Verify
        self.assertEqual(local_output, output)

        # Use a sample cross entry loss to verify backward and grad computation.
        loss = torch.nn.CrossEntropyLoss()
        attn_loss = loss(
            output,
            target,
        )
        attn_dup_loss = loss(
            local_output,
            target,
        )
        attn_loss.backward()
        attn_dup_loss.backward()

        gradient = sharded_embedding.weight.grad.redistribute(
            device_mesh, [Replicate()]
        ).to_local()

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
            DTensor.from_local(inp, device_mesh, [Replicate()]),
            sharded_embedding.weight,
            **kwargs,
        )
        self.assertEqual(local_output, sharded_output.full_tensor())

    @with_comms
    def test_sharded_embedding_colwise(self):
        self._run_embedding_op_test(1, [5, 4], 17, 12)
        self._run_embedding_op_test(1, [6, 7, 6], 21, 11)
        self._run_embedding_op_test(1, [8, 6, 5, 4], 23, 13)
        self._run_embedding_op_test(1, [8, 6, 5, 4, 7], 23, 16)
        self._run_embedding_op_test(1, [4], 15, 14)
        self._run_embedding_op_test(1, [34], 15, 14, padding_idx=10)
        self._run_embedding_op_test(1, [8, 6, 5, 4], 23, 13, padding_idx=12)

    @with_comms
    def test_sharded_embedding_colwise_errors(self):
        with self.assertRaisesRegex(
            NotImplementedError,
            "DTensor does not support sharded embedding operation with max_norm yet!",
        ):
            self._run_embedding_op_test(
                1, [8, 6, 5, 4], 23, 13, padding_idx=12, max_norm=2.0
            )

    @with_comms
    def test_sharded_embedding_rowwise(self):
        with self.assertRaisesRegex(
            NotImplementedError,
            "RowwiseParallel currently only support nn.Linear!",
        ):
            self._run_embedding_op_test(0, [5, 12], 16, 22)


if __name__ == "__main__":
    run_tests()
