# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import itertools
from copy import deepcopy
from typing import Dict, NamedTuple, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    loss_parallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.distributed.tensor.parallel.input_reshard import input_reshard
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    ModelArgs,
    NUM_DEVICES,
    skip_unless_torch_gpu,
    Transformer,
    with_comms,
)


c10d_functional = torch.ops.c10d_functional
reduce_scatter, all_gather, all_reduce = (
    c10d_functional.reduce_scatter_tensor,
    c10d_functional.all_gather_into_tensor,
    c10d_functional.all_reduce,
)


class ExpCommCounts(NamedTuple):
    fwd: Optional[Dict] = None
    bwd: Optional[Dict] = None
    optim: Optional[Dict] = None


class DistTensorParallelExampleTest(DTensorTestBase):
    def _check_module(self, m1, m2, check_grad=False):
        named_parameters = dict(m1.named_parameters())
        for name, param_m2 in m2.named_parameters():
            self.assertTrue(name in named_parameters)
            param_m1 = named_parameters[name]
            if check_grad:
                param_m2 = param_m2.grad
                param_m1 = param_m1.grad
            if isinstance(param_m2, DTensor):
                replicate = [Replicate()]
                param_m2 = param_m2.redistribute(
                    device_mesh=param_m2.device_mesh, placements=replicate
                ).to_local()
            self.assertEqual(param_m2, param_m1)

    def _test_mlp_training_e2e(self, is_seq_parallel=False, recompute_activation=False):
        inp_size = [8, 10]
        # Ensure all tp ranks have same input.
        rng_seed = self.rank if is_seq_parallel else 0
        torch.manual_seed(rng_seed)
        inp = torch.rand(*inp_size, device=self.device_type)
        model = MLPModule(self.device_type)
        model_tp = deepcopy(model)

        # Ensure model are initialized the same way.
        self._check_module(model, model_tp)

        # Shard module and initialize optimizer.
        LR = 0.25
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )
        parallelize_plan = {
            "net1": (
                ColwiseParallel(input_layouts=Shard(0))
                if is_seq_parallel
                else ColwiseParallel()
            ),
            "net2": (
                RowwiseParallel(output_layouts=Shard(0))
                if is_seq_parallel
                else RowwiseParallel()
            ),
        }
        model_tp = parallelize_module(model_tp, device_mesh, parallelize_plan)
        if recompute_activation:
            model_tp = input_reshard(
                checkpoint_wrapper(
                    model_tp, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                ),
                device_mesh,
                None if is_seq_parallel else 0,
            )
        optim = torch.optim.SGD(model.parameters(), lr=LR)
        optim_tp = torch.optim.SGD(model_tp.parameters(), lr=LR)

        output = model(inp)
        output.sum().backward()

        comm_mode = CommDebugMode()
        with comm_mode:
            output_tp = model_tp(inp)
            output_tp.sum().backward()

        self.assertEqual(output, output_tp)
        if is_seq_parallel:
            self.assertEqual(
                comm_mode.get_comm_counts()[c10d_functional.all_gather_into_tensor], 2
            )
            self.assertEqual(
                comm_mode.get_comm_counts()[c10d_functional.reduce_scatter_tensor], 1
            )
        else:
            self.assertEqual(comm_mode.get_comm_counts()[c10d_functional.all_reduce], 1)

        if is_seq_parallel:
            # Sum gradients from different ranks, since input
            # are different across ranks for sequence parallel.
            dist.all_reduce(model.net1.weight.grad)
            dist.all_reduce(model.net1.bias.grad)
            dist.all_reduce(model.net2.weight.grad)
            dist.all_reduce(model.net2.bias.grad)

        # Ensure gradients are same.
        self._check_module(model, model_tp, check_grad=True)

        optim.step()
        optim_tp.step()

        # Ensure model weights are still same after update.
        # Due to the trick we use for Partial aggregation, we only check the weight when local_rank = 0.
        self._check_module(model, model_tp)

        inp = torch.rand(*inp_size, device=self.device_type)
        output = model(inp)
        output_tp = model_tp(inp)
        self.assertEqual(output, output_tp)

    def _test_mlp_inference(self, device_mesh):
        inp_size = [8, 10]
        # Ensure all tp ranks have same input.
        torch.manual_seed(0)
        inp = torch.rand(*inp_size, device=self.device_type)
        model = MLPModule(self.device_type)
        model_tp = deepcopy(model)

        # Ensure model are initialized the same way.
        self._check_module(model, model_tp)

        # Shard module and initialize optimizer.
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        model_tp = parallelize_module(model_tp, device_mesh, parallelize_plan)

        output = model(inp)
        output_tp = model_tp(inp)
        self.assertEqual(output, output_tp)

    @with_comms
    @parametrize("is_seq_parallel", [True, False])
    # TODO: need to revisit input_reshard API about why it failed multi-gpu tests.
    # @parametrize("recompute_activation", [True, False])
    @parametrize("recompute_activation", [False])
    def test_mlp_training(self, is_seq_parallel, recompute_activation):
        self._test_mlp_training_e2e(
            is_seq_parallel=is_seq_parallel, recompute_activation=recompute_activation
        )

    @with_comms
    def test_mlp_inference(self):
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )
        with torch.inference_mode():
            self._test_mlp_inference(device_mesh)

    def _setup_single_gpu_model(self, model_args, dtype):
        return Transformer(model_args).to(device=self.device_type, dtype=dtype)

    def _setup_tp_model(self, model, is_seq_parallel, dtype):
        model_tp = deepcopy(model)
        self._check_module(model, model_tp)
        device_mesh = DeviceMesh(self.device_type, torch.arange(0, NUM_DEVICES))
        local_output_for_attn = dtype is torch.float64
        return Transformer.parallelize(
            model_tp,
            device_mesh,
            is_seq_parallel,
            local_output_for_attn=local_output_for_attn,
        )

    def _setup_optimizer(self, model, model_tp):
        # Step 3: Run test by comparing outputs from single-gpu and multi-gpu models.
        LR = 0.25
        optim = torch.optim.Adam(model.parameters(), lr=LR)
        optim_tp = torch.optim.Adam(model_tp.parameters(), lr=LR)
        return optim, optim_tp

    def _validate_fwd(
        self, model, model_tp, inp, expected_comms_dict=None, check_comms=True
    ):
        # Compare outputs on the same input.
        output = model(inp)
        with CommDebugMode() as comm_mode:
            output_tp = model_tp(inp)
        self.assertEqual(output, output_tp)
        if check_comms:
            self.assertDictEqual(comm_mode.get_comm_counts(), expected_comms_dict or {})
        return output, output_tp

    def _validate_bwd(
        self,
        model,
        model_tp,
        output,
        output_tp,
        expected_comms_dict=None,
        check_comms=True,
    ):
        # Ensure gradients are equal.
        output.sum().backward()
        with CommDebugMode() as comm_mode:
            output_tp.sum().backward()
        self._check_module(model, model_tp, check_grad=True)
        if check_comms:
            self.assertDictEqual(comm_mode.get_comm_counts(), expected_comms_dict or {})

    def _validate_optim_step(
        self,
        model,
        model_tp,
        optim,
        optim_tp,
        expected_comms_dict=None,
        check_comms=True,
    ):
        optim.step()  # Ensure model weights are still the same after update.
        from torch.distributed._tensor.experimental import implicit_replication

        with implicit_replication():
            with CommDebugMode() as comm_mode:
                optim_tp.step()
        self._check_module(model, model_tp)
        if check_comms:
            self.assertDictEqual(comm_mode.get_comm_counts(), expected_comms_dict or {})

    @staticmethod
    def _thaw_params(thaw_params, model, model_tp):
        if not thaw_params:
            return
        for target_model in [model, model_tp]:
            for n, p in target_model.named_parameters():
                if n not in thaw_params:
                    p.requires_grad_(False)

    @with_comms
    @skip_unless_torch_gpu
    @parametrize("is_seq_parallel", [True, False])
    @parametrize("dtype", [torch.float64, torch.float32])
    def test_transformer_training(self, is_seq_parallel, dtype: torch.dtype):
        EXP_BASE_CC = ExpCommCounts(
            fwd={all_reduce: 6, all_gather: 1}, bwd={all_reduce: 9}
        )
        EXP_SEQ_PARALLEL_CC = ExpCommCounts(
            fwd={reduce_scatter: 6, all_gather: 6},
            bwd={reduce_scatter: 5, all_gather: 6},
            optim={all_reduce: 30},
        )

        # Disable dropout in the test since we cannot reproduce the same random
        # behaviors when comparing single-gpu models with multi-gpu models.
        model_args = ModelArgs(dropout_p=0.0)
        model = self._setup_single_gpu_model(
            model_args, dtype
        )  # Step 1: Initialize single-gpu models.
        model_tp = self._setup_tp_model(
            model, is_seq_parallel, dtype
        )  # Step 2: Setup tp model, place onto device mesh.
        optim, optim_tp = self._setup_optimizer(
            model, model_tp
        )  # Step 3: Setup optimizers for both models

        # Initialize input and make sure all ranks have the same input.
        inp_size = [8, 8]  # [batch_size, seq_len]
        if is_seq_parallel:
            assert inp_size[1] % self.world_size == 0

        torch.manual_seed(0)
        steps = 10 if type(model) is torch.float64 else 1
        for _ in range(steps):
            inp = torch.randint(
                model_args.vocab_size, inp_size, device=self.device_type
            )
            expected_fwd_comms = (
                EXP_SEQ_PARALLEL_CC.fwd if is_seq_parallel else EXP_BASE_CC.fwd
            )
            output, output_tp = self._validate_fwd(
                model, model_tp, inp, expected_fwd_comms
            )
            expected_bwd_comms = (
                EXP_SEQ_PARALLEL_CC.bwd if is_seq_parallel else EXP_BASE_CC.bwd
            )
            self._validate_bwd(model, model_tp, output, output_tp, expected_bwd_comms)
            expected_optim_comms = (
                EXP_SEQ_PARALLEL_CC.optim if is_seq_parallel else EXP_BASE_CC.optim
            )
            self._validate_optim_step(
                model, model_tp, optim, optim_tp, expected_optim_comms
            )

    @with_comms
    @skip_unless_torch_gpu
    @parametrize(
        "thaw_params, is_seq_parallel, dtype, exp_cnts",
        [
            (
                None,  # all require grad seq_parallel float32 baseline
                True,
                torch.float32,
                ExpCommCounts(
                    bwd={reduce_scatter: 5, all_gather: 6}, optim={all_reduce: 30}
                ),
            ),
            (
                None,  # all require grad no seq_parallel float64 baseline
                False,
                torch.float64,
                ExpCommCounts(bwd={all_reduce: 9}),
            ),
            # test a subset of LayerNorm bwd output_masks
            (
                ("output.weight", "norm.weight", "norm.bias"),  # [False, True, True]
                True,
                torch.float32,
                ExpCommCounts(bwd={reduce_scatter: 1}, optim={all_reduce: 6}),
            ),
            (
                ("tok_embeddings.weight", "output.weight"),  # [True, False, False]
                True,
                torch.float32,
                ExpCommCounts(bwd={reduce_scatter: 5, all_gather: 5}),
            ),
            (
                (
                    "tok_embeddings.weight",
                    "output.weight",
                    "norm.weight",
                    "norm.bias",
                ),  # [True, True, True]
                True,
                torch.float32,
                ExpCommCounts(
                    bwd={reduce_scatter: 5, all_gather: 5}, optim={all_reduce: 6}
                ),
            ),
            (
                (
                    "tok_embeddings.weight",
                    "output.weight",
                    "norm.weight",
                    "norm.bias",
                    "layers.1.ffn_norm.weight",
                    "layers.1.ffn_norm.bias",
                ),  # a single transformerblock layernorm
                True,
                torch.float32,
                ExpCommCounts(
                    bwd={reduce_scatter: 5, all_gather: 5}, optim={all_reduce: 12}
                ),
            ),
            (
                (
                    "tok_embeddings.weight",
                    "layers.0.attention.wv.weight",
                    "layers.0.feed_forward.w1.bias",
                    "layers.1.ffn_norm.bias",
                    "layers.1.feed_forward.w2.weight",
                    "output.weight",
                ),  # varied layer/param types
                True,
                torch.float32,
                ExpCommCounts(
                    bwd={reduce_scatter: 5, all_gather: 5}, optim={all_reduce: 3}
                ),
            ),
        ],
        name_fn=lambda thaw, seq, dtype, *_: f"{'seq_parallel_' if seq else ''}"
        + f"{str(dtype).split('.')[-1]}_"
        + f"thaw_{'__'.join(sorted({n.rpartition('.')[0].replace('.', '_') for n in thaw})) if thaw else 'all'}",
    )
    def test_transformer_req_grad(self, thaw_params, is_seq_parallel, dtype, exp_cnts):
        # Sample a subset of `requires_grad` patterns

        # disabling dropout to facilitate single gpu to multi-device comparison
        # disable weight-tying to enable more fine-tuning configurations
        model_args = ModelArgs(dropout_p=0.0, weight_tying=False)
        model = self._setup_single_gpu_model(
            model_args, dtype
        )  # Step 1: Initialize single-gpu models.
        model_tp = self._setup_tp_model(
            model, is_seq_parallel, dtype
        )  # Step 2: Setup tp model, place onto device mesh.
        optim, optim_tp = self._setup_optimizer(
            model, model_tp
        )  # Step 3: Setup optimizers for both models
        DistTensorParallelExampleTest._thaw_params(
            thaw_params, model, model_tp
        )  # Step 4: set `requires_grad` patterns

        # Initialize input and make sure all ranks have the same input.
        inp_size = [8, 8]  # [batch_size, seq_len]
        if is_seq_parallel:
            assert inp_size[1] % self.world_size == 0

        torch.manual_seed(0)
        inp = torch.randint(model_args.vocab_size, inp_size, device=self.device_type)
        output, output_tp = self._validate_fwd(model, model_tp, inp, check_comms=False)
        self._validate_bwd(
            model, model_tp, output, output_tp, exp_cnts.bwd, check_comms=True
        )
        self._validate_optim_step(
            model, model_tp, optim, optim_tp, exp_cnts.optim, check_comms=True
        )

    @with_comms
    def test_weight_tying(self):
        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Initialize different weights for embedding and fc.
                torch.manual_seed(1)
                self.embedding = torch.nn.Embedding(16, 8)
                torch.manual_seed(2)
                self.fc = torch.nn.Linear(8, 16)

            def forward(self, x):
                return self.fc(self.embedding(x))

        model = TestModule().to(self.device_type)
        parallelize_plan = {
            "embedding": ColwiseParallel(),
            "fc": RowwiseParallel(),
        }
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        parallelize_module(model, device_mesh, parallelize_plan)

        input_size = [5]
        torch.manual_seed(0)
        inp = torch.randint(16, input_size, device=self.device_type)

        # Without weight tying.
        self.assertNotEqual(
            model.embedding.weight.to_local(), model.fc.weight.to_local()
        )
        output = model(inp)
        output.sum().backward()
        self.assertNotEqual(
            model.embedding.weight.grad.to_local(), model.fc.weight.grad.to_local()
        )
        model.zero_grad()

        # With weight tying.
        model.fc.weight = model.embedding.weight

        self.assertEqual(model.embedding.weight, model.fc.weight)
        self.assertEqual(id(model.embedding.weight), id(model.fc.weight))
        output = model(inp)
        output.sum().backward()
        self.assertEqual(model.embedding.weight.grad, model.fc.weight.grad)
        self.assertEqual(id(model.embedding.weight.grad), id(model.fc.weight.grad))

    @with_comms
    def test_loss_parallel(self):
        device_mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()

        channel_size, channel_dim = 16, 1
        test_setup = [
            (2, (8, channel_size), (8,)),  # calling aten.nll_loss_forward
            (3, (8, channel_size, 12), (8, 12)),  # calling aten.nll_loss2d_forward
        ]
        weight = torch.rand(channel_size, device=self.device_type)
        for input_ndim, input_size, target_size in test_setup:
            x = torch.rand(*input_size, device=self.device_type, requires_grad=True)
            target = torch.randint(channel_size, target_size, device=self.device_type)

            shard_dims = list(range(input_ndim))
            reductions = ["none", "mean", "sum"]
            for shard_dim, reduction in itertools.product(shard_dims, reductions):
                dist_x = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
                y = F.cross_entropy(x, target, weight, reduction=reduction)
                with loss_parallel():
                    if shard_dim == channel_dim:
                        with comm_mode:
                            dist_y = F.cross_entropy(
                                dist_x, target, weight, reduction=reduction
                            )
                            self.assertEqual(comm_mode.get_total_counts(), 3)
                            self.assertEqual(
                                comm_mode.get_comm_counts()[c10d_functional.all_reduce],
                                3,
                            )
                            self.assertTrue(dist_y.placements[0].is_replicate())
                            self.assertEqual(dist_y.to_local(), y)

                        with comm_mode:
                            if reduction == "none":
                                y.sum().backward()
                                dist_y.sum().backward()
                            else:
                                y.backward()
                                dist_y.backward()
                            self.assertEqual(comm_mode.get_total_counts(), 0)
                            self.assertTrue(
                                dist_x.grad.placements[0].is_shard(shard_dim)
                            )
                            self.assertEqual(dist_x.grad.full_tensor(), x.grad)
                        x.grad.zero_()
                    else:
                        with self.assertRaisesRegex(
                            ValueError,
                            "loss_parallel",
                        ):
                            dist_y = F.cross_entropy(
                                dist_x, target, reduction=reduction
                            )


instantiate_parametrized_tests(DistTensorParallelExampleTest)

if __name__ == "__main__":
    run_tests()
