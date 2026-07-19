# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import itertools
from copy import deepcopy
from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._local_tensor import maybe_run_for_local_tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    loss_parallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.distributed.tensor.parallel.input_reshard import input_reshard
from torch.testing._internal.common_device_type import skipXPUIf
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
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
    fwd: dict | None = None
    bwd: dict | None = None
    optim: dict | None = None


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
        from torch.distributed.tensor.experimental import implicit_replication

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
    @skipXPUIf(True, "https://github.com/intel/torch-xpu-ops/issues/1555")
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
            if inp_size[1] % self.world_size != 0:
                raise AssertionError(
                    f"Expected inp_size[1] % world_size == 0, got {inp_size[1]} % {self.world_size}"
                )

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
    @skipXPUIf(True, "https://github.com/intel/torch-xpu-ops/issues/1555")
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
            if inp_size[1] % self.world_size != 0:
                raise AssertionError(
                    f"Expected inp_size[1] % world_size == 0, got {inp_size[1]} % {self.world_size}"
                )

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

        @maybe_run_for_local_tensor
        def assert_not_equal(a, b):
            self.assertNotEqual(a, b)

        assert_not_equal(model.embedding.weight.to_local(), model.fc.weight.to_local())

        output = model(inp)
        output.sum().backward()

        assert_not_equal(
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

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_loss_parallel_multi_dim_mesh(self):
        """Test loss_parallel with multi-dimensional DeviceMesh (e.g. DP + TP)."""
        # Create a 2D mesh: (dp=2, tp=2) on 4 GPUs
        mesh_2d = DeviceMesh(
            self.device_type,
            torch.arange(self.world_size).reshape(2, 2),
            mesh_dim_names=("dp", "tp"),
        )

        channel_size, channel_dim = 16, 1
        x = torch.rand(8, channel_size, device=self.device_type, requires_grad=True)
        target = torch.randint(channel_size, (8,), device=self.device_type)
        weight = torch.rand(channel_size, device=self.device_type)

        # Input: Shard(0) on dp (batch), Shard(1) on tp (vocab/channel)
        dist_x = distribute_tensor(x, mesh_2d, [Shard(0), Shard(channel_dim)])
        # Target: Shard(0) on dp (batch), Replicate on tp
        dist_target = distribute_tensor(target, mesh_2d, [Shard(0), Replicate()])

        # reduction="sum"
        y_sum = F.cross_entropy(x, target, reduction="sum")
        with loss_parallel():
            dist_y = F.cross_entropy(dist_x, dist_target, reduction="sum")
            # Loss should be Partial("sum") on dp dim, Replicate on tp dim
            self.assertEqual(dist_y.placements[0], Partial("sum"))
            self.assertTrue(dist_y.placements[1].is_replicate())
            self.assertEqual(dist_y.full_tensor(), y_sum)

            dist_y.sum().backward()
            y_sum.sum().backward()
            self.assertEqual(dist_x.grad.full_tensor(), x.grad)
        x.grad = None

        # reduction="none": per-sample loss, sharded on dp, replicate on tp.
        y_none = F.cross_entropy(x, target, reduction="none")
        with loss_parallel():
            dist_x_none = distribute_tensor(x, mesh_2d, [Shard(0), Shard(channel_dim)])
            dist_y_none = F.cross_entropy(dist_x_none, dist_target, reduction="none")
            self.assertTrue(dist_y_none.placements[0].is_shard(0))
            self.assertTrue(dist_y_none.placements[1].is_replicate())
            self.assertEqual(dist_y_none.full_tensor(), y_none)

            # Force grad_output to arrive at the backward handler with placements
            # that do NOT match the forward output (Replicate on dp vs.
            # Shard(0) on dp): pass an explicit fully-replicated grad_output via
            # torch.autograd.grad. Without the grad_output.redistribute(...) in
            # _nll_loss_backward_handler, the local shape of grad_output would be
            # the full batch (8,), not the per-rank local batch (4,), and the
            # backward computation would shape-mismatch against x._local_tensor.
            grad_out = distribute_tensor(
                torch.ones_like(y_none), mesh_2d, [Replicate(), Replicate()]
            )
            (grad_x_none,) = torch.autograd.grad(
                outputs=dist_y_none, inputs=dist_x_none, grad_outputs=grad_out
            )
            y_none.sum().backward()
            self.assertTrue(grad_x_none.placements[0].is_shard(0))
            self.assertTrue(grad_x_none.placements[1].is_shard(channel_dim))
            self.assertEqual(grad_x_none.full_tensor(), x.grad)
        x.grad = None

        # reduction="none" with weight arg. Exercise the backward redistribute
        # path by passing an explicit fully-replicated grad_output (weight path
        # goes through the same backward handler but with weight != None).
        y_none_w = F.cross_entropy(x, target, weight, reduction="none")
        with loss_parallel():
            dist_x_none_w = distribute_tensor(
                x, mesh_2d, [Shard(0), Shard(channel_dim)]
            )
            dist_y_none_w = F.cross_entropy(
                dist_x_none_w, dist_target, weight, reduction="none"
            )
            self.assertTrue(dist_y_none_w.placements[0].is_shard(0))
            self.assertTrue(dist_y_none_w.placements[1].is_replicate())
            self.assertEqual(dist_y_none_w.full_tensor(), y_none_w)

            grad_out_w = distribute_tensor(
                torch.ones_like(y_none_w), mesh_2d, [Replicate(), Replicate()]
            )
            (grad_x_none_w,) = torch.autograd.grad(
                outputs=dist_y_none_w, inputs=dist_x_none_w, grad_outputs=grad_out_w
            )
            y_none_w.sum().backward()
            self.assertTrue(grad_x_none_w.placements[0].is_shard(0))
            self.assertTrue(grad_x_none_w.placements[1].is_shard(channel_dim))
            self.assertEqual(grad_x_none_w.full_tensor(), x.grad)
        x.grad = None

        # reduction="sum" with weight arg on multi-dim mesh
        y_weighted = F.cross_entropy(x, target, weight, reduction="sum")
        with loss_parallel():
            dist_x_w = distribute_tensor(x, mesh_2d, [Shard(0), Shard(channel_dim)])
            dist_y_w = F.cross_entropy(dist_x_w, dist_target, weight, reduction="sum")
            self.assertEqual(dist_y_w.placements[0], Partial("sum"))
            self.assertTrue(dist_y_w.placements[1].is_replicate())
            self.assertEqual(dist_y_w.full_tensor(), y_weighted)

            dist_y_w.sum().backward()
            y_weighted.sum().backward()
            self.assertEqual(dist_x_w.grad.full_tensor(), x.grad)

        # reduction="mean" is not supported on multi-dim mesh
        with loss_parallel():
            with self.assertRaisesRegex(NotImplementedError, "one-dimensional"):
                F.cross_entropy(dist_x, dist_target, reduction="mean")

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_loss_parallel_replicate_non_tp_dim(self):
        """Non-TP mesh dim = Replicate (not Shard) must also work."""
        mesh_2d = DeviceMesh(
            self.device_type,
            torch.arange(self.world_size).reshape(2, 2),
            mesh_dim_names=("rep", "tp"),
        )

        channel_size, channel_dim = 16, 1
        x = torch.rand(8, channel_size, device=self.device_type, requires_grad=True)
        target = torch.randint(channel_size, (8,), device=self.device_type)

        # Input: Replicate on the first dim, Shard(channel_dim) on TP
        dist_x = distribute_tensor(x, mesh_2d, [Replicate(), Shard(channel_dim)])
        dist_target = distribute_tensor(target, mesh_2d, [Replicate(), Replicate()])

        # reduction="sum": non-TP dim is Replicate, so it stays Replicate
        # (no Partial rewrite since the corresponding input placement is not Shard).
        y_sum = F.cross_entropy(x, target, reduction="sum")
        with loss_parallel():
            dist_y = F.cross_entropy(dist_x, dist_target, reduction="sum")
            self.assertTrue(dist_y.placements[0].is_replicate())
            self.assertTrue(dist_y.placements[1].is_replicate())
            self.assertEqual(dist_y.full_tensor(), y_sum)

            dist_y.backward()
            y_sum.backward()
            self.assertEqual(dist_x.grad.full_tensor(), x.grad)
        x.grad = None
        dist_x.grad = None

        # reduction="none": target placements = (Replicate(), Replicate()).
        y_none = F.cross_entropy(x, target, reduction="none")
        with loss_parallel():
            dist_y_none = F.cross_entropy(dist_x, dist_target, reduction="none")
            self.assertTrue(dist_y_none.placements[0].is_replicate())
            self.assertTrue(dist_y_none.placements[1].is_replicate())
            self.assertEqual(dist_y_none.full_tensor(), y_none)

            dist_y_none.sum().backward()
            y_none.sum().backward()
            self.assertEqual(dist_x.grad.full_tensor(), x.grad)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_loss_parallel_tp_not_last_dim(self):
        """TP mesh dim need not be the last dim: (tp, dp) ordering must also work."""
        # Create a 2D mesh with TP as the FIRST dim: (tp=2, dp=2)
        mesh_2d = DeviceMesh(
            self.device_type,
            torch.arange(self.world_size).reshape(2, 2),
            mesh_dim_names=("tp", "dp"),
        )

        channel_size, channel_dim = 16, 1
        x = torch.rand(8, channel_size, device=self.device_type, requires_grad=True)
        target = torch.randint(channel_size, (8,), device=self.device_type)

        # Input: Shard(channel_dim) on tp (first), Shard(0) on dp (second)
        dist_x = distribute_tensor(x, mesh_2d, [Shard(channel_dim), Shard(0)])
        # Target: Replicate on tp, Shard(0) on dp
        dist_target = distribute_tensor(target, mesh_2d, [Replicate(), Shard(0)])

        y_sum = F.cross_entropy(x, target, reduction="sum")
        with loss_parallel():
            dist_y = F.cross_entropy(dist_x, dist_target, reduction="sum")
            # TP is at dim 0 -> Replicate; DP at dim 1 -> Partial("sum")
            self.assertTrue(dist_y.placements[0].is_replicate())
            self.assertEqual(dist_y.placements[1], Partial("sum"))
            self.assertEqual(dist_y.full_tensor(), y_sum)

            dist_y.sum().backward()
            y_sum.sum().backward()
            self.assertEqual(dist_x.grad.full_tensor(), x.grad)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_loss_parallel_3d_input_non_batch_shard(self):
        """3-D input (batch, class, seq) with the non-TP mesh dim sharding the
        seq dim (d=2 > channel_dim=1). This exercises the ``d > channel_dim``
        dim-shift in target/output placements (Shard(2) on input → Shard(1) on
        the (batch, seq) target) and the ``nll_loss2d_forward/backward`` path.
        """
        mesh_2d = DeviceMesh(
            self.device_type,
            torch.arange(self.world_size).reshape(2, 2),
            mesh_dim_names=("cp", "tp"),
        )

        batch, channel_size, seq = 4, 16, 6
        channel_dim = 1
        x = torch.rand(
            batch, channel_size, seq, device=self.device_type, requires_grad=True
        )
        target = torch.randint(channel_size, (batch, seq), device=self.device_type)

        # Input: Shard(seq) on cp (non-TP), Shard(class) on tp.
        dist_x = distribute_tensor(x, mesh_2d, [Shard(2), Shard(channel_dim)])
        # Target is (batch, seq); the seq-sharded input dim shifts down to 1.
        dist_target = distribute_tensor(target, mesh_2d, [Shard(1), Replicate()])

        # reduction="sum"
        y_sum = F.cross_entropy(x, target, reduction="sum")
        with loss_parallel():
            dist_y = F.cross_entropy(dist_x, dist_target, reduction="sum")
            self.assertEqual(dist_y.placements[0], Partial())
            self.assertTrue(dist_y.placements[1].is_replicate())
            self.assertEqual(dist_y.full_tensor(), y_sum)

            dist_y.sum().backward()
            y_sum.sum().backward()
            self.assertEqual(dist_x.grad.full_tensor(), x.grad)
        x.grad = None

        # reduction="none": output shape (batch, seq); target placements
        # (Shard(1), Replicate()) are inherited directly.
        y_none = F.cross_entropy(x, target, reduction="none")
        with loss_parallel():
            dist_x_none = distribute_tensor(x, mesh_2d, [Shard(2), Shard(channel_dim)])
            dist_y_none = F.cross_entropy(dist_x_none, dist_target, reduction="none")
            self.assertTrue(dist_y_none.placements[0].is_shard(1))
            self.assertTrue(dist_y_none.placements[1].is_replicate())
            self.assertEqual(dist_y_none.full_tensor(), y_none)

            # Exercise the backward redistribute path with a mismatched
            # (fully-replicated) grad_output.
            grad_out = distribute_tensor(
                torch.ones_like(y_none), mesh_2d, [Replicate(), Replicate()]
            )
            (grad_x_none,) = torch.autograd.grad(
                outputs=dist_y_none, inputs=dist_x_none, grad_outputs=grad_out
            )
            y_none.sum().backward()
            self.assertTrue(grad_x_none.placements[0].is_shard(2))
            self.assertTrue(grad_x_none.placements[1].is_shard(channel_dim))
            self.assertEqual(grad_x_none.full_tensor(), x.grad)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_loss_parallel_invalid_non_tp_placement(self):
        """Non-TP mesh dim with a placement that is neither Shard nor Replicate is rejected."""
        mesh_2d = DeviceMesh(
            self.device_type,
            torch.arange(self.world_size).reshape(2, 2),
            mesh_dim_names=("dp", "tp"),
        )

        channel_size, channel_dim = 16, 1
        local_x = torch.rand(
            4, channel_size // 2, device=self.device_type, requires_grad=True
        )
        local_target = torch.randint(channel_size, (4,), device=self.device_type)

        # Force a Partial placement on the non-TP (dp) mesh dim via from_local.
        dist_x = DTensor.from_local(
            local_x, mesh_2d, [Partial("sum"), Shard(channel_dim)], run_check=False
        )
        dist_target = DTensor.from_local(
            local_target, mesh_2d, [Replicate(), Replicate()], run_check=False
        )

        with loss_parallel():
            for reduction in ("sum", "none", "mean"):
                with self.assertRaisesRegex(ValueError, "Shard or Replicate"):
                    F.cross_entropy(dist_x, dist_target, reduction=reduction)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_loss_parallel_plain_tensor_target_rejected_on_multi_dim(self):
        """On multi-dim mesh with a batch-sharded non-TP dim, a plain torch.Tensor
        target is ambiguous (full global vs. local slice) and must be rejected.
        """
        mesh_2d = DeviceMesh(
            self.device_type,
            torch.arange(self.world_size).reshape(2, 2),
            mesh_dim_names=("dp", "tp"),
        )

        channel_size, channel_dim = 16, 1
        x = torch.rand(8, channel_size, device=self.device_type, requires_grad=True)
        target = torch.randint(channel_size, (8,), device=self.device_type)

        dist_x = distribute_tensor(x, mesh_2d, [Shard(0), Shard(channel_dim)])

        with loss_parallel():
            with self.assertRaisesRegex(ValueError, "requires a DTensor"):
                F.cross_entropy(dist_x, target, reduction="sum")


instantiate_parametrized_tests(DistTensorParallelExampleTest)

DistTensorParallelExampleTestWithLocalTensor = create_local_tensor_test_class(
    DistTensorParallelExampleTest,
)

if __name__ == "__main__":
    run_tests()
