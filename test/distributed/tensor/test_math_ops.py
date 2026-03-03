# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import copy
import itertools
from pprint import pformat
from typing import NamedTuple

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor._ops.utils import is_tensor_partial, normalize_dim
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
    SequenceParallel,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorTestBase,
    map_local_for_rank,
    skip_unless_torch_gpu,
    with_comms,
)


funcol = torch.ops.c10d_functional


class DistMathOpsTest(DTensorTestBase):
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

    def linear_op_reductions(self, op_str):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        tensor = torch.randn(12, 8, 8)
        if op_str in ("any", "all"):
            # Test bool tensor for any() and all() reduction ops
            # Previously all() had a bug using sum reduction instead of product
            tensor = tensor < 0
        dtensor = distribute_tensor(tensor, device_mesh, shard_spec)

        op = getattr(tensor, op_str)
        op_dt = getattr(dtensor, op_str)

        keep_dim_or_not = [True, False, None]
        for dim in range(tensor.ndim):
            for keep_dim in keep_dim_or_not:
                args = (dim, keep_dim) if keep_dim is not None else (dim,)
                if op_str in ("max", "min"):
                    # min and max return a tuple when dim specified
                    dim_reduced_tensor, _ = op(*args)
                    dt_reduced, _ = op_dt(*args)
                else:
                    dim_reduced_tensor = op(*args)
                    dt_reduced = op_dt(*args)
                dt_dim_reduced_tensor = dt_reduced.full_tensor()
                self.assertEqual(dt_dim_reduced_tensor, dim_reduced_tensor)

        full_reduced_tensor = op()
        dt_full_reduced = op_dt().full_tensor()
        self.assertEqual(dt_full_reduced, full_reduced_tensor)

    @with_comms
    def test_linear_op_reductions(self):
        for op_str in (
            "all",
            "sum",
            "prod",
            "max",
            "min",
            "any",
            "amax",
            "amin",
            "var",
            "std",
        ):
            self.linear_op_reductions(op_str)

    @with_comms
    @skip_unless_torch_gpu
    def test_mean(self):
        self.linear_op_reductions("mean")

    # TODO: forward test can be removed once test_softmax_with_bwd passes on CPU
    @with_comms
    def test_softmax_fwd(self):
        device_mesh = self.build_device_mesh()

        x = torch.rand(8, 12, 16, device=self.device_type)
        dims = range(3)  # used to convert -1 to the actual dim
        softmax_dims = [-1, 0, 1, 2]
        shard_dims = [-1, 0, 1, 2]
        test_list = list(itertools.product(softmax_dims, shard_dims))

        for softmax_dim, shard_dim in test_list:
            local_y = torch.nn.functional.softmax(
                x, dim=softmax_dim, dtype=torch.float32
            )
            dist_x = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
            dist_y = torch.nn.functional.softmax(
                dist_x, dim=softmax_dim, dtype=torch.float32
            )
            shard_dim = normalize_dim(shard_dim, dist_x.ndim)
            if dims[shard_dim] == dims[softmax_dim]:
                self.assertTrue(dist_y.placements[0].is_replicate())
                self.assertEqual(dist_y.to_local(), local_y)
            else:
                self.assertTrue(dist_y.placements[0].is_shard(dim=shard_dim))
                self.assertEqual(dist_y.full_tensor(), local_y)

    # TODO: get test_softmax_with_bwd pass on CPU
    # DTensor's _softmax_backward_data produces wrong result on CPU on certain dimension.
    # fail_on_cpu_list = [(0, -1), (1, -1)]
    @with_comms
    @skip_unless_torch_gpu
    def test_softmax_with_bwd(self):
        device_mesh = self.build_device_mesh()

        dims = range(3)  # used to convert -1 to the actual dim
        softmax_dims = [-1, 0, 1, 2]
        shard_dims = [-1, 0, 1, 2]
        test_list = list(itertools.product(softmax_dims, shard_dims))

        for params in test_list:
            softmax_dim, shard_dim = params
            x = torch.rand(8, 12, 16, device=self.device_type, requires_grad=True)
            self.assertTrue(x.requires_grad)
            local_y = torch.nn.functional.softmax(
                x, dim=softmax_dim, dtype=torch.float32
            ).sum()
            local_y.backward()

            dist_x = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
            self.assertTrue(dist_x.requires_grad)
            dist_softmax = dist_x.softmax(dim=softmax_dim)
            shard_dim = normalize_dim(shard_dim, dist_x.ndim)
            if dims[softmax_dim] == dims[shard_dim]:
                self.assertTrue(dist_softmax.placements[0].is_replicate())
            else:
                self.assertTrue(dist_softmax.placements[0].is_shard(dim=shard_dim))
            dist_y = dist_softmax.sum()
            if dims[softmax_dim] == dims[shard_dim]:
                self.assertTrue(dist_y.placements[0].is_replicate())
            else:
                self.assertTrue(dist_y.placements[0].is_partial())
                dist_y = dist_y.redistribute(device_mesh, [Replicate()])
            self.assertEqual(dist_y.to_local(), local_y)
            self.assertIsNone(dist_x.grad)
            dist_y.backward()
            self.assertIsNotNone(dist_x.grad)
            if dims[softmax_dim] == dims[shard_dim]:
                self.assertTrue(dist_x.grad.placements[0].is_replicate())
            else:
                self.assertTrue(dist_x.grad.placements[0].is_shard(dim=shard_dim))
            self.assertEqual(dist_x.grad.full_tensor(), x.grad)

    @with_comms
    @skip_unless_torch_gpu
    def test_nll_loss_and_cross_entropy(self):
        device_mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()

        channel_size, channel_dim = 16, 1
        test_setup = [
            (2, (8, channel_size), (8,)),  # calling aten.nll_loss_forward
            (3, (8, channel_size, 12), (8, 12)),  # calling aten.nll_loss2d_forward
        ]
        for input_ndim, input_size, target_size in test_setup:
            x = torch.rand(*input_size, device=self.device_type, requires_grad=True)
            target = torch.randint(channel_size, target_size, device=self.device_type)
            dist_target = distribute_tensor(target, device_mesh, [Replicate()])

            shard_dims = list(range(input_ndim))
            reductions = ["none", "mean", "sum"]
            # Compared with nll_loss, cross_entropy additionally calls log_softmax first.
            # Testing them together as code can be reused.
            loss_functions = [
                torch.nn.functional.nll_loss,
                torch.nn.functional.cross_entropy,
            ]
            for shard_dim, reduction, loss_fn in itertools.product(
                shard_dims, reductions, loss_functions
            ):
                dist_x = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
                y = loss_fn(x, target, reduction=reduction)
                if reduction == "none":
                    y.sum().backward()
                else:
                    y.backward()
                with comm_mode:
                    dist_y = loss_fn(dist_x, dist_target, reduction=reduction)
                    if shard_dim == channel_dim:
                        self.assertEqual(comm_mode.get_total_counts(), 1)
                        self.assertEqual(
                            comm_mode.get_comm_counts()[funcol.all_gather_into_tensor],
                            1,
                        )
                        self.assertTrue(dist_y.placements[0].is_replicate())
                        self.assertEqual(dist_y.to_local(), y)
                    else:
                        self.assertEqual(comm_mode.get_total_counts(), 0)
                        if reduction == "none":
                            output_shard_dim = (
                                shard_dim if shard_dim < channel_dim else shard_dim - 1
                            )
                            self.assertTrue(
                                dist_y.placements[0].is_shard(dim=output_shard_dim)
                            )
                        else:
                            self.assertTrue(dist_y.placements[0].is_partial())
                        self.assertEqual(dist_y.full_tensor(), y)

                    if reduction == "none":
                        dist_y.sum().backward()
                    else:
                        dist_y.backward()
                    if shard_dim == channel_dim:
                        self.assertTrue(dist_x.grad.placements[0].is_replicate())
                        self.assertEqual(dist_x.grad.to_local(), x.grad)
                    else:
                        self.assertTrue(
                            dist_x.grad.placements[0].is_shard(dim=shard_dim)
                        )
                        self.assertEqual(dist_x.grad.full_tensor(), x.grad)
                    x.grad.zero_()

    @with_comms
    def test_shard_math_ops(self):
        mesh_shape = (2, self.world_size // 2)
        mesh = DeviceMesh(
            self.device_type,
            torch.arange(self.world_size).reshape(*mesh_shape),
        )
        global_tensor = torch.ones(4, 4)
        double_shard_tensor = distribute_tensor(
            global_tensor, mesh, [Shard(0), Shard(0)]
        )
        fully_shard_tensor = distribute_tensor(
            global_tensor, mesh, [Shard(0), Shard(1)]
        )

        # for op in [torch.add, torch.sub, torch.mul, torch.div]:
        for op in [torch.add, torch.sub, torch.mul, torch.div]:
            expect_rs = op(global_tensor, 2)
            double_shard_full_tensor = op(double_shard_tensor, 2).full_tensor()
            self.assertEqual(double_shard_full_tensor, expect_rs)

            fully_shard_full_tensor = op(fully_shard_tensor, 2).full_tensor()
            self.assertEqual(fully_shard_full_tensor, expect_rs)

    @with_comms
    def test_layer_norm_fwd(self):
        device_mesh = self.build_device_mesh()

        # NLP example from pytorch docs
        # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        batch, sentence_length, embedding_dim = 20, 5, 10
        x = torch.rand(batch, sentence_length, embedding_dim, device=self.device_type)
        norm_shape_idx_list = list(range(x.ndim))
        shard_dims = [-1, 0, 1, 2]
        elementwise_affine_list = [False, True]

        # Test RMSNorm as well if CUDA
        norm_types = [torch.nn.LayerNorm]
        if self.device_type == "cuda" and hasattr(torch.nn, "RMSNorm"):
            norm_types.append(torch.nn.RMSNorm)

        test_config_list = list(
            itertools.product(
                norm_types, shard_dims, norm_shape_idx_list, elementwise_affine_list
            )
        )

        # normalized shape is a torch.Size object
        for norm_type, shard_dim, norm_idx, elementwise_affine in test_config_list:
            normalized_shape = x.shape[norm_idx:]
            layer_norm = norm_type(
                normalized_shape,
                elementwise_affine=elementwise_affine,
                device=self.device_type,
            )
            layer_norm_local = copy.deepcopy(layer_norm).to(self.device_type)

            def _replicate_fn(name, module, device_mesh):
                for name, param in module.named_parameters():
                    # RMSNorm only has weight, LayerNorm has both weight and bias
                    if name in ["weight", "bias"]:
                        param_dist = torch.nn.Parameter(
                            distribute_tensor(param, device_mesh, [Replicate()])
                        )
                        module.register_parameter(name, param_dist)

            layer_norm_dist = distribute_module(layer_norm, device_mesh, _replicate_fn)

            x_local = x
            x_dist = distribute_tensor(x, device_mesh, [Shard(shard_dim)])

            y_local = layer_norm_local(x_local)
            # make sure that forward layer norm does not introduce extra collectives
            comm_mode = CommDebugMode()
            with comm_mode:
                y_dist = layer_norm_dist(x_dist)

            self.assertLessEqual(
                comm_mode.get_total_counts(),
                1,  # TODO: This should be 0!
                f"comm count={comm_mode.get_total_counts()}, norm_type={norm_type.__name__}, "
                f"shard_dim={shard_dim}, norm_shape={normalized_shape}, elem_affine={elementwise_affine}",
            )

            from torch.distributed.tensor._dtensor_spec import TensorMeta

            dtensor_meta = y_dist._spec.tensor_meta
            if not isinstance(dtensor_meta, TensorMeta):
                raise AssertionError(f"Expected TensorMeta, got {type(dtensor_meta)}")
            # make sure the right shape in sharding prop
            self.assertEqual(y_local.shape, dtensor_meta.shape)
            self.assertEqual(y_local, y_dist.full_tensor())

    @with_comms
    def test_layer_norm_bwd(self):
        device_mesh = self.build_device_mesh()

        # NLP example from pytorch docs
        # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        batch, sentence_length, embedding_dim = 20, 5, 10
        norm_shape_idx_list = list(range(3))
        shard_dims = [0, 1, 2]
        elementwise_affine_list = [False, True]

        # Test both LayerNorm and RMSNorm (if CUDA)
        norm_types = [torch.nn.LayerNorm]
        if self.device_type == "cuda" and hasattr(torch.nn, "RMSNorm"):
            norm_types.append(torch.nn.RMSNorm)

        test_config_list = list(
            itertools.product(
                norm_types, shard_dims, norm_shape_idx_list, elementwise_affine_list
            )
        )

        # normalized shape is a torch.Size object
        for norm_type, shard_dim, norm_idx, elementwise_affine in test_config_list:
            x = torch.rand(
                batch,
                sentence_length,
                embedding_dim,
                device=self.device_type,
                requires_grad=True,
            )
            normalized_shape = x.shape[norm_idx:]
            layer_norm = norm_type(
                normalized_shape,
                elementwise_affine=elementwise_affine,
                device=self.device_type,
            )
            layer_norm_local = copy.deepcopy(layer_norm).to(self.device_type)

            def _replicate_fn(name, module, device_mesh):
                for name, param in module.named_parameters():
                    if name in ["weight", "bias"]:
                        param_dist = torch.nn.Parameter(
                            distribute_tensor(param, device_mesh, [Replicate()])
                        )
                        module.register_parameter(name, param_dist)

            layer_norm_dist = distribute_module(layer_norm, device_mesh, _replicate_fn)

            if elementwise_affine:
                self.assertEqual(
                    layer_norm_local.weight, layer_norm_dist.weight.full_tensor()
                )
                # RMSNorm doesn't have bias
                if hasattr(layer_norm_local, "bias"):
                    self.assertEqual(
                        layer_norm_local.bias, layer_norm_dist.bias.full_tensor()
                    )

            x_local = x.detach().clone().requires_grad_(True)
            x_dist = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
            self.assertEqual(x_local, x_dist.full_tensor())

            y_local = layer_norm_local(x_local)
            # make sure that backward layer norm does not introduce extra collectives
            comm_mode = CommDebugMode()
            with comm_mode:
                y_dist = layer_norm_dist(x_dist)
                y_dist.sum().backward()

            expected_fwd_comm = 0 if shard_dim < norm_idx else 1

            self.assertEqual(
                sum(comm_mode.comm_module_counts["Global"]["forward"].values()),
                expected_fwd_comm,
                f"comm count={comm_mode.get_total_counts()}, norm_type={norm_type.__name__}, "
                f"shard_dim={shard_dim}, norm_shape={normalized_shape}, elem_affine={elementwise_affine}",
            )

            self.assertEqual(y_local, y_dist.full_tensor())

            # backward step
            y_local.sum().backward()

            expected_bwd_comm = 0 if shard_dim < norm_idx else 1

            self.assertEqual(
                sum(comm_mode.comm_module_counts["Global"]["backward"].values()),
                expected_bwd_comm,
                f"comm count={comm_mode.get_total_counts()}, norm_type={norm_type.__name__}, "
                f"shard_dim={shard_dim}, norm_shape={normalized_shape}, elem_affine={elementwise_affine}",
            )

            if elementwise_affine:
                # if input is sharded on any outer dimension, the gradient of weight
                # and bias should be Partial
                dim_map = x_dist._spec.dim_map
                outer_dims = range(norm_idx)
                needs_reduction = any(dim_map[d] >= 0 for d in outer_dims)
                self.assertEqual(
                    is_tensor_partial(layer_norm_dist.weight.grad._spec),
                    needs_reduction,
                )
                # RMSNorm doesn't have bias
                if hasattr(layer_norm_dist, "bias"):
                    self.assertEqual(
                        is_tensor_partial(layer_norm_dist.bias.grad._spec),
                        needs_reduction,
                    )
                self.assertEqual(
                    layer_norm_local.weight.grad,
                    layer_norm_dist.weight.grad.full_tensor(),
                )
                # RMSNorm doesn't have bias
                if hasattr(layer_norm_local, "bias"):
                    self.assertEqual(
                        layer_norm_local.bias.grad,
                        layer_norm_dist.bias.grad.full_tensor(),
                    )

            self.assertEqual(x_local.grad, x_dist.grad.full_tensor())

    @with_comms
    def test_layer_norm_bwd_req_grad(self):
        device_mesh = self.build_device_mesh()
        batch, seq_len, embedding_dim, vocab_size = 8, 8, 10, 32

        # Test both LayerNorm and RMSNorm (if CUDA)
        norm_types = [torch.nn.LayerNorm]
        if self.device_type == "cuda" and hasattr(torch.nn, "RMSNorm"):
            norm_types.append(torch.nn.RMSNorm)

        # build our subtest configurations and filter out invalid ones
        class SubTest(NamedTuple):
            norm_type: type
            multidim_norm: bool
            elementwise_affine: bool
            emb_req_grad: bool
            ln_req_grad: bool
            out_req_grad: bool

        subtest_fails = {}

        def valid_filter(cfg):
            return not (cfg.ln_req_grad and not cfg.elementwise_affine) and any(cfg[3:])

        subtest_cfgs = list(
            filter(
                valid_filter,
                [
                    SubTest(norm_type, *cfg)
                    for norm_type in norm_types
                    for cfg in itertools.product(*(((False, True),) * 5))
                ],
            )
        )

        for subtest_cfg in subtest_cfgs:
            try:
                (
                    norm_type,
                    multidim_norm,
                    elementwise_affine,
                    emb_req_grad,
                    ln_req_grad,
                    out_req_grad,
                ) = subtest_cfg
                normalized_shape = (
                    (seq_len, embedding_dim) if multidim_norm else (embedding_dim,)
                )

                # configure our local and parallelized models for this subtest
                class LnTpBlock(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.preln_embeddings = torch.nn.Embedding(
                            vocab_size, embedding_dim
                        )
                        self.layer_norm = norm_type(
                            normalized_shape, elementwise_affine=elementwise_affine
                        )
                        self.postln_linear = torch.nn.Linear(
                            embedding_dim, embedding_dim
                        )

                    def forward(self, tokens):
                        h = self.preln_embeddings(tokens)
                        h = self.layer_norm(h)
                        output = self.postln_linear(h)
                        return output

                parallel_plan = {
                    "preln_embeddings": RowwiseParallel(
                        input_layouts=Replicate(), output_layouts=Shard(1)
                    ),
                    "layer_norm": SequenceParallel(),
                    "postln_linear": ColwiseParallel(
                        input_layouts=Shard(1),
                        output_layouts=Replicate(),
                    ),
                }

                model = LnTpBlock()
                model_local = copy.deepcopy(model).to(device=self.device_type)
                model_dist = parallelize_module(model, device_mesh, parallel_plan)
                req_grad_map = {
                    "preln_embeddings": emb_req_grad,
                    "postln_linear": out_req_grad,
                    "layer_norm": ln_req_grad,
                }

                # apply the relevant `requires_grad` mask for this subtest to both models
                for target_model in [model_local, model_dist]:
                    for n, p in target_model.named_parameters():
                        if not req_grad_map.get(n.rpartition(".")[0], False):
                            p.requires_grad_(False)
                            if p.requires_grad:
                                raise AssertionError(
                                    f"Expected requires_grad to be False for {n}"
                                )
                        else:
                            if not p.requires_grad:
                                raise AssertionError(
                                    f"Expected requires_grad to be True for {n}"
                                )

                # forward step for both local and distributed models
                x = torch.randint(vocab_size, (batch, seq_len), device=self.device_type)
                x_local = x.detach().clone()
                output_local = model_local(x_local)

                with CommDebugMode() as comm_mode:
                    output_dist = model_dist(x)

                self.assertEqual(output_local, output_dist)

                # all requires_grad patterns should have the same forward comm counts
                expected_fwd_comm = {
                    funcol.reduce_scatter_tensor: 1,
                    funcol.all_gather_into_tensor: 2,
                }
                self.assertDictEqual(
                    comm_mode.comm_module_counts["Global"]["forward"], expected_fwd_comm
                )

                # backward step
                output_local.sum().backward()

                with CommDebugMode() as comm_mode:
                    output_dist.sum().backward()

                # ensure gradients (and parameters) remain equal between local and distributed models
                self._check_module(model_local, model_dist, check_grad=True)

                # different requires_grad patterns will have different bwd comm counts
                if out_req_grad and not any((emb_req_grad, ln_req_grad)):
                    expected_bwd_comm = {}
                elif ln_req_grad and not any((emb_req_grad, multidim_norm)):
                    expected_bwd_comm = {funcol.reduce_scatter_tensor: 1}
                elif multidim_norm:
                    expected_bwd_comm = {funcol.all_reduce: 1}
                    expected_bwd_comm[funcol.all_gather_into_tensor] = (
                        2 if emb_req_grad else 1
                    )
                else:
                    expected_bwd_comm = {
                        funcol.reduce_scatter_tensor: 1,
                        funcol.all_gather_into_tensor: 1,
                    }

                self.assertDictEqual(
                    comm_mode.comm_module_counts["Global"]["backward"],
                    expected_bwd_comm,
                )
                self.assertEqual(output_local, output_dist)

            except Exception as e:
                subtest_fails[subtest_cfg] = e
        # if any subtest fails, provide the failed subtests and report the overall failure
        if subtest_fails:
            raise AssertionError(
                f"{len(subtest_fails)}/{len(subtest_cfgs)} subtests failed: {pformat(subtest_fails)}"
            )

    @with_comms
    def test_topk(self):
        device_mesh = self.build_device_mesh()
        placement_combs = [Shard(0), Shard(1), Shard(2), Replicate()]

        comm_mode = CommDebugMode()

        tensor = torch.randn(12, 8, 8, requires_grad=True)
        global_topk = tensor.topk(3, dim=0)

        for placement in placement_combs:
            dtensor = distribute_tensor(tensor, device_mesh, (placement,))
            with comm_mode:
                out_dt = dtensor.topk(3, dim=0)
            if placement.is_shard(0):
                self.assertEqual(comm_mode.get_total_counts(), 1)
                self.assertEqual(
                    comm_mode.get_comm_counts()[funcol.all_gather_into_tensor],
                    1,
                )
            out_full_values = out_dt.values.full_tensor()
            self.assertEqual(global_topk.values, out_full_values)

            # TODO: support backward scatter
            # global_topk.values.sum().backward()
            # out_full_values.sum().backward()

    @with_comms
    def test_shard0_svd(self):
        device_mesh = self.build_device_mesh()
        torch.manual_seed(42)
        replicated_x = torch.randn((8, 8), device=self.device_type)
        sharded_x = distribute_tensor(replicated_x, device_mesh, (Shard(0),))
        with CommDebugMode() as comm_mode:
            U, S, V = torch.linalg.svd(sharded_x, full_matrices=False)
        ref_U, ref_S, ref_V = torch.linalg.svd(replicated_x, full_matrices=False)
        self.assertEqual(U.to_local(), ref_U)
        self.assertEqual(S.to_local(), ref_S)
        self.assertEqual(V.to_local(), ref_V)
        comm_counts = comm_mode.get_comm_counts()
        self.assertEqual(len(comm_counts), 1)
        self.assertEqual(comm_counts[funcol.all_gather_into_tensor], 1)

    @with_comms
    def test_vector_norm(self):
        device_mesh = self.build_device_mesh()

        grad = torch.randn(12, 8)

        sharded_grad = distribute_tensor(grad, device_mesh, [Shard(0)])

        # non-sharded op
        out = torch.ops.aten.linalg_vector_norm(grad, 2)

        # sharded op
        sharded_out = torch.ops.aten.linalg_vector_norm(sharded_grad, 2)

        self.assertEqual(sharded_out.full_tensor(), out)

    @with_comms
    def test_vector_norm_partial(self):
        device_mesh = self.build_device_mesh()

        all_ranks = list(range(self.world_size))

        local_grad = map_local_for_rank(
            self.rank, lambda rank: torch.tensor([rank, 1], dtype=torch.float32)
        )
        full_grad = torch.tensor([sum(all_ranks), self.world_size], dtype=torch.float32)

        partial_grad = DTensor.from_local(local_grad, device_mesh, [Partial()])

        # full result
        out = torch.ops.aten.linalg_vector_norm(full_grad, 2)

        # partial result
        partial_out = torch.ops.aten.linalg_vector_norm(partial_grad, 2)
        self.assertEqual(partial_out.full_tensor(), out)

    @with_comms
    def test_foreach_norm(self):
        device_mesh = self.build_device_mesh()

        grad0 = torch.randn(12, 8)
        grad1 = torch.randn(8, 8)

        sharded_grad0 = distribute_tensor(grad0, device_mesh, [Shard(0)])
        sharded_grad1 = distribute_tensor(grad1, device_mesh, [Shard(0)])

        # non-sharded op
        out = torch.ops.aten._foreach_norm([grad0, grad1], 2)

        # sharded op
        sharded_out = torch.ops.aten._foreach_norm([sharded_grad0, sharded_grad1], 2)

        for o, so in zip(out, sharded_out):
            self.assertEqual(so.full_tensor(), o)

    @with_comms
    def test_foreach_norm_partial(self):
        device_mesh = self.build_device_mesh()

        all_ranks = list(range(self.world_size))

        local_grad0 = map_local_for_rank(
            self.rank, lambda rank: torch.tensor([rank, 1], dtype=torch.float32)
        )
        local_grad1 = map_local_for_rank(
            self.rank, lambda rank: torch.tensor([rank + 1, 2], dtype=torch.float32)
        )

        grad0 = torch.tensor([sum(all_ranks), self.world_size], dtype=torch.float32)
        grad1 = torch.tensor(
            [sum(all_ranks) + self.world_size, 2 * self.world_size], dtype=torch.float32
        )

        partial_grad0 = DTensor.from_local(local_grad0, device_mesh, [Partial()])
        partial_grad1 = DTensor.from_local(local_grad1, device_mesh, [Partial()])

        # full result
        out = torch.ops.aten._foreach_norm([grad0, grad1], 2)

        # partial result
        partial_out = torch.ops.aten._foreach_norm([partial_grad0, partial_grad1], 2)

        for o, po in zip(out, partial_out):
            self.assertEqual(po.full_tensor(), o)

    @with_comms
    def test_powsum_sharded(self):
        """Test that linalg__powsum produces Partial(sum) placement for sharded input."""
        device_mesh = self.build_device_mesh()

        torch.manual_seed(42)
        grad = torch.randn(12, 8)
        sharded_grad = distribute_tensor(grad, device_mesh, [Shard(0)])

        # Test multiple ord values to validate output placements
        for ord in [1, 2, 3]:
            # powsum computes sum(|x|^ord) without the root
            sharded_out = torch.ops.aten.linalg__powsum(sharded_grad, ord)

            # The placement should be Partial("sum")
            self.assertEqual(sharded_out.placements, (Partial("sum"),))

            # Expected: sum(|x|^ord) over all elements
            expected = (grad.abs() ** ord).sum()
            self.assertEqual(sharded_out.full_tensor(), expected)

    @with_comms
    def test_vector_norm_special_norms_placement(self):
        """Test that inf/-inf/0/1 norms produce correct Partial placements."""
        device_mesh = self.build_device_mesh()

        torch.manual_seed(42)
        grad = torch.randn(12, 8).abs() + 0.1  # Ensure positive for proper test
        sharded_grad = distribute_tensor(grad, device_mesh, [Shard(0)])

        # Test inf norm -> Partial("max")
        out_inf = torch.ops.aten.linalg_vector_norm(sharded_grad, float("inf"))
        self.assertEqual(out_inf.full_tensor(), grad.max())
        self.assertTrue(out_inf.placements[0].is_partial())
        self.assertEqual(out_inf.placements[0].reduce_op, "max")

        # Test -inf norm -> Partial("min")
        out_neginf = torch.ops.aten.linalg_vector_norm(sharded_grad, float("-inf"))
        self.assertEqual(out_neginf.full_tensor(), grad.min())
        self.assertTrue(out_neginf.placements[0].is_partial())
        self.assertEqual(out_neginf.placements[0].reduce_op, "min")

        # Test 1-norm -> Partial("sum")
        out_1 = torch.ops.aten.linalg_vector_norm(sharded_grad, 1)
        self.assertEqual(out_1.full_tensor(), grad.abs().sum())
        self.assertTrue(out_1.placements[0].is_partial())
        self.assertEqual(out_1.placements[0].reduce_op, "sum")

        # Test 0-norm -> Partial("sum")
        out_0 = torch.ops.aten.linalg_vector_norm(sharded_grad, 0)
        self.assertEqual(out_0.full_tensor(), (grad != 0).sum().float())
        self.assertTrue(out_0.placements[0].is_partial())
        self.assertEqual(out_0.placements[0].reduce_op, "sum")

    @with_comms
    def test_foreach_powsum_sharded(self):
        """Test that _foreach_powsum produces correct values and placements for sharded input."""
        device_mesh = self.build_device_mesh()

        torch.manual_seed(42)
        grad0 = torch.randn(12, 8)
        grad1 = torch.randn(8, 8)

        sharded_grad0 = distribute_tensor(grad0, device_mesh, [Shard(0)])
        sharded_grad1 = distribute_tensor(grad1, device_mesh, [Shard(0)])

        # Test multiple ord values to validate output placements
        for ord in [1, 2, 3]:
            sharded_out = torch.ops.aten._foreach_powsum(
                [sharded_grad0, sharded_grad1], ord
            )

            # Output should be Partial("sum") since powsum computes sum(|x|^ord)
            self.assertEqual(sharded_out[0].placements, (Partial("sum"),))
            self.assertEqual(sharded_out[1].placements, (Partial("sum"),))

            # Check values: sum(|x|^ord) for each tensor
            expected0 = (grad0.abs() ** ord).sum()
            expected1 = (grad1.abs() ** ord).sum()

            self.assertEqual(sharded_out[0].full_tensor(), expected0)
            self.assertEqual(sharded_out[1].full_tensor(), expected1)

    @with_comms
    def test_foreach_norm_different_mesh(self):
        mesh_shape = (2, self.world_size // 2)
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=("x", "y")
        )

        mesh_x = mesh_2d["x"]
        mesh_y = mesh_2d["y"]

        torch.manual_seed(0)

        grad0 = torch.randn(12, 8)
        grad1 = torch.randn(8, 8)

        replica_grad0 = DTensor.from_local(grad0, mesh_x, [Replicate()])
        replica_grad1 = DTensor.from_local(grad1, mesh_y, [Replicate()])

        # could run sharded op without error
        out_tuple = torch.ops.aten._foreach_norm([replica_grad0, replica_grad1], 2)

        grad0_norm = out_tuple[0]
        grad1_norm = out_tuple[1]
        self.assertEqual(grad0_norm.device_mesh, mesh_x)
        self.assertEqual(grad1_norm.device_mesh, mesh_y)

    @with_comms
    def test_norm_0_on_psum(self):
        # L0 norm on P(sum) should not propagate -> P(sum), should replicate
        device_mesh = self.build_device_mesh()
        t = torch.tensor([0, 1, 0, 3, 0, 5], device=self.device_type).float()
        dt = distribute_tensor(t, device_mesh, [Partial()])
        out = torch.ops.aten.linalg_vector_norm(dt, 0)
        self.assertEqual(out.full_tensor().item(), 3.0)
        self.assertEqual(out.placements, (Replicate(),))

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_foreach_add_different_mesh(self):
        mesh_shape = (2, self.world_size // 2)
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=("x", "y")
        )

        mesh_x = mesh_2d["x"]
        mesh_y = mesh_2d["y"]

        inp00 = torch.ones(4, 8) * 2
        inp01 = torch.ones(8, 8) * 3
        inp10 = torch.ones(4, 8) * 4
        inp11 = torch.ones(8, 8) * 3

        replica_inp00 = DTensor.from_local(inp00, mesh_x, [Shard(0)])
        replica_inp01 = DTensor.from_local(inp01, mesh_x, [Replicate()])
        replica_inp10 = DTensor.from_local(inp10, mesh_y, [Shard(0)])
        replica_inp11 = DTensor.from_local(inp11, mesh_y, [Replicate()])

        # zipped foreach, could run sharded op without error
        out_tuple = torch.ops.aten._foreach_add(
            [replica_inp00, replica_inp10], [replica_inp01, replica_inp11]
        )

        out0, out1 = out_tuple
        self.assertEqual(out0.device_mesh, mesh_x)
        self.assertEqual(out1.device_mesh, mesh_y)

        with self.assertRaisesRegex(RuntimeError, "Sharding propagation failed"):
            torch.ops.aten._foreach_add(
                [replica_inp00, replica_inp01], [replica_inp10, replica_inp11]
            )

    @with_comms
    def test_foreach_compose(self):
        """Test composing multiple foreach operations."""
        device_mesh = self.build_device_mesh()
        local_shards = tuple(torch.randn(4, 8) for _ in range(3))
        dt_inputs = tuple(
            distribute_tensor(shard, device_mesh, [Shard(0)]) for shard in local_shards
        )
        dt_abs = torch._foreach_abs(dt_inputs)
        dt_max = torch._foreach_max(dt_abs)

        abs = torch._foreach_abs(local_shards)
        expected_max = torch._foreach_max(abs)

        for max_val, expected in zip(dt_max, expected_max):
            self.assertEqual(max_val.full_tensor(), expected)

    @with_comms
    def test_linalg_eigh(self):
        A = torch.randn(2, 2, dtype=torch.float64)
        mesh = self.build_device_mesh()
        dtensor_A = distribute_tensor(A, device_mesh=mesh, placements=[Replicate()])
        dtensor_A = dtensor_A + dtensor_A.mT
        dtensor_L, dtensor_Q = torch.linalg.eigh(dtensor_A)

        # TODO: we need to convert A, L, Q to local because we don't have a
        # sharding strategy registered for aten.dist.default yet.
        local_A, local_L, local_Q = (
            dtensor_A.to_local(),
            dtensor_L.to_local(),
            dtensor_Q.to_local(),
        )
        distance = torch.dist(local_Q @ torch.diag(local_L) @ local_Q.mT, local_A)
        self.assertEqual(distance.item(), 0.0)

    @with_comms
    def test_upsampling(self):
        input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
        mesh = self.build_device_mesh()
        input_dtensor = distribute_tensor(
            input, device_mesh=mesh, placements=[Shard(0)]
        )

        upsample_m = [
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            torch.nn.Upsample(scale_factor=2, mode="bicubic"),
        ]
        for m in upsample_m:
            result = m(input)
            dtensor_result = m(input_dtensor)
            self.assertEqual(result, dtensor_result.full_tensor())

    @with_comms
    def test_cumsum(self):
        mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()
        inp = torch.rand(3, 5, device=self.device_type)

        shard_dim = 0
        input_dtensor = distribute_tensor(
            inp, device_mesh=mesh, placements=[Shard(shard_dim)]
        )

        cumsum_dims = [0, 1]
        for dim in cumsum_dims:
            output = torch.cumsum(inp, dim=dim)
            with comm_mode:
                output_dtensor = torch.cumsum(input_dtensor, dim=dim)
                if dim == shard_dim:
                    self.assertEqual(comm_mode.get_total_counts(), 1)
                    self.assertEqual(
                        comm_mode.get_comm_counts()[funcol.all_gather_into_tensor],
                        1,
                    )
                    self.assertTrue(output_dtensor.placements[0].is_replicate())
                else:
                    self.assertEqual(comm_mode.get_total_counts(), 0)
                    self.assertTrue(output_dtensor.placements[0].is_shard(shard_dim))
                self.assertEqual(output_dtensor.full_tensor(), output)

    @with_comms
    def test_conj_complex_dtensor(self):
        mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()

        freqs_cis = torch.randn(
            1, 1, dtype=torch.complex64, requires_grad=False, device=self.device_type
        )
        freqs_cis_dt = distribute_tensor(
            freqs_cis, device_mesh=mesh, placements=[Replicate()]
        )

        local_result = freqs_cis.conj() + 1
        with comm_mode:
            dtensor_result = freqs_cis_dt.conj() + 1
            self.assertEqual(comm_mode.get_total_counts(), 0)

        self.assertEqual(local_result, dtensor_result.full_tensor())

    @with_comms
    def test_rotary_embedding_complex_ops(self):
        mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()

        def apply_rotary_emb(xq, freqs_cis):
            xq_ = torch.view_as_complex(xq)
            xq_out = torch.view_as_real(xq_ * freqs_cis)
            return xq_out

        xq = torch.randn(1, 1, 2, requires_grad=True, device=self.device_type)
        freqs_cis = torch.randn(
            1, 1, dtype=torch.complex64, requires_grad=False, device=self.device_type
        )

        xq_dt = distribute_tensor(xq, device_mesh=mesh, placements=[Replicate()])
        freqs_cis_dt = distribute_tensor(
            freqs_cis, device_mesh=mesh, placements=[Replicate()]
        )

        with comm_mode:
            xq_out_dt = apply_rotary_emb(xq_dt, freqs_cis_dt)
            xq_out_dt.sum().backward()
            self.assertEqual(comm_mode.get_total_counts(), 0)

        dtensor_grad = xq_dt.grad.full_tensor()

        xq.grad = None
        xq_out = apply_rotary_emb(xq, freqs_cis)
        xq_out.sum().backward()

        self.assertEqual(dtensor_grad, xq.grad)

    @with_comms
    def test_histc(self):
        # TODO - nicer to use parametrize here so its easy to run one sub-test by name,
        # but its too slow (10sec per process-group init) -> switch to MultiProcessContinuousTest
        device_mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()
        tensor = torch.randn(12, 8, 8, requires_grad=True)
        for min_max_specified in (True, False):
            for placement in [Shard(0), Shard(1), Shard(2), Replicate()]:
                min_ = tensor.min().item()
                max_ = tensor.max().item()
                global_bins = (
                    tensor.histc(min=min_, max=max_)
                    if min_max_specified
                    else tensor.histc()
                )

                dtensor = distribute_tensor(tensor, device_mesh, (placement,))
                with comm_mode:
                    out_dt = (
                        dtensor.histc(min=min_, max=max_)
                        if min_max_specified
                        else dtensor.histc()
                    )

                if placement.is_shard() and not min_max_specified:
                    self.assertEqual(comm_mode.get_total_counts(), 1)
                    self.assertEqual(
                        comm_mode.get_comm_counts()[funcol.all_gather_into_tensor], 1
                    )
                else:
                    self.assertEqual(comm_mode.get_total_counts(), 0)

                out_full = out_dt.full_tensor()
                self.assertEqual(global_bins, out_full)

    @with_comms
    def test_logsumexp(self):
        mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()
        inp = torch.rand(3, 5, device=self.device_type)

        shard_dim = 0
        input_dtensor = distribute_tensor(
            inp, device_mesh=mesh, placements=[Shard(shard_dim)]
        )

        logsumexp_dims = [0, 1]
        for dim in logsumexp_dims:
            output = torch.logsumexp(inp, dim=dim)
            with comm_mode:
                output_dtensor = torch.logsumexp(input_dtensor, dim=dim)
                if dim == shard_dim:
                    self.assertEqual(comm_mode.get_total_counts(), 1)
                    self.assertEqual(
                        comm_mode.get_comm_counts()[funcol.all_gather_into_tensor],
                        1,
                    )
                    self.assertTrue(output_dtensor.placements[0].is_replicate())
                else:
                    self.assertEqual(comm_mode.get_total_counts(), 0)
                    self.assertTrue(output_dtensor.placements[0].is_shard(shard_dim))
                self.assertEqual(output_dtensor.full_tensor(), output)

    @with_comms
    def test_partial_reduction_ops(self):
        mesh = self.build_device_mesh()
        rank = dist.get_rank()

        torch.manual_seed(rank)
        local_tensor = torch.rand(3, dtype=torch.float32, device=self.device_type)
        dt = DTensor.from_local(
            local_tensor, device_mesh=mesh, placements=[Partial("sum")]
        )
        out_without_redistribute = torch.norm(dt)

        dt = dt.redistribute(dt.device_mesh, placements=[Replicate()])
        out_with_redistribute = torch.norm(dt)

        self.assertEqual(out_without_redistribute, out_with_redistribute)

        local_tensor = torch.rand(3, dtype=torch.float32, device=self.device_type)
        dt = DTensor.from_local(
            local_tensor, device_mesh=mesh, placements=[Partial("sum")]
        )
        out_without_redistribute = torch.max(dt)

        dt = dt.redistribute(dt.device_mesh, placements=[Replicate()])
        out_with_redistribute = torch.max(dt)

        self.assertEqual(out_without_redistribute, out_with_redistribute)

        local_tensor = torch.rand(3, dtype=torch.float32, device=self.device_type)
        dt = DTensor.from_local(
            local_tensor, device_mesh=mesh, placements=[Partial("sum")]
        )
        out_without_redistribute = torch.min(dt)

        dt = dt.redistribute(dt.device_mesh, placements=[Replicate()])
        out_with_redistribute = torch.min(dt)

        self.assertEqual(out_without_redistribute, out_with_redistribute)

    @with_comms
    def test_matching_partial_reduction_ops(self):
        mesh = self.build_device_mesh()
        rank = dist.get_rank()

        torch.manual_seed(rank)
        local_tensor = torch.rand(3, dtype=torch.float32, device=self.device_type)
        dt = DTensor.from_local(
            local_tensor, device_mesh=mesh, placements=[Partial("max")]
        )
        out_without_redistribute = torch.max(dt)

        dt = dt.redistribute(dt.device_mesh, placements=[Replicate()])
        out_with_redistribute = torch.max(dt)

        self.assertTrue(out_without_redistribute.placements[0].is_partial())
        self.assertTrue(out_with_redistribute.placements[0].is_replicate())
        self.assertEqual(out_without_redistribute, out_with_redistribute)

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_std(self):
        mesh = DeviceMesh(self.device_type, torch.arange(4).reshape(2, 2))
        rank = self.rank
        comm_mode = CommDebugMode()

        global_tensor = map_local_for_rank(
            rank,
            lambda rank: torch.tensor(
                [[-20.0, -18.0, -12.0, 0.0], [-20.0, -18.0, -8.0, 4.0]]
            ),
        )

        dt = distribute_tensor(global_tensor, mesh, [Shard(0), Shard(1)])

        with comm_mode:
            res = dt.std(dim=1)
        expected_answer = torch.tensor([9.0, 11.0])

        self.assertEqual(comm_mode.get_total_counts(), 1)
        self.assertEqual(comm_mode.get_comm_counts()[funcol.all_gather_into_tensor], 1)
        self.assertEqual(res.placements, [Shard(0), Replicate()])
        self.assertEqual(res.full_tensor(), expected_answer)

    @with_comms
    def test_prims_pointwise_ops(self):
        device_mesh = self.build_device_mesh()
        x = torch.randn(12, 8)
        y = torch.randn(12, 8)
        dtensor_x = distribute_tensor(x, device_mesh, [Shard(0)])
        dtensor_y = distribute_tensor(y, device_mesh, [Shard(0)])

        for op in [
            torch.ops.prims.bessel_i0e,
            torch.ops.prims.bessel_i1,
            torch.ops.prims.bessel_i1e,
            torch.ops.prims.bessel_j0,
            torch.ops.prims.bessel_j1,
            torch.ops.prims.erfcx,
            torch.ops.prims.ndtri,
            torch.ops.prims.spherical_bessel_j0,
            torch.special.erfcx,
        ]:
            local_result = op(x.abs().clamp(0.1, 3.0))
            dtensor_result = op(dtensor_x.abs().clamp(0.1, 3.0))
            self.assertEqual(dtensor_result.full_tensor(), local_result)
            self.assertTrue(dtensor_result.placements[0].is_shard(dim=0))

        for op in [
            torch.ops.prims.div,
            torch.ops.prims.gcd,
            torch.ops.prims.ne,
            torch.ops.aten.ne,
            torch.ops.aten.gcd,
        ]:
            if op in [torch.ops.prims.gcd, torch.ops.aten.gcd]:
                local_result = op(x.int(), y.int())
                dtensor_result = op(dtensor_x.int(), dtensor_y.int())
            else:
                local_result = op(x, y)
                dtensor_result = op(dtensor_x, dtensor_y)
            self.assertEqual(dtensor_result.full_tensor(), local_result)

    @with_comms
    def test_prims_view_of(self):
        device_mesh = self.build_device_mesh()
        x = torch.randn(12, 8)
        dtensor = distribute_tensor(x, device_mesh, [Shard(0)])

        result = torch.ops.prims.view_of(dtensor)
        self.assertTrue(result.placements[0].is_shard(dim=0))
        self.assertEqual(result.full_tensor(), x)


DistMathOpsTestWithLocalTensor = create_local_tensor_test_class(
    DistMathOpsTest,
)

if __name__ == "__main__":
    run_tests()
