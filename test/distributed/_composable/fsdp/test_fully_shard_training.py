"""
TORCH_COMPILE_DEBUG=1 CUDA_VISIBLE_DEVICES=4,5 pytest -rx test/distributed/_composable/fsdp/test_fully_shard_training.py::TestFullyShard1DTrainingCore::test_train_parity_single_group >test_output1.txt 2>&1

TORCH_COMPILE_DEBUG=1 CUDA_VISIBLE_DEVICES=4,5 pytest -rx test/distributed/_composable/fsdp/test_fully_shard_training.py::TestFullyShard1DTrainingCore::test_train_parity_multi_group_full_graph_compile >test_output1.txt 2>&1

TORCH_COMPILE_DEBUG=1 CUDA_VISIBLE_DEVICES=4,5 pytest -rx test/distributed/_composable/fsdp/test_fully_shard_training.py::TestFullyShard1DTrainingCore::test_multi_forward_module >test_output1.txt 2>&1

TORCH_COMPILE_DEBUG=1 CUDA_VISIBLE_DEVICES=4,5 pytest -rx test/distributed/_composable/fsdp/test_fully_shard_training.py::TestFullyShard2DTraining::test_train_parity_2d_mlp >test_output1.txt 2>&1
"""

# Owner(s): ["oncall: distributed"]

import contextlib
import copy
import functools
import unittest
import logging
from typing import Dict, Iterable, List, Tuple, Union, Type

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed._composable import checkpoint, replicate
from torch.distributed._composable.fsdp import FSDP, fully_shard
from torch.distributed._tensor import DTensor, init_device_mesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_PREFIX,
    apply_activation_checkpointing,
    CheckpointWrapper,
)
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    FSDPTest,
    FSDPTestMultiThread,
    MLP,
    patch_all_gather,
    patch_reduce_scatter,
    test_graph_break_fsdp,
)
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,
    run_tests,
    wrapSwapTensorsTest,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)
from torch._dynamo import compiled_autograd
from torch.testing._internal.common_distributed import _dynamo_dist_per_rank_init
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir
from torch.testing._internal.logging_utils import logs_to_string
from torch._functorch._aot_autograd.fsdp_fx_passes import must_not_appear_ops_after_fsdp_fx_passes


torch_log = logging.getLogger("torch")

def prepare_capture_post_grad_graph_from_log():
    log_stream, ctx = logs_to_string(
        "torch._inductor.compile_fx", "post_grad_graphs"
    )
    return log_stream, ctx

def extract_graph_str(log_stream):
    lines = log_stream.getvalue().strip().split("\n")[3:]
    return "\n".join(lines).strip()

def remove_comments_from_graph_str(graph_str):
    lines = graph_str.split("\n")
    return "\n".join([line for line in lines if not line.strip().startswith("#")]).strip()

# TODO(yf225): reenable this once we settle on how the graph should look like
def check_expected_ops_in_graphs(unittest, log_stream_for_fwd_graph, log_stream_for_bwd_graph, compile_expected_ops, compile_expected_ops_count):
    # post_grad_fwd_graph_str = extract_graph_str(log_stream_for_fwd_graph)
    # post_grad_bwd_graph_str = extract_graph_str(log_stream_for_bwd_graph)
    # post_grad_fwd_graph_str_no_comment = remove_comments_from_graph_str(post_grad_fwd_graph_str)
    # post_grad_bwd_graph_str_no_comment = remove_comments_from_graph_str(post_grad_bwd_graph_str)
    # for op_str in must_not_appear_ops_after_fsdp_fx_passes:
    #     unittest.assertEqual(post_grad_fwd_graph_str_no_comment.count(op_str), 0, msg=f"'{op_str}' should not appear in graph. Graph: {post_grad_fwd_graph_str}")
    #     unittest.assertEqual(post_grad_bwd_graph_str_no_comment.count(op_str), 0, msg=f"'{op_str}' should not appear in graph. Graph: {post_grad_bwd_graph_str}")
    # for op_str, expected_count_fwd, expected_count_bwd in zip(compile_expected_ops, compile_expected_ops_count[0], compile_expected_ops_count[1]):
    #     count_fwd = post_grad_fwd_graph_str_no_comment.count(op_str)
    #     count_bwd = post_grad_bwd_graph_str_no_comment.count(op_str)
    #     unittest.assertEqual(count_fwd, expected_count_fwd, msg=f"'{op_str}' should appear {expected_count_fwd} times in graph, but it appears {count_fwd} times. Graph: {post_grad_fwd_graph_str}")
    #     unittest.assertEqual(count_bwd, expected_count_bwd, msg=f"'{op_str}' should appear {expected_count_bwd} times in graph, but it appears {count_bwd} times. Graph: {post_grad_bwd_graph_str}")
    pass

class TestFullyShardForwardInputs(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    @test_graph_break_fsdp()
    def test_root_move_forward_input_to_device(self):
        device = torch.device("cuda", 0)

        class ParamlessModule(nn.Module):
            def forward(self, x: torch.Tensor, ys: Tuple[torch.Tensor, ...]):
                # Check that FSDP moved the inputs to GPU, including recursing
                # into the tuple data structure
                assert x.device == device, f"Expects {device} but got {x.device}"
                assert (
                    ys[0].device == device
                ), f"Expects {device} but got {ys[0].device}"
                assert (
                    ys[1].device == device
                ), f"Expects {device} but got {ys[1].device}"
                y = ys[0] + ys[1]
                return x + y + 1

        model = ParamlessModule()
        fully_shard(model)
        x = torch.randn((3,))
        ys = (torch.randn((3,)), torch.randn((3,)))
        self.assertEqual(x.device, torch.device("cpu"))
        self.assertEqual(ys[0].device, torch.device("cpu"))
        self.assertEqual(ys[1].device, torch.device("cpu"))
        model(x, ys)


class TestFullyShardRegisteredParams(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 4

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_param_registration_after_forward(self):
        """Tests the parameter registration after forward."""
        device = torch.device("cuda", 0)
        # Single FSDP group
        for reshard_after_forward in (True, False, 2):
            torch.manual_seed(42)
            model = MLP(3, device)
            # Since seed is per process, not per thread, we broadcast to ensure
            # the same parameters across ranks
            for param in model.parameters():
                dist.broadcast(param, src=0)
            ref_model = copy.deepcopy(model)
            fully_shard(model, reshard_after_forward=reshard_after_forward)  # root only
            inp = torch.randn((2, 3), device="cuda")
            self._assert_dtensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())
            model(inp)  # root does not reshard after forward
            self._assert_tensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())
            model.reshard()  # however, we can manually reshard
            self._assert_dtensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())

        # Multiple FSDP groups
        for reshard_after_forward in (True, False, 2):
            torch.manual_seed(42)
            model = nn.Sequential(MLP(3, device), MLP(3, device))
            for param in model.parameters():
                dist.broadcast(param, src=0)
            ref_model = copy.deepcopy(model)
            fully_shard(model[0].in_proj, reshard_after_forward=reshard_after_forward)
            fully_shard(model[0].out_proj, reshard_after_forward=reshard_after_forward)
            fully_shard(model, reshard_after_forward=reshard_after_forward)

            self._assert_dtensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())
            model(inp)
            non_root_params = list(model[0].in_proj.parameters()) + list(
                model[0].out_proj.parameters()
            )
            root_params = list(set(model.parameters()) - set(non_root_params))
            if reshard_after_forward is False:
                self._assert_tensor_params(non_root_params)
            else:
                self._assert_dtensor_params(non_root_params)
            self._assert_tensor_params(root_params)
            self._assert_same_params(model.parameters(), ref_model.parameters())
            for module in model.modules():
                if isinstance(module, FSDP):
                    module.reshard()  # however, we can manually reshard
            self._assert_dtensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_param_registration_after_backward(self):
        """Tests the parameter registration after backward."""
        device = torch.device("cuda", 0)
        # Single FSDP group
        for reshard_after_forward in (True, False, 2):
            model = MLP(8, device)
            fully_shard(model, reshard_after_forward=reshard_after_forward)  # root only
            inp = torch.randn((2, 8), device="cuda")
            self._assert_dtensor_params(model.parameters())
            model(inp).sum().backward()
            self._assert_dtensor_params(model.parameters())

        # Multiple FSDP groups
        for reshard_after_forward in (True, False, 2):
            model = MLP(8, device)
            fully_shard(model.in_proj, reshard_after_forward=reshard_after_forward)
            fully_shard(model.out_proj, reshard_after_forward=reshard_after_forward)
            fully_shard(model, reshard_after_forward=reshard_after_forward)
            self._assert_dtensor_params(model.parameters())
            model(inp).sum().backward()
            self._assert_dtensor_params(model.parameters())

    def _assert_tensor_params(self, params: Iterable[nn.Parameter]):
        self.assertGreater(len(list(params)), 0)
        for param in params:
            self.assertNotIsInstance(param, DTensor)
            self.assertIsInstance(param, torch.Tensor)

    def _assert_dtensor_params(self, params: Iterable[nn.Parameter]):
        self.assertGreater(len(list(params)), 0)
        for param in params:
            self.assertIsInstance(param, DTensor)

    def _assert_same_params(
        self, params: Iterable[nn.Parameter], ref_params: Iterable[nn.Parameter]
    ):
        params, ref_params = list(params), list(ref_params)
        self.assertEqual(len(params), len(ref_params))
        for param, ref_param in zip(params, ref_params):
            if isinstance(param, DTensor):
                param = param.full_tensor()
            self.assertEqual(param.shape, ref_param.shape)
            self.assertEqual(param, ref_param)


class TestFullyShardCastAfterInit(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    @wrapSwapTensorsTest(True)
    def test_to_float64_after_init(self):
        """Tests that the user can cast the module to float64 after init."""
        # NOTE: Test fp64 instead of a lower precision dtype like bf16 for
        # better numerics. The important part is changing the dtype.
        torch.manual_seed(42)
        mlp_dim, device, dtype = 4, torch.device("cuda"), torch.float64
        model = MLP(mlp_dim, device=device)
        for param in model.parameters():
            dist.broadcast(param, src=0)
        ref_model = copy.deepcopy(model).to(dtype)
        replicate(ref_model)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for module in (model.in_proj, model.out_proj, model):
            fully_shard(module)
        model.to(dtype)
        for param in model.parameters():
            self.assertEqual(param.dtype, dtype)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        check_sharded_parity(self, ref_model, model)
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((2, mlp_dim), device="cuda", dtype=dtype)
        for iter_idx in range(10):
            losses: List[torch.Tensor] = []
            for _model in (ref_model, model):
                losses.append(_model(inp).sum())
                losses[-1].backward()
            self.assertEqual(losses[0], losses[1])
            check_sharded_parity(self, ref_model, model)
            for _optim in (ref_optim, optim):
                _optim.step()
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))


class TestFullyShard1DTrainingCore(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(8, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    @test_graph_break_fsdp()
    def test_train_parity_single_group(self):
        """Tests train parity with DDP for a single FSDP group."""
        self.run_subtests(
            {
                "lin_shapes_dict": [{0: [(16, 15), (15, 8)]}, {1: [(7, 15), (15, 3)]}],
                "compile_expected_ops": [[
                    "torch.ops.aten.empty.",
                    "torch.ops.aten.copy_.",
                    "torch.ops._c10d_functional.all_gather_into_tensor.",
                    "torch.ops._c10d_functional.reduce_scatter_tensor.",
                ]],
                "compile_expected_ops_count_dict": [{
                    0: ([1, 4, 1, 0], [3, 6, 1, 1]),
                    1: ([1, 4, 1, 0], [5, 8, 1, 1]),
                }],
            },
            self._test_train_parity_single_group,
        )

    def _test_train_parity_single_group(self, lin_shapes_dict: Dict[int, List[Tuple[int, int]]], compile_expected_ops: List[str]=None, compile_expected_ops_count_dict: Dict[int, Tuple[List[int], List[int]]]=None):
        full_graph_compile = True
        test_case_id = list(lin_shapes_dict.keys())[0]
        lin_shapes = list(lin_shapes_dict.values())[0]
        compile_expected_ops_count = compile_expected_ops_count_dict[test_case_id]
        with _dynamo_dist_per_rank_init(self.rank, self.world_size, init_pg=False, enabled=full_graph_compile):
            torch.manual_seed(42)
            model = nn.Sequential(
                nn.Linear(*lin_shapes[0]), nn.ReLU(), nn.Linear(*lin_shapes[1])
            )
            ref_model = copy.deepcopy(model).cuda()
            replicate(ref_model, device_ids=[self.rank])
            ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)

            model_for_eager = copy.deepcopy(model).cuda()
            fully_shard(model_for_eager)
            optim_for_eager = torch.optim.Adam(model_for_eager.parameters(), lr=1e-2)

            torch.manual_seed(42 + self.rank + 1)
            inp = (torch.randn((4, lin_shapes[0][0]), device="cuda"),)

            if full_graph_compile:
                # NOTE(yf225): we can't use the `post_grad_custom_post_pass` hook for checking graph, because the hook runs before the FSDP FX passes.
                log_stream_for_fwd_graph, ctx_for_fwd_graph = prepare_capture_post_grad_graph_from_log()
                log_stream_for_bwd_graph, ctx_for_bwd_graph = prepare_capture_post_grad_graph_from_log()
                def compiler_fn(gm):
                    return torch.compile(gm, backend="inductor", fullgraph=True)
                torch._dynamo.config.trace_distributed = True
                torch._functorch.config.move_view_chain_to_bwd_graph = True
                torch._inductor.config.triton.unique_kernel_names = True
                model_to_be_compiled = copy.deepcopy(model).cuda()
                fully_shard(model_to_be_compiled, reshard_after_forward=True, _reshard_after_forward_root=True)
                optim_for_compile = torch.optim.Adam(model_to_be_compiled.parameters(), lr=1e-2)

            compiled_model = None

            def get_compiled_model_and_optim(iter_idx):
                nonlocal compiled_model
                if not full_graph_compile:
                    return None, None, False
                if iter_idx > 0:
                    if compiled_model is None:
                        compiled_model = torch.compile(model_to_be_compiled, backend="inductor", fullgraph=True)
                    return compiled_model, optim_for_compile, True
                else:
                    return model_to_be_compiled, optim_for_compile, False

            for iter_idx in range(10):
                losses: List[float] = []
                for _model, _optim, _is_compile in ((ref_model, ref_optim, False), (model_for_eager, optim_for_eager, False), get_compiled_model_and_optim(iter_idx)):
                    if _model is None:
                        continue
                    # TODO(yf225): under compile, if we set `set_to_none=False`, compile numerical result is not correct in some cases. We need to fix it.
                    _optim.zero_grad(set_to_none=(iter_idx % 2 == 0) if not _is_compile else True)
                    if _is_compile:
                        ctx = compiled_autograd.enable(compiler_fn)
                    else:
                        ctx = contextlib.nullcontext()
                    with ctx:
                        # Assume compilation process to happen only on iter_idx=1
                        with (ctx_for_fwd_graph() if _is_compile and iter_idx == 1 else contextlib.nullcontext()):
                            loss = _model(*inp).sum()
                        losses.append(loss.item())
                        with (ctx_for_bwd_graph() if _is_compile and iter_idx == 1 else contextlib.nullcontext()):
                            loss.backward()
                    _optim.step()
                losses_tensors = [torch.tensor(x) for x in losses]
                self.assertTrue(all(torch.allclose(x, losses_tensors[0]) for x in losses_tensors), msg=f"iter_idx: {iter_idx}, losses_tensors: {losses_tensors}, losses: {losses}")

            if full_graph_compile:
                check_expected_ops_in_graphs(self, log_stream_for_fwd_graph, log_stream_for_bwd_graph, compile_expected_ops, compile_expected_ops_count)

    @skip_if_lt_x_gpu(2)
    def test_train_parity_multi_group_eager(self):
        """
        Tests train parity against DDP when using multiple parameter groups for
        communication (for communication and computation overlap plus memory
        reduction).
        """
        self.run_subtests(
            {
                "reshard_after_forward": [True, False, 2],
                "device_type": ["cuda"],
                "delay_after_forward": [False, True],
                "delay_before_all_gather": [False, True],
                "delay_before_reduce_scatter": [False, True],
                "delay_before_optim": [False, True],
            },
            self._test_train_parity_multi_group,
        )

    @skip_if_lt_x_gpu(2)
    @test_graph_break_fsdp()
    def test_train_parity_multi_group_graph_break_compile(self):
        self.run_subtests(
            {
                "reshard_after_forward": [True, False],
                "device_type": ["cuda"],
                "delay_after_forward": [False, True],
                "delay_before_all_gather": [False],
                "delay_before_reduce_scatter": [False],
                "delay_before_optim": [False, True],
            },
            self._test_train_parity_multi_group,
        )

    @skip_if_lt_x_gpu(2)
    def test_train_parity_multi_group_full_graph_compile(self):
        self.run_subtests(
            {
                "reshard_after_forward": [True],
                "device_type": ["cuda"],
                "delay_after_forward": [False, True],
                "delay_before_all_gather": [False],
                "delay_before_reduce_scatter": [False],
                "delay_before_optim": [False, True],
                "full_graph_compile": [True],
                "compile_expected_ops": [[
                    "torch.ops.aten.empty.",
                    "torch.ops.aten.copy_.",
                    "torch.ops._c10d_functional.all_gather_into_tensor.",
                    "torch.ops._c10d_functional.reduce_scatter_tensor.",
                ]],
                "compile_expected_ops_count": [
                    ([1, 12, 3, 0], [3, 12, 3, 3]),
                ],
            },
            self._test_train_parity_multi_group,
        )

    def _test_train_parity_multi_group(
        self,
        reshard_after_forward: Union[bool, int],
        device_type: str,
        delay_after_forward: bool,
        delay_before_all_gather: bool,
        delay_before_reduce_scatter: bool,
        delay_before_optim: bool,
        full_graph_compile: bool=False,
        compile_expected_ops: List[str]=None,
        compile_expected_ops_count: Tuple[List[int], List[int]]=None,
    ):
        # Only test individual delays or all four delays to save test time
        if (
            delay_after_forward
            + delay_before_all_gather
            + delay_before_reduce_scatter
            + delay_before_optim
            in (2, 3)
        ):
            return
        with _dynamo_dist_per_rank_init(self.rank, self.world_size, init_pg=False, enabled=full_graph_compile):
            assert device_type in ("cuda", "cpu"), f"{device_type}"
            torch.manual_seed(42)
            lin_dim = 32
            model = nn.Sequential(*[MLP(lin_dim, torch.device("cpu")) for _ in range(3)])
            ref_model = copy.deepcopy(model)
            if device_type == "cuda":
                replicate(ref_model.cuda(), device_ids=[self.rank])
            else:
                gloo_pg = dist.new_group(backend="gloo")
                replicate(ref_model, process_group=gloo_pg)
            ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
            mesh = init_device_mesh(device_type, (self.world_size,))
            model_for_eager = copy.deepcopy(model)
            for mlp in model_for_eager:
                fully_shard(mlp, mesh=mesh, reshard_after_forward=reshard_after_forward, _reshard_after_forward_root=reshard_after_forward if full_graph_compile else False)
            fully_shard(model_for_eager, mesh=mesh, reshard_after_forward=reshard_after_forward, _reshard_after_forward_root=reshard_after_forward if full_graph_compile else False)
            optim_for_eager = torch.optim.Adam(model_for_eager.parameters(), lr=1e-2)

            if full_graph_compile:
                # NOTE(yf225): we can't use the `post_grad_custom_post_pass` hook for checking graph, because the hook runs before the FSDP FX passes.
                log_stream_for_fwd_graph, ctx_for_fwd_graph = prepare_capture_post_grad_graph_from_log()
                log_stream_for_bwd_graph, ctx_for_bwd_graph = prepare_capture_post_grad_graph_from_log()
                def compiler_fn(gm):
                    return torch.compile(gm, backend="inductor", fullgraph=True)
                torch._dynamo.config.trace_distributed = True
                torch._functorch.config.move_view_chain_to_bwd_graph = True
                torch._inductor.config.triton.unique_kernel_names = True
                model_to_be_compiled = copy.deepcopy(model).cuda()
                for mlp in model_to_be_compiled:
                    fully_shard(mlp, mesh=mesh, reshard_after_forward=True, _reshard_after_forward_root=True)
                fully_shard(model_to_be_compiled, mesh=mesh, reshard_after_forward=True, _reshard_after_forward_root=True)
                optim_for_compile = torch.optim.Adam(model_to_be_compiled.parameters(), lr=1e-2)

            compiled_model = None

            def get_compiled_model_and_optim(iter_idx):
                nonlocal compiled_model
                if not full_graph_compile:
                    return None, None, False
                if iter_idx > 0:
                    if compiled_model is None:
                        compiled_model = torch.compile(model_to_be_compiled, backend="inductor", fullgraph=True)
                    return compiled_model, optim_for_compile, True
                else:
                    return model_to_be_compiled, optim_for_compile, False

            delay_in_ms = 100
            orig_all_gather = dist.all_gather_into_tensor
            orig_reduce_scatter = dist.reduce_scatter_tensor

            def delayed_all_gather(*args, **kwargs):
                torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
                return orig_all_gather(*args, **kwargs)

            def delayed_reduce_scatter(*args, **kwargs):
                torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
                return orig_reduce_scatter(*args, **kwargs)

            torch.manual_seed(42 + self.rank + 1)
            patch_all_gather_ctx = (
                patch_all_gather(delayed_all_gather)
                if delay_before_all_gather
                else contextlib.nullcontext()
            )
            patch_reduce_scatter_ctx = (
                patch_reduce_scatter(delayed_reduce_scatter)
                if delay_before_reduce_scatter
                else contextlib.nullcontext()
            )
            with patch_all_gather_ctx, patch_reduce_scatter_ctx:
                for iter_idx in range(10):
                    inp = torch.randn((8, lin_dim), device=torch.device(device_type))
                    losses: List[float] = []
                    for _model, _optim, _is_compile in ((ref_model, ref_optim, False), (model_for_eager, optim_for_eager, False), get_compiled_model_and_optim(iter_idx)):
                        if _model is None:
                            continue
                        # TODO(yf225): under compile, if we set `set_to_none=False`, compile numerical result is not correct in some cases. We need to fix it.
                        _optim.zero_grad(set_to_none=(iter_idx % 2 == 0) if not _is_compile else True)
                        if _is_compile:
                            ctx = compiled_autograd.enable(compiler_fn)
                        else:
                            ctx = contextlib.nullcontext()
                        with ctx:
                            # Assume compilation process to happen only on iter_idx=1
                            with (ctx_for_fwd_graph() if _is_compile and iter_idx == 1 else contextlib.nullcontext()):
                                loss = _model(inp).sum()
                            losses.append(loss.item())
                            if _model is model_for_eager and delay_after_forward:
                                torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
                            with (ctx_for_bwd_graph() if _is_compile and iter_idx == 1 else contextlib.nullcontext()):
                                loss.backward()
                        if _model is model_for_eager and delay_before_optim:
                            torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
                        _optim.step()
                    losses_tensors = [torch.tensor(x) for x in losses]
                    self.assertTrue(all(torch.allclose(x, losses_tensors[0]) for x in losses_tensors), msg=f"iter_idx: {iter_idx}, losses_tensors: {losses_tensors}, losses: {losses}")

                check_expected_ops_in_graphs(self, log_stream_for_fwd_graph, log_stream_for_bwd_graph, compile_expected_ops, compile_expected_ops_count)

    @skip_if_lt_x_gpu(2)
    @test_graph_break_fsdp()
    def test_non_root_forward_backward(self):
        """
        Tests running forward/backward through the root and then through a
        non-root. The non-root needs to synchronize streams/queue the callback.
        """
        torch.manual_seed(42)
        lin_dim = 32
        model = nn.Sequential(*[MLP(lin_dim, torch.device("cpu")) for _ in range(3)])
        ref_model = copy.deepcopy(model).cuda()
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for mlp in model:
            fully_shard(mlp)
        fully_shard(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        torch.manual_seed(42 + self.rank)
        inp = torch.randn((8, lin_dim), device=torch.device("cuda"))

        ref_root_loss = ref_model(inp).sum()
        ref_root_loss.backward()
        for param in ref_model.parameters():
            dist.all_reduce(param.grad)
            param.grad.detach().div_(self.world_size)
        ref_optim.step()
        ref_optim.zero_grad()
        ref_nonroot_loss = ref_model[0](inp).sum()
        ref_nonroot_loss.backward()
        for param in ref_model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
                param.grad.detach().div_(self.world_size)
        ref_optim.step()

        root_loss = model(inp).sum()
        root_loss.backward()
        torch.cuda._sleep(int(100 * get_cycles_per_ms()))
        optim.step()
        optim.zero_grad()
        nonroot_loss = model[0](inp).sum()
        nonroot_loss.backward()
        optim.step()

        self.assertEqual(ref_root_loss, root_loss)
        self.assertEqual(ref_nonroot_loss, nonroot_loss)
        self.assertEqual(ref_model(inp).sum(), model(inp).sum())

    @skip_if_lt_x_gpu(2)
    @test_graph_break_fsdp()
    def test_multi_forward_module(self):
        """
        Tests parity with DDP when running a module that participates multiple
        times in forward.
        """
        self.run_subtests(
            {
                "reshard_after_forward": [True, False, 2],
                "compile_expected_ops": [[
                    "torch.ops.aten.empty.",
                    "torch.ops.aten.copy_.",
                    "torch.ops._c10d_functional.all_gather_into_tensor.",
                    "torch.ops._c10d_functional.reduce_scatter_tensor.",
                ]],
                "compile_expected_ops_count": [
                    ([2, 4, 2, 0], [3, 4, 1, 2]),
                ],
            },
            self._test_multi_forward_module,
        )

    def _test_multi_forward_module(self, reshard_after_forward: Union[bool, int], compile_expected_ops: List[str]=None, compile_expected_ops_count: Tuple[List[int], List[int]]=None):
        full_graph_compile = False
        if reshard_after_forward is True:
            full_graph_compile = True
        class MultiForwardModule(nn.Module):
            def __init__(self, device: torch.device):
                super().__init__()
                self.inner = nn.Linear(4, 4, device=device)
                self.outer = nn.Linear(4, 5, device=device)

            def forward(self, x):
                i = self.inner(x)
                j = self.inner(x)
                return self.outer(i + j)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size, init_pg=False, enabled=full_graph_compile):
            torch.manual_seed(42)
            model = MultiForwardModule(device="cuda")
            ref_model = copy.deepcopy(model)
            replicate(ref_model, device_ids=[self.rank])
            ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
            model_for_eager = copy.deepcopy(model)
            fully_shard(model_for_eager.inner)
            fully_shard(model_for_eager)
            optim_for_eager = torch.optim.Adam(model_for_eager.parameters(), lr=1e-2)

            if full_graph_compile:
                # NOTE(yf225): we can't use the `post_grad_custom_post_pass` hook for checking graph, because the hook runs before the FSDP FX passes.
                log_stream_for_fwd_graph, ctx_for_fwd_graph = prepare_capture_post_grad_graph_from_log()
                log_stream_for_bwd_graph, ctx_for_bwd_graph = prepare_capture_post_grad_graph_from_log()
                def compiler_fn(gm):
                    return torch.compile(gm, backend="inductor", fullgraph=True)
                torch._dynamo.config.trace_distributed = True
                torch._functorch.config.move_view_chain_to_bwd_graph = True
                torch._inductor.config.triton.unique_kernel_names = True
                model_to_be_compiled = copy.deepcopy(model).cuda()
                fully_shard(model_to_be_compiled.inner, reshard_after_forward=True, _reshard_after_forward_root=True)
                fully_shard(model_to_be_compiled, reshard_after_forward=True, _reshard_after_forward_root=True)
                optim_for_compile = torch.optim.Adam(model_to_be_compiled.parameters(), lr=1e-2)

            compiled_model = None

            def get_compiled_model_and_optim(iter_idx):
                nonlocal compiled_model
                if not full_graph_compile:
                    return None, None, False
                if iter_idx > 0:
                    if compiled_model is None:
                        compiled_model = torch.compile(model_to_be_compiled, backend="inductor", fullgraph=True)
                    return compiled_model, optim_for_compile, True
                else:
                    return model_to_be_compiled, optim_for_compile, False

            torch.manual_seed(42 + self.rank)
            inp = torch.randn((32, 4), device="cuda")
            for iter_idx in range(10):
                losses: List[torch.Tensor] = []
                for _model, _optim, _is_compile in ((ref_model, ref_optim, False), (model_for_eager, optim_for_eager, False), get_compiled_model_and_optim(iter_idx)):
                    if _model is None:
                        continue
                    # TODO(yf225): under compile, if we set `set_to_none=False`, compile numerical result is not correct in some cases. We need to fix it.
                    _optim.zero_grad(set_to_none=(iter_idx % 2 == 0) if not _is_compile else True)
                    if _is_compile:
                        ctx = compiled_autograd.enable(compiler_fn)
                    else:
                        ctx = contextlib.nullcontext()
                    with ctx:
                        # Assume compilation process to happen only on iter_idx=1
                        with (ctx_for_fwd_graph() if _is_compile and iter_idx == 1 else contextlib.nullcontext()):
                            loss = _model(inp).sum()
                        losses.append(loss.item())
                        with (ctx_for_bwd_graph() if _is_compile and iter_idx == 1 else contextlib.nullcontext()):
                            loss.backward()
                    _optim.step()
                losses_tensors = [torch.tensor(x) for x in losses]
                self.assertTrue(all(torch.allclose(x, losses_tensors[0]) for x in losses_tensors), msg=f"iter_idx: {iter_idx}, losses_tensors: {losses_tensors}, losses: {losses}")

            if full_graph_compile:
                check_expected_ops_in_graphs(self, log_stream_for_fwd_graph, log_stream_for_bwd_graph, compile_expected_ops, compile_expected_ops_count)


class TestFullyShard1DTrainingCompose(FSDPTest):
    @property
    def world_size(self) -> int:
        # Since these tests run with a larger transformer model, they may see
        # some numeric drift with >2 GPUs
        return min(torch.cuda.device_count(), 2)

    @skip_if_lt_x_gpu(2)
    def test_train_parity_with_activation_checkpointing(self):
        """
        Tests train parity against DDP when composing with activation
        checkpointing.
        """
        self.run_subtests(
            {
                "reshard_after_forward": [True, False, 2],
                "checkpoint_impl": ["composable", "utils", "wrapper"],
            },
            self._test_train_parity_with_activation_checkpointing,
        )

    def _test_train_parity_with_activation_checkpointing(
        self, reshard_after_forward: Union[bool, int], checkpoint_impl: str
    ):
        assert checkpoint_impl in ("composable", "utils", "wrapper")
        torch.manual_seed(42)
        vocab_size = 1024
        with torch.device(torch.device("cuda")):
            model_args = ModelArgs(
                n_layers=3,
                n_heads=4,
                vocab_size=vocab_size,
                max_seq_len=64,
                dropout_p=0.1,
                checkpoint_activations=(checkpoint_impl == "utils"),
            )
            model = Transformer(model_args)
        ref_model = replicate(copy.deepcopy(model), device_ids=[self.rank])
        foreach = True
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2, foreach=foreach)
        fully_shard_fn = functools.partial(
            fully_shard,
            reshard_after_forward=reshard_after_forward,
        )
        if checkpoint_impl == "wrapper":
            prefixes_to_ignore = (_CHECKPOINT_PREFIX,)
            apply_activation_checkpointing(
                model, check_fn=lambda m: isinstance(m, TransformerBlock)
            )
            for module in model.modules():
                # Apply to `CheckpointWrapper`, which wraps `TransformerBlock`
                if isinstance(module, CheckpointWrapper):
                    fully_shard_fn(module)
        else:
            prefixes_to_ignore = ()
            for module in model.modules():
                if isinstance(module, TransformerBlock):
                    if checkpoint_impl == "composable":
                        checkpoint(module)
                    fully_shard_fn(module)
        fully_shard_fn(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=foreach)

        torch.manual_seed(42 + self.rank)
        # Reuse the same input across iterations to avoid loss explosion from
        # trying to learn from random inputs
        inp = torch.randint(0, vocab_size, (3, 64), device="cuda")
        check_sharded_parity(
            self, ref_model, model, prefixes_to_ignore=prefixes_to_ignore
        )
        for iter_idx in range(10):
            losses: List[torch.Tensor] = []
            for _model in (ref_model, model):
                torch.manual_seed(iter_idx + 1)  # for dropout determinism
                losses.append(_model(inp).sum())
                losses[-1].backward()
            check_sharded_parity(
                self, ref_model, model, prefixes_to_ignore=prefixes_to_ignore
            )
            self.assertEqual(losses[0], losses[1])
            for _optim in (ref_optim, optim):
                _optim.step()
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            check_sharded_parity(
                self, ref_model, model, prefixes_to_ignore=prefixes_to_ignore
            )


class TestFullyShardSharedParams(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    @test_graph_break_fsdp(compile_compute_on_module=TransformerBlock)
    def test_train_parity_with_shared_params_no_ac(self):
        self.run_subtests(
            {
                "reshard_after_forward": [False, True],
                "use_activation_checkpointing": [False],
            },
            self._test_train_shared_params,
        )

    @skip_if_lt_x_gpu(2)
    def test_train_parity_with_shared_params_ac(self):
        self.run_subtests(
            {
                "reshard_after_forward": [False, True],
                "use_activation_checkpointing": [True],
            },
            self._test_train_shared_params,
        )

    def _test_train_shared_params(
        self,
        reshard_after_forward: bool,
        use_activation_checkpointing: bool,
    ):
        torch.manual_seed(42)
        model_args = ModelArgs(n_layers=3, dropout_p=0.0, weight_tying=True)
        model = Transformer(model_args)
        ref_model = copy.deepcopy(model).cuda()
        replicate(ref_model, device_ids=[self.rank])
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                if use_activation_checkpointing:
                    checkpoint(module)
                fully_shard(module, reshard_after_forward=reshard_after_forward)
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        torch.manual_seed(42 + self.rank + 1)
        for iter_idx in range(10):
            inp = torch.randint(0, model_args.vocab_size, (2, 16), device="cuda")
            losses: List[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()
            self.assertEqual(losses[0], losses[1])


class TestFullyShardGradientAccumulation(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(2, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_set_requires_gradient_sync(self):
        """
        Tests the ``set_requires_gradient_sync`` API to exercise gradient
        accumulation without gradient reduction. This test includes mixing with
        gradient accumulation *with* gradient reduction.
        """
        self.run_subtests(
            {
                "reshard_after_forward": [True, False, 2],
                # For `True`, disable reduce-scatter for all MLPs, and for
                # `False`, only disable it for some MLPs
                "recurse": [True, False],
            },
            self._test_set_requires_gradient_sync,
        )

    def _test_set_requires_gradient_sync(
        self,
        reshard_after_forward: Union[bool, int],
        recurse: bool,
    ):
        torch.manual_seed(42)
        local_batch_size, lin_dim, num_mlps, num_microbatches = (2, 32, 3, 3)
        global_batch_size = local_batch_size * self.world_size
        if not recurse:
            num_mlps_to_disable_reduce_scatter = 2
        model = nn.Sequential(
            *[MLP(lin_dim, torch.device("cpu")) for _ in range(num_mlps)]
        )
        ref_model = copy.deepcopy(model).cuda()
        fully_shard_fn = functools.partial(
            fully_shard, reshard_after_forward=reshard_after_forward
        )
        for mlp in model:
            fully_shard_fn(mlp)
        fully_shard_fn(model)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        orig_reduce_scatter = dist.reduce_scatter_tensor
        reduce_scatter_count = 0

        def reduce_scatter_with_count(*args, **kwargs):
            nonlocal reduce_scatter_count
            reduce_scatter_count += 1
            return orig_reduce_scatter(*args, **kwargs)

        torch.manual_seed(1)  # same on all ranks
        for iter_idx in range(5):
            with patch_reduce_scatter(reduce_scatter_with_count):
                for microbatch_idx in range(num_microbatches):
                    is_last_microbatch = microbatch_idx == num_microbatches - 1
                    if recurse:
                        model.set_requires_gradient_sync(is_last_microbatch)
                    else:
                        for mlp in model[:num_mlps_to_disable_reduce_scatter]:
                            mlp.set_requires_gradient_sync(
                                is_last_microbatch, recurse=False
                            )
                    global_inp = torch.rand((global_batch_size, lin_dim), device="cuda")
                    local_inp = global_inp[
                        self.rank
                        * local_batch_size : (self.rank + 1)
                        * local_batch_size
                    ].detach()
                    losses: List[torch.Tensor] = []
                    for _model, _optim, inp in (
                        (ref_model, ref_optim, global_inp),
                        (model, optim, local_inp),
                    ):
                        losses.append(_model(inp).sum())
                        losses[-1].backward()
                    dist.all_reduce(losses[1])  # partial -> replicated
                    self.assertEqual(losses[0], losses[1])
            # Expect one reduce-scatter per MLP on the last microbatch
            expected_reduce_scatter_count = num_mlps
            if not recurse:
                # Expect additional reduce-scatters for non-disabled MLPs
                expected_reduce_scatter_count += (
                    num_mlps - num_mlps_to_disable_reduce_scatter
                ) * (num_microbatches - 1)
            self.assertEqual(reduce_scatter_count, expected_reduce_scatter_count)
            reduce_scatter_count = 0
            for param in ref_model.parameters():
                if param.grad is not None:
                    param.grad.div_(self.world_size)
            check_sharded_parity(self, ref_model, model)
            for _optim in (optim, ref_optim):
                _optim.step()
                # When `set_to_none=False`, we are exercising mixing
                # gradient accumulation with and without communication
                _optim.zero_grad(set_to_none=(iter_idx % 2))


class TestFullyShard2DTraining(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    def init_global_mesh(self) -> DeviceMesh:
        # Prefer to test with >=4 GPUs, but for 2 GPUs, use 2-way TP
        dp_size = 2 if self.world_size > 2 else 1
        return init_device_mesh(
            "cuda", (dp_size, self.world_size // dp_size), mesh_dim_names=("dp", "tp")
        )

    @skip_if_lt_x_gpu(2)
    def test_train_parity_2d_mlp(self):
        global_mesh = self.init_global_mesh()
        self.run_subtests(
            {
                # "reshard_after_forward": [False, True],
                # "use_activation_checkpointing": [False, True],
                # "mlp_dim": [3, 16, 17],
                "reshard_after_forward": [True],
                "use_activation_checkpointing": [False],
                "mlp_dim": [3, 16, 17],
                "full_graph_compile": [True],
            },
            functools.partial(self._test_train_parity_2d_mlp, global_mesh),
        )

    def _test_train_parity_2d_mlp(
        self,
        global_mesh: DeviceMesh,
        reshard_after_forward: bool,
        use_activation_checkpointing: bool,
        mlp_dim: int,
        full_graph_compile: bool,
    ):
        dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]
        dp_pg = dp_mesh.get_group()  # used for `replicate()`

        torch.manual_seed(42)
        model = nn.Sequential(
            nn.LayerNorm(mlp_dim, bias=False),
            # Use multiplier of 3 to exercise uneven case
            MLP(mlp_dim, dim_multiplier=3),
            MLP(mlp_dim),
            MLP(mlp_dim, dim_multiplier=3),
        )
        ref_model = copy.deepcopy(model).cuda()
        replicate(ref_model, device_ids=[self.rank], process_group=dp_pg)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)

        model_for_eager = copy.deepcopy(model).cuda()
        model_for_eager = parallelize_module(
            model_for_eager,
            device_mesh=tp_mesh,
            # Leave the layer norm as implicitly replicated
            parallelize_plan={
                # Pass `use_local_output=False` to keep as DTensor to preserve
                # uneven activation dims
                "1.in_proj": ColwiseParallel(use_local_output=False),
                "1.out_proj": RowwiseParallel(use_local_output=False),
                "2.in_proj": ColwiseParallel(use_local_output=False),
                "2.out_proj": RowwiseParallel(use_local_output=False),
                "3.in_proj": ColwiseParallel(use_local_output=False),
                "3.out_proj": RowwiseParallel(),
            },
        )
        for mlp in model_for_eager:
            if use_activation_checkpointing:
                checkpoint(mlp)
            fully_shard(mlp, mesh=dp_mesh, reshard_after_forward=reshard_after_forward)
        fully_shard(model_for_eager, mesh=dp_mesh, reshard_after_forward=reshard_after_forward)
        optim_for_eager = torch.optim.Adam(model_for_eager.parameters(), lr=1e-2)

        if full_graph_compile:
            def compiler_fn(gm):
                return torch.compile(gm, backend="inductor", fullgraph=True)
            model_to_be_compiled = copy.deepcopy(model).cuda()
            model_to_be_compiled = parallelize_module(
                model_to_be_compiled,
                device_mesh=tp_mesh,
                # Leave the layer norm as implicitly replicated
                parallelize_plan={
                    # Pass `use_local_output=False` to keep as DTensor to preserve
                    # uneven activation dims
                    "1.in_proj": ColwiseParallel(use_local_output=False),
                    "1.out_proj": RowwiseParallel(use_local_output=False),
                    "2.in_proj": ColwiseParallel(use_local_output=False),
                    "2.out_proj": RowwiseParallel(use_local_output=False),
                    "3.in_proj": ColwiseParallel(use_local_output=False),
                    "3.out_proj": RowwiseParallel(),
                },
            )
            for mlp in model_to_be_compiled:
                if use_activation_checkpointing:
                    checkpoint(mlp)
                fully_shard(mlp, mesh=dp_mesh, reshard_after_forward=True, _reshard_after_forward_root=True)
            fully_shard(model_to_be_compiled, mesh=dp_mesh, reshard_after_forward=True, _reshard_after_forward_root=True)
            optim_for_compile = torch.optim.Adam(model_to_be_compiled.parameters(), lr=1e-2)

        compiled_model = None

        def get_compiled_model_and_optim(iter_idx):
            nonlocal compiled_model
            if not full_graph_compile:
                return None, None, False
            if iter_idx > 0:
                if compiled_model is None:
                    compiled_model = torch.compile(model_to_be_compiled, backend="inductor", fullgraph=True)
                return compiled_model, optim_for_compile, True
            else:
                return model_to_be_compiled, optim_for_compile, False

        torch.manual_seed(42 + dp_pg.rank() + 1)
        device = torch.device("cuda")
        for iter_idx in range(10):
            inp = torch.randn((8, mlp_dim), device=device)
            losses: List[torch.Tensor] = []
            for _model, _optim, _is_compile in ((ref_model, ref_optim, False), (model_for_eager, optim_for_eager, False), get_compiled_model_and_optim(iter_idx)):
                if _model is None:
                    continue
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                if _is_compile:
                    ctx = compiled_autograd.enable(compiler_fn)
                else:
                    ctx = contextlib.nullcontext()
                with ctx:
                    loss = _model(inp).sum()
                    losses.append(loss.item())
                    loss.backward()
                _optim.step()
            losses_tensors = [torch.tensor(x) for x in losses]
            self.assertTrue(all(torch.allclose(x, losses_tensors[0]) for x in losses_tensors), msg=f"iter_idx: {iter_idx}, losses_tensors: {losses_tensors}, losses: {losses}")

    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_train_parity_2d_transformer_checkpoint_resume(self):
        """
        Tests train parity of a 2D transformer without checkpointing against a
        2D transformer with a checkpoint save/load.
        """
        self.run_subtests(
            {
                "use_seq_parallel": [False, True],
                # If reusing, then load into the same model/optimizer instance
                # else construct new ones (requiring eager optim state init)
                "reuse_model_optim": [False, True],
                "optimizer_class": [torch.optim.Adam, torch.optim.AdamW],
            },
            self._test_train_parity_2d_transformer_checkpoint_resume,
        )

    def _test_train_parity_2d_transformer_checkpoint_resume(
        self,
        use_seq_parallel: bool,
        reuse_model_optim: bool,
        optimizer_class: Type[torch.optim.Optimizer],
    ):
        def train_step(
            _model: nn.Module, _optim: torch.optim.Optimizer, _inp: torch.Tensor
        ) -> torch.Tensor:
            loss = _model(_inp).sum()
            loss.backward()
            _optim.step()
            _optim.zero_grad()
            return loss

        def parallelize(_model: Transformer, mesh: DeviceMesh, use_seq_parallel: bool):
            _model = Transformer.parallelize(_model, mesh["tp"], use_seq_parallel)
            for layer in _model.layers:
                fully_shard(layer, mesh=mesh["dp"])
            fully_shard(_model, mesh=mesh["dp"])
            return _model

        global_mesh = self.init_global_mesh()
        # Baseline: run two iterations without checkpointing
        seed = 42
        torch.manual_seed(seed)
        model_args = ModelArgs(dropout_p=0.0)
        model_no_cp = parallelize(
            Transformer(model_args), global_mesh, use_seq_parallel
        )
        optim_no_cp = optimizer_class(model_no_cp.parameters(), lr=1e-2)

        torch.manual_seed(42 + global_mesh["dp"].get_local_rank() + 1)
        inp = torch.randint(0, model_args.vocab_size, (3, 16), device="cuda")
        loss_no_cp1 = train_step(model_no_cp, optim_no_cp, inp)
        loss_no_cp2 = train_step(model_no_cp, optim_no_cp, inp)

        # Test: run one iteration, save checkpoint, zero states or init new
        # model/optimizer, load checkpoint, and run another iteration
        torch.manual_seed(seed)
        model_cp = parallelize(Transformer(model_args), global_mesh, use_seq_parallel)
        optim_cp = optimizer_class(model_cp.parameters(), lr=1e-2)

        loss_cp1 = train_step(model_cp, optim_cp, inp)
        self.assertEqual(loss_no_cp1, loss_cp1)

        sharded_sd = {
            "model": get_model_state_dict(model_cp),
            # Use `get_optimizer_state_dict` to handle eager optim state init
            # when constructing a new optimizer instance
            "optim": get_optimizer_state_dict(model_cp, optim_cp),
        }
        dcp.save(
            state_dict=sharded_sd,
            storage_writer=dcp.FileSystemWriter(self.temp_dir),
        )
        if reuse_model_optim:
            with torch.no_grad():
                for param in model_cp.parameters():
                    param.zero_()
                optim_sd = optim_cp.state_dict()
                for param_states in optim_sd["state"].values():
                    for state_value in param_states.values():
                        if torch.is_tensor(state_value):
                            state_value.zero_()
        else:
            torch.manual_seed(seed + 1)  # different seed
            model_cp = parallelize(
                Transformer(model_args), global_mesh, use_seq_parallel
            )
            optim_cp = optimizer_class(model_cp.parameters(), lr=1e-2)
        self.assertNotEqual(loss_no_cp2, train_step(model_cp, optim_cp, inp))

        sharded_sd = {
            "model": get_model_state_dict(model_cp),
            "optim": get_optimizer_state_dict(model_cp, optim_cp),
        }
        dcp.load(
            state_dict=sharded_sd,
            storage_reader=dcp.FileSystemReader(self.temp_dir),
        )
        self.assertGreater(len(optim_cp.state_dict()["state"]), 0)

        loss_cp2 = train_step(model_cp, optim_cp, inp)
        self.assertEqual(loss_no_cp2, loss_cp2)


class TestFullyShardHSDPTraining(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_train_parity_hsdp(self):
        shard_size = 2 if self.world_size > 2 else 1
        replicate_size = self.world_size // shard_size
        global_mesh = init_device_mesh(
            "cuda", (replicate_size, shard_size), mesh_dim_names=("replicate", "shard")
        )
        self.run_subtests(
            {
                "reshard_after_forward": [False, True],
                "use_activation_checkpointing": [False, True],
                "mlp_dim": [3, 16, 17],
                "sync_gradients_at_last_batch": [True, False],
            },
            functools.partial(self._test_train_parity_hsdp, global_mesh),
        )

    def _test_train_parity_hsdp(
        self,
        global_mesh: DeviceMesh,
        reshard_after_forward: bool,
        use_activation_checkpointing: bool,
        mlp_dim: int,
        sync_gradients_at_last_batch: bool,
    ):
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.LayerNorm(mlp_dim, bias=False),
            MLP(mlp_dim, dim_multiplier=3),
            MLP(mlp_dim),
            MLP(mlp_dim, dim_multiplier=3),
        )
        ref_model = copy.deepcopy(model).cuda()
        replicate(ref_model, device_ids=[self.rank])
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for mlp in model:
            if use_activation_checkpointing:
                checkpoint(mlp)
            fully_shard(
                mlp, mesh=global_mesh, reshard_after_forward=reshard_after_forward
            )
        fully_shard(
            model, mesh=global_mesh, reshard_after_forward=reshard_after_forward
        )
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        check_sharded_parity(self, ref_model, model)
        torch.manual_seed(42 + self.rank + 1)
        device = torch.device("cuda")
        num_microbatches = 3
        for iter_idx in range(5):
            for microbatch_idx in range(num_microbatches):
                is_last_microbatch = microbatch_idx == num_microbatches - 1
                if sync_gradients_at_last_batch:
                    model.set_requires_gradient_sync(is_last_microbatch)
                inp = torch.randn((8, mlp_dim), device=device)
                losses: List[torch.Tensor] = []
                for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                    losses.append(_model(inp).sum())
                    losses[-1].backward()
                self.assertEqual(losses[0], losses[1])
            check_sharded_parity(self, ref_model, model)
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.step()
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            check_sharded_parity(self, ref_model, model)


if __name__ == "__main__":
    run_tests()
