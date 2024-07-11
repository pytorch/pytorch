# Owner(s): ["oncall: distributed"]


import contextlib
import copy
import unittest

import torch
import torch._dynamo.testing
import torch.distributed._composable.fsdp._fsdp_param
from torch import nn
from torch._dynamo import compiled_autograd

from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._composable.fsdp._fsdp_common import TrainingState
from torch.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup
from torch.distributed._tensor import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, MLP
from torch.testing._internal.common_utils import run_tests, skipIfRocm
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)
from torch.utils._triton import has_triton


class TestFullyShardCompileCompute(FSDPTest):
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_disable_compiling_hooks(self):
        self.run_subtests(
            {
                "skip_fsdp_hooks": [False, True],
            },
            self._test_disable_compiling_hooks,
        )

    def _test_disable_compiling_hooks(
        self,
        skip_fsdp_hooks: bool,
    ):
        torch._dynamo.reset()
        trace_rules_check_count = 0
        HOOKS_FILE_NAME = "torch/distributed/_composable/fsdp/_fsdp_state.py"
        HOOK_WRAPPER_NAME = "fsdp_hook_wrapper"

        def patched_trace_rules_check(*args, **kwargs):
            nonlocal trace_rules_check_count
            f_code = args[0]
            if (
                hasattr(f_code, "co_filename")
                and f_code.co_filename.endswith(HOOKS_FILE_NAME)
                and f_code.co_name != HOOK_WRAPPER_NAME
            ):
                trace_rules_check_count += 1
            return orig_trace_rules_check(*args, **kwargs)

        original_skip_fsdp_hooks = torch._dynamo.config.skip_fsdp_hooks
        orig_trace_rules_check = torch._dynamo.trace_rules.check
        torch.distributed.barrier()
        torch._dynamo.config.skip_fsdp_hooks = skip_fsdp_hooks
        torch._dynamo.trace_rules.check = patched_trace_rules_check
        model = MLP(4)
        fully_shard(model)
        model.compile()
        model(torch.randn((4, 4), device="cuda"))
        torch.distributed.barrier()
        torch._dynamo.config.skip_fsdp_hooks = original_skip_fsdp_hooks
        torch._dynamo.trace_rules.check = orig_trace_rules_check
        if skip_fsdp_hooks:
            self.assertEqual(trace_rules_check_count, 0)
        else:
            self.assertTrue(trace_rules_check_count > 0)


class TestFullyShardCompile(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(2, torch.cuda.device_count())

    def test_dynamo_trace_use_training_state(self):
        torch._dynamo.reset()
        # Construct a dummy FSDPParamGroup, since we just want to test the `use_training_state` ctx manager.
        param_group = FSDPParamGroup(
            [],  # params: List[nn.Parameter],
            torch.nn.Linear(1, 1),  # module: nn.Module,
            None,  # mesh_info: FSDPMeshInfo,
            None,  # post_forward_mesh_info: Optional[FSDPMeshInfo],
            None,  # device: torch.device,
            None,  # mp_policy: MixedPrecisionPolicy,
            None,  # offload_policy: OffloadPolicy,
        )

        def f(x):
            param_group._training_state = TrainingState.IDLE
            with param_group.use_training_state(TrainingState.FORWARD):
                if param_group._training_state == TrainingState.FORWARD:
                    return x + 1
                else:
                    return x

        inp = torch.zeros(1)
        self.assertEqual(param_group._training_state, TrainingState.IDLE)

        eager_out = f(inp)
        self.assertEqual(param_group._training_state, TrainingState.IDLE)
        self.assertEqual(eager_out, inp + 1)

        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        compiled_out = torch.compile(f, backend=cnt, fullgraph=True)(inp)
        self.assertEqual(param_group._training_state, TrainingState.IDLE)
        self.assertEqual(eager_out, compiled_out)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)
        self.assertEqual(len(cnt.graphs), 1)

    def test_trace_fsdp_set_(self):
        @torch.library.custom_op("mylib::add_one_out", mutates_args={"out"})
        def add_one_out(x: torch.Tensor, out: torch.Tensor) -> None:
            torch.add(x, 1, out=out)

        def f(x):
            buf = torch.zeros(2)
            buf_view = buf.view(-1)
            torch.ops.mylib.add_one_out(x, out=buf_view)
            buf_view2 = buf.view(-1)
            torch.ops.fsdp.set_(x, buf_view2)

        ref_x = torch.zeros(2)
        x = copy.deepcopy(ref_x)
        f(ref_x)
        torch.compile(f, backend="aot_eager")(x)
        self.assertEqual(x, ref_x)

    @torch._dynamo.config.patch(inline_inbuilt_nn_modules=True)
    @torch._functorch.config.patch(recompute_views=True)
    @torch._functorch.config.patch(cse=False)
    def _test_traceable_fsdp(
        self, model_init_fn, input_creation_fn, backend, fullgraph
    ):
        def compiler_fn(compiled_autograd_backend):
            def _fn(gm):
                # fullgraph=True because graph-break in Compiled Autograd BWD graph is not supported by Traceable FSDP2 yet
                # (main difficulty comes from queue_callback not working well when BWD has graph break).
                return torch.compile(
                    gm, backend=compiled_autograd_backend, fullgraph=True
                )

            return _fn

        def run_iters(model, optim, n_iter=10, compiled_autograd_backend=None):
            torch.manual_seed(42)
            losses = []
            for i in range(n_iter):
                inp = input_creation_fn()
                if compiled_autograd_backend is not None:
                    maybe_compiled_autograd_ctx = compiled_autograd.enable(
                        compiler_fn(compiled_autograd_backend)
                    )
                else:
                    maybe_compiled_autograd_ctx = contextlib.nullcontext()
                with maybe_compiled_autograd_ctx:
                    out = model(inp)
                    loss = out.sum()
                    losses.append(loss.item())
                    loss.backward()
                optim.step()
                optim.zero_grad(set_to_none=True)
            return losses

        def test_compiled():
            model, optim = model_init_fn()
            # FSDP2 does lazy init using 1st run, so run it once to init using eager mode
            run_iters(model, optim, n_iter=1)

            model_compiled = torch.compile(model, backend=backend, fullgraph=True)
            res = run_iters(model_compiled, optim, compiled_autograd_backend=backend)
            return res

        def test_eager():
            model, optim = model_init_fn()
            # FSDP2 does lazy init using 1st run, so run it once to init using eager mode
            run_iters(model, optim, n_iter=1)

            res = run_iters(model, optim)
            return res

        losses_compiled = test_compiled()
        losses_eager = test_eager()
        for loss_compiled, loss_eager in zip(losses_compiled, losses_eager):
            self.assertTrue(
                torch.allclose(
                    torch.tensor(loss_compiled),
                    torch.tensor(loss_eager),
                    rtol=1e-5,
                    atol=1e-8,
                ),
                f"{loss_compiled} vs {loss_eager}",
            )

    def _create_simple_mlp_factory_fns(self):
        hidden_dim = 16

        def model_init_fn():
            torch.manual_seed(self.rank)
            fsdp_config = {}
            model = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, device="cuda"),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, device="cuda"),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, device="cuda"),
            )
            fully_shard(model, reshard_after_forward=True, **fsdp_config)
            optim = torch.optim.SGD(model.parameters(), lr=1e-4)
            return model, optim

        def input_creation_fn():
            torch.manual_seed(self.rank)
            inp = torch.randn((2, hidden_dim), device="cuda", requires_grad=False)
            return inp

        return model_init_fn, input_creation_fn

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_simple_mlp_fullgraph_backend_aot_eager(self):
        self._test_traceable_fsdp(
            *self._create_simple_mlp_factory_fns(), "aot_eager", fullgraph=True
        )

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_simple_mlp_fullgraph_backend_aot_eager_decomp_partition(self):
        self._test_traceable_fsdp(
            *self._create_simple_mlp_factory_fns(),
            "aot_eager_decomp_partition",
            fullgraph=True,
        )

    @skipIfRocm
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_simple_mlp_fullgraph_backend_inductor(self):
        self._test_traceable_fsdp(
            *self._create_simple_mlp_factory_fns(), "inductor", fullgraph=True
        )

    def _create_transformer_factory_fns(self):
        seq_len = 16
        vocab_size = 8

        def model_init_fn():
            torch.manual_seed(self.rank)
            fsdp_config = {}
            mesh = init_device_mesh("cuda", (self.world_size,))
            model_args = ModelArgs(vocab_size=vocab_size)
            model = Transformer(model_args)
            for layer_id, mod in enumerate(model.layers):
                fully_shard(mod, mesh=mesh, reshard_after_forward=True, **fsdp_config)
            model = fully_shard(
                model, mesh=mesh, reshard_after_forward=True, **fsdp_config
            )
            optim = torch.optim.SGD(model.parameters(), lr=1e-4)
            return model, optim

        def input_creation_fn():
            torch.manual_seed(self.rank)
            inp = torch.randint(
                0, vocab_size, (2, seq_len), device="cuda", requires_grad=False
            )
            return inp

        return model_init_fn, input_creation_fn

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_transformer_fullgraph_backend_aot_eager(self):
        self._test_traceable_fsdp(
            *self._create_transformer_factory_fns(), "aot_eager", fullgraph=True
        )

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    # TODO: native_dropout has worse accuracy after decomp, need to figure out why
    @torch._inductor.config.patch(fallback_random=True)
    def test_transformer_fullgraph_backend_aot_eager_decomp_partition(self):
        self._test_traceable_fsdp(
            *self._create_transformer_factory_fns(),
            "aot_eager_decomp_partition",
            fullgraph=True,
        )

    @skipIfRocm
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # TODO: native_dropout causes CUDA IMA error, need to figure out why
    @torch._inductor.config.patch(fallback_random=True)
    def test_transformer_fullgraph_backend_inductor(self):
        self._test_traceable_fsdp(
            *self._create_transformer_factory_fns(), "inductor", fullgraph=True
        )


if __name__ == "__main__":
    run_tests()
