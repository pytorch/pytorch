# Owner(s): ["oncall: distributed"]


import copy
import unittest

import torch
import torch._dynamo.testing
from torch.distributed.fsdp import (
    fully_shard,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype, MLP
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.inductor_utils import HAS_GPU


device_type = torch.device(get_devtype())


class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 1024, device=device_type),
            torch.nn.Linear(1024, 1024, device=device_type),
            torch.nn.Linear(1024, 4096, device=device_type),
        )

    def forward(self, x):
        return self.encoder(x)


class TestFullyShardCompileCompute(FSDPTest):
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_disable_compiling_hooks(self):
        """Verify that dynamo never traces into FSDP hooks."""
        torch._dynamo.reset()
        trace_rules_check_count = 0
        HOOKS_FILE_NAME = "torch/distributed/fsdp/_fully_shard/_fsdp_state.py"
        HOOK_WRAPPER_NAME = "wrapper"

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

        orig_trace_rules_check = torch._dynamo.trace_rules.check
        torch.distributed.barrier()
        torch._dynamo.trace_rules.check = patched_trace_rules_check
        model = MLP(4).to(device_type)
        fully_shard(model)
        model.compile()
        model(torch.randn((4, 4), device=device_type))
        torch.distributed.barrier()
        torch._dynamo.trace_rules.check = orig_trace_rules_check
        self.assertEqual(trace_rules_check_count, 0)


@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
class TestFullyShardCompile(FSDPTest):
    def test_dynamo_trace_use_training_state(self):
        torch._dynamo.reset()
        # Construct a dummy FSDPParamGroup, since we just want to test the `use_training_state` ctx manager.
        param_group = FSDPParamGroup(
            [],  # params: List[nn.Parameter],
            (torch.nn.Linear(1, 1),),  # module: Tuple[nn.Module, ...],
            None,  # mesh_info: FSDPMeshInfo,
            None,  # post_forward_mesh_info: Optional[FSDPMeshInfo],
            device_type,  # device: torch.device,
            None,  # shard_placement_fn: Optional[Callable],
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

    def test_trace_fsdp_copy_(self):
        @torch.library.custom_op("mylib::add_one_out", mutates_args={"out"})
        def add_one_out(x: torch.Tensor, out: torch.Tensor) -> None:
            torch.add(x, 1, out=out)

        def f(x):
            buf = torch.zeros(2)
            buf_view = buf.view(-1)
            torch.ops.mylib.add_one_out(x, out=buf_view)
            buf_view2 = buf.view(-1)
            torch.ops.fsdp.copy_(x, buf_view2)

        ref_x = torch.zeros(2)
        x = copy.deepcopy(ref_x)
        f(ref_x)
        torch.compile(f, backend="aot_eager")(x)
        self.assertEqual(x, ref_x)

    def test_dynamo_recompiles_on_fsdp_layers(self):
        m = Mod()
        for name, child in m.encoder.named_children():
            if isinstance(child, torch.nn.Linear):
                new_child = torch.compile(child)
                setattr(m.encoder, name, new_child)
        m = FSDP(m, sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True)
        inp = torch.randn(32, 784, device=device_type)
        m(inp)


if __name__ == "__main__":
    run_tests()
