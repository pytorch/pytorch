import torch
import torch.distributed._functional_collectives
from torch.testing._internal.composite_compliance import is_view_fn


# just port some minor fixes from xformers
# which should be backported here
from xformers.checkpoint import selective_checkpoint_context_fn, OPS_TO_ALWAYS_SKIP

torch._dynamo.config._experimental_support_context_fn_in_torch_utils_checkpoint = True

N = 8
_REAL_GATHER = torch.distributed.is_available() and torch.distributed.is_initialized()


# this is here just for ease of debugging / experimentation
# without having to spawn multi-process jobs

_my_new_lib = torch.library.Library("my_new_lib", "DEF")
_my_new_lib.define("all_gather(Tensor t) -> Tensor")
_my_new_lib.impl("all_gather", lambda x: torch.cat([x] * N, dim=0), "CUDA")
_my_new_lib.impl("all_gather", lambda x: torch.cat([x] * N, dim=0), "Meta")

_my_new_lib.define("reduce_scatter(Tensor t) -> Tensor")
_my_new_lib.impl("reduce_scatter", lambda x: x.chunk(N, dim=0)[0].clone(), "CUDA")
_my_new_lib.impl("reduce_scatter", lambda x: x.chunk(N, dim=0)[0], "Meta")

_my_new_lib.define("wait(Tensor t) -> Tensor")
_my_new_lib.impl("wait", lambda x: x.clone(), "CUDA")
_my_new_lib.impl("wait", lambda x: torch.empty_like(x), "Meta")


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        tag = ""
        if _REAL_GATHER:
            group_size = torch.distributed.get_world_size()
            ranks = list(range(group_size))
            rank = torch.distributed.get_rank()
            out = torch.ops.c10d_functional.all_gather_into_tensor(x, tag, ranks, group_size)
            out = torch.ops.c10d_functional.wait_tensor(out)
            return out
        else:
            x = torch.ops.my_new_lib.all_gather(x)
            return torch.ops.my_new_lib.wait(x)

    @staticmethod
    def backward(ctx, grad):
        if _REAL_GATHER:
            reduce_op = "sum"
            tag = ""
            group_size = torch.distributed.get_world_size()
            rankset = list(range(group_size))
            grad = torch.ops.c10d_functional.reduce_scatter_tensor(
                grad,
                reduce_op,
                tag,
                rankset,
                group_size
            )
            grad = torch.ops.c10d_functional.wait_tensor(grad)
        else:
            grad = torch.ops.my_new_lib.reduce_scatter(grad)
            grad = torch.ops.my_new_lib.wait(grad)
        return grad


def _my_all_gather(x):
    return _AllGather.apply(x)


class MyParam(torch.nn.Module):
    def forward(self, x):
        return _my_all_gather(x)

    def right_inverse(self, x):
        if _REAL_GATHER:
            group_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            return x.chunk(group_size, dim=0)[rank].clone()

        return x.chunk(N, dim=0)[0].clone()


def _policy_fn(mode, func, *args, **kwargs):
    if func in OPS_TO_ALWAYS_SKIP:
        return False
    if _REAL_GATHER:
        recomp = func not in {torch.ops.c10d_functional.all_gather_into_tensor.default, torch.ops.c10d_functional.wait_tensor.default}
    else:
        recomp = func not in {torch.ops.my_new_lib.all_gather.default, torch.ops.my_new_lib.wait.default}
    recomp = recomp and (not is_view_fn(func))
    # This is just a hack for eager for now
    if func is torch.ops.aten.add.Tensor:
        recomp = False

    return recomp


def context_fn():
    return selective_checkpoint_context_fn(_policy_fn)


class FSDP(torch.nn.Module):
    def apply_parametrization(self, module):
        params = list(module._parameters.items())
        for name, p in params:
            if p is not None:
                torch.nn.utils.parametrize.register_parametrization(
                    module, name, MyParam()
                )

    def __init__(self, modules):
        super().__init__()
        self.module = modules
        if True:
            modules = list(self.module.modules())
            for mod in modules:
                self.apply_parametrization(mod)

    def forward(self, *args, **kwargs):
        x = args[0]
        assert len(args) == 1
        assert len(kwargs) == 0
        for module in self.module:
            x = torch.utils.checkpoint.checkpoint(module, x, use_reentrant=False, context_fn=context_fn, **kwargs)
        return x
