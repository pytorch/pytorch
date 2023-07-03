import logging
from functools import partial

import torch
from ..backends.common import aot_autograd, mem_efficient_fusion_kwargs
from .registry import register_backend, register_debug_backend

log = logging.getLogger(__name__)


def prims_executor(gm, inputs, *, executor):
    from functorch.compile import make_boxed_func

    # This function is called once per forward/backward pass of a graph in AOT
    # Autograd. We use it to set up the nvFuser-specific FX graph and return
    # execute function.
    from torch._prims.context import TorchRefsNvfuserCapabilityMode
    from torch._prims.executor import execute
    from torch.fx.experimental.proxy_tensor import make_fx

    # AOT Autograd might not use the partitioner, so we need to make sure that
    # the graph is transformed to use nvFuser-compatible nodes.
    if not getattr(gm, "_nvprim_transformed", False):
        with TorchRefsNvfuserCapabilityMode():
            gm = make_fx(gm)(*inputs)

    # Then we return a callable that executes the "gm" graph
    return make_boxed_func(partial(execute, gm, executor=executor))


def nvprims_fw_bw_partition_fn(joint_module, joint_inputs, *, num_fwd_outputs):
    # This function is called once per forward+backward pass of a graph in AOT
    # Autograd. We use it to set up the nvFuser-specific FX graph that is later
    # passed to the executor.
    from functorch.compile import min_cut_rematerialization_partition

    from torch._prims.context import TorchRefsNvfuserCapabilityMode
    from torch.fx.experimental.proxy_tensor import make_fx

    # AOT Autograd expects arguments of the traced function to be named exactly
    # "primals, tangents"
    def func(primals, tangents):
        return joint_module(primals, tangents)

    # First we trace the graph conditionally decomposing nodes
    # that can be sent to the nvfuser executor
    with TorchRefsNvfuserCapabilityMode():
        prim_gm = make_fx(func)(*joint_inputs)

    # all nvprims for now
    recomputable_ops = {
        getattr(torch.ops.nvprims, prim)
        for prim in dir(torch.ops.nvprims)
        if isinstance(getattr(torch.ops.nvprims, prim), torch._ops.OpOverloadPacket)
        and getattr(torch.ops.nvprims, prim).is_recomputable
    }

    fw_gm, bw_gm = min_cut_rematerialization_partition(
        prim_gm,
        joint_inputs,
        recomputable_ops=recomputable_ops,
        num_fwd_outputs=num_fwd_outputs,
    )
    # AOT Autograd might not use the partitioner, so we need to make sure that
    # the graph is marked as already transformed to use nvFuser-compatible nodes
    fw_gm._nvprim_transformed = True
    bw_gm._nvprim_transformed = True
    return fw_gm, bw_gm


def create_nvprims_backend(*, executor):
    return aot_autograd(
        fw_compiler=partial(prims_executor, executor=executor),
        bw_compiler=partial(prims_executor, executor=executor),
        partition_fn=nvprims_fw_bw_partition_fn,
    )


aot_nvprims_nvfuser = create_nvprims_backend(executor="nvfuser")
aot_nvprims_aten = create_nvprims_backend(executor="aten")

# "nvprims" is a subset of PrimTorch primitives that are guaranteed to be
# supported by nvFuser. This is the preferred backend for nvFuser+PrimTorch.
register_backend(name="nvprims_nvfuser", compiler_fn=aot_nvprims_nvfuser)
# This is useful for debugging. Can be removed later.
register_debug_backend(name="nvprims_aten", compiler_fn=aot_nvprims_aten)


# Use min cut rematerialization and TorchScript+nvFuser with AOT Autograd
# aot_ts_nvfuser uses the memory efficient fusion algorithm from AOT Autograd.
# It uses min cut rematerialization algorithm, uses nvFuser as the
# compiler backend, and TorchScript as the frontend.
aot_mem_efficient_fusion = aot_autograd(**mem_efficient_fusion_kwargs(use_decomps=True))
aot_mem_efficient_fusion.backend_ctx_ctor = lambda: torch.jit.fuser("fuser2")
register_backend(name="aot_ts_nvfuser", compiler_fn=aot_mem_efficient_fusion)
