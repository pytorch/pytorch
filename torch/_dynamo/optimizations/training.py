import functools
import logging
import operator
from collections import defaultdict
from functools import partial
from importlib import import_module
from typing import Set

import torch
from torch.fx import GraphModule
from torch.fx.passes.backends.cudagraphs import partition_cudagraphs
from torch.multiprocessing.reductions import StorageWeakRef
from torch.nn import Module
from torch.utils._pytree import tree_map

from .. import config
from ..utils import clone_inputs, count_calls, counters
from .analysis import has_mutation
from .backends import BACKENDS
from .normalize import normalize_ir

log = logging.getLogger(__name__)


def is_aot_autograd_safe_to_run(gm, example_inputs):
    """
    There are some known issues with Aot Autograd. This is a workaround to catch
    such cases, and fallback to eager. We should fix these quickly.

    Issues
    1) LSTM - https://github.com/pytorch/torchdynamo/issues/1147
    2) LSTM - https://github.com/pytorch/functorch/issues/586
    3) Input mutation - https://github.com/pytorch/torchdynamo/issues/1301
    """

    def raise_or_warn(reason):
        msg = f"Unable to use Aot Autograd because of presence of {reason}"
        if config.raise_on_unsafe_aot_autograd:
            raise NotImplementedError(msg)
        else:
            log.warning(msg)
        return False

    import functorch.compile

    # 1) LSTM module (tts_angular) - https://github.com/pytorch/functorch/issues/586
    for submod in gm.modules():
        if submod.__class__.__name__ == "LSTM":
            return raise_or_warn("LSTM")

    # 2) Mutation in the graph
    mutated = False
    try:
        if functorch.compile.config.use_functionalize:
            # There are two problematic classes we still exclude for now with
            # functionalization:
            #   - data mutation of inputs (fixed when we stop recording the
            #   copy_ directly into the graph)
            #   - metadata mutation of inputs (fixed if we do an extra partition
            #   to avoid AotAutograd on the mutated inputs, or if we some how
            #   get custom autograd function to reflect metadata changes to the
            #   original tensor)
            mutated = has_mutation(gm, example_inputs, inputs_only=True)
        else:
            mutated = has_mutation(gm, example_inputs)
    except NotImplementedError as e:
        if "SparseTensorImpl" not in str(e):
            # TODO - TorchDynamo mutation analysis cannot handle sparse tensors.
            # So, there is a chance that we could call Aot Autograd when it is
            # unsafe.
            # The exception is fairly guarded with string check, so any other
            # mutation analysis bugs will raise exceptions and will be caught.
            raise e
        pass

    if mutated:
        return raise_or_warn("mutation")

    return True


class AotAutogradStrategy(object):
    """Base class for backend strategies that use AOT Autograd"""

    @classmethod
    def compile_fn(cls, gm: torch.fx.GraphModule, example_inputs):
        if count_calls(gm.graph) < 2:
            return gm  # no point for tiny graphs
        return cls(gm, example_inputs).verified_candidate()

    def __init__(self, gm: torch.fx.GraphModule, example_inputs):
        import functorch.compile

        functorch.compile.config.use_functionalize = True
        functorch.compile.config.use_fake_tensor = True

        super(AotAutogradStrategy, self).__init__()
        counters["aot_autograd"]["total"] += 1
        self.use_fallback = False
        self.original_example_inputs = example_inputs
        self.gm = gm

        if not functorch.compile.config.use_functionalize and config.normalize_ir:
            try:
                self.gm = normalize_ir(gm, self.example_inputs)
            except Exception:
                log.debug("TorchDynamo unable to remove mutation")
                self.use_fallback = True
                pass

        if not is_aot_autograd_safe_to_run(gm, example_inputs):
            self.use_fallback = True

    @property
    def example_inputs(self):
        return clone_inputs(self.original_example_inputs)

    def verified_candidate(self):
        if self.use_fallback:
            log.debug("Unable to use AOT Autograd because graph has mutation")
            counters["aot_autograd"]["not_ok"] += 1
            return self.gm
        cg = self.candidate()
        if cg is None:
            counters["aot_autograd"]["not_ok"] += 1
            raise RuntimeError("AOT Autograd failed to compile")
        counters["aot_autograd"]["ok"] += 1
        return cg

    def candidate(self):
        raise NotImplementedError()


class AotNop(AotAutogradStrategy):
    """Useful for debugging purpose"""

    def candidate(self):
        from functorch.compile import nop

        return BACKENDS["aot_autograd"](self.gm, self.example_inputs, fw_compiler=nop)


aot_eager = AotNop.compile_fn


class AotTorchscript(AotAutogradStrategy):
    """
    AOT Autograd with torchscript backend. Default partitioner.
    """

    def candidate(self):
        from functorch.compile import ts_compile

        return BACKENDS["aot_autograd"](
            self.gm, self.example_inputs, fw_compiler=ts_compile
        )


aot_ts = AotTorchscript.compile_fn

# Global counter to differentiate between different graphs.
graph_idx = 0


class AotPrint(AotNop):
    """Saves all the gm models so that we can run them separately"""

    def candidate(self):
        global graph_idx
        module_idx = "module_" + str(graph_idx)
        self.gm.to_folder(module_idx, "Bar")
        for idx, x in enumerate(self.example_inputs):
            torch.save(x, module_idx + "_tensor" + str(idx) + ".pt")
        graph_idx += 1
        return super(AotPrint, self).candidate()


aot_print = AotPrint.compile_fn


def mem_efficient_fusion_kwargs(use_decomps):
    from functorch.compile import (
        default_decompositions,
        min_cut_rematerialization_partition,
        ts_compile,
    )

    kwargs = {
        # these are taken from memory_efficient_fusion()
        "fw_compiler": ts_compile,
        "bw_compiler": ts_compile,
        "partition_fn": min_cut_rematerialization_partition,
    }

    if use_decomps:
        kwargs["decompositions"] = default_decompositions

    return kwargs


class AotMemEfficientFusion(AotAutogradStrategy):
    """Use Min cut rematerilization and TorchScript+nvFuser with AOT Autograd"""

    def candidate(self):
        kwargs = mem_efficient_fusion_kwargs(use_decomps=True)
        return BACKENDS["aot_autograd"](self.gm, self.example_inputs, **kwargs)


class AotMemEfficientFusionNoDecomps(AotAutogradStrategy):
    """Use Min cut rematerilization and TorchScript+nvFuser with AOT Autograd"""

    def candidate(self):
        kwargs = mem_efficient_fusion_kwargs(use_decomps=False)
        return BACKENDS["aot_autograd"](self.gm, self.example_inputs, **kwargs)


class AotInductorDebug(AotAutogradStrategy):
    """
    Uses TorchInductor Aot Autograd decopms and partitioner to isolate aot vs
    inductor problems.
    """

    def candidate(self):
        from functorch.compile import min_cut_rematerialization_partition, nop

        decompositions = import_module(
            f"{config.inductor_import}.compile_fx"
        ).select_decomp_table()

        kwargs = {
            # these are taken from memory_efficient_fusion()
            "fw_compiler": nop,
            "bw_compiler": nop,
            "decompositions": decompositions,
            "partition_fn": functools.partial(
                min_cut_rematerialization_partition, compiler="inductor"
            ),
        }
        return BACKENDS["aot_autograd"](self.gm, self.example_inputs, **kwargs)


aot_inductor_debug = AotInductorDebug.compile_fn


class AOTMemEfficientFusionWithContext:
    """Pass TorchScript+nvFuser context to TorchDynamo"""

    def __init__(self, use_decomps=True):
        self.backend_ctx_ctor = lambda: torch.jit.fuser("fuser2")
        self.use_decomps = use_decomps

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        if self.use_decomps:
            return AotMemEfficientFusion.compile_fn(gm, example_inputs)
        else:
            return AotMemEfficientFusionNoDecomps.compile_fn(gm, example_inputs)


aot_mem_efficient_fusion = AOTMemEfficientFusionWithContext(True)
aot_mem_efficient_fusion_no_decomp = AOTMemEfficientFusionWithContext(False)


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


def nvprims_fw_bw_partition_fn(joint_module, joint_inputs):
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
    } - {
        # Remove random ops from recomputable ops
        torch.ops.nvprims.rand_like,
        # Remove reduction/normalization ops from recomputable ops
        torch.ops.nvprims.sum,
        torch.ops.nvprims.var,
        torch.ops.nvprims.var_mean,
        torch.ops.nvprims.native_batch_norm,
        torch.ops.nvprims.amax,
        torch.ops.nvprims.amin,
    }

    fw_gm, bw_gm = min_cut_rematerialization_partition(
        prim_gm, joint_inputs, recomputable_ops=recomputable_ops
    )
    # AOT Autograd might not use the partitioner, so we need to make sure that
    # the graph is marked as already transformed to use nvFuser-compatible nodes
    fw_gm._nvprim_transformed = True
    bw_gm._nvprim_transformed = True
    return fw_gm, bw_gm


def create_nvprims_backend(*, executor):
    class NvPrims(AotAutogradStrategy):
        def __init__(self, gm: torch.fx.GraphModule, example_inputs):
            super(NvPrims, self).__init__(gm, example_inputs)
            self.executor = executor

        def candidate(self):
            from torch._dynamo import disable

            return BACKENDS["aot_autograd"](
                self.gm,
                self.example_inputs,
                fw_compiler=partial(prims_executor, executor=self.executor),
                bw_compiler=partial(prims_executor, executor=self.executor),
                partition_fn=disable(nvprims_fw_bw_partition_fn),
            )

    return NvPrims


aot_nvprims_nvfuser = create_nvprims_backend(executor="nvfuser").compile_fn
aot_nvprims_aten = create_nvprims_backend(executor="aten").compile_fn


def cloner(t):
    if isinstance(t, torch.Tensor):
        return t.clone()
    else:
        return t


class CudaGraphModule(Module):
    gm: GraphModule
    mutated_inputs: Set[int]

    def __init__(self, gm, mutated_inputs):
        super().__init__()
        self.gm = gm
        self.mutated_inputs = mutated_inputs

    warmed_up = False

    # these are all None or all filled
    graph = None
    static_inputs = None
    static_outputs = None

    # NB: we override __call__ as we don't need any nn.Module machinery
    # and to reduce overhead
    def __call__(self, *args):
        # TODO: once we've recorded here, we'd like to replace the __call__
        # implementation with compiled bytecode that copies into static, replays
        # the cuda graph, then copies out.  First condition is the hotpath,
        # needs optimizing
        if self.graph is not None:
            assert len(args) == len(self.static_inputs)
            for dst, src in zip(self.static_inputs, args):
                dst.copy_(src)
            self.graph.replay()
            for i in self.mutated_inputs:
                args[i].copy_(self.static_inputs[i])
            return tree_map(cloner, self.static_outputs)

        elif self.warmed_up:
            # record
            self.static_inputs = [x.clone() for x in args]
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_outputs = self.gm(*self.static_inputs)
            # NB: recording doesn't actually run the operations, so
            # now we immediately replay the graph to serve up the result
            self.graph.replay()
            for i in self.mutated_inputs:
                args[i].copy_(self.static_inputs[i])
            return tree_map(cloner, self.static_outputs)

        else:
            # warmup
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                r = self.gm(*args)
            torch.cuda.current_stream().wait_stream(stream)
            self.warmed_up = True
            return r


# Interpreter versions of these passes can be found at
# https://gist.github.com/ezyang/df2d746cac3b2c7d55c181e37c57ef23


def find_input_mutations(g):
    def meta_fk(meta):
        return meta["val"] if "val" in meta else meta["fake_result"]

    inputs = defaultdict(set)
    input_idx = 0
    mutated_inputs = set()
    for n in g.nodes:
        if n.op == "placeholder":
            inputs[StorageWeakRef(meta_fk(n.meta).storage())].add(input_idx)
            input_idx += 1
        elif n.op == "call_function":
            if n.target is operator.getitem:
                continue
            schema = n.target._schema
            for i, arg in enumerate(schema.arguments):
                if i < len(n.args):
                    argument = n.args[i]
                else:
                    if arg.name not in n.kwargs:
                        continue
                    argument = n.kwargs[arg.name]
                mut_arg = False
                if arg.alias_info:
                    if arg.alias_info.is_write:
                        mut_arg = True
                if mut_arg:
                    # TODO: not correct for args that contain tensors in a struct
                    # like list
                    mutated_inputs |= inputs[
                        StorageWeakRef(meta_fk(argument.meta).storage())
                    ]
        # TODO: error on unrecognized nodes
    return mutated_inputs


# Mutates input graph
def apply_cuda_graphs(gm):
    for n in gm.graph.nodes:
        if n.op == "call_module":
            assert not n.kwargs
            submod = gm.get_submodule(n.target)
            gm.delete_submodule(n.target)
            mutated_inputs = find_input_mutations(submod.graph)
            gm.add_submodule(n.target, CudaGraphModule(submod, mutated_inputs))
    # NB: we didn't actually change the graph, no need for recompile


def cudagraphs(model, inputs):
    model = partition_cudagraphs(model, inputs)
    apply_cuda_graphs(model)
    return model


def raw_aot_autograd_cudagraphs(model, inputs):
    kwargs = {
        # these are taken from memory_efficient_fusion()
        "fw_compiler": cudagraphs,
        "bw_compiler": cudagraphs,
    }

    def _wrapped_bw_compiler(*args, **kwargs):
        # stop TorchDynamo from trying to compile our generated backwards pass
        return disable(disable(bw_compiler)(*args, **kwargs))  # type: ignore[operator]

    bw_compiler = kwargs.get("bw_compiler") or kwargs["fw_compiler"]
    kwargs["bw_compiler"] = _wrapped_bw_compiler

    from functorch.compile import aot_module_simplified  # type: ignore[import]

    from .. import disable

    return aot_module_simplified(model, **kwargs)


class AotAutogradCudaGraphs(AotAutogradStrategy):
    def candidate(self):
        return raw_aot_autograd_cudagraphs(self.gm, self.example_inputs)


aot_cudagraphs = AotAutogradCudaGraphs.compile_fn


def create_aot_backends():
    """
    Register aliases for the AOT backends
    """
    # aot_eager uses AOT Autograd backend with nop compiler. It is helpful in debugging.
    BACKENDS["aot_eager"] = aot_eager

    # aot_eager uses AOT Autograd backend with print compiler. It prints the
    # graphs and also saves the graph modules that are sent to AOT Autograd.
    # This is helpful for debugging.
    BACKENDS["aot_print"] = aot_print

    # aot_ts uses torchscript backend. We can use this with both nnc and nvfuser
    # by using the relevant fuser with torch.jit.fuser(...)
    BACKENDS["aot_ts"] = aot_ts

    # "nvprims" is a subset of PrimTorch primitives that are guaranteed to be
    # supported by nvFuser. This is the preferred backend for nvFuser+PrimTorch.
    BACKENDS["nvprims_nvfuser"] = aot_nvprims_nvfuser
    # This is useful for debugging. Can be removed later.
    BACKENDS["nvprims_aten"] = aot_nvprims_aten

    # aot_ts_nvfuser uses the memory efficient fusion algorithm from AOT Autograd.
    # It uses min cut rematerialization algorithm, uses nvFuser as the
    # compiler backend, and TorchScript as the frontend.
    BACKENDS["aot_ts_nvfuser"] = aot_mem_efficient_fusion

    # Similar to aot_ts_nvfuser, but disables the decompositions. Decompositions
    # can cause accuracy deviations. This setting allows us to compare accuracy
    # without worrying about the impact of decomposisitons. More details at
    # https://github.com/pytorch/torchdynamo/issues/611
    BACKENDS["aot_ts_nvfuser_nodecomps"] = aot_mem_efficient_fusion_no_decomp

    # aot_cudagraphs only applies CUDA graphs to the graph.  It is also helpful
    # for debugging and can serve as a perf baseline.
    BACKENDS["aot_cudagraphs"] = aot_cudagraphs

    # aot_inductor_debug just replaces the inductor compiler with nop to help
    # isolate inductor vs aot_eager errors
    BACKENDS["aot_inductor_debug"] = aot_inductor_debug
