"""
This module implements distributed training optimizations for TorchDynamo backends.

It provides functionality to optimize models wrapped in DistributedDataParallel (DDP)
by intelligently splitting compiled graphs to align with DDP's gradient synchronization
boundaries. Key features include:

- Graph partitioning based on parameter bucket sizes
- Optimization of allreduce operations for distributed training
- Support for parameter ignoring and buffer handling
- Submodule compilation and management
- Debugging utilities for distributed training

The main component is the DDPOptimizer class, which handles graph splitting and
recompilation to enable efficient distributed training while maintaining the benefits
of compilation.
"""

import logging
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from unittest import mock

import torch
from torch import fx
from torch._dynamo.backends.registry import CompiledFn, CompilerFn
from torch._dynamo.output_graph import GraphCompileReason
from torch._dynamo.utils import deepcopy_to_fake_tensor, detect_fake_mode
from torch._logging import trace_structured
from torch.fx.node import Node


if TYPE_CHECKING:
    from torch._functorch._aot_autograd.schemas import ViewAndMutationMeta


# Regular log messages should go through 'log'.
# ddp_graph_log is a separate artifact logger reserved for dumping graphs.
# See docs/source/logging.rst for more info.
log = logging.getLogger(__name__)
ddp_graph_log = torch._logging.getArtifactLogger(__name__, "ddp_graphs")


def args_str(args: Any) -> str:
    # a debug helper
    if torch.is_tensor(args):
        return f"T[{args.shape}]"
    elif isinstance(args, tuple):
        return f"tuple({', '.join([args_str(x) for x in args])})"
    elif isinstance(args, list):
        return f"list({', '.join([args_str(x) for x in args])})"
    else:
        return str(args)


@dataclass
class Bucket:
    size: int = 0
    params: list[str] = field(default_factory=list)
    nodes: list[fx.Node] = field(default_factory=list)

    # param_ids is just used for unit testing
    param_ids: list[int] = field(default_factory=list)

    # keep track of any buckets that were extended for logging purposes
    opcount_increased_to_capture_external_output: int = 0
    paramsize_before_opcount_increase: int = 0


def bucket_has_external_output(bucket: Bucket) -> bool:
    nodes_in_bucket = set()
    # we want to iterate in reverse order, but clumsi-luckily the bucket.nodes list was already created backwards
    # so we don't reverse it here
    for node in bucket.nodes:
        # assume node.op != output, since those are filtered in the original iteration
        nodes_in_bucket.add(node)
        for user in node.users:
            if user not in nodes_in_bucket:
                return True
    return False


def pretty_print_buckets(buckets: list[Bucket], bucket_bytes_cap: int) -> None:
    headers = ("Index", "Size (b)", "Param Names")
    rows: list[tuple[int | None, int | None, str]] = []
    # pyrefly: ignore [implicit-any]
    extended_buckets = []
    for idx, bucket in enumerate(reversed(buckets)):
        if len(bucket.params) > 0:
            rows.append((idx, bucket.size, bucket.params[0]))
            rows.extend((None, None, param) for param in bucket.params[1:])
        if bucket.opcount_increased_to_capture_external_output > 0:
            extended_buckets.append(
                (
                    idx,
                    bucket.opcount_increased_to_capture_external_output,
                    bucket.size - bucket.paramsize_before_opcount_increase,
                )
            )

    if rows:
        log.info(
            "\nDDPOptimizer used bucket cap %s and created %d buckets. Enable debug logs for detailed bucket info.",
            bucket_bytes_cap,
            len(buckets),
        )

        if extended_buckets:
            log.warning(
                "Some buckets were extended beyond their requested parameter capacities"
                " in order to ensure each subgraph has an output node, required for fx graph partitioning."
                " This can be the case when a subgraph would have only contained nodes performing inplace mutation,"
                " and returning no logical outputs. This should not be a problem, unless it results in too few graph"
                " partitions for optimal DDP performance."
            )

        try:
            from tabulate import tabulate

            log.debug(
                "\nDDPOptimizer produced the following bucket assignments:\n%s",
                tabulate(rows, headers=headers, tablefmt="simple_grid"),
            )

            if extended_buckets:
                log.warning(
                    "DDPOptimizer extended these buckets to ensure per-subgraph output nodes:\n%s",
                    tabulate(
                        extended_buckets,
                        headers=("Index", "Extra Ops", "Extra Param Size (b)"),
                        tablefmt="simple_grid",
                    ),
                )
        except ImportError:
            log.debug(
                "Please `pip install tabulate` in order to display ddp bucket sizes and diagnostic information."
            )
    else:
        log.debug("DDPOptimizer captured no parameters and did not split this graph.")


def has_higher_order_op(gm: fx.GraphModule) -> bool:
    # Check if there is a higher order op in the graph
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            maybe_param = getattr(gm, node.target)
            if isinstance(maybe_param, torch.fx.GraphModule):
                return True
    return False


def propagate_metadata(orig_gm: fx.GraphModule, split_gm: fx.GraphModule) -> None:
    for name, module in split_gm.named_modules():
        if "." not in name and len(name):
            # TODO: add split id to CompileId: https://github.com/pytorch/tlparse/pull/83/files#r1880649384
            module.meta = orig_gm.meta
            module._param_name_to_source = orig_gm._param_name_to_source


def propagate_dynamo_source(orig_gm: fx.GraphModule, split_gm: fx.GraphModule) -> None:
    name_to_dynamo_source = {}
    for node in orig_gm.graph.find_nodes(op="placeholder"):
        name_to_dynamo_source[node.name] = node._dynamo_source

    for name, module in split_gm.named_modules():
        if "." not in name and len(name):
            for node in module.graph.find_nodes(op="placeholder"):
                # non-placeholder in original_gm may become placeholder in submodules
                node._dynamo_source = name_to_dynamo_source.get(node.name)


class DDPOptimizerContext:
    def __init__(self) -> None:
        self.curr_bucket: int = -1
        self.metadata_per_bucket: list[ViewAndMutationMeta] = []


# compile each of the partitioned submodules using the user-provided compiler
class SubmodCompiler(torch.fx.interpreter.Interpreter):
    def __init__(
        self,
        module: fx.GraphModule,
        compiler: CompilerFn,
        fake_mode: torch._subclasses.fake_tensor.FakeTensorMode,
    ) -> None:
        super().__init__(module)
        self.compiler = compiler
        self.fake_mode = fake_mode
        # See Note [DDPOptimizer and fw_metadata]
        ctx = torch._guards.TracingContext.try_get()
        if ctx is not None:
            ctx.ddp_optimizer_ctx = DDPOptimizerContext()

    def compile_submod(
        self, input_mod: fx.GraphModule, args: list[torch.Tensor], kwargs: Any
    ) -> Any:
        """
        Compile the submodule,
        using a wrapper to make sure its output is always a tuple,
        which is required by AotAutograd based compilers
        """
        assert len(kwargs) == 0, "We assume only args for these modules"

        class WrapperModule(torch.nn.Module):
            def __init__(
                self, submod: Callable[..., Any], unwrap_singleton_tuple: bool
            ) -> None:
                super().__init__()
                self.submod = submod
                self.unwrap_singleton_tuple = unwrap_singleton_tuple

            def forward(self, *args: Any) -> Any:
                x = self.submod(*args)
                # TODO(whc)
                # for some reason the isinstance check is necessary if I split one node per submod
                # - even though I supposedly wrapped the output in a tuple in those cases, the real
                # compiled module was still returning a tensor
                if self.unwrap_singleton_tuple and isinstance(x, (tuple, list)):
                    return x[0]
                return x

        unwrap_singleton_tuple = False
        for sn in input_mod.graph.nodes:
            if sn.op == "output":
                if not isinstance(sn.args[0], tuple):
                    unwrap_singleton_tuple = True
                    sn.args = (sn.args,)

        input_mod.recompile()
        input_mod.compile_subgraph_reason = GraphCompileReason(  # type: ignore[assignment]
            "DDPOptimizer intentional graph-break (See Note [DDPOptimizer])."
            " Set `torch._dynamo.config.optimize_ddp = False` to disable.",
            [
                # it's close to useless to get a real stacktrace here, and quite verbose.
                traceback.FrameSummary(__file__, 0, "DDPOptimizer"),
            ],
        )

        wrapper = WrapperModule(
            self.compiler(input_mod, args),
            unwrap_singleton_tuple,
        )
        return wrapper

    # Note:
    #
    # The way distributed works today around fake tensors can be somewhat confusing.
    # Some of these codepaths are shared in both runtime, and compile time. The presence
    # of a fake_mode, read off of fake tensor inputs, dictates how we will operate.
    #
    # A few things to keep in mind:
    #
    # 1) We invoke `compile_submod` with a real module. The output of that gets stored
    # on the graph via `self.module.add_submodule(n.target, compiled_submod_real)`.
    #
    # 2) When running a call_module targeted node, if we have a fake_mode, we fakify the
    # module we got from self.fetch_attr(n.target). Regardless of fake_mode, we then execute it.
    #
    # 3) Fake tensors should always be around during compile time.
    #
    # 4) Fake tensors should never be around at runtime.
    #
    # 5) We end up with a compilation mode that takes a real submodule and fake tensors,
    # to match what aot_autograd expects. See Note: [Fake Modules and AOTAutograd]
    def run_node(self, n: Node) -> Any:
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        new_args = []
        assert self.fake_mode
        for arg in args:
            if isinstance(arg, torch.Tensor) and not isinstance(
                arg, torch._subclasses.FakeTensor
            ):
                new_args.append(torch._dynamo.utils.to_fake_tensor(arg, self.fake_mode))
            else:
                new_args.append(arg)

        log.debug("run_node %s, %s got args %s", n.op, n.target, args_str(args))
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)

        if n.op == "call_module":
            real_mod = self.fetch_attr(str(n.target))
            if self.fake_mode:
                curr_submod = deepcopy_to_fake_tensor(real_mod, self.fake_mode)
            else:
                curr_submod = real_mod

            ddp_graph_log.debug("\n---%s graph---\n%s", n.target, curr_submod.graph)

            # When calling the compiler on the submod, inputs (new_args) are expected to
            # be FakeTensors already since Dynamo would have made them FakeTensors in the
            # non-DDP flow.  However, the parameters are _not_ expected to be FakeTensors,
            # since this wrapping happens during compilation

            # Note: Returning Fake Tensors on First AOT Autograd Call
            #
            # Inductor will optimize strides of outputs when it deems it profitable.
            # For instance, converting to channels last. When we split the graph here
            # into multiple inductor compilations, we need to make sure that the
            # output strides of one compilation is appropriately passed to the subsequent
            # compilations. However, the mapping from inductor output to dynamo output
            # is non-trivial due to aot_autograd's deduping, de-aliasing, mutation, re-writing,
            # subclass handling, etc. In order to replay all this logic we set a flag such that
            # the first invocation of inductor in aot_autograd will return Fake Tensors with
            # appropriate strides. Then, all of aot autograd's runtime logic is replayed.
            # This gives us the appropriately strided outputs here which will reflect runtime strides.

            class FakeifyFirstAOTInvocationGuard:
                def __init__(self) -> None:
                    self.tc = torch._guards.TracingContext.try_get()
                    assert self.tc
                    self.tc.fakify_first_call = True

                def __del__(self) -> None:
                    self.tc.fakify_first_call = False  # type: ignore[union-attr]

            # For aot_eager and other backends, tracing context is not set
            has_tracing_context = torch._guards.TracingContext.try_get() is not None
            if has_tracing_context:
                g = FakeifyFirstAOTInvocationGuard()  # noqa: F841

            from torch._dynamo.utils import counters

            init = counters["aot_autograd"]["total"]
            compiled_submod_real = self.compile_submod(real_mod, new_args, kwargs)

            # TODO - better way of doing this?
            # Only aot autograd handles fakifying first call
            invoked_aot_autograd = init != counters["aot_autograd"]["total"]

            # We update the original (outer) graph with a call into the compiled module
            # instead of the uncompiled one.
            self.module.delete_submodule(n.target)  # type: ignore[operator]
            n.target = "compiled_" + n.target  # type: ignore[operator]
            self.module.add_submodule(n.target, compiled_submod_real)  # type: ignore[operator]

            # Finally, we have to produce inputs for use compiling the next submodule,
            # and these need to be FakeTensors, so we execute the module under fake_mode
            # Because parameters are not fake we patch fake tensor mode to allow non fake inputs
            with (
                self.fake_mode,
                mock.patch.object(self.fake_mode, "allow_non_fake_inputs", True),
            ):
                if has_tracing_context and invoked_aot_autograd:
                    tracing_ctx = torch._guards.TracingContext.try_get()
                    assert tracing_ctx is not None
                    # DDPOptimizer maintains 1 dynamo graph -> N AOT graphs
                    # Dynamo only has 1 tracing context, so it needs to maintain all N AOT metadata instances
                    ddp_ctx = tracing_ctx.ddp_optimizer_ctx
                    assert ddp_ctx is not None
                    assert tracing_ctx.fw_metadata is not None
                    ddp_ctx.curr_bucket += 1
                    ddp_ctx.metadata_per_bucket.append(tracing_ctx.fw_metadata)

                    out = compiled_submod_real(*new_args, **kwargs)
                    # output should be fake or subclass
                    assert all(
                        (not isinstance(t, torch.Tensor) or type(t) is not torch.Tensor)
                        for t in (out if isinstance(out, (list, tuple)) else [out])
                    )
                    return out
                else:
                    return curr_submod(*new_args, **kwargs)
        else:
            # placeholder or output nodes don't need to get compiled, just executed
            return getattr(self, n.op)(n.target, new_args, kwargs)


class DDPOptimizer:
    """Note [DDPOptimizer]
    DDPOptimizer applies when dynamo compiles models wrapped in DistributedDataParallel (DDP),
    breaking the dynamo graph into chunks to compile separately, with the breaks aligning to
    the boundaries of gradient-allreduce buckets chosen by DDP.

    Background/Motivation
     - DDP uses allreduce collectives to synchronize partial gradients computed on different workers
     - DDP groups gradient allreduces into 'buckets' to optimize communication efficiency of all-reduce
     - Parameters grouped into buckets are assumed to be adjacent in time, so they become ready
       at around the same time during backward and thus can share the same allreduce efficiently
     - Allreduces must overlap with backward compute for optimal training performance
     - DDP schedules allreduces using 'hooks' fired from the c++ autograd engine in pytorch, which
       operates when individual grads become 'ready'
     - Dynamo+AOTAutograd produces a single fused graph that runs 'atomically' from the perspective of the
       autograd engine, such that all gradients become 'ready' at the same time.  Hooks fire after the whole
       fused backward function executes, preventing any overlap of compute and communication

    Algorithm
     - DDPOptimizer starts off with an FX graph traced by dynamo which represents forward.  It can traverse
       this graph in reverse order to determine the true order that gradients will become ready during backward.
     - Parameter sizes are counted in reverse order, up to a bucket size limit, at which point a new bucket is started
       and a graph break introduced
     - Each of the subgraphs is compiled by the compiler provided to dynamo by the user, and then fused back together
       into an outer module that is returned to the user

    Notes
     - It would be better to enforce (by adding an API to DDP) that the bucket splits chosen here are used by DDP,
       and that DDP does not need to detect or optimize bucket order by observing execution at runtime, as it does
       in eager.
     - If Dynamo can't capture a whole graph for the portion of the model wrapped by DDP, this algorithm will currently
       produce splits that do not necessarily align with the buckets used by DDP.  This should result in performance
       degradation approaching the baseline case where graph-splits are not used, but not worse.
     - If the backend compiler fails to compile a single subgraph, it will execute eagerly despite the rest of the
       subgraphs being compiled
     - DDP has a 'parameters_and_buffers_to_ignore' field, which DDPOptimizer attempts to honor by reading markers
       left by DDP on individual parameters.  In cases where other transformations, such as reparameterization, are
       also used, the ignore markers could be lost.  If DDPOptimizer fails to ignore a parameter ignored by DDP,
       it is not catastrophic but could impact performance by choosing sub-optimal bucket splits.
     - DDPOptimizer always ignores all buffers, regardless of their ignore flag, since buffers do not require gradients,
       and therefore aren't allreduced by DDP.  (They are broadcast during forward, but this is not covered by
       DDPOptimizer)

    Debugging
     - Generally, it is easiest to debug DDPOptimizer in a single process program, using pdb.
     - In many cases, the log messages are helpful (they show bucket size assignments)-
       just set TORCH_LOGS env to include any of 'dynamo', 'distributed', or 'dist_ddp'.
     - See `benchmarks/dynamo/distributed.py` for a simple harness that will run a toy model or a torchbench model
       in a single process (or with torchrun, in multiple processes)

    Args:
        bucket_bytes_cap (int): Controls the size of buckets, in bytes, used to determine graphbreaks.  Should be
            set to match the equivalent parameter on the original DDP module.

        backend_compile_fn (callable): A dynamo compiler function, to be invoked to compile each subgraph.

        first_bucket_cap (int): Controls the size of the first bucket.  Should match DDP's first bucket cap.  DDP
            special-cases the first bucket size since it is sometimes optimal to start a small allreduce early.

    """

    def __init__(
        self,
        bucket_bytes_cap: int,
        backend_compile_fn: CompilerFn,
        first_bucket_cap: int | None = None,
    ) -> None:
        if first_bucket_cap is not None:
            self.first_bucket_cap = first_bucket_cap
        elif torch.distributed.is_available():
            # this constant comes from C10D lib which is not always built
            self.first_bucket_cap = torch.distributed._DEFAULT_FIRST_BUCKET_BYTES
        else:
            self.first_bucket_cap = bucket_bytes_cap

        self.bucket_bytes_cap = bucket_bytes_cap
        assert self.first_bucket_cap <= self.bucket_bytes_cap, (
            "First bucket should be smaller/equal to other buckets to get comms warmed up ASAP"
        )

        self.backend_compile_fn = backend_compile_fn

    def _ignore_parameter(self, parameter: torch.nn.Parameter) -> bool:
        return hasattr(parameter, "_ddp_ignored") and parameter._ddp_ignored

    def add_param(self, bucket: Bucket, param: torch.nn.Parameter, name: str) -> None:
        bucket.size += param.untyped_storage().nbytes()
        bucket.params.append(name)
        bucket.param_ids.append(id(param))

    def add_module_params_to_bucket(
        self,
        mod: torch.nn.Module,
        bucket: Bucket,
        processed_modules: set[torch.nn.Module],
        prefix: str,
    ) -> None:
        processed_modules.add(mod)
        for name, param in mod.named_parameters():
            if param.requires_grad and not self._ignore_parameter(param):
                self.add_param(bucket, param, f"{prefix}_{name}")

    def add_param_args(self, bucket: Bucket, node: fx.Node) -> None:
        for arg in node.args:
            if not isinstance(arg, torch.fx.node.Node):
                continue
            if arg.op != "placeholder":
                continue
            param = arg.meta["example_value"]
            if (
                isinstance(param, torch.nn.Parameter)
                and param.requires_grad
                and not self._ignore_parameter(param)
            ):
                self.add_param(bucket, param, str(arg.target))

    def compile_fn(
        self, gm: fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> CompiledFn:
        """
        Implements graph splitting, first determining a set of of buckets by counting
        parameter sizes in reverse graph order, then invoking the user/backend compiler
        to compile each subgraph. Finally, stitches compiled graphs into one graphmodule
        and returns its callable.
        """
        # 1: compute the partition map according to DDP bucket logic
        buckets = [Bucket()]  # (size, param_names)
        processed_modules: set[torch.nn.Module] = set()
        for node in reversed(gm.graph.nodes):
            if node.op in ("output", "placeholder"):
                continue

            if (
                buckets[0].size >= self.bucket_bytes_cap
                or len(buckets) == 1
                and buckets[0].size >= self.first_bucket_cap
            ):
                if bucket_has_external_output(buckets[0]):
                    buckets.insert(0, Bucket())
                else:
                    # continue building this bucket past the point of filling its parameter capacity,
                    # to increase chances it contains at least one node that is either a global output or
                    # passed as input to a subsequent graph

                    if buckets[0].opcount_increased_to_capture_external_output == 0:
                        buckets[0].paramsize_before_opcount_increase = buckets[0].size
                    buckets[0].opcount_increased_to_capture_external_output += 1

            if node.op == "call_function":
                self.add_param_args(buckets[0], node)

            elif node.op == "call_module":
                target_mod = gm.get_submodule(node.target)
                if target_mod not in processed_modules:
                    self.add_module_params_to_bucket(
                        target_mod, buckets[0], processed_modules, node.target
                    )
            elif node.op == "call_method":
                if isinstance(node.args[0].target, str):
                    target_mod = None
                    try:
                        target_mod = gm.get_submodule(node.args[0].target)
                    except AttributeError:
                        pass
                    if target_mod is not None and target_mod not in processed_modules:
                        self.add_module_params_to_bucket(
                            target_mod, buckets[0], processed_modules, node.target
                        )
                    # This handles situations like  tmp = torch.mm(x, self.weight.t())
                    # t: "f32[512, 512]" = l_self_seq_2_weight.t();  l_self_seq_2_weight = None
                    # tmp: "f32[512, 512]" = torch.mm(input_2, t);  input_2 = t = None
                    self.add_param_args(buckets[0], node)

            elif node.op == "get_attr":
                maybe_param = getattr(gm, node.target)
                if (
                    isinstance(maybe_param, torch.nn.Parameter)
                    and maybe_param.requires_grad
                    and not self._ignore_parameter(maybe_param)
                ):
                    self.add_param(buckets[0], maybe_param, node.target)

            # All nodes have to be mapped to a bucket, even if they don't have their own params
            # Ignored params still end up in buckets, we just don't count them towards the capacity
            buckets[0].nodes.append(node)

        if len(buckets) > 1 and buckets[0].size == 0:
            # we collected a small preamble graph with ops that don't include parameters, fuse it back
            buckets[1].nodes.extend(buckets[0].nodes)
            assert len(buckets[0].params) == 0, "Params should be empty if size is 0"
            del buckets[0]

        # stash buckets for testing/debugging purposes
        self.buckets = buckets
        pretty_print_buckets(buckets, self.bucket_bytes_cap)

        if len(buckets) == 1:
            # bypass split/fuse logic if there is only one bucket
            return self.backend_compile_fn(gm, example_inputs)

        # 2: partition the graphmodule according to bucket capacity
        partition_map = {}
        for idx, b in enumerate(buckets):
            for node in b.nodes:
                partition_map[node] = idx

        split_gm = fx.passes.split_module.split_module(
            gm,
            None,  # type: ignore[arg-type]
            lambda node: partition_map[node],
        )

        # See note [Assumption on Dynamo Metadata]
        propagate_dynamo_source(gm, split_gm)
        propagate_metadata(gm, split_gm)

        debug_str = (
            f"\n---orig graph---\n{gm.graph}\n"
            + f"\n---split graph---\n{split_gm.graph}\n"
        )
        for name, module in split_gm.named_modules():
            if "." not in name and len(name):
                # only print the submod graphs, not their children
                debug_str += f"\n---{name} graph---\n{module.graph}\n"
        debug_str += "\n---------------\n"
        ddp_graph_log.debug(debug_str)

        trace_structured(
            "optimize_ddp_split_graph",
            payload_fn=lambda: split_gm.print_readable(print_output=False),
        )
        for name, module in split_gm.named_modules():
            if "." not in name and len(name):
                trace_structured(
                    "optimize_ddp_split_child",
                    lambda: {"name": name},
                    payload_fn=lambda: module.print_readable(print_output=False),
                )

        fake_mode = detect_fake_mode(example_inputs)
        if fake_mode is None:
            fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()

        submod_compiler = SubmodCompiler(split_gm, self.backend_compile_fn, fake_mode)
        with torch._dynamo.utils._disable_saved_tensors_hooks_during_tracing():
            submod_compiler.run(*example_inputs)
        split_gm.recompile()

        ddp_graph_log.debug(
            "\n---final graph---\n%s\n---------------\n", split_gm.graph
        )
        return split_gm
