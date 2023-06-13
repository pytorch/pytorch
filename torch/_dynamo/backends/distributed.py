import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch
from torch import fx
from torch._dynamo.output_graph import GraphCompileReason
from torch._dynamo.utils import deepcopy_to_fake_tensor, detect_fake_mode
from torch.fx.node import Node

log = logging.getLogger(__name__)
ddp_graph_log = torch._logging.getArtifactLogger(__name__, "ddp_graphs")


def args_str(args):
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
    params: List[str] = field(default_factory=list)
    nodes: List[fx.Node] = field(default_factory=list)

    # param_ids is just used for unit testing
    param_ids: List = field(default_factory=list)

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


def pretty_print_buckets(buckets: List[Bucket], bucket_bytes_cap: int):
    headers = ("Index", "Size (b)", "Param Names")
    rows = []
    extended_buckets = []
    for idx, bucket in enumerate(reversed(buckets)):
        if len(bucket.params) > 0:
            rows.append((idx, bucket.size, bucket.params[0]))
            for param in bucket.params[1:]:
                rows.append((None, None, param))
        if bucket.opcount_increased_to_capture_external_output > 0:
            extended_buckets.append(
                (
                    idx,
                    bucket.opcount_increased_to_capture_external_output,
                    bucket.size - bucket.paramsize_before_opcount_increase,
                )
            )

    if len(rows):
        log.info(
            "\nDDPOptimizer used bucket cap %s and created %d buckets. Enable debug logs for detailed bucket info.",
            bucket_bytes_cap,
            len(buckets),
        )

        if len(extended_buckets):
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

            if len(extended_buckets):
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
       just configure torch._dynamo.config.log_level to info or debug.
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
        backend_compile_fn,
        first_bucket_cap: Optional[int] = None,
    ):
        if first_bucket_cap is not None:
            self.first_bucket_cap = first_bucket_cap
        elif torch.distributed.is_available():
            # this constant comes from C10D lib which is not always built
            self.first_bucket_cap = torch.distributed._DEFAULT_FIRST_BUCKET_BYTES
        else:
            self.first_bucket_cap = bucket_bytes_cap

        self.bucket_bytes_cap = bucket_bytes_cap
        assert (
            self.first_bucket_cap <= self.bucket_bytes_cap
        ), "First bucket should be smaller/equal to other buckets to get comms warmed up ASAP"

        self.backend_compile_fn = backend_compile_fn

    def _ignore_parameter(self, parameter):
        return hasattr(parameter, "_ddp_ignored") and parameter._ddp_ignored

    def compile_fn(self, gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
        """
        Implements graph splitting, first determining a set of of buckets by counting
        parameter sizes in reverse graph order, then invoking the user/backend compiler
        to compile each subgraph. Finally, stiches compiled graphs into one graphmodule
        and returns its callable.
        """
        fake_mode = detect_fake_mode(example_inputs)
        if fake_mode is None:
            fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()

        # 1: compute the partition map according to DDP bucket logic
        buckets = [Bucket()]  # (size, param_names)
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

            if node.op == "call_module":
                target = gm.get_submodule(node.target)
                for name, param in target.named_parameters():
                    if param.requires_grad and not self._ignore_parameter(param):
                        buckets[0].size += param.untyped_storage().nbytes()
                        buckets[0].params.append(f"{node.target}_{name}")
                        buckets[0].param_ids.append(id(param))
            elif node.op == "get_attr":
                maybe_param = getattr(gm, node.target)
                if maybe_param.requires_grad and not self._ignore_parameter(
                    maybe_param
                ):
                    buckets[0].size += maybe_param.untyped_storage().nbytes()
                    buckets[0].params.append(node.target)
                    buckets[0].param_ids.append(id(maybe_param))

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
            gm, None, lambda node: partition_map[node]
        )

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

        # 3: compile each of the partitioned submodules using the user-provided compiler
        class SubmodCompiler(torch.fx.interpreter.Interpreter):
            def __init__(self, module, compiler):
                super().__init__(module)
                self.compiler = compiler

            def compile_submod(self, input_mod, args, kwargs):
                """
                Compile the submodule,
                using a wrapper to make sure its output is always a tuple,
                which is required by AotAutograd based compilers
                """
                assert len(kwargs) == 0, "We assume only args for these modules"

                class WrapperModule(torch.nn.Module):
                    def __init__(self, submod, unwrap_singleton_tuple):
                        super().__init__()
                        self.submod = submod
                        self.unwrap_singleton_tuple = unwrap_singleton_tuple

                    def forward(self, *args):
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
                input_mod.compile_subgraph_reason = GraphCompileReason(
                    "DDPOptimizer intentional graph-break (See Note [DDPOptimizer])."
                    " Set `torch._dynamo.config.optimize_ddp = False` to disable.",
                    [
                        # it's close to useless to get a real stacktrace here, and quite verbose.
                        traceback.FrameSummary(__file__, 0, DDPOptimizer),
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
                assert fake_mode
                for arg in args:
                    if isinstance(arg, torch.Tensor) and not isinstance(
                        arg, torch._subclasses.FakeTensor
                    ):
                        new_args.append(fake_mode.from_tensor(arg))
                    else:
                        new_args.append(arg)

                log.debug("run_node %s, %s got args %s", n.op, n.target, args_str(args))
                assert isinstance(args, tuple)
                assert isinstance(kwargs, dict)

                if n.op == "call_module":
                    real_mod = self.fetch_attr(n.target)
                    if fake_mode:
                        curr_submod = deepcopy_to_fake_tensor(real_mod, fake_mode)
                    else:
                        curr_submod = real_mod

                    ddp_graph_log.debug(
                        "\n---%s graph---\n%s", n.target, curr_submod.graph
                    )

                    # When calling the compiler on the submod, inputs (new_args) are expected to
                    # be FakeTensors already since Dynamo would have made them FakeTensors in the
                    # non-DDP flow.  However, the parameters are _not_ expected to be FakeTensors,
                    # since this wrapping happens during compilation
                    compiled_submod_real = self.compile_submod(
                        real_mod, new_args, kwargs
                    )

                    # We update the original (outer) graph with a call into the compiled module
                    # instead of the uncompiled one.
                    self.module.delete_submodule(n.target)
                    n.target = "compiled_" + n.target
                    self.module.add_submodule(n.target, compiled_submod_real)

                    # Finally, we have to produce inputs for use compiling the next submodule,
                    # and these need to be FakeTensors, so we execute the module under fake_mode
                    with fake_mode:
                        return curr_submod(*new_args, **kwargs)
                else:
                    # placeholder or output nodes don't need to get compiled, just executed
                    return getattr(self, n.op)(n.target, new_args, kwargs)

        submod_compiler = SubmodCompiler(split_gm, self.backend_compile_fn)
        submod_compiler.run(*example_inputs)
        split_gm.recompile()

        ddp_graph_log.debug(
            "\n---final graph---\n%s\n---------------\n", split_gm.graph
        )
        return split_gm
