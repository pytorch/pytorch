import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch
import torch.fx.traceback as fx_traceback
from torch import fx
from torch.fx.node import Node

log = logging.getLogger(__name__)


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


def pretty_print_buckets(buckets: List[Bucket]):
    headers = ("Index", "Size (b)", "Param Names")
    rows = []
    for idx, bucket in enumerate(reversed(buckets)):
        if len(bucket.params) > 0:
            rows.append((idx, bucket.size, bucket.params[0]))
            for param in bucket.params[1:]:
                rows.append((None, None, param))
    try:
        from tabulate import tabulate

        log.info(
            "\nDDPOptimizer bucket assignments\n"
            + tabulate(rows, headers=headers, tablefmt="simple_grid")
        )
    except ImportError:
        log.info(
            "Please `pip install tabulate` in order to pretty-print ddp bucket sizes"
        )


class DDPOptimizer:
    """
    DDPOptimizer applies when dynamo compiles models wrapped in DistributedDataParallel (DDP),
    breaking the dynamo graph into chunks to compile separately, with the breaks aligning to
    the boundaries of gradient-allreduce buckets chosen by DDP.

    Background/Motivation
     - DDP uses allreduce collectives to synchronize partial gradients computed on different workers
     - DDP groups gradient allreduces into 'buckets' to optimize communication efficiency of all-reduce
     - Parameters grouped into buckets are assumed to be adjacent in time, so they become ready
       at around the same time during backward and thus can share the same allreduce efficently
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
                buckets.insert(0, Bucket())

            if node.op == "call_module":
                target = gm.get_submodule(node.target)
                for name, p in target.named_parameters():
                    param = target.get_parameter(name)
                    if p.requires_grad and not self._ignore_parameter(param):
                        buckets[0].size += p._storage().nbytes()
                        buckets[0].params.append(f"{node.target}_{name}")
                        buckets[0].param_ids.append(id(param))
            elif node.op == "get_attr":
                maybe_param = getattr(gm, node.target)
                if maybe_param.requires_grad and not self._ignore_parameter(
                    maybe_param
                ):
                    buckets[0].size += maybe_param._storage().nbytes()
                    buckets[0].params.append(node.target)
                    buckets[0].param_ids.append(id(maybe_param))

            # All nodes have to be mapped to a bucket, even if they don't have their own params
            # Ignored params still end up in buckets, we just don't count them towards the capacity
            buckets[0].nodes.append(node)

        # stash buckets for testing/debugging purposes
        self.buckets = buckets
        log.info(
            f"DDPOptimizer used bucket cap {self.bucket_bytes_cap} and produced the following buckets:"
        )
        pretty_print_buckets(buckets)

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
        log.debug(debug_str)

        # 3: compile each of the partitioned submodules using the user-provided compiler
        class SubmodCompiler(torch.fx.interpreter.Interpreter):
            def __init__(self, module, compiler):
                super().__init__(module)
                self.compiler = compiler

            def compile_submod(self, submod, args, kwargs):
                """
                Compile the submodule,
                using a wrapper to make sure its output is always a tuple,
                which is required by AotAutograd based compilers
                """
                assert len(kwargs) == 0, "We assume only args for these modules"

                class WrapperModule(torch.nn.Module):
                    def __init__(self, compiled_submod, unwrap_singleton_tuple):
                        super().__init__()
                        self.compiled_submod = compiled_submod
                        self.unwrap_singleton_tuple = unwrap_singleton_tuple

                    def forward(self, *args):
                        x = self.compiled_submod(*args)
                        # TODO(whc)
                        # for some reason the isinstance check is necessary if I split one node per submod
                        # - even though I supposedly wrapped the output in a tuple in those cases, the real
                        # compiled module was still returning a tensor
                        if self.unwrap_singleton_tuple and isinstance(x, (tuple, list)):
                            return x[0]
                        return x

                unwrap_singleton_tuple = False
                for sn in submod.graph.nodes:
                    if sn.op == "output":
                        if not isinstance(sn.args[0], tuple):
                            unwrap_singleton_tuple = True
                            sn.args = (sn.args,)
                submod.recompile()

                wrapper = WrapperModule(
                    self.compiler(submod, args),
                    unwrap_singleton_tuple,
                )
                return wrapper

            def run_node(self, n: Node) -> Any:
                with fx_traceback.append_stack_trace(n.stack_trace):
                    args, kwargs = self.fetch_args_kwargs_from_env(n)
                    log.debug(f"run_node {n.op}, {n.target} got args {args_str(args)}")
                    assert isinstance(args, tuple)
                    assert isinstance(kwargs, dict)

                    # modify the currently running FX graph
                    # maybe this isn't sound in general, but only changing the target of a node might be ok?
                    if n.op == "call_module":
                        submod = self.fetch_attr(n.target)
                        log.debug(f"\n---{n.target} graph---\n" + str(submod.graph))
                        compiled_submod = self.compile_submod(submod, args, kwargs)
                        self.module.delete_submodule(n.target)
                        n.target = "compiled_" + n.target
                        self.module.add_submodule(n.target, compiled_submod)
                    # then we execute the modified node using the usual logic
                    return getattr(self, n.op)(n.target, args, kwargs)

        submod_compiler = SubmodCompiler(split_gm, self.backend_compile_fn)
        submod_compiler.run(*example_inputs)
        split_gm.recompile()

        log.debug("\n---final graph---\n" + str(split_gm.graph) + "\n---------------\n")

        return split_gm
