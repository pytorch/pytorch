from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch
import torch.fx.traceback as fx_traceback
from torch import fx
from torch.fx.node import Node


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
    param_ids: List = field(default_factory=list)
    nodes: List[fx.Node] = field(default_factory=list)


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

        print(tabulate(rows, headers=headers, tablefmt="simple_grid"))
    except ImportError:
        print("Please `pip install tabulate` in order to pretty-print ddp bucket sizes")


class DDPOptimizer:
    def __init__(
        self,
        bucket_bytes_cap: int,
        parameter_ids_to_ignore: List,
        backend_compile_fn,
        debug=False,
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

        self.parameter_ids_to_ignore = parameter_ids_to_ignore
        self.backend_compile_fn = backend_compile_fn
        self.debug = debug

    def _ignore_parameter(self, parameter):
        return id(parameter) in self.parameter_ids_to_ignore

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
                        buckets[0].size += p.storage().nbytes()
                        buckets[0].params.append(f"{node.target}_{name}")
                        buckets[0].param_ids.append(id(param))
            elif node.op == "get_attr":
                maybe_param = getattr(gm, node.target)
                if maybe_param.requires_grad and not self._ignore_parameter(
                    maybe_param
                ):
                    buckets[0].size += maybe_param.storage().nbytes()
                    buckets[0].params.append(node.target)
                    buckets[0].param_ids.append(id(maybe_param))

            # All nodes have to be mapped to a bucket, even if they don't have their own params
            # Ignored params still end up in buckets, we just don't count them towards the capacity
            buckets[0].nodes.append(node)

        # stash buckets for testing/debugging purposes
        self.buckets = buckets
        if self.debug:
            print(
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
        if self.debug:
            print("---orig graph---")
            print(str(gm.graph))
            print("\n---split graph---")
            print(str(split_gm.graph))
            for name, module in split_gm.named_modules():
                if "." not in name:
                    # only print the submod graphs, not their children
                    print(f"\n---{name} graph---")
                    print(str(module.graph))
            print("---------------")

        # 3: compile each of the partitioned submodules using the user-provided compiler
        class SubmodCompiler(torch.fx.interpreter.Interpreter):
            def __init__(self, module, compiler, debug=False):
                super().__init__(module)
                self.compiler = compiler
                self.debug = debug

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
                    if self.debug:
                        print(f"run_node {n.op}, {n.target} got args {args_str(args)}")
                    assert isinstance(args, tuple)
                    assert isinstance(kwargs, dict)

                    # modify the currently running FX graph
                    # maybe this isn't sound in general, but only changing the target of a node might be ok?
                    if n.op == "call_module":
                        submod = self.fetch_attr(n.target)
                        if self.debug:
                            with open("debug_ddp_optimizer.log", "a") as dump_file:
                                dump_file.write(f"\n---{n.target} graph---")
                                dump_file.write(str(submod.graph))
                        compiled_submod = self.compile_submod(submod, args, kwargs)
                        self.module.delete_submodule(n.target)
                        n.target = "compiled_" + n.target
                        self.module.add_submodule(n.target, compiled_submod)
                    # then we execute the modified node using the usual logic
                    return getattr(self, n.op)(n.target, args, kwargs)

        submod_compiler = SubmodCompiler(split_gm, self.backend_compile_fn, self.debug)
        submod_compiler.run(*example_inputs)
        split_gm.recompile()

        if self.debug:
            print("\n---final graph---")
            print(str(split_gm.graph))
            print("---------------")

        return split_gm
