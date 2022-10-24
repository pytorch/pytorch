from typing import Any, List

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


class DDPOptimizer:
    def __init__(
        self,
        bucket_bytes_cap: int,
        parameters_to_ignore: List[str],
        backend_compile_fn,
        debug=False,
    ):
        self.bucket_bytes_cap = bucket_bytes_cap
        self.parameters_to_ignore = parameters_to_ignore
        self.backend_compile_fn = backend_compile_fn
        self.debug = debug

    def compile_fn(self, gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
        """
        TODO:
        - handle params_and_buffers_to_ignore
        - handle kwargs
        """

        # 1: compute the partition map according to DDP bucket logic
        bucket_bytes = 0
        bucket_actual_sizes = []
        node_splits = [[]]
        for node in reversed(gm.graph.nodes):
            if node.op == "output" or node.op == "placeholder":
                continue

            if bucket_bytes >= self.bucket_bytes_cap:
                bucket_actual_sizes.insert(0, bucket_bytes)
                bucket_bytes = 0
                node_splits.insert(0, [])

            elif node.op == "call_module":
                target = gm.get_submodule(node.target)
                params_size_b = sum(
                    [
                        p.storage().nbytes()
                        for p in target.parameters()
                        if p.requires_grad
                    ]
                )
                bucket_bytes += params_size_b
                # print(f"accumulated {params_size_b} b from {node}")
            elif node.op == "get_attr":
                maybe_param = getattr(gm, node.target)
                if maybe_param.requires_grad:
                    bucket_bytes += maybe_param.storage().nbytes()
            else:
                # TODO(whc) confirm this:
                # (e.g. call_method, call_function aren't expected to 'have' parameters)
                pass

            node_splits[0].append(node)

        if len(node_splits) == 1:
            if self.debug:
                print(
                    "DDPOptimizer did not split graphs."
                    f" Accumulated {bucket_bytes} bytes, and bucket cap is {self.bucket_bytes_cap}"
                )
            return self.backend_compile_fn(gm, example_inputs)

        if len(bucket_actual_sizes) < len(node_splits):
            bucket_actual_sizes.insert(0, bucket_bytes)

        if self.debug:
            print(
                f"DDPOptimizer used bucket cap {self.bucket_bytes_cap}"
                f" and split graphs into parameter sizes {', '.join([str(b) for b in bucket_actual_sizes])}"
            )

        # 2: partition the graphmodule according to bucket capacity
        partition_map = {}
        for p, nodes in enumerate(node_splits):
            for node in nodes:
                partition_map[node] = p

        split_gm = fx.passes.split_module.split_module(
            gm, None, lambda node: partition_map[node]
        )
        if self.debug:
            with open("debug_ddp_optimizer.log", "w") as dump_file:
                dump_file.write("---orig graph---")
                dump_file.write(str(gm.graph))
                dump_file.write("\n---split graph---")
                dump_file.write(str(split_gm.graph))

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
            with open("debug_ddp_optimizer.log", "a") as dump_file:
                dump_file.write("\n---final graph---")
                dump_file.write(str(split_gm.graph))

        return split_gm
