import itertools
from typing import List

import sympy

from .. import config
from ..ir import Layout
from ..utils import ceildiv, sympy_product
from ..virtualized import V
from .common import IndentedBuffer, Kernel
from .triton_overrides import TritonOverrides
from .triton_utils import (
    config_of,
    IterationRangesRoot,
    signature_of,
    split_iteration_ranges,
    TritonCSEVariable,
    TritonPrinter,
)

texpr = TritonPrinter().doprint


class ListTracker:
    def __init__(self, var, arg_names, layouts):
        self.var: TritonCSEVariable = var
        self.arg_names: List[str] = arg_names
        self.layouts: List[Layout] = layouts
        self.indexers = [layout.make_indexer() for layout in layouts]
        self.range_trees: List[IterationRangesRoot] = []
        self.index_vars: List[sympy.symbol] = []
        name = "xindex"
        for layout in layouts:
            tree = IterationRangesRoot(
                "xindex", sympy_product(layout.size), name[0], 0, V.kernel
            )
            self.range_trees.append(tree)
            self.index_vars.append(tree.construct(layout.size))

    def codegen_tile_ptrs(self, code: IndentedBuffer, list_index: int):
        index_expr = self.index_expr(list_index)

        nodes = [
            V.kernel.range_tree_nodes[v]
            for v in sorted(index_expr.free_symbols, key=lambda s: s.name)
        ]
        for node in nodes:
            node.codegen_into(code)

        code.splice(
            f"{self.var}_tile_ptrs = {self.arg_names[list_index]} + ({index_expr})"
        )

    def index_expr(self, list_index: int):
        expr = self.indexers[list_index](self.index_vars[list_index])
        return V.graph.sizevars.simplify_with_ranges(
            expr, self.range_trees[list_index].var_ranges
        )


class ForeachKernel(Kernel):
    overrides = TritonOverrides

    @staticmethod
    def partition_schedule(node_schedule):
        """Creates one or more ForeachKernels if the number of args exceeds CUDA limits."""
        assert len(node_schedule) >= 1

        MAX_NUM_ARGS = 370  # number where I would no longer get triton errors
        partitions = []
        cur_count = 0
        cur_partition = []
        for node in node_schedule:
            read_writes = node.read_writes
            read_write_count = len(read_writes.reads) + len(read_writes.writes)
            if cur_count + read_write_count > MAX_NUM_ARGS:
                partitions.append(cur_partition)
                cur_partition = [node]
                cur_count = read_write_count
            else:
                cur_count += read_write_count
                cur_partition.append(node)

        if cur_partition:
            partitions.append(cur_partition)

        return partitions

    def __init__(self, num_sub_kernels):
        super().__init__()
        self.block_size = 1024  # Try tuning this value
        # self.grid = (
        #    ForeachKernel._compute_num_blocks(self.tensor_elem_counts, self.block_size),
        #    1,
        #    1,
        # )
        self.num_warps = 8
        self.load_buffers = [IndentedBuffer() for _ in range(num_sub_kernels)]
        self.compute_buffers = [IndentedBuffer() for _ in range(num_sub_kernels)]
        self.store_buffers = [IndentedBuffer() for _ in range(num_sub_kernels)]
        self.range_trees = []

    def set_index(self, index):
        assert index >= 0 and index < len(self.load_buffers)
        return self.swap_buffers(
            self.load_buffers[index],
            self.compute_buffers[index],
            self.store_buffers[index],
        )

    def split_and_set_ranges(self, lengths: List[List[sympy.Expr]]):
        """
        We may want to fuse `for i0 in s0*s1` into a tiled kernel with groups (s0, s1).

        To do this we need to split up the iteration space of i0 into something like:
            for i1 in s0:
              for i2 in s1:
                i0 = i1*s1 + i2
                ....

        This function matches and resplits lengths to the groups of
        this kernel to enable tiled + non-tiled fusions.
        """
        groups = [rt.numel for rt in self.range_trees]
        if not self.inside_reduction:
            groups[-1] = sympy.Integer(1)

        if len(lengths) == len(self.range_trees) and all(
            V.graph.sizevars.simplify(sympy_product(x) - g) == 0
            for x, g in zip(lengths, groups)
        ):
            return self.set_ranges(*lengths)

        new_ranges, return_getters_groups = split_iteration_ranges(groups, lengths)
        itervars = list(itertools.chain(*self.set_ranges(*new_ranges)))
        return [[fn(itervars) for fn in fns] for fns in return_getters_groups]

    def split_and_set_ranges(self, lengths: List[List[sympy.Expr]]):
        """
        We may want to fuse `for i0 in s0*s1` into a tiled kernel with groups (s0, s1).

        To do this we need to split up the iteration space of i0 into something like:
            for i1 in s0:
              for i2 in s1:
                i0 = i1*s1 + i2
                ....

        This function matches and resplits lengths to the groups of
        this kernel to enable tiled + non-tiled fusions.
        """
        groups = [rt.numel for rt in self.range_trees]
        if not self.inside_reduction:
            groups[-1] = sympy.Integer(1)

        if len(lengths) == len(self.range_trees) and all(
            V.graph.sizevars.simplify(sympy_product(x) - g) == 0
            for x, g in zip(lengths, groups)
        ):
            return self.set_ranges(*lengths)

        new_ranges, return_getters_groups = split_iteration_ranges(groups, lengths)
        itervars = list(itertools.chain(*self.set_ranges(*new_ranges)))
        return [[fn(itervars) for fn in fns] for fns in return_getters_groups]

    @staticmethod
    def _compute_num_blocks(tensor_elem_counts, block_size):
        num_blocks = 0
        for count in tensor_elem_counts:
            num_blocks += ceildiv(count, block_size)

        return num_blocks

    def _gen_tile_ptrs(self, code):
        block_count = 0
        for index, num_elems in enumerate(self.tensor_elem_counts):
            num_blocks = ceildiv(num_elems, self.block_size)
            upper_bound_pid = block_count + num_blocks
            lower_bound_pid = block_count
            last_block_elem_count = self.block_size - (
                num_blocks * self.block_size - num_elems
            )

            if block_count == 0:
                cond = "if"
                # initialize tile ptrs
                code.splice("xmask = tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE\n")
                for list_tracker in self.lists.values():
                    code.splice(
                        f"{list_tracker.var}_tile_ptrs = {list_tracker.arg_names[index]} + tl.arange(0, BLOCK_SIZE)"
                    )
            else:
                cond = "elif"

            code.splice(f"{cond} pid >= {lower_bound_pid} and pid < {upper_bound_pid}:")
            with code.indent():
                code.splice(f"xoffset = (pid - {lower_bound_pid}) * BLOCK_SIZE")
                code.splice("xindex = xoffset + tl.arange(0, BLOCK_SIZE)")
                for list_tracker in self.lists.values():
                    list_tracker.codegen_tile_ptrs(code, index)

                code.splice(f"if pid == {upper_bound_pid - 1}:")
                with code.indent():
                    code.splice(
                        f"xmask = tl.arange(0, BLOCK_SIZE) < {last_block_elem_count}"
                    )

            block_count += num_blocks

    def jit_line(self):
        _, _, signature = self.args.python_argdefs()
        triton_meta = {
            "signature": dict(enumerate(map(signature_of, signature))),
            "device": V.graph.scheduler.current_device.index,
            "constants": {},
        }
        triton_meta["configs"] = [config_of(signature)]
        return (
            f"@template(num_stages=1, num_warps={self.num_warps}, meta={triton_meta!r})\n"
            + "@triton.jit"
        )

    def _list_tracker(self, list_name, var, arg_names, layouts):
        self.lists[list_name] = ListTracker(var, arg_names, layouts)
        return self.lists[list_name]

    def get_list(self, list_name):
        return V.graph.lists[list_name][
            self.sublist_indices[0] : self.sublist_indices[1]
        ]

    # TODO: handle the case where we see the same list with different layouts
    def load_list(self, list_name: str, layouts: List[Layout]):
        if list_name not in self.lists:
            var = self.cse.newvar()
            arg_names = []

            for buffer_name in self.get_list(list_name):
                arg_names.append(self.args.input(buffer_name))

            self._list_tracker(list_name, var, arg_names, layouts)
            self.loads.writeline(f"{var} = tl.load({var}_tile_ptrs, mask=xmask)")

        return self.lists[list_name].var

    def store_list(self, list_name: str, layouts: List[Layout], value):
        if list_name not in self.lists:
            var = self.cse.newvar()

            arg_names = []
            for buffer_name in self.get_list(list_name):
                arg_names.append(self.args.output(buffer_name))

            self._list_tracker(list_name, var, arg_names, layouts)
            self.stores.writeline(f"tl.store({var}_tile_ptrs, {value}, mask=xmask)")

    def codegen_kernel(self, name=None):
        # from triton import next_power_of_2

        code = IndentedBuffer()

        code.splice(
            """
                import triton
                import triton.language as tl
                from torch._inductor.triton_heuristics import template
                from torch._inductor.utils import instance_descriptor
            """
        )
        argdefs, _, _ = self.args.python_argdefs()
        code.writeline(self.jit_line())
        code.writeline(f"def {name or 'KERNEL_NAME'}({', '.join(argdefs)}):")
        if config.benchmark_kernel:
            code.splice(
                """
                    from torch._dynamo.testing import rand_strided
                    from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
                    import torch
                    from torch._inductor.triton_heuristics import grid, template
                    from torch._inductor.utils import instance_descriptor
                """
            )

        with code.indent():
            code.splice("pid = tl.program_id(0)")
            code.splice(f"BLOCK_SIZE: tl.constexpr = {self.block_size}")

            self._gen_tile_ptrs(code)

            code.splice(self.loads)
            code.splice(self.compute)
            code.splice(self.stores)

        return code.getvalue()

    def call_kernel(self, code, name: str):
        _, call_args, _ = self.args.python_argdefs()
        # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + ".item()"
        if V.graph.cpp_wrapper:
            V.graph.wrapper_code.generate_kernel_call(
                name, call_args, V.graph.scheduler.current_device.index
            )
        else:
            # TODO: refactor generate_kernel_call
            call_args_str = ", ".join(call_args)
            stream_name = code.write_get_cuda_stream(
                V.graph.scheduler.current_device.index
            )
            code.writeline(
                f"{name}.run({call_args_str}, grid=({self.grid}), stream={stream_name})"
            )
