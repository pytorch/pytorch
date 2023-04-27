import collections

from .. import config
from ..utils import ceildiv, sympy_product
from ..virtualized import V

from .common import IndentedBuffer, Kernel

from .triton_overrides import TritonOverrides

from .triton_utils import config_of, signature_of


class ForeachKernel(Kernel):
    overrides = TritonOverrides

    @staticmethod
    def create_kernels(foreach_node):
        """Creates one or more ForeachKernels if the number of args exceeds CUDA limits."""
        node_schedule = foreach_node.get_nodes()
        assert len(node_schedule) >= 1
        layouts = node_schedule[0].node.get_layouts()
        list_length = len(layouts)
        num_lists = sum([node.node.list_arg_count() for node in node_schedule])

        MAX_NUM_ARGS = 370  # number where I would no longer get triton errors
        kernels = []
        max_list_length = int(MAX_NUM_ARGS / num_lists)
        for i in range(0, list_length, max_list_length):
            kernels.append(
                ForeachKernel(
                    layouts, sublist_indices=(i, min(i + max_list_length, list_length))
                )
            )
        return kernels

    def __init__(self, layouts, sublist_indices=None):
        sublist_indices = sublist_indices if sublist_indices else (0, len(layouts))
        if sublist_indices[0] > sublist_indices[1]:
            raise ValueError(
                f"Invalid list slice bounds in ForeachKernel: {sublist_indices}"
            )
        super().__init__()
        self.sublist_indices = sublist_indices
        self.list_name_to_var_name = {}
        self.list_name_to_arg_names = collections.defaultdict(list)
        self.tensor_elem_counts = [
            int(sympy_product(layout.size))
            for layout in layouts[sublist_indices[0] : sublist_indices[1]]
        ]
        self.block_size = 1024
        self.grid = (
            ForeachKernel._compute_num_blocks(self.tensor_elem_counts, self.block_size),
            1,
            1,
        )
        self.num_warps = 8

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
                for list_name, arg_names in self.list_name_to_arg_names.items():
                    var = self.list_name_to_var_name[list_name]
                    code.splice(f"{var}_tile_ptr = {arg_names[index]}")
            else:
                cond = "elif"

            code.splice(f"{cond} pid >= {lower_bound_pid} and pid < {upper_bound_pid}:")
            with code.indent():
                code.splice(f"elem_ind_offset = (pid - {lower_bound_pid}) * BLOCK_SIZE")
                for list_name, arg_names in self.list_name_to_arg_names.items():
                    var = self.list_name_to_var_name[list_name]
                    code.splice(
                        f"{var}_tile_ptr = {arg_names[index]} + elem_ind_offset"
                    )
                code.splice(f"if pid == {upper_bound_pid - 1}:")
                with code.indent():
                    code.splice(f"elem_count = {last_block_elem_count}")

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

    def get_list(self, list_name):
        return V.graph.lists[list_name][
            self.sublist_indices[0] : self.sublist_indices[1]
        ]

    def load(self, list_name: str, _):
        if list_name not in self.list_name_to_var_name:
            var = self.cse.newvar()
            self.list_name_to_var_name[list_name] = var

            for buffer_name in self.get_list(list_name):
                arg_name = self.args.input(buffer_name)
                self.list_name_to_arg_names[list_name].append(arg_name)

            self.loads.writeline(
                f"{var} = tl.load({var}_tile_ptr + tl.arange(0, BLOCK_SIZE), mask=mask)"
            )

        return self.list_name_to_var_name[list_name]

    def store(self, list_name: str, _, value, mode=None):
        if list_name not in self.list_name_to_var_name:
            var = self.cse.newvar()
            self.list_name_to_var_name[list_name] = var

            for buffer_name in self.get_list(list_name):
                arg_name = self.args.output(buffer_name)
                self.list_name_to_arg_names[list_name].append(arg_name)

            self.stores.writeline(
                f"tl.store({var}_tile_ptr + tl.arange(0, BLOCK_SIZE), {value}, mask=mask)"
            )

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
            code.splice(f"elem_count = {self.block_size}")

            self._gen_tile_ptrs(code)

            code.splice("mask = tl.arange(0, BLOCK_SIZE) < elem_count\n")
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
