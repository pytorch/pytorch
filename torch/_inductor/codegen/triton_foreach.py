import contextlib
import itertools

from .. import config
from ..utils import ceildiv
from ..virtualized import V
from .common import IndentedBuffer, Kernel
from .triton import TritonKernel
from .triton_utils import config_of, signature_of, TritonPrinter

texpr = TritonPrinter().doprint


class ForeachKernel(Kernel):
    @staticmethod
    def horizontal_partition(node_schedule):
        """Creates one or more ForeachKernels if the number of args exceeds CUDA limits."""
        assert len(node_schedule) >= 1

        MAX_NUM_ARGS = 370  # number where I would no longer get triton errors
        cur_count = 0
        kernels = []
        cur_partition = []
        for node in node_schedule:
            read_writes = node.read_writes
            read_write_count = len(read_writes.reads) + len(read_writes.writes)
            if cur_count + read_write_count > MAX_NUM_ARGS:
                kernels.append(ForeachKernel(cur_partition))
                cur_partition = [node]
                cur_count = read_write_count
            else:
                cur_count += read_write_count
                cur_partition.append(node)

        if cur_partition:
            kernels.append(ForeachKernel(cur_partition))

        return kernels

    def __init__(self, nodes):
        super().__init__()
        self.block_size = 1024  # Try tuning this value
        # self.grid = (
        #    ForeachKernel._compute_num_blocks(self.tensor_elem_counts, self.block_size),
        #    1,
        #    1,
        # )
        self.nodes = nodes
        self.num_warps = 8
        self.sub_kernels = []
        self.iter_vars_count = itertools.count()
        for node in nodes:
            _, els = node.group
            sub_kernel = TritonKernel(*els, index_dtype="tl.int32")
            sub_kernel.args = self.args
            sub_kernel.iter_vars_count = self.iter_vars_count
            self.sub_kernels.append(sub_kernel)
        self.kernel = None

    def set_kernel(self, index):
        assert index >= 0 and index < len(self.sub_kernels)
        exitstack = contextlib.ExitStack()

        @contextlib.contextmanager
        def _set_kernel():
            old_kernel = self.kernel
            self.kernel = self.sub_kernels[index]
            try:
                yield
            finally:
                self.kernel = old_kernel

        exitstack.enter_context(_set_kernel())

        return exitstack

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

    def codegen(self):
        self.codegen_sub_kernels()
        self.codegen_kernel()

    def codegen_sub_kernels(self):
        for node, kernel in zip(self.nodes, self.sub_kernels):
            with kernel:
                for sub_node in node.get_nodes():
                    sub_node.mark_run()

                for sub_node in node.get_nodes():
                    index_vars = kernel.split_and_set_ranges(sub_node.get_ranges())
                    sub_node.codegen(index_vars)

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

            for sub_kernel in self.sub_kernels:
                code.splice(sub_kernel.loads)
                code.splice(sub_kernel.compute)
                code.splice(sub_kernel.stores)

            # self._gen_tile_ptrs(code)

            # code.splice(self.loads)
            # code.splice(self.compute)
            # code.splice(self.stores)

        print(code.getvalue())
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
