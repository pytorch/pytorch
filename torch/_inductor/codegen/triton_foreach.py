import itertools

from .. import metrics
from ..utils import ceildiv, sympy_product
from ..virtualized import V
from .common import IndentedBuffer, Kernel
from .triton import TritonKernel
from .triton_utils import config_of, signature_to_meta


class ForeachKernel(Kernel):
    MAX_NUM_ARGS = 250  # number where I would no longer get triton errors

    @staticmethod
    def horizontal_partition(nodes):
        """Generates a list of list of nodes where each node sublist is
        guaranteed to not exceed CUDA limits for number of args (read/writes)."""
        assert len(nodes) >= 1

        cur_count = 0
        partitions = []
        cur_partition = []
        for node in nodes:
            read_writes = node.read_writes
            read_write_count = len(read_writes.reads) + len(read_writes.writes)
            if cur_count + read_write_count > ForeachKernel.MAX_NUM_ARGS:
                partitions.append(cur_partition)
                cur_partition = [node]
                cur_count = read_write_count
            else:
                cur_count += read_write_count
                cur_partition.append(node)

        if cur_partition:
            partitions.append(cur_partition)

        return partitions

    def __init__(self):
        super().__init__()
        self.block_size = 1024  # Try tuning this value
        self.num_warps = 8
        self.sub_kernels = []
        self.iter_vars_count = itertools.count()
        self.block_count = 0

    @staticmethod
    def _compute_num_blocks(tensor_elem_counts, block_size):
        num_blocks = 0
        for count in tensor_elem_counts:
            num_blocks += ceildiv(count, block_size)

        return num_blocks

    def codegen_pid_range(self, code, num_elems):
        num_blocks = ceildiv(num_elems, self.block_size)
        upper_bound_pid = self.block_count + num_blocks
        lower_bound_pid = self.block_count
        if self.block_count == 0:
            cond = "if"
        else:
            cond = "elif"
        code.splice(f"{cond} pid >= {lower_bound_pid} and pid < {upper_bound_pid}:")
        with code.indent():
            if self.block_count == 0:
                code.splice("pid_offset = pid")
            else:
                code.splice(f"pid_offset = pid - {lower_bound_pid}")
        self.block_count += num_blocks

    def create_sub_kernel(self, *groups, index_dtype, mutations, reduction_hint):
        sub_kernel = TritonKernel(
            *groups,
            index_dtype=index_dtype,
            mutations=mutations,
            pid_cache={"tl.program_id(0)": "pid_offset"},
            reduction_hint=reduction_hint,
        )
        metrics.generated_kernel_count -= 1
        sub_kernel.args = self.args
        sub_kernel.iter_vars_count = self.iter_vars_count
        sub_kernel.cse.iter_buffer_ids = self.cse.iter_buffer_ids
        self.sub_kernels.append(sub_kernel)
        return sub_kernel

    def jit_line(self):
        _, _, signature = self.args.python_argdefs()
        triton_meta = {
            "signature": signature_to_meta(signature, size_dtype=self.index_dtype),
            "device": V.graph.scheduler.current_device.index,
            "constants": {},
        }
        triton_meta["configs"] = [config_of(signature)]
        return (
            f"@foreach(num_warps={self.num_warps}, meta={triton_meta!r})\n"
            + "@triton.jit"
        )

    def grid(self):
        return (
            self.block_count,
            1,
            1,
        )

    def codegen_kernel(self, name=None):
        code = IndentedBuffer()

        code.splice(
            """
                import triton
                import triton.language as tl
                from torch._inductor.triton_heuristics import foreach
                from torch._inductor.utils import instance_descriptor
                from torch._inductor import triton_helpers
            """
        )
        argdefs, _, _ = self.args.python_argdefs()
        code.writeline(self.jit_line())
        code.writeline(f"def {name or 'KERNEL_NAME'}({', '.join(argdefs)}):")

        with code.indent():
            code.splice("pid = tl.program_id(0)")
            code.splice(f"XBLOCK: tl.constexpr = {self.block_size}")

            for sub_kernel in self.sub_kernels:
                num_elems = int(sympy_product(sub_kernel.numels))
                self.codegen_pid_range(code, num_elems)
                with code.indent():
                    code.splice(f"xnumel = {num_elems}")
                    sub_kernel.codegen_body()
                    code.splice(sub_kernel.body)

            code.splice("else:")
            with code.indent():
                code.splice("pass")

        return code.getvalue()

    def call_kernel(self, code, name: str):
        _, call_args, _ = self.args.python_argdefs()
        # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + ".item()"
        if V.graph.cpp_wrapper:
            V.graph.wrapper_code.generate_kernel_call(
                name, call_args, device_index=V.graph.scheduler.current_device.index
            )
        else:
            # TODO: refactor generate_kernel_call
            call_args_str = ", ".join(call_args)
            stream_name = code.write_get_cuda_stream(
                V.graph.scheduler.current_device.index
            )
            code.writeline(
                f"{name}.run({call_args_str}, grid=({self.grid()}), stream={stream_name})"
            )
