from typing import List, Optional

import sympy

from ...autotune_process import CUDABenchmarkRequest
from ...ir import Callable, CUDATemplateBuffer, IRNode, Layout, TensorBox
from ...select_algorithm import ChoiceCaller
from ...utils import sympy_product
from ...virtualized import V

from ..common import IndentedBuffer, Kernel, OpOverrides
from ..cpp import CppPrinter, DTYPE_TO_CPP


cexpr = CppPrinter().doprint


def _normalize_idx(index: int, total_length: int) -> int:
    return index if index >= 0 else index + total_length


class CUDAKernel(Kernel):
    """
    Kernels defined by the CUDA language.
    """

    overrides = OpOverrides


class CUDATemplateKernel(CUDAKernel):
    """
    Template kernels defined by the CUDA language.
    """

    _EXTRA_CPP_ARGS = "size_t* workspace_size, uint8_t* workspace, cudaStream_t stream"

    def __init__(
        self,
        kernel_name,
    ):
        super().__init__()
        self.kernel_name = kernel_name
        self.named_nodes = {}

    def arg_name(self, node: IRNode) -> Optional[str]:
        if node is None:
            return None
        return {**self.args.input_buffers, **self.args.output_buffers}.get(
            node.get_name(), None
        )

    def check_not_null(self, node: IRNode) -> str:
        if node is None:
            return ""

        size_str = self.size(node, 0, -1)
        name_str = self.arg_name(node)
        if name_str is None:
            return ""

        res = IndentedBuffer(initial_indent=2)
        res.tabwidth = 1
        res.splice(
            """
            {{
              if (!{name_str}) {{
                int64_t {name_str}_size = {size_str};
                if ({name_str}_size > 0) {{
                  throw std::runtime_error("input {name_str} is null but size is not 0!");
                }}
              }}
            }}
            """.format(
                name_str=name_str, size_str=size_str
            )
        )
        return res.getvalue()

    def def_kernel(
        self,
        inputs: List[IRNode],
        outputs: List[IRNode],
        names_str: str = "",
        input_reorder: List[int] = None,
    ) -> str:
        """
        Hook called from template code to generate function def and
        needed args.
        """

        names = [x.strip() for x in names_str.strip().split(",")]
        if len(inputs) + len(outputs) != len(names):
            raise RuntimeError(
                f"{len(inputs) + len(outputs)=} != {len(names)=}, {inputs=}, {outputs=}, {names=}"
            )

        if input_reorder is not None:
            assert len(inputs) == len(input_reorder)
        else:
            input_reorder = list(range(len(inputs)))

        for idx in input_reorder:
            name = names[idx]
            node = inputs[idx]
            if node is not None:
                self.named_nodes[name] = node
                self.args.input_buffers[node.get_name()] = name

        for name, node in zip(names[len(inputs) : len(inputs) + len(outputs)], outputs):
            if node is not None:
                self.named_nodes[name] = node
                self.args.output_buffers[node.get_name()] = name

        arg_defs, *_ = self.args.cpp_argdefs()
        return f"PT_EXPORT int {self.kernel_name}({', '.join(arg_defs)}, {self._EXTRA_CPP_ARGS})"

    def call_kernel(self, name: str, node: CUDATemplateBuffer) -> None:
        wrapper = V.graph.wrapper_code
        _, call_args, _ = self.args.python_argdefs()
        # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + ".item()"
            else:
                call_args[i] = f"c_void_p({call_args[i]}.data_ptr())"

        # workspace_size ptr is NULL to mark this call is not intended for retrieving workspace_size.
        # workspace_size should have already been retrieved prior to this call.
        call_args.append("None")

        if node.get_workspace_size() > 0:
            call_args.append(f"c_void_p({node.get_name()}_workspace.data_ptr())")
        else:
            call_args.append("None")

        wrapper.generate_kernel_call(
            name,
            call_args,
            V.graph.scheduler.current_device.index,
            cuda=True,
            triton=False,
        )

    def dtype(self, node: IRNode) -> str:
        if node is None:
            return "void"
        return DTYPE_TO_CPP.get(node.get_layout().dtype)

    def offset(self, node: IRNode) -> str:
        if node is None:
            return "0"
        return str(node.get_layout().offset)

    def ptr(self, node: IRNode, default_node: IRNode = None) -> str:
        if node is None:
            if default_node is not None:
                node = default_node
            else:
                return "nullptr"
        arg_name = self.arg_name(node)
        if arg_name is None:
            return "nullptr"
        offset = self.offset(node)
        return arg_name if offset == "0" else f"{arg_name} + {offset}"

    def size(
        self,
        node: IRNode,
        start_index: int,
        end_index: Optional[int] = None,
        default_value: int = 0,
    ) -> str:
        """
        Hook called from template code to get the size of an arg.
        Will add needed args to pass it in if it is dynamic.
        """

        if node is None:
            return str(default_value)

        start_index = _normalize_idx(start_index, len(node.get_size()))
        if end_index is None:
            end_index = start_index
        end_index = _normalize_idx(end_index, len(node.get_size()))

        sizes = node.get_size()[start_index : end_index + 1]
        if len(sizes) == 0:
            return str(default_value)

        val = sympy_product(sizes)
        return cexpr(self.rename_indexing(val))

    def stride(self, node: IRNode, index: int, default_value: int = 0) -> str:
        """
        Hook called from template code to get the stride of an arg.
        Will add needed args to pass it in if it is dynamic.
        """

        if node is None:
            return str(default_value)

        index = _normalize_idx(index, len(node.get_size()))
        if index < 0:
            return str(default_value)

        stride = node.get_stride()[index]
        return cexpr(self.rename_indexing(stride))

    def row_or_column_stride(self, node: IRNode, default_value: int = 0) -> str:
        """
        Hook called from template code to get the row or column stride of an arg.
        Will add needed args to pass it in if it is dynamic.
        If the node is in row_major, it returns stride[-2].
        If the node is in column_major, it returns stride[-1].
        """

        if node is None or len(node.get_stride()) < 2:
            return str(default_value)

        stride0 = node.get_stride()[-1]
        stride1 = node.get_stride()[-2]
        if stride0 == 1:
            return cexpr(self.rename_indexing(stride1))
        elif stride1 == 1:
            return cexpr(self.rename_indexing(stride0))
        else:
            raise RuntimeError(
                f"At least 1 stride should be 1. Strides: {node.get_stride()=}"
            )


class CUDATemplateCaller(ChoiceCaller):
    def __init__(
        self,
        name: str,
        category: str,
        input_nodes: List[IRNode],
        layout: Layout,
        make_kernel_render: Callable[[str], str],
        bmreq: CUDABenchmarkRequest,
    ):
        super().__init__(name, input_nodes, layout)
        self.category = category
        self.make_kernel_render = make_kernel_render
        self.bmreq = bmreq

    def benchmark(self, *args, out):
        assert self.bmreq is not None
        return self.bmreq.benchmark(*args, output_tensor=out)

    def __str__(self):
        return f"CUDATemplateCaller(source_file={self.bmreq.source_file})"

    def call_name(self) -> str:
        return f"cuda_template_kernels.{self.name}"

    def hash_key(self):
        return "-".join(
            [
                self.category,
                self.bmreq.hash_key,
            ]
        )

    def output_node(self):
        return TensorBox.create(
            CUDATemplateBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                make_kernel_render=self.make_kernel_render,
                workspace_size=self.bmreq.workspace_size,
            )
        )
