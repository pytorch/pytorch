# mypy: allow-untyped-defs
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu

from ...autotune_process import CUDABenchmarkRequest
from ...ir import (
    Buffer,
    ChoiceCaller,
    CUDATemplateBuffer,
    IRNode,
    Layout,
    PrimitiveInfoType,
    TensorBox,
)
from ...utils import sympy_product
from ...virtualized import V
from ..common import (
    IndentedBuffer,
    Kernel,
    OpOverrides,
    WorkspaceArg,
    WorkspaceZeroMode,
)
from ..cpp_utils import CppPrinter, DTYPE_TO_CPP


if TYPE_CHECKING:
    from torch._inductor.codegen.cuda.cuda_template import CUDATemplate

log = logging.getLogger(__name__)

cexpr = CppPrinter().doprint


def _normalize_idx(index: int, total_length: int) -> int:
    return index if index >= 0 else index + total_length


class CUDAKernel(Kernel):
    """
    Baseclass for CUDA / Cutlass based Kernels
    """

    overrides = OpOverrides  # type: ignore[assignment]


class CUDATemplateKernel(CUDAKernel):
    """
    Template kernels defined by CUDA / Cutlass in C++.
    """

    _EXTRA_CPP_ARGS = "size_t* workspace_size, uint8_t* workspace, cudaStream_t stream"

    def __init__(self, kernel_name) -> None:
        """
        Initializes a new instance of the CUDATemplateKernel class.

        Args:
            kernel_name (str): The name of the kernel.
        """
        super().__init__()
        self.kernel_name = kernel_name
        # Mapping from arg name to IRNode.
        self.named_nodes: Dict[str, IRNode] = {}

    def arg_name(self, node: IRNode) -> Optional[str]:
        """
        Returns arg name of a given input or output node.
        """
        if node is None:
            return None
        return {**self.args.input_buffers, **self.args.output_buffers}.get(
            node.get_name(), None
        )

    def check_not_null(self, node: IRNode) -> str:
        """
        Generates code to check that a node is not null.
        """

        if node is None:
            return ""

        size_str = self.size(node, 0, -1)
        name_str = self.arg_name(node)
        if name_str is None:
            return ""

        res = IndentedBuffer(initial_indent=2)
        res.tabwidth = 1
        res.splice(
            f"""
            {{
              if (!{name_str}) {{
                int64_t {name_str}_size = {size_str};
                if ({name_str}_size > 0) {{
                  throw std::runtime_error("input {name_str} is null but size is not 0!");
                }}
              }}
            }}
            """
        )
        return res.getvalue()

    def get_signature(self) -> str:
        return self.signature

    def def_kernel(
        self,
        inputs: List[IRNode],
        outputs: List[IRNode],
        names_str: str = "",
        input_reorder: Optional[List[int]] = None,
    ) -> str:
        """
        Hook called from template code to generate function definition and
        needed args.

        Args:
            inputs: List of input IRNodes
            outputs: List of output IRNodes
            names_str: Comma separated list of input + output argument names.
            input_reorder: The actual order of input nodes.
                           e.g. The template might have input argument defined as [X, W, Bias],
                           and the actual input passed into this template could be [Bias, X, W].
                           In this case, the `input_reorder` would be [2, 0, 1].
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
        signature = (
            f"int {self.kernel_name}({', '.join(arg_defs)}, {self._EXTRA_CPP_ARGS})"
        )
        self.signature = signature
        return signature

    def call_kernel(
        self,
        name: str,
        node: "CUDATemplateBuffer",  # type: ignore[name-defined]
    ) -> None:
        """
        Generates code to call the kernel through V.graph.wrapper_code.
        used from within torch._inductor.wrapper.PythonWrapperCodegen

        name: Name of kernel function.
        node: The CUDATemplateBuffer node which contains information about the kernel, it's fused epilogue nodes
        as well as all required inputs and outputs.
        """
        wrapper = V.graph.wrapper_code

        if V.graph.cpp_wrapper:
            # Make sure we initialize these kernels since they're exported as
            # C-style symbol names.
            assert isinstance(wrapper, CppWrapperCpu)
            wrapper.initialized_kernels[name] = self
            # We always originally initialize name with "KERNEL_NAME". So, we
            # we replace with the real kernel name passed as an arg to this function.
            self.signature = self.signature.replace("KERNEL_NAME", name)
            _, call_args, arg_types = self.args.cpp_argdefs()
        else:
            _, call_args, _, arg_types = self.args.python_argdefs()
        # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + ".item()"
            else:
                call_args[i] = (
                    call_args[i]
                    if V.graph.cpp_wrapper
                    else f"c_void_p({call_args[i]}.data_ptr())"
                )

        # workspace_size ptr is NULL to mark this call is not intended for retrieving workspace_size.
        # workspace_size should have already been retrieved prior to this call.
        # workspace_size is here.
        call_args.append("nullptr" if V.graph.cpp_wrapper else "None")
        if V.graph.cpp_wrapper:
            arg_types.append("size_t*")

        if node.get_workspace_size() > 0:
            ws = WorkspaceArg(
                count=node.get_workspace_size(),
                device=V.graph.get_current_device_or_throw(),
                zero_mode=WorkspaceZeroMode.UNINITIALIZED,
                outer_name=WorkspaceArg.unique_name(),
            )
            wrapper.generate_workspace_allocation(ws)
            data_ptr = f"{ws.outer_name}.data_ptr()"
            call_args.append(
                data_ptr if V.graph.cpp_wrapper else f"c_void_p({data_ptr})"
            )
        else:
            ws = None
            call_args.append("nullptr" if V.graph.cpp_wrapper else "None")
        if V.graph.cpp_wrapper:
            arg_types.append("uint8_t*")

        wrapper.generate_kernel_call(
            name,
            call_args,
            gpu=True,
            triton=False,
            arg_types=arg_types,
        )
        if ws:
            wrapper.generate_workspace_deallocation(ws)

    def dtype(self, node: IRNode) -> Optional[str]:
        """
        Generates code which represents dtype of a given node.
        """

        if node is None:
            return "void"
        return DTYPE_TO_CPP.get(node.get_layout().dtype)

    def cutlass_dtype(self, node: IRNode, default_dtype="void") -> Optional[str]:
        # Helper method, called into from CUTLASSGemmTemplate
        if node is None:
            return default_dtype
        from torch._inductor.codegen.cuda.cuda_template import CUTLASSTemplate

        return CUTLASSTemplate._DTYPE_TO_CUTLASS[node.get_layout().dtype]

    def max_valid_index(self, node: IRNode, default=-1):
        # Helper method, called into from CUTLASSGemmTemplate
        if node is None:
            return default
        max_valid_offset = 0
        for i in range(len(node.get_size())):
            max_valid_offset += (node.get_size()[i] - 1) * node.get_stride()[i]
        return max_valid_offset

    def offset(self, node: IRNode) -> str:
        """
        Generates code which represents offset of a given node.
        """

        if node is None:
            return "0"
        return str(node.get_layout().offset)  # type: ignore[union-attr]

    def ptr(self, node: IRNode) -> str:
        """
        Generates code which represents pointer of a given node.
        """

        if node is None:
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
        Generates code which represents size of a given node in [start_index, end_index).
        If node is None, returns default_value.

        TODO: Will add needed args to pass it in if it is dynamic.
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
        Generates code which represents stride of a given node at index.
        If node is None, returns default_value.

        TODO: Will add needed args to pass it in if it is dynamic.
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
        This is required by some CUTLASS 2.X APIs.
        If the node is in row_major, it returns stride[-2].
        If the node is in column_major, it returns stride[-1].

        TODO: Will add needed args to pass it in if it is dynamic.
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
    """
    CUDATemplateCaller

    This class represents a caller for CUDA template kernels. It is a subclass of ChoiceCaller.
    Attributes:
        name (str): The name of the caller.
        category (str): The category of the caller.
        bmreq (CUDABenchmarkRequest): The benchmark request for the caller.
        template_buffer (CUDATemplateBuffer): The template buffer for the caller.
    """

    def __init__(
        self,
        name: str,
        category: str,
        input_nodes: List[Buffer],
        layout: Layout,
        make_kernel_render: Callable[[CUDATemplateBuffer, Optional[List[IRNode]]], str],
        bmreq: CUDABenchmarkRequest,
        template: "CUDATemplate",  # type: ignore[name-defined]
        info_kwargs: Optional[Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]],  # type: ignore[type-arg]
        description: str,
    ) -> None:
        super().__init__(name, input_nodes, layout, description)
        self.category = category
        self.make_kernel_render = make_kernel_render
        self.bmreq = bmreq
        self.template = template
        self.info_kwargs = info_kwargs

    def precompile(self) -> None:
        assert self.bmreq is not None
        self.bmreq.precompile()

    def benchmark(self, *args, out) -> float:
        assert self.bmreq is not None
        return self.bmreq.benchmark(
            *args, output_tensor=out
        )  # @TODO: Hack for ensuring that Cutlass Kernel is preferred

    def __str__(self) -> str:
        return f"CUDATemplateCaller(source_file={self.bmreq.source_file})"

    def call_name(self) -> str:
        return f"cuda_template_kernels.{self.name}"

    def hash_key(self) -> str:
        return "-".join(
            [
                self.category,
                self.bmreq.hash_key,
            ]
        )

    def info_dict(self) -> Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
        if self.info_kwargs is not None and "op" in self.info_kwargs:
            op: Any = self.info_kwargs["op"]
            return {
                "backend": "CUDA",
                "op_type": type(op).__name__,
                "op_conf_name": str(op.configuration_name()),
                "op_arch": str(op.arch),
                "tile_shape": str(op.tile_description.tile_shape),
                "epilogue_schedule": str(op.epilogue_schedule),
                "kernel_schedule": str(op.kernel_schedule),
                "element_accumulator": str(op.accumulator_type()),
                "op_name": str(op.procedural_name()),
                "instruction_shape": str(
                    op.tile_description.math_instruction.instruction_shape
                ),
            }
        else:
            return {"backend": "CUDA", "op_type": "unknown"}

    def output_node(self) -> TensorBox:
        self.bmreq.update_workspace_size()
        return TensorBox.create(
            CUDATemplateBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                make_kernel_render=self.make_kernel_render,
                workspace_size=self.bmreq.workspace_size,
                template=self.template,
            )
        )
