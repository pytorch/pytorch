# mypy: allow-untyped-defs
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union, Set
from dataclasses import dataclass

from sympy import Symbol, Integer

from torch import dtype as torch_dtype
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
from ..common import IndentedBuffer, Kernel, OpOverrides
from ..cpp_utils import CppPrinter, DTYPE_TO_CPP


if TYPE_CHECKING:
    from torch._inductor.codegen.cuda.cuda_template import CUDATemplate

log = logging.getLogger(__name__)

cexpr = CppPrinter().doprint


def _normalize_idx(index: int, total_length: int) -> int:
    return index if index >= 0 else index + total_length

@dataclass(frozen=True)
class MyKernelArg:
    node: IRNode
    symbol: str   
    size_or_stride: str
    dim: int


class CUDAKernel(Kernel):
    """
    Baseclass for CUDA / Cutlass based Kernels
    """

    overrides = OpOverrides  # type: ignore[assignment]
    myargs: Dict[str, MyKernelArg]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.myargs = {}

    def find_symbol(self, node, size_or_stride, dim):
        def matches(my_kernel_arg):
            return (
                my_kernel_arg.node == node
                and my_kernel_arg.size_or_stride == size_or_stride
                and my_kernel_arg.dim == dim
            )
        matches =  [arg.symbol for arg in self.myargs.values() if matches(arg)]
        assert len(matches) <= 1, matches
        return None if len(matches) == 0 else matches[0]


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
        # TODO: figure out what to do with dynamic shapes.
        return ""
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

        self.size_args()
        size_args = [f"const int32_t {s}" for s in ("M", "K", "N", "lda", "ldb", "ldc", "ldd")]

        signature = f"int {self.kernel_name}({', '.join(arg_defs + size_args)}, {self._EXTRA_CPP_ARGS})"
        self.signature = signature
        return signature
    
    def size_args(self):
        X = self.named_nodes['X']
        W = self.named_nodes['W']
        Bias = self.named_nodes.get('Bias', None)
        Y = self.named_nodes['Y']

        mdim = _normalize_idx(-2, len(X.get_size()))        
        kdim = _normalize_idx(-1, len(X.get_size()))        
        ndim = _normalize_idx(-1, len(W.get_size()))        

        M = X.get_size()[mdim]
        K = X.get_size()[kdim]
        N = W.get_size()[ndim]

        self.myargs.setdefault('M', MyKernelArg(node=X, symbol='M', size_or_stride='size', dim=mdim))
        self.myargs.setdefault('K', MyKernelArg(node=W, symbol='K', size_or_stride='size', dim=kdim))
        self.myargs.setdefault('N', MyKernelArg(node=Y, symbol='N', size_or_stride='size', dim=ndim))

        def find_ld(node):
            if node is None:
                return 0
            dim = find_ld_idx(node)
            return node.get_stride()[dim]

        def find_ld_idx(node):
            strides = node.get_stride()   # ignore batch dim if present
            if V.graph.sizevars.statically_known_equals(strides[-1], 1):
                return _normalize_idx(-2, len(strides))

            assert V.graph.sizevars.statically_known_equals(strides[-2], 1), strides[-2]
            return _normalize_idx(-1, len(strides))
        
        lda_dim = find_ld_idx(X)
        ldb_dim = find_ld_idx(W)
        ldc_dim = find_ld_idx(Y)
        ldd_dim = find_ld_idx(Bias) if Bias else None

        self.myargs.setdefault('lda', MyKernelArg(node=X, symbol='lda', size_or_stride='stride', dim=lda_dim))
        self.myargs.setdefault('ldb', MyKernelArg(node=W, symbol='ldb', size_or_stride='stride', dim=ldb_dim))
        self.myargs.setdefault('ldc', MyKernelArg(node=Y, symbol='ldc', size_or_stride='stride', dim=ldc_dim))
        if Bias:
            self.myargs.setdefault('ldd', MyKernelArg(node=Bias, symbol='ldd', size_or_stride='stride', dim=ldd_dim))

        LDA = find_ld(X)
        LDB = find_ld(W)
        LDC = find_ld(Y)
        LDD = find_ld(Bias)
        return M, K, N, LDA, LDB, LDC, LDD

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

        call_args.extend(self.size_args())
        arg_types.extend(type(a) for a in self.size_args())
        # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + ".item()"
            else:
                if isinstance(arg_types[i], torch_dtype):
                    call_args[i] = (
                        call_args[i]
                        if V.graph.cpp_wrapper
                        else f"c_void_p({call_args[i]}.data_ptr())"
                    )
                elif issubclass(arg_types[i], Symbol):
                    call_args[i] = call_args[i]

        # workspace_size ptr is NULL to mark this call is not intended for retrieving workspace_size.
        # workspace_size should have already been retrieved prior to this call.
        # workspace_size is here.
        call_args.append("nullptr" if V.graph.cpp_wrapper else "None")
        if V.graph.cpp_wrapper:
            arg_types.append("size_t*")

        if node.get_workspace_size() > 0:
            wrapper.generate_workspace_allocation(
                node.get_workspace_size(), V.graph.scheduler.current_device, False
            )
            data_ptr = "workspace.data_ptr()"
            call_args.append(
                data_ptr if V.graph.cpp_wrapper else f"c_void_p({data_ptr})"
            )
        else:
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
        if node.get_workspace_size() > 0:
            wrapper.writeline(wrapper.make_free_by_names(["workspace"]))

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
        return str(node.get_layout().offset)

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

        sizes = [
            self.find_symbol(node, "size", i) or node.get_size()[i]
            for i in range(start_index, end_index + 1)
        ]

        val = sympy_product(sizes)

        return val


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

        if V.graph.sizevars.statically_known_leq(stride, 1):
            return str(stride)
        
        symbol = self.find_symbol(node, "stride", index)
        return symbol

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
