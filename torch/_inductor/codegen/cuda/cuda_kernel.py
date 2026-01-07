# mypy: allow-untyped-defs
import functools
import itertools
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Optional, TYPE_CHECKING, Union

from sympy import Expr, symbols

import torch._inductor.config as config
from torch import dtype as torch_dtype
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import do_bench_using_profiling, OrderedSet, Placeholder
from torch.utils._sympy.value_ranges import ValueRanges

from .cutlass_utils import DTYPE_TO_CUTLASS_TYPE


if TYPE_CHECKING:
    from .cuda_template import ArgInfo

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
    CSEVariable,
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


ValidLayoutSymbols = Literal["M", "N", "K", "B", "lda", "ldb", "ldc", "ldd"]
ValidLayoutAttrs = Literal["size", "stride"]


@dataclass(frozen=True)
class LayoutArg:
    node: IRNode
    symbol: ValidLayoutSymbols
    attr: ValidLayoutAttrs
    dim: int

    def matches(self, node, attr, dim) -> bool:
        return self.node == node and self.attr == attr and self.dim == dim


class CUDAKernel(Kernel):
    """
    Baseclass for CUDA / Cutlass based Kernels
    """

    overrides = OpOverrides  # type: ignore[assignment]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layout_args: dict[str, list[LayoutArg]] = defaultdict(list)
        self.size_args: list[Union[Expr, int]] = []
        # Mapping from arg name to IRNode.
        self.named_nodes: dict[str, IRNode] = {}

    def find_symbol(
        self, node: IRNode, attr: ValidLayoutAttrs, dim: int
    ) -> Optional[str]:
        arg = self.find_layout_arg(node, attr, dim)
        return arg.symbol if arg else None

    def find_layout_arg(
        self, node: IRNode, attr: ValidLayoutAttrs, dim: int
    ) -> Optional[LayoutArg]:
        matches = [
            arg
            for arg in itertools.chain.from_iterable(self.layout_args.values())
            if arg.matches(node, attr, dim)
        ]
        if len(matches) >= 1:
            # Verify all matches have the same node, attribute, and dimension
            # And if they come from the same node, whichever symbol we use is fine.
            # if in runtime the logic changes, this would trigger guard
            first_match = matches[0]
            if not all(
                match.node == first_match.node
                and match.attr == first_match.attr
                and match.dim == first_match.dim
                for match in matches
            ):
                raise AssertionError("All matching layout args should be identical")
            return first_match
        return None

    def add_layout_arg(
        self, symbol: ValidLayoutSymbols, node: IRNode, attr: ValidLayoutAttrs, dim: int
    ):
        arg = LayoutArg(node, symbol, attr, dim)
        self.layout_args[symbol].append(arg)

    def init_layout_args(self) -> None:
        X = self.named_nodes["X"]
        W = self.named_nodes["W"]
        Y = self.named_nodes["Y"]
        Bias = self.named_nodes.get("Bias", None)
        x_mdim = _normalize_idx(-2, len(X.get_size()))
        x_kdim = _normalize_idx(-1, len(X.get_size()))
        w_kdim = _normalize_idx(-2, len(W.get_size()))
        w_ndim = _normalize_idx(-1, len(W.get_size()))
        y_mdim = _normalize_idx(-2, len(Y.get_size()))
        y_ndim = _normalize_idx(-1, len(Y.get_size()))
        self.add_layout_arg("M", X, "size", x_mdim)
        self.add_layout_arg("K", X, "size", x_kdim)
        self.add_layout_arg("K", W, "size", w_kdim)
        self.add_layout_arg("N", W, "size", w_ndim)
        self.add_layout_arg("M", Y, "size", y_mdim)
        self.add_layout_arg("N", Y, "size", y_ndim)
        if len(X.get_size()) > 2:
            self.add_layout_arg("B", X, "size", 0)

        lda_dim = self.find_ld_idx(X)
        ldb_dim = self.find_ld_idx(W)
        ldc_dim = self.find_ld_idx(Bias) if Bias else None
        ldd_dim = self.find_ld_idx(Y)
        self.add_layout_arg("lda", X, "stride", lda_dim)
        self.add_layout_arg("ldb", W, "stride", ldb_dim)
        if Bias is not None and ldc_dim is not None:
            self.add_layout_arg("ldc", Bias, "stride", ldc_dim)
        self.add_layout_arg("ldd", Y, "stride", ldd_dim)

    def get_layout_args(self) -> tuple[Union[Expr, int], ...]:
        X = self.named_nodes["X"]
        W = self.named_nodes["W"]
        Y = self.named_nodes["Y"]
        Bias = self.named_nodes.get("Bias", None)
        mdim = _normalize_idx(-2, len(X.get_size()))
        ndim = _normalize_idx(-1, len(W.get_size()))
        kdim = _normalize_idx(-1, len(X.get_size()))

        def get_ld(node) -> Union[Expr, int]:
            dim = self.find_ld_idx(node)
            return node.get_stride()[dim]

        M = X.get_size()[mdim]
        N = W.get_size()[ndim]
        K = X.get_size()[kdim]
        B = X.get_size()[0] if len(X.get_size()) > 2 else 1
        LDA = get_ld(X)
        LDB = get_ld(W)
        LDC = get_ld(Bias) if Bias else 0
        LDD = get_ld(Y)
        return (M, N, K, B, LDA, LDB, LDC, LDD)

    def get_dynamic_shape_args(self) -> list[Union[Expr, int]]:
        return [*self.get_layout_args(), *self.size_args]

    def get_offset_args(self) -> list[Expr]:
        return [node.get_layout().offset for node in self.named_nodes.values()]

    @staticmethod
    def find_ld_idx(node: IRNode) -> int:
        strides = node.get_stride()
        # Handle 1D tensor case
        if V.graph.sizevars.statically_known_equals(strides[-1], 1):
            return _normalize_idx(-2, len(strides))

        assert V.graph.sizevars.statically_known_equals(strides[-2], 1), strides[-2]
        return _normalize_idx(-1, len(strides))


class CUDATemplateKernel(CUDAKernel):
    """
    Template kernels defined by CUDA / Cutlass in C++.
    """

    _EXTRA_CPP_ARGS = "size_t* workspace_size, uint8_t* workspace, cudaStream_t stream"

    def __init__(
        self,
        kernel_name: str,
        runtime_arg_info: list["ArgInfo"],
        runtime_arg_values: list[Any],
    ) -> None:
        """
        Initializes a new instance of the CUDATemplateKernel class.

        Args:
            kernel_name (str): The name of the kernel.
        """
        super().__init__()
        self.kernel_name = kernel_name
        self.runtime_arg_info = runtime_arg_info
        self.runtime_arg_values = runtime_arg_values

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
        inputs: list[IRNode],
        outputs: list[IRNode],
        names_str: str = "",
        input_reorder: Optional[list[int]] = None,
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
            additional_size_args: Additional size arguments for epilogue inputs
        """
        # NB: name order matters here, it's used to match up offsets
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

        free_symbols: OrderedSet[Expr] = OrderedSet()
        for name, node in zip(names[len(inputs) : len(inputs) + len(outputs)], outputs):
            if node is not None:
                # NB: named nodes must be populated in the order of names
                self.named_nodes[name] = node
                self.args.output_buffers[node.get_name()] = name

                if name not in (
                    "X",
                    "W",
                    "Bias",
                    "Y",
                ):  # we handle these symbolic shapes explicitly
                    for expr in itertools.chain(node.get_size(), node.get_stride()):
                        if isinstance(expr, Expr):
                            for s in expr.free_symbols:
                                free_symbols.add(s)  # type: ignore[arg-type]

        arg_defs, *_ = self.args.cpp_argdefs(DTYPE_TO_CUTLASS_TYPE)

        self.init_layout_args()
        size_vars = ["M", "N", "K", "B", "lda", "ldb", "ldc", "ldd"]
        size_vars.extend(str(s) for s in free_symbols)
        self.size_args.extend(free_symbols)
        size_args = [f"const int {s}" for s in size_vars]
        offset_args = [f"const int {name}_offset" for name in self.named_nodes]
        runtime_arg_decls = ",".join(
            [f"{arg.ty} {arg.name}" for arg in self.runtime_arg_info]
        )
        if runtime_arg_decls:
            runtime_arg_decls += ", "

        signature = (
            f"int {self.kernel_name}({', '.join(arg_defs + size_args + offset_args)},\
 {runtime_arg_decls}{self._EXTRA_CPP_ARGS})"
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

        arg_types: list[Any]
        if V.graph.cpp_wrapper:
            # Make sure we initialize these kernels since they're exported as
            # C-style symbol names.
            assert isinstance(wrapper, CppWrapperCpu)
            wrapper.initialized_kernels[name] = self
            # We always originally initialize name with "KERNEL_NAME". So, we
            # we replace with the real kernel name passed as an arg to this function.
            self.signature = self.signature.replace(str(Placeholder.KERNEL_NAME), name)
            _, call_args, arg_types = self.args.cpp_argdefs(DTYPE_TO_CUTLASS_TYPE)
        else:
            _, call_args, _, arg_types = self.args.python_argdefs()

        dynamic_shape_args = self.get_dynamic_shape_args()
        offset_args = self.get_offset_args()
        call_args.extend(dynamic_shape_args)  # type: ignore[arg-type]
        call_args.extend(offset_args)  # type: ignore[arg-type]
        for arg in self.runtime_arg_values:
            call_args.append(str(arg))
        arg_types.extend("const int" for _ in dynamic_shape_args)
        arg_types.extend("const int" for _ in offset_args)
        for arg in self.runtime_arg_info:
            arg_types.append(arg.ty)
        # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + ".item()"
            elif isinstance(arg_types[i], torch_dtype):
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
            workspace = str(ws.outer_name)
            call_args.append(
                workspace
                if V.graph.cpp_wrapper
                else f"c_void_p({workspace}.data_ptr())"
            )
        else:
            ws = None
            call_args.append("nullptr" if V.graph.cpp_wrapper else "None")
        if V.graph.cpp_wrapper:
            arg_types.append("uint8_t*")

        wrapper.generate_kernel_call(
            name,
            call_args,
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

    def ptr(self, node: IRNode) -> str:
        """
        Generates code which represents pointer of a given node.
        """

        if node is None:
            return "nullptr"
        arg_name = self.arg_name(node)
        if arg_name is None:
            return "nullptr"
        return f"{arg_name} + {arg_name}_offset"

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
        sizes = [
            self.find_symbol(node, "size", dim=i) or node.get_size()[i]
            for i in range(start_index, end_index + 1)
        ]
        if len(sizes) == 0:
            return str(default_value)

        sizes = [symbols(v) if isinstance(v, str) else v for v in sizes]
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
        return self.find_symbol(node, "stride", dim=index) or str(stride)

    def batch_stride(self, node: IRNode, default_value: int = 0) -> str:
        """
        Hook called from template code to get the batch stride of an arg.
        Returns 0 if batch dim is not present.

        This method assumes that batch stride is the largest stride.
        """

        if node is None:
            return str(default_value)

        if len(node.get_size()) < 3:
            return str(default_value)

        batch_stride = node.get_stride()[0]
        if V.graph.sizevars.statically_known_leq(batch_stride, 1):
            return str(batch_stride)

        return "{}*{}".format(
            self.find_symbol(node, "size", dim=1) or node.get_size()[1],
            self.find_symbol(node, "size", dim=2) or node.get_size()[2],
        )

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

    def load(self, name: str, index: Expr, mode: Any = None) -> CSEVariable:
        """
        Mock load function for memory planning to optimize allocations properly.
        """
        return self.create_cse_var(name, bounds=ValueRanges.unknown())

    def store(self, name: str, index: Expr, value: Any, mode: Any = None) -> None:
        """
        Mock store function for memory planning to optimize allocations properly.
        """
        self.store_buffer_names.add(name)


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
        input_nodes: list[Buffer],
        layout: Layout,
        make_kernel_render: Callable[
            [CUDATemplateBuffer, Optional[list[BaseSchedulerNode]]],
            tuple[CUDATemplateKernel, functools.partial[str]],
        ],
        bmreq: CUDABenchmarkRequest,
        supports_epilogue_fusion: bool,
        template: "CUDATemplate",  # type: ignore[name-defined]
        info_kwargs: Optional[
            dict[str, Union[PrimitiveInfoType, list[PrimitiveInfoType]]]
        ],  # type: ignore[type-arg]
        description: str,
    ) -> None:
        super().__init__(name, input_nodes, layout, description)
        self.category = category
        self.make_kernel_render = make_kernel_render
        self.bmreq = bmreq
        self.supports_epilogue_fusion = supports_epilogue_fusion
        self.template = template
        self.info_kwargs = info_kwargs

    def precompile(self) -> None:
        assert self.bmreq is not None
        self.bmreq.precompile()

    def benchmark(self, *args, out) -> float:
        assert self.bmreq is not None
        if config.profile_bandwidth_with_do_bench_using_profiling:
            algo = self.bmreq.make_run_fn(*args, out=out)
            return do_bench_using_profiling(algo)
        return self.bmreq.benchmark(*args, out=out)

    def __str__(self) -> str:
        return f"CUDATemplateCaller(source_file={self.bmreq.source_file})"

    def call_name(self) -> str:
        return f"cuda_template_kernels.{self.name}"

    def kernel_hash_key(self) -> str:
        """
        Return kernel hash key that does not depend on swizzle.
        """
        return "-".join(
            [
                self.category,
                self.bmreq.hash_key,
            ]
        )

    def hash_key(self) -> str:
        """
        Return kernel hash key that does not depend on swizzle.
        """
        swizzle_str: str = (
            str(self.info_kwargs.get("swizzle"))
            if isinstance(self.info_kwargs, dict)
            else "None"
        )
        return "-".join(
            [
                self.category,
                self.bmreq.hash_key,
                swizzle_str,
            ]
        )

    def info_dict(self) -> dict[str, Union[PrimitiveInfoType, list[PrimitiveInfoType]]]:
        """
        Information returned here is logged to the autotune log file when that is enabled.

        In general, we should avoid calling this function as it is expensive to compute,
        and can add up very fast.
        """
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
                "swizzle": str(self.info_kwargs["swizzle"]),
            }
        else:
            return {"backend": "CUDA", "op_type": "unknown"}

    def output_node(self) -> TensorBox:
        self.bmreq.update_workspace_size()
        buffer = CUDATemplateBuffer(
            layout=self.layout,
            inputs=self.input_nodes,
            make_kernel_render=self.make_kernel_render,
            workspace_size=self.bmreq.workspace_size,
            supports_epilogue_fusion=self.supports_epilogue_fusion,
            template=self.template,
        )
        # Pass KTC annotation to the buffer for encoding
        if "ktc" in self.annotations:
            buffer.annotations["ktc"] = self.annotations["ktc"]
        return TensorBox.create(buffer)
