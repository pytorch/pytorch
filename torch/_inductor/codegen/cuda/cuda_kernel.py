import logging
from typing import Callable, cast, Dict, List, Optional, Set, Union

from ...autotune_process import CUDABenchmarkRequest
from ...ir import Buffer, ComputedBuffer, IRNode, Pointwise, TemplateBuffer, TensorBox
from ...scheduler import Scheduler
from ...select_algorithm import ChoiceCaller
from ...utils import sympy_product
from ...virtualized import V

from ..common import IndentedBuffer, Kernel, OpOverrides
from ..cpp import CppPrinter, DTYPE_TO_CPP

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

    def __init__(self, kernel_name):
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
        return f"PT_EXPORT int {self.kernel_name}({', '.join(arg_defs)}, {self._EXTRA_CPP_ARGS})"

    def call_kernel(self, name: str, node: "CUDATemplateBuffer") -> None:
        """
        Generates code to call the kernel through V.graph.wrapper_code.
        used from within torch._inductor.wrapper.WrapperCodeGen

        name: Name of kernel function.
        node: The CUDATemplateBuffer node which contains information about the kernel, it's fused epilogue nodes
        as well as all required inputs and outputs.
        """
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
            device_index=V.graph.scheduler.current_device.index,
            cuda=True,
            triton=False,
        )

    def dtype(self, node: IRNode) -> Optional[str]:
        """
        Generates code which represents dtype of a given node.
        """

        if node is None:
            return "void"
        return DTYPE_TO_CPP.get(node.get_layout().dtype)

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


class CUDATemplateBuffer(TemplateBuffer):
    def __init__(
        self,
        template: "CUDATemplate",  # type: ignore[name-defined]
        op: "cutlass_gemm_op.GemmOperation",  # type: ignore[name-defined]
        epilogue_nodes: Optional[
            List[IRNode]
        ] = None,  # We need a new instance of this op every time we fuse
        workspace_size: Optional[
            Union[int, Callable[[], int]]
        ] = None,  # May be an int or a callback returning an int
        merged_input_nodes: Optional[
            List[IRNode]
        ] = None,  # input nodes of the template + epilogue nodes
        name: Optional[
            str
        ] = None,  # Name of the buffer, must be registered with the graph if passed.
        **render_kwargs,  # passed through to template_node.render
    ):
        """
        Initializes a new instance of the CUDATemplateBuffer class.

        Args:
            template (CUDATemplate): The CUDATemplate object that this buffer uses to generate the Kernel source code
            op (cutlass_gemm_op.GemmOperation): The GemmOperation object that represents the Cutlass GEMM operation
                                                as well as it's compile-time arguments such as tile-sizes
            epilogue_nodes (Optional[List[IRNode]]): A list of IRNodes representing the epilogue nodes for this
                                                    buffer. Default is None.
            workspace_size (Optional[Union[int, Callable[[], int]]]): The size of the workspace needed for this buffer.
                         It can be an integer or a callable function that returns an integer. Default is None.
            merged_input_nodes (Optional[List[IRNode]]): A list of IRNodes representing the merged input
                                                        nodes for this buffer. Default is None.
            name (Optional[str]): The name of the buffer. It must be already registered with the graph if passed.
                                                        Default is None. If None, a new name will be generated and
                                                        the name will be registered with the Graph
            **render_kwargs: Additional keyword arguments passed through to template_node.render.

        """
        if epilogue_nodes is None:
            epilogue_nodes = []
        if merged_input_nodes is None:
            assert (
                epilogue_nodes is None or len(epilogue_nodes) == 0
            ), "If epilogue nodes are passed, merged_input_nodes must be passed as well"
            input_nodes = template.input_nodes
        else:
            input_nodes = merged_input_nodes
        super().__init__(
            template.layout, input_nodes, self._make_kernel_render, name=name
        )
        self.template = template
        self.op = op
        self._epilogue_nodes = epilogue_nodes
        # TODO: Once we support non-pointwise fusions, layout might be modified by epilogues
        self.layout = template.layout
        self._render_kwargs = render_kwargs
        # Global memory (in bytes) needed for this template.
        self._workspace_size = workspace_size

    @property
    def epilogue_nodes(self):
        """
        Return the epilogue nodes for the CUDATemplateBuffer.
        This is a read-only property to signal it should not be mutated
        """
        return self._epilogue_nodes

    @property
    def workspace_size(self):
        """
        Read-only property that returns the workspace size for the CUDATemplateBuffer, possibly retrieved
        lazily via a callback.
        """
        return self.get_workspace_size()

    def get_workspace_size(self):
        """
        See self.workspace_size property
        """
        if callable(self._workspace_size):
            return self._workspace_size()
        return self._workspace_size if self._workspace_size is not None else 0

    def _make_kernel_render(
        self, output_node, epilogue_nodes: Optional[List[IRNode]] = None
    ):
        """
        Private method that may be passed as a callback bound to this instance,
        which returns the parameterless kernel render function and a CUDATemplateKernel.

        This callback is required for the parent class (TemplateBuffer) and is used by
        CUDAScheduling.codegen_template(...) to render the Kernel source.
        """
        assert output_node is self
        assert self.workspace_size >= 0
        kernel = CUDATemplateKernel(
            kernel_name="KERNEL_NAME",
        )

        def render():
            return self.template.render(
                kernel=kernel,
                template_node=self,
                op=self.op,
                epilogue_nodes=epilogue_nodes,
                **self._render_kwargs,
            )

        return kernel, render

    def can_fuse_epilogue(self, node) -> bool:
        """

        Check if the given node can be fused with the epilogue. At the moment, Kernels
        support fusion with Pointwise operations, wrapped in (named) ComputedBuffer nodes.

        Args:
            node: The IRNode to be check if it can be fused with the epilogue.

        Returns:
        - bool: True if the given node can be fused with the epilogue, False otherwise.

        """
        if not self.template.can_fuse_epilogue:
            # The used GEMM op does not support fusing epilogues
            return False

        if not isinstance(node, ComputedBuffer):
            return False
        if not isinstance(node.data, Pointwise):
            return False

        # We can fuse a Pointwise op that depends on the last fused epilogue node
        # if any. If there is no epilogue node yet, it needs to depend on the template
        # node
        node_name = node.name if node.name is not None else node.data.name
        if node_name is None:
            return False

        if len(self._epilogue_nodes) == 0:
            if self.name not in node.get_read_names():
                return False
        else:
            last_epilogue_node = self._epilogue_nodes[-1]
            last_epilogue_name = (
                last_epilogue_node.name
                if last_epilogue_node.name is not None
                else last_epilogue_node.data.name
            )
            if last_epilogue_name not in node.get_read_names():
                return False
        if node.layout != self.layout:
            return False
        try:
            from torch._inductor.codegen.cuda.cutlass_epilogue_gen import (
                CutlassEVTEpilogueArgumentFormatter,
                CutlassEVTEpilogueTypeFormatter,
            )

            CutlassEVTEpilogueTypeFormatter.ir_to_evt_string(
                self.name, "anything", [node]
            )
            CutlassEVTEpilogueArgumentFormatter.ir_to_evt_argument_string(
                self.name, [node]
            )
        except NotImplementedError as e:
            not_implemented_op = str(e)
            if not_implemented_op.startswith("_op_"):
                not_implemented_op = not_implemented_op[4:]
                log.warning(
                    f"Cannot fuse epilogue node {node} into {self.name}, likely due to unsupported operation: {not_implemented_op}"  # noqa: G004, B950
                )
                return False
            else:
                # Likely due to unsupported dtype.
                log.warning(
                    f"Cannot fuse epilogue node {node} into {self.name}. Reason: {not_implemented_op}"  # noqa: G004, B950
                )
                return False
        return True

    def create_fused_buffer(self, node, scheduler: Scheduler) -> "CUDATemplateBuffer":
        """
        Creates a new CUDATemplateBuffer that fuses the given node into the current one.
        This is non-mutating, it returns a new instance of CUDATemplateBuffer,
        self remains unchanged. The returned CUDATemplateBuffer gets the same name as self.

        It requires the current Scheduler in order to map epilogue inputs to IRNodes by their name
        """
        assert self.can_fuse_epilogue(node)
        epilogue_nodes = (
            self._epilogue_nodes if self._epilogue_nodes is not None else []
        )
        new_epilogue_nodes = list(epilogue_nodes) + [node]
        merged_input_nodes = CUDATemplateBuffer.merge_inputs(
            scheduler, new_epilogue_nodes, self.template
        )
        return CUDATemplateBuffer(
            self.template,
            self.op,
            new_epilogue_nodes,
            self._workspace_size,
            merged_input_nodes=merged_input_nodes,
            name=self.name,
            **self._render_kwargs,
        )

    @staticmethod
    def merge_inputs(
        scheduler: Scheduler, epilogue_nodes: List[IRNode], template: "CUDATemplate"  # type: ignore[name-defined]
    ) -> List[IRNode]:
        """
        Merge all inputs, including extra inputs from epilogue_nodes.

        Args:
            scheduler (Scheduler): The scheduler object, used to look up nodes by name.
            epilogue_nodes (List[IRNode]): The list of additional epilogue nodes.
            template (CUDATemplate): The CUDATemplate object.

        Returns:
        - List[IRNode]: The list of merged input nodes.
        """
        # Merge all inputs, including extra inputs from epilogue_nodes
        # input nodes are not hashable, so we cannot directly place them in sets

        template_input_id_set: Set[str] = {
            cast(Buffer, irnode).get_name() for irnode in template.input_nodes
        }
        intermediate_id_set: Set[str] = {template.name} | {
            cast(Buffer, n).get_name() for n in epilogue_nodes
        }
        covered_input_id_set = set(template_input_id_set) | intermediate_id_set
        extra_inputs = []
        for epilogue_node in epilogue_nodes:
            # IRNodes store no references to their inputs in general
            # we need to retrieve them indirectly from the current
            # V.graph.scheduler
            for node_name in epilogue_node.get_read_names():
                node = scheduler.name_to_node[node_name]
                assert hasattr(
                    node, "node"
                ), f"Scheduler node {node} does not have a node attribute"
                irnode = node.node
                if irnode.get_name() not in covered_input_id_set:
                    extra_inputs.append(irnode)
                    covered_input_id_set.add(irnode.get_name())
        input_nodes = list(template.input_nodes) + extra_inputs
        return input_nodes


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
        bmreq: CUDABenchmarkRequest,
        template_buffer: CUDATemplateBuffer,
    ):
        super().__init__(
            name, template_buffer.template.input_nodes, template_buffer.layout
        )
        self.category = category
        self.bmreq = bmreq

        self.template_buffer = template_buffer

    def benchmark(self, *args, out) -> float:
        assert self.bmreq is not None
        return self.bmreq.benchmark(*args, output_tensor=out)

    def __str__(self):
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

    def output_node(self) -> TensorBox:
        return TensorBox.create(self.template_buffer)
