# mypy: allow-untyped-defs
import logging
from typing import Callable, Dict, List, Optional, TYPE_CHECKING, Union

from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu

from ...ir import Buffer, ChoiceCaller, IRNode, Layout, PrimitiveInfoType, TensorBox
from ...virtualized import V
from ..common import Kernel, OpOverrides, WorkspaceArg, WorkspaceZeroMode
from ..cpp_utils import CppPrinter
from .rocm_benchmark_request import ROCmBenchmarkRequest
from .rocm_template_buffer import ROCmTemplateBuffer


if TYPE_CHECKING:
    from torch._inductor.codegen.rocm.rocm_template import ROCmTemplate

log = logging.getLogger(__name__)

cexpr = CppPrinter().doprint


def _normalize_idx(index: int, total_length: int) -> int:
    return index if index >= 0 else index + total_length


class ROCmKernel(Kernel):
    """
    Baseclass for ROCm based Kernels
    """

    overrides = OpOverrides


class ROCmTemplateKernel(ROCmKernel):
    """
    Template kernels defined by ROCm in C++.
    """

    _EXTRA_CPP_ARGS = "size_t* workspace_size, uint8_t* workspace, hipStream_t stream"

    def __init__(self, kernel_name) -> None:
        """
        Initializes a new instance of the ROCmTemplateKernel class.

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

    def get_signature(self):
        return self.signature

    def def_kernel(
        self,
        inputs: List[IRNode],
        outputs: List[IRNode],
        size_args: List[str],
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

        signature = f"int {self.kernel_name}({', '.join(arg_defs + size_args)}, {self._EXTRA_CPP_ARGS})"
        self.signature = signature
        return signature

    def call_kernel(
        self,
        name: str,
        node: "ROCmTemplateBuffer",
    ) -> None:
        """
        Generates code to call the kernel through V.graph.wrapper_code.
        used from within torch._inductor.wrapper.PythonWrapperCodegen

        name: Name of kernel function.
        node: The ROCmTemplateBuffer node which contains information about the kernel, it's fused epilogue nodes
        as well as all required inputs and outputs.
        """
        wrapper = V.graph.wrapper_code

        if V.graph.cpp_wrapper:
            # Make sure we initialize these kernels since they're exported as
            # C-style symbol names.
            assert isinstance(wrapper, CppWrapperCpu)
            wrapper.initialized_kernels[name] = self
            # Kinda hacky because we always originally initialize name with "KERNEL_NAME"
            # So, we replace with the real kernel name passed as an arg to this function.
            self.signature = self.signature.replace("KERNEL_NAME", name)
            _, call_args, arg_types = self.args.cpp_argdefs()
        else:
            _, call_args, _, arg_types = self.args.python_argdefs()
        kernel_args = []
        for arg in call_args:
            # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
            if V.graph.is_unspec_arg(arg):
                arg = arg + ".item()"
            else:
                if not V.graph.cpp_wrapper:
                    arg = f"c_void_p({arg}.data_ptr())"
            kernel_args.append(arg)

        # add size args
        size_args = [
            f"{V.graph.sizevars.simplify(sarg)}" for sarg in node.template.size_args()
        ]
        if V.graph.cpp_wrapper:
            kernel_args.extend(size_args)
        else:
            kernel_args.extend(f"c_int({sarg})" for sarg in size_args)

        if V.graph.cpp_wrapper:
            arg_types.extend(["int"] * len(node.template.size_args()))

        # workspace_size ptr is NULL to mark this call is not intended for retrieving workspace_size.
        # workspace_size should have already been retrieved prior to this call.
        kernel_args.append("nullptr" if V.graph.cpp_wrapper else "None")
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
            kernel_args.append(
                data_ptr if V.graph.cpp_wrapper else f"c_void_p({data_ptr})"
            )
        else:
            ws = None
            kernel_args.append("nullptr" if V.graph.cpp_wrapper else "None")
        if V.graph.cpp_wrapper:
            arg_types.append("uint8_t*")

        current_device = V.graph.get_current_device_or_throw()
        wrapper.generate_kernel_call(
            name,
            kernel_args,
            device_index=current_device.index,
            gpu=True,
            triton=False,
            arg_types=arg_types,
        )
        if ws:
            wrapper.generate_workspace_deallocation(ws)


class ROCmTemplateCaller(ChoiceCaller):
    """
    ROCmTemplateCaller

    This class represents a caller for ROCm template kernels. It is a subclass of ChoiceCaller.
    Attributes:
        name (str): The name of the caller.
        category (str): The category of the caller.
        bmreq (ROCmBenchmarkRequest): The benchmark request for the caller.
        template_buffer (ROCmTemplateBuffer): The template buffer for the caller.
    """

    def __init__(
        self,
        name: str,
        category: str,
        input_nodes: List[Buffer],
        layout: Layout,
        make_kernel_render: Callable[[ROCmTemplateBuffer, Optional[List[IRNode]]], str],
        bmreq: ROCmBenchmarkRequest,
        template: "ROCmTemplate",
        info_kwargs: Optional[
            Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]
        ],
    ) -> None:
        super().__init__(name, input_nodes, layout, description="")
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
        return self.bmreq.benchmark(*args, output_tensor=out)

    def __str__(self) -> str:
        return f"ROCmTemplateCaller(source_file={self.bmreq.source_file}, {self.info_dict()})"

    def call_name(self) -> str:
        return f"rocm_template_kernels.{self.name}"

    def hash_key(self) -> str:
        return "-".join(
            [
                self.category,
                self.bmreq.hash_key,
            ]
        )

    def info_dict(self) -> Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
        return {
            "backend": "ROCm",
            "name": self.name,
            **dict(self.info_kwargs["op"].dict_items()),  # type: ignore[union-attr, index]
        }

    def output_node(self) -> TensorBox:
        self.bmreq.update_workspace_size()
        return TensorBox.create(
            ROCmTemplateBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                make_kernel_render=self.make_kernel_render,
                workspace_size=self.bmreq.workspace_size,
                template=self.template,
            )
        )
