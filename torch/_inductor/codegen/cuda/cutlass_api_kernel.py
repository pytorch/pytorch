# mypy: allow-untyped-defs
import logging
from typing import Any, Optional

from torch._inductor.codegen.common import IndentedBuffer, Kernel
from torch._inductor.ir import Buffer, ReinterpretView
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


class CutlassAPIKernelWrapper:
    """Wrapper to provide .run() interface for cutlass_api kernels."""

    def __init__(self, kernel_fn, kernel_path: Optional[str] = None):
        self.kernel_fn = kernel_fn
        self.kernel_path = kernel_path

    def run(self, *args, stream=None, **kwargs):
        """Execute the cutlass_api kernel."""
        return self.kernel_fn(*args, stream=stream, **kwargs)


class CutlassAPITemplateKernel(Kernel):
    """
    Template kernel implementation for cutlass_api.

    Generates Python code that calls cutlass_api to execute GEMM operations.
    Unlike CuteDSL which uses Jinja templates, this generates simpler direct
    Python code.
    """

    def __init__(
        self,
        kernel_name: str,
        input_nodes: list[Buffer],
        output_node: Buffer,
        kernel_metadata: dict[str, Any],
        accumulator_type: Any,
    ) -> None:
        super().__init__()
        self.kernel_name = kernel_name
        self.input_nodes = input_nodes
        self.output_node = output_node
        self.kernel_metadata = kernel_metadata
        self.accumulator_type = accumulator_type

        self._template_input_args: list[tuple[str, Buffer]] = []
        self._seen_input_args: OrderedSet[str] = OrderedSet()

        for i, input_node in enumerate(input_nodes):
            param_name = f"in_ptr{i}"
            self._template_input_args.append((param_name, input_node))
            self._seen_input_args.add(param_name)

    def gen_imports(self) -> str:
        """Generate common imports for cutlass_api code."""
        imports = IndentedBuffer()
        imports.splice(
            """
            import torch
            import cutlass
            import cutlass_api
            """
        )
        return imports.getvalue()

    def render(self) -> str:
        """
        Render the cutlass_api kernel code as a Python source string.

        Generates Python code that:
        1. Looks up the cutlass_api kernel by name from the manifest (cached in
           _cutlass_api_kernel_cache to avoid repeated manifest searches)
        2. Creates GemmArguments with the input/output tensors and accumulator type
        3. Compiles the kernel for the specific tensor shapes/dtypes (cached in
           _cutlass_api_artifact_cache keyed by (shape, dtype) tuple)
        4. Runs the kernel with the compiled artifact and CUDA stream

        The caching strategy ensures:
        - Kernel lookup happens once per unique kernel name
        - Compilation happens once per unique (shape, dtype) combination
        - Runtime execution is just the kernel.run() call with cached artifact

        Returns:
            Python source code string to be written to a .py file and loaded
            via async_compile.cutlass_api()
        """
        code = IndentedBuffer()

        code.splice(self.gen_imports())
        code.writeline("")

        kernel_name_str = self.kernel_metadata["kernel_name"]
        code.writeline(f'_CUTLASS_API_KERNEL_NAME = "{kernel_name_str}"')
        code.writeline("_cutlass_api_kernel_cache = {}")
        code.writeline("_cutlass_api_artifact_cache = {}")
        code.writeline("")

        acc_dtype_str = str(self.accumulator_type).split(".")[-1]  # e.g., "float32"
        code.writeline(f"_ACCUMULATOR_DTYPE = torch.{acc_dtype_str}")
        code.writeline("")

        code.writeline("def _get_accumulator_type():")
        with code.indent():
            code.writeline("dtype_map = {")
            with code.indent():
                code.writeline("torch.float32: cutlass.Float32,")
                code.writeline("torch.float16: cutlass.Float16,")
                code.writeline("torch.bfloat16: cutlass.BFloat16,")
            code.writeline("}")
            code.writeline("return dtype_map.get(_ACCUMULATOR_DTYPE, cutlass.Float32)")
        code.writeline("")

        input_params = []
        for i, _ in enumerate(self.input_nodes):
            input_params.append(f"in_ptr{i}")

        input_params.append("out_ptr0")
        input_params.append("stream=None")

        params_str = ", ".join(input_params)
        code.writeline(f"def {self.kernel_name}_main({params_str}):")
        with code.indent():
            code.writeline(
                "global _cutlass_api_kernel_cache, _cutlass_api_artifact_cache"
            )
            code.writeline("")
            code.writeline(
                "if _CUTLASS_API_KERNEL_NAME not in _cutlass_api_kernel_cache:"
            )
            with code.indent():
                code.writeline("kernels = cutlass_api.get_kernels(")
                with code.indent():
                    code.writeline(
                        "metadata_filter=lambda m: m.kernel_name == _CUTLASS_API_KERNEL_NAME"
                    )
                code.writeline(")")
                code.writeline("if not kernels:")
                with code.indent():
                    code.writeline(
                        'raise RuntimeError(f"Could not find cutlass_api kernel: {_CUTLASS_API_KERNEL_NAME}")'
                    )
                code.writeline(
                    "_cutlass_api_kernel_cache[_CUTLASS_API_KERNEL_NAME] = kernels[0]"
                )
            code.writeline("")
            code.writeline(
                "kernel = _cutlass_api_kernel_cache[_CUTLASS_API_KERNEL_NAME]"
            )
            code.writeline("")

            code.writeline("args = cutlass_api.arguments.GemmArguments(")
            with code.indent():
                code.writeline("in_ptr0,")
                code.writeline("in_ptr1,")
                code.writeline("out_ptr0,")
                code.writeline("accumulator_type=_get_accumulator_type(),")
            code.writeline(")")
            code.writeline("")

            code.writeline(
                "cache_key = (in_ptr0.shape, in_ptr0.dtype, in_ptr1.shape, in_ptr1.dtype)"
            )
            code.writeline("if cache_key not in _cutlass_api_artifact_cache:")
            with code.indent():
                code.writeline(
                    "_cutlass_api_artifact_cache[cache_key] = kernel.compile(args)"
                )
            code.writeline("")
            code.writeline("artifact = _cutlass_api_artifact_cache[cache_key]")
            code.writeline("")

            code.writeline(
                "kernel.run(args, artifact, stream=stream, assume_supported_args=True)"
            )

        return code.getvalue()

    # TODO(nikhilap) Do this like cutedsl
    def _get_reinterpret_view(self, node: Buffer) -> Optional[ReinterpretView]:
        """Check if node is a ReinterpretView and return it."""
        if isinstance(node, ReinterpretView):
            return node
        return None

    def call_kernel(self, name: str, node=None):
        """
        Generate the kernel call in the wrapper code.

        Similar to CuteDSLTemplateKernel.call_kernel but simplified for cutlass_api.
        """
        wrapper = V.graph.wrapper_code

        call_args = []
        arg_types = []

        for _, input_node in self._template_input_args:
            reinterpret_view = self._get_reinterpret_view(input_node)
            if reinterpret_view is not None:
                call_args.append(reinterpret_view.codegen_reference())
            else:
                call_args.append(input_node.get_name())
            arg_types.append(V.graph.get_dtype(input_node.get_name()))

        output_name = self.output_node.get_name()
        call_args.append(output_name)
        arg_types.append(V.graph.get_dtype(output_name))

        # Generate the kernel call using triton=True for Python-based kernels
        wrapper.generate_kernel_call(name, call_args, triton=True, arg_types=arg_types)
