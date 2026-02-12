# mypy: allow-untyped-defs
import contextlib
import dataclasses
import logging
import textwrap
from collections.abc import Callable
from typing import Any, Optional

import sympy

import torch
from torch._inductor import config
from torch._inductor.codegen.common import (
    CSE,
    CSEVariable,
    IndentedBuffer,
    Kernel,
    ValueRanges,
)
from torch._inductor.ir import (
    BaseView,
    Buffer,
    ComputedBuffer,
    ExternKernel,
    InputBuffer,
    MutableBox,
    ReinterpretView,
)
from torch._inductor.ops_handler import StoreMode
from torch._inductor.utils import OrderedSet
from torch._inductor.virtualized import V

from ...utils import sympy_index_symbol
from .cutedsl_op_overrides import CuteDSLOpOverrides


# TODO setting the 'main' kernel w/ this suffix. We have 3 should probably just auto generate this
MAIN_SUFFIX = "main"


log = logging.getLogger(__name__)
kernel_code_log = torch._logging.getArtifactLogger(__name__, "kernel_code")


class CuteDSLKernelWrapper:
    """Wrapper to provide .run() interface for CuteDSL kernels"""

    def __init__(
        self, kernel_fn: Callable[..., Any], kernel_path: Optional[str] = None
    ):
        self.kernel_fn = kernel_fn
        self.kernel_path = kernel_path
        kernel_code_log.info("CuteDSL kernel path: %s", kernel_path)

    def run(self, *args, stream=None, **kwargs):
        """
        Execute the CuteDSL kernel.

        Args:
            *args: Arguments to pass to the kernel function
            stream: CUDA stream to pass to the kernel function
            **kwargs: Additional keyword arguments for the kernel

        Returns:
            Result of the kernel execution
        """
        return self.kernel_fn(*args, stream=stream, **kwargs)


@dataclasses.dataclass
class CuteDSLSubgraphInfo:
    """Minimal subgraph info for CuteDSL kernels."""

    body: IndentedBuffer
    template_mask: Optional[str] = None
    template_out: Optional[str] = None
    cse: Optional[CSE[Any]] = None

    def __post_init__(self):
        self.only_copy_if_non_none_fields = ("cse",)

    def to_dict(self):
        return {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }


class CuteDSLTemplateKernel(Kernel):
    """
    Template kernel implementation for CuteDSL (CUTLASS Python DSL).
    Handles code generation and argument management for CuteDSL CUDA kernels.
    Provides CuteDSL-specific functionality for tensor conversion and kernel configuration.
    """

    def __init__(
        self,
        kernel_name: str,
        input_nodes: list[Buffer],
        output_node: Buffer,
        subgraphs: Optional[list[Buffer]] = None,
    ) -> None:
        # Call parent Kernel constructor
        super().__init__()
        self.kernel_name = kernel_name
        self.input_nodes = input_nodes
        self.output_node = output_node
        self.subgraphs = subgraphs
        self.subgraph_bodies: dict[str, CuteDSLSubgraphInfo] = {}

        # Template attributes
        self.body: IndentedBuffer = IndentedBuffer()
        self.template_mask: Optional[str] = None
        self.template_out: Optional[str] = None
        self.template_indices: Optional[list[Any]] = None
        self.render_hooks: dict[str, Any] = {}

        # TODO Additional attributes needed by template system
        self.prologue_fused_inputs: OrderedSet[str] = OrderedSet()
        self.prologue_fused_inputs_preserve_zero: OrderedSet[str] = OrderedSet()
        self.named_input_nodes: dict[str, Buffer] = {}

        # Create named input nodes mapping
        for i, input_node in enumerate(input_nodes):
            node_name = getattr(input_node, "name", f"input_{i}")
            self.named_input_nodes[node_name] = input_node

        self.cse = CSE(name_prefix="tmp")

        # Track all tensor buffers added during modification processing
        self.collected_tensor_buffers: list[str] = []

    def kexpr(self, expr: sympy.Expr) -> str:
        """Convert sympy expression to CuteDSL string representation."""
        return str(expr)

    def gen_imports(self) -> str:
        """Generate common imports for CuteDSL templates."""
        imports = IndentedBuffer()
        imports.splice(
            """
            import torch
            import cutlass
            import cutlass.cute as cute
            from cutlass.cute.runtime import from_dlpack
            import cuda.bindings.driver as cuda
            from cutlass._mlir.dialects import math as mlir_math
            import operator
            from torch._inductor.codegen.cutedsl._cutedsl_utils import ssa_to_indexable, result_to_ssa
            """
        )
        return imports.getvalue()

    def gen_defines(self, **kwargs) -> str:
        """Generate CuteDSL parameter definitions from kwargs, similar to Triton's gen_defines."""
        params = IndentedBuffer()
        for name, val in kwargs.items():
            params.writeline(f"{name}: cutlass.Constexpr = {val}")
        return params.getvalue()

    def render(self, template, **kwargs):
        from torch._inductor.select_algorithm import PartialRender

        """Render the kernel using the template, returning PartialRender object with hooks."""
        # Available {{}} hooks for jinja rendering
        template_env = {
            "def_kernel": self.def_kernel,
            "gen_defines": lambda: self.gen_defines(**kwargs),
            "get_output": self.get_output,
            "get_tensor_buffers": self.get_tensor_buffers,
            "unpack_buffers": self.unpack_buffers,
            "modification": self.modification,
            "set_cute_hash": self.set_cute_hash,
        }

        # Render the template with the environment and provided kwargs
        rendered_code = template.render(
            kernel_name=self.kernel_name,
            input_nodes=self.input_nodes,
            output_node=self.output_node,
            **template_env,
            **kwargs,
        )

        # Always prepend the common imports
        imports = self.gen_imports()
        full_code = imports + rendered_code

        return PartialRender(full_code, self.render_hooks)

    @contextlib.contextmanager
    def set_subgraph_body(self, body_name: str):
        """Set the active subgraph body for template processing."""
        assert all(
            hasattr(self, field.name)
            for field in dataclasses.fields(CuteDSLSubgraphInfo)
        )
        old_state = {
            key.name: getattr(self, key.name)
            for key in dataclasses.fields(CuteDSLSubgraphInfo)
        }

        if body_name not in self.subgraph_bodies:
            self.subgraph_bodies[body_name] = CuteDSLSubgraphInfo(
                body=IndentedBuffer(),
                template_mask=None,
                template_out=None,
                cse=None,
            )

        subgraph = self.subgraph_bodies[body_name]
        for key, value in subgraph.to_dict().items():
            if value is None and key in getattr(
                subgraph, "only_copy_if_non_none_fields", ()
            ):
                continue
            setattr(self, key, value)

        try:
            yield
        finally:
            # Save current state back to subgraph
            self.subgraph_bodies[body_name] = CuteDSLSubgraphInfo(
                **{
                    key.name: getattr(self, key.name)
                    for key in dataclasses.fields(CuteDSLSubgraphInfo)
                }
            )
            # Restore old state
            for key, value in old_state.items():
                setattr(self, key, value)

    @contextlib.contextmanager
    def create_subgraph_body(self, body_name: str, *, clear_cse: bool = False):
        """Create a new subgraph body for template processing."""
        assert body_name not in self.subgraph_bodies, (
            f"Subgraph body '{body_name}' already exists"
        )
        new_cse = self.cse.clone() if clear_cse else None
        self.subgraph_bodies[body_name] = CuteDSLSubgraphInfo(
            body=IndentedBuffer(),
            template_mask=None,
            template_out=None,
            cse=new_cse,
        )
        with self.set_subgraph_body(body_name):
            yield

    def _get_reinterpret_view(self, node) -> ReinterpretView | None:
        """Extract or convert to ReinterpretView from a node, handling all views."""
        while isinstance(node, MutableBox):
            node = node.data
        if isinstance(node, BaseView):
            return ExternKernel.convert_to_reinterpret_view(node)
        return None

    def def_kernel(self, *argnames):
        """Define kernel function signature for CuteDSL templates.

        When inputs are ReinterpretViews of the same underlying buffer (e.g., Q/K/V
        from fused QKV projection), we generate separate arguments for each input
        even though they share the same underlying buffer.
        """
        renames = IndentedBuffer(initial_indent=1)

        # Track template input args - each input gets its own arg even if buffers are shared
        self._template_input_args: list[tuple[str, Buffer]] = []
        self._seen_input_args: OrderedSet[str] = OrderedSet()

        for i, input_node in enumerate(self.input_nodes):
            buf_name = input_node.get_name()
            # Register with args system (may deduplicate, but we track separately)
            self.args.input(buf_name)

            if i < len(argnames):
                template_name = argnames[i]
                arg_name = f"arg_{template_name}"
                self.args.input_buffers[buf_name] = arg_name
                renames.writeline(f"{template_name} = {arg_name}")
                self._template_input_args.append((arg_name, input_node))
                self._seen_input_args.add(arg_name)

        if self.output_node:
            self.args.output(self.output_node.get_name())

        def hook():
            # Generate signature with template input args plus additional args (output, sizevars)
            code = IndentedBuffer()
            code.writeline(f"# Kernel function signature: {self.kernel_name}")

            # Start with template input args
            params = [arg_name for arg_name, _ in self._template_input_args]

            # Get additional args from python_argdefs (output, sizevars, etc.)
            arg_defs, _, _, _ = self.args.python_argdefs()
            for arg_def in arg_defs:
                if arg_def.full_name() not in self._seen_input_args:
                    params.append(arg_def.full_name())

            params.append("stream")
            code.writeline(
                f"def {self.kernel_name}_{MAIN_SUFFIX}({', '.join(params)}):"
            )
            with code.indent():
                code.splice(renames.getvalue())
            return code.getvalue()

        assert "<DEF_KERNEL>" not in self.render_hooks
        # Placeholder-based rendering: hook will be called when template encounters "<DEF_KERNEL>"
        self.render_hooks["<DEF_KERNEL>"] = hook
        return "<DEF_KERNEL>"

    def get_output(self):
        """Get the actual argument name for the output buffer."""
        assert self.output_node, "Output node must exist to get output buffer name"
        buf_name = self.output_node.get_name()
        output = self.args.output_buffers.get(buf_name, None)
        if output is None:
            raise ValueError(f"Output buffer '{buf_name}' not found in args")
        return output

    def set_cute_hash(self, func_name: str, suffix: str = ""):
        """Generate code to set __cute_hash__ on a codegen function.

        This allows hash_callable in flash_attn to skip expensive runtime hashing
        for Inductor-generated functions. The hash is based on the kernel name
        which already contains a unique hash suffix.
        """
        hash_value = f"{self.kernel_name}_{suffix}" if suffix else self.kernel_name
        return f'{func_name}.__cute_hash__ = "{hash_value}"'

    def get_tensor_buffers(self):
        """Get list of tensor buffer names that were collected during modifications."""
        return self.collected_tensor_buffers

    def unpack_buffers(self, buffer_list_name: str, *, indent_width: int = 4):
        """Generate buffer unpacking code via render hook."""

        def hook():
            tensor_buffers = self.get_tensor_buffers()
            if not tensor_buffers:
                return ""

            # Generate unpacking assignments: in_ptr4 = buffers[0], etc.
            unpacking_lines = []
            for i, buffer_name in enumerate(tensor_buffers):
                # pyrefly: ignore [bad-argument-type]
                unpacking_lines.append(f"{buffer_name} = {buffer_list_name}[{i}]")

            indent = " " * indent_width
            return "\n" + indent + ("\n" + indent).join(unpacking_lines)

        # Register the hook and return placeholder
        placeholder = "<UNPACK_BUFFERS>"
        # TODO: I think double invoking is fine for this specific hook
        # assert placeholder not in self.render_hooks
        self.render_hooks[placeholder] = hook
        return placeholder

    def call_kernel(self, name: str, node=None):
        """Call the kernel function. Simplified version of TritonTemplateKernel.call_kernel.

        For inputs that are ReinterpretViews (e.g., Q/K/V slices from fused QKV),
        we generate reinterpret_tensor() calls to properly handle the views.
        """
        wrapper = V.graph.wrapper_code

        # Build call args matching the signature generated in `def_kernel`
        call_args = []
        arg_types = []

        for _, input_node in self._template_input_args:
            reinterpret_view = self._get_reinterpret_view(input_node)
            if reinterpret_view is not None:
                call_args.append(reinterpret_view.codegen_reference())
            else:
                call_args.append(input_node.get_name())
            arg_types.append(V.graph.get_dtype(input_node.get_name()))

        # Add additional args from python_argdefs (output, sizevars, ..)
        orig_arg_defs, orig_call_args, _, orig_arg_types = self.args.python_argdefs()
        for arg_def, call_arg, arg_type in zip(
            orig_arg_defs, orig_call_args, orig_arg_types
        ):
            # dedupe
            if arg_def.full_name() not in self._seen_input_args:
                call_args.append(call_arg)
                arg_types.append(arg_type)

        # TODO this karg really should not be called `triton`
        wrapper.generate_kernel_call(name, call_args, triton=True, arg_types=arg_types)

    def _get_subgraph(self, subgraph_number: int):
        """Get subgraph by number for modification processing."""
        assert isinstance(subgraph_number, int)
        assert isinstance(self.subgraphs, list)
        assert subgraph_number < len(self.subgraphs), (
            f"Invalid subgraph number provided to create_modification, {subgraph_number} must be < {len(self.subgraphs)}"
        )
        assert self.body.getvalue() == "", (
            "Body should be clear before adding a modification"
        )
        return self.subgraphs[subgraph_number]

    def modification(
        self,
        subgraph_number: int,
        output_name: Optional[str],
        mask: Optional[str] = None,
        **fixed_inputs,
    ) -> str:
        """Generate CuteDSL code for a subgraph modification."""
        # Find unique name to avoid collisions between multiple modifications of same subgraph
        num = 0
        while f"mod_{subgraph_number}_{num}" in self.subgraph_bodies:
            num += 1

        with self.create_subgraph_body(f"mod_{subgraph_number}_{num}", clear_cse=True):
            subgraph = self._get_subgraph(subgraph_number)
            modification_handler = ModificationWrapperCuteDSL(
                self, subgraph_number, fixed_inputs, mask
            )
            with V.set_kernel_handler(self), V.set_ops_handler(modification_handler):
                assert isinstance(subgraph, (ComputedBuffer, list)), (
                    f"Expected ComputedBuffer or List[ComputedBuffer], got {type(subgraph)}"
                )

                if isinstance(subgraph, list):
                    raise NotImplementedError(
                        "Scatter graphs are not supported for CuteDSL"
                    )

                if isinstance(subgraph.data, InputBuffer):
                    # grad_score_mod can be InputBuffers
                    out = subgraph.data.make_loader()(())
                else:
                    # Inline a pointwise lowering into the template
                    out = subgraph.data.inner_fn(())

            if output_name is not None:
                assert out is not None, (
                    f"Expected computation result for named output {output_name}"
                )
                self.body.writeline(f"{output_name} = {out.value}")
            else:
                # Side-effect only: no output assignment (currently only for scatter operations)
                raise NotImplementedError(
                    "Side-effect only modifications not yet supported for CuteDSL"
                )

            # Add Buffers that were added during modification
            self.collected_tensor_buffers.extend(modification_handler.tensor_buffers)

            return self.body.getvalue()


class ModificationWrapperCuteDSL(V.WrapperHandler):  # type: ignore[name-defined]
    """
    Wrapper handler that enables CuteDSL code generation during subgraph modifications.

    This class sits between the PyTorch IR and CuteDSL code generation, providing:
    1. Operation substitution: converts PyTorch ops to CuteDSL equivalents via CuteDSLOpOverrides
    2. Placeholder handling: resolves fixed_inputs during template processing
    3. Limited operation support: currently restricted to pointwise operations

    """

    def __init__(
        self,
        kernel,
        subgraph_number: int,
        fixed_inputs: dict[str, Any],
        mask: Optional[str],
    ):
        cutedsl_ops = CuteDSLOpOverrides()
        super().__init__(cutedsl_ops)
        self.name = f"CuteDSLPlaceholderSubstitution_{subgraph_number}"
        self.kernel = kernel
        self.fixed_inputs = fixed_inputs
        self.mask = mask
        # Track tensor buffers that get added during modification processing
        self.tensor_buffers: list[str] = []

    def _get_input_dtype(self, name: str) -> torch.dtype:
        """Get the dtype for an input from the kernel's named_input_nodes."""
        if name in self.kernel.named_input_nodes:
            return self.kernel.named_input_nodes[name].dtype
        # TODO: Fallback for common dimension names - should be replaced with proper dtype tracking
        return torch.float32 if name not in ("b", "h", "m", "n") else torch.int32

    def load(self, name: str, index: sympy.Expr):
        """Handle loading from tensor or fixed(template args) input for CuteDSL."""
        from torch._inductor.kernel.flex.flex_flash_attention import HierarchicalIndex

        if name not in self.fixed_inputs:
            var = self._add_kernel_input(name)
            buffer = V.graph.get_buffer(name)
            var_dtype = buffer.dtype

            cute_dtype = CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(
                var_dtype, "cutlass.Float32"
            )
            idx_vars = [
                self._emit_scalar_fragment(
                    self.kernel.kexpr(self.kernel.rename_indexing(dim_index)),
                    "cutlass.Int32",
                    torch.int32,
                )
                for dim_index in (
                    index.args if isinstance(index, HierarchicalIndex) else (index,)
                )
            ]

            val_frag = self.kernel.cse.newvar(dtype=var_dtype)
            self.kernel.body.writeline(
                f"{val_frag} = cute.make_rmem_tensor(1, {cute_dtype})"
            )
            self.kernel.body.writeline(
                f"{val_frag}[0] = ({var}[{', '.join(idx_vars)}])"
            )

            final_expr = f"{val_frag}.load()"

            if (
                var_dtype in (torch.float16, torch.bfloat16)
                and config.triton.codegen_upcast_to_fp32
            ):
                final_expr = f"({final_expr}).to(cutlass.Float32)"
                var_dtype = torch.float32

            out = self.kernel.cse.generate(
                self.kernel.body,
                final_expr,
                dtype=var_dtype,
                bounds=ValueRanges.unknown(),
            )
            return out

        value = self.fixed_inputs[name]
        dtype = self._get_input_dtype(name)

        return self.kernel.cse.generate(
            self.kernel.body, value, bounds=ValueRanges.unknown(), dtype=dtype
        )

    def _emit_scalar_fragment(
        self, expr_str: str, cute_dtype: str, torch_dtype: torch.dtype
    ) -> str:
        """
        Convert SSA expression to indexable scalar for tensor loads.

        Workaround for lack of gather support: SSA values cannot be used directly
        as indices. This generates code to convert SSA → indexable scalar.
        """
        result = self.kernel.cse.newvar(dtype=torch_dtype)
        self.kernel.body.writeline(
            f"{result} = ssa_to_indexable({expr_str}, {cute_dtype})"
        )
        return str(result)

    def indirect_indexing(self, index_var: str, size, check, wrap_neg=True):
        """Convert index variable to symbolic form."""
        return sympy_index_symbol(str(index_var))

    # pyrefly: ignore [bad-override]
    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> str:
        raise NotImplementedError(
            "Store operations not supported - CuteDSL limited to read-only operations"
        )

    def _add_kernel_input(self, name: str):
        """Add name as input to kernel and return input ref."""
        # Get the remapped name that will be used in the kernel
        remapped_name = self.kernel.args.input(name)
        # Track the remapped name for later collection
        if remapped_name not in self.tensor_buffers:
            self.tensor_buffers.append(remapped_name)
        return remapped_name

    def _process_indexing(self, index):
        """Process and rename indexing, adding symbols as kernel inputs."""
        renamed = self.kernel.rename_indexing(index)
        return self.kernel.kexpr(renamed)

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        try:
            return getattr(self._inner, name)(*args, **kwargs)
        except NotImplementedError as e:
            bar = "=" * 80
            msg = textwrap.dedent(f"""
                {bar}
                UNSUPPORTED CUTEDSL OPERATION: '{name}'
                {bar}
                This operation is not yet implemented in Inductor.

                Please open an issue at: https://github.com/pytorch/pytorch/issues
                with the following information:

                Operation: {name}
                Args: {args!r}
                Kwargs: {kwargs!r}

                Title your issue: [CuteDSL] Missing operation: {name}
                {bar}
            """).strip()
            raise NotImplementedError(msg) from e
