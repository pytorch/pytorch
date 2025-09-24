# mypy: allow-untyped-defs
import contextlib
import dataclasses
import logging
import textwrap
from typing import Any, Callable, Optional

import sympy

import torch
from torch._inductor.codegen.common import (
    CSE,
    CSEVariable,
    IndentedBuffer,
    Kernel,
    ValueRanges,
)
from torch._inductor.ir import Buffer, ComputedBuffer, InputBuffer
from torch._inductor.ops_handler import StoreMode
from torch._inductor.utils import OrderedSet
from torch._inductor.virtualized import V

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
            "modification": self.modification,
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
            )

        subgraph = self.subgraph_bodies[body_name]
        for key, value in subgraph.to_dict().items():
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
    def create_subgraph_body(self, body_name: str):
        """Create a new subgraph body for template processing."""
        assert body_name not in self.subgraph_bodies, (
            f"Subgraph body '{body_name}' already exists"
        )
        self.subgraph_bodies[body_name] = CuteDSLSubgraphInfo(
            body=IndentedBuffer(),
            template_mask=None,
            template_out=None,
        )
        with self.set_subgraph_body(body_name):
            yield

    def def_kernel(self, *argnames):
        """Define kernel function signature for CuteDSL templates."""
        renames = IndentedBuffer(initial_indent=1)

        for i, input_node in enumerate(self.input_nodes):
            buf_name = input_node.get_name()
            self.args.input(buf_name)

            # Template aliasing: converts template variables (e.g., "input_a") to function args (e.g., "arg_input_a")
            # and generates rename statements so template code can use the original names
            if i < len(argnames):
                template_name = argnames[i]
                arg_name = f"arg_{template_name}"
                self.args.input_buffers[buf_name] = arg_name
                renames.writeline(f"{template_name} = {arg_name}")

        if self.output_node:
            self.args.output(self.output_node.get_name())

        def hook():
            # Deferred execution: arg definitions must be collected after template processing adds all args
            arg_defs, *_ = self.args.python_argdefs()
            code = IndentedBuffer()
            code.writeline(f"# Kernel function signature: {self.kernel_name}")
            params = [x.full_name() for x in arg_defs] + ["stream"]
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

    def call_kernel(self, name: str, node=None):
        """Call the kernel function. Simplified version of TritonTemplateKernel.call_kernel."""
        wrapper = V.graph.wrapper_code
        _, call_args, _, arg_types = self.args.python_argdefs()
        # TODO triton should really be swapped w/ `python`
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

        with self.create_subgraph_body(f"mod_{subgraph_number}_{num}"):
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

    def _get_input_dtype(self, name: str) -> torch.dtype:
        """Get the dtype for an input from the kernel's named_input_nodes."""
        if name in self.kernel.named_input_nodes:
            return self.kernel.named_input_nodes[name].dtype
        # TODO: Fallback for common dimension names - should be replaced with proper dtype tracking
        return torch.float32 if name not in ("b", "h", "m", "n") else torch.int32

    def load(self, name: str, index: sympy.Expr):
        """Handle loading from tensor or fixed(template args) input for CuteDSL."""
        if name not in self.fixed_inputs:
            raise NotImplementedError(
                "Tensor loading not yet supported for CuteDSL - only fixed input substitution"
            )
        value = self.fixed_inputs[name]
        dtype = self._get_input_dtype(name)

        # ensure CSE wrapping
        return self.kernel.cse.generate(
            self.kernel.body, value, bounds=ValueRanges.unknown(), dtype=dtype
        )

    def indirect_indexing(self, index_var: str, size, check, wrap_neg=True):
        """Convert index variable to symbolic form."""
        raise NotImplementedError("Indirect indexing not supported")

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> str:
        raise NotImplementedError(
            "Store operations not supported - CuteDSL limited to read-only operations"
        )

    def _add_kernel_input(self, name: str):
        """Add name as input to kernel and return input ref."""
        return self.kernel.args.input(name)

    def _process_indexing(self, index):
        """Process and rename indexing, adding symbols as kernel inputs."""
        # Convert sympy expression to string representation for CuteDSL
        return str(index)  # Simplified for now

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
