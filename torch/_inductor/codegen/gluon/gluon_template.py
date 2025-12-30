# mypy: allow-untyped-defs
"""
Gluon template support for PyTorch Inductor.

Gluon is a DSL from the Triton project for writing Blackwell GPU kernels.
Since Gluon uses the same compilation infrastructure as Triton, this
implementation is kept minimal and reuses Triton's compilation pipeline.
"""

import functools
import hashlib
import itertools
from typing import Any, Optional

from ...ir import ChoiceCaller
from ..common import KernelTemplate


# Import TritonTemplateKernel only when needed to avoid circular imports
def _get_triton_kernel_class():
    from ...select_algorithm import TritonTemplateKernel

    return TritonTemplateKernel


# Cache for the actual GluonTemplateKernel subclass
_gluon_template_kernel_class = None


def _get_gluon_template_kernel_class():
    """
    Lazily create and cache the GluonTemplateKernel subclass.
    This avoids circular imports while ensuring proper inheritance.
    """
    global _gluon_template_kernel_class
    if _gluon_template_kernel_class is None:
        TritonTemplateKernel = _get_triton_kernel_class()

        class _GluonTemplateKernelImpl(TritonTemplateKernel):
            """
            Kernel class for Gluon templates.

            Inherits from TritonTemplateKernel and overrides jit_lines() to return
            @gluon.jit decorator with necessary imports.
            """

            def jit_lines(self):
                """
                Return gluon imports and @gluon.jit decorator since the wrapper kernel needs
                to be JIT-compiled to create layout objects and call the inner Gluon kernel functions.

                We call the parent's jit_lines() to get the @triton_heuristics.template() decorator
                which creates the CachingAutotuner, then replace @triton.jit with @gluon.jit.
                """
                import textwrap

                # Get the parent's jit_lines which includes @triton_heuristics.template(...)
                parent_jit_lines = super().jit_lines()

                # Remove leading whitespace from parent's jit_lines (it may be indented for template insertion)
                parent_jit_lines = textwrap.dedent(parent_jit_lines)

                # Add Gluon imports and replace @triton.jit with @gluon.jit
                result = """
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

""" + parent_jit_lines.replace("@triton.jit", "@gluon.jit")

                return result

            def render(
                self, template, kwargs, record_input_dependent_tracked_event=False
            ):
                """
                Override render to compute Gluon layout parameters before rendering template.

                We call get_default_for() to get layouts with appropriate swizzle widths,
                then extract parameters and pass them as template variables.
                """
                from triton.experimental.gluon import language as gl

                # Get block dimensions from kwargs
                BLOCK_M = kwargs.get("BLOCK_M", 64)
                BLOCK_N = kwargs.get("BLOCK_N", 64)
                BLOCK_K = kwargs.get("BLOCK_K", 64)

                # Get PyTorch dtype from input nodes
                import torch

                input_nodes = kwargs.get("input_nodes", [])
                if input_nodes:
                    torch_dtype = input_nodes[0].get_dtype()
                else:
                    torch_dtype = torch.bfloat16

                # Convert torch dtype to gluon dtype (following triton's approach)
                if torch_dtype == torch.float8_e5m2:
                    gluon_dtype = gl.float8e5
                    dtype_str = "gl.float8e5"
                elif torch_dtype == torch.float8_e4m3fn:
                    gluon_dtype = gl.float8e4nv
                    dtype_str = "gl.float8e4nv"
                else:
                    dtype_name = str(torch_dtype).split(".")[
                        1
                    ]  # 'torch.bfloat16' -> 'bfloat16'
                    gluon_dtype = getattr(gl, dtype_name)
                    dtype_str = f"gl.{dtype_name}"

                # Compute layouts using get_default_for() with the actual dtype
                a_layout = gl.NVMMASharedLayout.get_default_for(
                    [BLOCK_M, BLOCK_K], gluon_dtype
                )
                b_layout = gl.NVMMASharedLayout.get_default_for(
                    [1, BLOCK_N, BLOCK_K], gluon_dtype
                )
                c_layout = gl.NVMMASharedLayout.get_default_for(
                    [BLOCK_M, BLOCK_N], gluon_dtype
                )

                # Extract swizzle widths and add to kwargs
                kwargs["A_LAYOUT_SWIZZLE_BYTE_WIDTH"] = a_layout.swizzle_byte_width
                kwargs["B_LAYOUT_SWIZZLE_BYTE_WIDTH"] = b_layout.swizzle_byte_width
                kwargs["C_LAYOUT_SWIZZLE_BYTE_WIDTH"] = c_layout.swizzle_byte_width

                # Add dtype string for template
                kwargs["DTYPE"] = dtype_str

                # Call parent render with updated kwargs
                return super().render(
                    template, kwargs, record_input_dependent_tracked_event
                )

        _gluon_template_kernel_class = _GluonTemplateKernelImpl

    return _gluon_template_kernel_class


class GluonTemplateKernel:
    """
    Factory class for creating GluonTemplateKernel instances.

    This class acts as a factory to create instances of the actual GluonTemplateKernel
    implementation, which is created lazily to avoid circular imports.
    """

    def __new__(cls, *args, **kwargs):
        # Get the cached subclass
        kernel_class = _get_gluon_template_kernel_class()
        # Create an instance of the actual subclass
        instance = kernel_class(*args, **kwargs)
        return instance


class GluonTemplate(KernelTemplate):
    """
    Template for Gluon kernels.

    Uses TritonTemplate infrastructure but with GluonTemplateKernel which
    overrides compilation to use Gluon's ASTSource and extended IR builder.
    """

    index_counter = itertools.count()
    all_templates: dict[str, "GluonTemplate"] = {}

    def __init__(
        self,
        name: str,
        grid: Any,
        source: str,
        debug: bool = False,
    ) -> None:
        super().__init__(name, hash=hashlib.sha256(source.encode("utf-8")).hexdigest())
        self.grid = grid
        self.source = source
        self.template = GluonTemplate._template_from_string(source)
        assert name not in self.all_templates, f"duplicate template name: {name}"
        GluonTemplate.all_templates[name] = self
        self.debug = debug

        # Create TritonTemplate with GluonTemplateKernel
        from ...select_algorithm import TritonTemplate

        self._triton_template = TritonTemplate(
            name=self.name,
            grid=self.grid,
            source=self.source,
            debug=self.debug,
        )
        # Override the kernel_type for this specific instance
        self._triton_template.kernel_type = GluonTemplateKernel

    @staticmethod
    @functools.lru_cache(None)  # type: ignore[misc]
    def _template_from_string(source: str) -> Any:  # pyre-fixme[40]
        return KernelTemplate._template_from_string(source)

    @property
    def uid(self) -> str:
        return f"gluon::{self.name}"

    def maybe_append_choice(
        self, choices: list[Any], **kwargs: Any
    ) -> Optional[NotImplementedError]:
        """
        Maybe generates a new ChoiceCaller and appends it into existing choices.
        Returns None if success, otherwise returns the error.
        """
        try:
            choice = self.generate(**kwargs)
            if choice is not None:
                choices.append(choice)
            return None
        except NotImplementedError as e:
            return e
        except Exception as e:
            return NotImplementedError(f"Gluon template failed: {e}")

    def generate(self, **kwargs: Any) -> Optional[ChoiceCaller]:  # type: ignore[override]
        """
        Generate a Gluon kernel choice.

        Uses TritonTemplate infrastructure with GluonTemplateKernel which
        overrides compilation to use Gluon's ASTSource and extended IR builder.
        """
        import torch

        from ...select_algorithm import identity

        # Compute element bitwidth from input dtype
        input_nodes = kwargs.get("input_nodes", ())
        if input_nodes:
            dtype = input_nodes[0].get_dtype()
            element_bitwidth = torch.tensor([], dtype=dtype).element_size() * 8
        else:
            element_bitwidth = 16

        # Extract required positional/named arguments for TritonTemplate.generate
        layout = kwargs.pop("layout")
        num_stages = kwargs.pop("num_stages", 1)
        num_warps = kwargs.pop("num_warps", 4)

        # Remove input_nodes from kwargs since we'll pass it as positional
        kwargs.pop("input_nodes", None)

        # Add element bitwidth to remaining kwargs (these become template variables)
        kwargs["ELEMENT_BITWIDTH"] = element_bitwidth

        # Use TritonTemplate's generate with fusion disabled
        # GluonTemplateKernel will override compilation to use Gluon's compiler
        result = self._triton_template.generate(
            input_nodes=input_nodes,
            layout=layout,
            num_stages=num_stages,
            num_warps=num_warps,
            epilogue_fn=identity,
            subgraphs=None,
            workspace_arg=None,
            **kwargs,
        )
        return result
