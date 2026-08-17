# mypy: allow-untyped-defs
"""
Gluon template support for PyTorch Inductor.

Gluon is a DSL from the Triton project for writing Blackwell GPU kernels.
Since Gluon uses the same compilation infrastructure as Triton, this
implementation is kept minimal and reuses Triton's compilation pipeline.
"""

import functools
import hashlib
import logging
from typing import Any

from ...ir import ChoiceCaller
from ..common import KernelTemplate


log = logging.getLogger(__name__)


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

            def _jit_decorator(self):
                return "@gluon.jit"

            def _constexpr(self):
                return "gl.constexpr"

            def _index_dtype_expr(self, dtype: str):
                return _gluon_index_dtype(dtype)

            def jit_lines(self):
                import textwrap

                parent_jit_lines = super().jit_lines()
                parent_jit_lines = textwrap.dedent(parent_jit_lines)
                return (
                    """
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

"""
                    + parent_jit_lines
                )

            def size(self, name: str | None, index: int):
                return self._size_expr(name, index)

            def stride(self, name, index=None):
                return self._stride_expr(name, index)

            def output_stride(self, index):
                return self._output_stride_expr(index)

            def _add_layout_kwargs(self, kwargs):
                import torch

                input_nodes = getattr(self, "input_nodes", [])
                if input_nodes:
                    input_torch_dtype = input_nodes[0].get_dtype()
                else:
                    input_torch_dtype = torch.bfloat16
                output_torch_dtype = self.output_node.get_dtype()  # type: ignore[attr-defined]

                def torch_dtype_to_gluon_str(dtype):
                    if dtype == torch.float8_e5m2:
                        return "gl.float8e5"
                    elif dtype == torch.float8_e4m3fn:
                        return "gl.float8e4nv"
                    else:
                        return f"gl.{str(dtype).split('.')[1]}"

                kwargs["INPUT_DTYPE"] = torch_dtype_to_gluon_str(input_torch_dtype)
                kwargs["OUTPUT_DTYPE"] = torch_dtype_to_gluon_str(output_torch_dtype)
                kwargs["INDEX_DTYPE_EXPR"] = _gluon_index_dtype(self.index_dtype)

                # Flip to True manually (no config knob) to enable Proton
                # profiling scopes inside the generated kernel.
                kwargs["ENABLE_PROTON_PROFILING"] = False

                return kwargs

            def render(
                self, template, kwargs, record_input_dependent_tracked_event=False
            ):
                kwargs = self._add_layout_kwargs(kwargs)

                # Call parent render with updated kwargs
                return super().render(
                    template, kwargs, record_input_dependent_tracked_event
                )

        _gluon_template_kernel_class = _GluonTemplateKernelImpl

    return _gluon_template_kernel_class


def _gluon_index_dtype(dtype: str):
    return {
        "tl.int32": "gl.int32",
        "tl.int64": "gl.int64",
    }[dtype]


# Cache for the GluonTritonTemplate subclass
_gluon_triton_template_class = None


def _get_gluon_triton_template_class():
    """
    Lazily create and cache the TritonTemplate subclass that drives Gluon
    codegen. This avoids circular imports while ensuring proper inheritance.
    """
    global _gluon_triton_template_class
    if _gluon_triton_template_class is None:
        from ...select_algorithm import TritonTemplate

        class GluonTritonTemplate(TritonTemplate):
            def _constexpr(self):
                return "gl.constexpr"

            def _index_dtype_expr(self, dtype: str):
                return _gluon_index_dtype(dtype)

        _gluon_triton_template_class = GluonTritonTemplate

    return _gluon_triton_template_class


class GluonTemplate(KernelTemplate):
    """
    Template for Gluon kernels.

    Uses TritonTemplate infrastructure but with GluonTemplateKernel which
    overrides compilation to use Gluon's ASTSource and extended IR builder.
    """

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
        assert name not in self.all_templates, f"duplicate template name: {name}"  # noqa: S101
        GluonTemplate.all_templates[name] = self
        self.debug = debug

        self._triton_template = _get_gluon_triton_template_class()(
            name=self.name,
            grid=self.grid,
            source=self.source,
            debug=self.debug,
            kernel_name_prefix="",
        )
        # Override the kernel_type for this specific instance
        self._triton_template.kernel_type = _get_gluon_template_kernel_class()

    @staticmethod
    @functools.lru_cache(None)  # type: ignore[misc]
    def _template_from_string(source: str) -> Any:  # pyre-fixme[40]
        return KernelTemplate._template_from_string(source)

    @property
    def uid(self) -> str:
        return f"gluon::{self.name}"

    def maybe_append_choice(
        self, choices: list[Any], **kwargs: Any
    ) -> NotImplementedError | None:
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
            log.info(
                "Cannot Append Choice: %s. KernelTemplate type is %s",
                e,
                type(self),
                stack_info=log.getEffectiveLevel() < logging.INFO,
            )
            return e

    def generate(self, **kwargs: Any) -> ChoiceCaller | None:  # type: ignore[override]
        """
        Generate a Gluon kernel choice.

        Uses TritonTemplate infrastructure with GluonTemplateKernel which
        overrides compilation to use Gluon's ASTSource and extended IR builder.
        """
        from ...select_algorithm import identity

        input_nodes = kwargs.get("input_nodes", ())

        # Extract required positional/named arguments for TritonTemplate.generate
        layout = kwargs.pop("layout")
        num_stages = kwargs.pop("num_stages", 1)
        num_warps = kwargs.pop("num_warps", 4)

        # Remove input_nodes from kwargs since we'll pass it as positional
        kwargs.pop("input_nodes", None)

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
        if result is not None:
            result.log_info["backend"] = "Gluon"
        return result
