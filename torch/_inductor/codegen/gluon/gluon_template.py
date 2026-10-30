# mypy: allow-untyped-defs
"""
Gluon template support for PyTorch Inductor.

Gluon is a DSL from the Triton project for writing Blackwell GPU
kernels. It shares Triton's compilation infrastructure, so this
implementation reuses Triton's pipeline.
"""

import functools
import hashlib
import logging
import re
from typing import Any

from ...ir import ChoiceCaller
from ..common import KernelTemplate


log = logging.getLogger(__name__)

# The tl.* creation ops fail in Gluon kernels (no default layout); the gl.* ones use AutoLayout.
_TL_CREATION_OPS = re.compile(r"\btl\.(full|arange|zeros|zeros_like)\(")


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
        from torch.utils._ordered_set import OrderedSet

        from ...utils import can_use_tma, triton_type
        from ...virtualized import V
        from ..triton import texpr, TritonKernelOverrides

        TritonTemplateKernel = _get_triton_kernel_class()

        class _TmaStoreCapture(V.WrapperHandler):  # type: ignore[name-defined]
            def __init__(self, inner, tma_store_vars):
                super().__init__(inner)
                self.tma_store_vars = tma_store_vars

            def store(self, name, index, value, mode=None):
                var = self.tma_store_vars.get(name)
                if var is None:
                    return self._inner.store(name, index, value, mode)
                V.kernel.store_buffer_names.add(name)
                V.kernel.cse.store_cache[name] = value
                dtype = triton_type(V.graph.get_dtype(name))
                V.kernel.compute.writeline(f"{var} = {value}.to({dtype})")

        class _GluonKernelOverrides(TritonKernelOverrides):
            # Gluon broadcasting needs equal ranks; scalars broadcast to any rank.
            @classmethod
            def constant(cls, value, dtype):
                return cls._shaped_constant(value, dtype, shape=[])

        class _GluonTemplateKernelImpl(TritonTemplateKernel):
            """
            Kernel class for Gluon templates.

            Overrides jit_lines() to emit the @gluon.jit decorator
            with the necessary imports.
            """

            overrides = _GluonKernelOverrides  # type: ignore[assignment]
            # Gluon can't broadcast a scalar to a shape without a layout.
            broadcast_scalar_loads = False
            _tma_store_vars: dict[str, str] | None = None

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

            def _find_tma_store_targets(self, epilogue_nodes):
                scheduler = V.graph.scheduler
                sizevars = V.graph.sizevars
                output = self.output_node
                names = [output.get_name()]
                names += [
                    name for node in epilogue_nodes for name in node.get_buffer_names()
                ]
                fused_ops = OrderedSet(
                    scheduler.name_to_buf[name].defining_op_name() for name in names
                )
                targets = []
                for name in names:
                    if scheduler.can_buffer_be_removed_through_fusion(name, fused_ops):
                        continue
                    buf = V.graph.get_buffer(name)
                    itemsize = buf.get_dtype().itemsize
                    # can_use_tma checks unguarded size hints; TMA needs this for every symbol value.
                    aligned = all(
                        sizevars.statically_known_multiple_of(stride * itemsize, 16)
                        for stride in buf.get_stride()[:-1]
                    )
                    # The staging buffer is sized for the output dtype, so wider dtypes don't fit.
                    if (
                        not buf.get_mutation_names()
                        and sizevars.statically_known_list_equals(
                            buf.get_size(), output.get_size()
                        )
                        and sizevars.statically_known_equals(buf.get_stride()[-1], 1)
                        and itemsize <= output.get_dtype().itemsize
                        and aligned
                        and can_use_tma(output_layout=buf.get_layout())
                    ):
                        targets.append(name)
                return targets

            def store_output(self, *args, **kwargs):
                if not self._tma_store_vars:
                    return super().store_output(*args, **kwargs)
                capture = functools.partial(
                    _TmaStoreCapture, tma_store_vars=self._tma_store_vars
                )
                # Also captures the template output's own store.
                with V.set_ops_handler(capture(V.get_ops_handler())):
                    subgraph_name = super().store_output(*args, **kwargs)
                self.subgraph_bodies[subgraph_name].ops_handler = capture
                return subgraph_name

            def _make_codegen_hook(self, subgraph_name, indent_width=0):
                hook = super()._make_codegen_hook(subgraph_name, indent_width)
                return lambda: _TL_CREATION_OPS.sub(r"gl.\1(", hook())

            def _add_layout_kwargs(self, kwargs):
                import torch

                input_nodes = getattr(self, "input_nodes", [])
                if input_nodes:
                    input_torch_dtype = input_nodes[0].get_dtype()
                else:
                    input_torch_dtype = torch.bfloat16
                output_torch_dtype = self.output_node.get_dtype()  # type: ignore[attr-defined]

                kwargs["INPUT_DTYPE"] = _gluon_dtype(input_torch_dtype)
                kwargs["OUTPUT_DTYPE"] = _gluon_dtype(output_torch_dtype)
                kwargs["INDEX_DTYPE_EXPR"] = _gluon_index_dtype(self.index_dtype)

                from torch._inductor import config

                kwargs["ENABLE_PROTON_PROFILING"] = config.gluon_enable_proton_profiling

                return kwargs

            def _epilogue_fusion_kwargs(self, use_tma_store):
                epilogue_nodes_by_subgraph = getattr(
                    self, "_epilogue_nodes_by_subgraph", None
                )
                epilogue_nodes = (
                    epilogue_nodes_by_subgraph[0]
                    if epilogue_nodes_by_subgraph is not None
                    else []
                )
                has_epilogue_fusion = bool(epilogue_nodes)
                targets = (
                    self._find_tma_store_targets(epilogue_nodes)
                    if has_epilogue_fusion and use_tma_store
                    else []
                )
                self._tma_store_vars = {name: f"c{i}" for i, name in enumerate(targets)}
                tma_store_targets = []
                for name, var in self._tma_store_vars.items():
                    buf = V.graph.get_buffer(name)
                    strides = [texpr(self.rename_indexing(s)) for s in buf.get_stride()]
                    tma_store_targets.append(
                        {
                            "var": var,
                            "ptr": self.args.output(name),
                            "dtype": _gluon_dtype(buf.get_dtype()),
                            "strides": strides,
                        }
                    )
                return {
                    "HAS_EPILOGUE_FUSION": has_epilogue_fusion,
                    "TMA_STORE_TARGETS": tma_store_targets,
                }

            def render(
                self, template, kwargs, record_input_dependent_tracked_event=False
            ):
                kwargs = self._add_layout_kwargs(kwargs)
                fusion_kwargs = self._epilogue_fusion_kwargs(
                    kwargs.get("USE_TMA_STORE", False)
                )
                return super().render(
                    template,
                    kwargs | fusion_kwargs,
                    record_input_dependent_tracked_event,
                )

        _gluon_template_kernel_class = _GluonTemplateKernelImpl

    return _gluon_template_kernel_class


def _gluon_dtype(dtype):
    from ...utils import triton_type

    return triton_type(dtype).replace("tl.", "gl.", 1)


def _gluon_index_dtype(dtype: str):
    return {
        "tl.int32": "gl.int32",
        "tl.int64": "gl.int64",
    }[dtype]


# Cache for the GluonTritonTemplate subclass
_gluon_triton_template_class = None


def _get_gluon_triton_template_class():
    """
    Lazily create and cache the TritonTemplate subclass driving Gluon
    codegen, avoiding circular imports.
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

    Uses TritonTemplate infrastructure, but GluonTemplateKernel
    compiles via Gluon's ASTSource and extended IR builder.
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
        Maybe generate a ChoiceCaller and append it to choices.
        Returns None on success, otherwise the error.
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

        Uses TritonTemplate infrastructure, but GluonTemplateKernel
        compiles via Gluon's ASTSource and extended IR builder.
        """
        from ...select_algorithm import identity

        input_nodes = kwargs.get("input_nodes", ())

        # Arguments TritonTemplate.generate takes positionally.
        layout = kwargs.pop("layout")
        num_stages = kwargs.pop("num_stages", 1)
        num_warps = kwargs.pop("num_warps", 4)

        # Passed positionally below.
        kwargs.pop("input_nodes", None)

        # Fusion is disabled; GluonTemplateKernel overrides compilation
        # to use Gluon's compiler.
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
