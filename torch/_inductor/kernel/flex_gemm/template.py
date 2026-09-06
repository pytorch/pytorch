# mypy: allow-untyped-defs
import dataclasses
from typing import Any
from typing_extensions import override

from torch._higher_order_ops.flex_gemm import FlexGemmOpSpec
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.codegen.cutedsl.cutedsl_kernel import CuteDSLTemplateKernel
from torch._inductor.codegen.cutedsl.cutedsl_template import (
    CuteDSLTemplate,
    CuteDSLTemplateCaller,
)
from torch._inductor.kernel.flex_gemm.constraints import (
    FlexGemmGroupedMainOutputTransform,
    FlexGemmLocalReduceGeometry,
    LOCAL_REDUCE_PREPASS_FN_SUFFIX,
)
from torch._inductor.kernel.flex_gemm.output_layout import FlexGemmOutputLayout
from torch._inductor.select_algorithm import PartialRender
from torch.utils._ordered_set import OrderedSet


@dataclasses.dataclass(frozen=True)
class FlexGemmEpilogueLocalReduceConfig:
    """Template-time local-reduce metadata for output and/or feed-main consumers."""

    geometry: FlexGemmLocalReduceGeometry
    out_index: int | None = None
    output_layout: FlexGemmOutputLayout | None = None
    feeds_main: bool = False
    combine: str | None = None
    finalize: str | None = None
    store_finalize: str | None = None
    prepass_combine: str | None = None
    prepass_finalize: str | None = None

    @classmethod
    def from_output_plan(
        cls,
        local_reduce: Any | None,
        out_index: int | None,
        *,
        combine: str | None = None,
        finalize: str | None = None,
        store_finalize: str | None = None,
        prepass_combine: str | None = None,
        prepass_finalize: str | None = None,
    ) -> "FlexGemmEpilogueLocalReduceConfig | None":
        """Translate lowering's output-consumer plan into template metadata."""
        if local_reduce is None:
            return None
        return FlexGemmEpilogueLocalReduceConfig(
            local_reduce.match.geometry,
            out_index,
            (None if local_reduce.store is None else local_reduce.store.output_layout),
            local_reduce.feeds_main,
            combine,
            finalize,
            store_finalize,
            prepass_combine,
            prepass_finalize,
        )


@dataclasses.dataclass(frozen=True)
class FlexGemmEpilogueConfig:
    """Metadata needed to render one Inductor-owned QuACK GEMM epilogue choice.

    Attributes:
        epilogue_name: Name of the generated CuTeDSL epilogue callable.
        epilogue_source: Python source that defines ``epilogue_name``.
        gemm_op: Original aten GEMM op spec used to map inputs into QuACK.
        alpha: Static alpha multiplier for addmm/baddbmm inputs.
        beta: Static beta multiplier for addmm/baddbmm bias inputs.
        quack_config_constraints: Optional native QuACK config field constraints.
        quack_config: Exact QuACK GemmConfig fields pinned for this choice, or
            None to take QuACK's untuned default within the constraints.
        epilogue_arg_indices: Template input indices for read-only epilogue captures.
        epilogue_arg_kinds: Broadcast kind for each captured epilogue tensor.
        aux_out_indices: Template input indices for same-shape aux outputs.
        local_reduce: Concrete local-reduce consumer rendered into runtime kwargs.
    """

    epilogue_name: str
    epilogue_source: str
    gemm_op: FlexGemmOpSpec
    alpha: float
    beta: float
    quack_config_constraints: tuple[tuple[str, Any], ...]
    quack_config: tuple[tuple[str, Any], ...] | None
    epilogue_arg_indices: tuple[int, ...]
    epilogue_arg_kinds: tuple[str, ...]
    aux_out_indices: tuple[int, ...]
    local_reduce: FlexGemmEpilogueLocalReduceConfig | None
    main_transform: FlexGemmGroupedMainOutputTransform | None


class FlexGemmEpilogueKernel(CuteDSLTemplateKernel):
    """Render one generated FlexGEMM EpiMod wrapper."""

    @override
    def render(self, template, **kwargs):
        config = kwargs.pop("config")
        if kwargs:
            raise RuntimeError(f"unexpected FlexGEMM epilogue options: {kwargs}")

        self._template_input_args = []
        self._seen_input_args = OrderedSet()
        for index, input_node in enumerate(self.input_nodes):
            buf_name = input_node.get_name()
            self.args.input(buf_name)
            arg_name = f"arg{index}"
            self.args.input_buffers[buf_name] = arg_name
            self._template_input_args.append((arg_name, input_node))
            self._seen_input_args.add(arg_name)

        self.args.output(self.output_node.get_name())
        arg_defs, _, _, _ = self.args.python_argdefs()
        params = [arg_name for arg_name, _ in self._template_input_args]
        params.extend(
            arg_def.full_name()
            for arg_def in arg_defs
            if arg_def.full_name() not in self._seen_input_args
        )
        params.append("stream")

        template_input_arg_names = [
            arg_name for arg_name, _ in self._template_input_args
        ]
        call_args, call_kwargs = self._gemm_call_args(template_input_arg_names, config)
        call_kwargs += self._epilogue_kwargs(template_input_arg_names, config)
        call_kwargs += f", out={self.get_output()}, stream=stream"

        code = IndentedBuffer()
        code.writeline("import torch")
        code.splice(
            """
            from torch._inductor.kernel.flex_gemm.constraints import (
                FlexGemmGroupedMainOutputTransform,
                FlexGemmLocalReduceGeometry,
            )
            from torch._inductor.kernel.flex_gemm import (
                output_layout as flex_gemm_output_layout,
            )
            from torch._inductor.kernel.flex_gemm.runtime import (
                FlexGemmEpiModLocalReducePlan,
                gemm_epimod as flex_gemm_runtime,
            )
            """
        )
        code.splice(config.epilogue_source)
        code.splice(
            f"""
            def {self.kernel_name}_main({", ".join(params)}):
                flex_gemm_runtime({", ".join((*call_args, config.epilogue_name))}{call_kwargs})

            def {self.kernel_name}_precompile(**metadata):
                # Compile workers cannot initialize CUDA; the template caller
                # precompiles each choice's pinned QuACK kernel from the parent
                # through Inductor's pool instead (see FlexGemmEpilogueCaller).
                pass
            """
        )
        return PartialRender(code.getvalue(), self.render_hooks)

    def _gemm_call_args(
        self, input_args: list[str], config: FlexGemmEpilogueConfig
    ) -> tuple[list[str], str]:
        """Return positional GEMM operands and scalar/bias kwargs for runtime dispatch."""
        op = config.gemm_op
        call_args = [input_args[op.mat1_index], input_args[op.mat2_index]]
        if op.bias_index is None:
            return call_args, ""
        return call_args, (
            f", C={input_args[op.bias_index]}, alpha={config.alpha!r}, beta={config.beta!r}"
        )

    @staticmethod
    def _callback_reference(name: str) -> str:
        """Render a built-in finalizer name or generated callable reference."""
        return repr(name) if name == "mean" else name

    def _local_reduce_kwargs(
        self,
        input_args: list[str],
        local_reduce: FlexGemmEpilogueLocalReduceConfig,
        epilogue_name: str,
    ) -> str:
        """Render one structural local-reduce plan for runtime dispatch."""
        plan = f"FlexGemmEpiModLocalReducePlan({local_reduce.geometry!r}"
        if local_reduce.out_index is not None:
            plan += f", out={input_args[local_reduce.out_index]}"
        if local_reduce.output_layout is not None:
            plan += f", output_layout={local_reduce.output_layout.codegen_reference()}"
        if local_reduce.feeds_main:
            plan += ", feeds_main=True"
        plan += f", combine={local_reduce.combine!r}"
        if local_reduce.finalize is not None:
            plan += f", finalize={self._callback_reference(local_reduce.finalize)}"
        if local_reduce.store_finalize is not None:
            plan += (
                ", store_finalize="
                f"{self._callback_reference(local_reduce.store_finalize)}"
            )
        if local_reduce.prepass_combine is not None:
            plan += (
                f", prepass={epilogue_name}{LOCAL_REDUCE_PREPASS_FN_SUFFIX}, "
                f"prepass_combine={local_reduce.prepass_combine!r}"
            )
        if local_reduce.prepass_finalize is not None:
            plan += (
                ", prepass_finalize="
                f"{self._callback_reference(local_reduce.prepass_finalize)}"
            )
        return f", local_reduce={plan})"

    def _epilogue_kwargs(
        self, input_args: list[str], config: FlexGemmEpilogueConfig
    ) -> str:
        """Render captured tensor and aux-output kwargs for runtime dispatch."""
        epilogue_args = [input_args[index] for index in config.epilogue_arg_indices]
        kwargs = []
        if config.quack_config is not None:
            kwargs.append(f", config={config.quack_config!r}")
        if config.quack_config_constraints:
            kwargs.append(f", config_constraints={config.quack_config_constraints!r}")
        if epilogue_args:
            kwargs.append(
                f", epilogue_args=({', '.join(epilogue_args)},), "
                f"epilogue_arg_kinds={config.epilogue_arg_kinds!r}"
            )
        if config.aux_out_indices:
            aux_outs = ", ".join(input_args[index] for index in config.aux_out_indices)
            kwargs.append(f", aux_outs=({aux_outs},)")
        if config.local_reduce is not None:
            kwargs.append(
                self._local_reduce_kwargs(
                    input_args, config.local_reduce, config.epilogue_name
                )
            )
        if config.main_transform is not None:
            kwargs.append(f", main_transform={config.main_transform!r}")
        return "".join(kwargs)


class FlexGemmEpilogueCaller(CuteDSLTemplateCaller):
    @override
    def _build_description(
        self, name: str, template_kwargs: dict[str, Any] | None
    ) -> str:
        if template_kwargs is None:
            raise AssertionError("FlexGEMM template kwargs must include a config")
        config = template_kwargs["config"]
        quack_config = config.quack_config
        description = "default" if quack_config is None else dict(quack_config)
        return f"CuteDSL template {name} (QUACK config={description})"

    def precompile(self, *, wait: bool = True) -> None:
        """Compile this choice's pinned QuACK kernel through Inductor's worker pool."""
        from torch._inductor.async_compile import AsyncCompile
        from torch._inductor.kernel.flex_gemm.runtime import precompile_flex_gemm_kernel

        if not AsyncCompile.wait_process_pool_ready():
            return
        inputs = [meta.to_tensor() for meta in self.bmreq.input_tensor_meta]
        run = self.bmreq.make_run_fn(
            *inputs, out=self.bmreq.output_tensor_meta.to_tensor()
        )
        precompile_flex_gemm_kernel(run, wait=wait)


class FlexGemmEpilogueTemplate(CuteDSLTemplate):
    kernel_type = FlexGemmEpilogueKernel
    caller_type = FlexGemmEpilogueCaller

    def __init__(self) -> None:
        super().__init__("flex_gemm_epilogue", source="")


flex_gemm_epilogue_template = FlexGemmEpilogueTemplate()
