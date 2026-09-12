# mypy: allow-untyped-defs
import dataclasses
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from typing import Any, TYPE_CHECKING
from typing_extensions import override

import sympy

import torch
from torch._higher_order_ops.flex_gemm import FlexGemmOpSpec
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.codegen.cutedsl.compile_lock import CUTEDSL_COMPILE_LOCK
from torch._inductor.codegen.cutedsl.cutedsl_kernel import CuteDSLTemplateKernel
from torch._inductor.codegen.cutedsl.cutedsl_template import (
    CuteDSLTemplate,
    CuteDSLTemplateCaller,
)
from torch._inductor.heuristics.template.flex_gemm import QuackConfigKey
from torch._inductor.kernel.flex_gemm.constraints import (
    FlexGemmLocalReduceGeometry,
    FlexGemmOutputContraction,
    LOCAL_REDUCE_COMBINE_NAMES,
    LOCAL_REDUCE_FINALIZE_NAMES,
    LOCAL_REDUCE_PREPASS_FN_SUFFIX,
)
from torch._inductor.kernel.flex_gemm.output_layout import FlexGemmOutputStorageLayout
from torch._inductor.select_algorithm import PartialRender
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from torch._inductor.ir import TensorBox
    from torch._inductor.kernel.flex_gemm.compile_pool import InductorCompilePool
    from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import FlexGemmEpiModSource
    from torch._inductor.kernel.flex_gemm.runtime import FlexGemmRuntimeLocalReducePlan


_BUILTIN_CALLBACKS = LOCAL_REDUCE_COMBINE_NAMES | LOCAL_REDUCE_FINALIZE_NAMES


@dataclasses.dataclass(frozen=True)
class FlexGemmEpilogueBlockScaledConfig:
    """Shared QuACK A/B block-scaled format and template positions of SFA/SFB."""

    format: str
    sfa_index: int
    sfb_index: int


@dataclasses.dataclass(frozen=True)
class FlexGemmEpilogueIndexedOutputConfig:
    """Template input positions for one row-indexed auxiliary output."""

    out_index: int
    indices_index: int


@dataclasses.dataclass(frozen=True)
class FlexGemmEpilogueLocalReduceConfig:
    """Template-time local-reduce metadata; geometry is in physical accumulator columns."""

    geometry: FlexGemmLocalReduceGeometry
    out_index: int | None = None
    output_layout: FlexGemmOutputStorageLayout | None = None
    feeds_main: bool = False
    combine: str | None = None
    finalize: str | None = None
    finalize_operands: tuple[str, ...] = ()
    store_finalize: str | None = None
    binary_store_finalize: bool = False
    prepass_combine: str | None = None
    prepass_finalize: str | None = None
    reduce_planes: int = 1
    fragment_reduced: bool = False

    @classmethod
    def from_plan(
        cls,
        local_reduce: Any | None,
        out_index: int | None,
        source: "FlexGemmEpiModSource",
    ) -> "FlexGemmEpilogueLocalReduceConfig | None":
        """Pair lowering's output-consumer plan with the generated callback names."""
        if local_reduce is None:
            return None
        return FlexGemmEpilogueLocalReduceConfig(
            local_reduce.match.physical_geometry,
            out_index,
            (None if local_reduce.store is None else local_reduce.store.output_layout),
            local_reduce.feeds_main,
            source.local_reduce_combine,
            source.local_reduce_finalize,
            source.local_reduce_finalize_operands,
            source.local_reduce_store_finalize,
            source.local_reduce_binary_store_finalize,
            source.local_reduce_prepass_combine,
            source.local_reduce_prepass_finalize,
            source.local_reduce_planes,
            source.local_reduce_fragment_reduced,
        )

    def runtime_plan(
        self, resolve: Callable[[str], Any], epilogue_name: str
    ) -> "FlexGemmRuntimeLocalReducePlan":
        """The runtime plan (without its ``out`` buffer) with generated callback
        names resolved through ``resolve``; built-in names pass through."""
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmRuntimeLocalReducePlan,
        )

        def callback(name: str | None) -> Any:
            return name if name is None or name in _BUILTIN_CALLBACKS else resolve(name)

        prepass_name = f"{epilogue_name}{LOCAL_REDUCE_PREPASS_FN_SUFFIX}"
        return FlexGemmRuntimeLocalReducePlan(
            self.geometry,
            stores=self.out_index is not None,
            feeds_main=self.feeds_main,
            combine=callback(self.combine),
            finalize=callback(self.finalize),
            finalize_operands=self.finalize_operands,
            reduce_planes=self.reduce_planes,
            fragment_reduced=self.fragment_reduced,
            store_finalize=callback(self.store_finalize),
            binary_store_finalize=self.binary_store_finalize,
            prepass=None if self.prepass_combine is None else resolve(prepass_name),
            prepass_combine=self.prepass_combine,
            prepass_finalize=callback(self.prepass_finalize),
            output_layout=self.output_layout,
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
        blockscaled: Shared block-scaled format and SFA/SFB input positions.
        quack_config: Exact QuACK GemmConfig fields pinned for this choice;
            None only before lowering has selected the candidates.
        epilogue_arg_indices: Template input indices for read-only epilogue captures.
        epilogue_arg_kinds: Broadcast kind for each captured epilogue tensor.
        aux_out_indices: Template input indices for same-shape aux outputs.
        indexed_output: Runtime input positions for one indexed auxiliary output.
        local_reduce: Concrete local-reduce consumer rendered into runtime kwargs.
    """

    epilogue_name: str
    epilogue_source: str
    gemm_op: FlexGemmOpSpec
    alpha: float
    beta: float
    blockscaled: FlexGemmEpilogueBlockScaledConfig | None
    quack_config: QuackConfigKey | None
    epilogue_arg_indices: tuple[int, ...]
    epilogue_arg_kinds: tuple[str, ...]
    aux_out_indices: tuple[int, ...]
    indexed_output: FlexGemmEpilogueIndexedOutputConfig | None
    local_reduce: FlexGemmEpilogueLocalReduceConfig | None
    output_contraction: FlexGemmOutputContraction | None

    def epimod(
        self,
        epilogue_fn: Any,
        input_dtypes: Sequence[torch.dtype],
        resolve: Callable[[str], Any],
    ) -> Any:
        """Build the QuACK EpiMod the generated ``_main`` builds at runtime.

        ``input_dtypes`` are the template inputs' dtypes and ``resolve`` maps a
        generated local-reduce callback name to its callable. Lowering passes
        the selection stub for both; compile workers pass the loaded module.
        """
        from torch._inductor.kernel.flex_gemm.runtime import (
            flex_gemm_epimod,
            quack_epilogue_dtype,
        )

        indexed = self.indexed_output
        return flex_gemm_epimod(
            epilogue_fn,
            tuple(
                quack_epilogue_dtype(input_dtypes[i]) for i in self.epilogue_arg_indices
            ),
            self.epilogue_arg_kinds,
            len(self.aux_out_indices),
            None
            if indexed is None
            else (
                quack_epilogue_dtype(input_dtypes[indexed.out_index]),
                input_dtypes[indexed.indices_index],
            ),
            None
            if self.local_reduce is None
            else self.local_reduce.runtime_plan(resolve, self.epilogue_name),
            self.output_contraction,
        )


class FlexGemmEpilogueKernel(CuteDSLTemplateKernel):
    """Render generated FlexGEMM epilogue modules with a compile-only hook."""

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
        for arg_def in arg_defs:
            if arg_def.full_name() not in self._seen_input_args:
                params.append(arg_def.full_name())
        params.append("stream")

        template_input_arg_names = [
            arg_name for arg_name, _ in self._template_input_args
        ]
        call_args, call_kwargs = self._gemm_call_args(template_input_arg_names, config)
        call_kwargs += self._epilogue_kwargs(template_input_arg_names, config)
        call_kwargs += f", out={self.get_output()}, stream=stream"

        code = IndentedBuffer()
        code.splice(
            """
            import torch
            from torch._inductor.kernel.flex_gemm.constraints import (
                FlexGemmOutputContraction,
                FlexGemmLocalReduceGeometry,
            )
            from torch._inductor.kernel.flex_gemm import (
                output_layout as flex_gemm_output_layout,
            )
            from torch._inductor.kernel.flex_gemm.runtime import (
                FlexGemmRuntimeLocalReducePlan,
                gemm_epilogue as flex_gemm_epilogue,
            )
            """
        )
        code.splice(config.epilogue_source)
        # QuACK fingerprints the epilogue by (module, qualname, source); the
        # benchmark and final wrapper modules differ, so give the generated
        # functions a stable module or the selected kernel compiles twice.
        code.splice(
            f"""
            for _fn in list(globals().values()):
                if getattr(_fn, "__module__", None) == __name__ and callable(_fn):
                    _fn.__module__ = "torch._inductor.kernel.flex_gemm.{config.epilogue_name}"
            """
        )
        code.splice(
            f"""
            def {self.kernel_name}_main({", ".join(params)}):
                flex_gemm_epilogue({", ".join((*call_args, config.epilogue_name))}{call_kwargs})

            def {self.kernel_name}_precompile(**metadata):
                # The template caller compiles each choice's pinned QuACK
                # kernel itself (see FlexGemmEpilogueCaller.precompile).
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
        """Render a built-in callback name or generated callable reference."""
        return repr(name) if name in _BUILTIN_CALLBACKS else name

    def _local_reduce_geometry(
        self, local_reduce: FlexGemmEpilogueLocalReduceConfig
    ) -> str:
        """Render the shared grouped M/N local-reduce geometry."""
        geometry = local_reduce.geometry
        return (
            "FlexGemmLocalReduceGeometry("
            f"group={geometry.group!r}, axis={geometry.axis!r})"
        )

    def _local_reduce_kwargs(
        self,
        input_args: list[str],
        local_reduce: FlexGemmEpilogueLocalReduceConfig,
        epilogue_name: str,
    ) -> str:
        """Render one structural local-reduce plan for runtime dispatch."""
        geometry = self._local_reduce_geometry(local_reduce)
        plan = f"FlexGemmRuntimeLocalReducePlan({geometry}"
        if local_reduce.out_index is not None:
            plan += f", stores=True, out={input_args[local_reduce.out_index]}"
        if local_reduce.output_layout is not None:
            plan += f", output_layout={local_reduce.output_layout.codegen_reference()}"
        if local_reduce.feeds_main:
            plan += ", feeds_main=True"
        if local_reduce.combine is None:
            raise RuntimeError("FlexGEMM EpiMod local reductions require a combine")
        plan += f", combine={self._callback_reference(local_reduce.combine)}"
        if local_reduce.reduce_planes != 1:
            plan += f", reduce_planes={local_reduce.reduce_planes}"
        if local_reduce.fragment_reduced:
            plan += ", fragment_reduced=True"
        if local_reduce.finalize is not None:
            plan += f", finalize={self._callback_reference(local_reduce.finalize)}"
        if local_reduce.finalize_operands:
            plan += f", finalize_operands={local_reduce.finalize_operands!r}"
        if local_reduce.store_finalize is not None:
            plan += (
                ", store_finalize="
                f"{self._callback_reference(local_reduce.store_finalize)}"
            )
        if local_reduce.binary_store_finalize:
            plan += ", binary_store_finalize=True"
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
        if config.quack_config is None:
            raise AssertionError("rendered FlexGEMM choices require a pinned config")
        kwargs = [f", config={config.quack_config!r}"]
        if config.blockscaled is not None:
            kwargs.append(
                f", SFA={input_args[config.blockscaled.sfa_index]}, "
                f"SFB={input_args[config.blockscaled.sfb_index]}, "
                f"blockscaled_format={config.blockscaled.format!r}"
            )
        if epilogue_args:
            kwargs.append(
                f", epilogue_args=({', '.join(epilogue_args)},), "
                f"epilogue_arg_kinds={config.epilogue_arg_kinds!r}"
            )
        if config.aux_out_indices:
            aux_outs = ", ".join(input_args[index] for index in config.aux_out_indices)
            kwargs.append(f", aux_outs=({aux_outs},)")
        if config.indexed_output is not None:
            kwargs.append(
                f", indexed_out={input_args[config.indexed_output.out_index]}, "
                f"indexed_indices={input_args[config.indexed_output.indices_index]}"
            )
        if config.local_reduce is not None:
            kwargs.append(
                self._local_reduce_kwargs(
                    input_args, config.local_reduce, config.epilogue_name
                )
            )
        if config.output_contraction is not None:
            kwargs.append(f", output_contraction={config.output_contraction!r}")
        return "".join(kwargs)


class FlexGemmEpilogueCaller(CuteDSLTemplateCaller):
    def __init__(self, *args: Any, template_kwargs: dict[str, Any], **kwargs: Any):
        super().__init__(*args, template_kwargs=template_kwargs, **kwargs)
        self.config: FlexGemmEpilogueConfig = template_kwargs["config"]

    @override
    def output_node(self) -> "TensorBox":
        """Guard the problem-size rules QuACK applied to the selected config.

        Selection pruned on concrete shape hints; these are the pruning
        conditions that depend on physical N, so a dynamic graph recompiles
        instead of launching a config QuACK would have rejected (``swap_ab``
        needs ``N % 8 == 0``; ``GroupedMainStore.supports_problem`` needs
        ``tile_n <= N``). Only the selected choice is guarded, in both the
        direct and multi-template paths.
        """
        from torch._inductor.virtualized import V

        config = self.config
        if config.quack_config is None:
            raise AssertionError("selected FlexGEMM choice has no pinned config")
        fields = dict(config.quack_config)
        n = self.input_nodes[config.gemm_op.mat2_index].get_size()[-1]
        sizevars = V.graph.sizevars
        if fields["swap_ab"]:
            sizevars.check(sympy.Eq(sympy.Mod(n, 8), 0))
        if config.output_contraction is not None:
            sizevars.check_leq(fields["tile_n"], n)
        return super().output_node()

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

    def precompile(self, *, use_workers: bool = True) -> None:
        """Compile this choice's pinned QuACK kernel, in a compile worker if any.

        Running the kernel once with a compile pool active ships the cold
        ``jit_cache`` miss to an Inductor compile worker (NOTE [FlexGEMM compile
        workers]). Without workers, or with ``use_workers=False`` for a lone
        choice that would only pay the worker's cold import, it compiles
        in-process.
        """
        from torch._inductor.kernel.flex_gemm import compile_pool

        if use_workers and compile_pool.workers_ready():
            bmreq = self.bmreq
            pool = compile_pool.InductorCompilePool(
                compile_pool.FlexGemmCompileRecipe(
                    bmreq.module_cache_key,
                    bmreq.module_path,
                    self.config,
                    tuple(meta.dtype for meta in bmreq.input_tensor_meta),
                )
            )
            sha = self._run_once(pool)
            if sha is None or pool.wait(sha):
                return
        self._run_once(None)

    def _run_once(self, pool: "InductorCompilePool | None") -> str | None:
        """Run the kernel on fresh buffers under the compile lock.

        Inductor precompiles from several threads and CuTeDSL compilation is
        not thread-safe. Buffers live only inside this call so a thread
        waiting on the lock or on a worker holds none. Returns the pending
        ``jit_cache`` key when ``pool`` took the compile, else None.
        """
        from torch._vendor.quack.cache.async_compile import CompilePending, pool_active

        bmreq = self.bmreq
        with CUTEDSL_COMPILE_LOCK:
            inputs = [meta.to_tensor() for meta in bmreq.input_tensor_meta]
            out = bmreq.output_tensor_meta.to_tensor()
            run = bmreq.make_run_fn(*inputs, out=out)
            try:
                with pool_active(pool) if pool is not None else nullcontext():
                    run()
            except CompilePending as pending:
                return pending.sha
        return None


class FlexGemmEpilogueTemplate(CuteDSLTemplate):
    kernel_type = FlexGemmEpilogueKernel
    caller_type = FlexGemmEpilogueCaller

    def __init__(self) -> None:
        super().__init__("flex_gemm_epilogue", source="")


flex_gemm_epilogue_template = FlexGemmEpilogueTemplate()
