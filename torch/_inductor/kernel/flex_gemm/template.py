# mypy: allow-untyped-defs
import dataclasses
import logging
from typing import Any
from typing_extensions import override

import torch
from torch._dynamo.utils import counters
from torch._higher_order_ops.flex_gemm import FlexGemmOpSpec
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.codegen.cutedsl.cutedsl_kernel import CuteDSLTemplateKernel
from torch._inductor.codegen.cutedsl.cutedsl_template import (
    CuteDSLTemplate,
    CuteDSLTemplateCaller,
)
from torch._inductor.kernel.flex_gemm.runtime import inductor_quack_cache_dir
from torch._inductor.select_algorithm import PartialRender
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class FlexGemmEpilogueConfig:
    """Metadata needed to render one Inductor-owned QuACK GEMM epilogue choice.

    The epilogue fields identify the generated CuTeDSL callable, ``gemm_op``
    describes how to map the original aten op's operands into QuACK's dense GEMM
    adapter, and ``quack_config_key`` pins the exact QuACK config selected by
    Inductor autotuning or default selection.
    """

    epilogue_name: str
    epilogue_source: str
    gemm_op: FlexGemmOpSpec
    alpha: float
    beta: float
    out_dtype: Any | None = None
    quack_config_key: tuple[Any, ...] | None = None


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
        params.append("device_capacity_override=None")
        quack_cache_dir_param = f"quack_cache_dir={inductor_quack_cache_dir()!r}"
        params.append(quack_cache_dir_param)

        call_args, call_kwargs = self._gemm_call_args(
            [arg_name for arg_name, _ in self._template_input_args], config
        )
        call_kwargs += (
            f", out={self.get_output()}, "
            f"expected_ndim={config.gemm_op.input_ndim}, "
            "device_capacity_override=device_capacity_override, "
            "quack_cache_dir=quack_cache_dir"
        )
        if config.quack_config_key is not None:
            call_kwargs += f", config_key={tuple(config.quack_config_key)!r}"

        output_name = self.get_output()
        template_input_arg_names = [
            arg_name for arg_name, _ in self._template_input_args
        ]

        code = IndentedBuffer()
        code.splice(
            """
            import torch
            from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue as flex_gemm_epilogue
            """
        )
        code.splice(config.epilogue_source)
        code.splice(
            f"""
            def {self.kernel_name}_main({", ".join(params)}):
                flex_gemm_epilogue(
                    {", ".join(call_args)},
                    {config.epilogue_name},
                    {config.epilogue_name!r}{call_kwargs},
                )

            def {self.kernel_name}_precompile(
                precompile_shapes,
                precompile_strides,
                precompile_dtypes,
                device_index=0,
                device_capability=None,
                hw_info=None,
                quack_cache_dir={inductor_quack_cache_dir()!r},
            ):
                from torch._vendor.quack.cache import compile_only_mode

                device = f"cuda:{{device_index}}"
                with compile_only_mode():
            """
        )
        with code.indent():
            with code.indent():
                for arg_name, _ in self._template_input_args:
                    code.writeline(
                        f"{arg_name} = torch.empty_strided(tuple(precompile_shapes[{arg_name!r}]), "
                        f"tuple(precompile_strides[{arg_name!r}]), device=device, "
                        f"dtype=getattr(torch, precompile_dtypes[{arg_name!r}]))"
                    )
                code.writeline(
                    f"{output_name} = torch.empty_strided(tuple(precompile_shapes['output']), "
                    "tuple(precompile_strides['output']), device=device, "
                    "dtype=getattr(torch, precompile_dtypes['output']))"
                )
                code.writeline(f"{self.kernel_name}_main(")
                with code.indent():
                    for arg_name in template_input_arg_names:
                        code.writeline(f"{arg_name},")
                    code.writeline(f"{output_name}={output_name},")
                    code.writeline("stream=None,")
                    code.writeline("device_capacity_override=device_capability,")
                    code.writeline("quack_cache_dir=quack_cache_dir,")
                code.writeline(")")
        return PartialRender(code.getvalue(), self.render_hooks)

    def _gemm_call_args(self, input_args, config):
        out_dtype = (
            "" if config.out_dtype is None else f", out_dtype={config.out_dtype!r}"
        )
        op = config.gemm_op
        call_args = [input_args[op.mat1_index], input_args[op.mat2_index]]
        if op.bias_index is None:
            return call_args, out_dtype
        return call_args, (
            f", C={input_args[op.bias_index]}, alpha={config.alpha!r}, beta={config.beta!r}"
            f"{out_dtype}"
        )


class FlexGemmEpilogueCaller(CuteDSLTemplateCaller):
    def precompile(self) -> None:
        """Warm the generated FlexGEMM epilogue module's QuACK object cache."""
        metadata = self.precompile_metadata()
        if metadata is None:
            return
        from torch._inductor.async_compile import AsyncCompile

        if not AsyncCompile.use_process_pool():
            AsyncCompile.wait_pool_ready()
        if not AsyncCompile.use_process_pool():
            return
        AsyncCompile().cutedsl(
            self.bmreq.kernel_name,
            self.bmreq.source_code,
            precompile_metadata=metadata,
        ).result()

    def precompile_metadata(self) -> dict[str, object] | None:
        """Build the generated FlexGEMM precompile hook's tensor metadata."""
        precompile_shapes = {}
        precompile_strides = {}
        precompile_dtypes = {}
        tensor_metas = [
            *(
                (f"arg{index}", tensor_meta)
                for index, tensor_meta in enumerate(self.bmreq.input_tensor_meta)
            ),
            ("output", self.bmreq.output_tensor_meta),
        ]
        try:
            for name, tensor_meta in tensor_metas:
                precompile_shapes[name] = [int(size) for size in tensor_meta.sizes]
                precompile_strides[name] = [
                    int(stride) for stride in tensor_meta.strides
                ]
                precompile_dtypes[name] = str(tensor_meta.dtype).removeprefix("torch.")
        except (TypeError, RuntimeError, ValueError):
            counters["inductor"]["flex_gemm_precompile_skipped_dynamic"] += 1
            log.debug("Skipping FlexGEMM precompile for symbolic tensor metadata")
            return None
        device_index = self.bmreq.output_tensor_meta.device.index or 0
        device_capability = None
        if torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability(device_index)
        return {
            "precompile_shapes": precompile_shapes,
            "precompile_strides": precompile_strides,
            "precompile_dtypes": precompile_dtypes,
            "device_index": device_index,
            "device_capability": device_capability,
            "quack_cache_dir": inductor_quack_cache_dir(),
        }


class FlexGemmEpilogueTemplate(CuteDSLTemplate):
    kernel_type = FlexGemmEpilogueKernel
    caller_type = FlexGemmEpilogueCaller

    def __init__(self) -> None:
        super().__init__("flex_gemm_epilogue", source="")


flex_gemm_epilogue_template = FlexGemmEpilogueTemplate()
