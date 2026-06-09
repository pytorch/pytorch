# mypy: allow-untyped-defs
from torch._inductor import ir
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.codegen.cutedsl.cutedsl_kernel import CuteDSLTemplateKernel
from torch._inductor.codegen.cutedsl.cutedsl_template import CuteDSLTemplate
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


class QuackGemmEpilogueKernel(CuteDSLTemplateKernel):
    """Renders the QuACK GEMM epilogue runtime wrapper."""

    def render(self, template, **kwargs):
        config = kwargs.pop("config")
        if kwargs:
            raise RuntimeError(f"unexpected QuACK GEMM epilogue options: {kwargs}")

        self._template_input_args = []
        self._seen_input_args = OrderedSet()
        for index, input_node in enumerate(self.input_nodes):
            buf_name = input_node.get_name()
            self.args.input(buf_name)
            arg_name = f"arg{index}"
            self.args.input_buffers[buf_name] = arg_name
            self._template_input_args.append((arg_name, input_node))
            self._seen_input_args.add(arg_name)

        output_name = self.output_node.get_name()
        output_arg = None
        if output_name not in V.graph.removed_buffers:
            self.args.output(output_name)
            output_arg = self.get_output()
        arg_defs, _, _, _ = self.args.python_argdefs()
        params = [arg_name for arg_name, _ in self._template_input_args]
        for arg_def in arg_defs:
            if arg_def.full_name() not in self._seen_input_args:
                params.append(arg_def.full_name())
        params.append("stream")

        input_args = [arg_name for arg_name, _ in self._template_input_args]
        call_fn, matmul_args, call_kwargs = self._gemm_call(input_args, config)
        call_kwargs += self._epilogue_kwargs(input_args, config)
        call_kwargs += self._local_reduce_kwargs(input_args, config)
        if output_arg is not None:
            call_kwargs += f", out={output_arg}"
        call_kwargs += (
            f", tuned={config.tuned!r}, epilogue_source={config.epilogue_source!r}"
        )

        code = IndentedBuffer()
        code.writeline("import torch")
        code.splice(config.epilogue_source)
        code.writeline(f"def {self.kernel_name}_main({', '.join(params)}):")
        with code.indent():
            self._write_imports(code, config)
            code.writeline(
                f"{call_fn}({matmul_args}, {config.epilogue_name}, "
                f"{config.epilogue_name!r}{call_kwargs})"
            )
        return PartialRender(code.getvalue(), self.render_hooks)

    def _write_imports(self, code: IndentedBuffer, config: ir.QuackGemmEpilogueConfig):
        if config.gemm_op == "scaled_grouped_mm":
            code.writeline(
                "from quack.gemm_blockscaled_interface import "
                "mxfp8_varlen_k_scaled_mm_epilogue, "
                "mxfp8_varlen_m_scaled_mm_epilogue"
            )
        else:
            code.writeline("from quack.gemm_epilogue_interface import gemm_epilogue")

    def _epilogue_kwargs(
        self, input_args: list[str], config: ir.QuackGemmEpilogueConfig
    ) -> str:
        epilogue_args = [input_args[i] for i in config.epilogue_arg_indices]
        kwargs = ""
        if epilogue_args:
            kwargs = f", epilogue_args=({', '.join(epilogue_args)},)"
            kwargs += f", epilogue_arg_kinds={config.epilogue_arg_kinds!r}"
        if config.main_output_transform is not None:
            kwargs += f", main_output_transform={config.main_output_transform!r}"
            if config.main_output_transform_group is not None:
                kwargs += (
                    f", main_output_transform_group="
                    f"{config.main_output_transform_group!r}"
                )
        if config.concat_layout:
            kwargs += f", concat_layout={config.concat_layout!r}"
        return kwargs

    def _local_reduce_kwargs(
        self, input_args: list[str], config: ir.QuackGemmEpilogueConfig
    ) -> str:
        kwargs = ""
        if config.aux_out_index is not None:
            kwargs += f", aux_out={input_args[config.aux_out_index]}"
        if config.local_reduce_group is None:
            return kwargs

        kwargs += (
            f", local_reduce_group={config.local_reduce_group!r}, "
            f"local_reduce_dim={config.local_reduce_dim!r}"
        )
        if config.local_reduce_out_index is not None:
            kwargs += f", local_reduce_out={input_args[config.local_reduce_out_index]}"
        if config.local_reduce_op is not None:
            kwargs += f", local_reduce_op={config.local_reduce_op!r}"
        if config.local_reduce_scale != 1.0:
            kwargs += f", local_reduce_scale={config.local_reduce_scale!r}"
        if (
            config.local_reduce_op == "mx_e8m0_scale"
            or config.local_reduce_max_power != 8
        ):
            kwargs += f", local_reduce_max_power={config.local_reduce_max_power!r}"
        if config.local_reduce_feeds_main:
            kwargs += ", local_reduce_feeds_main=True"
        if config.local_reduce_source_from_epilogue:
            kwargs += ", local_reduce_source_from_epilogue=True"
        return kwargs

    def _gemm_call(
        self, input_args: list[str], config: ir.QuackGemmEpilogueConfig
    ) -> tuple[str, str, str]:
        call_args, call_kwargs = self._gemm_call_args(input_args, config)
        if config.gemm_op == "scaled_grouped_mm":
            call_fn = (
                "mxfp8_varlen_m_scaled_mm_epilogue"
                if config.scaled_grouped_mm_kind == "varlen_m"
                else "mxfp8_varlen_k_scaled_mm_epilogue"
            )
            return call_fn, ", ".join(call_args[:5]), call_kwargs
        return "gemm_epilogue", ", ".join(call_args[:2]), call_kwargs

    def _gemm_call_args(
        self, input_args: list[str], config: ir.QuackGemmEpilogueConfig
    ) -> tuple[list[str], str]:
        out_dtype_kwargs = (
            "" if config.out_dtype is None else f", out_dtype={config.out_dtype!r}"
        )
        if config.gemm_op in ("mm", "bmm"):
            return [input_args[0], input_args[1]], out_dtype_kwargs
        if config.gemm_op == "scaled_mm":
            scale_b_index = 2 + config.scaled_mm_scale_a_len
            kwargs = (
                f", scale_a={input_args[2]}, scale_b={input_args[scale_b_index]}, "
                f"out_dtype={config.out_dtype!r}"
            )
            if config.scaled_mm_scale_a_len == 2 and config.scaled_mm_scale_b_len == 2:
                scale_b_global_index = scale_b_index + 1
                if (
                    scale_b_global_index >= len(input_args)
                    or scale_b_global_index in config.epilogue_arg_indices
                ):
                    scale_b_global_index = 3
                kwargs += (
                    f", scale_a_global={input_args[3]}, "
                    f"scale_b_global={input_args[scale_b_global_index]}"
                )
            return [input_args[0], input_args[1]], kwargs
        if config.gemm_op == "grouped_mm":
            return [input_args[0], input_args[1]], (
                f", offs={input_args[2]}, out_dtype={config.out_dtype!r}"
            )
        if config.gemm_op == "scaled_grouped_mm":
            if config.scaled_grouped_mm_kind == "varlen_m":
                mat_a = input_args[0]
                mat_b = f"{input_args[1]}.permute(2, 1, 0)"
            else:
                mat_a = f"{input_args[0]}.t().contiguous().t()"
                mat_b = f"{input_args[1]}.contiguous().t()"
            return [mat_a, mat_b, input_args[2], input_args[3], input_args[4]], (
                f", out_dtype={config.out_dtype!r}"
            )

        return [input_args[1], input_args[2]], (
            f", C={input_args[0]}, alpha={config.alpha!r}, beta={config.beta!r}"
            f"{out_dtype_kwargs}"
        )


class QuackGemmEpilogueTemplate(CuteDSLTemplate):
    kernel_type = QuackGemmEpilogueKernel

    def __init__(self) -> None:
        super().__init__("quack_gemm_epilogue", source="")


quack_gemm_epilogue_template = QuackGemmEpilogueTemplate()
