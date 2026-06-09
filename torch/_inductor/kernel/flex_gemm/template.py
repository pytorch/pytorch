# mypy: allow-untyped-defs
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.codegen.cutedsl.cutedsl_kernel import CuteDSLTemplateKernel
from torch._inductor.codegen.cutedsl.cutedsl_template import CuteDSLTemplate
from torch._inductor.select_algorithm import PartialRender
from torch.utils._ordered_set import OrderedSet


class FlexGemmEpilogueKernel(CuteDSLTemplateKernel):
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

        call_args, call_kwargs = self._gemm_call_args(
            [arg_name for arg_name, _ in self._template_input_args], config
        )
        call_kwargs += (
            f", out={self.get_output()}, tuned=False, "
            f"epilogue_source={config.epilogue_source!r}"
        )

        code = IndentedBuffer()
        code.writeline("import torch")
        code.writeline(
            "from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue as flex_gemm_epilogue"
        )
        code.splice(config.epilogue_source)
        code.writeline(f"def {self.kernel_name}_main({', '.join(params)}):")
        with code.indent():
            code.writeline(
                f"flex_gemm_epilogue({', '.join(call_args)}, "
                f"{config.epilogue_name}, {config.epilogue_name!r}{call_kwargs})"
            )
        return PartialRender(code.getvalue(), self.render_hooks)

    def _gemm_call_args(self, input_args, config):
        out_dtype = (
            "" if config.out_dtype is None else f", out_dtype={config.out_dtype!r}"
        )
        if config.gemm_op == "mm":
            return [input_args[0], input_args[1]], out_dtype
        if config.gemm_op == "addmm":
            return [input_args[1], input_args[2]], (
                f", C={input_args[0]}, alpha={config.alpha!r}, beta={config.beta!r}"
                f"{out_dtype}"
            )
        raise NotImplementedError(f"unsupported FlexGEMM op: {config.gemm_op}")


class FlexGemmEpilogueTemplate(CuteDSLTemplate):
    kernel_type = FlexGemmEpilogueKernel

    def __init__(self) -> None:
        super().__init__("flex_gemm_epilogue", source="")


flex_gemm_epilogue_template = FlexGemmEpilogueTemplate()
