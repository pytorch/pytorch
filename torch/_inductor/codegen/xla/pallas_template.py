from __future__ import annotations

from torch._inductor.codegen.common import KernelTemplate
from torch._inductor.ir import ChoiceCaller


class PallasTemplate(KernelTemplate):
    """
    A template for Pallas kernels. This class is responsible for generating
    a benchmarkable ChoiceCaller from a Pallas kernel definition.
    """

    def __init__(self, name: str, source: str):
        super().__init__(name)
        # The Pallas kernel code is likely stored as a Jinja2 template string
        self.template = self._template_from_string(source)

    def generate(self, **kwargs) -> ChoiceCaller:
        # TODO: Implement the logic to render, compile, and wrap the Pallas kernel.
        return PallasTemplateCaller("dummy_name", [], None, None)


class PallasTemplateCaller(ChoiceCaller):
    """
    A ChoiceCaller for Pallas kernels. This object is the "athlete" that the
    autotuner will benchmark.
    """

    def __init__(self, name, inputs, layout, compiled_pallas_kernel):
        super().__init__(name, inputs, layout)
        self.compiled_pallas_kernel = compiled_pallas_kernel

    def benchmark(self, *args, out: torch.Tensor) -> float:
        # TODO: Implement the logic to run the compiled Pallas kernel
        # and return its performance.
        return 0.0
