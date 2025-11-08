from __future__ import annotations

from torch._inductor.codegen.wrapper import PythonWrapperCodegen


class PallasWrapperCodegen(PythonWrapperCodegen):
    """
    This class is responsible for generating the Python code that calls
    the compiled XLA kernels. It will need to override methods to handle
    kernel definition and invocation specific to the XLA runtime.
    """

    def __init__(self):
        super().__init__()
        # The device for this wrapper is 'xla'
        self.device = "xla"