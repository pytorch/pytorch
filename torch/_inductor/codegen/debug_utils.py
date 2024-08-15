# mypy: allow-untyped-defs
from __future__ import annotations

import os
from typing import List, Optional

from .. import config
from ..virtualized import V
from .common import TensorArg


class DebugPrinterManager:
    def __init__(
        self,
        enable_debug_printer: bool,
        args_to_print: List[str],
        kernel_name: str,
        kernel=None,
        arg_types: Optional[List[type]] = None,
    ):
        self.wrapper = V.graph.wrapper_code
        self.enable_debug_printer = enable_debug_printer
        self.args_to_print = args_to_print
        self.kernel_name = kernel_name
        self.arg_types: Optional[List[type]] = None
        self.kernel = kernel

    def __enter__(self):
        if self.enable_debug_printer:
            V.graph.all_codegen_kernel_names.add(self.kernel_name)
            self.codegen_intermediate_tensor_value_printer(
                self.args_to_print,
                self.kernel_name,
                before_launch=True,
                arg_types=self.arg_types,
            )

    def __exit__(self, args_to_print, kernel_name, arg_types):
        if self.enable_debug_printer:
            self.codegen_intermediate_tensor_value_printer(
                self.args_to_print,
                self.kernel_name,
                before_launch=False,
                arg_types=self.arg_types,
            )

    def get_debug_filtered_kernel_names(self) -> List[str]:
        return [
            x.strip()
            for x in os.environ.get(
                "AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT",
                ",".join(V.graph.all_codegen_kernel_names),
            )
            .lower()
            .split(",")
        ]

    def codegen_intermediate_tensor_value_printer(
        self,
        args_to_print,
        kernel_name,
        before_launch=True,
        arg_types: Optional[List[type]] = None,
    ) -> None:
        # when invoking this codegen_intermediate_tensor_value_printer function,
        # we already assured that the AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER env var is set to 1,
        # so we can directly use get method for filtered kernel info here
        filtered_kernel_names_to_print = []
        if V.graph.cpp_wrapper:
            filtered_kernel_names_to_print = self.get_debug_filtered_kernel_names()

        for i, arg in enumerate(args_to_print):
            if arg_types is not None and not isinstance(arg_types[i], TensorArg):
                continue
            if (
                len(filtered_kernel_names_to_print) > 0
                and kernel_name not in filtered_kernel_names_to_print
            ):
                continue
            launch_prefix = "before_launch" if before_launch else "after_launch"
            if V.graph.cpp_wrapper:
                if config.abi_compatible:
                    self.wrapper.writeline(
                        f'aoti_torch_print_tensor_handle({arg}, "{launch_prefix} - {kernel_name} - {arg}");'
                    )
                else:
                    # TODO: add non-abi compatible mode debug printing info
                    pass
            else:
                line = f"print('{launch_prefix} {kernel_name} - {arg}', {arg})"
                self.wrapper.writeline(line)
