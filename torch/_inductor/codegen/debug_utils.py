# mypy: allow-untyped-defs
from __future__ import annotations

import functools
import os
from typing import List, Optional

from .. import config
from ..virtualized import V
from .common import TensorArg


class DebugPrinterManager:
    DEBUG_FILTER_DEFAULT_PRINT_ALL = "default"

    def __init__(
        self,
        enable_debug_printer: bool,
        args_to_print: List[str] | None = None,
        kernel_name: str = "",
        kernel=None,
        arg_signatures: Optional[List[type]] = None,
    ):
        self.enable_debug_printer = enable_debug_printer
        if args_to_print is None:
            args_to_print = []
        self.args_to_print = args_to_print
        self.kernel_name = kernel_name
        self.arg_signatures: Optional[List[type]] = None
        self.kernel = kernel
        self.filtered_kernel_names_to_print = self.get_debug_filtered_kernel_names()

    def __enter__(self):
        if self.enable_debug_printer:
            V.graph.all_codegen_kernel_names.add(self.kernel_name)
            self.codegen_intermediate_tensor_value_printer(
                self.args_to_print,
                self.kernel_name,
                before_launch=True,
                arg_signatures=self.arg_signatures,
            )

    def __exit__(self, args_to_print, kernel_name, arg_signatures):
        if self.enable_debug_printer:
            self.codegen_intermediate_tensor_value_printer(
                self.args_to_print,
                self.kernel_name,
                before_launch=False,
                arg_signatures=self.arg_signatures,
            )

    def set_printer_args(
        self,
        args_to_print: List[str],
        kernel_name: str,
        arg_signatures: Optional[List[type]],
        kernel,
    ):
        self.args_to_print = args_to_print
        self.kernel_name = kernel_name
        self.arg_signatures = arg_signatures
        self.kernel = kernel

    @functools.lru_cache  # noqa: B019
    def get_debug_filtered_kernel_names(self) -> List[str]:
        return [
            x.strip()
            for x in os.environ.get(
                "AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT",
                self.DEBUG_FILTER_DEFAULT_PRINT_ALL,
            )
            .lower()
            .split(",")
        ]

    def codegen_intermediate_tensor_value_printer(
        self,
        args_to_print,
        kernel_name,
        before_launch=True,
        arg_signatures: Optional[List[type]] = None,
    ) -> None:
        for i, arg in enumerate(args_to_print):
            if arg_signatures is not None and not isinstance(
                arg_signatures[i], TensorArg
            ):
                continue
            if (
                len(self.filtered_kernel_names_to_print) > 0
                and self.filtered_kernel_names_to_print[0]
                != self.DEBUG_FILTER_DEFAULT_PRINT_ALL
                and kernel_name not in self.filtered_kernel_names_to_print
            ):
                continue
            launch_prefix = "before_launch" if before_launch else "after_launch"
            if V.graph.cpp_wrapper:
                if config.abi_compatible:
                    V.graph.wrapper_code.writeline(
                        f'aoti_torch_print_tensor_handle({arg}, "{launch_prefix} - {kernel_name} - {arg}");'
                    )
                else:
                    # TODO: add non-abi compatible mode debug printing info
                    pass
            else:
                line = f"print('{launch_prefix} - {kernel_name} - {arg}', {arg})"
                V.graph.wrapper_code.writeline(line)
