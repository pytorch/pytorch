# mypy: allow-untyped-defs
from __future__ import annotations

<<<<<<< HEAD
import functools
=======
>>>>>>> 5709375d565 ([AOTI][tooling][1/n] Add intermediate value debug printer (#132323))
import os
from typing import List, Optional

from .. import config
from ..virtualized import V
from .common import TensorArg


class DebugPrinterManager:
<<<<<<< HEAD
    DEBUG_FILTER_DEFAULT_PRINT_ALL = "default"

    def __init__(
        self,
        enable_debug_printer: bool,
        args_to_print: Optional[List[str]] = None,
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
=======
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
>>>>>>> 5709375d565 ([AOTI][tooling][1/n] Add intermediate value debug printer (#132323))

    def __enter__(self):
        if self.enable_debug_printer:
            V.graph.all_codegen_kernel_names.add(self.kernel_name)
            self.codegen_intermediate_tensor_value_printer(
                self.args_to_print,
                self.kernel_name,
                before_launch=True,
<<<<<<< HEAD
                arg_signatures=self.arg_signatures,
            )

    def __exit__(self, args_to_print, kernel_name, arg_signatures):
=======
                arg_types=self.arg_types,
            )

    def __exit__(self, args_to_print, kernel_name, arg_types):
>>>>>>> 5709375d565 ([AOTI][tooling][1/n] Add intermediate value debug printer (#132323))
        if self.enable_debug_printer:
            self.codegen_intermediate_tensor_value_printer(
                self.args_to_print,
                self.kernel_name,
                before_launch=False,
<<<<<<< HEAD
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
=======
                arg_types=self.arg_types,
            )

>>>>>>> 5709375d565 ([AOTI][tooling][1/n] Add intermediate value debug printer (#132323))
    def get_debug_filtered_kernel_names(self) -> List[str]:
        return [
            x.strip()
            for x in os.environ.get(
                "AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT",
<<<<<<< HEAD
                self.DEBUG_FILTER_DEFAULT_PRINT_ALL,
=======
                ",".join(V.graph.all_codegen_kernel_names),
>>>>>>> 5709375d565 ([AOTI][tooling][1/n] Add intermediate value debug printer (#132323))
            )
            .lower()
            .split(",")
        ]

    def codegen_intermediate_tensor_value_printer(
        self,
        args_to_print,
        kernel_name,
        before_launch=True,
<<<<<<< HEAD
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
=======
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
>>>>>>> 5709375d565 ([AOTI][tooling][1/n] Add intermediate value debug printer (#132323))
            ):
                continue
            launch_prefix = "before_launch" if before_launch else "after_launch"
            if V.graph.cpp_wrapper:
                if config.abi_compatible:
<<<<<<< HEAD
                    V.graph.wrapper_code.writeline(
=======
                    self.wrapper.writeline(
>>>>>>> 5709375d565 ([AOTI][tooling][1/n] Add intermediate value debug printer (#132323))
                        f'aoti_torch_print_tensor_handle({arg}, "{launch_prefix} - {kernel_name} - {arg}");'
                    )
                else:
                    # TODO: add non-abi compatible mode debug printing info
                    pass
            else:
<<<<<<< HEAD
                line = f"print('{launch_prefix} - {kernel_name} - {arg}', {arg})"
                V.graph.wrapper_code.writeline(line)
=======
                line = f"print('{launch_prefix} {kernel_name} - {arg}', {arg})"
                self.wrapper.writeline(line)
>>>>>>> 5709375d565 ([AOTI][tooling][1/n] Add intermediate value debug printer (#132323))
