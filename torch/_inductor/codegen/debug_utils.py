# mypy: allow-untyped-defs
from __future__ import annotations

import functools
import logging
import os
from enum import Enum
from typing import Callable, Optional

import torch
from torch import dtype as torch_dtype

from .. import config
from ..virtualized import V
from .multi_kernel import MultiKernel


log = logging.getLogger(__name__)


def _print_debugging_tensor_value_info(msg, arg):
    # helper for printing debugging stats for intermediate tensor values
    # at jit inductor level codegen
    max_numel_to_print = 64
    print(msg)
    if not isinstance(arg, torch.Tensor):
        print("Value: ", arg)
        return
    numel = arg.float().numel()
    # print the debug printing stats
    if numel <= max_numel_to_print:
        print(arg)
    print("Number of elements: ", numel)
    print("Size: ", arg.float().size())
    print("Dtype: ", arg.float().mean().item())
    print("Mean: ", arg.float().mean().item())
    print("Min: ", arg.float().min().item())
    print("Max: ", arg.float().max().item())
    print("Std: ", arg.float().std().item())


# AOTI debug printing related configs
class IntermediateValueDebuggingLevel(Enum):
    # OFF: No intermediate tensor value debug info will be printed or saved.
    OFF = "0"
    # LEVEL 1: Save all intermediate tensor values to individual `.pt` files. No debug printing will be displayed.
    SAVE_ONLY = "1"
    # LEVEL 2: Print all intermediate tensor values by default to the console. No debug saving will be performed.
    PRINT_ONLY = "2"
    # LEVEL 3: Print all kernel names to the console only. No debug saving/printing for input tensor value info will be performed.
    # This mode can be helpful in cases when you just want to pinpointing what kernel is running into a CUDA IMA issue, etc.
    PRINT_KERNEL_NAMES_ONLY = "3"


class DebugPrinterManager:
    def __init__(
        self,
        debug_printer_level,
        use_array_ref: bool,
        writeline: Optional[Callable[..., None]] = None,
        args_to_print_or_save: Optional[list[str]] = None,
        kernel_name: str = "",
        kernel=None,
        arg_signatures: Optional[list[type]] = None,
        kernel_type=None,
    ):
        self.debug_printer_level = IntermediateValueDebuggingLevel(debug_printer_level)
        self.use_array_ref = use_array_ref
        if args_to_print_or_save is None:
            args_to_print_or_save = []
        self.args_to_print_or_save = args_to_print_or_save
        self.kernel_name = kernel_name
        self.arg_signatures: Optional[list[type]] = None
        self.kernel = kernel
        self.filtered_kernel_names_to_print = self._get_debug_filtered_kernel_names()
        self.kernel_type = None

    def __enter__(self):
        self._perform_debug_print_or_save_helper(
            self.args_to_print_or_save,
            self.kernel_name,
            before_launch=True,
            arg_signatures=self.arg_signatures,
        )

    def __exit__(self, args_to_print_or_save, kernel_name, arg_signatures):
        self._perform_debug_print_or_save_helper(
            args_to_print_or_save,
            kernel_name,
            before_launch=False,
            arg_signatures=arg_signatures,
        )

    def _perform_debug_print_or_save_helper(
        self,
        args_to_print_or_save,
        kernel_name,
        before_launch,
        arg_signatures: Optional[list[type]] = None,
    ):
        if self.debug_printer_level == IntermediateValueDebuggingLevel.OFF:
            return
        if self.debug_printer_level == IntermediateValueDebuggingLevel.SAVE_ONLY:
            # by default save all the tensor values before launch
            self.codegen_intermediate_tensor_value_save(
                self.args_to_print_or_save,
                self.kernel_name,
                before_launch,
                arg_signatures=self.arg_signatures,
            )
        if self.debug_printer_level == IntermediateValueDebuggingLevel.PRINT_ONLY:
            # by default print all the tensor values before launch
            self.codegen_intermediate_tensor_value_print(
                self.args_to_print_or_save,
                self.kernel_name,
                before_launch,
                arg_signatures=self.arg_signatures,
            )
        if (
            self.debug_printer_level
            == IntermediateValueDebuggingLevel.PRINT_KERNEL_NAMES_ONLY
        ):
            # Print all kernel names to the console only
            self.codegen_intermediate_tensor_value_print(
                [],
                self.kernel_name,
                before_launch,
            )

    @functools.lru_cache  # noqa: B019
    def _get_debug_filtered_kernel_names(self) -> list[str]:
        if config.aot_inductor.filtered_kernel_names is None:
            return []
        return [
            x.strip()
            for x in config.aot_inductor.filtered_kernel_names.lower().split(",")
        ]

    def set_printer_args(
        self,
        args_to_print_or_save: list[str],
        kernel_name: str,
        arg_signatures: Optional[list[type]],
        kernel,
        kernel_type=None,
    ):
        # Note: MultiKernel debug printing is not supported for now
        if isinstance(kernel, MultiKernel):
            log.info(
                "MultiKernel type is not supported in AOTI debug printer tool yet."
            )
            self.debug_printer_level = IntermediateValueDebuggingLevel.OFF

        self.kernel_type = kernel_type
        # Note: if the kernel type is an extern kernel (or cpp kernel), we do a special handling to
        # get the list of args_to_print_or_save
        # TODO: Find a more reliable way to detect kernel args types to print for extern kernel calls
        if kernel_type == "extern":
            args_to_print_or_save_extern = [
                arg for arg in args_to_print_or_save if arg.startswith(("buf", "arg"))
            ]
            self.args_to_print_or_save = args_to_print_or_save_extern
        elif kernel_type == "cpp":
            self.args_to_print_or_save = [
                (
                    f"copy_arrayref_tensor_to_tensor({arg})"
                    if self.use_array_ref
                    else arg
                )
                for arg in args_to_print_or_save
                if arg.startswith(("buf", "arg"))
            ]
        else:
            self.args_to_print_or_save = args_to_print_or_save
        self.kernel_name = kernel_name
        self.arg_signatures = arg_signatures
        self.kernel = kernel

    def codegen_model_inputs_value_print(self, input_args_to_print: list[str]) -> None:
        if self.debug_printer_level != IntermediateValueDebuggingLevel.PRINT_ONLY:
            return
        for arg in input_args_to_print:
            if V.graph.cpp_wrapper:
                V.graph.wrapper_code.prefix.writeline(
                    f'aoti_torch_print_tensor_handle({arg}, "aoti_model_inputs - {arg}");'
                )

    def codegen_intermediate_tensor_value_save(
        self,
        args_to_save,
        kernel_name,
        before_launch=True,
        arg_signatures: Optional[list[type]] = None,
    ) -> None:
        for i, arg in enumerate(args_to_save):
            if arg_signatures is not None and not isinstance(
                arg_signatures[i], torch_dtype
            ):
                # infer from the arg data type (has torch.dtype) to see if it is a tensor type
                continue
            launch_prefix = "before_launch" if before_launch else "after_launch"
            if V.graph.cpp_wrapper:
                V.graph.wrapper_code.writeline(
                    f'aoti_torch_save_tensor_handle({arg}, "{arg}", "{launch_prefix}", "{kernel_name}");'
                )
            else:
                cwd = os.getcwd()
                saved_dir = cwd + "/tmp/jit_inductor/"
                if not os.path.exists(saved_dir):
                    log.info(
                        "Creating directory to save inductor intermediate tensor values."
                    )
                    os.makedirs(saved_dir)
                # Save the model to the directory
                saved_path = saved_dir + f"{launch_prefix}_{kernel_name}_{arg}.pt"
                log.info(
                    "Saved intermediate tensor %s for %s to %s",
                    arg,
                    kernel_name,
                    saved_path,
                )
                line = f"torch.save({arg}, '{saved_path}')"
                V.graph.wrapper_code.writeline(line)

    def codegen_intermediate_tensor_value_print(
        self,
        args_to_print,
        kernel_name,
        before_launch=True,
        arg_signatures: Optional[list[type]] = None,
    ) -> None:
        launch_prefix = "before_launch" if before_launch else "after_launch"

        # if the debug printing level is PRINT_KERNEL_NAMES_ONLY
        # we only print the kernel name to the console
        if (
            self.debug_printer_level
            == IntermediateValueDebuggingLevel.PRINT_KERNEL_NAMES_ONLY
        ):
            if V.graph.cpp_wrapper:
                V.graph.wrapper_code.writeline(
                    f'printf("[ {launch_prefix}: {kernel_name} ]\\n");'
                )
            return

        if self.debug_printer_level != IntermediateValueDebuggingLevel.PRINT_ONLY:
            return
        for i, arg in enumerate(args_to_print):
            # when debug printing is enabled i.e. IntermediateValueDebuggingLevel.PRINT_ONLY,
            # check if filtered kernel name list is provided
            if (
                len(self.filtered_kernel_names_to_print) > 0
                and kernel_name.lower() not in self.filtered_kernel_names_to_print
            ):
                continue
            if V.graph.cpp_wrapper:
                if arg_signatures is not None and isinstance(
                    arg_signatures[i], torch_dtype
                ):
                    # infer from the arg data type (has torch.dtype) to see if it is a tensor type
                    V.graph.wrapper_code.writeline(
                        f'aoti_torch_print_tensor_handle({arg}, "{launch_prefix} - {kernel_name} - {arg}");'
                    )
                elif arg_signatures is not None and isinstance(
                    arg_signatures[i],
                    (
                        type(torch._inductor.codegen.wrapper.SymbolicCallArg),
                        type(int),
                        type(float),
                        type(bool),
                    ),
                ):
                    V.graph.wrapper_code.writeline(
                        f'printf("[  {launch_prefix} - {kernel_name} - {arg}: %ld  ]", {arg}); printf("\\\\n");'
                    )
                else:
                    if arg_signatures is None and self.kernel_type == "cpp" or "extern":
                        V.graph.wrapper_code.writeline(
                            f'aoti_torch_print_tensor_handle({arg}, "{launch_prefix} - {kernel_name} - {arg}");'
                        )
            else:
                V.graph.wrapper_code.writeline(
                    f'_print_debugging_tensor_value_info("inductor: {launch_prefix} - {kernel_name} - {arg}", {arg})'
                )
