import shlex
from typing import Any

import lldb  # type: ignore[import]


def get_target() -> Any:
    target = lldb.debugger.GetSelectedTarget()
    if not target:
        print("[-] error: no target available. please add a target to lldb.")
        return None
    return target


class DisableBreakpoints:
    """
    Context-manager to temporarily disable all lldb breakpoints, useful if
    there is a risk to hit one during the evaluation of one of our custom
    commands
    """

    def __enter__(self) -> None:
        target = get_target()

        if target.DisableAllBreakpoints() is False:
            print("[-] error: failed to disable all breakpoints.")

    def __exit__(self, etype: Any, evalue: Any, tb: Any) -> None:
        target = get_target()

        if target.EnableAllBreakpoints() is False:
            print("[-] error: failed to enable all breakpoints.")


def print_tensor(debugger: Any, command: Any, result: Any, internal_dict: Any) -> None:
    """
    Print a human readable representation of the given at::Tensor.
    Usage: print_tensor EXP

    at::Tensor instances do not have a C++ implementation of a repr method: in
    pytorch, this is done by pure-Python code. As such, print_tensor
    internally creates a Python wrapper for the given tensor and call repr()
    on it.
    """
    with DisableBreakpoints():
        command_str = shlex.split(command)
        assert (
            len(command_str) == 1
        ), "print_tensor only accepts the tensor name as argument"
        tensor_arg = command_str[0]
        target = get_target()
        # This will allocate the the char* array
        result = target.EvaluateExpression(f"torch::gdb::tensor_repr({tensor_arg})")
        print(f"Python-level repr of {tensor_arg}:")
        str_result = str(result)
        print(str_result[str_result.find("tensor") : -1])
        # torch::gdb::tensor_repr returns a malloc()ed buffer, let's free it
        target.EvaluateExpression(f"(void)free({result.GetValue()})")


# And the initialization code to add your commands
def __lldb_init_module(debugger: Any, internal_dict: Any) -> None:
    debugger.HandleCommand(
        "command script add -f pytorch_lldb.print_tensor print_tensor"
    )
    print('The "print_tensor" python command has been installed and is ready for use.')
