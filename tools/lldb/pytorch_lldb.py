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


def IntArrayRef_summary(valobj: Any, internal_dict: Any, options: Any) -> str:
    """Print human readable representation of c10::IntArrayRef"""
    with DisableBreakpoints():
        target = get_target()
        tensor = valobj.GetName()
        result = target.EvaluateExpression(
            f"torch::gdb::int_array_ref_string({tensor})"
        )
        str_result = str(result)
        str_result = str_result[str_result.find('"') + 1 : -1]
        return str_result


def DispatchKeyset_summary(valobj: Any, internal_dict: Any, options: Any) -> str:
    """Print human readable representation of c10::DispatchKeyset"""
    with DisableBreakpoints():
        target = get_target()
        keyset = valobj.GetName()
        result = target.EvaluateExpression(
            f"torch::gdb::dispatch_keyset_string({keyset})"
        )
        str_result = str(result)
        str_result = str_result[str_result.find('"') + 1 : -1]
        return str_result


def Tensor_summary(valobj: Any, internal_dict: Any, options: Any) -> str:
    """Print a human readable representation of the given at::Tensor.

    at::Tensor instances do not have a C++ implementation of a repr method: in
    pytorch, this is done by pure-Python code. As such, print <tensor>
    internally creates a Python wrapper for the given tensor and call repr()
    on it.
    Usage:
        print self
    """
    with DisableBreakpoints():
        target = get_target()
        tensor = valobj.GetName()
        result = target.EvaluateExpression(f"torch::gdb::tensor_repr({tensor})")
        str_result = str(result)
        target.EvaluateExpression(f"(void)free({result.GetValue()})")
        str_result = "\n" + str_result[str_result.find("tensor") : -1]
        return str_result


# And the initialization code to add your commands
def __lldb_init_module(debugger: Any, internal_dict: Any) -> Any:
    debugger.HandleCommand(
        "type summary add c10::IntArrayRef -F pytorch_lldb.IntArrayRef_summary -w torch"
    )
    debugger.HandleCommand(
        "type summary add c10::DispatchKeySet -F pytorch_lldb.DispatchKeyset_summary -w torch"
    )
    debugger.HandleCommand(
        "type summary add at::Tensor -F pytorch_lldb.Tensor_summary -w torch"
    )
    print(
        "Pretty Printing lldb summary for PyTorch AT types has been installed and is ready for use. "
        "This category is enabled by default. To disable run: `type category disable torch`"
    )
    print(
        "Usage:\n\tprint <at::tensor>\n\tprint <c10::IntArrayRef>\n\tprint <c10::DispatchKeySet>"
    )
