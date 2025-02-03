import textwrap
from typing import Any

import gdb  # type: ignore[import]


class DisableBreakpoints:
    """
    Context-manager to temporarily disable all gdb breakpoints, useful if
    there is a risk to hit one during the evaluation of one of our custom
    commands
    """

    def __enter__(self) -> None:
        self.disabled_breakpoints = []
        for b in gdb.breakpoints():
            if b.enabled:
                b.enabled = False
                self.disabled_breakpoints.append(b)

    def __exit__(self, etype: Any, evalue: Any, tb: Any) -> None:
        for b in self.disabled_breakpoints:
            b.enabled = True


class TensorRepr(gdb.Command):  # type: ignore[misc, no-any-unimported]
    """
    Print a human readable representation of the given at::Tensor.
    Usage: torch-tensor-repr EXP

    at::Tensor instances do not have a C++ implementation of a repr method: in
    pytorch, this is done by pure-Python code. As such, torch-tensor-repr
    internally creates a Python wrapper for the given tensor and call repr()
    on it.
    """

    __doc__ = textwrap.dedent(__doc__).strip()

    def __init__(self) -> None:
        gdb.Command.__init__(
            self, "torch-tensor-repr", gdb.COMMAND_USER, gdb.COMPLETE_EXPRESSION
        )

    def invoke(self, args: str, from_tty: bool) -> None:
        args = gdb.string_to_argv(args)
        if len(args) != 1:
            print("Usage: torch-tensor-repr EXP")
            return
        name = args[0]
        with DisableBreakpoints():
            res = gdb.parse_and_eval(f"torch::gdb::tensor_repr({name})")
            print(f"Python-level repr of {name}:")
            print(res.string())
            # torch::gdb::tensor_repr returns a malloc()ed buffer, let's free it
            gdb.parse_and_eval(f"(void)free({int(res)})")


class IntArrayRefRepr(gdb.Command):  # type: ignore[misc, no-any-unimported]
    """
    Print human readable representation of c10::IntArrayRef
    """

    def __init__(self) -> None:
        gdb.Command.__init__(
            self, "torch-int-array-ref-repr", gdb.COMMAND_USER, gdb.COMPLETE_EXPRESSION
        )

    def invoke(self, args: str, from_tty: bool) -> None:
        args = gdb.string_to_argv(args)
        if len(args) != 1:
            print("Usage: torch-int-array-ref-repr EXP")
            return
        name = args[0]
        with DisableBreakpoints():
            res = gdb.parse_and_eval(f"torch::gdb::int_array_ref_string({name})")
            res = str(res)
            print(res[res.find('"') + 1 : -1])


class DispatchKeysetRepr(gdb.Command):  # type: ignore[misc, no-any-unimported]
    """
    Print human readable representation of c10::DispatchKeyset
    """

    def __init__(self) -> None:
        gdb.Command.__init__(
            self,
            "torch-dispatch-keyset-repr",
            gdb.COMMAND_USER,
            gdb.COMPLETE_EXPRESSION,
        )

    def invoke(self, args: str, from_tty: bool) -> None:
        args = gdb.string_to_argv(args)
        if len(args) != 1:
            print("Usage: torch-dispatch-keyset-repr EXP")
            return
        keyset = args[0]
        with DisableBreakpoints():
            res = gdb.parse_and_eval(f"torch::gdb::dispatch_keyset_string({keyset})")
            res = str(res)
            print(res[res.find('"') + 1 : -1])


TensorRepr()
IntArrayRefRepr()
DispatchKeysetRepr()
