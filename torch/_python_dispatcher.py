import re

import torch._C as C


"""
PythonDispatcher class is a thin python-binding to C++ dispatcher and it
is designed to show how dispatcher precompute works. In particular,
it shows for a certain op `foo`, what the computed dispatch table looks
like after user register their kernels to certains dispatch keys.

In the real C++ dispatcher we support many dispatch keys for different
functionalities. For simplicity PythonDispatcher only supports dispatch
keys for a single example of each use case. These use cases are listed below:

- CPU/AutogradCPU: represents in-tree backends which we usually have dedicated inference &
    autograd kernel in pytorch core library.
    E.g. CPU, CUDA
- FPGA/AutogradOther: represents in-tree backends which we usually have backend specific
    inference kernels, but they share the same autograd kernel specified in AutogradOther.
    E.g. FPGA, SparseCsrCPU
- XLA/AutogradXLA: represents out-of-tree backends which we don't have either inference or autograd
    kernel defined in pytorch core library. Backend owner is responsible for registering both
    inference & autograd kernels in their extensions(e.g. torch-xla) for the operators they support.
    E.g. XLA, XPU, MPS
- CompositeExplicitAutograd: alias key mapped to inference kernels of all backends like CPU, CUDA, XLA etc.
    Kernels registered to this key MUST work for inference for all backends.
- Autograd: alias key mapped to autograd of all backends like AutogradCPU, AutogradXLA, AutogradOther.
    Kernels registered to this key MUST work for autograd for all backends.
- CompositeImplicitAutograd: alias key CompositeImplicitAutograd = CompositeExplicitAutograd + Autograd
    Kernels registered to this key MUST work for both inference + autograd for all backends.

Note we only allow registrations to alias keys inside pytorch core library. E.g
you shouldn't register a CompositeImplicitAutograd or CompositeExplicitAutograd
kernel from torch-xla extension, instead you should upstream the kernel into
pytorch/pytorch repo so that it's available for all backends and continuously
tested even without the extension.

Usage:
  dispatcher = PythonDispatcher()
  dispatcher.register(["CPU", "XLA", "CompositeImplicitAutograd"])
  print(dispatcher.dispatchTable()) # This tells you exactly which kernel is used for certain backend.
  # For more debugging information
  # print(dispatcher.keys())
  # print(dispatcher.registrations())
  # print(dispatcher.rawRegistrations())
  # print(dispatcher.rawDispatchTable())
PythonDispatcher calls C++ dispatcher under the hood for to precompute dispatch table.
This file only provides the simplified API for developers, relevant test code is located in
test/test_dispatch.py
"""


class PythonDispatcher:
    namespace = "__test__"
    name = "foo"
    # fmt: off
    runtime_keys = [
        "CPU", "AutogradCPU",
        "FPGA", "AutogradOther",
        "XLA", "AutogradXLA",
        "Lazy", "AutogradLazy",
    ]
    # fmt: on
    alias_keys = [
        "CompositeExplicitAutograd",
        "Autograd",
        "CompositeImplicitAutograd",
    ]
    supported_keys = runtime_keys + alias_keys

    def __init__(self):
        C._dispatch_check_invariants(self.name)  # type: ignore[attr-defined]
        self.ref = C._dispatch_library("FRAGMENT", self.namespace, "")
        self.ref.def_("foo(Tensor x) -> Tensor")

    """
    Returns a list of dispatch keys supported by PythonDispatcher.
    You can register kernels to these keys.
    """

    def keys(self):
        return self.supported_keys

    """
    Register kernels to the target dispatchKeys.
    dispatchKeys(list[str]): a list of dispatch keys that you want to register
      your own kernel. Note that you don't need to write the kernel yourself in
      this PythonDispatcher.E.g. for CPU key, a kernel(e.g fn_CPU for CPU) is
      automatically generated and registered.
    """

    def register(self, dispatchKeys):
        # Overriden is not supported and triggers a warning in C++ dispatcher.
        if len(set(dispatchKeys)) != len(dispatchKeys):
            raise RuntimeError(
                f"Overriden is not allowed but found duplicates in {dispatchKeys}."
            )
        # We currently forbid this in codegen instead of C++ dispatcher.
        if (
            "CompositeImplicitAutograd" in dispatchKeys
            and "CompositeExplicitAutograd" in dispatchKeys
        ):
            raise RuntimeError(
                "Registration to both CompositeImplicitAutograd and CompositeExplicitAutograd is not allowed."
            )
        for key in dispatchKeys:
            if key not in self.supported_keys:
                raise RuntimeError(
                    f"{key} is not supported, please select a dispatch key in {self.supported_keys}."
                )
            self.ref.impl_t_t("foo", dispatch=key, debug="fn_" + key)

    """
    Helper function to format (key, kernel).
    """

    def _format_line(self, key, kernel):
        return "{:<15} {}\n".format(key, kernel)

    """
    Helper function to print a table header.
    """

    def _format_header(self, header):
        s = f"""
{header}
"""
        s += self._format_line("key", "kernel")
        s += "---------------------------\n"
        return s

    """
    Returns raw output of all registration info for debugging only.
    Use registrations() for a simplified version.
    """

    def rawRegistrations(self):
        return C._dispatch_dump("{}::{}".format(self.namespace, self.name))  # type: ignore[attr-defined]

    """
    Returns raw output of computed dispatch table for debugging only.
    Use dispatchTable() for a simplified version.
    """

    def rawDispatchTable(self):
        return C._dispatch_dump_table("{}::{}".format(self.namespace, self.name))  # type: ignore[attr-defined]

    """
    Returns a table(str) including all the registrations from users.
    Note this includes registrations to both runtime keys and alias keys.
    """

    def registrations(self):
        output = self._format_header("Registered Kernels")
        state = self.rawRegistrations()
        state_entries = state.split("\n")
        for line in state_entries:
            first = line.split(":")[0]
            if any(first.startswith(k) for k in self.supported_keys):
                kernel = line.split("::")[0].split(" ")[1]
                output += self._format_line(first, kernel)
        return output

    """
    Returns the computed dispatch table(str). Note this only include
    runtime keys, registrations to alias keys have been decoded to their
    mapped runtime keys.
    """

    def dispatchTable(self):
        output = self._format_header("Computed Dispatch Table")
        table = self.rawDispatchTable()
        table_entries = table.split("\n")
        regex = re.compile(r"registered at .*FallbackKernel\.cpp.*(\[)")
        for line in table_entries:
            k = line.split(":")[0]
            if k in self.runtime_keys:
                entry = regex.sub("[", line)
                output += self._format_line(k, entry.split(": ")[1])
        return output
