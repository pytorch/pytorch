"""Freezing

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""

from typing import Optional, List

import torch
from torch.jit._script import RecursiveScriptModule, ScriptModule


def freeze(mod, preserved_attrs: Optional[List[str]] = None):
    r"""
    Freezing a :class:`ScriptModule` will clone it and attempt to inline the cloned
    module's submodules, parameters, and attributes as constants in the TorchScript IR Graph.
    By default, `forward` will be preserved, as well as attributes & methods specified in
    `preserved_attrs`. Additionally, any attribute that is modified within a preserved
    method will be preserved.

    Freezing currently only accepts ScriptModules that are in eval mode.

    Arguments:
        mod (:class:`ScriptModule`): a module to be frozen

        preserved_attrs (Optional[List[str]]): a list of attributes to preserve in addition to the forward method.
        Attributes modified in preserved methods will also be preserved.

    Returns:
        Frozen :class:`ScriptModule`.

    Example (Freezing a simple module with a Parameter):

    .. testcode::
        import torch
        class MyModule(torch.nn.Module):
            def __init__(self, N, M):
                super(MyModule, self).__init__()
                self.weight = torch.nn.Parameter(torch.rand(N, M))
                self.linear = torch.nn.Linear(N, M)

            def forward(self, input):
                output = self.weight.mm(input)
                output = self.linear(output)
                return output

        scripted_module = torch.jit.script(MyModule(2, 3).eval())
        frozen_module = torch.jit.freeze(scripted_module)
        # parameters have been removed and inlined into the Graph as constants
        assert len(list(frozen_module.named_parameters())) == 0
        # See the compiled graph as Python code
        print(frozen_module.code)

    Example (Freezing a module with preserved attributes)

    .. testcode::
        import torch
        class MyModule2(torch.nn.Module):
            def __init__(self):
                super(MyModule2, self).__init__()
                self.modified_tensor = torch.tensor(10.)
                self.version = 1

            def forward(self, input):
                self.modified_tensor += 1
                return input + self.modified_tensor

        scripted_module = torch.jit.script(MyModule2().eval())
        frozen_module = torch.jit.freeze(scripted_module, preserved_attrs=["version"])
        # we've manually preserved `version`, so it still exists on the frozen module and can be modified
        assert frozen_module.version == 1
        frozen_module.version = 2
        # `modified_tensor` is detected as being mutated in the forward, so freezing preserves
        # it to retain model semantics
        assert frozen_module(torch.tensor(1)) == torch.tensor(12)
        # now that we've run it once, the next result will be incremented by one
        assert frozen_module(torch.tensor(1)) == torch.tensor(13)

    Note:
        If you're not sure why an attribute is not being inlined as a constant, you can run
        `dump_alias_db` on frozen_module.forward.graph to see if freezing has detected the
        attribute is being modified.
    """
    if not isinstance(mod, ScriptModule):
        raise RuntimeError(
            "Freezing expects a ScriptModule as input. "
            "Please use torch.jit.script or torch.jit.trace to script your 'nn.Module'."
        )

    if mod.training:
        raise RuntimeError(
            "Freezing is currently only implemented for modules in eval mode. "
            "Please call .eval() on your module before freezing."
        )

    preserved_attrs = preserved_attrs if preserved_attrs is not None else []

    out = RecursiveScriptModule(torch._C._freeze_module(mod._c, preserved_attrs))
    RecursiveScriptModule._finalize_scriptmodule(out)

    return out
