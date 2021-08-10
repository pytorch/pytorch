"""Freezing

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""

from typing import Optional, List

import torch
from torch.jit._script import RecursiveScriptModule, ScriptModule


def freeze(mod, preserved_attrs: Optional[List[str]] = None, optimize_numerics: bool = True):
    r"""
    Freezing a :class:`ScriptModule` will clone it and attempt to inline the cloned
    module's submodules, parameters, and attributes as constants in the TorchScript IR Graph.
    By default, `forward` will be preserved, as well as attributes & methods specified in
    `preserved_attrs`. Additionally, any attribute that is modified within a preserved
    method will be preserved.

    Freezing currently only accepts ScriptModules that are in eval mode.

    Freezing applies generic optimization that will speed up your model regardless of machine.
    To further optimize using server-specific settings, run `optimize_for_inference` after
    freezing.

    Args:
        mod (:class:`ScriptModule`): a module to be frozen

        preserved_attrs (Optional[List[str]]): a list of attributes to preserve in addition to the forward method.
        Attributes modified in preserved methods will also be preserved.

        optimize_numerics (bool): If ``True``, a set of optimization passes will be run that does not strictly
        preserve numerics. Full details of optimization can be found at `torch.jit.run_frozen_optimizations`.

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

    Note:
        Because freezing makes weights constants and removes module hierarchy, `to` and other
        nn.Module methods to manipulate device or dtype no longer work. As a workaround,
        You can remap devices by specifying `map_location` in `torch.jit.load`, however
        device-specific logic may have been baked into the model.
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
    run_frozen_optimizations(out, optimize_numerics)

    return out

def run_frozen_optimizations(mod, optimize_numerics: bool = True):
    r"""
    Runs a series of optimizations looking for patterns that occur in frozen graphs.
    The current set of optimizations is:
        - Dropout Removal
        - Conv -> Batchnorm folding
        - Conv -> Add/Sub folding
        - Conv -> Mul/Div folding

    Args:
        mod (:class:`ScriptModule`): a frozen module to be optimized

        optimize_numerics (bool): If ``True``, a set of optimization passes will be run that does not strictly
        preserve numerics. These optimizations preserve default rtol and atol of `torch.testing.assert_allclose`
        when applied on a single transformation, however in a module where many transformations are applied
        the rtol or atol may no longer fall within the default `assert_allclose` tolerance. Conv -> Batchnorm folding,
        Conv-Add/Sub, and Conv -> Mul/Div folding all may alter numerics.

    Returns:
        None

    Note:
        In rare occassions, this can result in slower execution.

    Example (Freezing a module with Conv->Batchnorm)
    .. code-block:: python
        import torch
        in_channels, out_channels = 3, 32
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=True)
        bn = torch.nn.BatchNorm2d(out_channels, eps=.001)
        mod = torch.nn.Sequential(conv, bn)
        # set optimize to False here, by default freezing runs run_frozen_optimizations
        frozen_mod = torch.jit.freeze(torch.jit.script(mod.eval()), optimize=False)
        # inspect frozen mod
        assert "batch_norm" in str(frozen_mod.graph)
        torch.jit.run_frozen_optimizations(frozen_mod)
        assert "batch_norm" not in str(frozen_mod.graph)

    """
    # xxx: keep in sync with frozen_graph_optimization.cpp
    # intentionally duplicated to make to make it easier to create custom optimization sequence
    torch._C._jit_pass_remove_dropout(mod._c)
    if optimize_numerics:
        # run a couple times to capture Conv -> Mul -> Add etc
        for _ in range(2):
            torch._C._jit_pass_fold_frozen_conv_bn(mod.graph)
            torch._C._jit_pass_fold_frozen_conv_add_or_sub(mod.graph)
            torch._C._jit_pass_fold_frozen_conv_mul_or_div(mod.graph)

def optimize_for_inference(mod: ScriptModule) -> ScriptModule:
    """
    Performs a set of optimization passes to optimize a model for the
    purposes of inference. If the model is not already frozen, optimize_for_inference
    will invoke `torch.jit.freeze` automatically.

    In addition to generic optimizations that should speed up your model regardless
    of environment, prepare for inference will also bake in build specific settings
    such as the presence of CUDNN or MKLDNN, and may in the future make transformations
    which speed things up on one machine but slow things down on another. Accordingly,
    serialization is not implemented following invoking `optimize_for_inference` and
    is not guaranteed.

    This is still in prototype, and may have the potential to slow down your model.
    Primary use cases that have been targeted so far have been vision models on cpu
    and gpu to a lesser extent.
    """
    if not isinstance(mod, ScriptModule):
        raise RuntimeError(
            "optimize_for_inference expects a ScriptModule as input. "
            "Please use torch.jit.script or torch.jit.trace to script your 'nn.Module'.")

    if hasattr(mod, "training"):
        mod = freeze(mod.eval())

    torch._C._jit_pass_convert_frozen_ops_to_mkldnn(mod.graph)
    torch._C._jit_pass_fuse_frozen_conv_add_relu(mod.graph)
    return mod
