from typing import Any

import torch


@torch.jit.script
class MyScriptClass:
    """Intended to be scripted."""

    def __init__(self, x):
        self.foo = x

    def set_foo(self, x):
        self.foo = x


@torch.jit.script
def uses_script_class(x):
    """Intended to be scripted."""
    foo = MyScriptClass(x)
    return foo.foo


class IdListFeature:
    def __init__(self):
        self.id_list = torch.ones(1, 1)

    def returns_self(self) -> "IdListFeature":
        return IdListFeature()


class UsesIdListFeature(torch.nn.Module):
    def forward(self, feature: Any):
        if isinstance(feature, IdListFeature):
            return feature.id_list
        else:
            return feature
