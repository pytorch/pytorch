import torch


@torch.library.impl_abstract_class("_TorchScriptTesting::_Foo")
class FakeFoo:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def from_metadata(foo_meta):
        x, y = foo_meta
        return FakeFoo(x, y)

    def add_tensor(self, z):
        return (self.x + self.y) * z


@torch.library.impl_abstract("_TorchScriptTesting::takes_foo")
def fake_takes_foo(foo, z):
    return foo.add_tensor(z)
