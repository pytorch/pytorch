import torch


@torch.library.impl_abstract("_TorchScriptTesting::takes_foo")
def fake_takes_foo(foo, z):
    return foo.add_tensor(z)
