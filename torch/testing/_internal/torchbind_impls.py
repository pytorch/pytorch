import torch


def register_if_not(qualname):
    entry = torch._library.simple_registry.singleton.find(qualname)
    if entry.abstract_impl.kernel is None:
        return torch.library.impl_abstract(qualname)
    else:

        def dummy_wrapper(fn):
            return fn

        return dummy_wrapper


# put these under a function because the corresponding library might not be loaded yet.
def register_fake_operators():
    @register_if_not("_TorchScriptTesting::takes_foo_python_meta")
    def fake_takes_foo(foo, z):
        return foo.add_tensor(z)

    @register_if_not("_TorchScriptTesting::queue_pop")
    def fake_queue_pop(tq):
        return tq.pop()

    @register_if_not("_TorchScriptTesting::queue_push")
    def fake_queue_push(tq, x):
        return tq.push(x)

    @register_if_not("_TorchScriptTesting::queue_size")
    def fake_queue_size(tq):
        return tq.size()
