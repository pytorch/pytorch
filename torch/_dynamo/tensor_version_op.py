# mypy: allow-untyped-defs
import torch
from torch._prims import _make_prim, RETURN_TYPE
from torch._subclasses import FakeTensorMode
from torch._subclasses.functional_tensor import FunctionalTensorMode


_tensor_version = _make_prim(
    schema="_tensor_version(Tensor self) -> SymInt",
    return_type=RETURN_TYPE.NEW,
    meta=torch.ops.aten._version.default,
    impl_aten=torch.ops.aten._version.default,
    doc="Tracable unbacked SymInt version of torch.Tensor._version",
)


@_tensor_version.py_impl(FakeTensorMode)
def _tensor_version_fake(fake_mode, self_tensor):
    """
    The initial dynamo capture of _tensor_version + _unsafe_set_version_counter turns the
    `._version` into an unbacked SymInt so that we don't need to specialize on the `._version`
    of input tensors to the graph.
    """
    return fake_mode.shape_env.create_unbacked_symint()


_unsafe_set_version_counter = _make_prim(
    schema="_unsafe_set_version_counter(Tensor self, SymInt version) -> ()",
    return_type=RETURN_TYPE.NEW,
    meta=lambda self, version: None,
    impl_aten=torch._C._autograd._unsafe_set_version_counter,
    doc="Tracable+SymInt version of torch._C._autograd._unsafe_set_version_counter",
)
torch.fx.node.has_side_effect(_unsafe_set_version_counter)


"""
When we functionalize _tensor_version + _unsafe_set_version_counter,
the ops disappear from the traced graph.  We run them eagerly on the
fake tensors used for tracing, in order to get past asserts that would
fail in autograd.

Why is this ok?
1) Versions on functional tensors don't make any sense since you can't mutate a functional tensor.
2) The whole point of version munging is to trick autograd into doing what we want, and after
   AotAtuograd there is no longer any need for these ops.

Note this is similar to how no_grad is handled.
"""


@_tensor_version.py_impl(FunctionalTensorMode)
def _tensor_version_functional(mode, self):
    return self._version


@_unsafe_set_version_counter.py_impl(FunctionalTensorMode)
def _unsafe_set_version_counter_functional(ctx, self, version):
    torch._C._autograd._unsafe_set_version_counter(self, version)
