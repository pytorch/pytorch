import torch
from torch import Tensor
import contextlib
from typing import Iterator
from torch.utils._pytree import tree_map
from functools import partial
from torch.utils._python_dispatch import enable_python_mode

# TODO: move this into library proper
@contextlib.contextmanager
def no_dispatch() -> Iterator[None]:
    guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
    try:
        yield
    finally:
        del guard

def check_attr_consistency(wrapper_tensor, metadata_name, metadata_accessor):
    elem = wrapper_tensor.elem
    metadata_wrapper_tensor = metadata_accessor(wrapper_tensor)
    metadata_elem = metadata_accessor(elem)
    if metadata_wrapper_tensor == metadata_elem:
        return
    raise RuntimeError(
        f"This operator is not CompositeImplicitAutograd compliant: the "
        f"{metadata_name} of the tensor was modified directly without "
        f"going through the PyTorch dispatcher.")

def check_metadata_consistency(wrapper_tensor):
    if not isinstance(wrapper_tensor, CompositeCompliantTensor):
        return
    things_to_check = {
        'shape': Tensor.size,
        'dtype': lambda x: x.dtype,
        'device': lambda x: x.device,
        'numel': Tensor.numel,
        'stride': Tensor.stride,
        'storage_offset': Tensor.storage_offset,
    }
    for metadata_name, metadata_accessor in things_to_check.items():
        check_attr_consistency(wrapper_tensor, metadata_name, metadata_accessor)

def is_view_fn(func):
    return func.__name__ in {
        'as_strided',
        'detach',
        'diagonal',
        'expand',
        'expand_as',
        'movedim',
        'narrow',
        'permute',
        'select',
        'squeeze',
        'transpose',
        't',
        'real',
        'imag',
        'view_as_real',
        'view_as_complex',
        'unflatten',
        'unfold',
        'unsqueeze',
        'view',
        'view_as',
        'unbind',
        'split',
        'split_with_sizes',
        'vsplit',
        'hsplit',
        'tensor_split',
        'chunk',
        'swapaxes',
        'slice',
        '_reshape_alias',
        '_unsafe_view',
        '_conj',
        'alias',
    }

# manually populated from native_functions that have inplace_view: True.
# In the future we will probably be able to grab that list directly
def is_inplace_view_fn(func):
    return func.__name__ in {
        'as_strided_',
        'detach_',
        'squeeze_',
        'swapaxes_',
        'swapdims_',
        't_',
        'transpose_',
        'unsqueeze_',
    }

class CompositeCompliantTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem']

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        # The storage of CompositeCompliantTensor should never be used directly
        # by a CompositeImplicitAutograd operation; if the CompositeImplicitAutograd
        # operator attempts to read from the storage without dispatching then it'll
        # raise a RuntimeError due to it being a meta storage.
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, elem.size(),
            dtype=elem.dtype, layout=elem.layout,
            device=elem.device, requires_grad=elem.requires_grad,
            strides=elem.stride(), storage_offset=elem.storage_offset())
        r.elem = elem
        return r

    def __repr__(self):
        return f"CompositeCompliantTensor({self.elem})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            return e.elem if isinstance(e, CompositeCompliantTensor) else e

        def wrap(e):
            return CompositeCompliantTensor(e) if isinstance(e, torch.Tensor) else e

        if func.__name__ in ('set_', 'resize_'):
            raise RuntimeError(
                f"{func.__name__} is not allowed to be called inside of "
                f"CompositeImplicitAutograd operators.")

        with no_dispatch():
            unwrapped_args = tree_map(unwrap, args)
            unwrapped_kwargs = tree_map(unwrap, kwargs)
            unwrapped_rs = func(*unwrapped_args, **unwrapped_kwargs)
            rs = tree_map(wrap, unwrapped_rs)

        if is_view_fn(func):
            # Autograd asserts that for B = A.view_fn(...), B and A's storages
            # are the same. Here we try to make B alias A to avoid those asserts.
            # See https://github.com/pytorch/pytorch/issues/65339 for more information
            # about the issue.
            with no_dispatch():
                # Idea: this is a weird way of getting a storage that aliases the input.
                # This is a workaround for #65339.
                # 1. under no_dispatch, all of the wrapper tensors look like regular
                #    tensors with special storage (the storage is nullptr and
                #    advertises CPU/CUDA device.
                # 2. we run func, which ends up running the view operation
                # 3. All view operations reuse the input's storage and return
                #    result Tensor(s) with new sizes/strides/offset that alias
                #    the input.
                # 4. we set the storage (and sizes/strides/offset) of the wrapper
                #    tensor results to be that of the tensors that alias the input
                result = func(*args, **kwargs)
                if isinstance(result, tuple) or isinstance(result, list):
                    for a, b in zip(rs, result):
                        a.set_(b)
                else:
                    rs.set_(result)

        # Some operations are allowed to in-place modify the metadata of the
        # inputs. The only ones are the "inplace view functions"; when we
        # run into these, we manually modify the metadata of the input.
        with no_dispatch():
            if is_inplace_view_fn(func):
                func(*args, **kwargs)

        # For each CompositeCompliantTensor t, we check that t and t.elem
        # have consistent metadata. If they don't have consistent metadata,
        # that means the operator did something fishy.
        check = partial(check_metadata_consistency)
        tree_map(check, args)
        tree_map(check, kwargs)
        tree_map(check, rs)
        return rs

# The general strategy is to wrap all Tensor args and kwargs in
# CompositeCompliantTensor wrappers. If an operator that is
# CompositeImplicitAutograd does any non-compliant behavior,
# CompositeCompliantTensor will raise an error.
def _check_composite_compliance(op, args, kwargs):
    def wrap(e):
        return CompositeCompliantTensor(e) if isinstance(e, torch.Tensor) else e

    args = tree_map(wrap, args)
    kwargs = tree_map(wrap, kwargs)
    try:
        with enable_python_mode(CompositeCompliantTensor):
            op(*args, **kwargs)
    except RuntimeError as err:
        raise RuntimeError("CompositeImplicitAutograd compilance check failed with "
                           "the above error. If you are adding an OpInfo of an "
                           "existing operator, please feel free to skip this test "
                           "because the problem was pre-existing and file an issue. "
                           "Otherwise, if you added a new operator, please read "
                           "through the CompositeImplicitAutograd Compliance section in "
                           "aten/src/ATen/native/README.md for how to resolve this. "
                           ) from err
