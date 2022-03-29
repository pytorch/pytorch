import torch
from torch import Tensor
import contextlib
import itertools
from typing import Iterator
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from functools import partial
from torch.utils._python_dispatch import enable_python_mode
import re

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
        f"This operator is not Composite Compliant: the "
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
    return func.overloadpacket.__name__ in {
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
    return func.overloadpacket.__name__ in {
        'as_strided_',
        'detach_',
        'squeeze_',
        'swapaxes_',
        'swapdims_',
        't_',
        'transpose_',
        'unsqueeze_',
    }


# Introspection please save us
def is_inplace(func):
    name = func.overloadpacket.__name__
    if re.match('__i.+__', name):
        return True
    if re.match('__.+__', name):
        return False
    return name[-1] == '_'


class CompositeCompliantTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem']

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        # The storage of CompositeCompliantTensor should never be used directly
        # by a Composite operation; if the Composite
        # operator attempts to read from the storage without dispatching then it'll
        # raise a RuntimeError due to it being a meta storage.
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, elem.size(),
            dtype=elem.dtype, layout=elem.layout,
            device=elem.device, requires_grad=elem.requires_grad,
            strides=elem.stride(), storage_offset=elem.storage_offset())

        # CompositeCompliantTensor steals the "requires_grad"-ness.
        if elem.requires_grad:
            # Why clone? Because sometimes OpInfo shares inputs between tests...
            r.elem = elem.detach().clone()
        else:
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

        if func.overloadpacket.__name__ in ('set_', 'resize_'):
            raise RuntimeError(
                f"{func.__name__} is not allowed to be called inside of "
                f"Composite operators.")

        if is_inplace(func):
            # NB: We are making an assumption that if the function is in-place,
            # then the first argument is being written to. Introspection please save us!
            mutated_argument = args[0]
            if not isinstance(mutated_argument, CompositeCompliantTensor) and \
                    any([isinstance(a, CompositeCompliantTensor) for a in args[1:]]):
                raise RuntimeError(
                    'Not composite compliant: performing in-place operation '
                    f'{func.__name__} where the Tensor being written to is '
                    'regular Tensor but the other tensors are Tensor Subclasses. '
                    'Please try to avoid this in-place operation.')

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


def is_tensorlist(lst):
    if not isinstance(lst, list) and not isinstance(lst, tuple):
        return False
    if len(lst) == 0:
        return False
    all_tensors = all([isinstance(elt, torch.Tensor) for elt in lst])
    if all_tensors:
        return True
    exists_one_tensor = all([isinstance(elt, torch.Tensor) for elt in lst])
    if exists_one_tensor:
        raise RuntimeError('This test assumes that PyTorch APIs cannot take '
                           'mixed lists of Tensor and other things')
    return False


def maybe_map(fn, should_map, arg):
    return fn(arg) if should_map else arg


def wrap(arg):
    if isinstance(arg, torch.Tensor):
        return CompositeCompliantTensor(arg)
    if is_tensorlist(arg):
        return [CompositeCompliantTensor(a) for a in arg]
    raise RuntimeError("wrap assumes that the input can be wrapped")


# Given a list of flat arguments, some of which may be Tensors, return all
# possible ways some of the arguments could be CompositeCompliantTensors (CCT).
# For example, given Tensors A, B, C and flat_args = [A, 1, B],
# We would return the following 4 options:
# [CCT(A), 1, CCT(B)]
# [CCT(A), 1, B]
# [A, 1, CCT(B)]
# [A, 1, B]
# NB: Yes, this is exponential. No, we don't care too much because PyTorch ops
# don't accept that many input Tensors.
def generate_subclass_choices(flat_args):
    is_tensor_likes = [isinstance(arg, torch.Tensor) or is_tensorlist(arg) for arg in flat_args]
    subclass_options = [[False, True] if is_tensor_like else [False] for is_tensor_like in is_tensor_likes]

    for which_args_are_wrapped in itertools.product(*subclass_options):
        result = [maybe_map(wrap, should_wrap_arg, arg)
                  for should_wrap_arg, arg in zip(which_args_are_wrapped, flat_args)]
        yield result, which_args_are_wrapped


# For an operation f(*args, **kwargs), each Tensor argument may either be
# a regular Tensor or a Tensor Subclass. This iterator iterates through
# all of those options.
def generate_subclass_choices_args_kwargs(args, kwargs):
    flat_kwargs, spec = tree_flatten(kwargs)
    flat_args_kwargs = list(args) + list(flat_kwargs)
    for choice, debug_metadata in generate_subclass_choices(flat_args_kwargs):
        new_args = choice[:len(args)]
        new_kwargs = tree_unflatten(choice[len(args):], spec)
        which_args_are_wrapped = debug_metadata[:len(args)]
        which_kwargs_are_wrapped = tree_unflatten(debug_metadata[len(args):], spec)
        yield new_args, new_kwargs, which_args_are_wrapped, which_kwargs_are_wrapped


def raise_composite_compliance_error(err, additional_info=''):
    raise RuntimeError(
        "Composite compilance check failed with "
        "the above error.\n"
        f"{additional_info}"
        "If you are adding an OpInfo of an "
        "existing operator, please feel free to skip this test "
        "because the problem was pre-existing and file an issue. "
        "Otherwise, if you added a new operator, please read "
        "through the Composite Compliance section in "
        "aten/src/ATen/native/README.md for how to resolve this. "
    ) from err


# This test checks ALL possible permutations of calling `op` with arguments
# that are individually either a regular Tensor or a Tensor subclass.
#
# The general strategy is to wrap some Tensor args and kwargs in
# CompositeCompliantTensor wrappers and call the operation.

# If some composite operation does any non-compliant behavior,
# CompositeCompliantTensor will raise an error.
def check_all_permutations(op, args, kwargs):
    def wrap(e):
        return CompositeCompliantTensor(e) if isinstance(e, torch.Tensor) else e

    for choice in generate_subclass_choices_args_kwargs(args, kwargs):
        new_args, new_kwargs, which_args_are_wrapped, which_kwargs_are_wrapped = choice

        try:
            op(*new_args, **new_kwargs)
        # NOTE: [What errors are Composite Compiance trying to catch?]
        #
        # There's two things we want to catch:
        # - errors that would raise within the torch_dispatch impl
        # - data_ptr accesses
        # The first is easy to filter for (we could make the error a different
        # error class), the second is always going to be a RuntimeError due to
        # how it is implemented (if you try to access the data_ptr of thex
        # wrapper Tensor, it raises you some internal RuntimeError).
        #
        # So the most general thing to catch here was RuntimeError. If you
        # are here and debugging why your test failed, it's plausible that
        # the operator itself is broken and that there are other tests failing.
        except RuntimeError as err:
            raise_composite_compliance_error(
                err,
                f"- wrapped_args: {which_args_are_wrapped}\n"
                f"- wrapped_kwargs: {which_kwargs_are_wrapped}\n"
            )

# Checks via the usage of Python mode certain anti-patterns that
# are not composite compliant.
#
# In particular, the anti-pattern we are trying to prevent is a user
# creating an empty tensor and then resize_-ing it. Python Mode helps
# here because all factory functions will create tensors that are
# CompositeCompliantTensor.
#
# The general strategy is to wrap all Tensor args and kwargs in
# CompositeCompliantTensor wrappers. If an operator that is
# Composite does any non-compliant behavior,
# CompositeCompliantTensor will raise an error.
def check_with_mode(op, args, kwargs):
    def wrap(e):
        return CompositeCompliantTensor(e) if isinstance(e, torch.Tensor) else e

    args = tree_map(wrap, args)
    kwargs = tree_map(wrap, kwargs)
    try:
        with enable_python_mode(CompositeCompliantTensor):
            op(*args, **kwargs)
    # see NOTE: [What errors are Composite Compiance trying to catch?]
    except RuntimeError as err:
        raise_composite_compliance_error(err)

def gather_leaf_tensors(args, kwargs):
    leaf_tensors = []
    args, args_spec = tree_flatten(args)
    kwargs, kwargs_spec = tree_flatten(kwargs)
    args = args + kwargs
    for arg in args:
        if not isinstance(arg, torch.Tensor):
            continue
        if arg.requires_grad:
            leaf_tensors.append(arg)
    return leaf_tensors


# Checks if the backward formula is composite compliant by testing
# all possible permutations of {inputs, grad_outputs} being
# CompositeCompliantTensor or regular Tensors.
def check_backward_formula(op, args, kwargs):
    assert op.supports_autograd

    for choice in generate_subclass_choices_args_kwargs(args, kwargs):
        new_args, new_kwargs, which_args_are_wrapped, which_kwargs_are_wrapped = choice
        leaf_tensors = gather_leaf_tensors(new_args, new_kwargs)
        assert len(leaf_tensors) > 0

        try:
            results = op(*new_args, **new_kwargs)
        # see NOTE: [What errors are Composite Compiance trying to catch?]
        except RuntimeError as err:
            raise_composite_compliance_error(
                err,
                f"- wrapped_args: {which_args_are_wrapped}\n"
                f"- wrapped_kwargs: {which_kwargs_are_wrapped}\n"
            )

        # Hack: tree_flatten doesn't handle torch.return_types yet,
        # so we're gonna convert them to tuple.
        # TODO: https://github.com/pytorch/pytorch/issues/74624
        if isinstance(results, tuple):
            results = tuple(results)
        flat_results, _ = tree_flatten(results)
        flat_diff_results = [r for r in flat_results if r.requires_grad]
        assert len(flat_diff_results) > 0

        # NB: ones, not ones_like, so we get a regular Tensor here
        grads = [torch.ones(r.shape, device=r.device, dtype=r.dtype)
                 for r in flat_diff_results]
        for flat_new_grads, which_grad_is_batched in generate_subclass_choices(grads):
            try:
                torch.autograd.grad(flat_diff_results, leaf_tensors, flat_new_grads,
                                    allow_unused=True, retain_graph=True)
            # see NOTE: [What errors are Composite Compiance trying to catch?]
            except RuntimeError as err:
                raise_composite_compliance_error(
                    err,
                    f"- wrapped_args: {which_args_are_wrapped}\n"
                    f"- wrapped_kwargs: {which_kwargs_are_wrapped}\n"
                    f"- wrapped_grads: {which_grad_is_batched}\n"
                )
