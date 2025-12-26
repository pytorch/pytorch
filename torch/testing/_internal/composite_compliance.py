# mypy: ignore-errors

import torch
from torch import Tensor
import itertools

from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from torch.utils import _pytree as pytree
from functools import partial
from torch.utils._mode_utils import no_dispatch, all_same_mode
import torch.autograd.forward_ad as fwAD
from collections.abc import Callable
import re


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

def check_metadata_consistency(wrapper_tensor, CCT):
    # CCT: CompositeCompliantTensor class which is generated using generate_cct
    if not isinstance(wrapper_tensor, CCT):
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


def generate_cct_and_mode(autograd_view_consistency=True):
    # This function returns a new class CompositeCompliantTensor
    # The two arguments control the behaviour described below.

    # autograd_view_consistency:
    #   If True, alias result using `set_` if func returns a view
    #   (See Note [Alias Result]).
    #   Since Forward AD doesn't work with `set_`
    #   we disable it by setting alias to False.

    class CompositeCompliantTensor(torch.Tensor):
        elem: torch.Tensor

        __slots__ = ['elem']

        @staticmethod
        def __new__(cls, elem, mode, *args, **kwargs):
            assert type(elem) is not cls, \
                "Wrapping a CompositeCompliantTensor in a CompositeCompliantTensor is not supported"

            # The storage of CompositeCompliantTensor should never be used directly
            # by a Composite operation; if the Composite
            # operator attempts to read from the storage without dispatching then it'll
            # raise a RuntimeError due to it being a meta storage.
            r = torch.Tensor._make_wrapper_subclass(
                cls, elem.size(),
                dtype=elem.dtype, layout=elem.layout,
                device=elem.device, requires_grad=elem.requires_grad,
                strides=elem.stride(), storage_offset=elem.storage_offset())

            if elem.requires_grad:
                # CompositeCompliantTensor steals the "requires_grad"-ness.
                # Why a new copy of `elem`? Because sometimes OpInfo shares inputs between tests...
                tmp = torch.empty(
                    (),
                    dtype=elem.dtype,
                    device=elem.device,
                    layout=elem.layout,
                    requires_grad=False,
                )
                # Use set_ rather than empty_strided() + copy_ so that we can preserve
                # things like storage_offset.
                tmp.set_(
                    source=elem.untyped_storage().clone(),
                    storage_offset=elem.storage_offset(),
                    size=elem.size(),
                    stride=elem.stride(),
                )
                r.elem = tmp
            else:
                r.elem = elem

            assert r.stride() == r.elem.stride()

            # Propagate conjugate bits to the wrapper tensor
            # Ref: https://github.com/albanD/subclass_zoo/issues/24
            # Ref: https://github.com/albanD/subclass_zoo/issues/21
            torch._C._set_conj(r, r.elem.is_conj())
            torch._C._set_neg(r, r.elem.is_neg())

            r.mode = mode
            return r

        def __repr__(self):
            return f"CompositeCompliantTensor({self.elem})"

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            all_args = pytree.arg_tree_leaves(*args, **(kwargs or {}))
            modes = tuple(e.mode for e in all_args if isinstance(e, CompositeCompliantTensor))
            if not all_same_mode(modes):
                raise RuntimeError("Multiple CompositeCompliantTensorModes NYI")
            with modes[0]:
                return func(*args, **kwargs)

    class CompositeCompliantTensorMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            def unwrap(e):
                return e.elem if isinstance(e, CompositeCompliantTensor) else e

            def wrap(e):
                return CompositeCompliantTensor(e, self) if isinstance(e, torch.Tensor) else e

            if func is torch.ops.aten._local_scalar_dense.default:
                raise RuntimeError(
                    ".item() is not allowed to be called inside of composite "
                    "functions in the PyTorch library because not all backends "
                    "and/or Tensor subclasses (e.g. vmap, ProxyTensor) support them.")

            if func.overloadpacket.__name__ in ('set_', 'resize_'):
                raise RuntimeError(
                    f"{func.__name__} is not allowed to be called inside of "
                    f"Composite operators.")

            if is_inplace(func):
                # NB: We are making an assumption that if the function is in-place,
                # then the first argument is being written to. Introspection please save us!
                mutated_argument = args[0]
                if not isinstance(mutated_argument, CompositeCompliantTensor) and \
                        any(isinstance(a, CompositeCompliantTensor) for a in args[1:]):
                    raise RuntimeError(
                        'Not composite compliant: performing in-place operation '
                        f'{func.__name__} where the Tensor being written to is '
                        'regular Tensor but the other tensors are Tensor Subclasses. '
                        'Please try to avoid this in-place operation.')

            unwrapped_args = tree_map(unwrap, args)
            unwrapped_kwargs = tree_map(unwrap, kwargs)
            unwrapped_rs = func(*unwrapped_args, **unwrapped_kwargs)
            rs = tree_map(wrap, unwrapped_rs)

            if is_view_fn(func) and autograd_view_consistency:
                # Note [Alias Result]
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
                    if isinstance(result, (tuple, list)):
                        for a, b in zip(rs, result, strict=True):
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
            check = partial(check_metadata_consistency, CCT=CompositeCompliantTensor)
            pytree.tree_map_(check, args)
            pytree.tree_map_(check, kwargs)
            pytree.tree_map_(check, rs)
            return rs

    return CompositeCompliantTensor, CompositeCompliantTensorMode()

def is_tensorlist(lst):
    if not isinstance(lst, list) and not isinstance(lst, tuple):
        return False
    if len(lst) == 0:
        return False
    all_tensors = all(isinstance(elt, torch.Tensor) for elt in lst)
    if all_tensors:
        return True
    exists_one_tensor = all(isinstance(elt, torch.Tensor) for elt in lst)
    if exists_one_tensor:
        raise RuntimeError('This test assumes that PyTorch APIs cannot take '
                           'mixed lists of Tensor and other things')
    return False


def maybe_map(fn, should_map, arg):
    return fn(arg) if should_map else arg


def wrap(arg, CCT, cct_mode):
    # CCT: CompositeCompliantTensor class which is generated using generate_cct_and_mode
    if isinstance(arg, torch.Tensor):
        return CCT(arg, cct_mode)
    if is_tensorlist(arg):
        return [CCT(a, cct_mode) for a in arg]
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
def generate_subclass_choices(flat_args, CCT, cct_mode):
    # CCT: CompositeCompliantTensor class which is generated using generate_cct_and_mode
    is_tensor_likes = [isinstance(arg, torch.Tensor) or is_tensorlist(arg) for arg in flat_args]
    subclass_options = [[False, True] if is_tensor_like else [False] for is_tensor_like in is_tensor_likes]

    for which_args_are_wrapped in itertools.product(*subclass_options):

        result = [maybe_map(partial(wrap, CCT=CCT, cct_mode=cct_mode), should_wrap_arg, arg)
                  for should_wrap_arg, arg in zip(which_args_are_wrapped, flat_args, strict=True)]
        yield result, which_args_are_wrapped


# For an operation f(*args, **kwargs), each Tensor argument may either be
# a regular Tensor or a Tensor Subclass. This iterator iterates through
# all of those options.
def generate_subclass_choices_args_kwargs(args, kwargs, CCT, cct_mode):
    # CCT: CompositeCompliantTensor class which is generated using generate_cct_and_mode
    flat_kwargs, spec = tree_flatten(kwargs)
    flat_args_kwargs = list(args) + list(flat_kwargs)
    for choice, debug_metadata in generate_subclass_choices(flat_args_kwargs, CCT, cct_mode):
        new_args = choice[:len(args)]
        new_kwargs = tree_unflatten(choice[len(args):], spec)
        which_args_are_wrapped = debug_metadata[:len(args)]
        which_kwargs_are_wrapped = tree_unflatten(debug_metadata[len(args):], spec)
        yield new_args, new_kwargs, which_args_are_wrapped, which_kwargs_are_wrapped


def raise_composite_compliance_error(err, additional_info=''):
    raise RuntimeError(
        "Composite compliance check failed with "
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
def check_all_permutations(op, args, kwargs, assert_equal_fn):
    CCT, cct_mode = generate_cct_and_mode()
    expected = op(*args, **kwargs)
    for choice in generate_subclass_choices_args_kwargs(args, kwargs, CCT, cct_mode):
        new_args, new_kwargs, which_args_are_wrapped, which_kwargs_are_wrapped = choice

        try:
            actual = op(*new_args, **new_kwargs)
        # NOTE: [What errors are Composite Compliance trying to catch?]
        #
        # There's two things we want to catch:
        # - errors that would raise within the torch_dispatch impl
        # - data_ptr accesses
        # The first is easy to filter for (we could make the error a different
        # error class), the second is always going to be a RuntimeError due to
        # how it is implemented (if you try to access the data_ptr of the
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

        def unwrap(e):
            return e.elem if isinstance(e, CCT) else e

        assert_equal_fn(tree_map(unwrap, actual), expected)

# Checks via the usage of torch dispatch mode certain anti-patterns that
# are not composite compliant.
#
# In particular, the anti-pattern we are trying to prevent is a user
# creating an empty tensor and then resize_-ing it. Torch Dispatch Mode helps
# here because all factory functions will create tensors that are
# CompositeCompliantTensor.
#
# The general strategy is to wrap all Tensor args and kwargs in
# CompositeCompliantTensor wrappers. If an operator that is
# Composite does any non-compliant behavior,
# CompositeCompliantTensor will raise an error.
def check_with_mode(op, args, kwargs, assert_equal_fn):
    CCT, cct_mode = generate_cct_and_mode()

    def wrap(e):
        return CCT(e, cct_mode) if isinstance(e, torch.Tensor) else e

    expected = op(*args, **kwargs)

    args = tree_map(wrap, args)
    kwargs = tree_map(wrap, kwargs)
    try:
        with cct_mode:
            actual = op(*args, **kwargs)
    # see NOTE: [What errors are Composite Compliance trying to catch?]
    except RuntimeError as err:
        raise_composite_compliance_error(err)

    def unwrap(e):
        return e.elem if isinstance(e, CCT) else e

    assert_equal_fn(tree_map(unwrap, actual), expected)

def gather_leaf_tensors(args, kwargs):
    leaf_tensors = []
    args, _args_spec = tree_flatten(args)
    kwargs, _kwargs_spec = tree_flatten(kwargs)
    args = args + kwargs
    for arg in args:
        if not isinstance(arg, torch.Tensor):
            continue
        if arg.requires_grad:
            leaf_tensors.append(arg)
    return leaf_tensors


def compute_expected_grads(op, args, kwargs, output_process_fn_grad=None, gradcheck_wrapper=None):
    if gradcheck_wrapper is None:
        results = op(*args, **kwargs)
    else:
        results = gradcheck_wrapper(op, *args, **kwargs)

    if output_process_fn_grad is not None:
        results = output_process_fn_grad(results)

    flat_results = pytree.tree_leaves(results)
    flat_results = [r for r in flat_results if isinstance(r, torch.Tensor)]
    flat_diff_results = [r for r in flat_results if r.requires_grad]
    assert len(flat_diff_results) > 0

    grads = [torch.ones(r.shape, device=r.device, dtype=r.dtype) for r in flat_diff_results]
    leaf_tensors = gather_leaf_tensors(args, kwargs)
    assert len(leaf_tensors) > 0
    return torch.autograd.grad(flat_diff_results, leaf_tensors,
                               grads, allow_unused=True, retain_graph=True)


# Checks if the backward formula is composite compliant by testing
# all possible permutations of {inputs, grad_outputs} being
# CompositeCompliantTensor or regular Tensors.
#
# NB: it is important that op is accepted as a Callable and not an OpInfo,
# this means we can apply check_backward_formula to things that aren't OpInfos
# while debugging.
def check_backward_formula(op: Callable, args, kwargs,
                           output_process_fn_grad=None,
                           gradcheck_wrapper=None, assert_equal_fn=None):
    CCT, cct_mode = generate_cct_and_mode()

    expected = compute_expected_grads(op, args, kwargs, output_process_fn_grad, gradcheck_wrapper)

    for choice in generate_subclass_choices_args_kwargs(args, kwargs, CCT, cct_mode):
        new_args, new_kwargs, which_args_are_wrapped, which_kwargs_are_wrapped = choice
        leaf_tensors = gather_leaf_tensors(new_args, new_kwargs)
        assert len(leaf_tensors) > 0

        try:
            if gradcheck_wrapper is None:
                results = op(*new_args, **new_kwargs)
            else:
                results = gradcheck_wrapper(op, *new_args, **new_kwargs)
            if output_process_fn_grad is not None:
                results = output_process_fn_grad(results)
        # see NOTE: [What errors are Composite Compliance trying to catch?]
        except RuntimeError as err:
            raise_composite_compliance_error(
                err,
                f"- wrapped_args: {which_args_are_wrapped}\n"
                f"- wrapped_kwargs: {which_kwargs_are_wrapped}\n"
            )

        flat_results = pytree.tree_leaves(results)
        flat_results = [r for r in flat_results if isinstance(r, torch.Tensor)]
        flat_diff_results = [r for r in flat_results if r.requires_grad]
        assert len(flat_diff_results) > 0

        # NB: ones, not ones_like, so we get a regular Tensor here
        grads = [torch.ones(r.shape, device=r.device, dtype=r.dtype)
                 for r in flat_diff_results]
        for flat_new_grads, which_grad_is_batched in generate_subclass_choices(grads, CCT, cct_mode):
            try:
                actual = torch.autograd.grad(flat_diff_results, leaf_tensors, flat_new_grads,
                                             allow_unused=True, retain_graph=True)
            # see NOTE: [What errors are Composite Compliance trying to catch?]
            except RuntimeError as err:
                raise_composite_compliance_error(
                    err,
                    f"- wrapped_args: {which_args_are_wrapped}\n"
                    f"- wrapped_kwargs: {which_kwargs_are_wrapped}\n"
                    f"- wrapped_grads: {which_grad_is_batched}\n"
                )

            def unwrap(e):
                return e.elem if isinstance(e, CCT) else e

            assert_equal_fn(tuple(map(unwrap, actual)), expected, equal_nan=True)

# Checks if the forward AD formula is composite compliant by testing
# all possible permutations of {primals, tangents} being
# CompositeCompliantTensor or regular Tensors.
#
# NB: it is important that op is accepted as a Callable and not an OpInfo,
# this means we can apply check_forward_ad_formula to things that aren't OpInfos
# while debugging.
def check_forward_ad_formula(op: Callable, args, kwargs, gradcheck_wrapper=None, assert_equal_fn=None):
    CCT, cct_mode = generate_cct_and_mode(autograd_view_consistency=False)

    def maybe_tangent(t):
        assert type(t) is not CCT
        # Generate `tangent` tensor
        # if given object is a Tensor and requires grad is set.
        if isinstance(t, torch.Tensor) and t.requires_grad:
            return torch.randn_like(t)
        elif is_tensorlist(t):
            return [torch.randn_like(e) if e.requires_grad else None for e in t]
        return None

    tangent_args = tuple(maybe_tangent(arg) for arg in args)
    flat_kwargs, spec = tree_flatten(kwargs)
    flat_tangent_kwargs = tuple(maybe_tangent(arg) for arg in flat_kwargs)
    tangent_kwargs = tree_unflatten(flat_tangent_kwargs, spec)

    with fwAD.dual_level():
        def maybe_make_dual(dual):
            # Returns dual tensor if primal is a tensor/tensor subclass
            # with requires_grad set.
            primal, tangent = dual
            if isinstance(primal, torch.Tensor) and primal.requires_grad:
                return fwAD.make_dual(primal.detach(), tangent)
            elif is_tensorlist(primal):
                return tuple(fwAD.make_dual(pri.detach(), tang) if tang is not None else pri
                             for pri, tang in zip(primal, tangent, strict=True))
            return primal

        def compute_expected_grad(args, tangent_args, kwargs, tangent_kwargs):
            op_args = tuple(map(maybe_make_dual, zip(args, tangent_args, strict=True)))
            op_kwargs = {k: maybe_make_dual((v, tangent_kwargs[k])) for k, v in kwargs.items()}

            if gradcheck_wrapper is None:
                return op(*op_args, **op_kwargs)
            return gradcheck_wrapper(op, *op_args, **op_kwargs)

        expected = compute_expected_grad(args, tangent_args, kwargs, tangent_kwargs)
        expected = tree_map(fwAD.unpack_dual, expected)
        expected_primals = tree_map(
            lambda x: x.primal,
            expected,
            is_leaf=lambda x: type(x) is fwAD.UnpackedDualTensor,
        )
        expected_tangents = tree_map(
            lambda x: x.tangent,
            expected,
            is_leaf=lambda x: type(x) is fwAD.UnpackedDualTensor,
        )

        # Permutations of arg and kwargs in CCT.
        for choice in generate_subclass_choices_args_kwargs(args, kwargs, CCT, cct_mode):
            new_args, new_kwargs, which_args_are_wrapped, which_kwargs_are_wrapped = choice

            # Permutations tangent arg and tangent kwargs in CCT.
            for tang_choice in generate_subclass_choices_args_kwargs(tangent_args, tangent_kwargs, CCT, cct_mode):
                new_tang_args, new_tang_kwargs, \
                    which_tang_args_are_wrapped, which_tang_kwargs_are_wrapped = tang_choice

                op_args = tuple(map(maybe_make_dual, zip(new_args, new_tang_args, strict=True)))
                op_kwargs = {k: maybe_make_dual((v, new_tang_kwargs[k])) for k, v in new_kwargs.items()}

                try:
                    if gradcheck_wrapper is None:
                        actual = op(*op_args, **op_kwargs)
                    else:
                        actual = gradcheck_wrapper(op, *op_args, **op_kwargs)
                # see NOTE: [What errors are Composite Compliance trying to catch?]
                except RuntimeError as err:
                    raise_composite_compliance_error(
                        err,
                        f"- wrapped_args: {which_args_are_wrapped}\n"
                        f"- wrapped_kwargs: {which_kwargs_are_wrapped}\n"
                        f"- wrapped_tangent_args: {which_tang_args_are_wrapped}\n"
                        f"- wrapped_tangent_kwargs: {which_tang_kwargs_are_wrapped}\n"
                    )

                def unwrap(e):
                    return e.elem if isinstance(e, CCT) else e

                actual = tree_map(fwAD.unpack_dual, actual)
                actual_primals = tree_map(
                    lambda x: unwrap(x.primal),
                    actual,
                    is_leaf=lambda x: type(x) is fwAD.UnpackedDualTensor,
                )
                actual_tangents = tree_map(
                    lambda x: unwrap(x.tangent),
                    actual,
                    is_leaf=lambda x: type(x) is fwAD.UnpackedDualTensor,
                )
                assert_equal_fn(actual_primals, expected_primals, equal_nan=True)
                assert_equal_fn(actual_tangents, expected_tangents, equal_nan=True)
