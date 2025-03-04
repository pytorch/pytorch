# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import unittest
from collections import namedtuple

from functorch_additional_op_db import additional_op_db

import torch
import torch.utils._pytree as pytree
from functorch import vmap
from torch.testing._internal.autograd_function_db import autograd_function_db
from torch.testing._internal.common_device_type import toleranceOverride
from torch.testing._internal.common_methods_invocations import DecorateInfo, op_db
from torch.testing._internal.common_modules import module_db
from torch.testing._internal.custom_op_db import custom_op_db
from torch.testing._internal.opinfo.core import sample_skips_and_xfails, XFailRule


IS_FBCODE = os.getenv("FUNCTORCH_TEST_FBCODE") == "1"


def loop(op, in_dims, out_dim, batch_size, *batched_args, **kwarg_values):
    outs = []
    out_spec = None
    for idx in range(batch_size):
        flat_args, args_spec = pytree.tree_flatten(batched_args)
        flat_dims, dims_spec = pytree.tree_flatten(in_dims)
        assert args_spec == dims_spec
        new_args = [
            a.select(in_dim, idx) if in_dim is not None else a
            for a, in_dim in zip(flat_args, flat_dims)
        ]
        out = op(*pytree.tree_unflatten(new_args, args_spec), **kwarg_values)
        flat_out, out_spec = pytree.tree_flatten(out)
        outs.append(flat_out)

    # use the same out_dim for all outputs
    if isinstance(out_dim, int):
        flat_out_dim = [out_dim for _ in flat_out]
    else:
        flat_out_dim, _ = pytree.tree_flatten(out_dim)

    outs = zip(*outs)

    result = []
    for i, out_lst in enumerate(outs):
        if flat_out_dim[i] is not None:
            if not all(isinstance(x, torch.Tensor) for x in out_lst):
                raise ValueError(
                    f"vmap `{op}` must only return "
                    "Tensors. Did you mean to set out_dims= to None for output?"
                )
            result.append(torch.stack(out_lst))
        else:
            # not batched over, result should be the same for all batches
            result.append(out_lst[0])
    return pytree.tree_unflatten(result, out_spec)


# Like loop helper function but for 2 levels of vmap. If we need more levels than this, probably possible
# to generalize the loops function but it seemed too complicated for this
def loop2(
    op,
    in_dims1,
    in_dims2,
    out_dim1,
    out_dim2,
    batch_size1,
    batch_size2,
    *batched_args,
    **kwarg_values,
):
    outs = []
    flat_args, args_spec = pytree.tree_flatten(batched_args)
    flat_dims1, dims_spec1 = pytree.tree_flatten(in_dims1)
    flat_dims2, dims_spec2 = pytree.tree_flatten(in_dims2)
    assert args_spec == dims_spec1
    assert args_spec == dims_spec2
    assert len(flat_dims1) == len(flat_dims2)
    for idx1 in range(batch_size1):
        out_split = []
        arg_split = [
            a.select(in_dim1, idx1) if in_dim1 is not None else a
            for a, in_dim1 in zip(flat_args, flat_dims1)
        ]
        for idx2 in range(batch_size2):
            new_args = [
                a.select(in_dim, idx2) if in_dim is not None else a
                for a, in_dim in zip(arg_split, flat_dims2)
            ]
            out = op(*pytree.tree_unflatten(new_args, args_spec), **kwarg_values)
            out_split.append(out)
        outs.append(out_split)

    loop_out = []
    for out_split in outs:
        if isinstance(out_split[0], torch.Tensor):
            loop_out.append(torch.stack(out_split, out_dim1))
        else:
            new_out = []
            for idx in range(len(out_split[0])):
                new_out.append(torch.stack([i[idx] for i in out_split], out_dim1))
            loop_out.append(new_out)

    new_out = []
    if isinstance(loop_out, torch.Tensor):
        new_out = torch.stack(loop_out, out_dim2)
    else:
        for idx in range(len(loop_out[0])):
            new_out.append(torch.stack([i[idx] for i in loop_out], out_dim2))
    return new_out


def is_valid_inplace_sample_input(sample_input, op, inplace_variant):
    if inplace_variant is None:
        return False
    if sample_input.broadcasts_input:
        return False
    if not isinstance(sample_input.input, torch.Tensor):
        return False

    # Check if input's dtype matches the output's dtype
    args = (sample_input.input,) + sample_input.args
    kwargs = sample_input.kwargs
    output_dtype = op(*args, **kwargs).dtype
    return sample_input.input.dtype == output_dtype


# This is kind of dangerous, please think carefully before using it.
# Known risks:
# - the return better not be mutated so it's best to return immutable types
# (e.g. prefer tuples to list)
# - Don't hash tensors in a global context, that'll keep them around forever
def memoize(fn):
    memo = {}

    def wrapped(*args):
        if args not in memo:
            memo[args] = fn(*args)
        return memo[args]

    return wrapped


# NB: This is O(2 ** num_tensors).
# num_tensors ranges from 1 to 10, with 2-4 being most common.
# Try not to extravagate it if you're modifying it.
@memoize
def get_bdim_choices(num_tensors):
    choices = []

    # full of zeros
    choices.append((0,) * num_tensors)

    # All permutations of (-1, None)
    options = (-1, None)
    choices.extend(itertools.product(options, repeat=num_tensors))

    assert choices[-1] == (None,) * num_tensors
    return tuple(choices[:-1])


# NB: This is O(2 ** num_tensors).
# num_tensors ranges from 1 to 10, with 2-4 being most common.
# Try not to extravagate it if you're modifying it.
def get_bdim_choices_batch_norm(
    num_tensors, _, running_mean=None, running_var=None, *args
):
    choices = []
    options = (-1, None)

    # instance norm turns these into unbatched 0 tensors, so we cannot batch the input if either is not specified
    if running_mean is None or running_var is None:
        choices.append((None,) + (0,) * (num_tensors - 1))
        for choice in itertools.product(options, repeat=num_tensors - 1):
            choices.append((None,) + choice)

    else:
        # running_mean and running_var are specified as tensors. Batch norm doesn't work if the input is batched but
        # running_mean/var are unbatched, so this tests all other cases
        choices.append((0,) * num_tensors)
        for choice in itertools.product(options, repeat=num_tensors):
            input_bdim = choice[0]
            running_mean_bdim = choice[1]
            running_var_bdim = choice[2]
            if input_bdim and (not running_mean_bdim or not running_var_bdim):
                continue
            choices.append(choice)

    assert choices[-1] == (None,) * num_tensors
    return tuple(choices[:-1])


def add_batch_dim(arg, bdim, batch_size=3):
    assert bdim == 0 or bdim == -1
    assert isinstance(arg, torch.Tensor)
    if bdim == 0:
        shape = [1] * len(arg.shape)
        shape.insert(bdim, batch_size)
        return (arg.repeat(shape), bdim)
    if bdim == -1:
        arg = arg.unsqueeze(-1).expand(*arg.shape, batch_size).contiguous()
        return (arg, bdim)


def construct_in_dims(bdim_choice_for_tensors, is_tensors):
    result = []
    bdim = iter(bdim_choice_for_tensors)
    for is_tensor in is_tensors:
        if not is_tensor:
            result.append(None)
            continue
        result.append(next(bdim))
    return tuple(result)


def is_batch_norm_training(op_name, kwarg_values):
    batch_norm_fns = (
        "nn.functional.batch_norm",
        "nn.functional.instance_norm",
    )  # instance norm calls batch norm
    if op_name not in batch_norm_fns:
        return False

    # batch norm and instance norm require the value to be a plain bool
    default_training = (
        op_name == "nn.functional.instance_norm"
    )  # instance norm defaults to training, batch norm doesn't
    is_training = tuple(
        arg for arg in tuple(kwarg_values.values()) if isinstance(arg, bool)
    )
    if len(is_training) == 0:
        return default_training
    else:
        assert len(is_training) == 1
        return is_training[0]


def generate_vmap_inputs(
    arg_values, kwarg_values, is_batch_norm_and_training=False, batch_size=2
):
    flat_args, arg_spec = pytree.tree_flatten(tuple(arg_values))
    is_tensors = [isinstance(a, torch.Tensor) for a in flat_args]
    num_tensors = sum(is_tensors)
    # For Batch Norm, if there's only an input, we can't
    # batch it since running_mean/var will be seen as unbatched tensors
    if num_tensors == 1 and is_batch_norm_and_training:
        return
    bdim_choices = (
        get_bdim_choices_batch_norm(num_tensors, *arg_values)
        if is_batch_norm_and_training
        else get_bdim_choices(num_tensors)
    )

    @memoize
    def get_batched_arg(arg, bdim):
        assert isinstance(arg, torch.Tensor)
        assert bdim is not None
        result, _ = add_batch_dim(arg, bdim, batch_size)
        return result

    for bdim_choice in bdim_choices:
        flat_in_dims = construct_in_dims(bdim_choice, is_tensors)

        flat_batched_args = tuple(
            arg if in_dim is None else get_batched_arg(arg, in_dim)
            for arg, in_dim in zip(flat_args, flat_in_dims)
        )
        batched_args = pytree.tree_unflatten(flat_batched_args, arg_spec)
        in_dims = pytree.tree_unflatten(flat_in_dims, arg_spec)
        yield batched_args, in_dims, kwarg_values


def clone_if_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    return x


# Helper function to compare output of `vmap` against the
# `for-loop` version.
def _compute_quantities_for_vmap_test(
    op,
    orig_batched_args,
    orig_kwarg_values,
    in_dims,
    out_dim,
    batch_size,
    compute_loop_out=True,
    clone_inputs=False,
):
    def maybe_clone_inputs():
        if clone_inputs:
            batched_args = pytree.tree_map(clone_if_tensor, orig_batched_args)
            kwarg_values = pytree.tree_map(clone_if_tensor, orig_kwarg_values)
            return batched_args, kwarg_values
        return orig_batched_args, orig_kwarg_values

    batched_args, kwarg_values = maybe_clone_inputs()

    if compute_loop_out:
        loop_out = loop(op, in_dims, out_dim, batch_size, *batched_args, **kwarg_values)
    else:
        loop_out = None

    # Used for debugging the resulting operations
    # from functorch import make_fx
    # def f(a):
    #     return op(a)
    # t = make_fx(vmap(f, in_dims=in_dims, out_dims=out_dim))(*batched_args, **kwarg_values)
    # print(in_dims, [arg.shape for arg in batched_args], kwarg_values)
    batched_args, kwarg_values = maybe_clone_inputs()
    batched_out = vmap(op, in_dims=in_dims, out_dims=out_dim)(
        *batched_args, **kwarg_values
    )

    # Tests case where we dispatch to a batching rule with no bdims
    # This should be handled by autogenerated plumbing. For vmap support
    # added via a manual plumbing you may need to handle this specially.
    def add_bdim_if_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.unsqueeze(1)
        return x

    def f(dummy, *args, **kwargs):
        return op(*args, **kwargs)

    dummy = torch.ones(batch_size, 1)
    vmapvmap_expected = pytree.tree_map(add_bdim_if_tensor, batched_out)

    inner_in_dims = (0,) + pytree.tree_map(lambda x: None, in_dims)
    outer_in_dims = (0,) + in_dims
    batched_args, kwarg_values = maybe_clone_inputs()
    vmapvmap_output = vmap(
        vmap(f, inner_in_dims, out_dims=out_dim), outer_in_dims, out_dims=out_dim
    )(dummy, *batched_args, **kwarg_values)

    yield (batched_out, loop_out, vmapvmap_output, vmapvmap_expected)


# Function with more friendly return types
# compared to `_compute_quantities_for_vmap_test`
def compute_quantities_for_vmap_test(
    op,
    orig_batched_args,
    orig_kwarg_values,
    in_dims,
    out_dim=0,
    batch_size=2,
    compute_loop_out=True,
    clone_inputs=False,
):
    for quantities in _compute_quantities_for_vmap_test(
        op,
        orig_batched_args,
        orig_kwarg_values,
        in_dims,
        out_dim,
        batch_size,
        compute_loop_out,
        clone_inputs,
    ):
        yield (quantities[0], quantities[1])
        yield (quantities[2], quantities[3])


def get_fallback_and_vmap_exhaustive(
    op,
    arg_values,
    kwarg_values,
    is_batch_norm_and_training=False,
    compute_loop_out=True,
):
    out_dim = 0
    batch_size = 2

    def make_batched(t):
        if isinstance(t, torch.Tensor):
            shape = list(t.shape)
            shape.insert(out_dim, batch_size)
            return t.expand(*shape)
        return t

    # Inputs generated by `generate_vmap_inputs` just copy/expand the unbatched inputs
    # over the batched dimension. Thus we can compute the expected value once and just
    # expand it based on the `out_dim` and `batch_size`.
    expected_unbatched = op(*arg_values, **kwarg_values)
    expected_batched = pytree.tree_map(make_batched, expected_unbatched)
    generator = generate_vmap_inputs(
        arg_values, kwarg_values, is_batch_norm_and_training
    )
    for batched_args, in_dims, kwarg_values in generator:
        for quantities in _compute_quantities_for_vmap_test(
            op,
            batched_args,
            kwarg_values,
            in_dims,
            out_dim,
            batch_size,
            compute_loop_out=False,
        ):
            assert quantities[1] is None
            yield (quantities[0], expected_batched)
            yield (quantities[2], quantities[3])


def opinfo_in_dict(opinfo, d):
    return (opinfo.name in d) or (f"{opinfo.name}.{opinfo.variant_test_name}" in d)


DecorateMeta = namedtuple(
    "DecorateMeta",
    [
        "op_name",
        "variant_name",
        "decorator",
        "device_type",
        "dtypes",
    ],
)


def decorate(
    op_name, variant_name="", *, decorator=None, device_type=None, dtypes=None
):
    assert decorator is not None
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=decorator,
        device_type=device_type,
        dtypes=dtypes,
    )


def xfail(op_name, variant_name="", *, device_type=None, dtypes=None):
    return decorate(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.expectedFailure,
        device_type=device_type,
        dtypes=dtypes,
    )


# fail_fn should be a callable that accepts a single SampleInput and returns True if failure
# is expected
def xfailIf(op_name, fail_fn, variant_name="", *, device_type=None, dtypes=None):
    return decorate(
        op_name=op_name,
        variant_name=variant_name,
        decorator=sample_skips_and_xfails(
            [
                XFailRule(
                    # op matching is already handled by DecorateMeta
                    op_match_fn=lambda device, op: True,
                    # device matching is already handled by DecorateMeta
                    sample_match_fn=lambda device, sample: fail_fn(sample),
                )
            ]
        ),
        device_type=device_type,
        dtypes=dtypes,
    )


def skip(op_name, variant_name="", *, device_type=None, dtypes=None):
    return decorate(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.skip("Skipped!"),
        device_type=device_type,
        dtypes=dtypes,
    )


def skipOps(test_case_name, base_test_name, to_skip):
    all_opinfos = op_db + additional_op_db + autograd_function_db + custom_op_db
    for decorate_meta in to_skip:
        matching_opinfos = [
            o
            for o in all_opinfos
            if o.name == decorate_meta.op_name
            and o.variant_test_name == decorate_meta.variant_name
        ]
        assert len(matching_opinfos) > 0, f"Couldn't find OpInfo for {decorate_meta}"
        assert len(matching_opinfos) == 1, (
            "OpInfos should be uniquely determined by their (name, variant_name). "
            f"Got more than one result for ({decorate_meta.op_name}, {decorate_meta.variant_name})"
        )
        opinfo = matching_opinfos[0]
        decorators = list(opinfo.decorators)
        new_decorator = DecorateInfo(
            decorate_meta.decorator,
            test_case_name,
            base_test_name,
            device_type=decorate_meta.device_type,
            dtypes=decorate_meta.dtypes,
        )
        decorators.append(new_decorator)
        opinfo.decorators = tuple(decorators)

    # This decorator doesn't modify fn in any way
    def wrapped(fn):
        return fn

    return wrapped


def decorateForModules(decorator, module_classes, device_type=None, dtypes=None):
    # This decorator doesn't modify fn in any way
    def wrapped(
        fn,
        module_classes=module_classes,
        decorator=decorator,
        device_type=device_type,
        dtypes=dtypes,
    ):
        name_parts = fn.__qualname__.split(".")
        assert (
            len(name_parts) == 2
        ), "Decorator only applies to a test function of a test class"
        test_case_name, base_test_name = name_parts
        for module_cls in module_classes:
            matching_module_infos = [m for m in module_db if m.module_cls == module_cls]
            assert (
                len(matching_module_infos) == 1
            ), f"Couldn't find single ModuleInfo for {module_cls}"
            module_info = matching_module_infos[0]
            decorators = list(module_info.decorators)
            new_decorator = DecorateInfo(
                decorator,
                test_case_name,
                base_test_name,
                device_type=device_type,
                dtypes=dtypes,
            )
            decorators.append(new_decorator)
            module_info.decorators = tuple(decorators)
        return fn

    return wrapped


def expectedFailureIf(condition):
    def decorator(fn):
        if condition:
            return unittest.expectedFailure(fn)
        return fn

    return decorator


def tol2(op_name, variant_name, override_dct, *, device_type=None):
    return (op_name, variant_name, override_dct, device_type)


def tol1(op_name, override_dct, *, device_type=None):
    return tol2(op_name, "", override_dct, device_type=device_type)


def opsToleranceOverride(test_case_name, base_test_name, overrides):
    all_opinfos = op_db + additional_op_db
    for override in overrides:
        op_name, variant_name, override, device_type = override
        matching_opinfos = [
            o
            for o in all_opinfos
            if o.name == op_name and o.variant_test_name == variant_name
        ]
        assert len(matching_opinfos) == 1, f"Couldn't find OpInfo for {override}"
        opinfo = matching_opinfos[0]
        decorators = list(opinfo.decorators)
        decorators.append(
            DecorateInfo(
                toleranceOverride(override),
                test_case_name,
                base_test_name,
                device_type=device_type,
            )
        )
        opinfo.decorators = tuple(decorators)

    # This decorator doesn't modify fn in any way
    def wrapped(fn):
        return fn

    return wrapped


class DisableVmapFallback:
    def __enter__(self):
        self.prev_state = torch._C._functorch._is_vmap_fallback_enabled()
        torch._C._functorch._set_vmap_fallback_enabled(False)

    def __exit__(self, *ignored):
        torch._C._functorch._set_vmap_fallback_enabled(self.prev_state)


def check_vmap_fallback(test_case, thunk, opinfo, dry_run=False):
    try:
        with DisableVmapFallback():
            thunk()
    except Exception:
        if not dry_run:
            raise
        if opinfo.variant_test_name:
            print(f"xfail('{opinfo.name}', '{opinfo.variant_test_name}'),")
        else:
            print(f"xfail('{opinfo.name}'),")
