# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import torch
import functorch
from functorch import vmap
import torch.utils._pytree as pytree
from functorch_lagging_op_db import functorch_lagging_op_db
from functorch_additional_op_db import additional_op_db
from torch.testing._internal.common_methods_invocations import DecorateInfo
import os
import unittest
from torch.testing._internal.common_device_type import toleranceOverride

IS_FBCODE = os.getenv('FUNCTORCH_TEST_FBCODE') == '1'


def loop(op, in_dims, out_dim, batch_size, *batched_args, **kwarg_values):
    outs = []
    for idx in range(batch_size):
        flat_args, args_spec = pytree.tree_flatten(batched_args)
        flat_dims, dims_spec = pytree.tree_flatten(in_dims)
        assert(args_spec == dims_spec)
        new_args = [a.select(in_dim, idx) if in_dim is not None else a for a, in_dim in zip(flat_args, flat_dims)]
        out = op(*pytree.tree_unflatten(new_args, args_spec), **kwarg_values)
        outs.append(out)

    loop_out = []
    if isinstance(outs[0], torch.Tensor):
        loop_out = torch.stack(outs)
    else:
        for idx in range(len(outs[0])):
            loop_out.append(torch.stack([i[idx] for i in outs], out_dim))
    return loop_out


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
    for choice in itertools.product(options, repeat=num_tensors):
        choices.append(choice)

    assert choices[-1] == (None,) * num_tensors
    return tuple(choices[:-1])

# NB: This is O(2 ** num_tensors).
# num_tensors ranges from 1 to 10, with 2-4 being most common.
# Try not to extravagate it if you're modifying it.
def get_bdim_choices_batch_norm(num_tensors, _, running_mean=None, running_var=None, *args):
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

def get_exhaustive_batched_inputs(arg_values, kwarg_values, batch_size=2):
    flat_args, arg_spec = pytree.tree_flatten(tuple(arg_values))
    is_tensors = [isinstance(a, torch.Tensor) for a in flat_args]
    bdim_choices = get_bdim_choices(sum(is_tensors))

    @memoize
    def get_batched_arg(arg, bdim):
        assert isinstance(arg, torch.Tensor)
        assert bdim is not None
        result, _ = add_batch_dim(arg, bdim, batch_size)
        return result

    for bdim_choice in bdim_choices:
        flat_in_dims = construct_in_dims(bdim_choice, is_tensors)

        flat_batched_args = tuple(arg if in_dim is None else get_batched_arg(arg, in_dim)
                                  for arg, in_dim in zip(flat_args, flat_in_dims))
        batched_args = pytree.tree_unflatten(flat_batched_args, arg_spec)
        in_dims = pytree.tree_unflatten(flat_in_dims, arg_spec)
        yield batched_args, in_dims, kwarg_values


def is_batch_norm_training(op_name, kwarg_values):
    batch_norm_fns = ("nn.functional.batch_norm", "nn.functional.instance_norm")  # instance norm calls batch norm
    if op_name not in batch_norm_fns:
        return False

    # batch norm and instance norm require the value to be a plain bool
    default_training = op_name == "nn.functional.instance_norm"  # instance norm defaults to training, batch norm doesn't
    is_training = tuple(arg for arg in tuple(kwarg_values.values()) if isinstance(arg, bool))
    if len(is_training) == 0:
        return default_training
    else:
        assert len(is_training) == 1
        return is_training[0]


def get_exhaustive_batched_inputs_batch_norm_is_training(arg_values, kwarg_values, batch_size=2):
    flat_args, arg_spec = pytree.tree_flatten(tuple(arg_values))
    is_tensors = [isinstance(a, torch.Tensor) for a in flat_args]
    num_tensors = sum(is_tensors)
    if num_tensors == 1:  # if there's only an input, can't batch it since running_mean/var will be seen as unbatched tensors
        return
    bdim_choices = get_bdim_choices_batch_norm(num_tensors, *arg_values)

    @memoize
    def get_batched_arg(arg, bdim):
        assert isinstance(arg, torch.Tensor)
        assert bdim is not None
        result, _ = add_batch_dim(arg, bdim, batch_size)
        return result

    for bdim_choice in bdim_choices:
        flat_in_dims = construct_in_dims(bdim_choice, is_tensors)

        flat_batched_args = tuple(arg if in_dim is None else get_batched_arg(arg, in_dim)
                                  for arg, in_dim in zip(flat_args, flat_in_dims))
        batched_args = pytree.tree_unflatten(flat_batched_args, arg_spec)
        in_dims = pytree.tree_unflatten(flat_in_dims, arg_spec)
        yield batched_args, in_dims, kwarg_values


def get_fallback_and_vmap_exhaustive(op, arg_values, kwarg_values, is_batch_norm_and_training=False, compute_loop_out=True):
    out_dim = 0
    batch_size = 2

    if is_batch_norm_and_training:
        generator = get_exhaustive_batched_inputs_batch_norm_is_training(arg_values, kwarg_values, batch_size)
    else:
        generator = get_exhaustive_batched_inputs(arg_values, kwarg_values, batch_size)

    for batched_args, in_dims, kwarg_values in generator:
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
        batched_out = vmap(op, in_dims=in_dims, out_dims=out_dim)(*batched_args, **kwarg_values)
        yield (loop_out, batched_out)

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
        expected = pytree.tree_map(add_bdim_if_tensor, batched_out)

        inner_in_dims = (0,) + pytree.tree_map(lambda x: None, in_dims)
        outer_in_dims = (0,) + in_dims
        output = vmap(vmap(f, inner_in_dims), outer_in_dims)(dummy, *batched_args, **kwarg_values)
        yield (expected, output)


def opinfo_in_dict(opinfo, d):
    return (opinfo.name in d) or (f'{opinfo.name}.{opinfo.variant_test_name}' in d)


def xfail(op_name, variant_name='', *, device_type=None, dtypes=None):
    return (op_name, variant_name, device_type, dtypes, True)

# TODO: this doesn't work in python < 3.8


def skip(op_name, variant_name='', *, device_type=None, dtypes=None):
    return (op_name, variant_name, device_type, dtypes, False)


def skipOps(test_case_name, base_test_name, to_skip):
    all_opinfos = functorch_lagging_op_db + additional_op_db
    for xfail in to_skip:
        op_name, variant_name, device_type, dtypes, expected_failure = xfail
        matching_opinfos = [o for o in all_opinfos
                            if o.name == op_name and o.variant_test_name == variant_name]
        assert len(matching_opinfos) >= 1, f"Couldn't find OpInfo for {xfail}"
        for opinfo in matching_opinfos:
            decorators = list(opinfo.decorators)
            if expected_failure:
                decorator = DecorateInfo(unittest.expectedFailure,
                                         test_case_name, base_test_name,
                                         device_type=device_type, dtypes=dtypes)
                decorators.append(decorator)
            else:
                decorator = DecorateInfo(unittest.skip("Skipped!"),
                                         test_case_name, base_test_name,
                                         device_type=device_type, dtypes=dtypes)
                decorators.append(decorator)
            opinfo.decorators = tuple(decorators)

    # This decorator doesn't modify fn in any way
    def wrapped(fn):
        return fn
    return wrapped


def tol2(op_name, variant_name, override_dct, *, device_type=None):
    return (op_name, variant_name, override_dct, device_type)


def tol1(op_name, override_dct, *, device_type=None):
    return tol2(op_name, '', override_dct, device_type=device_type)


def opsToleranceOverride(test_case_name, base_test_name, overrides):
    all_opinfos = functorch_lagging_op_db + additional_op_db
    for override in overrides:
        op_name, variant_name, override, device_type = override
        matching_opinfos = [o for o in all_opinfos
                            if o.name == op_name and o.variant_test_name == variant_name]
        assert len(matching_opinfos) == 1, f"Couldn't find OpInfo for {override}"
        opinfo = matching_opinfos[0]
        decorators = list(opinfo.decorators)
        decorators.append(DecorateInfo(
            toleranceOverride(override),
            test_case_name, base_test_name, device_type=device_type))
        opinfo.decorators = tuple(decorators)

    # This decorator doesn't modify fn in any way
    def wrapped(fn):
        return fn
    return wrapped


class DisableVmapFallback:
    def __enter__(self):
        self.prev_state = functorch._C._is_vmap_fallback_enabled()
        functorch._C._set_vmap_fallback_enabled(False)

    def __exit__(self, *ignored):
        functorch._C._set_vmap_fallback_enabled(self.prev_state)


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
