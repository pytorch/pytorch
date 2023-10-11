import warnings
from typing import Callable, Union

import torch
import torch.utils._pytree as pytree
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import (
    FakeTensorMode,
    tree_flatten_only,
    UnsupportedFakeTensorException,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


aten = torch._ops.ops.aten


def outputs_alias_inputs(outputs, inputs):
    input_storages = {
        inp._typed_storage()._cdata
        for inp in tree_flatten_only(torch.Tensor, inputs)
        if torch._C._has_storage(inp)
    }
    return any(
        torch._C._has_storage(out) and out._typed_storage()._cdata in input_storages
        for out in tree_flatten_only(torch.Tensor, outputs)
    )


def outputs_are_inputs(outputs, inputs):
    input_ids = {id(inp) for inp in tree_flatten_only(torch.Tensor, inputs)}
    return any(id(out) in input_ids for out in tree_flatten_only(torch.Tensor, outputs))


def output_alias_each_other(outputs):
    storages = set()
    for out in tree_flatten_only(torch.Tensor, outputs):
        if not torch._C._has_storage(out):
            continue
        stor = out._typed_storage()._cdata
        if stor in storages:
            return True
        storages.add(stor)
    return False


class CrossRefFakeMode(TorchDispatchMode):
    def __init__(
        self,
        ignore_op_fn: Union[Callable[[OpOverload], bool], None] = None,
        *,
        check_strides=True,
        check_aliasing=True,
    ):
        self.ignore_op_fn = (
            ignore_op_fn if ignore_op_fn is not None else lambda fn: False
        )
        self.check_strides = check_strides
        self.check_aliasing = check_aliasing

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        fake_r = None

        # empty_like excluded for now due to sparse complex
        # aten._to_dense.default this one is getting called with csc
        if (
            func
            not in (
                aten.lift_fresh.default,
                aten.lift_fresh_copy.default,
                aten.set_.source_Storage_storage_offset,
            )
            and not self.ignore_op_fn(func)
            and torch.Tag.dynamic_output_shape not in func.tags
            and torch.Tag.inplace_view not in func.tags
            and torch.Tag.data_dependent_output not in func.tags
        ):
            try:
                with FakeTensorMode() as fake_mode:
                    fake_args, fake_kwargs = pytree.tree_map_only(
                        torch.Tensor, fake_mode.from_tensor, (args, kwargs)
                    )
                    with warnings.catch_warnings():
                        fake_r = func(*fake_args, **fake_kwargs)
            except UnsupportedFakeTensorException:
                pass

        context = (
            f"When comparing the output of {func} on FakeTensor and concrete Tensors, "
            f"found"
        )
        r = func(*args, **kwargs)
        if fake_r is not None:
            r_flat, _ = tree_flatten(r)
            f_flat, _ = tree_flatten(fake_r)
            assert len(r_flat) == len(
                r_flat
            ), f"{context} mismatch in number of returns {len(f_flat)} != {len(r_flat)}"

            if self.check_aliasing:
                r_aliasing = outputs_alias_inputs(r, (args, kwargs))
                f_aliasing = outputs_alias_inputs(fake_r, (fake_args, fake_kwargs))
                assert (
                    r_aliasing == f_aliasing
                ), f"{context} mismatch in outputs_alias_inputs check {f_aliasing} != {r_aliasing}"

                r_identity_eq = outputs_are_inputs(r, (args, kwargs))
                f_identity_eq = outputs_are_inputs(fake_r, (fake_args, fake_kwargs))
                assert (
                    r_identity_eq == f_identity_eq
                ), f"{context} mismatch in outputs_are_inputs check {f_identity_eq} != {r_identity_eq}"

                r_output_alias_each_other = output_alias_each_other(r)
                f_output_alias_each_other = output_alias_each_other(fake_r)
                assert r_output_alias_each_other == f_output_alias_each_other, (
                    f"{context} mismatch in outputs_alias_each_other check "
                    f"{f_output_alias_each_other} != {r_output_alias_each_other}"
                )

            for idx, (r_out, fake_out) in enumerate(
                zip(tree_flatten(r)[0], tree_flatten(fake_r)[0])
            ):
                r_is_ten = isinstance(r_out, torch.Tensor)
                assert r_is_ten == isinstance(
                    fake_out, torch.Tensor
                ), f"{context} mismatched number of tensor outputs"
                if r_is_ten:
                    assert r_out.requires_grad == fake_out.requires_grad, (
                        f"{context} mismatched requires_grad-ness of outputs. "
                        f"This usually means that you have added autograd support "
                        f"for your operator at a dispatch key other than Autograd, "
                        f"which will lead to problems"
                    )
                    if torch._C._has_storage(r_out):
                        r_offset = r_out.storage_offset()
                        f_offset = fake_out.storage_offset()
                        assert (
                            r_offset == f_offset
                        ), f"{context} mismatched storage offset"

                    try:
                        torch._prims.utils.compare_tensor_meta(
                            r_out, fake_out, check_strides=self.check_strides
                        )
                    except Exception as e:
                        if (
                            func is aten._scaled_dot_product_flash_attention.default
                            and idx in (6, 7)
                            and "Devices" in repr(e)
                        ):
                            continue
                        if (
                            func is aten._scaled_dot_product_efficient_attention.default
                            and idx in (2, 3)
                            and "Devices" in repr(e)
                        ):
                            continue
                        error_message = (
                            f"{context} mismatched tensor metadata: {e}"
                            if len(r_flat) == 1
                            else f"{context} mismatched tensor metadata for output[{idx}]: {e}"
                        )
                        raise RuntimeError(error_message) from e
        return r
