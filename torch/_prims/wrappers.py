import torch
import torch._prims as prims
from torch._prims.utils import (
    Number,
    NumberType,
    TensorLike,
    TensorLikeType,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
)
import torch._prims.utils as utils
from torch.utils._pytree import tree_flatten

from typing import Callable, Sequence, Union
import inspect
from functools import wraps, reduce
import operator
import warnings
from itertools import chain

# TODO: implement ref.cast with an option to enforce safe casting
def _maybe_convert_to_dtype(
    a: Union[TensorLikeType, NumberType, Sequence], dtype: torch.dtype
) -> Union[TensorLikeType, NumberType, Sequence]:
    if isinstance(a, TensorLike):
        if a.dtype != dtype:
            # NOTE: this is incorrect on the CPU
            # See https://github.com/pytorch/pytorch/issues/77553
            return prims.convert_element_type(a, dtype)
        return a
    if isinstance(a, Number):
        return utils.dtype_to_type(dtype)(a)
    if isinstance(a, Sequence):
        return tuple(_maybe_convert_to_dtype(x, dtype) for x in a)

    raise ValueError(
        "Received type {0} that is neither a tensor or a number!".format(type(a))
    )


def _maybe_convert_to_type(a: NumberType, typ: type) -> NumberType:
    if not isinstance(a, Number):
        msg = "Found unknown type {0} when trying to convert scalars!".format(type(a))
        raise ValueError(msg)
    if not utils.is_weakly_lesser_type(type(a), typ):
        msg = "Scalar {0} of type {1} cannot be safely cast to type {2}!".format(
            a, type(a), typ
        )
        raise ValueError(msg)

    return typ(a)


def _annotation_has_type(*, typ, annotation):
    if hasattr(annotation, "__args__"):
        for a in annotation.__args__:
            if _annotation_has_type(typ=typ, annotation=a):
                return True
        return False

    return typ is annotation


class elementwise_type_promotion_wrapper(object):
    """
    Adds elementwise type promotion to a Python reference implementation.

    Takes two kwargs, type_promoting_args and type_promotion_kind.

    type_promoting_args must be a string Sequence specifiying the argument names of all
    arguments that participate in type promotion (and should be type promoted). If the
    arg specifies a Sequence-type then every element of the Sequence will participate in
    type promotion.

    type_promotion_kind must be one of the kinds specified by ELEMENTWISE_TYPE_PROMOTION_KIND.
    See its documentation for details.

    Other type promotion behavior, like validating the Python type of scalar arguments, must
    be handled separately.
    """

    def __init__(
        self,
        *,
        type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND,
        type_promoting_args: Sequence[str] = None,
    ):
        self.type_promoting_arg_names = type_promoting_args
        self.type_promotion_kind = type_promotion_kind

    def __call__(self, fn: Callable) -> Callable:
        sig = inspect.signature(fn)

        @wraps(fn)
        def _fn(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            type_promoting_args = tuple(
                bound.arguments[x]
                for x in self.type_promoting_arg_names  # type: ignore[union-attr]
                if x in bound.arguments.keys()
            )

            flattened_type_promoting_args = tree_flatten(type_promoting_args)[0]
            compute_dtype, result_dtype = utils.elementwise_dtypes(
                *flattened_type_promoting_args,
                type_promotion_kind=self.type_promotion_kind,
            )

            promoted_args = {
                x: _maybe_convert_to_dtype(bound.arguments[x], compute_dtype)
                for x in self.type_promoting_arg_names  # type: ignore[union-attr]
                if x in bound.arguments.keys()
            }
            bound.arguments.update(promoted_args)

            result = fn(**bound.arguments)

            # FIXME?: assumes result is a single tensor
            assert isinstance(result, TensorLike)
            return _maybe_convert_to_dtype(result, result_dtype)

        _fn.__signature__ = sig  # type: ignore[attr-defined]
        return _fn


# TODO: handle tuples of tensors
def _maybe_resize_out(out: TensorLikeType, shape):
    if out.numel() == 0:
        return prims.resize(out, shape)

    if out.numel() != reduce(operator.mul, shape, 1):
        msg = (
            "An output with one or more elements was resized since it had shape {0} "
            "which does not match the required output shape {1}. "
            "This behavior is deprecated, and in a future PyTorch release outputs will not "
            "be resized unless they have zero elements. "
            "You can explicitly reuse an out tensor t by resizing it, inplace, to zero elements with t.resize_(0).".format(
                str(out.shape), str(shape)
            )
        )
        warnings.warn(msg)
        return prims.resize(out, shape)

    return out


def _safe_copy_out(*, copy_from: TensorLikeType, copy_to: TensorLikeType):
    # Checks same device
    if copy_from.device != copy_to.device:
        msg = "Attempting to copy from device {0} to device {1}, but cross-device copies are not allowed!".format(
            copy_from.device, copy_to.device
        )
        raise RuntimeError(msg)

    # Checks safe cast
    if not utils.can_safe_cast_to(cast_from=copy_from.dtype, cast_to=copy_to.dtype):
        msg = "Attempting to cast from {0} to out tensor with dtype {1}, but this can't be cast because it is not safe!".format(
            copy_from.dtype, copy_to.dtype
        )
        raise RuntimeError(msg)

    return prims.copy_to(copy_to, copy_from)


# FIXME: only supports out parameter that is literally called "out"
def out_wrapper(fn: Callable) -> Callable:
    """
    Adds the out parameter to a Python reference.

    Note that this currently only supports operations that return a single tensor.
    """

    @wraps(fn)
    def _fn(*args, out=None, **kwargs):
        result = fn(*args, **kwargs)
        if out is not None:
            assert isinstance(out, TensorLike)
            out = _maybe_resize_out(out, result.shape)
            return _safe_copy_out(copy_from=result, copy_to=out)  # type: ignore[arg-type]
            return out
        return result

    sig = inspect.signature(fn)
    out_param = inspect.Parameter(
        "out",
        kind=inspect.Parameter.KEYWORD_ONLY,
        default=None,
        annotation=TensorLikeType,
    )
    params = chain(sig.parameters.values(), (out_param,))
    _fn.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=params, return_annotation=sig.return_annotation  # type: ignore[arg-type]
    )
    _fn.__annotations__ = fn.__annotations__
    _fn.__annotations__["out"] = TensorLikeType
    return _fn


def out_wrapper_multi(*out_names):
    def go(fn: Callable) -> Callable:
        @wraps(fn)
        def _fn(*args, **kwargs):
            out_kwargs = {}
            has_out_kwargs = None
            for o in out_names:
                out_kwargs[o] = kwargs.pop(o, None)
                # Either all of the out kwargs are set or none of them
                if has_out_kwargs is None:
                    has_out_kwargs = out_kwargs[o] is not None
                else:
                    assert has_out_kwargs == (out_kwargs[o] is not None)
            result = fn(*args, **kwargs)
            assert isinstance(result, tuple)
            if has_out_kwargs:
                final_result = []
                for i, o in enumerate(out_names):
                    out = out_kwargs[o]
                    assert isinstance(out, TensorLike)
                    out = _maybe_resize_out(out, result[i].shape)
                    final_result.append(_safe_copy_out(copy_from=result[i], copy_to=out))  # type: ignore[arg-type]
                return tuple(final_result)
            return result

        sig = inspect.signature(fn)
        out_params = []
        for o in out_names:
            out_params.append(
                inspect.Parameter(
                    o,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=TensorLikeType,
                )
            )
        params = chain(sig.parameters.values(), out_params)
        _fn.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            parameters=params, return_annotation=sig.return_annotation  # type: ignore[arg-type]
        )
        _fn.__annotations__ = fn.__annotations__
        for o in out_names:
            _fn.__annotations__[o] = TensorLikeType
        return _fn

    return go
