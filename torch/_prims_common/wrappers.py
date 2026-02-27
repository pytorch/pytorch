# mypy: allow-untyped-defs
import inspect
import types
import warnings
from collections.abc import Callable, Sequence
from functools import wraps
from types import GenericAlias
from typing import NamedTuple, overload, TypeVar
from typing_extensions import ParamSpec

import torch
import torch._prims_common as utils
from torch._prims_common import (
    CustomOutParamAnnotation,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    Number,
    NumberType,
    ShapeType,
    TensorLike,
    TensorLikeType,
)
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten


_T = TypeVar("_T")
_P = ParamSpec("_P")


@overload
# pyrefly: ignore [bad-return]
def _maybe_convert_to_dtype(a: TensorLikeType, dtype: torch.dtype) -> TensorLikeType:
    pass


@overload
# pyrefly: ignore [bad-return]
def _maybe_convert_to_dtype(a: NumberType, dtype: torch.dtype) -> NumberType:
    pass


@overload
# pyrefly: ignore [bad-return]
def _maybe_convert_to_dtype(a: Sequence, dtype: torch.dtype) -> Sequence:
    pass


@overload
def _maybe_convert_to_dtype(a: None, dtype: torch.dtype) -> None:
    pass


# TODO: implement ref.cast with an option to enforce safe casting
def _maybe_convert_to_dtype(a, dtype):
    if isinstance(a, TensorLike):
        if a.dtype != dtype:
            return a.to(dtype)
        return a
    if isinstance(a, Number):
        return utils.dtype_to_type_ctor(dtype)(a)  # type: ignore[arg-type]
    if isinstance(a, Sequence):
        return tuple(_maybe_convert_to_dtype(x, dtype) for x in a)
    # Passthrough None because some functions wrapped with type promotion
    # wrapper might have optional args
    if a is None:
        return None

    raise ValueError(
        f"Received unsupported type {type(a)}. Expected TensorLike, Number, or Sequence."
    )


def _maybe_convert_to_type(a: NumberType, typ: type) -> NumberType:
    if not isinstance(a, Number):
        msg = f"Found unknown type {type(a)} when trying to convert scalars!"
        raise ValueError(msg)
    if not utils.is_weakly_lesser_type(type(a), typ):
        msg = f"Scalar {a} of type {type(a)} cannot be safely cast to type {typ}!"
        raise ValueError(msg)

    return typ(a)


def _annotation_has_type(*, typ, annotation):
    if hasattr(annotation, "__args__"):
        for a in annotation.__args__:
            if _annotation_has_type(typ=typ, annotation=a):
                return True
        return False

    return typ is annotation


class elementwise_type_promotion_wrapper:
    """
    Adds elementwise type promotion to a Python reference implementation.

    Takes two kwargs, type_promoting_args and type_promotion_kind.

    type_promoting_args must be a string Sequence specifying the argument names of all
    arguments that participate in type promotion (and should be type promoted). If the
    arg specifies a Sequence-type then every element of the Sequence will participate in
    type promotion.

    type_promotion_kind must be one of the kinds specified by ELEMENTWISE_TYPE_PROMOTION_KIND.
    See its documentation for details.

    The return_dtype will be coerced to the wrapped function's dtype arg if it is available and
    not None.

    Other type promotion behavior, like validating the Python type of scalar arguments, must
    be handled separately.
    """

    def __init__(
        self,
        *,
        type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND,
        type_promoting_args: Sequence[str] | None = None,
    ):
        self.type_promoting_arg_names = type_promoting_args
        self.type_promotion_kind = type_promotion_kind

    def __call__(self, fn: Callable) -> Callable:
        sig = inspect.signature(fn)

        # TorchDynamo tracing of inspect causes fake tensor dynamo_wrapped tests to fail
        # PYTORCH_TEST_WITH_DYNAMO=1 python test/test_fake_tensor.py FakeTensorTest.test_basic
        @torch._disable_dynamo
        @wraps(fn)
        def _fn(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            type_promoting_args = tuple(
                bound.arguments[x]
                for x in self.type_promoting_arg_names  # type: ignore[union-attr]
                if x in bound.arguments
            )

            flattened_type_promoting_args = pytree.arg_tree_leaves(*type_promoting_args)
            compute_dtype, result_dtype = utils.elementwise_dtypes(
                *flattened_type_promoting_args,
                type_promotion_kind=self.type_promotion_kind,
            )

            promoted_args = {
                x: _maybe_convert_to_dtype(bound.arguments[x], compute_dtype)
                for x in self.type_promoting_arg_names  # type: ignore[union-attr]
                if x in bound.arguments
            }
            bound.arguments.update(promoted_args)

            result = fn(**bound.arguments)

            # Override the return_dtype if a dtype arg is present and not None
            if "dtype" in bound.arguments:
                maybe_dtype = bound.arguments["dtype"]
                if maybe_dtype:  # dtype cannot be None
                    result_dtype = maybe_dtype

            if isinstance(result, TensorLike):
                return _maybe_convert_to_dtype(result, result_dtype)
            if isinstance(result, Sequence):
                return tuple(_maybe_convert_to_dtype(x, result_dtype) for x in result)
            raise AssertionError(f"Unhandled result type: {type(result)}")

        _fn.__signature__ = sig  # type: ignore[attr-defined]
        return _fn


# Returns True if resize is necessary
def _resize_output_check(out: TensorLikeType, shape: ShapeType):
    # If the shapes are correct there's nothing to do
    if utils.same_shape(out.shape, shape):
        return False
    if out.numel() != 0:
        msg = (
            f"An output with one or more elements was resized since it had shape {str(out.shape)} "
            "which does not match the required output shape {str(shape)}. "
            "This behavior is deprecated, and in a future PyTorch release outputs will not "
            "be resized unless they have zero elements. "
            "You can explicitly reuse an out tensor t by resizing it, inplace, to zero elements with t.resize_(0)."
        )
        warnings.warn(msg, stacklevel=2)
    return True


# TODO: handle tuples of tensors
def _maybe_resize_out(
    out: TensorLikeType,
    shape: ShapeType,
    memory_format: torch.memory_format | None = None,
):
    if _resize_output_check(out, shape):
        return out.resize_(shape, memory_format=memory_format)
    else:
        return out


def is_cpu_scalar(x: TensorLikeType) -> bool:
    return x.dim() == 0 and x.device.type == "cpu"


def check_copy_devices(*, copy_from: TensorLikeType, copy_to: TensorLikeType) -> None:
    if copy_from.device != copy_to.device:
        msg = (
            f"Attempting to copy from device {copy_from.device} "
            f"to device {copy_to.device}, but cross-device copies are not allowed!"
        )
        raise RuntimeError(msg)


def _safe_copy_out(
    *, copy_from: TensorLikeType, copy_to: TensorLikeType, exact_dtype: bool = False
):
    # Checks same device
    if not is_cpu_scalar(copy_from):
        check_copy_devices(copy_from=copy_from, copy_to=copy_to)

    # Checks safe cast
    if exact_dtype:
        torch._check(
            copy_from.dtype == copy_to.dtype,
            lambda: f"Expected out tensor to have dtype {copy_from.dtype} "
            f"but got {copy_to.dtype} instead",
        )
    else:
        torch._check(
            utils.can_safe_cast_to(cast_from=copy_from.dtype, cast_to=copy_to.dtype),
            lambda: f"Attempting to cast from {copy_from.dtype} to out tensor with dtype {copy_to.dtype}, "
            "but this can't be cast because it is not safe!",
        )

    return copy_to.copy_(copy_from)


def out_wrapper(
    *out_names: str,
    exact_dtype: bool = False,
    pass_is_out: bool = False,
    preserve_memory_format: bool = False,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    # The wrapped function needs to convert the output parameters to ensure
    # compatibility between the Python API (which always uses "out" as the
    # parameter name and may be a tuple) and the Aten API (which may have
    # multiple output parameters and use different parameter names such as
    # "grad_input", "indices" or "values".)

    default_out_names = ("out",)
    if len(out_names) == 0:
        # Use default in out name
        out_names = default_out_names

    is_tensor = len(out_names) == 1

    def maybe_compute_memory_format(t):
        return utils.suggest_memory_format(t) if preserve_memory_format else None

    def _out_wrapper(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        """
        Adds the out parameter to a Python reference.
        """
        out_type = (
            TensorLikeType
            if is_tensor
            else GenericAlias(
                tuple, tuple(TensorLikeType for _ in range(len(out_names)))
            )
        )
        # For backward compatibility - should be able to remove once PEP585
        # conversion is complete.
        bc_out_type = (
            TensorLikeType
            if is_tensor
            else types.GenericAlias(
                tuple, tuple(TensorLikeType for _ in range(len(out_names)))
            )
        )
        return_type = (
            TensorLikeType
            if is_tensor
            else NamedTuple(
                f"return_types_{fn.__name__}",
                # pyrefly: ignore [bad-argument-count]
                [(o, TensorLikeType) for o in out_names],
            )
        )

        sig = inspect.signature(fn)
        factory_kwargs = ("device", "dtype")
        is_factory_fn = all(p in sig.parameters for p in factory_kwargs)

        @wraps(fn)
        def _fn(*args: _P.args, **kwargs: _P.kwargs):
            out = kwargs.pop("out", None)
            if is_factory_fn and out is not None:
                for k in factory_kwargs:
                    out_attr = getattr(out, k)
                    if k not in kwargs:
                        kwargs[k] = out_attr

            def maybe_check_copy_devices(out):
                if isinstance(out, TensorLike) and isinstance(args[0], TensorLike):
                    check_copy_devices(copy_from=args[0], copy_to=out)

            if isinstance(out, (tuple, list)):
                for o in out:
                    maybe_check_copy_devices(o)
            else:
                maybe_check_copy_devices(out)

            if pass_is_out:
                result = fn(*args, is_out=(out is not None), **kwargs)  # type: ignore[arg-type]
            else:
                result = fn(*args, **kwargs)
            if result is NotImplemented:
                return NotImplemented
            if not (
                (isinstance(result, TensorLike) and is_tensor)
                or (
                    isinstance(result, tuple)  # type: ignore[arg-type]
                    and len(result) == len(out_names)  # type: ignore[arg-type]
                )
                or (
                    fn.__name__ == "unbind" and isinstance(result, (list, tuple))  # type: ignore[arg-type]
                )
            ):
                raise AssertionError(
                    f"Unexpected result type: {type(result)}, is_tensor={is_tensor}, "
                    f"out_names={out_names}"
                )
            # unbind_copy is a special case: see https://github.com/pytorch/pytorch/issues/130829
            if out is not None:
                # Naively you might expect this assert to be true, but
                # it's not:
                #
                #   assert type(out) is type(result)
                #
                # The reason is that functions under this wrapper can
                # get registered to the Meta dispatch key, and that
                # means they can be executed in a context where tensor
                # subclasses are disabled (with no_dispatch), which is a
                # handy way for an is-a tensor subclass (e.g.,
                # FakeTensor) to have the normal meta backend create a
                # meta tensor, to be wrapped once it gets returned.
                # In this situation, you will get a FakeTensor as
                # the output tensor, but not the result--which will
                # be a normal meta tensor, but this is perfectly
                # harmless.
                if is_tensor and fn.__name__ != "unbind":
                    if not isinstance(out, TensorLike):
                        raise AssertionError(
                            f"out must be TensorLike, got {type(out)}"
                        )  # mypy
                    # These two operations are done in-place
                    _maybe_resize_out(
                        out,
                        result.shape,  # type: ignore[union-attr]
                        maybe_compute_memory_format(result),
                    )
                    _safe_copy_out(
                        copy_from=result,  # type: ignore[arg-type]
                        copy_to=out,
                        exact_dtype=exact_dtype,
                    )
                else:
                    if fn.__name__ != "unbind":
                        if not isinstance(out, tuple):
                            raise AssertionError(f"out must be tuple, got {type(out)}")  # type: ignore[arg-type]  # mypy
                    else:
                        if not isinstance(out, (list, tuple)):
                            raise AssertionError(
                                f"out must be list or tuple, got {type(out)}"
                            )  # type: ignore[arg-type]  # mypy
                    torch._check_type(
                        len(out) == len(result),  # type: ignore[arg-type]
                        lambda: f"expected tuple of {len(result)} elements but got {len(out)}",  # type: ignore[arg-type]
                    )
                    for r, o in zip(result, out):  # type: ignore[arg-type]
                        # These two operations are done in-place
                        _maybe_resize_out(o, r.shape, maybe_compute_memory_format(r))
                        _safe_copy_out(copy_from=r, copy_to=o, exact_dtype=exact_dtype)  # type: ignore[arg-type]
            else:
                out = result
            # mypy does not see through  the definition of out_type given that it's in a different scope
            return out if is_tensor else return_type(*out)  # type: ignore[operator]

        out_param = inspect.Parameter(
            "out",
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=out_type,
        )
        # Mark that the function now returns a tuple
        if not (
            isinstance(sig.return_annotation, (str, TypeVar))
            or sig.return_annotation in (sig.empty, out_type, bc_out_type)
        ):
            raise AssertionError(
                f"Unexpected return annotation: {sig.return_annotation}, "
                f"expected str, TypeVar, empty, {out_type}, or {bc_out_type}"
            )
        params = *sig.parameters.values(), out_param

        # If there's a Parameter.VAR_KEYWORD parameter (like **kwds), it must appear
        # after the out= parameter, which is Parameter.KEYWORD_ONLY. Sorting by
        # Parameter.kind guarantees that all the parameters are in legal order.
        params = sorted(params, key=lambda p: p.kind)

        _fn.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            parameters=params,
            return_annotation=return_type,  # type: ignore[arg-type]
        )

        _fn.__annotations__ = dict(getattr(fn, "__annotations__", {}))
        _fn.__annotations__["out"] = out_type
        _fn.__annotations__["return"] = return_type

        # In the special case of having a single tensor out parameter with a
        # name other than out, add a special annotation to name the parameter
        if is_tensor and out_names != default_out_names:
            _fn.__annotations__[CustomOutParamAnnotation] = out_names[0]

        # Add an indicator attribute that can be used in special cases
        # where having a function wrapped by `out_wrapper` is not desirable e.g.
        # jit
        _fn._torch_decompositions_out_wrapper = (  # type: ignore[attr-defined]
            f"This function is wrapped by {out_wrapper.__module__}.out_wrapper"
        )

        return _fn

    return _out_wrapper


def _maybe_remove_out_wrapper(fn: Callable):
    return inspect.unwrap(
        fn,
        stop=lambda f: not hasattr(f, "_torch_decompositions_out_wrapper"),
    )


def backwards_not_supported(prim):
    def redispatch_prim(args, kwargs):
        with torch._C._AutoDispatchBelowAutograd():
            return prim(*args, **kwargs)

    class BackwardsNotSupported(torch.autograd.Function):
        @staticmethod
        # pyrefly: ignore [bad-override]
        def forward(ctx, args_spec, *flat_args):
            args, kwargs = tree_unflatten(flat_args, args_spec)  # type: ignore[arg-type]
            return redispatch_prim(args, kwargs)

        @staticmethod
        def backward(ctx, *args):
            raise RuntimeError("backwards not supported on prim")

    @wraps(prim)
    def _autograd_impl(*args, **kwargs):
        flat_args, args_spec = tree_flatten((args, kwargs))
        if torch.is_grad_enabled() and any(
            a.requires_grad for a in flat_args if isinstance(a, torch.Tensor)
        ):
            # TODO: There is a subtle bug here: prims like copy_to
            # return their input argument after mutating it; and custom
            # autograd function will incorrectly turn the result into
            # a view which will fail test_python_ref_executor tests.
            # At the moment, we sidestep this by observing that the
            # unit tests don't ever try to run the executor with
            # autograd, so we don't exercise the buggy case, but if
            # you ever want to feed autograd through this, be aware
            # of it!  We need a way of properly implementing autograd
            # for mutating operations in Python to do this.
            return BackwardsNotSupported.apply(args_spec, *flat_args)
        else:
            return redispatch_prim(args, kwargs)

    return _autograd_impl


# TODO: when tracing this will add torch tensors and not TensorMeta objects
# to the trace -- we should fix this by adding a tracing context and NumberMeta classes
# TODO: this wrapper is currently untested
def elementwise_unary_scalar_wrapper(
    fn: Callable[_P, _T],
) -> Callable[_P, _T | NumberType]:
    """
    Allows unary operators that accept tensors to work with Python numbers.
    """
    sig = inspect.signature(fn)

    @wraps(fn)
    def _fn(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], Number):
            dtype = utils.type_to_dtype(type(args[0]))
            args_ = list(args)
            args_[0] = torch.tensor(args[0], dtype=dtype)
            # pyrefly: ignore [invalid-param-spec]
            result = fn(*args_, **kwargs)
            if not isinstance(result, torch.Tensor):
                raise AssertionError(f"Expected torch.Tensor, got {type(result)}")
            return result.item()

        # pyrefly: ignore [invalid-param-spec]
        return fn(*args, **kwargs)

    _fn.__signature__ = sig  # type: ignore[attr-defined]
    # pyrefly: ignore [bad-return]
    return _fn
