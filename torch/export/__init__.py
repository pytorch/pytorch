import builtins
import copy
import dataclasses
import inspect
import io
import pathlib
import sys
import typing
from enum import auto, Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import sympy

import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.fx._compatibility import compatibility

from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint

from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager

from torch.utils._pytree import (
    FlattenFunc,
    FromDumpableContextFn,
    ToDumpableContextFn,
    UnflattenFunc,
)


__all__ = [
    "Constraint",
    "Dim",
    "ExportBackwardSignature",
    "ExportGraphSignature",
    "ExportedProgram",
    "ModuleCallEntry",
    "ModuleCallSignature",
    "constrain_as_size",
    "constrain_as_value",
    "dims",
    "dynamic_dim",
    "export",
    "load",
    "register_dataclass",
    "save",
]


from .exported_program import (
    ExportBackwardSignature,
    ExportedProgram,
    ExportGraphSignature,
    ModuleCallEntry,
    ModuleCallSignature,
)


PassType = Callable[[torch.fx.GraphModule], Optional[PassResult]]


@dataclasses.dataclass
class _ConstraintTarget:
    """
    This represents input tensor dimensions.  Don't create this
    class directly; instead, use :func:`dynamic_dim`.
    """

    w_tensor: Any  # weakref to torch.Tensor
    # TODO: We don't need t_id; we can get it off of w_tensor
    t_id: int
    dim: int


class _ConstraintFactory(type):
    """
    Metaclass that ensures a private constructor for :class:`Constraint`
    """

    def __call__(cls, *args, **kwargs):
        raise TypeError(
            f"{cls.__module__}.{cls.__qualname__} has no public constructor. "
            f"Please use torch.export.dynamic_dim() to create one"
        )

    def _create(
        cls, w_tensor, t_id, dim, constraint_range, shared=None, debug_name=None
    ):
        return super().__call__(
            w_tensor, t_id, dim, constraint_range, shared, debug_name
        )


def _create_constraint(
    w_tensor, t_id, dim, constraint_range, shared=None, debug_name=None
):
    return Constraint._create(w_tensor, t_id, dim, constraint_range, shared, debug_name)


@dataclasses.dataclass
class Constraint(_ConstraintTarget, metaclass=_ConstraintFactory):
    """

    .. warning::
        Do not construct :class:`Constraint` directly, use :func:`dynamic_dim` instead.

    This represents constraints on input tensor dimensions, e.g., requiring
    them to be fully polymorphic or within some range.

    """

    # NOTE(avik): In the future, this could be Union[StrictMinMaxConstraint, <other kinds>]
    constraint_range: StrictMinMaxConstraint
    # Represent that `constraint_range` is shared with another _ConstraintTarget, which
    # typically arises because of a specified equality with another dynamic dimension.
    shared: Optional[_ConstraintTarget] = None
    debug_name: Optional[str] = None

    def _clone_with_range(self, lower=2, upper=sympy.oo):
        from torch.utils._sympy.value_ranges import ValueRanges

        constraint_range = StrictMinMaxConstraint(
            vr=self.constraint_range.vr & ValueRanges(lower=lower, upper=upper),
            warn_only=False,
        )
        return _create_constraint(
            self.w_tensor,
            self.t_id,
            self.dim,
            constraint_range,
            self.shared,
            self.debug_name,
        )

    def __ge__(self, lower):
        return self._clone_with_range(lower=lower)

    def __gt__(self, lower):
        return self._clone_with_range(lower=lower + 1)

    def __le__(self, upper):
        return self._clone_with_range(upper=upper)

    def __lt__(self, upper):
        return self._clone_with_range(upper=upper - 1)

    def __bool__(self):
        # NOTE(avik): We do not support compound expressions like a <= x <= b.
        # This is because Python implicitly desugars them into bool(a <= x) and bool(x <= b),
        # and moreover, enforces that any overload of __bool__ must return True or False.
        # FWIW, sympy also raises TypeError in this case.
        raise TypeError(
            "Cannot determine truth value of Constraint. "
            "If you are trying to combine Constraint's with logical connectives, "
            "you can specify them separately instead."
        )

    @property
    def serializable_spec(self):
        # We need a serialization compatible format of the constraint so that it
        # can be savedin the graph module w/o breaking the module serialization.
        # The saved constraints will be used directly for the post-exporting pass
        # that converts constraints to runtime assertion. The saved constraints
        # will not be saved in the serialized module.
        # TODO: A better way is needed. Currently we use 't_id' to map the constraint,
        # which is not reliable
        return {
            "t_id": self.t_id,
            "dim": self.dim,
            "min": self.constraint_range.vr.lower,
            "max": self.constraint_range.vr.upper,
            "shared": (
                None
                if self.shared is None
                else {
                    "t_id": self.shared.t_id,
                    "dim": self.shared.dim,
                }
            ),
        }

    def __eq__(self, other):
        if not isinstance(other, Constraint):
            raise TypeError(
                "A dynamic dim can be specified equal only to another dynamic dim. "
                f"Equality with {type(other)} is not supported."
            )
        constraint_range = StrictMinMaxConstraint(
            vr=self.constraint_range.vr & other.constraint_range.vr,
            warn_only=False,
        )
        if self.debug_name is None:
            debug_name = other.debug_name
        else:
            assert other.debug_name is None or self.debug_name == other.debug_name
            debug_name = self.debug_name
        return _create_constraint(
            self.w_tensor,
            self.t_id,
            self.dim,
            constraint_range,
            shared=_ConstraintTarget(other.w_tensor, other.t_id, other.dim),
            debug_name=debug_name,
        )


def constrain_as_value(symbol, min: Optional[int] = None, max: Optional[int] = None):
    """
    Hint :func:`export` about the constraint of an intermediate scalar value so that subsequent
    branching behaviors that check on the range of aforementioned scalar value can be
    soundly traced.

    .. warning::
        (Note that if the intermediate scalar value will be used like a size, including
        being passed as size arg to a tensor factory or view, call :func:`constrain_as_size`
        instead.)

    Args:
        symbol: Intermediate scalar value (int-only now) to apply range constraint on.
        min (Optional[int]): Minimum possible value of given symbol (inclusive)
        max (Optional[int]): Maximum possible value of given symbol (inclusive)

    Returns:
        None

    For example, following program can not be traced soundly::

        def fn(x):
            v = x.max().item()
            if v > 1024:
                return x
            else:
                return x * 2

    ``v`` is a data-dependent value, which is assumed to have a range of (-inf, inf).
    :func:`export()` a hint about which branch to take would not be able to determine
    if the traced branching decision is correct or not. Thus :func:`export()`
    would give following error::

        torch._dynamo.exc.UserError: Consider annotating your code using
        torch.export.constrain_as_size() or torch.export().constrain_as_value() APIs.
        It appears that you're trying to get a value out of symbolic int/float whose value
        is data-dependent (and thus we do not know the true value.)  The expression we were
        trying to evaluate is f0 > 1024 (unhinted: f0 > 1024).

    Assuming the actual range of ``v`` can be between [10, 200], you can add a call to
    :func:`constrain_as_value` in the source code like this::

        def fn(x):
            v = x.max().item()

            # Give export() a hint
            torch.export.constrain_as_value(v, min=10, max=200)

            if v > 1024:
                return x
            else:
                return x * 2

    With the additional hint, :func:`export` would be able to trace the program correctly by taking
    the ``else`` branch, resulting in following graph::

        graph():
            %arg0_1 := placeholder[target=arg0_1]

            # v = x.max().item()
            %max_1 := call_function[target=torch.ops.aten.max.default](args = (%arg0_1,))
            %_local_scalar_dense := call_function[target=torch.ops.aten._local_scalar_dense.default](args = (%max_1,))

            # Asserting 10 <= v <= 200
            %ge := call_function[target=operator.ge](args = (%_local_scalar_dense, 10))
            %scalar_tensor := call_function[target=torch.ops.aten.scalar_tensor.default](args = (%ge,))
            %_assert_async := call_function[target=torch.ops.aten._assert_async.msg](
                args = (%scalar_tensor, _local_scalar_dense is outside of inline constraint [10, 200].))
            %le := call_function[target=operator.le](args = (%_local_scalar_dense, 200))
            %scalar_tensor_1 := call_function[target=torch.ops.aten.scalar_tensor.default](args = (%le,))
            %_assert_async_1 := call_function[target=torch.ops.aten._assert_async.msg](
                args = (%scalar_tensor_1, _local_scalar_dense is outside of inline constraint [10, 200].))
            %sym_constrain_range := call_function[target=torch.ops.aten.sym_constrain_range.default](
                args = (%_local_scalar_dense,), kwargs = {min: 10, max: 200})

            # Always taking `else` branch to multiply elements `x` by 2 due to hints above
            %mul := call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 2), kwargs = {})
            return (mul,)

    """
    from torch._export.constraints import constrain_as_value

    return constrain_as_value(symbol, min, max)


def constrain_as_size(symbol, min: Optional[int] = None, max: Optional[int] = None):
    """
    Hint :func:`export` about the constraint of an intermediate scalar value that
    represents shape of a tensor so that subsequent tensor constructors can be
    traced correctly because many operators need to make assumption about range
    of sizes.

    Args:
        symbol: Intermediate scalar value (int-only now) to apply range constraint on.
        min (Optional[int]): Minimum possible value of given symbol (inclusive)
        max (Optional[int]): Maximum possible value of given symbol (inclusive)

    Returns:
        None

    For example, following program can not be traced soundly wihout using
    :func:`constrain_as_size` to give :func:`export` a hint about shape ranges::

        def fn(x):
            d = x.max().item()
            return torch.ones(v)

    :func:`export` would give following error::

        torch._dynamo.exc.Unsupported: guard on data-dependent symbolic int/float

    Assuming the actual range of ``d`` can be between [3, 10], you can add a call to
    :func:`constrain_as_size` in the source code like this::

        def fn(x):
            d = x.max().item()
            torch.export.constrain_as_size(d, min=3, max=10)
            return torch.ones(d)

    With the additional hint, :func:`export` would be able to trace the program correctly by taking
    the ``else`` branch, resulting in following graph::

        graph():
            %arg0_1 := placeholder[target=arg0_1]

            # d = x.max().item()
            %max_1 := call_function[target=torch.ops.aten.max.default](args = (%arg0_1,))
            %_local_scalar_dense := call_function[target=torch.ops.aten._local_scalar_dense.default](args = (%max_1,))

            # Asserting 3 <= d <= 10
            %ge := call_function[target=operator.ge](args = (%_local_scalar_dense, 3))
            %scalar_tensor := call_function[target=torch.ops.aten.scalar_tensor.default](args = (%ge,))
            %_assert_async := call_function[target=torch.ops.aten._assert_async.msg](
                args = (%scalar_tensor, _local_scalar_dense is outside of inline constraint [3, 10].))
            %le := call_function[target=operator.le](args = (%_local_scalar_dense, 10))
            %scalar_tensor_1 := call_function[target=torch.ops.aten.scalar_tensor.default](args = (%le,))
            %_assert_async_1 := call_function[target=torch.ops.aten._assert_async.msg](
                args = (%scalar_tensor_1, _local_scalar_dense is outside of inline constraint [3, 10].))
            %sym_constrain_range_for_size := call_function[target=torch.ops.aten.sym_constrain_range_for_size.default](
                args = (%_local_scalar_dense,), kwargs = {min: 3, max: 10})

            # Constructing new tensor with d
            %full := call_function[target=torch.ops.aten.full.default](
                args = ([%_local_scalar_dense], 1),
                kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})

            ......


    .. warning::
        if your size is intended to be dynamic, do NOT test if sizes are equal to 0 or 1,
        these will SILENTLY report false and be bypassed

    """

    from torch._export.constraints import constrain_as_size

    return constrain_as_size(symbol, min, max)


def dynamic_dim(t: torch.Tensor, index: int):
    """
    .. warning::
        (This feature is DEPRECATED. See :func:`Dim` instead.)

    :func:`dynamic_dim` constructs a :class:`Constraint` object that describes the dynamism of
    a dimension ``index`` of tensor ``t``. :class:`Constraint` objects should be passed to
    ``constraints`` argument of :func:`export`.

    Args:
        t (torch.Tensor): Example input tensor that have dynamic dimension size(s)
        index (int): Index of dynamic dimension

    Returns:
        A :class:`Constraint` object that describes shape dynamism. It can be passed to :func:`export` so
        that :func:`export` does not assume static size of specified tensor, i.e. keeping it dynamic
        as a symbolic size rather than specializing according to size of example tracing input.

    Specifically :func:`dynamic_dim` can be used to express following types of dynamism.

    - Size of a dimension is dynamic and unbounded::

        t0 = torch.rand(2, 3)
        t1 = torch.rand(3, 4)

        # First dimension of t0 can be dynamic size rather than always being static size 2
        constraints = [dynamic_dim(t0, 0)]
        ep = export(fn, (t0, t1), constraints=constraints)

    - Size of a dimension is dynamic with a lower bound::

        t0 = torch.rand(10, 3)
        t1 = torch.rand(3, 4)

        # First dimension of t0 can be dynamic size with a lower bound of 5 (inclusive)
        # Second dimension of t1 can be dynamic size with a lower bound of 2 (exclusive)
        constraints = [
            dynamic_dim(t0, 0) >= 5,
            dynamic_dim(t1, 1) > 2,
        ]
        ep = export(fn, (t0, t1), constraints=constraints)

    - Size of a dimension is dynamic with an upper bound::

        t0 = torch.rand(10, 3)
        t1 = torch.rand(3, 4)

        # First dimension of t0 can be dynamic size with a upper bound of 16 (inclusive)
        # Second dimension of t1 can be dynamic size with a upper bound of 8 (exclusive)
        constraints = [
            dynamic_dim(t0, 0) <= 16,
            dynamic_dim(t1, 1) < 8,
        ]
        ep = export(fn, (t0, t1), constraints=constraints)

    - Size of a dimension is dynamic and it is always equal to size of another dynamic dimension::

        t0 = torch.rand(10, 3)
        t1 = torch.rand(3, 4)

        # Sizes of second dimension of t0 and first dimension are always equal
        constraints = [
            dynamic_dim(t0, 1) == dynamic_dim(t1, 0),
        ]
        ep = export(fn, (t0, t1), constraints=constraints)

    - Mix and match all types above as long as they do not express conflicting requirements

    """
    from torch._export import dynamic_dim

    return dynamic_dim(t, index)


class _Dim(type):
    """
    Metaclass for :func:`Dim` types.
    """

    @staticmethod
    def readable(name, min_, max_):
        if min_ == 2:
            min_ = None
        if max_ == sys.maxsize - 1:
            max_ = None
        if min_ is None and max_ is None:
            return f"Dim('{name}')"
        if min_ is None:
            return f"Dim('{name}', max={max_})"
        if max_ is None:
            return f"Dim('{name}', min={min_})"
        return f"Dim('{name}', min={min_}, max={max_})"


def Dim(name: str, *, min: Optional[int] = None, max: Optional[int] = None):
    """
    :func:`Dim` constructs a type analogous to a named symbolic integer with a range.
    It can be used to describe multiple possible values of a dynamic tensor dimension.
    Note that different dynamic dimensions of the same tensor, or of different tensors,
    can be described by the same type.

    Args:
        name (str): Human-readable name for debugging.
        min (Optional[int]): Minimum possible value of given symbol (inclusive)
        max (Optional[int]): Maximum possible value of given symbol (inclusive)

    Returns:
        A type that can be used in dynamic shape specifications for tensors.
    """
    _min = 2 if min is None else builtins.max(min, 2)
    _max = sys.maxsize - 1 if max is None else builtins.min(max, sys.maxsize - 1)
    assert _max > _min, f"Cannot create Dim with inconsistent min={min}, max={max}"
    dim = _Dim(name, (int,), {"min": _min, "max": _max})
    dim.__module__ = inspect.getmodule(inspect.stack()[1][0]).__name__  # type: ignore[union-attr]
    return dim


def dims(*names: str, min: Optional[int] = None, max: Optional[int] = None):
    """
    Util to create multiple :func:`Dim` types.
    """
    return tuple(Dim(name, min=min, max=max) for name in names)


def export(
    f: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    constraints: Optional[List[Constraint]] = None,
    dynamic_shapes: Optional[Dict[str, Any]] = None,
) -> ExportedProgram:
    """
    :func:`export` takes an arbitrary Python callable (an nn.Module, a function or
    a method) and produces a traced graph representing only the Tensor
    computation of the function in an Ahead-of-Time (AOT) fashion, which can
    subsequently be executed with different outputs or serialized.  The traced
    graph (1) produces a normalized operator set consisting only of functional
    `Core ATen Operator Set <https://pytorch.org/docs/stable/ir.html>`_
    and user specified custom operators, (2) has eliminated all Python control
    flow and data structures (except for certain
    conditions), and (3) has the set of shape constraints needed to show that
    this normalization and control flow elimination is sound for a future
    input.

    **Soundness Guarantee**

    While tracing, :func:`export()` takes note of shape-related assumptions
    made by the user program and the underlying PyTorch operator kernels.
    The output :class:`ExportedProgram` is considered valid only when these
    assumptions hold true.

    Tracing makes assumptions on the shapes (not values) of input tensors.
    Such assumptions must be validated at graph capture time for :func:`export`
    to succeed. Specifically:

    - Assumptions on static shapes of input tensors are automatically validated without additional effort.
    - Assumptions on dynamic shape of input tensors require explicit specification
      by using the :func:`Dim` API to construct dynamic dimensions and by associating
      them with example inputs through the ``dynamic_shapes`` argument.

    If any assumption can not be validated, a fatal error will be raised. When that happens,
    the error message will include suggested fixes to the specification that are needed
    to validate the assumptions. For example :func:`export` might suggest the
    following fix to the definition of a dynamic dimension ``dim0_x``, say appearing in the
    shape associated with input ``x``, that was previously defined as ``Dim("dim0_x")``::

        dim = Dim("dim0_x", max=5)

    This example means the generated code requires dimension 0 of input ``x`` to be less
    than or equal to 5 to be valid. You can inspect the suggested fixes to dynamic dimension
    definitions and then copy them verbatim into your code without needing to change the
    ``dynamic_shapes`` argument to your :func:`export` call.

    Args:
        f: The callable to trace.

        args: Example positional inputs.

        kwargs: Optional example keyword inputs.

        constraints: [DEPRECATED: use ``dynamic_shapes`` instead, see below]
         An optional list of constraints on the dynamic arguments
         that specify their possible range of shapes. By default, shapes of
         input torch.Tensors are assumed to be static. If an input torch.Tensor
         is expected to have dynamic shapes, please use :func:`dynamic_dim`
         to define :class:`Constraint` objects that specify the dynamics and the possible
         range of shapes. See :func:`dynamic_dim` docstring for examples on
         how to use it.

        dynamic_shapes: Should be a dict from argument names of ``f`` to their dynamic shape specifications,
         as follows. The dynamic shape of a tensor argument can be specified as either
         (1) a dict from dynamic dimension indices to :func:`Dim` types, where it is
         not required to include static dimension indices in this dict, but when they are,
         they should be mapped to None; or (2) a tuple / list of :func:`Dim` types or None,
         where the :func:`Dim` types correspond to dynamic dimensions, and static dimensions
         are denoted by None. Arguments that are dicts or tuples / lists of tensors are
         recursively specified by using mappings or sequences of contained specifications.

    Returns:
        An :class:`ExportedProgram` containing the traced callable.

    **Acceptable input/output types**

    Acceptable types of inputs (for ``args`` and ``kwargs``) and outputs include:

    - Primitive types, i.e. ``torch.Tensor``, ``int``, ``float``, ``bool`` and ``str``.
    - Dataclasses, but they must be registered by calling :func:`register_dataclass` first.
    - (Nested) Data structures comprising of ``dict``, ``list``, ``tuple``, ``namedtuple`` and
      ``OrderedDict`` containing all above types.

    """

    from torch._export import export, export__RC__

    if constraints is not None:
        return export(f, args, kwargs, constraints)
    else:
        return export__RC__(f, args, kwargs, dynamic_shapes=dynamic_shapes)


def save(
    ep: ExportedProgram,
    f: Union[str, pathlib.Path, io.BytesIO],
    *,
    extra_files: Optional[Dict[str, Any]] = None,
    opset_version: Optional[Dict[str, int]] = None,
) -> None:
    """

    .. warning::
        Under active development, saved files may not be usable in newer versions
        of PyTorch.

    Saves an :class:`ExportedProgram` to a file-like object. It can then be
    loaded using the Python API :func:`torch.export.load <torch.export.load>`.

    Args:
        ep (ExportedProgram): The exported program to save.

        f (Union[str, pathlib.Path, io.BytesIO): A file-like object (has to
         implement write and flush) or a string containing a file name.

        extra_files (Optional[Dict[str, Any]]): Map from filename to contents
         which will be stored as part of f.

        opset_version (Optional[Dict[str, int]]): A map of opset names
         to the version of this opset


    Example::

        import torch
        import io

        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + 10

        ep = torch.export.export(MyModule(), (torch.randn(5),))

        # Save to file
        torch.export.save(ep, 'exported_program.pt2')

        # Save to io.BytesIO buffer
        buffer = io.BytesIO()
        torch.export.save(ep, buffer)

        # Save with extra files
        extra_files = {'foo.txt': b'bar'.decode('utf-8')}
        torch.export.save(ep, 'exported_program.pt2', extra_files=extra_files)

    """
    from torch._export import save

    save(ep, f, extra_files=extra_files, opset_version=opset_version)


def load(
    f: Union[str, pathlib.Path, io.BytesIO],
    *,
    extra_files: Optional[Dict[str, Any]] = None,
    expected_opset_version: Optional[Dict[str, int]] = None,
) -> ExportedProgram:
    """

    .. warning::
        Under active development, saved files may not be usable in newer versions
        of PyTorch.

    Loads an :class:`ExportedProgram` previously saved with
    :func:`torch.export.save <torch.export.save>`.

    Args:
        ep (ExportedProgram): The exported program to save.

        f (Union[str, pathlib.Path, io.BytesIO): A file-like object (has to
         implement write and flush) or a string containing a file name.

        extra_files (Optional[Dict[str, Any]]): The extra filenames given in
         this map would be loaded and their content would be stored in the
         provided map.

        expected_opset_version (Optional[Dict[str, int]]): A map of opset names
         to expected opset versions

    Returns:
        An :class:`ExportedProgram` object

    Example::

        import torch
        import io

        # Load ExportedProgram from file
        ep = torch.export.load('exported_program.pt2')

        # Load ExportedProgram from io.BytesIO object
        with open('exported_program.pt2', 'rb') as f:
            buffer = io.BytesIO(f.read())
        buffer.seek(0)
        ep = torch.export.load(buffer)

        # Load with extra files.
        extra_files = {'foo.txt': ''}  # values will be replaced with data
        ep = torch.export.load('exported_program.pt2', extra_files=extra_files)
        print(extra_files['foo.txt'])
        print(ep(torch.randn(5)))
    """
    from torch._export import load

    return load(
        f, extra_files=extra_files, expected_opset_version=expected_opset_version
    )


def register_dataclass(typ: Any) -> None:
    """
    Registers a dataclass as a valid input/output type for :func:`torch.export.export`.

    Args:
        typ: the dataclass type to register

    Example::

        @dataclass
        class InputDataClass:
            feature: torch.Tensor
            bias: int

        class OutputDataClass:
            res: torch.Tensor

        torch.export.register_dataclass(InputDataClass)
        torch.export.register_dataclass(OutputDataClass)

        def fn(o: InputDataClass) -> torch.Tensor:
            res = res=o.feature + o.bias
            return OutputDataClass(res=res)

        ep = torch.export.export(fn, (InputDataClass(torch.ones(2, 2), 1), ))
        print(ep)

    """

    from torch._export.utils import register_dataclass_as_pytree_node

    return register_dataclass_as_pytree_node(typ)
