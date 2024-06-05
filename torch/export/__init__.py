import builtins
import copy
import dataclasses
import inspect
import io
import os
import sys
import typing
import warnings
from enum import auto, Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

import torch
import torch.utils._pytree as pytree
from torch.fx._compatibility import compatibility

from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager

from torch.utils._pytree import (
    FlattenFunc,
    FromDumpableContextFn,
    ToDumpableContextFn,
    UnflattenFunc,
)

if TYPE_CHECKING:
    # Import the following modules during type checking to enable code intelligence features,
    # Do not import unconditionally, as they import sympy and importing sympy is very slow
    from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint


__all__ = [
    "Constraint",
    "Dim",
    "ExportBackwardSignature",
    "ExportGraphSignature",
    "ExportedProgram",
    "ModuleCallEntry",
    "ModuleCallSignature",
    "dims",
    "dynamic_dim",
    "export",
    "load",
    "register_dataclass",
    "save",
    "unflatten",
    "FlatArgsAdapter",
    "UnflattenedModule",
]


from .dynamic_shapes import Constraint, Dim, dims, dynamic_dim, ShapesCollection
from .exported_program import ExportedProgram, ModuleCallEntry, ModuleCallSignature
from .graph_signature import ExportBackwardSignature, ExportGraphSignature
from .unflatten import FlatArgsAdapter, unflatten, UnflattenedModule


PassType = Callable[[torch.fx.GraphModule], Optional[PassResult]]


def export(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]] = None,
    strict: bool = True,
    preserve_module_call_signature: Tuple[str, ...] = (),
) -> ExportedProgram:
    """
    :func:`export` takes an arbitrary Python callable (an nn.Module, a function or
    a method) along with example inputs, and produces a traced graph representing
    only the Tensor computation of the function in an Ahead-of-Time (AOT) fashion,
    which can subsequently be executed with different inputs or serialized.  The
    traced graph (1) produces normalized operators in the functional ATen operator set
    (as well as any user-specified custom operators), (2) has eliminated all Python control
    flow and data structures (with certain exceptions), and (3) records the set of
    shape constraints needed to show that this normalization and control-flow elimination
    is sound for future inputs.

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
        mod: We will trace the forward method of this module.

        args: Example positional inputs.

        kwargs: Optional example keyword inputs.

        dynamic_shapes:
         An optional argument where the type should either be:
         1) a dict from argument names of ``f`` to their dynamic shape specifications,
         2) a tuple that specifies dynamic shape specifications for each input in original order.
         If you are specifying dynamism on keyword args, you will need to pass them in the order that
         is defined in the original function signature.

         The dynamic shape of a tensor argument can be specified as either
         (1) a dict from dynamic dimension indices to :func:`Dim` types, where it is
         not required to include static dimension indices in this dict, but when they are,
         they should be mapped to None; or (2) a tuple / list of :func:`Dim` types or None,
         where the :func:`Dim` types correspond to dynamic dimensions, and static dimensions
         are denoted by None. Arguments that are dicts or tuples / lists of tensors are
         recursively specified by using mappings or sequences of contained specifications.

        strict: When enabled (default), the export function will trace the program through
         TorchDynamo which will ensure the soundness of the resulting graph. Otherwise, the
         exported program will not validate the implicit assumptions baked into the graph and
         may cause behavior divergence between the original model and the exported one. This is
         useful when users need to workaround bugs in the tracer, or simply want incrementally
         enable safety in their models. Note that this does not affect the resulting IR spec
         to be different and the model will be serialized in the same way regardless of what value
         is passed here.
         WARNING: This option is experimental and use this at your own risk.

    Returns:
        An :class:`ExportedProgram` containing the traced callable.

    **Acceptable input/output types**

    Acceptable types of inputs (for ``args`` and ``kwargs``) and outputs include:

    - Primitive types, i.e. ``torch.Tensor``, ``int``, ``float``, ``bool`` and ``str``.
    - Dataclasses, but they must be registered by calling :func:`register_dataclass` first.
    - (Nested) Data structures comprising of ``dict``, ``list``, ``tuple``, ``namedtuple`` and
      ``OrderedDict`` containing all above types.

    """
    from ._trace import _export

    if not isinstance(mod, torch.nn.Module):
        raise ValueError(
            f"Expected `mod` to be an instance of `torch.nn.Module`, got {type(mod)}."
        )

    return _export(
        mod,
        args,
        kwargs,
        dynamic_shapes,
        strict=strict,
        preserve_module_call_signature=preserve_module_call_signature,
        pre_dispatch=True,
    )


def save(
    ep: ExportedProgram,
    f: Union[str, os.PathLike, io.BytesIO],
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

        f (Union[str, os.PathLike, io.BytesIO): A file-like object (has to
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

    if not isinstance(ep, ExportedProgram):
        raise TypeError(
            f"The 'ep' parameter must be an instance of 'ExportedProgram', got '{type(ep).__name__}' instead."
        )

    save(ep, f, extra_files=extra_files, opset_version=opset_version)


def load(
    f: Union[str, os.PathLike, io.BytesIO],
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

        f (Union[str, os.PathLike, io.BytesIO): A file-like object (has to
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


def register_dataclass(
    cls: Type[Any],
    *,
    serialized_type_name: Optional[str] = None,
) -> None:
    """
    Registers a dataclass as a valid input/output type for :func:`torch.export.export`.

    Args:
        cls: the dataclass type to register
        serialized_type_name: The serialized name for the dataclass. This is
        required if you want to serialize the pytree TreeSpec containing this
        dataclass.

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

    return register_dataclass_as_pytree_node(
        cls, serialized_type_name=serialized_type_name
    )
