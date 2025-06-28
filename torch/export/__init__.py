import builtins
import copy
import dataclasses
import inspect
import os
import sys
import typing
import warnings
import zipfile
from collections.abc import Iterator
from enum import auto, Enum
from typing import Any, Callable, Optional, TYPE_CHECKING, Union

import torch
import torch.utils._pytree as pytree
from torch.fx._compatibility import compatibility
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager
from torch.types import FileLike
from torch.utils._pytree import (
    FlattenFunc,
    FromDumpableContextFn,
    ToDumpableContextFn,
    UnflattenFunc,
)


if TYPE_CHECKING:
    # Import the following modules during type checking to enable code intelligence features,
    # Do not import unconditionally, as they import sympy and importing sympy is very slow
    from torch._ops import OpOverload
    from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint


__all__ = [
    "Constraint",
    "Dim",
    "ExportBackwardSignature",
    "ExportGraphSignature",
    "ExportedProgram",
    "CustomDecompTable",
    "ModuleCallEntry",
    "ModuleCallSignature",
    "default_decompositions",
    "dims",
    "export",
    "export_for_training",
    "load",
    "register_dataclass",
    "save",
    "unflatten",
    "FlatArgsAdapter",
    "UnflattenedModule",
    "AdditionalInputs",
    "draft_export",
]

# To make sure export specific custom ops are loaded
import torch.export.custom_ops

from .decomp_utils import CustomDecompTable
from .dynamic_shapes import AdditionalInputs, Constraint, Dim, dims, ShapesCollection
from .exported_program import (
    default_decompositions,
    ExportedProgram,
    ModuleCallEntry,
    ModuleCallSignature,
)
from .graph_signature import ExportBackwardSignature, ExportGraphSignature
from .unflatten import FlatArgsAdapter, unflatten, UnflattenedModule


PassType = Callable[[torch.fx.GraphModule], Optional[PassResult]]


def export_for_training(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Optional[dict[str, Any]] = None,
    *,
    dynamic_shapes: Optional[Union[dict[str, Any], tuple[Any], list[Any]]] = None,
    strict: bool = False,
    preserve_module_call_signature: tuple[str, ...] = (),
) -> ExportedProgram:
    """
    :func:`export_for_training` takes any nn.Module along with example inputs, and produces a traced graph representing
    only the Tensor computation of the function in an Ahead-of-Time (AOT) fashion,
    which can subsequently be executed with different inputs or serialized. The
    traced graph (1) produces normalized operators in the all ATen operator set
    (as well as any user-specified custom operators), (2) has eliminated all Python control
    flow and data structures (with certain exceptions), and (3) records the set of
    shape constraints needed to show that this normalization and control-flow elimination
    is sound for future inputs. This API is intended for PT2 quantization training use cases
    and will soon be the default IR of torch.export.export in the near future. To read further about
    the motivation behind this change, please refer to
    https://dev-discuss.pytorch.org/t/why-pytorch-does-not-need-a-new-standardized-operator-set/2206
    With this API, and :func:`run_decompositions()`, you should be able to get inference IR with
    your custom decomposition behaviour.

    **Soundness Guarantee**

    See :func:`export()` docstring for more details.

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

        preserve_module_call_signature: A list of submodule paths for which the original
         calling conventions are preserved as metadata. The metadata will be used when calling
         torch.export.unflatten to preserve the original calling conventions of modules.

    Returns:
        An :class:`ExportedProgram` containing the traced callable.

    **Acceptable input/output types**

    Acceptable types of inputs (for ``args`` and ``kwargs``) and outputs include:

    - Primitive types, i.e. ``torch.Tensor``, ``int``, ``float``, ``bool`` and ``str``.
    - Dataclasses, but they must be registered by calling :func:`register_dataclass` first.
    - (Nested) Data structures comprising of ``dict``, ``list``, ``tuple``, ``namedtuple`` and
      ``OrderedDict`` containing all above types.

    """
    from ._trace import _export_for_training

    if not isinstance(mod, torch.nn.Module):
        raise ValueError(
            f"Expected `mod` to be an instance of `torch.nn.Module`, got {type(mod)}."
        )
    if isinstance(mod, torch.jit.ScriptModule):
        raise ValueError(
            "Exporting a ScriptModule is not supported. "
            "Maybe try converting your ScriptModule to an ExportedProgram "
            "using `TS2EPConverter(mod, args, kwargs).convert()` instead."
        )
    return _export_for_training(
        mod,
        args,
        kwargs,
        dynamic_shapes,
        strict=strict,
        preserve_module_call_signature=preserve_module_call_signature,
    )


def export(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Optional[dict[str, Any]] = None,
    *,
    dynamic_shapes: Optional[Union[dict[str, Any], tuple[Any], list[Any]]] = None,
    strict: bool = False,
    preserve_module_call_signature: tuple[str, ...] = (),
) -> ExportedProgram:
    """
    :func:`export` takes any nn.Module along with example inputs, and produces a traced graph representing
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

        strict: When disabled (default), the export function will trace the program through
         Python runtime, which by itself will not validate some of the implicit assumptions
         baked into the graph. It will still validate most critical assumptions like shape
         safety. When enabled (by setting ``strict=True``), the export function will trace
         the program through TorchDynamo which will ensure the soundness of the resulting
         graph. TorchDynamo has limited Python feature coverage, thus you may experience more
         errors. Note that toggling this argument does not affect the resulting IR spec to be
         different and the model will be serialized in the same way regardless of what value
         is passed here.

        preserve_module_call_signature: A list of submodule paths for which the original
         calling conventions are preserved as metadata. The metadata will be used when calling
         torch.export.unflatten to preserve the original calling conventions of modules.

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
    if isinstance(mod, torch.jit.ScriptModule):
        raise ValueError(
            "Exporting a ScriptModule is not supported. "
            "Maybe try converting your ScriptModule to an ExportedProgram "
            "using `TS2EPConverter(mod, args, kwargs).convert()` instead."
        )

    try:
        return _export(
            mod,
            args,
            kwargs,
            dynamic_shapes,
            strict=strict,
            preserve_module_call_signature=preserve_module_call_signature,
            pre_dispatch=True,
        )
    except Exception as e:
        draft_export_msg = (
            "The error above occurred when calling torch.export.export. If you would "
            "like to view some more information about this error, and get a list "
            "of all other errors that may occur in your export call, you can "
            "replace your `export()` call with `draft_export()`."
        )

        # For errors that we know can be caught by draft-export, add the message
        # to ask users to try out draft-export
        if isinstance(
            e,
            (
                torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode,
                torch._subclasses.fake_tensor.UnsupportedOperatorException,
                torch._dynamo.exc.UserError,
                torch.fx.experimental.symbolic_shapes.ConstraintViolationError,
            ),
        ):
            new_msg = str(e) + "\n\n" + draft_export_msg
            e.args = (new_msg,)
        elif isinstance(e, RuntimeError) and "no fake impl registered" in str(e):
            new_msg = str(e) + "\n\n" + draft_export_msg
            e.args = (new_msg,)
        raise e


DEFAULT_PICKLE_PROTOCOL = 2


def save(
    ep: ExportedProgram,
    f: FileLike,
    *,
    extra_files: Optional[dict[str, Any]] = None,
    opset_version: Optional[dict[str, int]] = None,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> None:
    """

    .. warning::
        Under active development, saved files may not be usable in newer versions
        of PyTorch.

    Saves an :class:`ExportedProgram` to a file-like object. It can then be
    loaded using the Python API :func:`torch.export.load <torch.export.load>`.

    Args:
        ep (ExportedProgram): The exported program to save.

        f (str | os.PathLike[str] | IO[bytes]) A file-like object (has to
         implement write and flush) or a string containing a file name.

        extra_files (Optional[Dict[str, Any]]): Map from filename to contents
         which will be stored as part of f.

        opset_version (Optional[Dict[str, int]]): A map of opset names
         to the version of this opset

        pickle_protocol: can be specified to override the default protocol

    Example::

        import torch
        import io


        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + 10


        ep = torch.export.export(MyModule(), (torch.randn(5),))

        # Save to file
        torch.export.save(ep, "exported_program.pt2")

        # Save to io.BytesIO buffer
        buffer = io.BytesIO()
        torch.export.save(ep, buffer)

        # Save with extra files
        extra_files = {"foo.txt": b"bar".decode("utf-8")}
        torch.export.save(ep, "exported_program.pt2", extra_files=extra_files)

    """
    if not isinstance(ep, ExportedProgram):
        raise TypeError(
            f"The 'ep' parameter must be an instance of 'ExportedProgram', got '{type(ep).__name__}' instead."
        )

    from torch.export.pt2_archive._package import package_pt2

    package_pt2(
        f,
        exported_programs={"model": ep},
        extra_files=extra_files,
        pickle_protocol=pickle_protocol,
        opset_version=opset_version,
    )


def load(
    f: FileLike,
    *,
    extra_files: Optional[dict[str, Any]] = None,
    expected_opset_version: Optional[dict[str, int]] = None,
) -> ExportedProgram:
    """

    .. warning::
        Under active development, saved files may not be usable in newer versions
        of PyTorch.

    Loads an :class:`ExportedProgram` previously saved with
    :func:`torch.export.save <torch.export.save>`.

    Args:
        f (str | os.PathLike[str] | IO[bytes]): A file-like object (has to
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
        ep = torch.export.load("exported_program.pt2")

        # Load ExportedProgram from io.BytesIO object
        with open("exported_program.pt2", "rb") as f:
            buffer = io.BytesIO(f.read())
        buffer.seek(0)
        ep = torch.export.load(buffer)

        # Load with extra files.
        extra_files = {"foo.txt": ""}  # values will be replaced with data
        ep = torch.export.load("exported_program.pt2", extra_files=extra_files)
        print(extra_files["foo.txt"])
        print(ep(torch.randn(5)))
    """
    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)

    extra_files = extra_files or {}

    from torch.export.pt2_archive._package import load_pt2, PT2ArchiveContents

    try:
        pt2_contents = load_pt2(
            f,
            expected_opset_version=expected_opset_version,
        )
    except RuntimeError:
        pt2_contents = PT2ArchiveContents({}, {}, {})

    if len(pt2_contents.exported_programs) > 0 or len(pt2_contents.extra_files) > 0:
        for k, v in pt2_contents.extra_files.items():
            extra_files[k] = v

        return pt2_contents.exported_programs["model"]

    # TODO: For backward compatibility, we support loading a zip file from 2.7. Delete this path in 2.9(?)
    warnings.warn(
        "This version of file is deprecated. Please generate a new pt2 saved file."
    )
    with zipfile.ZipFile(f, "r") as zipf:
        # Check the version
        version = zipf.read("version").decode().split(".")
        from torch._export.serde.schema import (
            SCHEMA_VERSION,  # todo change archive version to schema version
        )

        assert len(version) == len(SCHEMA_VERSION), (
            "Version in the saved file has incorrect length, double check if the file is generated by torch.export.save()"
        )
        if version[0] != str(SCHEMA_VERSION[0]):
            raise RuntimeError(
                f"Serialized version {version} does not match our current "
                f"schema version {SCHEMA_VERSION}."
            )

        from torch._export.serde.serialize import deserialize, SerializedArtifact

        # Load serialized_ep and serialized_state_dict from the zip file

        serialized_exported_program: Optional[bytes] = None
        serialized_state_dict: Optional[bytes] = None
        serialized_constants: Optional[bytes] = None
        serialized_example_inputs: Optional[bytes] = None

        for file_info in zipf.infolist():
            file_content = zipf.read(file_info.filename)

            if file_info.filename == "serialized_exported_program.json":
                serialized_exported_program = file_content
            elif file_info.filename == "serialized_state_dict.json":
                warnings.warn("This version of file is deprecated")
                serialized_state_dict = file_content
            elif file_info.filename == "serialized_constants.json":
                warnings.warn("This version of file is deprecated")
                serialized_constants = file_content
            elif file_info.filename == "serialized_state_dict.pt":
                serialized_state_dict = file_content
            elif file_info.filename == "serialized_constants.pt":
                serialized_constants = file_content
            elif file_info.filename == "serialized_example_inputs.pt":
                serialized_example_inputs = file_content
            elif file_info.filename.startswith("extra_files"):
                filename = file_info.filename.split("/", 1)[1]
                extra_files[filename] = file_content.decode("utf-8")

        assert serialized_exported_program is not None
        assert serialized_state_dict is not None
        assert serialized_constants is not None
        assert serialized_example_inputs is not None
        artifact: SerializedArtifact = SerializedArtifact(
            serialized_exported_program,
            serialized_state_dict,
            serialized_constants,
            serialized_example_inputs,
        )

        # Deserialize ExportedProgram
        ep = deserialize(artifact, expected_opset_version)

        return ep


def draft_export(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Optional[dict[str, Any]] = None,
    *,
    dynamic_shapes: Optional[Union[dict[str, Any], tuple[Any], list[Any]]] = None,
    preserve_module_call_signature: tuple[str, ...] = (),
    strict: bool = False,
) -> ExportedProgram:
    """
    A version of torch.export.export which is designed to consistently produce
    an ExportedProgram, even if there are potential soundness issues, and to
    generate a report listing the issues found.
    """
    from ._draft_export import draft_export

    return draft_export(
        mod=mod,
        args=args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
        preserve_module_call_signature=preserve_module_call_signature,
        strict=strict,
    )


def register_dataclass(
    cls: type[Any],
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

        import torch
        from dataclasses import dataclass


        @dataclass
        class InputDataClass:
            feature: torch.Tensor
            bias: int


        @dataclass
        class OutputDataClass:
            res: torch.Tensor


        torch.export.register_dataclass(InputDataClass)
        torch.export.register_dataclass(OutputDataClass)


        class Mod(torch.nn.Module):
            def forward(self, x: InputDataClass) -> OutputDataClass:
                res = x.feature + x.bias
                return OutputDataClass(res=res)


        ep = torch.export.export(Mod(), (InputDataClass(torch.ones(2, 2), 1),))
        print(ep)

    """
    pytree.register_dataclass(cls, serialized_type_name=serialized_type_name)
