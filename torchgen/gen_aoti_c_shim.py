from __future__ import annotations

import difflib
import os
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchgen.aoti.fallback_ops import aten_shimified_ops, inductor_fallback_ops
from torchgen.api.types import DispatcherSignature
from torchgen.api.types.signatures import CppSignature, CppSignatureGroup
from torchgen.context import method_with_native_function
from torchgen.model import (
    Argument,
    BackendIndex,
    BaseTy,
    BaseType,
    DispatchKey,
    FunctionSchema,
    is_cuda_dispatch_key,
    ListType,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
    OptionalType,
    Type,
)
from torchgen.utils import FileManager, mapMaybe


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Optional


base_type_to_c_type = {
    BaseTy.Tensor: "AtenTensorHandle",
    BaseTy.bool: "int32_t",  # Use int to pass bool
    BaseTy.int: "int64_t",
    BaseTy.SymInt: "int64_t",  # Inductor-generated code won't see a SymInt
    BaseTy.Scalar: "double",  # Use double to pass both integer and floating point
    BaseTy.float: "double",  # TODO: how about other floating point types?
    BaseTy.str: "const char*",
    BaseTy.DeviceIndex: "int32_t",
    BaseTy.Layout: "int32_t",  # Represent enum as int
    BaseTy.MemoryFormat: "int32_t",  # Represent enum as int
    BaseTy.ScalarType: "int32_t",  # Represent enum as int
    BaseTy.Generator: "AtenGeneratorHandle",
}

base_type_to_aten_type = {
    BaseTy.Tensor: "at::Tensor",
    BaseTy.bool: "bool",
    BaseTy.int: "int64_t",
    BaseTy.SymInt: "c10::SymInt",
    BaseTy.Scalar: "c10::Scalar",
    BaseTy.float: "double",
    BaseTy.str: "::std::string_view",
    BaseTy.DeviceIndex: "c10::DeviceIndex",
    BaseTy.Layout: "c10::Layout",
    BaseTy.MemoryFormat: "c10::MemoryFormat",
    BaseTy.ScalarType: "c10::ScalarType",
    BaseTy.Generator: "at::Generator",
}

base_type_to_callsite_expr = {
    BaseTy.Tensor: "resolve_tensor_dispatch_flags",
    BaseTy.bool: "",
    BaseTy.int: "",
    BaseTy.SymInt: "",
    BaseTy.Scalar: "",
    BaseTy.float: "",
    BaseTy.str: "",
    BaseTy.DeviceIndex: "static_cast<c10::DeviceIndex>",
    BaseTy.Layout: "static_cast<c10::Layout>",
    BaseTy.MemoryFormat: "static_cast<c10::MemoryFormat>",
    BaseTy.ScalarType: "static_cast<c10::ScalarType>",
    BaseTy.Generator: "*generator_handle_to_generator_pointer",
}


# convert args to C types, names in declarations, and expressions in function bodies
def convert_arg_type_and_name(
    typ: Type,
    name: str,
    is_write: bool = False,
) -> tuple[list[str], list[str], list[str], list[str]]:
    if isinstance(typ, BaseType):
        if typ.name in base_type_to_c_type:
            if typ.name == BaseTy.Tensor and is_write:
                # For output tensors, our normal call to resolve_tensor_dispatch_flags
                # results in an rvalue tensor, which can't be passed to at::Tensor&.
                # Override this case specifically.
                callsite_expr = [f"*tensor_handle_to_tensor_pointer({name})"]
            else:
                callsite_expr = [
                    f"{base_type_to_callsite_expr[typ.name]}({name})"
                    if base_type_to_callsite_expr[typ.name]
                    else name
                ]

            return (
                [base_type_to_c_type[typ.name]],
                [name],
                [base_type_to_aten_type[typ.name]],
                callsite_expr,
            )
        elif typ.name == BaseTy.Device:
            return (
                ["int32_t", "int32_t"],
                [name, name + "_index_"],
                ["c10::Device"],
                [
                    f"c10::Device(static_cast<c10::DeviceType>({name}), static_cast<c10::DeviceIndex>({name}_index_))"
                ],
            )
        else:
            # TODO: BaseTy.Dimname, etc.
            raise NotImplementedError(f"TODO: add support for arg type {repr(typ)}")
    elif isinstance(typ, OptionalType):
        c_types, names, aten_types, callsite_exprs = convert_arg_type_and_name(
            typ.elem, name
        )
        j = 0  # index for names
        new_aten_types = []
        new_callsite_exprs = []
        for aten_type in aten_types:
            # Use pointer to denote optional type
            c_types[j] = c_types[j] + "*"
            if aten_type.startswith("c10::ArrayRef<"):
                # ArrayRef is passed as pointer + size, but no need to add "*" to the size argument
                new_aten_types.append(f"::std::optional<{aten_type}>")
                base_type = aten_type[len("c10::ArrayRef<") : -1]
                new_callsite_exprs.append(
                    f"pointer_to_optional_list<{base_type}>({names[j]}, {names[j + 1]})"
                )
                j += 2
            elif aten_type == "c10::Device":
                # Device is passed as device_type + device_index
                new_aten_types.append("::std::optional<c10::Device>")
                new_callsite_exprs.append(
                    f"pointer_to_optional_device({names[j]}, {names[j + 1]})"
                )
                j += 2
            elif aten_type == "at::Tensor":
                new_aten_types.append(f"::std::optional<{aten_type}>")
                new_callsite_exprs.append(f"resolve_tensor_dispatch_flags({names[j]})")
                j += 1
            else:
                new_aten_types.append(f"::std::optional<{aten_type}>")
                new_callsite_exprs.append(
                    f"pointer_to_optional<{aten_type}>({names[j]})"
                )
                j += 1

        return (
            c_types,
            names,
            new_aten_types,
            new_callsite_exprs,
        )
    elif isinstance(typ, ListType):
        # Need to explicitly pass the list as pointer + length
        c_types, names, aten_types, _ = convert_arg_type_and_name(typ.elem, name)
        assert len(c_types) == 1, "ListType with unsupported element type " + repr(typ)

        # The list content should never be modified
        c_types[0] = f"const {c_types[0]}*"
        c_types.append("int64_t")
        name = names[0]
        names.append(name + "_len_")

        atype = aten_types[0]
        callsite_exprs = []
        if atype == "bool":
            # no converter from std::vector<bool> to c10::ArrayRef<bool>
            # construct std::array<bool, N> instead
            assert typ.size is not None
            callsite_exprs.append(f"pointer_to_list<{typ.size}>({name})")
        elif atype == "at::Tensor" and not is_write:
            callsite_exprs.append(
                f"resolve_tensor_list_dispatch_flags({name}, {name}_len_)"
            )
        elif atype == "::std::optional<at::Tensor>":
            # convert from std::vector<::std::optional<at::Tensor>> to c10::List<::std::optional<at::Tensor>>
            callsite_exprs.append(
                f"c10::List<{atype}>(c10::ArrayRef<{atype}>(resolve_tensor_list_dispatch_flags({name}, {name}_len_)))"
            )
        else:
            callsite_exprs.append(f"pointer_to_list<{atype}>({name}, {name}_len_)")

        aten_types = [f"c10::ArrayRef<{t}>" for t in aten_types]
        return (
            c_types,
            names,
            aten_types,
            callsite_exprs,
        )
    raise NotImplementedError(f"Argument type {repr(typ)} not supported!")


def zip_type_and_name(types: list[str], names: list[str]) -> list[str]:
    return [typ + " " + name for typ, name in zip(types, names)]


# Generate argument declarations and callsite expressions
def gen_arguments(
    flat_arguments: Sequence[Argument], skipped_args: set[str]
) -> tuple[list[str], list[str]]:
    types: list[str] = []
    new_names: list[str] = []
    callsite_exprs: list[str] = []
    for arg in flat_arguments:
        if arg.name in skipped_args:
            callsite_exprs.append("std::nullopt")
            continue
        new_types, names, _, new_callsite_exprs = convert_arg_type_and_name(
            arg.type, arg.name, arg.is_write
        )
        types.extend(new_types)
        new_names.extend(names)
        callsite_exprs.extend(new_callsite_exprs)
    return zip_type_and_name(types, new_names), callsite_exprs


# Return values are passed out as pointer arguments because all the C shim functions
# are expected to return AOTITorchError.
# Generate returns as declarations and callsite expressions
def gen_returns(schema: FunctionSchema) -> tuple[list[str], list[str]]:
    types = []
    names = []
    for idx, ret in enumerate(schema.returns):
        names.append(f"ret{idx}")
        if isinstance(ret.type, BaseType) and ret.type.name in base_type_to_c_type:
            types.append(base_type_to_c_type[ret.type.name] + "*")
        else:
            raise NotImplementedError(
                f"TODO: add support for return type {repr(ret.type)}"
            )

    def convert_return(typ: BaseType, val: str) -> str:
        if typ.name == BaseTy.Tensor:
            return f"new_tensor_handle(std::move({val}))"
        elif typ.name == BaseTy.SymInt:
            return f"{val}.expect_int()"
        elif typ.name == BaseTy.Scalar:
            return f"{val}.toDouble()"
        else:
            return val

    ret_pointer_can_be_null = False
    unambiguous_name = schema.name.unambiguous_name()
    for name in (
        "_functional_sym_constrain_range",
        "_scaled_dot_product_cudnn_attention",
        "_scaled_dot_product_efficient_attention_backward",
        "_scaled_dot_product_efficient_attention",
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_fused_attention_overrideable",
        "_thhn_fused_lstm_cell_backward_impl",
        "convolution_backward",
        "grid_sampler_2d_backward",
        "grid_sampler_3d_backward",
        "linear_backward",
    ):
        if name in unambiguous_name:
            ret_pointer_can_be_null = True
            break

    callsite_exprs: list[str] = []
    for idx, ret in enumerate(schema.returns):
        tmp = "tmp_result" if len(names) == 1 else f"std::get<{idx}>(tmp_result)"
        assert isinstance(ret.type, BaseType)
        rval = convert_return(ret.type, tmp)
        if ret_pointer_can_be_null:
            callsite_exprs.append(f"if ({names[idx]}) {{ *{names[idx]} = {rval}; }}")
        else:
            callsite_exprs.append(f"*{names[idx]} = {rval};")

    return zip_type_and_name(types, names), callsite_exprs


# gen.py generates header first and then src, so caching the result here to avoid duplicate work
declaration_definition_cache: dict[tuple[str, str, str], tuple[str, str]] = {}


def gen_declaration_and_definition(
    schema: FunctionSchema,
    device: str,
    backend_call: str,
    version_info: dict[str, list[str]],
) -> tuple[str, str]:
    base_name = schema.name.unambiguous_name()
    if (base_name, device, backend_call) in declaration_definition_cache:
        return declaration_definition_cache[(base_name, device, backend_call)]

    # Check the validity of version_info. The format should look like
    # {"v2" : ["new_arg1"], "v3": ["new_arg2, new_arg3"]}.
    indexed_version_info: dict[int, list[str]] = {1: []}
    for ver_str, new_args in sorted(version_info.items()):
        assert ver_str.startswith("v"), (
            f"Version number for {base_name} is {ver_str}, not starting with 'v'"
        )
        try:
            ver_id = int(ver_str[1:])
        except ValueError as e:
            raise AssertionError(
                f"Version number for {base_name} is {ver_str}, not a valid integer after 'v'"
            ) from e
        assert ver_id not in indexed_version_info, (
            f"{ver_str} for {base_name} has already been defined"
        )
        indexed_version_info[ver_id] = new_args

    declarations: list[str] = []
    definitions: list[str] = []
    skipped_args: set[str] = set()

    for ver_id, new_args in sorted(indexed_version_info.items(), reverse=True):
        # Iterate in the reverse order, so the latest version of an op will get generated first
        # with all the arguments included, while a set of to-be-trimmed args is carried down
        # to generate earlier version of the op.
        func_name = base_name if ver_id == 1 else f"{base_name}_v{ver_id}"
        if schema.is_out_fn():
            # out_variant has out arguments in the front, and it's ok to ignore return values
            # because C shim functions only return AOTITorchError
            args, callsite_exprs = gen_arguments(
                [*schema.arguments.out, *schema.arguments.flat_non_out], skipped_args
            )
            ret_assignments: list[str] = []
        else:
            args, callsite_exprs = gen_arguments(
                schema.arguments.flat_all, skipped_args
            )
            # ignore return values for inplace ops
            ret_declarations, ret_assignments = (
                ([], []) if schema.name.name.inplace else gen_returns(schema)
            )
            args.extend(ret_declarations)

        declaration = textwrap.dedent(
            f"AOTITorchError aoti_torch_{device}_{func_name}({', '.join(args)})"
        )

        tmp_result = "auto tmp_result = " if ret_assignments else ""
        indent = "\t\t"
        ret_assignments_str = (
            "\n".join(indent + r for r in ret_assignments) if ret_assignments else ""
        )
        definition = (
            textwrap.dedent(f"""
        {declaration} {{
            AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({{
                {tmp_result}{backend_call}(
                    {", ".join(callsite_exprs)}
                );
        """)
            + ret_assignments_str
            + textwrap.dedent("""
            });
        }
        """)
        )
        skipped_args.update(new_args)
        declarations.append(f"AOTI_TORCH_EXPORT {declaration};")
        definitions.append(definition)

    declaration_definition_cache[(base_name, device, backend_call)] = (
        "\n".join(declarations),
        "\n".join(definitions),
    )
    return declaration_definition_cache[(base_name, device, backend_call)]


def gen_static_dispatch_backend_call_signature(
    sig: CppSignature | DispatcherSignature,
    f: NativeFunction,
) -> CppSignature:
    sig = DispatcherSignature.from_schema(f.func)
    cpp_sigs = CppSignatureGroup.from_native_function(
        f, method=False, fallback_binding=False
    )
    if sig.symint and f.func.has_symint():
        cpp_sig = cpp_sigs.symint_signature
    else:
        cpp_sig = cpp_sigs.signature
    assert cpp_sig is not None
    return cpp_sig


def gen_static_dispatch_backend_call(
    f: NativeFunction,
    backend_index: Optional[BackendIndex] = None,
) -> str:
    sig = DispatcherSignature.from_schema(f.func)
    cpp_sig = gen_static_dispatch_backend_call_signature(sig, f)
    if backend_index is None:
        return f"at::{cpp_sig.name()}"
    else:
        return f"at::{backend_index.dispatch_key.lower()}::{cpp_sig.name()}"


def get_backend_index_for_aoti(
    func: NativeFunction,
    func_group_mapping: dict[OperatorName, NativeFunctionsGroup],
    dispatch_key: Optional[DispatchKey],
    backend_indices: dict[DispatchKey, BackendIndex],
    extend_aoti_c_shim: bool,
) -> BackendIndex | None:
    backend_index = None

    if dispatch_key is None:
        return backend_index

    if backend_indices[dispatch_key].has_kernel(func) or (
        func.structured_delegate is not None
        and func.structured_delegate in func_group_mapping
        and backend_indices[dispatch_key].has_kernel(
            func_group_mapping[func.structured_delegate]
        )
    ):
        backend_index = backend_indices[dispatch_key]
    else:
        # for the extend out-of-tree kernels, we don't need to
        # duplicatly create C shim wrappers for other dispatch keys
        if extend_aoti_c_shim:
            return backend_index

        elif backend_indices[DispatchKey.CompositeExplicitAutograd].has_kernel(func):
            # We need to create C shim wrappers for CompositeExplicitAutograd kernels
            backend_index = backend_indices[DispatchKey.CompositeExplicitAutograd]
        elif backend_indices[
            DispatchKey.CompositeExplicitAutogradNonFunctional
        ].has_kernel(func):
            # We need to create C shim wrappers for CompositeExplicitAutogradNonFunctional kernels
            backend_index = backend_indices[
                DispatchKey.CompositeExplicitAutogradNonFunctional
            ]
        elif backend_indices[DispatchKey.CompositeImplicitAutograd].has_kernel(func):
            backend_index = backend_indices[DispatchKey.CompositeImplicitAutograd]

    return backend_index


def get_header_for_aoti(
    func: NativeFunction,
    func_group_mapping: dict[OperatorName, NativeFunctionsGroup],
    dispatch_key: Optional[DispatchKey],
    backend_indices: dict[DispatchKey, BackendIndex],
    extend_aoti_c_shim: bool,
) -> str | None:
    backend_index = get_backend_index_for_aoti(
        func, func_group_mapping, dispatch_key, backend_indices, extend_aoti_c_shim
    )
    if backend_index is None:
        if dispatch_key is None:
            return f"#include <ATen/ops/{func.root_name}.h>"
        return None

    return f"#include <ATen/ops/{func.root_name}_{backend_index.dispatch_key.lower()}_dispatch.h>"


def get_fallback_op_name(func: NativeFunction) -> str:
    return (
        f"{func.namespace}.{func.func.name.name}.{func.func.name.overload_name}"
        if func.func.name.overload_name
        else f"{func.namespace}.{func.func.name.name}.default"
    )


def gen_c_shim(
    func: NativeFunction,
    version_info: dict[str, list[str]],
    func_group_mapping: dict[OperatorName, NativeFunctionsGroup],
    dispatch_key: Optional[DispatchKey],
    backend_indices: dict[DispatchKey, BackendIndex],
    header: bool,
    extend_aoti_c_shim: bool,
) -> str | None:
    backend_index = get_backend_index_for_aoti(
        func, func_group_mapping, dispatch_key, backend_indices, extend_aoti_c_shim
    )
    if backend_index is None and dispatch_key is not None:
        return None

    schema = func.func
    device = "aten" if dispatch_key is None else dispatch_key.lower()
    backend_call = gen_static_dispatch_backend_call(
        func,
        backend_index,
    )

    try:
        if header:
            declaration, _ = gen_declaration_and_definition(
                schema, device, backend_call, version_info
            )
            return declaration
        else:
            _, definition = gen_declaration_and_definition(
                schema, device, backend_call, version_info
            )
            return definition

    except NotImplementedError:
        return None


@dataclass(frozen=True)
class ShimGenerator:
    inductor_fallback_ops: dict[str, dict[str, list[str]]]
    func_group_mapping: dict[OperatorName, NativeFunctionsGroup]
    dispatch_key: Optional[DispatchKey]
    backend_indices: dict[DispatchKey, BackendIndex]
    header: bool  # True to generate .h and False to generate .cpp
    extend_aoti_c_shim: bool

    @method_with_native_function
    def __call__(
        self,
        func: NativeFunction,
    ) -> str | None:
        version_info = self.inductor_fallback_ops[get_fallback_op_name(func)]
        result = gen_c_shim(
            func,
            version_info,
            self.func_group_mapping,
            self.dispatch_key,
            self.backend_indices,
            self.header,
            self.extend_aoti_c_shim,
        )
        return result


def gen_aoti_c_shim(
    native_functions: Sequence[NativeFunction],
    inductor_fallback_ops: dict[str, dict[str, list[str]]],
    func_group_mapping: dict[OperatorName, NativeFunctionsGroup],
    dispatch_key: Optional[DispatchKey],
    backend_indices: dict[DispatchKey, BackendIndex],
    header: bool,
    extend_aoti_c_shim: bool,
    includes: str = "",
) -> str:
    body = "\n".join(
        list(
            mapMaybe(
                ShimGenerator(
                    inductor_fallback_ops,
                    func_group_mapping,
                    dispatch_key,
                    backend_indices,
                    header,
                    extend_aoti_c_shim,
                ),
                native_functions,
            )
        )
    )
    device = "aten" if dispatch_key is None else dispatch_key.lower()
    include_device_functions = (
        "#include <ATen/Functions.h>"
        if dispatch_key is None
        else f"#include <ATen/{str(dispatch_key)}Functions.h>"
    )
    aten_warning = (
        (
            "\n\n// This file corresponds to the aten_shimified_ops list in torchgen/aoti/fallback_ops.py\n"
        )
        if dispatch_key is None
        else ""
    )
    warning = """

// WARNING: THIS FILE IS AUTOGENERATED BY torchgen. DO NOT MODIFY BY HAND.
// See https://github.com/pytorch/pytorch/blob/7e86a7c0155295539996e0cf422883571126073e/torchgen/gen.py#L2424-L2436 for details"""

    if header:
        return (
            warning
            + aten_warning
            + textwrap.dedent("""

            #pragma once

            #include <torch/csrc/inductor/aoti_torch/c/shim.h>

            #ifdef __cplusplus
            extern "C" {
            #endif

            """)
            + body
            + textwrap.dedent("""

            #ifdef __cplusplus
            } // extern "C"
            #endif
            """)
        )
    else:
        return (
            warning
            + aten_warning
            + textwrap.dedent(f"""

            #include <torch/csrc/inductor/aoti_torch/generated/{"extend/" if extend_aoti_c_shim else ""}c_shim_{device}.h>
            #include <torch/csrc/inductor/aoti_torch/utils.h>

            #ifndef AT_PER_OPERATOR_HEADERS
            {include_device_functions}
            #include <ATen/CompositeExplicitAutogradFunctions.h>
            #include <ATen/CompositeExplicitAutogradNonFunctionalFunctions.h>
            #include <ATen/CompositeImplicitAutogradFunctions.h>
            #else
            """)
            + includes
            + textwrap.dedent("""
            #endif // AT_PER_OPERATOR_HEADERS

            using namespace torch::aot_inductor;

            """)
            + body
        )


def gen_aoti_c_shim_files(
    aoti_fm: FileManager,
    aoti_backends: set[Optional[DispatchKey]],
    native_functions: Sequence[NativeFunction],
    backend_indices: dict[DispatchKey, BackendIndex],
    structured_native_functions: Sequence[NativeFunctionsGroup],
    extra_cuda_headers: str,
    extend_aoti_c_shim: bool,
    update_aoti_c_shim: bool,
) -> None:
    structured_func_group_dict = {}
    for func_group in structured_native_functions:
        for func in func_group.functions():
            if func.structured_delegate is not None:
                structured_func_group_dict[func.structured_delegate] = func_group
                break

    for dispatch_key in aoti_backends:
        # Use aten_shimified_ops for the aten backend, inductor_fallback_ops for others
        fallback_ops_dict = (
            aten_shimified_ops if dispatch_key is None else inductor_fallback_ops
        )
        fallbacks = {}
        for func in native_functions:
            op_name = get_fallback_op_name(func)
            if op_name in fallback_ops_dict:
                fallbacks[op_name] = func
        fallback_native_functions = tuple(
            value for _, value in sorted(fallbacks.items())
        )

        # Use "aten" as the device name when dispatch_key is Generic
        device_name = "aten" if dispatch_key is None else dispatch_key.lower()

        # header files were checked in for ABI-compatiblilty checking
        header_file_name = f"c_shim_{device_name}.h"
        new_header = gen_aoti_c_shim(
            fallback_native_functions,
            fallback_ops_dict,
            structured_func_group_dict,
            dispatch_key,
            backend_indices,
            header=True,
            extend_aoti_c_shim=extend_aoti_c_shim,
            includes="",
        )
        if update_aoti_c_shim:
            aoti_fm.write(
                header_file_name,
                lambda: new_header,
            )
        else:
            try:
                with open(
                    os.path.join(aoti_fm.install_dir, header_file_name)
                ) as old_file:
                    old_header = old_file.read()

                    if old_header != new_header:
                        diff = "\n".join(
                            difflib.unified_diff(
                                old_header.splitlines(),
                                new_header.splitlines(),
                                fromfile="expected",
                                tofile="actual",
                                lineterm="",
                            )
                        )

                        raise RuntimeError(f"""
The generated AOTInductor C shim header files have unexpectedly changed. This
indicates an AOTInductor fallback operator ABI backward compatibility breakage!!!
Only in a limited number of situations, this is allowed:

1. You added a fallback op to the inductor_fallback_ops list in torchgen/aoti/fallback_ops.py.
If that's the case, run `python torchgen/gen.py --update-aoti-c-shim` to add a new entry to
existing C shim header files.

2. You added a new default argument to an existing fallback op. This is clearly a BC breaking
change in the AOTInductor land. You need to annotate the new default argument in
torchgen/aoti/fallback_ops.py, and then run `python torchgen/gen.py --update-aoti-c-shim` to
update the C shim header files by creating different versions of the fallback op. See
https://github.com/pytorch/pytorch/pull/154848 as an example.

{diff}
                    """)
            except FileNotFoundError:
                print(
                    f"{os.path.join(aoti_fm.install_dir, header_file_name)} not found"
                )

        # cpp files are always generated on-the-fly
        def headers_for_aoti() -> str:
            headers = []
            for func in fallback_native_functions:
                header = get_header_for_aoti(
                    func,
                    structured_func_group_dict,
                    dispatch_key,
                    backend_indices,
                    extend_aoti_c_shim=extend_aoti_c_shim,
                )
                if header is not None:
                    headers.append(header)
            return "\n".join(sorted(set(headers)))

        extra_headers = (
            extra_cuda_headers
            if dispatch_key is not None and is_cuda_dispatch_key(dispatch_key)
            else ""
        )

        aoti_fm.write(
            f"c_shim_{device_name}.cpp",
            lambda: gen_aoti_c_shim(
                fallback_native_functions,
                inductor_fallback_ops,
                structured_func_group_dict,
                dispatch_key,
                backend_indices,
                header=False,
                extend_aoti_c_shim=extend_aoti_c_shim,
                includes=headers_for_aoti() + "\n" + extra_headers,
            ),
        )
