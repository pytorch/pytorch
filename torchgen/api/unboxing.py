from __future__ import annotations

from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignatureGroup, CType
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    ListType,
    NativeFunction,
    OptionalType,
    Type,
)


# This file generates the code for unboxing wrappers, i.e., the glue logic to unbox a boxed operator and convert the
# ivalues from stack to correct arguments to the unboxed kernel, based on corresponding JIT schema. This codegen is
# an alternative way to generate unboxing wrappers similar to the existing C++ metaprogramming approach but gets the
# job done statically. These generated unboxing wrappers will be useful under the scenario where we need to register
# a fixed set of operators known at compile time and thus can save some time in runtime initialization phase.
#
# Here's an example on how the codegen works:
#
# - Function Schema (source of truth)
#
#      aten::empty.names(int[] size, *, Dimname[]? names,
#                        ScalarType? dtype=None, Layout? layout=None,
#                        Device? device=None, bool? pin_memory=None,
#                        MemoryFormat? memory_format=None) -> Tensor
# - Argument Conversion
#       Generates C++ code to convert an ivalue (from stack) to its underlying C++ type.
#    - int[] size
#        ```cpp
#           const c10::List<c10::IValue> size_list_in = (std::move(peek(stack, 0, 7))).toList();
#
#           std::vector<int64_t> size_vec;
#           for (c10::IValue size_elem: size_list_in) {
#               int64_t size_base = size_elem.to<int64_t>();
#               size_vec.push_back(size_base);
#           }
#           at::ArrayRef<int64_t> size_list_out(size_vec);
#                                 ~~~~~~~~~~~~~ <-- The converted argument from ivalues in the stack.
#                                                   Will be passed to unboxed kernel.
#       ```
#    - Dimname[]? names
#       ```cpp
#           ::std::optional<c10::IValue> names_opt = (std::move(peek(stack, 1, 7))).toOptional<c10::IValue>();
#           ::std::optional<at::ArrayRef<at::Dimname>> names_opt_out;
#           if (names_opt.has_value()) {
#                         ~~~~~~~~~~~ <-- Unwrapping optional shell
#               const c10::IValue names_opt_in = names_opt.value();
#               const c10::List<c10::IValue> names_list_in = names_opt_in.toList();
#
#               std::vector<at::Dimname> names_vec;
#               for (c10::IValue names_elem: names_list_in) {
#                                ~~~~~~~~~~~~~~~~~~~~~~~~~ <-- Unrolling list, then convert elements one by one.
#                   at::Dimname names_base = names_elem.to<at::Dimname>();
#                   names_vec.push_back(names_base);
#               }
#               at::ArrayRef<at::Dimname> names_list_out(names_vec);
#
#               names_opt_out = ::std::optional<at::ArrayRef<at::Dimname>>(names_list_out);
#           } else {
#               names_opt_out = ::std::optional<at::ArrayRef<at::Dimname>>();
#           }
#       ```
#    - ScalarType? dtype (similarly for the rest of the arguments)
#       ```cpp
#           ::std::optional<c10::IValue> dtype_opt = (std::move(peek(stack, 2, 7))).toOptional<c10::IValue>();
#           ::std::optional<at::ScalarType> dtype_opt_out;
#           if (dtype_opt.has_value()) {
#               const c10::IValue dtype_opt_in = dtype_opt.value();
#               at::ScalarType dtype_base = dtype_opt_in.to<at::ScalarType>();
#                                                        ~~~~~~~~~~~~~~~~~~~~ <-- For base types, convert ivalue to it
#                                                                                 directly using ".to<T>()" API.
#               dtype_opt_out = ::std::optional<at::ScalarType>(dtype_base);
#           } else {
#               dtype_opt_out = ::std::optional<at::ScalarType>();
#           }
#       ```
#
# - Unboxed Kernel Call
#   ```cpp
#       auto result_ = torch::empty(
#           size_list_out,
#           names_opt_out,
#           options,
#           memory_format_opt_out
#       );
#   ```
#
# - Push Result Back to Stack
#   ```cpp
#       drop(stack, 7);
#       pack(stack, std::move(result_));
#   ```
connector = "\n\t"


# Return unboxing function name for a NativeFunction
def name(f: NativeFunction) -> str:
    return f.func.name.unambiguous_name()


# Convert all the arguments in a NativeFunction to C++ code
def convert_arguments(f: NativeFunction) -> tuple[list[Binding], list[str]]:
    # we need the 'self' argument so method needs to be False
    args = (
        CppSignatureGroup.from_native_function(f, method=False)
        .most_faithful_signature()
        .arguments()
    )
    code_list = [
        f"c10::IValue {args[i].name} = std::move(peek(stack, {i}, {len(args)}));"
        for i in range(len(args))
    ] + [""]
    binding_list = []
    for arg in args:
        # expecting only Argument
        if not isinstance(arg.argument, Argument):
            raise Exception(  # noqa: TRY002
                f"Unexpected argument type, expecting `Argument` but got {arg}"
            )
        argument: Argument = arg.argument
        unboxed_name, _, code, decl = argumenttype_ivalue_convert(
            argument.type,
            argument.name,
            mutable=argument.is_write,
        )
        code_list.extend(decl)
        code_list.extend(code)
        binding_list.append(arg.with_name(unboxed_name))
    return binding_list, code_list


# Takes in the type, name and mutability corresponding to an argument, and generates a tuple of:
# (1) the C++ code necessary to unbox the argument
# (2) A Binding corresponding to the newly created unboxed variable, including variable name and its CType
def argumenttype_ivalue_convert(
    t: Type, arg_name: str, *, mutable: bool = False
) -> tuple[str, CType, list[str], list[str]]:
    # Unboxing is for mobile, which doesn't care about SymInts
    ctype = cpp.argumenttype_type(
        t=t, mutable=mutable, binds=arg_name, symint=False
    ).type

    if isinstance(t, BaseType):
        out_name = f"{arg_name}_base"
        code, decl = _gen_code_base_type(
            arg_name=arg_name, out_name=out_name, ctype=ctype
        )
    elif isinstance(t, OptionalType):
        out_name = f"{arg_name}_opt_out"
        code, decl = _gen_code_optional_type(
            arg_name=arg_name,
            out_name=out_name,
            t=t,
            ctype=ctype,
        )
    elif isinstance(t, ListType):
        out_name = f"{arg_name}_list_out"
        code, decl = _gen_code_list_type(
            arg_name=arg_name,
            out_name=out_name,
            t=t,
            ctype=ctype,
        )
    else:
        raise Exception(f"Cannot handle type {t}. arg_name: {arg_name}")  # noqa: TRY002
    return out_name, ctype, code, decl


def _gen_code_base_type(
    arg_name: str, out_name: str, ctype: CType
) -> tuple[list[str], list[str]]:
    return [
        f"{ctype.cpp_type(strip_ref=True)} {out_name} = {arg_name}.to<{ctype.cpp_type(strip_ref=True)}>();"
    ], []


def _gen_code_optional_type(
    arg_name: str, out_name: str, t: OptionalType, ctype: CType
) -> tuple[list[str], list[str]]:
    in_name = f"{arg_name}_opt_in"
    res_name, _, res_code, decl = argumenttype_ivalue_convert(t.elem, in_name)
    return (
        f"""
auto {arg_name}_opt = {arg_name}.toOptional<c10::IValue>();
{ctype.cpp_type(strip_ref=True)} {out_name};
if ({arg_name}_opt.has_value()) {{
    const c10::IValue {in_name} = {arg_name}_opt.value();
    {connector.join(res_code)}
    {out_name} = {ctype.cpp_type(strip_ref=True)}({res_name});
}} else {{
    {out_name} = {ctype.cpp_type(strip_ref=True)}();
}}
        """.split(
            "\n"
        ),
        decl,
    )


def _gen_code_list_type(
    arg_name: str, out_name: str, t: ListType, ctype: CType
) -> tuple[list[str], list[str]]:
    in_name = f"{arg_name}_list_in"
    elem_name = f"{arg_name}_elem"
    code = [f"const c10::List<c10::IValue> {in_name} = {arg_name}.toList();"]
    res_name, res_ctype, res_code, decl = argumenttype_ivalue_convert(t.elem, elem_name)
    # handle list type with size, e.g., bool[4]
    if isinstance(t.elem, BaseType) and t.elem.name == BaseTy.bool and t.size:
        code.extend(
            f"""
{ctype.cpp_type(strip_ref=True)} {out_name} = as_array<{res_ctype.cpp_type(strip_ref=True)}, {t.size}>({in_name});
            """.split(
                "\n"
            )
        )
    # we have to use c10::List for optional element. e.g., Tensor?[] -> c10::List<::std::optional<at::Tensor>>
    elif isinstance(t.elem, OptionalType):
        code.extend(
            f"""
{ctype.cpp_type(strip_ref=True)} {out_name};
for (c10::IValue {elem_name}: {in_name}) {{
    {connector.join(res_code)}
    {out_name}.push_back({res_name});
}}
            """.split(
                "\n"
            )
        )
    else:
        # use ArrayRef as default.
        vec_name = arg_name + "_vec"
        # need to bring vector instantiation out of scope so that ArrayRef has valid data
        decl.append(f"std::vector<{res_ctype.cpp_type(strip_ref=True)}> {vec_name};")
        code.extend(
            f"""
for (c10::IValue {elem_name}: {in_name}) {{
    {connector.join(res_code)}
    {vec_name}.push_back({res_name});
}}
{ctype.cpp_type(strip_ref=True)} {out_name}({vec_name});
            """.split(
                "\n"
            )
        )
    return code, decl
