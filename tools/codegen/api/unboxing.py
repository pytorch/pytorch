from typing import List, Tuple

from tools.codegen.api.cpp import argumenttype_type
from tools.codegen.api.types import CType, Expr
from tools.codegen.model import (
    Argument,
    Type,
    BaseType,
    OptionalType,
    ListType,
    BaseTy,
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
#           c10::optional<c10::IValue> names_opt = (std::move(peek(stack, 1, 7))).toOptional<c10::IValue>();
#           c10::optional<at::ArrayRef<at::Dimname>> names_opt_out;
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
#               names_opt_out = c10::optional<at::ArrayRef<at::Dimname>>(names_list_out);
#           } else {
#               names_opt_out = c10::optional<at::ArrayRef<at::Dimname>>();
#           }
#       ```
#    - ScalarType? dtype (similarly for the rest of the arguments)
#       ```cpp
#           c10::optional<c10::IValue> dtype_opt = (std::move(peek(stack, 2, 7))).toOptional<c10::IValue>();
#           c10::optional<at::ScalarType> dtype_opt_out;
#           if (dtype_opt.has_value()) {
#               const c10::IValue dtype_opt_in = dtype_opt.value();
#               at::ScalarType dtype_base = dtype_opt_in.to<at::ScalarType>();
#                                                        ~~~~~~~~~~~~~~~~~~~~ <-- For base types, convert ivalue to it
#                                                                                 directly using ".to<T>()" API.
#               dtype_opt_out = c10::optional<at::ScalarType>(dtype_base);
#           } else {
#               dtype_opt_out = c10::optional<at::ScalarType>();
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


# Convert all the arguments in a NativeFunction to C++ code, including TensorOptions.
def convert_arguments(args: List[Argument]) -> Tuple[List[Expr], List[str]]:
    argument_str = "c10::IValue {arg_name} = std::move(peek(stack, {pos}, {args_num}));"
    expr_list = []
    pop_ivalue = []
    code_list = []
    for i, arg in enumerate(args):
        pop_ivalue.append(argument_str.format(arg_name=arg.name, pos=i, args_num=len(args)))
        expr, code = argumenttype_ivalue_convert(arg.type, arg.name, mutable=arg.is_write)
        expr_list.append(expr)
        code_list.extend(code)
    pop_ivalue.append("")
    return expr_list, pop_ivalue + code_list


# Take an argument in JIT type format, returns the C++ code to convert an ivalue from stack to corresponding C++ type.
def argumenttype_ivalue_convert(t: Type, arg_name: str, *, mutable: bool = False) -> Tuple[Expr, List[str]]:
    nctype = argumenttype_type(t=t, mutable=mutable, binds=arg_name)
    ctype = nctype.cpp_type(strip_ref=True)

    if isinstance(t, BaseType):
        out_name = f"{arg_name}_base"
        code = [f"{ctype} {arg_name}_base = {arg_name}.to<{ctype}>();"]
    elif isinstance(t, OptionalType):
        out_name = arg_name + "_opt_out"
        code = _gen_code_optional_type(arg_name=arg_name, out_name=out_name, t=t, ctype=ctype)
    elif isinstance(t, ListType):
        out_name = arg_name + "_list_out"
        code = _gen_code_list_type(arg_name=arg_name, out_name=out_name, t=t, ctype=ctype)
    else:
        raise Exception(f"Cannot handle type {t}. arg_name: {arg_name}")
    expr = Expr(expr=out_name, type=nctype)
    return expr, code


def _gen_code_optional_type(arg_name: str, out_name: str, t: OptionalType, ctype: str) -> List[str]:
    in_name = arg_name + "_opt_in"
    res_expr, res_code = argumenttype_ivalue_convert(t.elem, in_name)
    return f"""
c10::optional<c10::IValue> {arg_name + "_opt"} = {arg_name}.toOptional<c10::IValue>();
{ctype} {out_name};
if ({arg_name + "_opt"}.has_value()) {{
    const c10::IValue {in_name} = {arg_name + "_opt"}.value();
    {connector.join(res_code)}
    {out_name} = {ctype}({res_expr.expr});
}} else {{
    {out_name} = {ctype}();
}}
        """.split("\n")


def _gen_code_list_type(arg_name: str, out_name: str, t: ListType, ctype: str) -> List[str]:
    in_name = arg_name + "_list_in"
    elem_name = arg_name + "_elem"
    code = [f"const c10::List<c10::IValue> {in_name} = {arg_name}.toList();"]
    res_expr, res_code = argumenttype_ivalue_convert(t.elem, elem_name)
    # handle list type with size, e.g., bool[4]
    if isinstance(t.elem, BaseType) and t.elem.name == BaseTy.bool and t.size:
        code.extend(
            f"""
{ctype} {out_name} = as_array<{res_expr.type.cpp_type(strip_ref=True)}, {t.size}>({in_name});
            """.split(
                "\n"
            )
        )
    # we have to use c10::List for optional element. e.g., Tensor?[] -> c10::List<c10::optional<at::Tensor>>
    elif isinstance(t.elem, OptionalType):
        code.extend(
            f"""
{ctype} {out_name};
for (c10::IValue {elem_name}: {in_name}) {{
    {connector.join(res_code)}
    {out_name}.push_back({res_expr.expr});
}}
            """.split(
                "\n"
            )
        )
    else:
        # use ArrayRef as default.
        vec_name = arg_name + "_vec"
        code.extend(
            f"""
std::vector<{res_expr.type.cpp_type(strip_ref=True)}> {vec_name};
for (c10::IValue {elem_name}: {in_name}) {{
    {connector.join(res_code)}
    {vec_name}.push_back({res_expr.expr});
}}
{ctype} {out_name}({vec_name});
            """.split(
                "\n"
            )
        )
    return code
