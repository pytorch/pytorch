from dataclasses import dataclass
from typing import Dict, List, Optional

from tools.codegen.api import cpp
from tools.codegen.api.cpp import argumenttype_type
from tools.codegen.api.types import BaseCType, tensorOptionsT, CppSignature, CType, voidT
from tools.codegen.model import (
    Argument,
    Type,
    BaseType,
    OptionalType,
    ListType,
    BaseTy,
    TensorOptionsArguments,
    NativeFunction,
    Variant,
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


@dataclass(frozen=True)
class ArgumentCppCode:
    val_name: str
    code: List[str]
    ctype: str


# Convert all the arguments in a NativeFunction to C++ code, including TensorOptions.
def convert_arguments(
    args: List[Argument], tensor_option_arg: Optional[TensorOptionsArguments]
) -> Dict[str, ArgumentCppCode]:
    argument_str = "(std::move(peek(stack, {pos}, {args_num})))"
    arguments: Dict[str, ArgumentCppCode] = {}
    tensor_option_args: Dict[str, str] = {}
    for i, arg in enumerate(args):
        ivalue_str = argument_str.format(pos=i, args_num=len(args))
        res = argumenttype_ivalue_convert(arg.type, ivalue_str, arg.name)
        # handle tensor options and other keyword arguments
        if tensor_option_arg and arg.name in tensor_option_arg.__dict__:
            tensor_option_args[arg.name] = res.val_name
        arguments[arg.name] = res

        # only generate tensor options when native function is taking it
    if tensor_option_arg:
        arguments["options"] = ArgumentCppCode(
            val_name="options",
            code=[
                f"""
        const auto options = TensorOptions()
            .dtype({tensor_option_args['dtype']})
            .layout({tensor_option_args['layout']})
            .device({tensor_option_args['device']})
            .pinned_memory({tensor_option_args['pin_memory']});
                """
            ],
            ctype="at::TensorOptions",
        )
    return arguments


# Take an argument in JIT type format, returns the C++ code to convert an ivalue from stack to corresponding C++ type.
def argumenttype_ivalue_convert(t: Type, ival: str, arg_name: str) -> ArgumentCppCode:
    if isinstance(t, BaseType):
        ctype = argumenttype_type(t=t, mutable=True, binds=arg_name).cpp_type(
            strip_ref=True
        )
        return ArgumentCppCode(
            code=[f"{ctype} {arg_name}_base = {ival}.to<{ctype}>();"],
            val_name=f"{arg_name}_base",
            ctype=ctype,
        )
    elif isinstance(t, OptionalType):
        in_name = arg_name + "_opt_in"
        out_name = arg_name + "_opt_out"
        connector = "\n\t"
        res = argumenttype_ivalue_convert(t.elem, in_name, arg_name)
        ctype = f"c10::optional<{res.ctype}>"
        code = f"""
c10::optional<c10::IValue> {arg_name + "_opt"} = {ival}.toOptional<c10::IValue>();
{ctype} {out_name};
if ({arg_name + "_opt"}.has_value()) {{
    const c10::IValue {in_name} = {arg_name + "_opt"}.value();
    {connector.join(res.code)}
    {out_name} = {ctype}({res.val_name});
}} else {{
    {out_name} = {ctype}();
}}
        """.split(
            "\n"
        )
        return ArgumentCppCode(
            code=code,
            val_name=out_name,
            ctype=ctype,
        )
    elif isinstance(t, ListType):
        in_name = arg_name + "_list_in"
        out_name = arg_name + "_list_out"
        elem_name = arg_name + "_elem"
        code = [f"const c10::List<c10::IValue> {in_name} = {ival}.toList();"]
        res = argumenttype_ivalue_convert(t.elem, elem_name, arg_name)
        connector = "\n\t"
        # handle list type with size, e.g., bool[4]
        if isinstance(t.elem, BaseType) and t.elem.name == BaseTy.bool and t.size:
            ctype = f"std::array<{res.ctype}, {t.size}>"
            code.extend(
                f"""
{ctype} {out_name} = as_array<{res.ctype}, {t.size}>({in_name});
            """.split(
                    "\n"
                )
            )
        # we have to use c10::List for optional element. e.g., Tensor?[] -> c10::List<c10::optional<at::Tensor>>
        elif isinstance(t.elem, OptionalType):
            ctype = f"c10::List<{res.ctype}>"
            code.extend(
                f"""
{ctype} {out_name};
for (c10::IValue {elem_name}: {in_name}) {{
    {connector.join(res.code)}
    {out_name}.push_back({res.val_name});
}}
            """.split(
                    "\n"
                )
            )
        else:
            # use ArrayRef as default.
            ctype = f"at::ArrayRef<{res.ctype}>"
            vec_name = arg_name + "_vec"
            code.extend(
                f"""
std::vector<{res.ctype}> {vec_name};
for (c10::IValue {elem_name}: {in_name}) {{
    {connector.join(res.code)}
    {vec_name}.push_back({res.val_name});
}}
{ctype} {out_name}({vec_name});
            """.split(
                    "\n"
                )
            )
        return ArgumentCppCode(
            code=code,
            val_name=out_name,
            ctype=ctype,
        )
    else:
        raise Exception(f"Cannot handle type {t}. ival: {ival}, arg_name: {arg_name}")


# Generate code to call C++ unboxed kernel with argument variable names
def generate_unboxed_kernel_call(
    f: NativeFunction, sig: CppSignature, arguments: Dict[str, ArgumentCppCode]
) -> List[str]:
    use_tensor_options = any(
        isinstance(a.nctype.type, BaseCType) and a.nctype.type.type == tensorOptionsT
        for a in sig.arguments()
    )
    arg_connector = ",\n\t"
    # find dispatch native function namespace
    if use_tensor_options:
        namespace = "torch"
    else:
        namespace = "at"
    # handle void return type
    ret_type: CType = cpp.returns_type(f.func.returns)
    if isinstance(ret_type, BaseCType) and ret_type.type == voidT:
        ret_str = ""
        push_str = ""
    else:
        ret_str = "auto result_ = "
        push_str = """
pack(stack, std::move(result_));
        """
    if Variant.method in f.variants:
        self_arg = f.func.arguments.self_arg
        assert self_arg is not None, "No self argument"
        arg_list = arg_connector.join(
            [
                arguments[a.name].val_name
                for a in sig.arguments()
                if a.name != self_arg.argument.name
            ]
        )
        if arg_list:
            arg_list = f"\n\t{arg_list}\n"
        function_call = f"""
{ret_str}{arguments[self_arg.argument.name].val_name}.{sig.name()}({arg_list});
    """
    else:
        arg_list = arg_connector.join(
            [arguments[a.name].val_name for a in sig.arguments()]
        )
        if arg_list:
            arg_list = f"\n\t{arg_list}\n"
        function_call = f"""
{ret_str}{namespace}::{sig.name()}({arg_list});
    """
    return (function_call + push_str).split("\n")
