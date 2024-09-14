from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, TYPE_CHECKING

from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    ListType,
    NativeFunction,
    OptionalType,
    Type,
)


if TYPE_CHECKING:
    from torchgen.api.types import Binding, CType, NamedCType


connector = "\n\t"


# Return unboxing function name for a NativeFunction
def name(f: NativeFunction) -> str:
    return f.func.name.unambiguous_name()


@dataclass(frozen=True)
class Unboxing:
    """
    Takes a sequence of Bindings and unbox EValues to these Bindings. Return generated code that performs correct unboxing.
    A sample generated code:
    // aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    void mul_out(EValue** stack) {
        EValue& self = *stack[0];
        EValue& other = *stack[1];
        EValue& out = *stack[2];
        const torch::executor::Tensor & self_base = self.to<torch::executor::Tensor>();
        const torch::executor::Tensor & other_base = other.to<torch::executor::Tensor>();
        torch::executor::Tensor & out_base = out.to<torch::executor::Tensor>();

        EXECUTORCH_SCOPE_PROF("native_call_mul.out");
        torch::executor::mul_outf(self_base, other_base, out_base);


    }
    """

    # this is a callable that converts a JIT argument, into its C++ type.
    # Translates (type, mutability, binds) to NamedCType. E.g., torchgen.api.cpp.argumenttype_type.
    argument_type_gen: Callable[
        ...,
        NamedCType,
    ]

    # Convert all the arguments in a NativeFunction to C++ code
    def convert_arguments(
        self, args: Sequence[Binding]
    ) -> tuple[list[Binding], list[str]]:
        code_list = [f"EValue& {args[i].name} = *stack[{i}];" for i in range(len(args))]
        binding_list = []
        for arg in args:
            # expecting only Argument
            if not isinstance(arg.argument, Argument):
                raise Exception(  # noqa: TRY002
                    f"Unexpected argument type, expecting `Argument` but got {arg}"
                )
            argument: Argument = arg.argument
            unboxed_name, _, code, decl = self.argumenttype_evalue_convert(
                argument.type, argument.name, mutable=argument.is_write
            )
            code_list.extend(decl)
            code_list.extend(code)
            binding_list.append(arg.with_name(unboxed_name))
        return binding_list, code_list

    def argumenttype_evalue_convert(
        self, t: Type, arg_name: str, *, mutable: bool = False
    ) -> tuple[str, CType, list[str], list[str]]:
        """
        Takes in the type, name and mutability corresponding to an argument, and generates a tuple of:
        (1) the C++ code necessary to unbox the argument
        (2) A Binding corresponding to the newly created unboxed variable, including variable name and its CType
        :param t: a `Type` of an argument
        :param arg_name: argument name
        :param mutable: boolean for whether this argument type is mutable
        :return: unboxed result
        """
        ctype = self.argument_type_gen(t, mutable=mutable, binds=arg_name).type

        if isinstance(t, BaseType):
            out_name = f"{arg_name}_base"
            code, decl = self._gen_code_base_type(
                arg_name=arg_name, out_name=out_name, ctype=ctype
            )
        elif isinstance(t, OptionalType):
            out_name = f"{arg_name}_opt_out"
            code, decl = self._gen_code_optional_type(
                arg_name=arg_name, out_name=out_name, t=t, ctype=ctype
            )
        elif isinstance(t, ListType):
            out_name = f"{arg_name}_list_out"
            code, decl = self._gen_code_list_type(
                arg_name=arg_name, out_name=out_name, t=t, ctype=ctype
            )
        else:
            raise Exception(  # noqa: TRY002
                f"Cannot handle type {t}. arg_name: {arg_name}"
            )  # noqa: TRY002
        return out_name, ctype, code, decl

    def _gen_code_base_type(
        self, arg_name: str, out_name: str, ctype: CType
    ) -> tuple[list[str], list[str]]:
        return [
            f"{ctype.cpp_type()} {out_name} = {arg_name}.to<{ctype.cpp_type(strip_ref=True)}>();"
        ], []

    def _gen_code_optional_type(
        self, arg_name: str, out_name: str, t: OptionalType, ctype: CType
    ) -> tuple[list[str], list[str]]:
        in_name = f"{arg_name}_opt_in"
        res_name, base_type, res_code, decl = self.argumenttype_evalue_convert(
            t.elem, in_name
        )
        return (
            f"""
    auto {out_name} = {arg_name}.toOptional<{base_type.cpp_type(strip_ref=True)}>();
            """.split(
                "\n"
            ),
            decl,
        )

    def _gen_code_list_type(
        self, arg_name: str, out_name: str, t: ListType, ctype: CType
    ) -> tuple[list[str], list[str]]:
        in_name = f"{arg_name}_list_in"
        elem_name = f"{arg_name}_elem"
        code = []
        res_name, res_ctype, res_code, decl = self.argumenttype_evalue_convert(
            t.elem, elem_name
        )

        if isinstance(t.elem, BaseType) and t.elem.name == BaseTy.Tensor:
            code.extend(
                f"""
    auto {out_name} = {arg_name}.toTensorList();
                """.split(
                    "\n"
                )
            )
        elif isinstance(t.elem, BaseType) and (
            t.elem.name == BaseTy.int or t.elem.name == BaseTy.SymInt
        ):
            code.extend(
                f"""
    auto {out_name} = {arg_name}.toIntList();
                """.split(
                    "\n"
                )
            )
        elif isinstance(t.elem, BaseType) and t.elem.name == BaseTy.float:
            code.extend(
                f"""
    auto {out_name} = {arg_name}.toDoubleList();
                """.split(
                    "\n"
                )
            )
        elif isinstance(t.elem, BaseType) and t.elem.name == BaseTy.bool:
            # handle list type with size, e.g., bool[4]
            code.extend(
                f"""
#ifdef USE_ATEN_LIB
std::array<bool, {t.size}> {out_name};
auto {in_name} = {arg_name}.toBoolList();
size_t _i = 0;
for (auto {elem_name}: {in_name}) {{
    {out_name}[_i++] = {elem_name};
}}
#else
auto {out_name} = {arg_name}.toBoolList();
#endif
                """.split(
                    "\n"
                )
            )
        # pytorch codegen:
        # we have to use c10::List for optional element. e.g., Tensor?[] -> c10::List<::std::optional<at::Tensor>>
        elif (
            isinstance(t.elem, OptionalType)
            and isinstance(t.elem.elem, BaseType)
            and t.elem.elem.name == BaseTy.Tensor
        ):
            code.extend(
                f"""
#ifdef USE_ATEN_LIB
auto {in_name} = {arg_name}.toListOptionalTensor();
c10::List<::std::optional<at::Tensor>> {out_name};
for (auto {elem_name}: {in_name}) {{
    {out_name}.push_back({elem_name});
}}
#else
auto {out_name} = {arg_name}.toListOptionalTensor();
#endif
                """.split(
                    "\n"
                )
            )
        else:
            # use ArrayRef as default.
            vec_name = arg_name + "_vec"
            # need to bring vector instantiation out of scope so that ArrayRef has valid data
            decl.append(
                f"std::vector<{res_ctype.cpp_type(strip_ref=True)}> {vec_name};"
            )
            code.extend(
                f"""
    for (EValue {elem_name}: {in_name}) {{
        {connector.join(res_code)}
        {vec_name}.push_back({res_name});
    }}
    {ctype.cpp_type(strip_ref=True)} {out_name}({vec_name});
                """.split(
                    "\n"
                )
            )
        return code, decl
