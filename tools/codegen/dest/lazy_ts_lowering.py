from typing import Union
from tools.codegen.model import (NativeFunction, NativeFunctionsGroup)
from tools.codegen.api.lazy import LazyIrSchema, isValueType
from tools.codegen.api.types import OptionalCType


def ts_lowering_body(f: Union[NativeFunctionsGroup, NativeFunction]) -> str:
    # for now, we just want one IR class decl and soon after also the method defs
    # and we use the functional version not out/inplace.
    func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
    schema = LazyIrSchema(func)

    emplace_arguments = []
    for value in schema.positional_arg_types:
        if isValueType(value.type):
            if isinstance(value.type, OptionalCType):
                emplace_arguments.append(f"has_{value.name} ? loctx->GetOutputOp(operand(i++)) : nullptr")
                continue
            emplace_arguments.append('loctx->GetOutputOp(operand(i++))')
            continue
        emplace_arguments.append(f'"{value.name}", {value.name}')

    emplace_arguments_str = "\n    ".join(
        [f"arguments.emplace_back({a});" for a in emplace_arguments])
    emplace_kwarg_values = [f'"{t.name}", loctx->GetOutputOp(operand(i++))' for t in schema.keyword_values]
    emplace_kwarg_scalars = [f'"{t.name}", {t.name}' for t in schema.keyword_scalars]
    emplace_kwarguments = "\n    ".join(
        [f"kwarguments.emplace_back({a});" for a in emplace_kwarg_values + emplace_kwarg_scalars])
    return f"""\
    std::vector<torch::jit::NamedValue> arguments;
    std::vector<torch::jit::NamedValue> kwarguments;
    arguments.reserve({len(emplace_arguments)});
    kwarguments.reserve({len(emplace_kwarg_values + emplace_kwarg_scalars)});
    size_t i = 0;
    {emplace_arguments_str}
    {emplace_kwarguments}
    torch::lazy::TSOpVector {schema.aten_name}_out = torch::lazy::LowerTSBuiltin(function, op().op, arguments, kwarguments);
    CHECK_EQ({schema.aten_name}_out.size(), {len(func.returns)});

    return {schema.aten_name}_out;
"""
