from torchgen.api.lazy import LazyIrSchema
from torchgen.api.types import OptionalCType


def ts_lowering_body(schema: LazyIrSchema) -> str:
    # for now, we just want one IR class decl and soon after also the method defs
    # and we use the functional version not out/inplace.
    emplace_arguments = []
    for arg in schema.positional_args:
        if arg.is_lazy_value:
            if isinstance(arg.lazy_type, OptionalCType):
                emplace_arguments.append(
                    f"has_{arg.name} ? loctx->GetOutputOp(operand(i++)) : nullptr"
                )
                continue
            emplace_arguments.append("loctx->GetOutputOp(operand(i++))")
            continue
        emplace_arguments.append(f'"{arg.name}", {arg.name}')

    emplace_arguments_str = "\n    ".join(
        [f"arguments.emplace_back({a});" for a in emplace_arguments]
    )
    emplace_kwarg_values = [
        f'"{arg.name}", loctx->GetOutputOp(operand(i++))'
        for arg in schema.keyword_values
    ]
    emplace_kwarg_scalars = [
        f'"{arg.name}", {arg.name}' for arg in schema.keyword_scalars
    ]
    emplace_kwarguments = "\n    ".join(
        [
            f"kwarguments.emplace_back({a});"
            for a in emplace_kwarg_values + emplace_kwarg_scalars
        ]
    )
    return f"""\
    std::vector<torch::jit::NamedValue> arguments;
    std::vector<torch::jit::NamedValue> kwarguments;
    arguments.reserve({len(emplace_arguments)});
    kwarguments.reserve({len(emplace_kwarg_values + emplace_kwarg_scalars)});
    size_t i = 0;
    {emplace_arguments_str}
    {emplace_kwarguments}
    torch::lazy::TSOpVector {schema.aten_name}_out = torch::lazy::LowerTSBuiltin(function, op().op, arguments, kwarguments);
    TORCH_CHECK_EQ({schema.aten_name}_out.size(), {len(schema.returns)});

    return {schema.aten_name}_out;
"""
