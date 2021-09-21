from typing import List, Union
from typing_extensions import Literal
from dataclasses import dataclass
from enum import Enum
from tools.codegen.context import method_with_native_function
from tools.codegen.model import (NativeFunction, NativeFunctionsGroup,
                                 BackendIndex)
from tools.codegen.api.lazy import LazyIrSchema


@dataclass(frozen=True)
class LazyTsLowering:
    backend_index: BackendIndex

    TsLoweringTarget = Enum('TsLoweringTarget', (
        'DISPATCH',  # an entry in the top-level Lower func that dispatches to impls
        'LOWERING',  # an impl of a particular lowering
    ))

    target: Union[
        Literal[TsLoweringTarget.DISPATCH],
        Literal[TsLoweringTarget.LOWERING],
    ]

    @method_with_native_function
    def __call__(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        return self.gen(f)

    def gen(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        # for now, we just want one IR class decl and soon after also the method defs
        # and we use the functional version not out/inplace.
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        schema = LazyIrSchema(func)

        if self.target == LazyTsLowering.TsLoweringTarget.DISPATCH:
            return [f"""\
case at::aten::{schema.aten_name}:
    return Lower{schema.node_name}(function,
                loctx,
                ir::NodeCast<ir::ops::{schema.node_name}>(node, ir::OpKind(at::aten::{schema.aten_name})));
""", ]

        elif self.target == LazyTsLowering.TsLoweringTarget.LOWERING:
            emplace_arg_values = [f'loctx->GetOutputOp(node->operand({i}))' for i in range(len(schema.positional_values))]
            emplace_arg_scalars = [f'"{t.name}", node->{t.name}_' for t in schema.positional_scalars]
            emplace_arguments = "\n    ".join(
                [f"arguments.emplace_back({a});" for a in emplace_arg_values + emplace_arg_scalars])
            emplace_kwarg_values = [f'loctx->GetOutputOp(node->operand({i}))' for i in range(len(schema.keyword_values))]
            emplace_kwarg_scalars = [f'"{t.name}", node->{t.name}_' for t in schema.keyword_scalars]
            assert len(schema.keyword_values) == 0, "TODO the logic for operand(i) is broken if there are kw values"
            emplace_kwarguments = "\n    ".join(
                [f"kwarguments.emplace_back({a});" for a in emplace_kwarg_values + emplace_kwarg_scalars])
            return [f"""\
TSOpVector Lower{schema.node_name}(std::shared_ptr<torch::jit::GraphFunction> function,
                                   ts_backend::TSLoweringContext* loctx,
                                   const ir::ops::{schema.node_name}* node) {{
    std::vector<torch::jit::NamedValue> arguments;
    std::vector<torch::jit::NamedValue> kwarguments;
    arguments.reserve({len(emplace_arg_values + emplace_arg_scalars)});
    kwarguments.reserve({len(emplace_kwarg_values + emplace_kwarg_scalars)});
    {emplace_arguments}
    {emplace_kwarguments}
    TSOpVector {schema.aten_name}_out = LowerBuiltin(function, node, arguments, kwarguments);
    LTC_CHECK_EQ({schema.aten_name}_out.size(), {len(func.returns)});

    // TODO: need to call GenerateClone sometimes? Or else return LowerBuiltIn() directly
    return {schema.aten_name}_out;
}}
""", ]
