from typing import List, Optional, Union, Tuple
import itertools
from typing_extensions import Literal
from dataclasses import dataclass
from enum import Enum
import textwrap
from tools.codegen import local
from tools.codegen.context import method_with_native_function, native_function_manager
from tools.codegen.utils import Target, mapMaybe
from tools.codegen.model import (BaseTy, BaseType, OptionalType, DispatchKey, NativeFunction,
                                 NativeFunctionsGroup, SchemaKind, FunctionSchema,
                                 TensorOptionsArguments, ListType,
                                 DeviceCheckType, Argument, assert_never,
                                 is_cuda_dispatch_key, BackendIndex,
                                 gets_generated_out_inplace_wrapper, OperatorName,
                                 SelfArgument, Arguments)
from tools.codegen.api.types import (BaseCppType, BaseCType, OptionalCType,
                                     Binding, ConstRefCType, NamedCType,
                                     CppSignature, CppSignatureGroup,
                                     Expr, MutRefCType, kernel_signature,
                                     DispatcherSignature, ListCType)
import tools.codegen.api.meta as meta
import tools.codegen.api.cpp as cpp
import tools.codegen.api.structured as structured
from tools.codegen.api.translate import translate
from tools.codegen.selective_build.selector import SelectiveBuilder
from tools.codegen.api.lazy import ir_node_name, valueT, update_schema_for_lazy_ir, isValueType

def separate_args_kwargs_types(func: FunctionSchema) -> Tuple[List[NamedCType], List[NamedCType], List[NamedCType], List[NamedCType], List[NamedCType]]:
    """
    Since there are significant differences in how Values and 'everything else' are handled in the IR node class,
    it's useful to have a way to get a list of just the values, just the scalars, or all of them.
    """
    args = func.arguments
    types = [NamedCType(arg.name, arg.type) for arg in func.schema_order_arguments()]
    positional_values = [NamedCType(arg.name, arg.type) for arg in args.flat_positional if isValueType(arg.type)]
    positional_scalars = [NamedCType(arg.name, arg.type) for arg in args.flat_positional if not isValueType(arg.type)]
    kw_values = [NamedCType(arg.name, arg.type) for arg in args.flat_kwarg_only if isValueType(arg.type)]
    kw_scalars = [NamedCType(arg.name, arg.type) for arg in args.flat_kwarg_only if not isValueType(arg.type)]
    
    return types, positional_values, positional_scalars, kw_values, kw_scalars

@dataclass(frozen=True)
class LazyTsLowering:
    backend_index: BackendIndex

    # Names of operators we want to codegen for, a subset of backend_index
    codegen: List[OperatorName]

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
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        if func.name in self.codegen:
            return self.gen(f)
        else:
            return []

    def gen(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        # for now, we just want one IR class decl and soon after also the method defs
        # and we use the functional version not out/inplace.
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func

        if self.target == LazyTsLowering.TsLoweringTarget.DISPATCH:
            return [f"""\
case at::aten::{func.name}:
    return Lower{ir_node_name(func)}(function, loctx, ir::NodeCast<ir::ops::{ir_node_name(func)}>(node, ir::OpKind(at::aten::{func.name})));
""", ]

        elif self.target == LazyTsLowering.TsLoweringTarget.LOWERING:
            schema = update_schema_for_lazy_ir(func)
            types, positional_values, positional_scalars, kw_values, kw_scalars = separate_args_kwargs_types(schema)
            emplace_arg_values = [f'loctx->GetOutputOp(node->operand({i}))' for i in range(len(positional_values))]
            emplace_arg_scalars = [f'"{t.name}", node->{t.name}_' for t in positional_scalars]
            emplace_arguments = "\n    ".join(
                [f"arguments.emplace_back({a});" for a in emplace_arg_values + emplace_arg_scalars])
            emplace_kwarg_values = [f'loctx->GetOutputOp(node->operand({i}))' for i in range(len(kw_values))]
            emplace_kwarg_scalars = [f'"{t.name}", node->{t.name}_' for t in kw_scalars]
            assert len(kw_values) == 0, "TODO the logic for operand(i) is broken if there are kw values"
            emplace_kwarguments = "\n    ".join(
                [f"kwarguments.emplace_back({a});" for a in emplace_kwarg_values + emplace_kwarg_scalars])
            return [f"""\
TSOpVector Lower{ir_node_name(func)}(std::shared_ptr<torch::jit::GraphFunction> function, ts_backend::TSLoweringContext* loctx, const ir::ops::{ir_node_name(func)}* node) {{
    std::vector<torch::jit::NamedValue> arguments;
    std::vector<torch::jit::NamedValue> kwarguments;
    arguments.reserve({len(emplace_arg_values + emplace_arg_scalars)});
    kwarguments.reserve({len(emplace_kwarg_values + emplace_kwarg_scalars)});
    {emplace_arguments}
    {emplace_kwarguments}
    TSOpVector {func.name}_out = LowerBuiltin(function, node, arguments, kwarguments);
    LTC_CHECK_EQ({func.name}_out.size(), {len(func.returns)});
    
    // TODO: need to call GenerateClone sometimes? Or else return LowerBuiltIn() directly
    return {func.name}_out;
}}
""", ]
