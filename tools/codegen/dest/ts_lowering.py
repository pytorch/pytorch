from typing import List, Optional, Union, Tuple
import itertools
from typing_extensions import Literal
from dataclasses import dataclass
from enum import Enum
import textwrap
from tools.codegen import local
from tools.codegen.context import method_with_native_function, native_function_manager
from tools.codegen.utils import Target, mapMaybe
from tools.codegen.model import (BaseType, OptionalType, DispatchKey, NativeFunction,
                                 NativeFunctionsGroup, SchemaKind, FunctionSchema,
                                 TensorOptionsArguments, ListType,
                                 DeviceCheckType, Argument, assert_never,
                                 is_cuda_dispatch_key, BackendIndex,
                                 gets_generated_out_inplace_wrapper, OperatorName,
                                 SelfArgument, Arguments)
from tools.codegen.api.types import (BaseTy, BaseCppType, BaseCType, OptionalCType,
                                     Binding, ConstRefCType, NamedCType,
                                     CppSignature, CppSignatureGroup,
                                     Expr, MutRefCType, kernel_signature,
                                     DispatcherSignature, ListCType)
import tools.codegen.api.meta as meta
import tools.codegen.api.cpp as cpp
import tools.codegen.api.structured as structured
from tools.codegen.api.translate import translate
from tools.codegen.selective_build.selector import SelectiveBuilder
from .lazy_ir import ir_node_name, valueT, valueListT, update_schema_for_lazy_ir, separate_value_scalar_types


@dataclass(frozen=True)
class TsLowering:
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

        if self.target == TsLowering.TsLoweringTarget.DISPATCH:
            return [f"""\
case at::aten::{func.name}:
    return Lower{ir_node_name(func)}(function, loctx, ir::NodeCast<ir::ops::{ir_node_name(func)}>(node, ir::OpKind(at::aten::{func.name})));
""", ]

        elif self.target == TsLowering.TsLoweringTarget.LOWERING:
            schema = update_schema_for_lazy_ir(func)
            all_types, value_types, scalar_types = separate_value_scalar_types(schema)
            emplace_values = [f"loctx->GetOutputOp(node->operand({i}))" for i in range(len(value_types))]
            emplace_scalars = [f"node->{t.name}_" for t in scalar_types]
            emplace_arguments = "\n    ".join(
                [f"arguments.emplace_back({a});" for a in emplace_values + emplace_scalars])
            return [f"""\
TSOpVector Lower{ir_node_name(func)}(std::shared_ptr<torch::jit::GraphFunction> function, ts_backend::TSLoweringContext* loctx, const ir::ops::{ir_node_name(func)}* node) {{
    std::vector<torch::jit::NamedValue> arguments;
    arguments.reserve({len(all_types)});
    {emplace_arguments}
    TSOpVector {func.name}_out = LowerBuiltin(function, node, arguments);
    LTC_CHECK_EQ({func.name}_out.size(), {len(func.returns)});
    
    // TODO: need to call GenerateClone sometimes? Or else return LowerBuiltIn() directly
    return {func.name}_out;
}}
""", ]
