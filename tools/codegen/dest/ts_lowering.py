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
                                 gets_generated_out_inplace_wrapper)
from tools.codegen.api.types import (BaseTy, BaseCppType, BaseCType, OptionalCType,
                                     Binding, ConstRefCType, NamedCType,
                                     CppSignature, CppSignatureGroup,
                                     Expr, MutRefCType, kernel_signature,
                                     DispatcherSignature)
import tools.codegen.api.meta as meta
import tools.codegen.api.cpp as cpp
import tools.codegen.api.structured as structured
from tools.codegen.api.translate import translate
from tools.codegen.selective_build.selector import SelectiveBuilder
from .lazy_ir import process_ir_types, ir_node_name


@dataclass(frozen=True)
class TsLowering:
    backend_index: BackendIndex
    TsLoweringTarget = Enum('TsLoweringTarget', (
        'DISPATCH',  # an entry in the top-level Lower func that dispatches to impls
        'LOWERING',  # an impl of a particular lowering
    ))

    target: Union[
        Literal[TsLoweringTarget.DISPATCH],
        Literal[TsLoweringTarget.LOWERING],
    ]

    # The namespace that the kernels are written in. This is just `at::native` for in-tree kernels.
    cpp_namespace: str

    @method_with_native_function
    def __call__(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        if func.name in self.backend_index.index:
            return self.gen(f)
        else:
            return []

    def gen(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        # for now, we just want one IR class decl and soon after also the method defs
        # and we use the functional version not out/inplace.
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        class_name = str(func.name).lower().capitalize()

        if self.target == TsLowering.TsLoweringTarget.DISPATCH:
            return [f"""\
case at::aten::{func.name}:
    return Lower{ir_node_name(func)}(node);
""", ]


        elif self.target == TsLowering.TsLoweringTarget.LOWERING: 
            all_types, value_types, scalar_types = process_ir_types(func)
            emplace_values = [f"loctx()->GetOutputOp(node->operand({i}))" for i in range(len(value_types))]
            emplace_scalars = [f"node->{t.name}()" for t in scalar_types]
            emplace_arguments = "\n    ".join([f"arguments.emplace_back({a});" for a in emplace_values + emplace_scalars])
            return [f"""\
TSOpVector Lower{ir_node_name(func)}(const ir::ops::{ir_node_name(func)}* node) {{
    std::vector<torch::jit::NamedValue> arguments;
    arguments.reserve({len(all_types)});
    {emplace_arguments}
    TSOpVector {func.name}_out = LowerBuiltin(node, arguments);
    LTC_CHECK_EQ({func.name}_out.size(), len(func.returns));
    
    // TODO: GenerateClone always?
    // TODO: handle multi-return cases
    return GenerateClone({func.name}_out.front()); 
}}
""", ]
