from typing import List, Optional, Union
import itertools
from typing_extensions import Literal
from dataclasses import dataclass
import textwrap

from tools.codegen.context import method_with_native_function, native_function_manager
from tools.codegen.utils import Target, mapMaybe
from tools.codegen.model import (DispatchKey, NativeFunction,
                                 NativeFunctionsGroup, SchemaKind,
                                 TensorOptionsArguments,
                                 DeviceCheckType, Argument, assert_never,
                                 is_cuda_dispatch_key, BackendIndex,
                                 gets_generated_out_inplace_wrapper)
from tools.codegen.api.types import (BaseCType, Binding, ConstRefCType,
                                     CppSignature, CppSignatureGroup,
                                     Expr, MutRefCType, kernel_signature,
                                     NativeSignature, tensorT, NamedCType,
                                     DispatcherSignature)
import tools.codegen.api.meta as meta
import tools.codegen.api.cpp as cpp
import tools.codegen.api.structured as structured
from tools.codegen.api.translate import translate
from tools.codegen.selective_build.selector import SelectiveBuilder

# Generates {backend}_lazy_ir.h and .cpp
#


@dataclass(frozen=True)
class LazyIR:
    backend_index: BackendIndex

    target: Union[
        Literal[Target.DEFINITION],
        Literal[Target.DECLARATION],
    ]

    # TODO(whc) probably use selector instead of building a separate index for ops to codegen
    # Selector object to determine which operators to generate
    # registration code for.
    selector: SelectiveBuilder

    # The namespace that the kernels are written in. This is just `at::native` for in-tree kernels.
    cpp_namespace: str

    @method_with_native_function
    def __call__(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        if f.func.name in self.backend_index.index:
            return self.gen(f)
        else:
            return []

    def gen(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        # for now, we just want one IR class decl and soon after also the method defs
        return ["// blah"]

# TODO(whc) do we stick with the GenericOp concept or just make a class per IR?
#   NodePtr {name}(const Value& input) {                       \
    # return GenericOp(OpKind({sym}), {input}, input.shape()); \
  