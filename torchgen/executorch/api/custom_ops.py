from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Set, Union

from torchgen import dest

from torchgen.context import method_with_native_function

from torchgen.gen import get_native_function_schema_registrations
from torchgen.model import BackendIndex, DispatchKey, NativeFunction, Variant

# disable import sorting to avoid circular dependency.
from torchgen.api.types import CppSignatureGroup  # isort:skip
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import concatMap, FileManager, mapMaybe, Target

# Generates RegisterKernelStub.cpp, which provides placeholder kernels for custom operators. This will be used at model authoring side.
@dataclass(frozen=True)
class ComputeNativeFunctionStub:
    # pyre-fixme[31]: Expression `Target.DEFINITION` is not a literal value.
    # pyre-fixme[31]: Expression `Target.REGISTRATION` is not a literal value.
    # pyre-fixme[11]: Annotation `Target` is not defined as a type.
    target: Union[Literal[Target.REGISTRATION], Literal[Target.DEFINITION]]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if Variant.function not in f.variants:
            return None
        sig_group = CppSignatureGroup.from_native_function(
            f, method=False, fallback_binding=f.manual_cpp_binding
        )

        sig = sig_group.most_faithful_signature()
        assert sig is not None
        if len(f.func.returns) == 0:
            ret_name = ""
        elif len(f.func.returns) == 1:
            if f.func.arguments.out:
                ret_name = f.func.arguments.out[0].name
            else:
                ret_name = next(
                    a.name
                    for a in f.func.arguments.flat_non_out
                    if a.type == f.func.returns[0].type
                )
            if not ret_name:
                raise Exception(f"Can't handle this return type {f.func}")
        else:
            assert len(f.func.arguments.out) == len(f.func.returns), (
                "Out variant number of returns need to match the number of out arguments."
                f" Got outs {str(f.func.arguments.out)} but returns {str(f.func.returns)}"
            )
            # returns a tuple of out arguments
            tensor_type = "at::Tensor &"
            comma = ", "
            ret_name = f"""::std::tuple<{comma.join([tensor_type] * len(f.func.returns))}>(
                {comma.join([r.name for r in f.func.arguments.out])}
            )"""
        ret_str = f"return {ret_name};" if len(f.func.returns) > 0 else ""
        # pyre-fixme[16]: `Enum` has no attribute `DEFINITION`.
        if self.target is Target.DEFINITION:
            return f"""
{sig.defn()} {{
    {ret_str}
}}
    """
        # pyre-fixme[16]: `Enum` has no attribute `REGISTRATION`.
        elif self.target is Target.REGISTRATION:
            return f"""
    m.impl("{f.func.name}", {sig.name()});
    """


def gen_custom_ops(
    *,
    native_functions: Sequence[NativeFunction],
    selector: SelectiveBuilder,
    backend_indices: Dict[DispatchKey, BackendIndex],
    cpu_fm: FileManager,
    functions_keys: Set[DispatchKey],
    dispatch_keys: Sequence[DispatchKey],
    rocm: bool,
):
    ns_grouped_native_functions: Dict[str, List[NativeFunction]] = defaultdict(list)
    for native_function in native_functions:
        ns_grouped_native_functions[native_function.namespace].append(native_function)
    impl_registrations = ""
    newline = "\n\t"
    for namespace, funcs in ns_grouped_native_functions.items():
        if len(funcs) == 0:
            continue
        impl_registrations_body = list(
            # pyre-fixme[19]: Expected 0 positional arguments.
            # pyre-fixme[16]: `Enum` has no attribute `REGISTRATION`.
            mapMaybe(ComputeNativeFunctionStub(Target.REGISTRATION), funcs)
        )
        impl_registrations += f"""
TORCH_LIBRARY_IMPL({namespace}, CPU, m) {{
{newline.join(impl_registrations_body)}
}};"""
    cpu_fm.write(
        "RegisterKernelStub.cpp",
        lambda: {
            "native_function_definition_stub": list(
                # pyre-fixme[19]: Expected 0 positional arguments.
                # pyre-fixme[16]: `Enum` has no attribute `DEFINITION`.
                mapMaybe(ComputeNativeFunctionStub(Target.DEFINITION), native_functions)
            ),
            "static_init_dispatch_registrations": impl_registrations,
        },
    )

    def operator_headers() -> List[str]:
        headers = ['#include "NativeFunctions.h"']
        return headers

    (
        aten_schema_registrations,
        schema_registrations,
    ) = get_native_function_schema_registrations(
        native_functions=native_functions,
        schema_selector=selector,
    )
    cpu_fm.write(
        "RegisterSchema.cpp",
        lambda: {
            "schema_registrations": schema_registrations,
            "aten_schema_registrations": aten_schema_registrations,
        },
    )
    ns_grouped_custom_ops = defaultdict(list)
    for grouped_native_function in native_functions:
        namespace = (
            grouped_native_function.namespace
            if isinstance(grouped_native_function, NativeFunction)
            # pyre-fixme[16]: NativeFunction has no attribute functional.
            else grouped_native_function.functional.namespace
        )
        ns_grouped_custom_ops[namespace].append(grouped_native_function)
    for dispatch_key in dispatch_keys:
        backend_index = backend_indices[dispatch_key]

        static_init_dispatch_registrations = ""
        for namespace, functions in ns_grouped_custom_ops.items():

            dispatch_registrations_body = "\n".join(
                list(
                    concatMap(
                        # pyre-fixme[19]: Expected 2 positional arguments.
                        dest.RegisterDispatchKey(
                            backend_index,
                            Target.REGISTRATION,
                            selector,
                            rocm=rocm,
                            symint=False,
                            class_method_name=None,
                            skip_dispatcher_op_registration=False,
                        ),
                        functions,
                    )
                )
            )
            static_init_dispatch_registrations += f"""
TORCH_LIBRARY_IMPL({namespace}, {dispatch_key}, m) {{
    {dispatch_registrations_body}
}};"""
        cpu_fm.write_with_template(
            f"Register{dispatch_key}CustomOps.cpp",
            "RegisterDispatchKeyCustomOps.cpp",
            lambda: {
                "ops_headers": operator_headers(),
                "DispatchKey": dispatch_key,
                "dispatch_namespace": dispatch_key.lower(),
                "dispatch_namespaced_definitions": "",
                "dispatch_anonymous_definitions": list(
                    concatMap(
                        # pyre-fixme[19]: Expected 2 positional arguments.
                        dest.RegisterDispatchKey(
                            backend_index,
                            # pyre-fixme[16]: `Enum` has no attribute
                            #  `ANONYMOUS_DEFINITION`.
                            Target.ANONYMOUS_DEFINITION,
                            selector,
                            rocm=rocm,
                            symint=False,
                            class_method_name=None,
                            skip_dispatcher_op_registration=False,
                        ),
                        native_functions,
                    )
                ),
                "static_init_dispatch_registrations": static_init_dispatch_registrations,
            },
        )
