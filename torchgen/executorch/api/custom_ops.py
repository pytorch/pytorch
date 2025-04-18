from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchgen import dest


# disable import sorting to avoid circular dependency.
from torchgen.api.types import DispatcherSignature  # usort: skip
from torchgen.context import method_with_native_function
from torchgen.model import BaseTy, BaseType, DispatchKey, NativeFunction, Variant
from torchgen.utils import concatMap, Target


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torchgen.executorch.model import ETKernelIndex
    from torchgen.selective_build.selector import SelectiveBuilder


# Generates RegisterKernelStub.cpp, which provides placeholder kernels for custom operators. This will be used at
# model authoring side.
@dataclass(frozen=True)
class ComputeNativeFunctionStub:
    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str | None:
        if Variant.function not in f.variants:
            return None

        sig = DispatcherSignature.from_schema(
            f.func, prefix=f"wrapper_CPU_{f.func.name.overload_name}_", symint=False
        )
        assert sig is not None
        if len(f.func.returns) == 0:
            ret_name = ""
        elif len(f.func.returns) == 1:
            if f.func.arguments.out:
                ret_name = f.func.arguments.out[0].name
            else:
                ret_name = next(
                    (
                        a.name
                        for a in f.func.arguments.flat_non_out
                        if a.type == f.func.returns[0].type
                    ),
                    "",
                )
            if not ret_name:
                # if return type is tensor
                if f.func.returns[0].type == BaseType(BaseTy.Tensor):
                    # Returns an empty tensor
                    ret_name = "at::Tensor()"
                else:
                    raise Exception(  # noqa: TRY002
                        f"Can't handle this return type {f.func}"
                    )  # noqa: TRY002
        elif len(f.func.arguments.out) == len(f.func.returns):
            # Returns a tuple of out arguments
            tensor_type = "at::Tensor &"
            comma = ", "
            ret_name = f"""::std::tuple<{comma.join([tensor_type] * len(f.func.returns))}>(
                {comma.join([r.name for r in f.func.arguments.out])}
            )"""
        else:
            assert all(a.type == BaseType(BaseTy.Tensor) for a in f.func.returns), (
                f"Only support tensor returns but got {f.func.returns}"
            )
            # Returns a tuple of empty tensors
            tensor_type = "at::Tensor"
            comma = ", "
            ret_name = f"""::std::tuple<{comma.join([tensor_type] * len(f.func.returns))}>(
                {comma.join(["at::Tensor()" for _ in f.func.returns])}
            )"""
        ret_str = f"return {ret_name};" if len(f.func.returns) > 0 else ""
        return f"""
{sig.defn()} {{
    {ret_str}
}}
    """


def gen_custom_ops_registration(
    *,
    native_functions: Sequence[NativeFunction],
    selector: SelectiveBuilder,
    kernel_index: ETKernelIndex,
    rocm: bool,
) -> tuple[str, str]:
    """
    Generate custom ops registration code for dest.RegisterDispatchKey.

    :param native_functions: a sequence of `NativeFunction`
    :param selector: for selective build.
    :param kernel_index: kernels for all the ops.
    :param rocm: bool for dest.RegisterDispatchKey.
    :return: generated C++ code to register custom operators into PyTorch
    """

    # convert kernel index to BackendIndex. This is because we can't handle ETKernelIndex yet.
    # TODO larryliu: evaluate if this code is still needed. If yes let it handle ETKernelIndex.

    dispatch_key = DispatchKey.CPU
    backend_index = kernel_index._to_backend_index()
    static_init_dispatch_registrations = ""
    ns_grouped_native_functions: dict[str, list[NativeFunction]] = defaultdict(list)
    for native_function in native_functions:
        ns_grouped_native_functions[native_function.namespace].append(native_function)

    for namespace, functions in ns_grouped_native_functions.items():
        if len(functions) == 0:
            continue
        dispatch_registrations_body = "\n".join(
            list(
                concatMap(
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
}}"""
    anonymous_definition = "\n".join(
        list(
            concatMap(
                dest.RegisterDispatchKey(
                    backend_index,
                    Target.ANONYMOUS_DEFINITION,
                    selector,
                    rocm=rocm,
                    symint=False,
                    class_method_name=None,
                    skip_dispatcher_op_registration=False,
                ),
                native_functions,
            )
        )
    )
    return anonymous_definition, static_init_dispatch_registrations
