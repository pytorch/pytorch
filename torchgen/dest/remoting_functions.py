from __future__ import annotations

import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.types import kernel_signature
from torchgen.context import with_native_function_and_index
from torchgen.model import BackendIndex, NativeFunction, NativeFunctionsGroup, ListType, BaseTy
from torchgen.utils import mapMaybe

def torch_api_key_word_prefix(bankend_index: BackendIndex) -> str:
    if bankend_index.external:
        return ""

    # Although Intel GPU ATen library is out-of-tree, it still utilizes torchgen to produce structrued
    # kernels. Regarding these produced structured kernels, they should be visible for the Intel GPU ATen
    # library. Therefore, we need to add "TORCH_XPU_API" prefix to these structured kernels,
    # rather than "TORCH_API". Because the semantic of "TORCH_API" is "hidden" for out-of-tree backends.
    # For other in-tree backends like cpu and cuda, they still use "TORCH_API" prefix with "visible" semantic.
    device_torch_api_key_word_mapping = {
        "XPU": "TORCH_XPU_API",
    }

    return (
        device_torch_api_key_word_mapping.get(
            bankend_index.dispatch_key.name, "TORCH_API"
        )
        + " "
    )

def get_scalar_return_value(return_ty, name=None):
    name = name or (return_ty.name and f"ret_{return_ty.name}") or "ret"
    if return_ty.type.is_list_like():
        if not return_ty.type.is_base_ty_like(BaseTy.Tensor):
            import pdb;pdb.set_trace()

        return (
            [f"std::vector<at::Tensor> {name}"],
            name,
        )

    if  return_ty.type.is_base_ty_like(BaseTy.bool):
        # boolean
        return (
            [f"bool {name} = true"],
            name,
        )

    if  return_ty.type.is_base_ty_like(BaseTy.float):
        # boolean
        return (
            [f"float {name} = 0.0"],
            name,
        )

    if  return_ty.type.is_base_ty_like(BaseTy.SymInt):
        # boolean
        return (
            [f"c10::SymInt {name} = 0"],
            name,
        )

    if  return_ty.type.is_base_ty_like(BaseTy.Scalar):
        # scalar (?)
        if return_ty.is_write:
            import pdb;pdb.set_trace()

        return (
                [f"at::Scalar {name} = at::Scalar()"],
                name,
            )

    if  return_ty.type.is_base_ty_like(BaseTy.ScalarType):

        # scalartype (?)
        if return_ty.is_write:
            import pdb;pdb.set_trace()

        return (
                [f"at::ScalarType {name} = at::ScalarType()"],
                name,
            )

    if return_ty.type.is_base_ty_like(BaseTy.Tensor):
        if return_ty.is_write:
            # tensor &
            return (
                [f"at::Tensor *{name} = new at::Tensor()"],
                f"*{name}",
            )
        else:
            # tensor
            return (
                [f"at::Tensor {name} = at::Tensor()"],
                name,
            )

    if return_ty.type.is_base_ty_like(BaseTy.int):

        return (
            [f"int64_t {name} = 0"],
            name,
        )

    print(return_ty.type)
    import pdb;pdb.set_trace()
    pass


def get_return_value(returns):
    if len(returns) == 0:
        return ([], "")

    elif len(returns) == 1:
        return get_scalar_return_value(returns[0])
    else:
        pass

    # tuple
    prepare_return_statements = []
    tuple_values = []
    for idx, tuple_return_value in enumerate(returns):
        name = tuple_return_value.name or f"elt{idx}"
        tuple_value_prepare_statement, tuple_value = get_scalar_return_value(tuple_return_value, (name and f"ret_{name}") or None)
        prepare_return_statements += tuple_value_prepare_statement
        tuple_values.append(tuple_value)

    tuple_content = ", ".join([f"std::ref({tuple_name})" for tuple_name in tuple_values])

    prepare_return_statements += [
        f"auto tuple = std::make_tuple({tuple_content})",
    ]

    return (
        prepare_return_statements,
        "tuple"
    )


# need to use MutRefCType # torchgen/api/cpp.py

@with_native_function_and_index
def gen_unstructured_remoting_frontend(f: NativeFunction, backend_index: BackendIndex) -> str | None:
    sig = kernel_signature(f, backend_index)
    metadata = backend_index.get_kernel(f)
    if metadata is None:
        return None
    if "legacy::" in metadata.kernel:
        return None

    if "_mps" not in metadata.kernel:
        return None

    if metadata.kernel in ("_mps_convolution_out_symint", "_mps_convolution_transpose_out_symint", "lstm_mps_backward_out", "_lstm_mps_out"):
        return

    prefix = "static" if backend_index.external else "TORCH_API"

    ret_type = sig.returns_type().cpp_type()
    #print(ret_type)

    prepare_return_statements, return_value = get_return_value(sig.func.returns)

    code = f"""{prefix} {sig.decl(name=metadata.kernel)} {{

    {';\n    '.join(prepare_return_statements)};

    return {return_value};
}}
    """
    #print(code)

    return code


@with_native_function_and_index
def gen_structured_remoting_frontend(g: NativeFunctionsGroup, backend_index: BackendIndex) -> list[str]:
    meta_name = meta.name(g)
    out_args = structured.impl_arguments(g)
    metadata = backend_index.get_kernel(g)
    if metadata is None:
        return []
    prefix = torch_api_key_word_prefix(backend_index)
    return [
        f"""\
struct {prefix}structured_{metadata.kernel} : public at::meta::structured_{meta_name} {{
    void impl({', '.join(a.decl() for a in out_args)}) {{
        return;
    }}
}};
"""
    ]


# Generates NativeFunctions.h, a list of forward declarations of all
# actual kernel definitions we keep in aten/src/ATen/native/
@with_native_function_and_index
def compute_native_function_remoting_frontend(
    g: NativeFunctionsGroup | NativeFunction, backend_index: BackendIndex
) -> list[str]:
    #raise Error("hell")
    metadata = backend_index.get_kernel(g)
    if isinstance(g, NativeFunctionsGroup):
        if metadata is not None and metadata.structured:
            if backend_index.external:
                # Structured hasn't been tested with external backends yet.
                raise AssertionError(
                    "Structured external backend functions are not implemented yet."
                )
            else:
                return gen_structured_remoting_frontend(g, backend_index)
        else:
            return list(
                mapMaybe(lambda f: gen_unstructured_remoting_frontend(f, backend_index), g.functions())
            )
    else:
        x = gen_unstructured_remoting_frontend(g, backend_index)
        return [] if x is None else [x]
