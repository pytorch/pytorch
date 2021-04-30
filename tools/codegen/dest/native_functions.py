from typing import List, Union, Optional

from tools.codegen.context import with_native_function_and_index
from tools.codegen.utils import mapMaybe
from tools.codegen.model import (NativeFunction, NativeFunctionsGroup,
                                 BackendIndex, is_structured_dispatch_key)
from tools.codegen.api.types import kernel_signature
import tools.codegen.api.meta as meta
import tools.codegen.api.structured as structured

@with_native_function_and_index
def gen_unstructured(f: NativeFunction, backend_index: BackendIndex) -> Optional[str]:
    sig = kernel_signature(f, backend_index)
    name = backend_index.kernel(f)
    is_external = backend_index.is_external(f)
    # sigh... I should be able to better enforce that these are all or nothing.
    if sig is None or name is None or is_external is None:
        return None
    if "legacy::" in name:
        return None
    else:
        prefix = 'static' if is_external else 'TORCH_API'
        return f"{prefix} {sig.decl(name=name)};"

@with_native_function_and_index
def gen_structured(g: NativeFunctionsGroup, backend_index: BackendIndex) -> List[str]:
    # TODO: consider baking is_structured_dispatch_key directly into BackendIndex's notion of structured.
    # I'm not sure if that will play nicely with CompositeExplicitAutograd though.
    if is_structured_dispatch_key(backend_index.dispatch_key):
        # only out has dispatch
        meta_name = meta.name(g)
        out_args = structured.impl_arguments(g)
        name = backend_index.kernel(backend_index.primary(g))
        is_external = backend_index.is_external(g)
        if name is None or is_external is None:
            return []
        prefix = 'static' if is_external else 'TORCH_API'
        return [f"""\
struct {prefix} structured_{name} : public at::meta::{meta_name} {{
void impl({', '.join(a.decl() for a in out_args)});
}};
"""]
    else:
        return list(mapMaybe(lambda f: gen_unstructured(f, backend_index), g.functions()))

# Generates NativeFunctions.h, a list of forward declarations of all
# actual kernel definitions we keep in aten/src/ATen/native/
@with_native_function_and_index
def compute_native_function_declaration(
        g: Union[NativeFunctionsGroup, NativeFunction],
        backend_index: BackendIndex
) -> List[str]:
    structured = backend_index.structured(g)
    is_external = backend_index.is_external(g)
    if isinstance(g, NativeFunctionsGroup):
        if structured:
            if is_external:
                # Structured hasn't been tested with external backends yet.
                raise AssertionError("Structured external backend functions are not implemented yet.")
            else:
                return gen_structured(g, backend_index)
        else:
            return list(mapMaybe(lambda f: gen_unstructured(f, backend_index), g.functions()))
    else:
        x = gen_unstructured(g, backend_index)
        return [] if x is None else [x]
