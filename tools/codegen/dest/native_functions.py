from typing import List, Union, Set, Any

from tools.codegen.context import *
from tools.codegen.utils import *
from tools.codegen.model import *
from tools.codegen.api.types import *
import tools.codegen.api.meta as meta
import tools.codegen.api.native as native
import tools.codegen.api.structured as structured

@with_native_function
def gen_unstructured(f: NativeFunction) -> List[str]:
    ns = list(f.dispatch.values())

    rs = []
    # Sometimes a function name shows up multiple times; only generate
    # it once!
    seen = set()
    for n in ns:
        if n in seen:
            continue
        if "legacy::" in n:
            continue
        seen.add(n)
        returns_type = native.returns_type(f.func.returns)
        args = native.arguments(f.func)
        rs.append(f"TORCH_API {returns_type} {n}({', '.join(a.decl() for a in args)});")

    return rs

@with_native_function
def gen_structured(g: NativeFunctionsGroup) -> List[str]:
    # only out has dispatch
    meta_name = meta.name(g)
    rs = []
    seen: Set[Any] = set()
    out_args = structured.impl_arguments(g)
    for k, n in g.out.dispatch.items():
        if n in seen:
            continue
        if not is_structured_dispatch_key(k):
            continue
        seen.add(n)
        rs.append(f"""\
struct TORCH_API structured_{n} : public at::meta::{meta_name} {{
void impl({', '.join(a.decl() for a in out_args)});
}};
""")

    seen = set()
    for f in g.functions():
        returns_type = native.returns_type(f.func.returns)
        args = native.arguments(f.func)
        for k, n in f.dispatch.items():
            if n in seen:
                continue
            if is_structured_dispatch_key(k):
                continue
            seen.add(n)
            args_str = ', '.join(a.decl() for a in args)
            rs.append(f"TORCH_API {returns_type} {n}({args_str});")

    return rs

# Generates NativeFunctions.h, a list of forward declarations of all
# actual kernel definitions we keep in aten/src/ATen/native/
@with_native_function
def compute_native_function_declaration(g: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
    if isinstance(g, NativeFunctionsGroup):
        if g.structured:
            return gen_structured(g)
        else:
            return list(concatMap(gen_unstructured, g.functions()))
    else:
        return gen_unstructured(g)
