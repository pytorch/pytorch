import argparse
import os
import pathlib
from dataclasses import dataclass
from io import TextIOWrapper
from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

import yaml

# Parse native_functions.yaml into a sequence of NativeFunctions and Backend Indices.
from torchgen import dest
from torchgen.api import cpp as aten_cpp
from torchgen.api.translate import translate
from torchgen.api.types import CppSignatureGroup
from torchgen.context import method_with_native_function, with_native_function_and_index
from torchgen.executorch.api import cpp as et_cpp, unboxing
from torchgen.executorch.api.types import CppSignature, NativeSignature
from torchgen.executorch.api.unboxing import Unboxing
from torchgen.executorch.api.custom_ops import gen_custom_ops
from torchgen.gen import (
    get_custom_build_selector,
    get_grouped_native_functions,
    get_native_function_declarations,
    LineLoader,
    parse_native_yaml,
    ParsedYaml,
)
from torchgen.model import (
    BackendIndex,
    DispatchKey,
    is_generic_dispatch_key,
    Location,
    NativeFunction,
    NativeFunctionsGroup,
    Variant,
)
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import context, FileManager, make_file_manager, mapMaybe, Target


def static_dispatch(
    sig: CppSignature,
    f: NativeFunction,
    backend_indices: List[BackendIndex],
) -> str:
    """
    For a given `NativeFunction`, find out the corresponding native function and dispatch to it. If zero or more than one
    native function exists, error out. A simplified version of register_dispatch_key.py
    Arguments:
        sig: A CppSignature for this native function we want to use.
        f: NativeFunction to generate static dispatch.
        backend_indices: All available backends.
    Return:
        C++ code to call backend-specific functions, e.g., "return at::native::add(self, other, scale);"
    """
    if len(backend_indices) == 0 or f.manual_kernel_registration:
        return ""

    backends = [b for b in backend_indices if b.has_kernel(f)]
    static_block = None
    if len(backends) == 1:
        backend_metadata = backends[0].get_kernel(f)
        if backend_metadata:
            native_sig = NativeSignature(func=f.func)
            exprs = translate(sig.arguments(), native_sig.arguments())
            comma = ", "
            static_block = f"return {backend_metadata.cpp_namespace}::{backend_metadata.kernel}({comma.join(e.expr for e in exprs)});"
    else:
        static_block = f'ET_ASSERT_UNREACHABLE_MSG("The number of native function(s) binding to {f.func.name} is {len(backends)}.");'
    return f"""
// {f.namespace}::{f.func}
TORCH_API inline {sig.decl()} {{
    {static_block}
}}
"""


# Generates Functions.h, which provides the functional public C++ API,
# and the scaffolding to call into the dispatcher from these functions.
@dataclass(frozen=True)
class ComputeFunction:
    static_dispatch_backend_indices: List[BackendIndex]

    selector: SelectiveBuilder

    use_aten_lib: bool

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if not self.selector.is_root_operator(f"{f.namespace}::{f.func.name}"):
            return None
        if Variant.function not in f.variants:
            return None

        if self.use_aten_lib:
            comma = ", "
            sig = CppSignatureGroup.from_native_function(
                f, method=False, fallback_binding=f.manual_cpp_binding
            ).most_faithful_signature()
            return f"""
// {f.namespace}::{f.func}
TORCH_API inline {sig.decl()} {{
    return at::{sig.name()}({comma.join(e.name for e in sig.arguments())});
}}
            """

        else:
            sig = CppSignature.from_native_function(f)
            return static_dispatch(
                sig,
                f,
                backend_indices=self.static_dispatch_backend_indices,
            )


# Generates UnboxingFunctions.h & UnboxingFunctions.cpp.
@dataclass(frozen=True)
class ComputeUnboxingFunctions:
    # pyre-fixme[31]: Expression `Target.DECLARATION` is not a literal value.
    # pyre-fixme[31]: Expression `Target.DEFINITION` is not a literal value.
    # pyre-fixme[11]: Annotation `Target` is not defined as a type.
    target: Union[Literal[Target.DECLARATION], Literal[Target.DEFINITION]]

    selector: SelectiveBuilder

    use_aten_lib: bool

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:
        if not self.selector.is_root_operator(f"{f.namespace}::{f.func.name}"):
            return ""

        # pyre-fixme[16]: `Enum` has no attribute `DECLARATION`.
        if self.target is Target.DECLARATION:
            # Note [The ATen Codegen Unboxing API]
            # Similar to the ATen Operators API, ATen Codegen Unboxing API lives in the at::unboxing namespace, and
            # will be used by codegen unboxing wrappers (CodegenUnboxingWrappers.cpp).
            # The Wrappers will be registered into torch::jit::OperatorRegistry using RegisterOperators API.
            #
            # Important characteristics about the Codegen Unboxing API:
            # (1) It follows the OperatorRegistry API.
            #     This is kind of necessary to avoid overhead.
            #     For example: if it followed the C++ API, then all of the faithful C++ factory functions
            #     would need to wrap their arguments into TensorOptions only to unwrap them again.
            # (2) Under the hood it calls C++ API.
            return f"""
// {f.namespace}::{f.func}
void {f.func.name.unambiguous_name()}(EValue** stack);
"""
        else:
            if self.use_aten_lib:
                sig = CppSignatureGroup.from_native_function(
                    f, method=False, fallback_binding=f.manual_cpp_binding
                ).most_faithful_signature()
                argument_type_gen = aten_cpp.argumenttype_type
                return_type_gen = aten_cpp.returns_type
            else:
                sig = CppSignature.from_native_function(f)
                argument_type_gen = et_cpp.argumenttype_type
                return_type_gen = et_cpp.returns_type
            # parse arguments into C++ code
            binding_list, code_list = Unboxing(
                argument_type_gen=argument_type_gen
            ).convert_arguments(sig.arguments())

            # for each C++ argument, generate the conversion code
            code_connector = "\n\t"
            arg_connector = ", "

            args_str = f"{arg_connector.join(e.name for e in binding_list)}"

            if len(f.func.returns) == 0:
                if len(f.func.arguments.out) == 0:
                    raise Exception(
                        f"Can't handle native function {f.func} with no returns and no out yet."
                    )
                out = f.func.arguments.out[0]
                return_assignment = f"""stack[{len(binding_list)}] = &{out.name};"""
                ret_prefix = ""
            else:
                if len(f.func.arguments.out) == 0:
                    return_assignment = (
                        f"""*stack[{len(binding_list)}] = EValue(result_);"""
                    )
                    ret_prefix = (
                        return_type_gen(f.func.returns).cpp_type() + " result_ = "
                    )
                else:
                    return_assignment = ""
                    ret_prefix = ""
            return f"""
// {f.namespace}::{f.func}
void {f.func.name.unambiguous_name()}(EValue** stack) {{
    {code_connector.join(code_list)}

    EXECUTORCH_SCOPE_PROF("native_call_{f.func.name}");
    {ret_prefix}torch::executor::{sig.name()}({args_str});

    {return_assignment}
}}
"""


# Generates RegisterCodegenUnboxedKernels.cpp.
@dataclass(frozen=True)
class ComputeCodegenUnboxedKernels:
    selector: SelectiveBuilder

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:
        if not self.selector.is_root_operator(f"{f.namespace}::{f.func.name}"):
            return ""
        # We unconditionally generate function wrappers,
        return f"""
Operator(
    "{f.namespace}::{f.func.name}",
    [](EValue** stack) {{
        unboxing::{unboxing.name(f)}(stack);
    }}
),
"""


def gen_unboxing(
    *,
    native_functions: Sequence[NativeFunction],
    cpu_fm: FileManager,
    selector: SelectiveBuilder,
    use_aten_lib: bool,
) -> None:
    def key_func(fn: Union[NativeFunction, NativeFunctionsGroup]) -> str:
        return fn.root_name

    cpu_fm.write_sharded(
        "UnboxingFunctions.cpp",
        native_functions,
        key_fn=key_func,
        env_callable=lambda fn: {
            # pyre-fixme[19]: Expected 0 positional arguments.
            # pyre-fixme[16]: `Enum` has no attribute `DEFINITION`.
            "definitions": [ComputeUnboxingFunctions(Target.DEFINITION, selector, use_aten_lib)(fn)],
        },
        num_shards=1,
        sharded_keys={"definitions"},
    )
    cpu_fm.write_sharded(
        "RegisterCodegenUnboxedKernels.cpp",
        native_functions,
        key_fn=key_func,
        env_callable=lambda fn: {
            "unboxed_ops": [ComputeCodegenUnboxedKernels(selector)(fn)],
        },
        num_shards=1,
        sharded_keys={"unboxed_ops"},
    )


@with_native_function_and_index
def compute_native_function_declaration(
    g: Union[NativeFunctionsGroup, NativeFunction], backend_index: BackendIndex
) -> List[str]:
    assert isinstance(g, NativeFunction)
    sig = NativeSignature(func=g.func)
    metadata = backend_index.get_kernel(g)
    if metadata is None:
        return []
    prefix = "static" if backend_index.external else "TORCH_API"
    return [f"{prefix} {sig.decl(name=metadata.kernel)};"]


def gen_aggregated_headers(
    *,
    native_functions: Sequence[NativeFunction],
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    static_dispatch_idx: List[BackendIndex],
    selector: SelectiveBuilder,
    backend_indices: Dict[DispatchKey, BackendIndex],
    cpu_fm: FileManager,
    functions_keys: Set[DispatchKey],
    dispatch_keys: Sequence[DispatchKey],
    rocm: bool,
    use_aten_lib: bool,
) -> None:

    cpu_fm.write(
        "Functions.h",
        lambda: {
            "static_dispatch_extra_headers": "#include <ATen/Functions.h>"
            if use_aten_lib
            else '#include "NativeFunctions.h"',
            "Functions_declarations": list(
                mapMaybe(
                    ComputeFunction(
                        static_dispatch_backend_indices=static_dispatch_idx,
                        selector=selector,
                        use_aten_lib=use_aten_lib,
                    ),
                    native_functions,
                )
            ),
        },
    )

    cpu_fm.write(
        "NativeFunctions.h",
        lambda: {
            "nativeFunctions_declarations": get_native_function_declarations(
                grouped_native_functions=native_functions,
                backend_indices=backend_indices,
                native_function_decl_gen=dest.compute_native_function_declaration
                if use_aten_lib
                else compute_native_function_declaration,
            ),
        },
    )
    cpu_fm.write(
        "UnboxingFunctions.h",
        lambda: {
            "declarations": list(
                mapMaybe(
                    # pyre-fixme[19]: Expected 1 positional arguments.
                    # pyre-fixme[16]: `Enum` has no attribute `DECLARATION`.
                    ComputeUnboxingFunctions(
                        Target.DECLARATION, selector, use_aten_lib
                    ),
                    native_functions,
                )
            ),
        },
    )


def gen_headers(
    *,
    native_functions: Sequence[NativeFunction],
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    static_dispatch_idx: List[BackendIndex],
    selector: SelectiveBuilder,
    backend_indices: Dict[DispatchKey, BackendIndex],
    core_fm: FileManager,
    cpu_fm: FileManager,
    ops_fm: FileManager,
    dispatch_keys: Sequence[DispatchKey],
    functions_keys: Set[DispatchKey],
    rocm: bool,
    per_operator_headers: bool,
    use_aten_lib: bool,
) -> None:
    if per_operator_headers:
        raise RuntimeError("Not supported")
    else:
        gen_aggregated_headers(
            native_functions=native_functions,
            grouped_native_functions=grouped_native_functions,
            static_dispatch_idx=static_dispatch_idx,
            selector=selector,
            backend_indices=backend_indices,
            cpu_fm=cpu_fm,
            dispatch_keys=dispatch_keys,
            functions_keys=functions_keys,
            rocm=rocm,
            use_aten_lib=use_aten_lib,
        )

    def gen_aten_interned_strings() -> Dict[str, str]:
        attrs = set()  # All function argument names
        names = set()  # All ATen function names
        for func in native_functions:
            names.add(str(func.func.name.name))
            # Some operators don't have a functional variant but we still create a
            # symbol without the underscore
            names.add(func.func.name.name.base)

            for arg in func.func.schema_order_arguments():
                attrs.add(arg.name)

        # These are keywords in C++, so aren't valid symbol names
        # https://en.cppreference.com/w/cpp/language/operator_alternative
        names -= {
            "and",
            "and_eq",
            "bitand",
            "bitor",
            "compl",
            "not",
            "not_eq",
            "or",
            "or_eq",
            "xor",
            "xor_eq",
        }

        return {
            "aten_symbols": " \\\n".join(
                [f"_(aten, {name})" for name in sorted(names)]
            ),
            "attr_symbols": " \\\n".join(
                [f"_(attr, {name})" for name in sorted(attrs)]
            ),
        }

    core_fm.write("aten_interned_strings.h", gen_aten_interned_strings)


def translate_native_yaml(
    tags_yaml_path: str,
    aten_yaml_path: str,
    native_yaml_path: str,
    use_aten_lib: bool,
    out_file: TextIOWrapper,
) -> None:
    """Translates Executorch DSL dialect to use the same syntax as
    native_functions.yaml. The major difference is that Executorch DSL dialect
    supports "op" key, where it refers to the operator name in native_functions.yaml.

    For example, a functions.yaml may have the following entry:

    - op: add.out
      ...

    It needs to be translated to the following:

    - func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
      ...

    We go in aten_yaml_path and find the operator schema for "add.out" and add it
    to the original functions.yaml. We also add required field "variants", where for
    Executorch it will always be "function".

    For ATen mode we don't have to do the translation because native_yaml_path is
    the same as native_functions.yaml.

    Args:
        tags_yaml_path: Path to a tags.yaml file to satisfy codegen parsing.
            It is not optional.
        aten_yaml_path: Path to ATen operator yaml file native_functions.yaml.
        native_yaml_path: Path to a functions.yaml file to parse.
            If the path does not exist in the filesystem, it is treated as an
            empty file. If `custom_ops_yaml_path` exists, the contents of that
            file are appended to the yaml input to be parsed.
        use_aten_lib: We use this flag to determine if we want to generate native
            functions. In ATen mode we should generate out= variants.
        out_file: The IO object that we are writing into.
    Returns:
        None
    """
    if use_aten_lib:
        with open(native_yaml_path, "r") as native_yaml:
            out_file.writelines(native_yaml.readlines())
        return
    aten_parsed_yaml = parse_native_yaml(
        aten_yaml_path,
        tags_yaml_path,
        None,
        skip_native_fns_gen=False,
    )
    aten_native_functions = aten_parsed_yaml.native_functions
    schema_dict = {
        f"{f.namespace}::{f.func.name}": str(f.func) for f in aten_native_functions
    }

    with open(native_yaml_path, "r") as native_yaml:
        native_es = yaml.load(native_yaml, Loader=LineLoader)
        for e in native_es:
            assert isinstance(e.get("__line__"), int), e
            loc = Location(native_yaml_path, e.pop("__line__"))
            with context(lambda: f"in {loc}:\n  "):
                if "variants" not in e:
                    e["variants"] = "function"
                if "func" in e:
                    continue
                assert isinstance(e.get("op"), str), e
                opname = e.pop("op")
                if "::" not in opname:
                    opname = "aten::" + opname
                assert opname in schema_dict
                e["func"] = schema_dict.get(opname)
        yaml.dump(native_es, out_file, width=1000)


def parse_yaml_files(
    tags_yaml_path: str,
    aten_yaml_path: str,
    native_yaml_path: Optional[str],
    custom_ops_yaml_path: Optional[str],
    use_aten_lib: bool,
) -> Tuple[ParsedYaml, Optional[ParsedYaml]]:
    """Parses functions.yaml and custom_ops.yaml files.

    Args:
        tags_yaml_path: Path to a tags.yaml file to satisfy codegen parsing.
            It is not optional.
        aten_yaml_path: Path to ATen operator yaml file native_functions.yaml.
        native_yaml_path: Path to a functions.yaml file to parse.
            If the path does not exist in the filesystem, it is treated as an
            empty file. If `custom_ops_yaml_path` exists, the contents of that
            file are appended to the yaml input to be parsed.
        custom_ops_yaml_path: Path to a custom_ops.yaml file to parse. If
            the path does not exist in the filesystem, it is ignored.
        use_aten_lib: We use this flag to determine if we want to generate native
            functions. In ATen mode we should generate out= variants.
    Returns:
        A tuple with two elements:
        [0]: The parsed results of concatenating the contents of
             `native_yaml_path` and `custom_ops_yaml_path`.
        [1]: The parsed results of the contents of `custom_ops_yaml_path`, if
             present. If not present, None.
    """
    import tempfile
    gen_native_fns = use_aten_lib and native_yaml_path
    with tempfile.TemporaryDirectory() as tmpdirname:
        # If native_yaml_path doesn't exist, point to an empty file.
        if not native_yaml_path or not os.path.exists(native_yaml_path):
            native_yaml_path = os.path.join(tmpdirname, "functions.yaml")
            with open(native_yaml_path, "w"):
                pass

        # If custom_ops_yaml_path exists, combine both files.
        if custom_ops_yaml_path and os.path.exists(custom_ops_yaml_path):
            combined_yaml_path = os.path.join(tmpdirname, "combined.yaml")
            with open(combined_yaml_path, "w") as tmp:
                with open(native_yaml_path, "r") as native:
                    for line in native:
                        tmp.write(line)
                with open(custom_ops_yaml_path, "r") as custom:
                    for line in custom:
                        tmp.write(line)
            custom_ops_parsed_yaml = parse_native_yaml(
                custom_ops_yaml_path, tags_yaml_path, None, skip_native_fns_gen=True
            )
        else:
            # No custom_ops; just parse native_yaml_path.
            custom_ops_parsed_yaml = None
            combined_yaml_path = native_yaml_path
        translated_yaml_path = os.path.join(tmpdirname, "translated.yaml")
        with open(translated_yaml_path, "w") as translated:
            translate_native_yaml(
                tags_yaml_path,
                aten_yaml_path,
                combined_yaml_path,
                use_aten_lib,
                translated,
            )
        parsed_yaml = parse_native_yaml(
            translated_yaml_path,
            tags_yaml_path,
            None,
            skip_native_fns_gen=(not gen_native_fns),
        )

    return parsed_yaml, custom_ops_parsed_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate operator source files")
    # Although we don't refer to --source-path directly, make_file_manager()
    # expects it to point to a directory that contains a templates/ subdirectory
    # containing the file templates.
    parser.add_argument(
        "-s",
        "--source-path",
        help="path to source directory for kernel templates",
    )
    parser.add_argument(
        "--functions_yaml_path",
        help="path to the functions.yaml file to use. Optional, but at least "
        "one of --functions_yaml_path and --custom_ops_yaml_path must be "
        "specified.",
    )
    parser.add_argument(
        "--custom_ops_yaml_path",
        help="path to the custom_ops.yaml file to use. Optional, but at least "
        "one of --functions_yaml_path and --custom_ops_yaml_path must be "
        "specified.",
    )
    parser.add_argument(
        "--aten_yaml_path",
        help="path to native_functions.yaml file.",
    )
    # Note that make_file_manager() also looks at --install-dir.
    parser.add_argument(
        "-d", "--install_dir", help="output directory", default="build/generated"
    )
    parser.add_argument(
        "-o",
        "--output-dependencies",
        help="output a list of dependencies into the given file and exit",
    )
    # Although we don't refer to --dry-run directly, make_file_manager() looks
    # for it.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run without writing any files (still updates outputs)",
    )
    parser.add_argument(
        "--static_dispatch_backend",
        nargs="*",
        help="generate static dispatch code for the specific backend (if set)",
    )
    parser.add_argument(
        "--backend_whitelist",
        nargs="*",
        help="filter dispatch backend by the whitelist (if set), "
        "e.g.: CPU CUDA QuantizedCPU ...",
    )
    parser.add_argument(
        "--op_registration_whitelist",
        nargs="*",
        help="filter op registrations by the whitelist (if set); "
        "each item is `namespace`::`operator name` without overload name; "
        "e.g.: aten::empty aten::conv2d ...",
    )
    parser.add_argument(
        "--op_selection_yaml_path",
        help="Provide a path to the operator selection (for custom build) YAML "
        "that contains the information about the set of selected operators "
        "and their categories (training, ...). Each operator is either a "
        "full operator name with overload or just a bare operator name. "
        "The operator names also contain the namespace prefix (e.g. aten::)",
    )
    parser.add_argument(
        "--skip_dispatcher_op_registration",
        action="store_true",
        help="Avoid registering operators into the dispatcher.",
    )
    parser.add_argument(
        "--tags-path",
        help="Path to tags.yaml. Required by yaml parsing in codegen system.",
    )
    parser.add_argument(
        "--rocm",
        action="store_true",
        help="reinterpret CUDA as ROCm/HIP and adjust filepaths accordingly",
    )
    parser.add_argument(
        "--per-operator-headers",
        action="store_true",
        help="generate separate headers per operator in ATen/ops",
    )
    parser.add_argument(
        "--use_aten_lib",
        action="store_true",
        help="a boolean flag to indicate whether we use ATen kernels or not, in the future this flag will be per operator",
    )
    parser.add_argument(
        "--force_schema_registration",
        action="store_true",
        help="force it to generate schema-only registrations for all ops, including"
        "those that are not listed on --op_registration_whitelist",
    )
    parser.add_argument(
        "--generate",
        type=str,
        nargs="*",
        choices=["headers", "sources"],
        default=["headers", "sources"],
        help="Generate only a subset of files",
    )
    options = parser.parse_args()
    assert options.tags_path, "tags.yaml is required by codegen yaml parsing."
    parsed_yaml, custom_ops_parsed_yaml = parse_yaml_files(
        aten_yaml_path=options.aten_yaml_path,
        tags_yaml_path=options.tags_path,
        native_yaml_path=options.functions_yaml_path,
        custom_ops_yaml_path=options.custom_ops_yaml_path,
        use_aten_lib=options.use_aten_lib,
    )
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )
    custom_ops_native_functions = (
        custom_ops_parsed_yaml.native_functions if custom_ops_parsed_yaml else None
    )

    ops_install_dir = f"{options.install_dir}/ops"
    pathlib.Path(ops_install_dir).mkdir(parents=True, exist_ok=True)
    core_install_dir = f"{options.install_dir}/core"
    pathlib.Path(core_install_dir).mkdir(parents=True, exist_ok=True)

    core_fm = make_file_manager(options=options, install_dir=core_install_dir)
    cpu_fm = make_file_manager(options=options)
    ops_fm = make_file_manager(options=options, install_dir=ops_install_dir)
    grouped_native_functions = get_grouped_native_functions(native_functions)

    from torchgen.model import dispatch_keys

    if options.backend_whitelist:
        dispatch_keys = [
            k
            for k in dispatch_keys
            if is_generic_dispatch_key(k) or str(k) in options.backend_whitelist
        ]
    functions_keys = {
        DispatchKey.CPU,
        DispatchKey.CUDA,
        DispatchKey.CompositeImplicitAutograd,
        DispatchKey.CompositeImplicitAutogradNestedTensor,
        DispatchKey.CompositeExplicitAutograd,
        DispatchKey.CompositeExplicitAutogradNonFunctional,
        DispatchKey.Meta,
    }

    selector = get_custom_build_selector(
        options.op_registration_whitelist,
        options.op_selection_yaml_path,
    )

    static_dispatch_idx: List[BackendIndex] = []
    if options.static_dispatch_backend:
        static_dispatch_idx = [
            backend_indices[DispatchKey.parse(key)]
            for key in options.static_dispatch_backend
        ]
        static_dispatch_idx = list(
            filter(
                lambda i: i.dispatch_key != DispatchKey.Undefined, static_dispatch_idx
            )
        )

        for key in options.static_dispatch_backend:
            dp_key = DispatchKey.parse(key)
            if dp_key not in functions_keys:
                functions_keys.add(dp_key)

    if "headers" in options.generate:
        gen_headers(
            native_functions=native_functions,
            grouped_native_functions=grouped_native_functions,
            static_dispatch_idx=static_dispatch_idx,
            selector=selector,
            backend_indices=backend_indices,
            core_fm=core_fm,
            cpu_fm=cpu_fm,
            ops_fm=ops_fm,
            dispatch_keys=dispatch_keys,
            functions_keys=functions_keys,
            rocm=options.rocm,
            per_operator_headers=options.per_operator_headers,
            use_aten_lib=options.use_aten_lib,
        )

    if "sources" in options.generate:
        gen_unboxing(
            native_functions=native_functions,
            cpu_fm=cpu_fm,
            selector=selector,
            use_aten_lib=options.use_aten_lib,
        )
        if custom_ops_native_functions:
            gen_custom_ops(
                native_functions=custom_ops_native_functions,
                selector=selector,
                backend_indices=backend_indices,
                cpu_fm=cpu_fm,
                functions_keys=functions_keys,
                dispatch_keys=dispatch_keys,
                rocm=options.rocm,
            )

    if options.output_dependencies:
        depfile_path = pathlib.Path(options.output_dependencies).resolve()
        depfile_name = depfile_path.name
        depfile_stem = depfile_path.stem

        for fm, prefix in [
            (cpu_fm, ""),
            (core_fm, "core_"),
            (ops_fm, "ops_"),
        ]:
            varname = prefix + depfile_stem
            path = depfile_path.parent / (prefix + depfile_name)
            fm.write_outputs(varname, str(path))


if __name__ == "__main__":
    main()
