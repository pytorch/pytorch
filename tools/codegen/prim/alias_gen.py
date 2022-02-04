from typing import List, Dict, Tuple, Optional
import os
from string import Template

from tools.codegen.prim.aliases import alias_infos

from tools.codegen.model import \
    (NativeFunction, FunctionSchema, OperatorName, BaseOperatorName, Variant,
     DeviceCheckType, BaseType, BaseTy, Argument, ListType, OptionalType, Arguments,
     DispatchKey, BackendMetadata, Return)

# This file defines 'native_functions_callback', which generates
#   PyTorch aliases at run time.


# TODO: extend this map as needed
# Note: native functions code generation does not appear to support non-const
#   optionals
_optional_elem_type_to_sig_cpp_map = {
    BaseTy.Tensor: 'const c10::optional<Tensor>&',
    BaseTy.Scalar: 'const c10::optional<Scalar>&'
}

def optional_type_to_sig_cpp(arg: Argument) -> str:
    assert isinstance(arg.type, OptionalType), "Trying to generate an OptionalType's C++ for a non-OptionalType arg!"
    ot: OptionalType = arg.type
    assert isinstance(ot.elem, BaseType), "Failed to extract BaseType elem from OptionalType!"
    return _optional_elem_type_to_sig_cpp_map[ot.elem.name]


# TODO: extend this map as needed
_list_elem_type_to_sig_cpp_map = {
    BaseTy.Tensor: 'TensorList',
    BaseTy.int: 'IntArrayRef'
}

def list_type_to_sig_cpp(arg: Argument) -> str:
    assert isinstance(arg.type, ListType), "Trying to generate a ListType's C++ for a non-ListType arg!"
    lt: ListType = arg.type
    assert isinstance(lt.elem, BaseType), "Failed to extract BaseType elem from ListType!"
    return _list_elem_type_to_sig_cpp_map[lt.elem.name]

# TODO: expand this map as needed
_base_type_to_sig_cpp_map = {
    BaseTy.Tensor: 'Tensor&',  # Note: may be const Tensor& (handled below)
    BaseTy.Dimname: 'Dimname',
    BaseTy.int: 'int64_t',
    BaseTy.Scalar: 'const Scalar&'
}

# Generates the C++ representing a type
def arg_to_sig_cpp(arg: Argument) -> str:
    cpp: str

    # Computes the C++ type
    if isinstance(arg.type, ListType):
        cpp = list_type_to_sig_cpp(arg)
    elif isinstance(arg.type, OptionalType):
        cpp = optional_type_to_sig_cpp(arg)
    else:
        assert isinstance(arg.type, BaseType), "Cannot generate signature C++ for a non-Base, non-Optional, non-List type!"
        cpp = _base_type_to_sig_cpp_map[arg.type.name]

        # Unmodified tensors are passed as const references
        if arg.type.name is BaseTy.Tensor:
            is_modified = arg.annotation and arg.annotation.is_write
            if not is_modified:
                cpp = "const " + cpp

    # Adds the name
    cpp += " " + arg.name
    return cpp

# Returns True when Arguments specifies one or more out arguments, False otherwise
def has_out(args: Arguments) -> bool:
    return len(args.out) > 0

# Returns True when Arguments specifies a self arg, False otherwise
def has_self(args: Arguments) -> bool:
    return args.self_arg is not None

# Returns True when FunctionSchema specifies an inplace function, False otherwise
def is_inplace(fs: FunctionSchema) -> bool:
    return fs.name.name.inplace

# Constructs the C++ for a method call
def generate_method_call_cpp(fn_cpp: str, args: Arguments) -> str:
    assert has_self(args), "Request to generate method call C++ with Arguments containing no self arg!"
    assert not has_out(args), "Request to generate method call C++ with Arguments containing out arg!"

    call_args = list(args.pre_self_positional)
    call_args.extend(args.post_self_positional)
    call_args.extend(args.pre_tensor_options_kwarg_only)
    call_args.extend(args.post_tensor_options_kwarg_only)

    arg_cpp = ", ".join(arg.name for arg in call_args)
    return "self." + fn_cpp + "(" + arg_cpp + ")"

# Generates the C++ for a function call
# Note: assumes the function is in the at namespace
def generate_function_call_cpp(fn_cpp: str, args: Arguments) -> str:
    call_args = list(args.out)
    call_args.extend(args.pre_self_positional)
    if has_self(args):
        assert args.self_arg is not None  # This is for mypy, which can't figure out the has_self() call
        call_args.append(args.self_arg.argument)
    call_args.extend(args.post_self_positional)
    call_args.extend(args.pre_tensor_options_kwarg_only)
    call_args.extend(args.post_tensor_options_kwarg_only)

    arg_cpp = ", ".join(arg.name for arg in call_args)
    return "at::" + fn_cpp + "(" + arg_cpp + ")"

# Generates the C++ for a function signature
def generate_signature_cpp(return_cpp: str, fn_cpp: str, args: Arguments) -> str:
    signature_args = list(args.pre_self_positional)
    if has_self(args):
        assert args.self_arg is not None  # This is for mypy, which can't figure out the has_self() call
        signature_args.append(args.self_arg.argument)
    signature_args.extend(args.post_self_positional)
    signature_args.extend(args.pre_tensor_options_kwarg_only)
    signature_args.extend(args.post_tensor_options_kwarg_only)
    signature_args.extend(args.out)

    arg_cpp = ", ".join(arg_to_sig_cpp(arg) for arg in signature_args)
    return return_cpp + " " + fn_cpp + "(" + arg_cpp + ")"

# Generates the C++ for the name of a called function
# Call name is ...
    #   <base name>_<out> when there's an out arg
    #   <base name>_ when the operation is inplace
    #   <base name> otherwise
def generate_function_call_name_cpp(fs: FunctionSchema) -> str:
    assert not (is_inplace(fs) and has_out(fs.arguments)), "Aliases for inplace operators with out kwargs are not supported."

    fn_name = fs.name.name.base
    if is_inplace(fs):
        fn_name += "_"
    elif has_out(fs.arguments):
        fn_name += "_" + "out"

    return fn_name

# Generates the C++ for name of a function as it appears in a C++ definition
# Note: this is distinct from the name used when calling the function
# Signature name is ...
    #   <base name>__<overload name> when <overload name>
    #     is nonempty and the operation is inplace
    #   <base_name>_<overload name> when <overload name> is
    #     is nonempty and the operation is not inplace
    #   <base name>_ when <overload name> is empty and
    #     the operation is inplace
    #   <base name> otherwise
def generate_signature_name_cpp(fs: FunctionSchema) -> str:
    on = fs.name
    has_overload_name = True if on.overload_name else False

    fn_name = on.name.base
    if is_inplace(fs):
        fn_name += "_"
    if has_overload_name:
        fn_name += "_" + on.overload_name

    return fn_name

# Generates the C++ for the return type of a function as it appears in a C++ definition
# Note: Return values that are annotated as written to are references
def generate_return_cpp(rets: Tuple[Return, ...]) -> str:
    assert len(rets) == 1, "Aliases for ops that return multiple objects are not supported."
    return_type = rets[0].type
    assert isinstance(return_type, BaseType), "Aliass for ops with non-BaseType returns are not supported."
    # TODO: this can be relaxed if we find aliases for ops with non-tensor returns
    assert return_type.name is BaseTy.Tensor, "Aliases for ops with non-tensor returns are not supported."

    # Modified tensors are returned by reference, unmodified tensors are returned by value
    is_modified = rets[0].annotation is not None and rets[0].annotation.is_write
    cpp: str
    if is_modified:
        cpp = "Tensor&"
    else:
        cpp = "Tensor"

    return cpp


# Creates native function and C++ source for primTorch defined aliases.
#   See the [primTorch Aliases] note for more details.
# This callback that appends alias information to native functions internal datastructures after they're parsed
#   and, if the "src_path" kwarg is provided, writes the C++ for them.
# Note that the callback structure is because native_functions.yaml is parsed multiple times when
#   PyTorch is built.
# Additional kwargs:
#   - src_path, the ATen source path. If provided this will write the C++ for the aliases to
#       ATen/native/aliases.cpp, using the template ATen/native/aliases.cpp.in.
#   - prim_cpp_dry_run, a boolean for whether to perform a "dry run". Useful when debugging.
#       A dry run will print the generated C++ instead of writing it.
#       Usage: python -m tools.codegen.gen --prim_cpp_dry_run
def native_functions_callback(rs: List[NativeFunction],
                              bs: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]],
                              *,
                              src_path: Optional[str] = None,
                              is_dry_run: bool = False) -> \
        List[Tuple[NativeFunction, Dict[DispatchKey, Dict[OperatorName, BackendMetadata]]]]:
    # Return values
    # pairs is used to update the internal native function data structures
    # cpp is the C++ defining the aliases to write to ATen/native/Aliases.cpp if src_path is specified
    pairs: List[Tuple[NativeFunction, Dict[DispatchKey, Dict[OperatorName, BackendMetadata]]]] = []
    cpp = ""

    # TODO: construct a map of alias_infos so this isn't nf * alias_infos?
    for nf in rs:
        for alias in alias_infos:
            if alias.alias_for == nf.func.name.name.base:

                # Determines which variants (if any) to this alias overload has.
                # Note [Function variants]: An alias overload has a function variant when
                #   the aliased overload has a function variant.
                # Note [Method variants]: By default, alias overloads have method variants when the
                #   aliased overload has a method variant and the alias is not in a subnamespace
                #   of torch (like torch.linalg).
                #   If the has_method attribute of an AliasInfo is true then every alias overload
                #   except out variants have method variants. (out variants cannot be methods.)
                #   If the has_method attribute of an AliasInfo is false then no alias overload has
                #   a method variant, and if the aliased overload only had a method variant then
                #   no alias is generated.
                # Note [Dunder Methods]: All aliases set dunder_method to False.

                original_is_out_variant = len(nf.func.arguments.out) > 0
                ao_has_method_variant = Variant.method in nf.variants
                ao_is_method_only = ao_has_method_variant and (len(nf.variants) == 1)
                has_method_variant = alias.has_method
                if has_method_variant is None:
                    has_method_variant = ao_has_method_variant and (alias.namespace is None)

                variants = {Variant.function, Variant.method}
                if Variant.function not in nf.variants:
                    variants.remove(Variant.function)
                if not has_method_variant or original_is_out_variant:
                    variants.remove(Variant.method)

                # TODO: should this silently fail?
                assert len(variants) > 0, "Aliased overload is method-only but alias overload would not have a method variant!"

                # Creates the alias overload's FunctionSchema by mimicking the alias overload's
                #   FunctionSchema, except for:
                #     - the BaseOperatorName's "base" attribute is replaced by the alias's name and its
                #         dunder_method attribute is always set to False
                ao_bon = nf.func.name.name
                bon = BaseOperatorName(base=alias.name, inplace=ao_bon.inplace, dunder_method=False)
                on = OperatorName(name=bon, overload_name=nf.func.name.overload_name)
                fs = FunctionSchema(name=on, arguments=nf.func.arguments, returns=nf.func.returns)

                # Creates the alias overload's NativeFunction by mimicking the aliased overload's
                #   NativeFunction, except for:
                #     - device_guard is always False (because the alias overload calls the aliased overload)
                #     - device_check is always DeviceCheckType.NoCheck
                #     - python_module is set according to the alias's namespace
                #     - variants may be modified per the [Function variants] and [Method variants] notes
                #     - manual_kernel_registration is always False (because this can't generate it!)
                #     - manual_cpp_binding is always False (because this can't generate it!)
                #     - structured is always False, and structured_delegate, structured_inherits
                #         and precomputed are always None (because the alias delegates to the aliased)
                #     - has_composite_implicit_autograd_kernel is always True (and
                #         has_composite_explicit_autograd_kernel is always False)
                alias_nf = NativeFunction(
                    func=fs,
                    use_const_ref_for_mutable_tensors=nf.use_const_ref_for_mutable_tensors,
                    device_guard=False,
                    device_check=DeviceCheckType.NoCheck,
                    python_module=alias.namespace,
                    category_override=nf.category_override,
                    variants=variants,
                    manual_kernel_registration=False,
                    manual_cpp_binding=False,
                    loc=nf.loc,
                    structured=False,
                    structured_delegate=None,
                    structured_inherits=None,
                    precomputed=None,
                    cpp_no_default_args=nf.cpp_no_default_args,
                    is_abstract=nf.is_abstract,
                    has_composite_implicit_autograd_kernel=True,
                    has_composite_explicit_autograd_kernel=False,
                    tag=nf.tag)

                # Creates dispatch and backend metadata
                kernel = str(on).replace('.', '_')
                bm = BackendMetadata(kernel=kernel, structured=False)
                alias_bs = {DispatchKey.CompositeImplicitAutograd: {on: bm}}

                # Updates native function data structures
                pairs.append((alias_nf, alias_bs))
                # pairs.append((alias_nf, {}))

                # Generates C++ src for the operator if given a source path
                if src_path is None:
                    continue

                # Validates function arguments
                args = alias_nf.func.arguments
                assert args.tensor_options is None, "Aliases for operations with tensor options are not supported."
                assert len(args.out) <= 1, "Aliases for operations with multiple out tensors are not supported."
                is_method_call = nf.variants == {Variant.method}

                # Creates function signature
                return_cpp = generate_return_cpp(alias_nf.func.returns)
                alias_name_cpp = generate_signature_name_cpp(alias_nf.func)
                alias_sig_cpp = generate_signature_cpp(return_cpp, alias_name_cpp, args)

                # Creates alias call
                # NOTES:
                #   - out arguments are passed first when calling from C++
                #   - a method call is generated if the called operator has only a method variant
                call_name_cpp = generate_function_call_name_cpp(nf.func)
                call_cpp = ""
                if is_method_call:
                    call_cpp = generate_method_call_cpp(call_name_cpp, args)
                else:
                    # Function call case
                    call_cpp = generate_function_call_cpp(call_name_cpp, args)

                # Composes C++
                cpp += alias_sig_cpp + "{\n"
                cpp += "\treturn " + call_cpp + ";\n"
                cpp += "}\n"

    # Dry runs are for debugging C++ code generation, and do not write the results
    if is_dry_run:
        print(cpp)
    elif src_path is not None:
        alias_template_path = os.path.join(src_path, 'native/Aliases.cpp.in')
        alias_source_path = os.path.join(src_path, 'native/Aliases.cpp')
        with open(alias_template_path, 'r') as f:
            template = Template(f.read())
        cpp_src = template.substitute(alias_definitions=cpp)
        # TODO: write once?
        with open(alias_source_path, 'w') as f:
            f.write(cpp_src)

    return pairs
