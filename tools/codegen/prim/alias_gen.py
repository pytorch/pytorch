from typing import List, Dict, Tuple, Optional
from itertools import product, filterfalse

from tools.codegen.prim.aliases import alias_infos

from tools.codegen.model import \
    (NativeFunction, FunctionSchema, OperatorName, BaseOperatorName, Variant,
     DeviceCheckType, Arguments, DispatchKey, BackendMetadata)
from tools.codegen.utils import \
    (FileManager,)
import tools.codegen.api.dispatcher as dispatcher
from tools.codegen.context import native_function_manager

# This file defines 'native_functions_callback', which generates
#   PyTorch aliases defined as part of primTorch at run time.

# Returns True when Arguments specifies one or more out arguments, False otherwise
def has_out(args: Arguments) -> bool:
    return len(args.out) > 0

# Returns True when Arguments specifies a self arg, False otherwise
def has_self(args: Arguments) -> bool:
    return args.self_arg is not None

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

# Creates native function datastructures and C++ source for primTorch defined aliases.
#   See the [primTorch Aliases] note for more details.
# Note that the callback structure is because native_functions.yaml is parsed multiple times when
#   PyTorch is built.
def native_functions_callback(rs: List[NativeFunction],
                              bs: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]],
                              *,
                              fm: Optional[FileManager] = None,
                              debug: bool = False) -> \
        List[Tuple[NativeFunction, Dict[DispatchKey, Dict[OperatorName, BackendMetadata]]]]:
    # pairs is used to update the internal native function data structures
    # cpp is the C++ defining the aliases
    pairs: List[Tuple[NativeFunction, Dict[DispatchKey, Dict[OperatorName, BackendMetadata]]]] = []
    cpp = ""

    # Iterates over matching native function and alias entries
    for nf, alias in filterfalse(lambda x: x[0].func.name.name.base != x[1].alias_for, product(rs, alias_infos)):

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

        # Generates C++ (if necessary)
        if fm is None and not debug:
            continue

        # Generates signature, reusing dispatcher API logic
        # TODO: This is extremely similar to compute_registration_declarations
        #   in tools/codegen/gen.py -- maybe that could be refactored and the two merged?
        signature_cpp: str
        with native_function_manager(alias_nf):
            dispatcher_name = dispatcher.name(alias_nf.func)
            returns_type = dispatcher.returns_type(alias_nf.func.returns).cpp_type_registration_declarations()
            dispatcher_args = dispatcher.arguments(alias_nf.func)
            dispatcher_args_str = ', '.join(a.no_default().decl_registration_declarations() for a in dispatcher_args)
            signature_cpp = f"""{returns_type} {dispatcher_name}({dispatcher_args_str}) """

        # Validates function arguments
        args = alias_nf.func.arguments
        assert args.tensor_options is None, "Aliases for operations with tensor options are not supported."
        assert len(args.out) <= 1, "Aliases for operations with multiple out tensors are not supported."
        is_method_call = nf.variants == {Variant.method}

        # Creates alias call
        # NOTES:
        #   - out arguments are passed first when calling from C++
        #   - a method call is generated if the called operator has only a method variant
        call_name_cpp: str
        with native_function_manager(nf):
            call_name_cpp = dispatcher.name(nf.func)

        if is_method_call:
            call_cpp = generate_method_call_cpp(call_name_cpp, args)
        else:
            # Function call case
            call_cpp = generate_function_call_cpp(call_name_cpp, args)

        # Composes C++
        cpp += signature_cpp + "{\n"
        cpp += "\treturn " + call_cpp + ";\n"
        cpp += "}\n"

    # Debug mode prints the C++
    if debug:
        print(cpp)

    # Writes C++
    if fm is not None:
        env = {'alias_definitions': cpp}
        fm.write_with_template('Aliases.cpp', 'Aliases.cpp.in', lambda: env)

    return pairs
