import re

from dataclasses import dataclass
from typing import List, Dict, Optional, Iterator, Tuple, Set, NoReturn, Sequence, Callable, Union
from enum import Enum, auto
import itertools

from tools.codegen.utils import *

# A little trick from https://github.com/python/mypy/issues/6366
# for getting mypy to do exhaustiveness checking
# TODO: put this somewhere else, maybe
def assert_never(x: NoReturn) -> NoReturn:
    raise AssertionError("Unhandled type: {}".format(type(x).__name__))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           DATA MODEL
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# Some general principles for our data model.
#
# - Stop using C++ data types as the internal data representation
#   format.  Instead, the internal data structures are centered
#   around JIT schema representation.  This avoid a big problem
#   with the old codegen where we read in all the types from
#   native_functions.yaml and then immediately had to retranslate
#   them into C++ types.
#
# - More semantic data representation.  Instead of representing
#   everything as dicts and strings, we define dataclasses for
#   every interesting entity the code generation has to deal with.
#   These dataclasses have strong semantic invariants: for example,
#   we generally require them to roundtrip losslessly into the
#   form they were parsed from.  These structures are immutable
#   and you're expected to populate information once during
#   construction.

# Represent a source location; used for better error reporting
@dataclass(frozen=True)
class Location:
    file: str
    line: int

    def __str__(self) -> str:
        return "{}:{}".format(self.file, self.line)

# Valid values of the 'variants' field in native_functions.yaml
Variant = Enum('Variant', ('function', 'method'))

# NOTE: Keep the list in sync with `DispatchKey` in c10/core/DispatchKey.h
class DispatchKey(Enum):
    Undefined = 0
    CatchAll = Undefined

    CPU = auto()
    CUDA = auto()
    HIP = auto()
    FPGA = auto()
    MSNPU = auto()
    XLA = auto()
    Vulkan = auto()
    Metal = auto()
    XPU = auto()
    MKLDNN = auto()
    OpenGL = auto()
    OpenCL = auto()
    IDEEP = auto()
    QuantizedCPU = auto()
    QuantizedCUDA = auto()
    QuantizedXPU = auto()
    CustomRNGKeyId = auto()
    MkldnnCPU = auto()
    SparseCPU = auto()
    SparseCUDA = auto()
    SparseHIP = auto()
    SparseXPU = auto()
    NestedTensor = auto()
    PrivateUse1 = auto()
    PrivateUse2 = auto()
    PrivateUse3 = auto()
    EndOfBackendKeys = PrivateUse3

    Meta = auto()
    BackendSelect = auto()
    Named = auto()
    AutogradOther = auto()
    AutogradCPU = auto()
    AutogradCUDA = auto()
    AutogradXLA = auto()
    AutogradNestedTensor = auto()
    AutogradXPU = auto()
    AutogradPrivateUse1 = auto()
    AutogradPrivateUse2 = auto()
    AutogradPrivateUse3 = auto()
    Tracer = auto()
    Autocast = auto()
    Batched = auto()
    VmapMode = auto()
    TESTING_ONLY_GenericWrapper = auto()
    TESTING_ONLY_GenericMode = auto()
    NumDispatchKeys = auto()
    Autograd = auto()
    CompositeImplicitAutograd = auto()
    CompositeExplicitAutograd = auto()
    EndOfAliasKeys = CompositeExplicitAutograd

    CPUTensorId = CPU
    CUDATensorId = CUDA
    PrivateUse1_PreAutograd = AutogradPrivateUse1
    PrivateUse2_PreAutograd = AutogradPrivateUse2
    PrivateUse3_PreAutograd = AutogradPrivateUse3

    def __str__(self) -> str:
        return self.name

    def lower(self) -> str:
        return str(self).lower()

    @staticmethod
    def parse(value: str) -> 'DispatchKey':
        for k, v in DispatchKey.__members__.items():
            if k == value:
                return v
        raise AssertionError(f'unknown dispatch key {value}')

STRUCTURED_DISPATCH_KEYS = {DispatchKey.CUDA, DispatchKey.CPU}

# Dispatch keys that "support all backends".  These codegen slightly differently
# then backend specific keys.
def is_generic_dispatch_key(dk: DispatchKey) -> bool:
    return dk in {DispatchKey.CompositeExplicitAutograd, DispatchKey.CompositeImplicitAutograd}

# CUDA specific dispatch keys
def is_cuda_dispatch_key(dk: DispatchKey) -> bool:
    return dk in {
        DispatchKey.CUDA,
        DispatchKey.QuantizedCUDA,
        DispatchKey.SparseCUDA,
        DispatchKey.AutogradCUDA,
        DispatchKey.CUDATensorId,
    }

# Structured kernel generation is only supported for certain key types;
# otherwise use old-style
def is_structured_dispatch_key(dk: DispatchKey) -> bool:
    return dk in STRUCTURED_DISPATCH_KEYS

# The basic input to the code generation is native_functions.yaml.
# The name "native", BTW, comes from the distinction between native
# functions and legacy TH functions.  The legacy TH functions are gone,
# but the "native" descriptor has stuck.
#
# NativeFunction models a single entry in native_functions.yaml.  Its
# fields roughly correspond to what you would see in the YAML itself,
# but after canonicalization and parsing has occurred.
#
# You can see some of the overall design patterns for how we setup
# dataclasses in this class, but we will defer a complete discussion
# of this at FunctionSchema.
@dataclass(frozen=True)
class NativeFunction:
    # The function schema of the operator in question.  This schema
    # has been parsed; see FunctionSchema for more about its structure.
    # (This type is quoted as we are forward referencing a type
    # defined later in the file.  I opted for this ordering of the
    # classes for expository clarity.)
    func: 'FunctionSchema'

    # Whether or not to omit automatic generation of a DeviceGuard
    device_guard: bool

    # What python module to put the function in
    python_module: Optional[str]

    # TODO: figure out what this does
    category_override: Optional[str]

    # If no variants are specified in native_functions.yaml, this is
    # assumed to be {'function'}.
    variants: Set[Variant]

    # Whether or not we should skip generating registrations for
    # this kernel.  This is a bit of a double-edged sword, as manual
    # registrations don't participate in codegen-based selective build!
    manual_kernel_registration: bool

    # Whether or not to skip generating TensorMethod/Functions bindings
    # for this kernel.  Technically, this doesn't actually skip generating
    # the binding; instead, the binding gets generated to __dispatch_{funcname}
    # so you can make use of the normal binding if you need it.
    manual_cpp_binding: bool

    # A mapping of dispatch keys to names of functions implementing
    # them.  In native_functions.yaml, the dispatch entry is optional; in that
    # case, that is equivalent to having written:
    #
    #   dispatch:
    #       CompositeImplicitAutograd: $operator_name
    dispatch: Dict[DispatchKey, str]

    # The location in the YAML file were this native function entry was
    # defined.  This is for conveniently reporting error messages!
    loc: 'Location'

    # Whether or not this out functions is a "structured kernel".  Structured
    # kernels are defined a little differently from normal kernels; in
    # particular, their shape checking logic is defined separately from
    # the kernel.  Only out functions can be structured; other functions
    # delegate to the out function using the structured_delegate keyword.
    # Every structured kernel must have at least an out and a functional
    # variant.
    structured: bool

    # Whether or not this non-out function is a structured kernel, defined
    # in terms of the out kernel referenced by the string here.
    structured_delegate: Optional['OperatorName']

    # Only valid for structured kernels.  Specifies alternative of what
    # to inherit from when defining the meta class for the structured
    # operator.  This will usually be TensorIteratorBase.  This also
    # changes the semantics of set_output to call the parent class.
    structured_inherits: Optional[str]

    # Argument names whose default  should be excluded from the C++ interface.
    # Intended for resolving overload ambiguities between signatures.
    cpp_no_default_args: Set[str]

    # Note [Abstract ATen methods]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # An abstract ATen method is one whose dispatch differs between
    # types.  These are implemented in derived types (with a
    # standard (throwing) definition in Type).  A concrete ATen
    # method is one which has the same dispatch for all types;
    # we just implement it in the base Type.  This is exposed
    # in Declarations.yaml via a field named 'abstract'.
    @property
    def is_abstract(self) -> bool:
        if self.structured_delegate:
            # Structured functions MUST have a dispatch table
            return True
        else:
            return self.dispatch.keys() != {DispatchKey.CompositeImplicitAutograd}

    # NB: The benefit of defining a dataclass is that we automatically get
    # a constructor defined for all the fields we specify.  No need
    # to explicitly write it out.

    @staticmethod
    def from_yaml(ei: Dict[str, object], loc: 'Location') -> 'NativeFunction':
        """
        Parse a NativeFunction from a dictionary as directly parsed
        from native_functions.yaml
        """
        e = ei.copy()

        funcs = e.pop('func')
        assert isinstance(funcs, str), f'not a str: {funcs}'
        func = FunctionSchema.parse(funcs)

        cpp_no_default_args_list = e.pop('cpp_no_default_args', [])
        assert isinstance(cpp_no_default_args_list, list)
        cpp_no_default_args = set(cpp_no_default_args_list)

        variants_s = e.pop('variants', 'function')
        assert isinstance(variants_s, str)
        variants: Set[Variant] = set()
        for v in variants_s.split(', '):
            if v == 'function':
                variants.add(Variant.function)
            elif v == 'method':
                variants.add(Variant.method)
            else:
                raise AssertionError(f'illegal variant {v}')

        manual_kernel_registration = e.pop('manual_kernel_registration', False)
        assert isinstance(manual_kernel_registration, bool), f'not a bool: {manual_kernel_registration}'

        manual_cpp_binding = e.pop('manual_cpp_binding', False)
        assert isinstance(manual_cpp_binding, bool), f'not a bool: {manual_cpp_binding}'

        device_guard = e.pop('device_guard', True)
        assert isinstance(device_guard, bool), f'not a bool: {device_guard}'

        structured = e.pop('structured', False)
        assert isinstance(structured, bool), f'not a bool: {structured}'

        structured_delegate_s = e.pop('structured_delegate', None)
        assert structured_delegate_s is None or isinstance(structured_delegate_s, str), f'not a str: {structured_delegate}'
        structured_delegate: Optional[OperatorName] = None
        if structured_delegate_s is not None:
            structured_delegate = OperatorName.parse(structured_delegate_s)

        structured_inherits = e.pop('structured_inherits', None)
        assert structured_inherits is None or isinstance(structured_inherits, str), f'not a str: {structured_inherits}'

        python_module = e.pop('python_module', None)
        assert python_module is None or isinstance(python_module, str), f'not a str: {python_module}'

        category_override = e.pop('category_override', None)
        assert category_override is None or isinstance(category_override, str), f'not a str: {category_override}'

        from tools.codegen.api import cpp

        raw_dispatch = e.pop('dispatch', None)
        assert raw_dispatch is None or isinstance(raw_dispatch, dict), e
        dispatch: Dict[DispatchKey, str] = {}
        if raw_dispatch is not None:
            assert not manual_kernel_registration, \
                "cannot specify both manual_kernel_registration and dispatch; with " \
                "manual registration, dispatch has no effect!"
            for ks, v in raw_dispatch.items():
                if ks == '__line__':
                    continue  # not worth tracking line numbers for dispatch entries
                assert isinstance(ks, str), e
                assert isinstance(v, str), e
                for k in ks.split(","):
                    dispatch_key = DispatchKey.parse(k.strip())
                    dispatch[dispatch_key] = v
            assert dispatch != {DispatchKey.CompositeImplicitAutograd: cpp.name(func)}, \
                "unnecessary dispatch table for this function; just delete the dispatch " \
                "key entirely"
            assert dispatch.keys() != {DispatchKey.CompositeImplicitAutograd}, \
                f"unexpected name for singleton CompositeImplicitAutograd dispatch entry: expected {cpp.name(func)} " \
                f"but got {dispatch[DispatchKey.CompositeImplicitAutograd]}.  Rename your implementation to the expected " \
                "name, then delete the dispatch table"
        elif not structured and structured_delegate is None:
            dispatch[DispatchKey.CompositeImplicitAutograd] = cpp.name(func)

        assert not (DispatchKey.CompositeExplicitAutograd in dispatch and DispatchKey.CompositeImplicitAutograd in dispatch), \
            "cannot specify both CompositeExplicitAutograd and CompositeImplicitAutograd on a single kernel; each " \
            "strictly subsumes the other.  If you wanted to provide an explicit autograd " \
            "implementation, specify CompositeExplicitAutograd; otherwise specify CompositeImplicitAutograd only"

        e.pop('__line__')
        assert not e, f"leftover entries: {e}"

        return NativeFunction(
            func=func,
            variants=variants,
            structured=structured,
            structured_delegate=structured_delegate,
            structured_inherits=structured_inherits,
            manual_kernel_registration=manual_kernel_registration,
            manual_cpp_binding=manual_cpp_binding,
            python_module=python_module,
            category_override=category_override,
            dispatch=dispatch,
            device_guard=device_guard,
            loc=loc,
            cpp_no_default_args=cpp_no_default_args,
        )

    def validate_unstructured(self) -> None:
        # TODO: probably better to accumulate these errors and report them all
        # at once
        assert not self.structured, "This function is structured, but there was " \
            "no valid functional variant of it."
        assert self.structured_delegate, "This function delegates to another structured out function, " \
            "but no valid function was found (the delegate may not exist, or it has the wrong type)"

    # __post_init__ functions in dataclasses can be used to do extra
    # validation after construction.
    #
    # Notice that we don't do any type validation here.  In fact, we
    # rely exclusively on mypy to check if you've done types correctly!
    # Validation is for nontrivial invariants that cannot be (conveniently)
    # encoded in the type system.
    def __post_init__(self) -> None:
        if self.func.arguments.out:
            assert self.variants == {Variant.function}, "Native functions with out arguments MUST " \
                "be declared with only function variant; e.g., variants: function; " \
                "otherwise you will tickle a Python argument binding bug " \
                "(which usually manifests itself as the result variable being undefined.)"
        if self.structured:
            assert self.func.kind() == SchemaKind.out, "Put structured field on the out= " \
                "variant of a function; did you mean structured_delegate?"
            assert self.device_guard, "device_guard: False is not respected by structured kernels"
        if self.structured_delegate:
            assert self.func.kind() != SchemaKind.out, "structured_delegate field not allowed " \
                "on out= functions; did you mean structured?"
            assert self.device_guard, "device_guard: False is not respected by structured kernels"
        # Technically, with the asserts above, this assert is impossible to
        # happen
        assert not (self.structured and self.structured_delegate), \
            "Cannot have both structured and structured_delegate on function"
        defaulted_arguments = {a.name for a in self.func.schema_order_arguments()
                               if a.default is not None}
        invalid_args = set.difference(self.cpp_no_default_args, defaulted_arguments)
        assert len(invalid_args) == 0, f'Invalid cpp_no_default_args: {invalid_args}'
        if self.structured_inherits is not None:
            assert self.structured, "structured_inherits must also imply structured: True"
        if self.structured_delegate is not None:
            for k in STRUCTURED_DISPATCH_KEYS:
                assert k not in self.dispatch, \
                    f"if structured_delegate, then must not have {k} in dispatch dictionary " \
                    "(it is delegated!)"

SchemaKind = Enum('SchemaKind', ('functional', 'inplace', 'out'))

# A structured kernel is guaranteed to have a functional and out variant, and
# optionally an inplace variant.
#
# NB: we create NativeFunctionsGroup *even if* the function is not
# actually annotated structured.  Test the structured boolean to see if it
# actually is structured or not.
@dataclass(frozen=True)
class NativeFunctionsGroup:
    functional: NativeFunction
    inplace: Optional[NativeFunction]
    out: NativeFunction

    @property
    def structured(self) -> bool:
        return self.out.structured

    def __post_init__(self) -> None:
        test_sig: FunctionSchema = self.functional.func.signature()
        for f in self.functions():
            if test_sig != f.func.signature():
                raise AssertionError(
                    "NativeFunctionsGroup constructed from two NativeFunctions "
                    f"that don't have matching signatures: {test_sig} != {f.func.signature()}"
                )
        assert self.functional.func.kind() == SchemaKind.functional
        assert self.out.func.kind() == SchemaKind.out
        if self.inplace is not None:
            assert self.inplace.func.kind() == SchemaKind.inplace

        if self.structured:
            # For now, structured composite kernels are not supported (need some
            # design work to figure out how to make the composite case work)
            assert self.out.dispatch.keys() != {DispatchKey.CompositeImplicitAutograd}

            assert self.functional.structured_delegate == self.out.func.name, \
                f"{self.functional.func.name} delegates to {self.functional.structured_delegate} " \
                f"but its actual delegate is {self.out.func.name}"
            if self.inplace is not None:
                assert self.inplace.structured_delegate == self.out.func.name

    def signature(self) -> 'FunctionSchema':
        return self.out.func.signature()

    def functions(self) -> Iterator[NativeFunction]:
        yield self.out
        yield self.functional
        if self.inplace is not None:
            yield self.inplace

    @staticmethod
    def from_dict(d: Dict[SchemaKind, NativeFunction]) -> Optional['NativeFunctionsGroup']:
        assert d
        if len(d) == 1:
            return None
        d = dict(d)  # non-destructive updates please
        functional = d.pop(SchemaKind.functional, None)
        inplace = d.pop(SchemaKind.inplace, None)
        out = d.pop(SchemaKind.out, None)
        assert not d
        assert functional is not None
        # There are a few operators which only have functional/inplace variants;
        # these don't count as structured for our purposes here
        if out is None:
            return None
        return NativeFunctionsGroup(
            functional=functional,
            inplace=inplace,
            out=out,
        )

def is_foreach_op(name: str) -> bool:
    return str(name) in set([
        '_amp_foreach_non_finite_check_and_unscale_',
        '_foreach_add_.ScalarList',
        '_foreach_sub_.ScalarList',
        '_foreach_mul_.ScalarList',
        '_foreach_div_.ScalarList',
        '_foreach_add_.Scalar',
        '_foreach_sub_.Scalar',
        '_foreach_mul_.Scalar',
        '_foreach_div_.Scalar',
        '_foreach_add_.List',
        '_foreach_sub_.List',
        '_foreach_mul_.List',
        '_foreach_div_.List',
        '_foreach_exp_',
        '_foreach_sqrt_',
        '_foreach_abs_',
        '_foreach_acos_',
        '_foreach_asin_',
        '_foreach_atan_',
        '_foreach_ceil_',
        '_foreach_cos_',
        '_foreach_cosh_',
        '_foreach_erf_',
        '_foreach_erfc_',
        '_foreach_expm1_',
        '_foreach_floor_',
        '_foreach_log_',
        '_foreach_log10_',
        '_foreach_log1p_',
        '_foreach_log2_',
        '_foreach_neg_',
        '_foreach_tan_',
        '_foreach_tanh_',
        '_foreach_sin_',
        '_foreach_sinh_',
        '_foreach_round_',
        '_foreach_lgamma_',
        '_foreach_frac_',
        '_foreach_reciprocal_',
        '_foreach_sigmoid_',
        '_foreach_trunc_',
        '_foreach_addcmul_.Scalar',
        '_foreach_addcdiv_.Scalar',
        '_foreach_addcmul_.ScalarList',
        '_foreach_addcdiv_.ScalarList',
        '_foreach_zero_'])

# The function schema is undoubtedly the most important data structure
# in all of the codegen, as it defines the type signature for operators,
# and most of the code generation we do is type directed (e.g., look at
# the types, decide what to do.  Think about how we code generate
# C++ function stubs!)
#
# We will also see in this class the general structure for how we model
# data in this code generation.  A few notable properties to point out
# ahead of time:
#
#   - These dataclasses are a *lossless* representation of the strings
#     they are parsed from.  In fact, we assert that given the
#     information stored in the dataclass, we can exactly reconstruct
#     the string we parsed from (and assert this inside the parse
#     definition).  There are a few reasons for this:
#
#       - If you find that it is difficult to reconstruct the string
#         given a dataclass, that is a clue that you are data
#         representation is wrong.
#
#       - It helps ensure that all relevant information is present
#         in the dataclass, so that downstream users aren't tempted
#         to reparse the original string to get some information
#         that was omitted.
#
#       - It forces you to represent the data in-memory in the same way
#         it is recorded textually, which makes the dataclasses easier
#         to understand for someone who is familiar with the
#         textual format.  (As a tradeoff, it means you have to model
#         the syntax, even when it is inconvenient.  But maybe that means
#         the syntax is bad!)  If you don't understand the internal
#         representation, go look at the printing code to see how
#         it maps onto the surface syntax!
#
#       - It makes it easy to test the parsing code, as parsing code
#         that is inconsistent with the string code will fail early
#         and loudly.  (As a tradeoff, it makes the parsing code a bit
#         brittle (in particular, with trivial whitespace changes you
#         are likely to trigger an assert error).
#
#     In general, try to make the __str__ code as simple as possible
#     (even at the cost of more complex parsing logic.)  Additionally,
#     try to minimize redundancy in data representation.  (Precomputed
#     fields are OK though: they are defined as a simple function on
#     the canonical representation in question.)
#
#   - These dataclasses are all frozen; once constructed their
#     values never change.  This makes it easy to tell where any
#     given data came from: just look to the constructor.  As a
#     tradeoff, you can't easily "decorate" a schema with extra
#     information from a post-facto analysis.  We impose this
#     restriction to make these structures more understandable.
#
@dataclass(frozen=True)
class FunctionSchema:
    # The name of the operator this function schema describes.
    name: 'OperatorName'

    arguments: 'Arguments'

    # TODO: Need to handle collisions with argument names at some point
    returns: Tuple['Return', ...]

    def schema_order_arguments(self) -> Iterator['Argument']:
        return itertools.chain(
            self.arguments.flat_positional,
            self.arguments.flat_kwarg_only,
            self.arguments.out
        )

    @staticmethod
    def parse(func: str) -> 'FunctionSchema':
        # We should probably get a proper parser here
        assert ' -> ' in func, "function schema missing return type (spaces are mandatory)"
        func_decl, return_decl = [x.strip() for x in func.split(' -> ')]
        ops, args = func_decl.split('(', 1)
        assert args[-1] == ")", "Expecting closing )"
        args = args[:-1]
        name = OperatorName.parse(ops)
        arguments = Arguments.parse(args)
        returns = parse_returns(return_decl)
        r = FunctionSchema(
            name=name,
            arguments=arguments,
            returns=returns
        )
        assert str(r) == func, f'{str(r)} != {func}'
        return r

    def __post_init__(self) -> None:
        for arg, ret in zip(self.arguments.out, self.returns):
            assert arg.annotation == ret.annotation, \
                "Out arguments must have matching return Tensor; furthermore, " \
                "the ith-argument needs to correspond to the ith return"
        # Invariant: we expect out arguments to appear as keyword arguments in the schema.
        # This means that all mutable returns should be aliased to a keyword argument
        # (except for "self", which we explicitly don't treat as an out argument because of its use in methods)
        # See Note [is_out_fn]
        out_and_self = list(self.arguments.out) + [arg for arg in self.arguments.flat_positional if arg.name == "self"]
        mutable_returns = [ret for ret in self.returns if ret.annotation is not None and ret.annotation.is_write]
        for ret in mutable_returns:
            assert any([ret.annotation == arg.annotation for arg in out_and_self]), \
                "All mutable returns must be aliased either to a keyword argument, or to \"self\". " \
                "Did you forget to mark an out argument as keyword-only?"
        if self.arguments.out:
            assert len(self.arguments.out) == len(self.returns), \
                "Must return as many arguments as there are out arguments"
        if self.name.name.inplace:
            # TODO: fixme
            if not is_foreach_op(str(self.name)):
                assert len(self.returns) == 1

    def is_out_fn(self) -> bool:
        # Note [is_out_fn]
        #
        # out functions are the variants which take an explicit out= argument
        # to populate into.  We need to know if a schema corresponds to an
        # out function for several reasons:
        #
        #   - They codegen differently in C++ API
        #       - codegen to at::add_out rather than at::add
        #       - out argument is moved to front of C++ argument list
        #
        # out functions are DEFINED to be any function with a keyword-only
        # argument that is mutable.  In principle, this could lead to a
        # false positive if you define a function that mutates a
        # kwarg only argument, but this isn't the "true" output of this
        # function.  A more robust definition that would work in this
        # case would also look at:
        #
        #   - The output types.  Out functions take in the arguments
        #     they mutate and then return them again; this is sort
        #     of "definitionally" what makes something an out function.
        #     Historically, we DO check this for consistency.
        #   - Correspondence with pure variant.  An out function
        #     should have a signature equivalent to its pure variant,
        #     but just with extra kwargs for the output elements.  This
        #     is difficult to actually check for and historically
        #     we only do this check in tools/
        return bool(self.arguments.out)

    def kind(self) -> SchemaKind:
        """
        What kind of schema is this?  A functional schema is one
        that returns a newly allocated output; an inplace schema
        modifies the self argument inplace; an out schema writes
        the result into an explicitly provided out argument.
        """
        is_inplace = self.name.name.inplace
        is_out = bool(self.arguments.out)
        assert not (is_inplace and is_out)
        if is_inplace:
            return SchemaKind.inplace
        elif is_out:
            return SchemaKind.out
        else:
            return SchemaKind.functional

    def signature(self, *, strip_default: bool = False) -> 'FunctionSchema':
        """
        Certain schemas are 'related', in that they are simply
        inplace/out/functional versions of the same function.  This method
        factors these schemas into the "core" functional signature which
        is equal across all versions.

        Here is what normalization happens to the schema to convert
        it to a signature:
        - The overload name is stripped (name is retained, since
          it expresses semantic content about what the function does)
        - Inplace is set False
        - Out arguments are stripped
        - Mutability annotations are stripped  (this is sound
          because you cannot overload on mutability annotation)
        - Return names are stripped since they are not overloadable and
          some variants have return names but some not
        """

        def strip_ret_annotation(r: Return) -> Return:
            return Return(
                name=None,
                type=r.type,
                annotation=None,
            )

        return FunctionSchema(
            name=OperatorName(
                name=BaseOperatorName(
                    base=self.name.name.base,
                    inplace=False,
                    dunder_method=self.name.name.dunder_method,
                ),
                overload_name="",  # stripped
            ),
            arguments=self.arguments.signature(strip_default=strip_default),
            returns=tuple(map(strip_ret_annotation, self.returns)),
        )

    def __str__(self) -> str:
        all_arguments_str = str(self.arguments)
        if len(self.returns) == 1:
            returns = str(self.returns[0])  # omit parentheses
        else:
            returns = '(' + ', '.join(map(str, self.returns)) + ')'
        return f'{self.name}({all_arguments_str}) -> {returns}'

# Here is the rest of the data model, described more briefly.

# Simplified version for what actually shows up in built-ins.
# Look at alias_info.h for expanded syntax.  If you need the structure,
# you also need to make this structure recursive so it can be lined
# up with the type components too.  For primitives this isn't really
# necessary
@dataclass(frozen=True)
class Annotation:
    # Typically only has one element.  Not actually a set so
    # we can conveniently assume it is canonically ordered
    alias_set: Tuple[str, ...]
    is_write: bool

    @staticmethod
    def parse(ann: str) -> 'Annotation':
        m = re.match(r'^([a-z])(!?)$', ann)
        assert m is not None, f'unrecognized alias annotation {ann}'
        alias_set = (m.group(1),)
        is_write = m.group(2) == '!'
        r = Annotation(alias_set=alias_set, is_write=is_write)
        assert str(r) == ann, f'{r} != {ann}'
        return r

    def __str__(self) -> str:
        alias_set = '|'.join(self.alias_set)
        is_write = '!' if self.is_write else ''
        return f'{alias_set}{is_write}'

# The base class for the type system.  This is also loosely modeled
# off of jit_type.h, but we've simplified the hierarchy to focus
# in on the aspects of the type system that matter for code generation
# (for example, there's no SingleElementType subclass anymore).
# You never actually construct a Type; usually it's going to be one
# of the subclasses.  If Python had ADTs this would be one!
@dataclass(frozen=True)
class Type:
    @staticmethod
    def parse(t: str) -> 'Type':
        r = Type._parse(t)
        assert str(r) == t, f'{r} != {t}'
        return r

    @staticmethod
    def _parse(t: str) -> 'Type':
        m = re.match(r'^(.+)\?$', t)
        if m is not None:
            return OptionalType(Type.parse(m.group(1)))
        m = re.match(r'^(.+)\[([0-9]+)?\]$', t)
        if m is not None:
            size = int(m.group(2)) if m.group(2) is not None else None
            return ListType(elem=Type.parse(m.group(1)), size=size)
        try:
            return BaseType(BaseTy[t])
        except KeyError:
            raise RuntimeError(f"unrecognized type {t}")

    def __str__(self) -> str:
        raise NotImplementedError

    # WARNING: These concepts are not very well-defined.  For example,
    # is "int?" nullable? How about "int?[]".  They are defined
    # so we can conveniently generate legacy Declarations.yaml but
    # really we should probably just remove these at some point

    def is_tensor_like(self) -> bool:
        raise NotImplementedError

    def is_nullable(self) -> bool:
        raise NotImplementedError

    def is_list_like(self) -> Optional['ListType']:
        raise NotImplementedError

# Base types are simple, atomic types with no further structure
BaseTy = Enum('BaseTy', (
    'Generator',
    'ScalarType',
    'Tensor',
    'int',
    'Dimname',
    'float',
    'str',
    'bool',
    'Layout',
    'Device',
    'Scalar',
    'MemoryFormat',
    'QScheme',
    'Storage',
    'Stream',
    'ConstQuantizerPtr',  # TODO: rename
))

@dataclass(frozen=True)
class BaseType(Type):
    name: BaseTy

    def __str__(self) -> str:
        return f'{self.name.name}'

    def is_tensor_like(self) -> bool:
        return self.name == BaseTy.Tensor

    def is_nullable(self) -> bool:
        return False

    def is_list_like(self) -> Optional['ListType']:
        return None

# Optional types may be specified, or may also be validly given None
@dataclass(frozen=True)
class OptionalType(Type):
    elem: Type

    def __str__(self) -> str:
        return f'{self.elem}?'

    def is_tensor_like(self) -> bool:
        return self.elem.is_tensor_like()

    def is_nullable(self) -> bool:
        return True

    def is_list_like(self) -> Optional['ListType']:
        return self.elem.is_list_like()

# List types specify that we may have multiples of an element.  We
# also support explicit sizes on list types, but these have
# some nontrivial semantics!  (However, for C++ API purposes, explicit
# sizes are mostly erased from the type system.)
#
# DANGER WILL ROBINSON: C++ elaboration depends on elem type; e.g.,
# int[] elaborates differently than bool[3]!
@dataclass(frozen=True)
class ListType(Type):
    elem: Type
    size: Optional[int]

    def __str__(self) -> str:
        size = f'{self.size}' if self.size else ''
        return f'{self.elem}[{size}]'

    def is_tensor_like(self) -> bool:
        return self.elem.is_tensor_like()

    def is_nullable(self) -> bool:
        return self.elem.is_nullable()

    def is_list_like(self) -> Optional['ListType']:
        return self

@dataclass(frozen=True)
class Argument:
    # NB: I didn't put kwarg_only as a boolean field here, unlike
    # c10::Argument, so that printing works correctly

    name: str
    type: Type
    default: Optional[str]

    # The semantics of the annotation field are a little strange.
    #
    # Alias annotations parametrize Tensors (since Tensors are the only things
    # that can alias.)  This motivates why I write Tensor(a!)?  (and not, for
    # example, Tensor?(a!)), because the (a!) describes aliasing on the tensor,
    # which may be optional (i.e., the alias annotation should bind first to
    # Tensor, before the optional postfix annotation).
    #
    # However, despite being a property of Tensor, we (and c10::Argument)
    # store the annotation at the top level of the Argument, rather than
    # inside the embedded Tensor type.  In the C++ version of this
    # class, we then go through great lengths to mimic the type
    # structure in the annotation structure so we can correlate
    # annotations with types.
    #
    # Now, it turns out, in all applications in code generation, the
    # structure of annotated types is very simple.  So we just hard
    # code it here.  But if we ever do get anything more complex, this
    # model will have to change!
    annotation: Optional[Annotation]

    @staticmethod
    def parse(arg: str) -> 'Argument':
        name: str
        default: Optional[str]
        type_and_annot, name_and_default = arg.rsplit(' ', 1)
        if '=' in name_and_default:
            name, default = name_and_default.split('=')
        else:
            name = name_and_default
            default = None
        # TODO: deduplicate annotation matching with Return
        match = re.match(r'Tensor\((.+)\)(.*)', type_and_annot)
        annotation: Optional[Annotation]
        if match:
            # If you update this, make sure the __str__ still works too
            assert match.group(2) in ['', '?', '[]'], 'unrecognized alias analysis form with Tensor'
            type_s = 'Tensor' + match.group(2)
            annotation = Annotation.parse(match.group(1))
        else:
            type_s = type_and_annot
            annotation = None
        type = Type.parse(type_s)
        r = Argument(
            name=name,
            type=type,
            default=default,
            annotation=annotation,
        )
        assert str(r) == arg, f'{str(r)} != {arg}'
        return r

    @property
    def is_write(self) -> bool:
        return self.annotation is not None and self.annotation.is_write

    def __str__(self) -> str:
        type = f'{self.type}'
        if self.annotation:
            assert type in ['Tensor', 'Tensor?', 'Tensor[]']
            type = type.replace('Tensor', f'Tensor({self.annotation})')
        if self.name is None:
            return type
        else:
            mb_default = ''
            if self.default:
                mb_default = f'={self.default}'
            return f"{type} {self.name}{mb_default}"


@dataclass(frozen=True)
class Return:
    name: Optional[str]
    type: Type
    annotation: Optional[Annotation]

    @staticmethod
    def parse(arg: str) -> 'Return':
        name: Optional[str]
        if ' ' in arg:
            type_and_annot, name = arg.rsplit(' ', 1)
        else:
            type_and_annot = arg
            name = None
        match = re.match(r'Tensor\((.+)\)(.*)', type_and_annot)
        annotation: Optional[Annotation]
        if match:
            # If you update this, make sure the __str__ still works too
            assert match.group(2) in ['', '?', '[]'], 'unrecognized alias analysis form with Tensor'
            type_s = 'Tensor' + match.group(2)
            annotation = Annotation.parse(match.group(1))
        else:
            type_s = type_and_annot
            annotation = None
        type = Type.parse(type_s)
        r = Return(
            name=name,
            type=type,
            annotation=annotation,
        )
        assert str(r) == arg, f'{str(r)} != {arg}'
        return r

    @property
    def is_write(self) -> bool:
        return self.annotation is not None and self.annotation.is_write

    def __str__(self) -> str:
        type = f'{self.type}'
        if self.annotation:
            assert type in ['Tensor', 'Tensor?', 'Tensor[]']
            type = type.replace('Tensor', f'Tensor({self.annotation})')
        if self.name is None:
            return type
        else:
            return f"{type} {self.name}"


# Represents the self argument for functions that may be methods
@dataclass(frozen=True)
class SelfArgument:
    argument: Argument

# Bundle of arguments that represent a TensorOptions.  This is mostly
# relevant for the public C++ API but we bake it into the core data
# model because other APIs often have to interact with it
@dataclass(frozen=True)
class TensorOptionsArguments:
    dtype: Argument
    layout: Argument
    device: Argument
    pin_memory: Argument

    def all(self) -> Sequence[Argument]:
        return [self.dtype, self.layout, self.device, self.pin_memory]

@dataclass(frozen=True)
class Arguments:
    # pre_self_positional is usually empty, but is notably non-empty
    # for where.self, where the condition argument comes before the
    # self argument
    pre_self_positional: Tuple[Argument, ...]
    self_arg: Optional[SelfArgument]
    post_self_positional: Tuple[Argument, ...]

    pre_tensor_options_kwarg_only: Tuple[Argument, ...]
    tensor_options: Optional[TensorOptionsArguments]
    # post_tensor_options is typically memory format, which should be
    # part of tensor options but isn't right now, and is usually
    # placed after the tensor options arguments
    post_tensor_options_kwarg_only: Tuple[Argument, ...]

    # Unlike in the previous codegen, we have factored out 'out' arguments
    # in the canonical representation, removing them from kwarg
    # arguments.  This choice is justified by numerous downstream
    # transformations which treat out arguments specially; additionally,
    # you can see that canonicity is not violated!
    out: Tuple[Argument, ...]  # these are also kwarg-only

    @property
    def flat_non_out(self) -> Sequence[Argument]:
        ret: List[Argument] = []
        ret.extend(self.flat_positional)
        ret.extend(self.flat_kwarg_only)
        return ret

    @property
    def flat_positional(self) -> Sequence[Argument]:
        ret: List[Argument] = []
        ret.extend(self.pre_self_positional)
        if self.self_arg is not None:
            ret.append(self.self_arg.argument)
        ret.extend(self.post_self_positional)
        return ret

    # NB: doesn't contain out arguments
    @property
    def flat_kwarg_only(self) -> Sequence[Argument]:
        ret: List[Argument] = []
        ret.extend(self.pre_tensor_options_kwarg_only)
        if self.tensor_options is not None:
            ret.extend(self.tensor_options.all())
        ret.extend(self.post_tensor_options_kwarg_only)
        return ret

    @property
    def non_out(self) -> Sequence[Union[Argument, SelfArgument, TensorOptionsArguments]]:
        ret: List[Union[Argument, SelfArgument, TensorOptionsArguments]] = []
        ret.extend(self.positional)
        ret.extend(self.kwarg_only)
        return ret

    @property
    def positional(self) -> Sequence[Union[Argument, SelfArgument]]:
        ret: List[Union[Argument, SelfArgument]] = []
        ret.extend(self.pre_self_positional)
        if self.self_arg is not None:
            ret.append(self.self_arg)
        ret.extend(self.post_self_positional)
        return ret

    @property
    def kwarg_only(self) -> Sequence[Union[Argument, TensorOptionsArguments]]:
        ret: List[Union[Argument, TensorOptionsArguments]] = []
        ret.extend(self.pre_tensor_options_kwarg_only)
        if self.tensor_options is not None:
            ret.append(self.tensor_options)
        ret.extend(self.post_tensor_options_kwarg_only)
        return ret

    # NB: contains out args.
    @property
    def tensor_args(self) -> Sequence[Union[Argument, SelfArgument]]:
        ret: List[Union[Argument, SelfArgument]] = []
        candidates: List[Union[Argument, SelfArgument]] = []
        candidates.extend(self.positional)
        candidates.extend(self.pre_tensor_options_kwarg_only)
        candidates.extend(self.post_tensor_options_kwarg_only)
        candidates.extend(self.out)
        for arg in candidates:
            a = arg.argument if isinstance(arg, SelfArgument) else arg
            if a.type.is_tensor_like():
                ret.append(arg)
        return ret


    def signature(self, *, strip_default: bool = False) -> 'Arguments':
        # dataclasses.replace could be used here, but it is less
        # type safe so for now I've opted to type everything out
        def strip_arg_annotation(a: Argument) -> Argument:
            return Argument(
                name=a.name,
                type=a.type,
                default=a.default if not strip_default else None,
                annotation=None,
            )

        return Arguments(
            pre_self_positional=tuple(map(strip_arg_annotation, self.pre_self_positional)),
            self_arg=SelfArgument(
                strip_arg_annotation(self.self_arg.argument)
            ) if self.self_arg is not None else None,
            post_self_positional=tuple(map(strip_arg_annotation, self.post_self_positional)),
            pre_tensor_options_kwarg_only=tuple(map(strip_arg_annotation, self.pre_tensor_options_kwarg_only)),
            # NB: tensor_options guaranteed to not have any alias annotations
            tensor_options=self.tensor_options,
            post_tensor_options_kwarg_only=tuple(map(strip_arg_annotation, self.post_tensor_options_kwarg_only)),
            # out arguments are dropped in signature
            out=(),
        )


    @staticmethod
    def _preparse(args: str) -> Tuple[List[Argument], List[Argument], List[Argument]]:
        positional: List[Argument] = []
        kwarg_only: List[Argument] = []
        out: List[Argument] = []
        arguments_acc = positional

        # TODO: Use a real parser here; this will get bamboozled
        # by signatures that contain things like std::array<bool, 2> (note the space)
        for arg in args.split(', '):
            if not arg:
                continue
            if arg == '*':
                assert arguments_acc is positional, "invalid syntax: kwarg-only specifier * can only occur once"
                arguments_acc = kwarg_only
                continue
            parg = Argument.parse(arg)
            # Currently, we rely directly on the invariant that there are NO
            # kwarg-only mutating arguments.  If you want to relax this,
            # we will need a more semantic way of matching that takes
            # into account return arguments.  In that case, you will have
            # to manage out computation a level up, in FunctionSchema.  See Note
            # [is_out_fn]
            if parg.annotation is not None and parg.annotation.is_write:
                if arguments_acc is positional:
                    pass  # do nothing
                elif arguments_acc is kwarg_only:
                    arguments_acc = out
            else:
                assert arguments_acc is not out
            arguments_acc.append(parg)

        return positional, kwarg_only, out

    @staticmethod
    def parse(args: str) -> 'Arguments':
        """
        Input: 'int x, int y, int z'
        """

        # We do this in two phases.  First we parse into three
        # main categories: positional, kwarg_only, out.
        # Then, we reparse positional and kwarg_only to separate
        # out the self argument and tensor options arguments.

        positional, kwarg_only, out = Arguments._preparse(args)

        # Split self argument
        self_ix = None
        for i, a in enumerate(positional):
            if a.name == "self":
                self_ix = i
                break
        pre_self_positional: List[Argument]
        self_arg: Optional[SelfArgument]
        post_self_positional: List[Argument]
        if self_ix is not None:
            pre_self_positional = positional[:self_ix]
            self_arg = SelfArgument(positional[self_ix])
            post_self_positional = positional[self_ix + 1:]
        else:
            pre_self_positional = []
            self_arg = None
            post_self_positional = positional

        # Group tensor options arguments
        pre_tensor_options_kwarg_only: List[Argument] = []
        tensor_options: Optional[TensorOptionsArguments] = None
        post_tensor_options_kwarg_only: List[Argument] = []
        kwarg_only_acc = pre_tensor_options_kwarg_only

        def pred(name: str, ty: Type) -> Callable[[Argument], bool]:
            return lambda a: a.name == name and a.type in [ty, OptionalType(ty)]
        predicates = [  # order matters
            pred('dtype', Type.parse('ScalarType')),
            pred('layout', Type.parse('Layout')),
            pred('device', Type.parse('Device')),
            pred('pin_memory', Type.parse('bool')),
        ]

        i = 0
        while i < len(kwarg_only):
            # If there is enough space...
            if i <= len(kwarg_only) - len(predicates):
                # And the next len(predicates) arguments look like TensorOptions arguments
                if all(p(a) for p, a in zip(predicates, kwarg_only[i : i + len(predicates)])):
                    assert kwarg_only_acc is pre_tensor_options_kwarg_only
                    # Group them together as one argument
                    tensor_options = TensorOptionsArguments(
                        dtype=kwarg_only[i],
                        layout=kwarg_only[i + 1],
                        device=kwarg_only[i + 2],
                        pin_memory=kwarg_only[i + 3],
                    )
                    i += len(predicates)
                    kwarg_only_acc = post_tensor_options_kwarg_only
                    continue
            kwarg_only_acc.append(kwarg_only[i])
            i += 1

        return Arguments(
            pre_self_positional=tuple(pre_self_positional),
            self_arg=self_arg,
            post_self_positional=tuple(post_self_positional),
            pre_tensor_options_kwarg_only=tuple(pre_tensor_options_kwarg_only),
            tensor_options=tensor_options,
            post_tensor_options_kwarg_only=tuple(post_tensor_options_kwarg_only),
            out=tuple(out),
        )


    def __str__(self) -> str:
        all_arguments: List[str] = []
        all_arguments.extend(map(str, self.flat_positional))
        if self.flat_kwarg_only or self.out:
            all_arguments.append('*')
        all_arguments.extend(map(str, self.flat_kwarg_only))
        all_arguments.extend(map(str, self.out))
        return ', '.join(all_arguments)

    def __post_init__(self) -> None:
        # TODO: These invariants are weirdly asymmetric?
        # TODO: Fancier types?
        if self.self_arg is None:
            assert not self.pre_self_positional
        if self.tensor_options is None:
            assert not self.post_tensor_options_kwarg_only


# Names that validly are __iXXX__ indicating inplace operations.
# Taken from https://www.python.org/dev/peps/pep-0203/#new-methods
# NB: PyTorch hasn't actually implemented all of these
AUGMENTED_ASSIGNMENT_NAMES = ['add', 'sub', 'mul', 'div', 'mod', 'pow', 'lshift', 'rshift', 'and', 'xor', 'or']

# A BaseOperatorName is what we think of the operator name, without
# the overload name.  Unusually, we don't represent this as just a
# string; instead, we directly represent a few important semantic
# bits of information we derive from the string: namely whether
# or not it's inplace (add_) and whether or not it's a double-underscore
# method (__add__)
@dataclass(frozen=True)
class BaseOperatorName:
    base: str
    inplace: bool
    dunder_method: bool

    @staticmethod
    def parse(op: str) -> 'BaseOperatorName':
        assert op != ''
        assert not op.endswith('_out'), \
            "_out suffix is reserved and not permitted for operator names; " \
            "did you mean to specify an out overload name instead?"
        m = re.match(r'^__([^_]+)__$', op)
        if m is not None:
            dunder_method = True
            base = m.group(1)
            if any(base == f'i{n}' for n in AUGMENTED_ASSIGNMENT_NAMES):
                inplace = True
                base = base[1:]
            else:
                inplace = False
                # temporary, this is not intrinsically true but
                # has been historically true for dunder methods
                # we support  (but, if we ever got, say, __int__, this would
                # be wrong!)
                assert base[0] != 'i'
        else:
            dunder_method = False
            base = op
            if base[-1] == '_':
                inplace = True
                base = base[:-1]
            else:
                inplace = False
        r = BaseOperatorName(base=base, inplace=inplace, dunder_method=dunder_method)
        assert str(r) == op, f'{str(r)} != {op}'
        return r

    def __str__(self) -> str:
        if self.dunder_method:
            i = 'i' if self.inplace else ''
            return f'__{i}{self.base}__'
        else:
            i = '_' if self.inplace else ''
            return f'{self.base}{i}'

# Operator name is the base operator name along with the (typically not
# user visible) overload string.
@dataclass(frozen=True)
class OperatorName:
    name: BaseOperatorName
    overload_name: str

    @staticmethod
    def parse(op_name: str) -> 'OperatorName':
        if '.' in op_name:
            name, overload_name = op_name.split('.', 1)
        else:
            name = op_name
            overload_name = ''
        r = OperatorName(
            name=BaseOperatorName.parse(name),
            overload_name=overload_name
        )
        assert str(r) == op_name, f'{str(r)} != {op_name}'
        return r

    def __str__(self) -> str:
        if self.overload_name:
            return f"{self.name}.{self.overload_name}"
        else:
            return f"{self.name}"

# Helper functions for parsing argument lists (both inputs and returns)

def parse_returns(return_decl: str) -> Tuple[Return, ...]:
    """
    Input: '()'
    Output: []
    """
    if return_decl == '()':
        return ()
    if return_decl[0] == '(' and return_decl[-1] == ')':
        return_decl = return_decl[1:-1]
    return tuple(Return.parse(arg) for arg in return_decl.split(', '))
