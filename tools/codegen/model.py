import re

from dataclasses import dataclass
from typing import List, Sequence, Dict, Optional, Iterator, Tuple, Set, NoReturn
from enum import Enum
import itertools

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

UseC10Dispatcher = Enum('UseC10Dispatcher', (
    'full',
    'with_codegenerated_unboxing_wrapper'
))

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

    # Corresponds to the 'use_c10_dispatcher' field.  The default
    # is 'with_codegenerated_unboxing_wrapper'
    use_c10_dispatcher: UseC10Dispatcher

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

    # Distinguish between a missing dispatch dict (historically, this
    # means to register a catch-all kernel) and a present but empty
    # dispatch dict (this means register nothing; arguably, this should
    # subsume manual_kernel_registration).
    #
    # TODO: str key could be replaced with more explicit enum
    dispatch: Optional[Dict[str, str]]

    # The location in the YAML file were this native function entry was
    # defined.  This is for conveniently reporting error messages!
    loc: 'Location'

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

        use_c10_dispatcher_s = e.pop('use_c10_dispatcher', None)
        if use_c10_dispatcher_s is None:
            use_c10_dispatcher = UseC10Dispatcher.with_codegenerated_unboxing_wrapper
        elif use_c10_dispatcher_s == 'full':
            use_c10_dispatcher = UseC10Dispatcher.full
        else:
            raise AssertionError(
                f'use_c10_dispatcher must be unset or set to full, got {use_c10_dispatcher}')

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

        device_guard = e.pop('device_guard', True)
        assert isinstance(device_guard, bool), f'not a bool: {device_guard}'

        python_module = e.pop('python_module', None)
        assert python_module is None or isinstance(python_module, str), f'not a str: {python_module}'

        category_override = e.pop('category_override', None)
        assert category_override is None or isinstance(category_override, str), f'not a str: {category_override}'

        raw_dispatch = e.pop('dispatch', None)
        assert raw_dispatch is None or isinstance(raw_dispatch, dict), e
        dispatch: Optional[Dict[str, str]] = None
        if raw_dispatch is not None:
            dispatch = {}
            for ks, v in raw_dispatch.items():
                if ks == '__line__':
                    continue  # not worth tracking line numbers for dispatch entries
                assert isinstance(ks, str), e
                assert isinstance(v, str), e
                for k in ks.split(","):
                    dispatch[k.strip()] = v

        e.pop('__line__')
        assert not e, f"leftover entries: {e}"

        return NativeFunction(
            func=func,
            use_c10_dispatcher=use_c10_dispatcher,
            variants=variants,
            manual_kernel_registration=manual_kernel_registration,
            python_module=python_module,
            category_override=category_override,
            dispatch=dispatch,
            device_guard=device_guard,
            loc=loc,
        )

    # __post_init__ functions in dataclasses can be used to do extra
    # validation after construction.
    #
    # Notice that we don't do any type validation here.  In fact, we
    # rely exclusively on mypy to check if you've done types correctly!
    # Validation is for nontrivial invariants that cannot be (conveniently)
    # encoded in the type system.
    def __post_init__(self) -> None:
        if self.func.out_arguments:
            assert self.variants == {Variant.function}, "Native functions with out arguments MUST " \
                "be declared with only function variant; e.g., variants: function; " \
                "otherwise you will tickle a Python argument binding bug " \
                "(which usually manifests itself as the result variable being undefined.)"

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

    # NB: Sequence here is intentional, to make it read only
    arguments: Sequence['Argument']
    kwarg_only_arguments: Sequence['Argument']  # but not including out args
    # Unlike in the previous codegen, we have factored out 'out' arguments
    # in the canonical representation, removing them from kwarg
    # arguments.  This choice is justified by numerous downstream
    # transformations which treat out arguments specially; additionally,
    # you can see that canonicity is not violated!
    out_arguments: Sequence['Argument']  # these are also kwarg-only

    # TODO: Need to handle collisions with argument names at some point
    returns: Sequence['Return']

    def schema_order_arguments(self) -> Iterator['Argument']:
        return itertools.chain(self.arguments, self.kwarg_only_arguments, self.out_arguments)

    @staticmethod
    def parse(func: str) -> 'FunctionSchema':
        # We should probably get a proper parser here
        assert ' -> ' in func, "function schema missing return type (spaces are mandatory)"
        func_decl, return_decl = [x.strip() for x in func.split(' -> ')]
        ops, args = func_decl.split('(', 1)
        assert args[-1] == ")", "Expecting closing )"
        args = args[:-1]
        name = OperatorName.parse(ops)
        arguments, kwarg_only_arguments, out_arguments = parse_arguments(args)
        returns = parse_returns(return_decl)
        r = FunctionSchema(
            name=name,
            arguments=arguments,
            kwarg_only_arguments=kwarg_only_arguments,
            out_arguments=out_arguments,
            returns=returns
        )
        assert str(r) == func, f'{str(r)} != {func}'
        return r

    def __post_init__(self) -> None:
        for arg, ret in zip(self.out_arguments, self.returns):
            assert arg.annotation == ret.annotation, \
                "Out arguments must have matching return Tensor; furthermore, " \
                "the ith-argument needs to correspond to the ith return"
        if self.out_arguments:
            assert len(self.out_arguments) == len(self.returns), \
                "Must return as many arguments as there are out arguments"
        if self.name.name.inplace:
            # TODO: fixme
            if str(self.name) not in [
                    '_amp_non_finite_check_and_unscale_',
                    '_foreach_add_.Scalar',
                    '_foreach_add_.ScalarList',
                    '_foreach_sub_.ScalarList',
                    '_foreach_mul_.ScalarList',
                    '_foreach_div_.ScalarList',
                    '_foreach_sub_.Scalar',
                    '_foreach_mul_.Scalar',
                    '_foreach_div_.Scalar',
                    '_foreach_add_.List',
                    '_foreach_sub_.List',
                    '_foreach_mul_.List',
                    '_foreach_div_.List',
                    '_foreach_exp_',
                    '_foreach_sqrt_',
                    '_foreach_addcmul_',
                    '_foreach_addcdiv_']:
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
        return bool(self.out_arguments)

    def __str__(self) -> str:
        all_arguments: List[str] = []
        all_arguments.extend(map(str, self.arguments))
        if self.kwarg_only_arguments or self.out_arguments:
            all_arguments.append('*')
        all_arguments.extend(map(str, self.kwarg_only_arguments))
        all_arguments.extend(map(str, self.out_arguments))
        all_arguments_str = ', '.join(all_arguments)
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
    alias_set: Sequence[str]
    is_write: bool

    @staticmethod
    def parse(ann: str) -> 'Annotation':
        m = re.match(r'^([a-z])(!?)$', ann)
        assert m is not None, f'unrecognized alias annotation {ann}'
        alias_set = [m.group(1)]
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

def parse_returns(return_decl: str) -> Sequence[Return]:
    """
    Input: '()'
    Output: []
    """
    if return_decl == '()':
        return []
    if return_decl[0] == '(' and return_decl[-1] == ')':
        return_decl = return_decl[1:-1]
    returns = []
    for arg in return_decl.split(', '):
        returns.append(Return.parse(arg))
    return returns

def parse_arguments(args: str) -> Tuple[Sequence[Argument], Sequence[Argument], Sequence[Argument]]:
    """
    Input: 'int x, int y, int z'
    Output: positional args, kwarg only args
    """
    arguments: List[Argument] = []
    kwarg_only_arguments: List[Argument] = []
    out_arguments: List[Argument] = []
    arguments_acc = arguments

    # TODO: Use a real parser here; this will get bamboozled
    # by signatures that contain things like std::array<bool, 2> (note the space)
    for arg in args.split(', '):
        if not arg:
            continue
        if arg == '*':
            assert arguments_acc is arguments, "invalid syntax: kwarg-only specifier * can only occur once"
            arguments_acc = kwarg_only_arguments
            continue
        parg = Argument.parse(arg)
        # Currently, we rely directly on the invariant that there are NO
        # kwarg-only mutating arguments.  If you want to relax this,
        # we will need a more semantic way of matching that takes
        # into account return arguments.  In that case, you will have
        # to manage out_arguments computation a level up, in
        # FunctionSchema.  See Note [is_out_fn]
        if parg.annotation is not None and parg.annotation.is_write:
            if arguments_acc is arguments:
                pass  # do nothing
            elif arguments_acc is kwarg_only_arguments:
                arguments_acc = out_arguments
        else:
            assert arguments_acc is not out_arguments
        arguments_acc.append(parg)

    return arguments, kwarg_only_arguments, out_arguments
