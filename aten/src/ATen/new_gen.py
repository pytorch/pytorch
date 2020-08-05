import sys
from os import path
import contextlib
import textwrap
import re
import pprint
import dataclasses
from dataclasses import dataclass, field
from typing import List, Sequence, Dict, Optional, Iterator, Tuple, Set, NoReturn
from enum import Enum
import yaml

# Reusing CodeTemplate from existing codegen
sys.path.append(path.dirname(path.abspath(__file__)))
from code_template import CodeTemplate

try:
    # use faster C loader if available
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader  # type: ignore

# A little trick from https://github.com/python/mypy/issues/6366
# for getting mypy to do exhaustiveness checking
def _assert_never(x: NoReturn) -> NoReturn:
    assert False, "Unhandled type: {}".format(type(x).__name__)

# Welcome to the ATen code generator v2!  The ATen code generator is
# responsible for parsing native_functions.yaml and then generating
# various generated files (e.g., TypeDefault.cpp) based on the operators
# defined in this file.  This means that the code generator knows how to
# parse function schema, and then translate this into various C++ types
# and boilerplate code.
#
# I went into this rewrite with some goals:
#
# - Completely excise all legacy TH handling.  Declarations.cwrap isn't
#   a thing.  Preprocessing declarations isn't a thing.
#   native_functions.yaml is the only place where we get information
#   about operators.
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
#
# - Strict mypy typechecking.  You can typecheck this file using
#   `mypy -config mypy-strict.ini aten/src/ATen/gen_cpu.py` and
#   this will enforce that everything is annotated.
#
# Some non-goals:
#
# - Change native_functions.yaml format.  One step at a time!
#
# The general structure:
#
# - We define a lot of dataclasses to represent all of the various
#   semantic entities in native_functions.yaml (schema! types!)
#   These classes come with parsing and pretty-printing functionality.
#
# - We parse native_functions.yaml into our dataclasses
#
# - We do code generation on it (under construction!)
#

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           DATA MODEL
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

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

    # Corresponds to the 'use_c10_dispatcher' field.  Historically,
    # this field could take several possible strings, but right
    # now you can have it in any color you like, as long as it's 'full'
    use_c10_dispatcher_full: bool

    # If no variants are specified in native_functions.yaml, this is
    # assumed to be {'function'}.
    variants: Set['Variant']

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
    def from_yaml(e: Dict[str, object], loc: 'Location') -> 'NativeFunction':
        """
        Parse a NativeFunction from a dictionary as directly parsed
        from native_functions.yaml
        """
        funcs = e.get('func')
        assert isinstance(funcs, str), f'not a str: {funcs}'
        func = FunctionSchema.parse(funcs)

        use_c10_dispatcher = e.get('use_c10_dispatcher')
        assert use_c10_dispatcher is None or use_c10_dispatcher == 'full', \
            f'use_c10_dispatcher must be unset or set to full, got {use_c10_dispatcher}'
        use_c10_dispatcher_full = use_c10_dispatcher == 'full'

        variants_s = e.get('variants', 'function')
        assert isinstance(variants_s, str)
        variants: Set[Variant] = set()
        for v in variants_s.split(', '):
            if v == 'function':
                variants.add(Variant.function)
            elif v == 'method':
                variants.add(Variant.method)
            else:
                assert False, f'illegal variant {v}'

        manual_kernel_registration = e.get('manual_kernel_registration', False)
        assert isinstance(manual_kernel_registration, bool), f'not a bool: {manual_kernel_registration}'

        raw_dispatch = e.get('dispatch')
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

        return NativeFunction(
            func=func,
            use_c10_dispatcher_full=use_c10_dispatcher_full,
            variants=variants,
            manual_kernel_registration=manual_kernel_registration,
            dispatch=dispatch,
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
    returns: Sequence['Argument']

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
        returns = parse_return_arguments(return_decl)
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
            if str(self.name) not in ['_amp_non_finite_check_and_unscale_']:
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

# Valid values of the 'variants' field in native_functions.yaml
Variant = Enum('Variant', ('function', 'method'))

# Simplified version for what actually shows up in built-ins.
# Look at alias_info.h for expanded syntax.  If you need the structure,
# you also need to make this structure recursive so it can be lined
# up with the type components too.  For primitives this isn't really
# necessary
@dataclass
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
@dataclass
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
        raise NotImplemented

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

@dataclass
class BaseType(Type):
    name: BaseTy
    def __str__(self) -> str:
        return f'{self.name.name}'

# Optional types may be specified, or may also be validly given None
@dataclass
class OptionalType(Type):
    elem: Type
    def __str__(self) -> str:
        return f'{self.elem}?'

# List types specify that we may have multiples of an element.  We
# also support explicit sizes on list types, but these have
# some nontrivial semantics!  (However, for C++ API purposes, explicit
# sizes are mostly erased from the type system.)
#
# DANGER WILL ROBINSON: C++ elaboration depends on elem type; e.g.,
# int[] elaborates differently than bool[3]!
@dataclass
class ListType(Type):
    elem: Type
    size: Optional[int]
    def __str__(self) -> str:
        size = f'{self.size}' if self.size else ''
        return f'{self.elem}[{size}]'

# Arguments represent both input arguments, as well as return types from
# a function (we support named returns, so the data structure works
# in both caes.)
@dataclass
class Argument:
    # NB: I didn't put kwarg_only as a boolean field here, unlike
    # c10::Argument, so that printing works correctly

    name: Optional[str]
    type: Type
    # INVARIANT: if name is None, default is None
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
        name: Optional[str]
        default: Optional[str]
        if ' ' in arg:
            type_and_annot, name_and_default = arg.rsplit(' ', 1)
            if '=' in name_and_default:
                name, default = name_and_default.split('=')
            else:
                name = name_and_default
                default = None
        else:
            type_and_annot = arg
            name = None
            default = None
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

def parse_return_arguments(return_decl: str) -> Sequence[Argument]:
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
        returns.append(Argument.parse(arg))
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           PROCESSING
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Conveniently add error context to exceptions raised.  Lets us
# easily say that an error occurred while processing a specific
# context.
@contextlib.contextmanager
def context(msg: str) -> Iterator[None]:
    try:
        yield
    except Exception as e:
        # TODO: this does the wrong thing with KeyErorr
        msg = textwrap.indent(msg, '  ')
        msg = f'{e.args[0]}\n{msg}' if e.args else msg
        e.args = (msg,) + e.args[1:]
        raise

# Represent a source location; used for better error reporting
@dataclass
class Location:
    file: str
    line: int

    def __str__(self) -> str:
        return "{}:{}".format(self.file, self.line)

# A custom loader for YAML to let us also keep track of line numbers
# of each entry in the YAML file
class LineLoader(Loader):
    def construct_mapping(self, node, deep=False):  # type: ignore
        mapping = super().construct_mapping(node, deep=deep)  # type: ignore
        # Add 1 so line numbering starts at 1
        mapping['__line__'] = node.start_mark.line + 1
        return mapping

# Parse native_functions.yaml into a sequence of NativeFunctions
def parse_native_yaml(path: str) -> List[NativeFunction]:
    with open(path, 'r') as f:
        es = yaml.load(f, Loader=LineLoader)
    assert isinstance(es, list)
    rs: List[NativeFunction] = []
    for e in es:
        assert isinstance(e.get('__line__'), int), e
        loc = Location(path, e['__line__'])
        funcs = e.get('func')
        with context(f'in {loc}:\n  {funcs}'):
            rs.append(NativeFunction.from_yaml(e, loc))
    return rs

native_functions = parse_native_yaml('aten/src/ATen/native/native_functions.yaml')
# pprint.pprint([dataclasses.asdict(f) for f in native_functions])

# TODO: TensorOptions argument detection
# TODO: Extra enforcement of inplace functions having mutable self

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           CODE GENERATION
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# This is not fully constructed yet, but eventually this will actually
# do the code generation

TEMPLATE_PATH = "aten/src/ATen/templates"
TYPE_DERIVED_CPP = CodeTemplate.from_file(TEMPLATE_PATH + "/TypeDerived.h")

# When we interpret things as C++ types, there are a bunch of
# different modalities we have to consider
#
# - Return versus argument type
# - Mutable type (inplace, out argument)
# - Public API versus internal calling convention versus legacy calling
#   convention
#
# I'm not really sure how to structure this logic yet, but here is a
# sketch.  This function is ONLY correct for CPUType.h at the moment;
# I bet I am going to need another parameter before I'm done
def cpp_type(t: Type, *, mutable: bool, argument: bool) -> str:
    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            if mutable:
                return 'Tensor &'
            else:
                if argument:
                    return 'const Tensor &'
                else:
                    return 'Tensor'
        elif t.name == BaseTy.int:
            return 'int64_t'
        elif t.name == BaseTy.float:
            return 'double'
        elif t.name == BaseTy.str:
            return 'std::string'
        elif t.name in [BaseTy.bool, BaseTy.QScheme, BaseTy.Scalar,
                BaseTy.ScalarType, BaseTy.Generator, BaseTy.Storage,
                BaseTy.Layout, BaseTy.Device, BaseTy.MemoryFormat]:
            # These C++ names coincidentally line up with their schema
            # names
            return t.name.name
        else:
            assert False, f"unsupported type: {t}"
    elif isinstance(t, OptionalType):
        # TODO: these arguments are smoothed over by the hacky wrapper
        if str(t.elem) == 'Tensor' and argument:
            if mutable:
                return 'Tensor &'
            else:
                return 'const Tensor &'
        elem = cpp_type(t.elem, mutable=mutable, argument=argument)
        return f"c10::optional<{elem}>"
    elif isinstance(t, ListType):
        if str(t.elem) == 'bool':
            assert t.size is not None
            return f"std::array<bool,{t.size}>"
        # TODO: remove this special case
        if str(t.elem) == 'int' and argument:
            return f"IntArrayRef"
        elif str(t.elem) == 'Tensor' and argument:
            return f"TensorList"
        elem = cpp_type(t.elem, mutable=mutable, argument=argument)
        if argument:
            # TODO: explicitly qualify namespace here
            return f"ArrayRef<{elem}>"
        else:
            assert t.size is None, f"fixed size list returns not supported: {t}"
            return f"std::vector<{elem}>"
    else:
        assert False

def cpp_type_return(rs: Sequence[Argument]) -> str:
    if len(rs) == 0:
        return 'void'
    elif len(rs) == 1:
        return cpp_type(rs[0].type, mutable=rs[0].is_write, argument=False)
    else:
        args = ','.join([cpp_type(r.type, mutable=r.is_write, argument=False) for r in rs])
        return f'std::tuple<{args}>'

# Some simple code to exercise some of the functions we've been building
type_derived_method_declarations: List[str] = []
for f in native_functions:
    if f.dispatch is None or 'CPU' not in f.dispatch:
        continue

    name = str(f.func.name.name)
    # TODO: delete this!
    if f.func.is_out_fn():
        name += '_out'
    if f.func.name.overload_name:
        name += f'_{f.func.name.overload_name}'

    cpp_return = cpp_type_return(f.func.returns)

    def format_arg(a: Argument) -> str:
        return f"{cpp_type(a.type, mutable=a.is_write, argument=True)} {a.name}"
    cpp_args: List[str] = []
    cpp_args.extend(map(format_arg, f.func.out_arguments))
    cpp_args.extend(map(format_arg, f.func.arguments))

    # Discover TensorOptions
    topt_names = ['dtype', 'layout', 'device', 'pin_memory']
    kwargs = list(f.func.kwarg_only_arguments)  # short name
    i = 0
    while i < len(kwargs):
        if i <= len(kwargs) - len(topt_names) and all(kwargs[i+j].name == topt_names[j] for j in range(len(topt_names))):
            cpp_args.append('const TensorOptions & options')
            i += len(topt_names)
        else:
            cpp_args.append(format_arg(kwargs[i]))
            i += 1

    type_derived_method_declarations.append(f"{cpp_return} {name}({', '.join(cpp_args)});")

comment = "@" + "generated by aten/src/ATen/gen.py from TypeDerived.h"

env = {
    'generated_comment': comment,
    'Type': 'CPUType',
    'Generator': 'CPUGeneratorImpl',
    'Backend': 'CPU',  # TODO: rename this to DispatchKey
    'extra_cuda_headers': '',
    'legacy_th_headers': '#include <ATen/LegacyTHFunctionsCPU.h>',
    'type_derived_method_declarations': type_derived_method_declarations,
    'function_registrations': [],
}

print(TYPE_DERIVED_CPP.substitute(env))
