import itertools
from dataclasses import dataclass
from typing import Optional, Union, Sequence, Set, List, Tuple, Dict

from tools.codegen.api.types import *
import tools.codegen.api.cpp as cpp
import tools.codegen.local as local
from tools.codegen.gen import pythonify_default
from tools.codegen.model import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           Data Models
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# The Python binding codegen produces code that takes the input list of
# PyObjects, finds the matching ATen C++ function using PythonArgParser,
# converts the PyObjects into C++ types and calls the ATen C++ function:
#
# +--------+  parsing   +------------------------+  binding   +-----------------------+
# | PyObjs | ---------> | PythonArgParser Output | ---------> | Cpp Function Dispatch |
# +--------+            +------------------------+            +-----------------------+
#
# The following examples demonstrate the data models the Python binding
# codegen needs to deal with and the tasks it needs to accomplish. It
# helps understand the purpose of the new data types we introduced below.
#
#  - Function Schema (source of truth)
#
#      aten::empty.names(int[] size, *, Dimname[]? names,
#                        ScalarType? dtype=None, Layout? layout=None,
#                        Device? device=None, bool? pin_memory=None,
#                        MemoryFormat? memory_format=None) -> Tensor
#
#  - Python Signature
#
#    It's used to generate input schema string for PythonArgParser.
#    Note: TensorOptions fields are reordered and the additional
#    'requires_grad' field is added:
#
#      empty(IntArrayRef size, *, DimnameList? names,
#            MemoryFormat? memory_format=None, ScalarType dtype=None,
#            Layout layout=torch.strided, Device device=None,
#            bool pin_memory=False, bool requires_grad=False)
#
#  - C++ Signature
#
#    It's used to generate C++ lambda formals & dispatch call.
#    Note: the scattered TensorOptions fields are packed into 'options'.
#
#      auto dispatch_empty =
#          [](IntArrayRef size, c10::optional<DimnameList> names,
#             const TensorOptions & options,
#             c10::optional<MemoryFormat> memory_format) -> Tensor {
#          pybind11::gil_scoped_release no_gil;
#          return torch::empty(size, names, options, memory_format);
#      };
#
#  - Binding between Python Arguments and C++ Arguments
#
#    Given a set of Python Arguments in scope, we need produce the
#    binding expressions that translate the Python API into C++ API:
#
#            Python Args               Cpp Args       Binding Exprs
#     -----------------------------------------------------------------
#         0: size                      size           '_r.intlist(0)'
#         1: names                     names          'names' [special init]
#         2: memory_format -------+
#         3: dtype         -----+-|--> options        'options' [special packing]
#         4: layout            /  |
#         5: device           /   +--> memory_format  '_r.memoryformatOptional(2)'
#         6: pin_memory      /
#         7: requires_grad -+
#
#    So the full dispatch expression would look like:
#
#      dispatch_empty(_r.intlist(0), names, options,
#                     _r.memoryformatOptional(2))
#
#    Where does 'names' come from? It involves special local init:
#
#      auto __names = _r.toDimnameListOptional(1);
#      c10::optional<DimnameList> names =
#          __names ? c10::make_optional(DimnameList(__names.value()))
#                  : c10::nullopt;
#
#    Where does 'options' come from? It involves special local init
#    for TensorOptions. Note that Python side has the additional
#    'requires_grad' field:
#
#      const auto options = TensorOptions()
#          .dtype(_r.scalartype(3))
#          .device(_r.device(5))
#          .layout(_r.layoutOptional(4))
#          .requires_grad(_r.toBool(7))
#          .pinned_memory(_r.toBool(6));
#
#    In some other cases one Python Argument can map to multiple C++
#    Arguments. For example:
#
#     aten::max.names_dim(Tensor self, Dimname dim, bool keepdim=False)
#       -> (Tensor values, Tensor indices)
#
#            Python Args               Cpp Args          Binding Exprs
#     ---------------------------------------------------------------------
#                               +----> max               'out[0]'
#                              /-----> max_values        'out[1]
#         0: input            /        self              '_r.tensor(0)'
#         1: dim             /         dim               '_r.dimname(1)'
#         2: keepdim        /          keepdim           '_r.toBool(2)'
#         3: out      -----+           [local init] out  '_r.tensorlist_n<2>(3)'
#
#    As demonstrated above, the binding can involve reordering,
#    packing, unpacking and special local inits.
#
#
#  Let's look at a concrete example:
#
#      static PythonArgParser parser({
#        "abs(Tensor input, *, Tensor out=None)",
#        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         ^
#         +--- Python Schema, represented by PythonSignature and PythonArgument
#
#      }, /*traceable=*/true);
#
#      ParsedArgs<2> parsed_args;
#      auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
#
#      ...
#
#      if (_r.isNone(1)) {
#          ~~~~~~~~~~~~  <--- Scattered PythonArgParser output (arg name = 'out')
#                             represented by PythonArgParserOutputExpr
#
#        // aten::abs(Tensor self) -> Tensor
#        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         ^
#         +--- NativeFunction schema, base version
#
#        auto dispatch_abs = [](const Tensor & self) -> Tensor {
#                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                             ^
#                             +--- dispatch_lambda_args / dispatch_lambda_return_str
#                                  generated from NativeFunction / CppSignature
#                                  (deprecated PythonSignature is special)
#                                  arguments are represented by DispatchLambdaArgument
#
#          pybind11::gil_scoped_release no_gil;
#          return self.abs();
#                 ~~~~~~~~~~~  <--- cpp_dispatch_target / cpp_dispatch_exprs
#                                   generated from NativeFunction / CppSignature
#        };
#        return wrap(dispatch_abs(_r.tensor(0)));
#                                 ~~~~~~~~~~~~~
#                                  ^
#                                  +--- dispatch_lambda_exprs
#                                       binding PythonArgParserOutputExpr (python args)
#                                       and DispatchLambdaArgument (c++ args)
#
#      } else {
#        // aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
#        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         ^
#         +--- NativeFunction schema, out-variant
#
#        auto dispatch_abs_out = [](Tensor out, const Tensor & self) -> Tensor {
#          pybind11::gil_scoped_release no_gil;
#          return at::abs_out(out, self);
#        };
#        return wrap(dispatch_abs_out(_r.tensor(1), _r.tensor(0)));
#      }
#

@dataclass(frozen=True)
class PythonArgument:
    name: str
    type: Type
    default: Optional[str]

    # Used to generate the default init expr for some PythonArgParser outputs, e.g.:
    #
    #   _r.layoutWithDefault(3, layout_from_backend(self.options().backend())))
    #                           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                            ^
    #                            +--- default_init str
    default_init: Optional[str]

    # Compute argument formal for python argument parsing.
    # Needs to be consistent with torch/csrc/utils/python_arg_parser.h.
    def argument_str(self, *, method: bool = False) -> str:
        type_str = argument_type_str(self.type)

        # s/self/input/ outside method bindings
        # [old codegen] TODO: remove this? doesn't rename in codegen, it's just
        # for the parse string
        name = self.name
        if name == 'self' and type_str == 'Tensor' and not method:
            name = 'input'

        # add default
        if self.default is not None:
            default = {
                'nullptr': 'None',
                'c10::nullopt': 'None',
                '{}': 'None',
            }.get(self.default, self.default)
            return f'{type_str} {name}={default}'
        else:
            return f'{type_str} {name}'

@dataclass(frozen=True)
class PythonOutArgument(PythonArgument):
    # In Python signature multiple output fields are packed into one 'out' argument.
    # When binding to C++, it's first binded to a local 'out' variable:
    #   'auto out = _r.tensorlist_n<2>(2);',
    # then binded to scattered C++ output arguments as 'out[0]', 'out[1]', and etc.
    # TODO: maybe don't need keep scattered out fields for python signature?
    outputs: Tuple[PythonArgument, ...]

    @staticmethod
    def from_outputs(outputs: Tuple[PythonArgument, ...]) -> Optional['PythonOutArgument']:
        if not outputs:
            return None

        size = len(outputs)
        if size == 1:
            return PythonOutArgument(
                name=outputs[0].name,
                type=outputs[0].type,
                default='None',
                default_init=None,
                outputs=outputs,
            )
        elif size > 1:
            if any(map(lambda a: not a.type.is_tensor_like(), outputs)):
                raise RuntimeError(f'Unsupported output type: {outputs}')
            return PythonOutArgument(
                name='out',
                type=ListType(BaseType(BaseTy.Tensor), size),
                default='None',
                default_init=None,
                outputs=outputs,
            )
        raise AssertionError(r'Unexpected PythonOutArgument size')

@dataclass(frozen=True)
class PythonSignature:
    # Base operator name, without inplace/outplace suffix.
    name: str

    # Positional arguments.
    # TODO: create a dedicated SelfArgument type for 'self'?
    input_args: Tuple[PythonArgument, ...]

    # Keyword arguments excluding the 'out' argument and scattered kwargs belonging
    # to TensorOptions (dtype, layout, device, pin_memory, requires_grad, etc).
    input_kwargs: Tuple[PythonArgument, ...]

    output_args: Optional[PythonOutArgument]

    # These are scattered kwargs arguments belonging to TensorOptions.
    # When binding to C++, they are packed into a TensorOptions object 'options'.
    # It's possible that the C++ signature doesn't take TensorOptions object (e.g.
    # for out variant), in which case they will be used as scattered fields without
    # being packed into 'options'.
    # TODO: maybe create a PythonTensorOptionsArgument?
    tensor_options_args: Tuple[PythonArgument, ...]

    # method or function signature?
    method: bool

    @property
    def deprecated(self) -> bool:
        return False

    def arguments(
        self, *, skip_outputs: bool = False, skip_tensor_options: bool = False
    ) -> Tuple[Union[PythonArgument, PythonOutArgument], ...]:
        result: List[Union[PythonArgument, PythonOutArgument]] = []
        result.extend(self.input_args)
        result.extend(self.input_kwargs)
        if self.output_args is not None and not skip_outputs:
            result.append(self.output_args)
        if not skip_tensor_options:
            result.extend(self.tensor_options_args)
        return tuple(result)

    def arguments_count(self) -> int:
        return len(self.arguments())

    def output_idx(self) -> int:
        return len(self.input_args) + len(self.input_kwargs)

    # [old codegen] Compute the Python function signature for argument parsing,
    # as specified in torch/csrc/utils/python_arg_parser.h.  WARNING:
    # this is NOT the same type signature as specified by PEP 484
    # as understood by mypy; our format was independently developed
    # and has some quirks to make it more suitable specifically
    # for error parsing.
    #
    # For a translation to mypy-valid type signatures, see
    # tools/gen_pyi.py.  If you change any logic here, please
    # check that file too.
    def signature_str(self, *, skip_outputs: bool = False) -> str:
        schema_formals: List[str] = \
            list(map(lambda a: a.argument_str(method=self.method),
                     self.arguments(skip_outputs=skip_outputs)))
        positional_argc = len(self.input_args)
        if len(schema_formals) > positional_argc:
            schema_formals.insert(positional_argc, '*')

        return f'{self.name}({", ".join(schema_formals)})'

# The deprecated python signature involves some special logic, so create a
# dedicated data model to store these extra properties.
@dataclass(frozen=True)
class PythonSignatureDeprecated(PythonSignature):
    # We need keep the order of arguments in deprecated signature.
    # Particularly, method signature might have 'self' not at the beginning, e.g.:
    #   addmm(Scalar beta, Tensor self, Tensor mat1, Tensor mat2)
    # When generating lambda function signature we need follow the exact order (even for method=True):
    #   [](Scalar beta, const Tensor & self, const Tensor & mat1, const Tensor & mat2) -> Tensor
    deprecated_args_names: Tuple[str, ...]

    # The deprecated signature might miss some arguments that the corresponding
    # C++ signature expects. We need store the constant default values to pass in.
    # For example:
    #   [deprecate signature]: addmm(Scalar beta, Tensor self, Tensor mat1, Tensor mat2)
    #   [func schema]: aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
    #   [func call]: self.addmm(mat1, mat2, beta, 1)
    # We store ['self', 'mat1', 'mat2', 'beta', '1'] in this case.
    deprecated_args_exprs: Tuple[str, ...]

    @property
    def deprecated(self) -> bool:
        return True

    def signature_str(self, *, skip_outputs: bool = False) -> str:
        return PythonSignature.signature_str(self, skip_outputs=skip_outputs) + '|deprecated'

# This struct is used to hold the PythonSignature and its corresponding
# NativeFunction BEFORE grouping base and out-variant functions.
# Why not store NativeFunction in PythonSignature or construct PythonSignature
# from NativeFunction? Because they are not 1-1 mapped.
# One native function could have both deprecated and non-deprecated python
# signatures - NativeFunction doesn't contain information to construct the
# deprecated python signature.
# One python signature is used to handle both the base and the out-variant
# function - see 'PythonSignatureGroup'.
@dataclass(frozen=True)
class PythonSignatureNativeFunctionPair:
    signature: PythonSignature
    function: NativeFunction

# We merge pairs of functions with signatures that are equivalent mod
# output arguments, and use a single entry in the python_arg_parser sig
# list for both (output arguments become optional).
@dataclass(frozen=True)
class PythonSignatureGroup:
    # The signature used for Python argument parsing. The outplace signature
    # is preferred if exists, because it can be used to parse inputs for both
    # the out-place variant and the base version (with output omitted).
    signature: PythonSignature

    # The regular ATen declaration (e.g. conv2d)
    base: NativeFunction

    # The out variant (e.g. conv2d_out)
    outplace: Optional[NativeFunction]

# C++ function dispatch is wrapped in a lambda function. The lambda function
# has almost the same signature as the C++ function, only with some small
# variants - see details below.
# This data model is used to represent arguments of the lambda function
# signature.
@dataclass(frozen=True)
class DispatchLambdaArgument:
    name: str
    type_str: str
    is_out_arg: bool

# To pass PyObjects arguments to C++ function (via the lambda wrapper),
# we need first convert PyObjects into simple C++ objects. This work
# is done by PythonArgParser.
# This data model is used to represent the output of PythonArgParser.
# It has 1-1 mapping with PythonArgument in PythonSignature.
@dataclass(frozen=True)
class PythonArgParserOutputExpr:
    # argument name
    name: str

    # RHS expression to reference PythonArgParser output.
    expr: str

    # In some special cases we need create different expr, e.g.:
    # '_r.isNone(1)' instead of '_r.tensor(1)'.
    index: int

    # The python argument it maps to.
    argument: PythonArgument

    @property
    def is_none_expr(self) -> str:
        return f'_r.isNone({self.index})'

# To pass PythonArgParser output to the lambda wrapper, we need bind
# PythonArgParserOutputExpr to DispatchLambdaArgument.
# They are not always 1-1 mapped, e.g. scattered TensorOptions fields
# need be packed into a TensorOptions object, which is the argument
# that the lambda function wrapper takes.
@dataclass(frozen=True)
class DispatchLambdaArgumentExprs:
    # The exprs that provide the binding for lambda arguments, e.g.:
    #
    #   'self' -> '_r.tensor(0)'
    #   'min' -> 'out[0]' / 'min_indices' -> 'out[1]'
    #   'options' -> 'options'
    #
    # It has 1-1 mapping with DispatchLambdaArgument.
    exprs: Sequence[str]

    # Special local inits, which might introduce new variables that
    # the 'exprs' above reference, e.g.:
    #
    #   'auto out = _r.tensorlist_n<2>(2);'
    #
    inits: Sequence[str]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                          Helper Functions
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def _cpp_signature(f: NativeFunction, *, method: bool = False) -> CppSignature:
    return CppSignatureGroup.from_schema(f.func, method=method).signature

def has_tensor_options(f: NativeFunction) -> bool:
    return any(filter(lambda a: isinstance(a, TensorOptionsArguments),
                      cpp.group_arguments(f.func, method=False)))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                          Python Signature
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def argument_type_str(t: Type) -> str:
    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            return 'Tensor'
        elif t.name == BaseTy.int:
            return 'int64_t'
        elif t.name == BaseTy.float:
            return 'double'
        elif t.name == BaseTy.str:
            return 'std::string'
        elif t.name in [BaseTy.bool, BaseTy.QScheme, BaseTy.Scalar,
                        BaseTy.ScalarType, BaseTy.Generator, BaseTy.Storage,
                        BaseTy.Layout, BaseTy.Device, BaseTy.MemoryFormat,
                        BaseTy.Dimname, BaseTy.Stream, BaseTy.ConstQuantizerPtr]:
            # These python schema type names line up with their function schema names
            return t.name.name

    elif isinstance(t, OptionalType):
        elem = argument_type_str(t.elem)
        if elem == 'Layout':
            # TODO: fix this special case in PythonArgParser?
            return 'Layout'
        else:
            return f'{elem}?'

    elif isinstance(t, ListType):
        if str(t.elem) == 'bool':
            assert t.size is not None
            return f'std::array<bool,{t.size}>'
        elif str(t.elem) == 'int':
            return f'IntArrayRef[{t.size}]' if t.size is not None else 'IntArrayRef'
        elif str(t.elem) == 'Tensor':
            return f'TensorList[{t.size}]' if t.size is not None else 'TensorList'
        elif str(t.elem) == 'Tensor?':
            # TODO: clone the old codegen behavior but does it make sense?
            return 'TensorList?'
        elif str(t.elem) == 'Dimname':
            return f'DimnameList[{t.size}]' if t.size is not None else 'DimnameList'
        elem = argument_type_str(t.elem)
        return f'ArrayRef<{elem}>'

    raise RuntimeError(f'unrecognized type {repr(t)}')

def argument(cpp_arg: CppArgument) -> PythonArgument:
    a = cpp_arg.argument
    if not isinstance(a, Argument):
        # cpp's TensorOptionsArguments is ignored, we will reintroduce the
        # scattered fields in tensor_options_args.
        raise RuntimeError(f'unsupported cpp argument: \'{cpp_arg}\'')
    return PythonArgument(
        name=a.name,
        type=a.type,
        # TODO: directly translate a.default to python default
        default=str(pythonify_default(cpp.default_expr(a.default, a.type)))
        if a.default is not None else None,
        default_init=None,
    )

def signature(f: NativeFunction, *, method: bool = False) -> PythonSignature:
    # Use cpp api to gather TensorOptions fields from kwargs.
    # Always set 'method' to false as ThisArgument is not relevant - 'self'
    # is still included as regular Argument type.
    # TODO: maybe directly generate from FunctionSchema to avoid slicing back
    # into args/kwargs/outputs?
    cpp_sig = _cpp_signature(f, method=False)

    # Skip ThisArgument if this is method signature.
    # Skip TensorOptionsArguments in C++ signature. Python side TensorOptions
    # arguments are created based on different rules - see below.
    cpp_arguments = tuple(filter(lambda a: not (method and a.name == 'self') and
                                 not isinstance(a.argument, TensorOptionsArguments), cpp_sig.arguments()))

    kwarg_only_set = set(a.name for a in f.func.kwarg_only_arguments)
    out_arg_set = set(a.name for a in f.func.out_arguments)

    input_args = tuple(map(argument,
                           filter(lambda a: not (a.name in kwarg_only_set or a.name in out_arg_set), cpp_arguments)))
    input_kwargs = tuple(map(argument, filter(lambda a: a.name in kwarg_only_set, cpp_arguments)))
    outputs = tuple(map(argument, filter(lambda a: a.name in out_arg_set, cpp_arguments)))

    # Reintroduce the scattered fields of TensorOptions for Python.
    # Compared to the cpp counterpart, the python arguments have new property
    # (default_init) and a new argument 'requires_grad', which require some
    # special handlings.
    # [old codegen] TODO: because these aren't guaranteed to be 100% faithful
    # to the original versions in the yaml, this recreation is a potential
    # source of drift between eager and JIT. Pull this logic out to a shared place.

    has_tensor_input_arg = any(a.type.is_tensor_like()
                               for a in itertools.chain(f.func.arguments, f.func.kwarg_only_arguments))
    if any(a.name == 'requires_grad' for a in f.func.schema_order_arguments()):
        raise ValueError('argument named requires_grad is reserved, should not explicitly add it in the schema')

    # [old codegen] this probably won't work if one of the returns is not a tensor,
    # but it will produce a compile-time error that is obvious.
    has_tensor_return = any(r.type.is_tensor_like() for r in f.func.returns)

    name: str = cpp.name(f.func)
    is_factory_function = f.category_override == 'factory' or (has_tensor_return and not has_tensor_input_arg)
    is_like_or_new_function = f.category_override in ('new', 'like') or name.startswith('new_') or name.endswith('_like')

    tensor_options_args: List[PythonArgument] = []
    if is_factory_function or is_like_or_new_function:
        tensor_options_args.append(PythonArgument(
            name='dtype',
            type=BaseType(BaseTy.ScalarType),
            default=_dtype_default_type_hack(name),
            default_init='self.scalar_type()' if is_like_or_new_function else None,
        ))
        tensor_options_args.append(PythonArgument(
            name='layout',
            type=OptionalType(BaseType(BaseTy.Layout)),
            default='torch.strided',
            default_init='layout_from_backend(self.options().backend())' if is_like_or_new_function else None,
        ))
        tensor_options_args.append(PythonArgument(
            name='device',
            type=BaseType(BaseTy.Device),
            default='None',
            default_init='self.device()' if is_like_or_new_function else None,
        ))
        tensor_options_args.append(PythonArgument(
            name='pin_memory',
            type=BaseType(BaseTy.bool),
            default='False',
            default_init=None,
        ))
        tensor_options_args.append(PythonArgument(
            name='requires_grad',
            type=BaseType(BaseTy.bool),
            default='False',
            default_init=None,
        ))

    return PythonSignature(
        name=str(f.func.name.name),
        input_args=input_args,
        input_kwargs=input_kwargs,
        output_args=PythonOutArgument.from_outputs(outputs),
        tensor_options_args=tuple(tensor_options_args),
        method=method,
    )

# TODO blowtorch
def _dtype_default_type_hack(name: str) -> str:
    if name.startswith('randperm') or name == 'tril_indices' or name == 'triu_indices':
        return 'torch.int64'
    else:
        return 'None'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                        C++ Function Dispatch
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# This section provides APIs to generate the code that does C++ function
# dispatch. The C++ function call is wrapped by a lambda function.
# For example:
#
#    // aten::selu_(Tensor(a!) self) -> Tensor(a!)
#    auto dispatch_selu_ = [](Tensor self) -> Tensor {
#      pybind11::gil_scoped_release no_gil;
#      return at::selu_(self);
#    };
#
# The lambda function's signature follows the C++ signature in common
# cases, e.g.:
#
#   // aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
#   [](const Tensor & self, const Tensor & other, Scalar alpha) -> Tensor
#
# For out variant the 'out' argument's type is changed from 'Tensor &'
# to 'Tensor'. It's because when calling the lambda it passes in the
# PythonArgParser output '_r.tensor(3)', which is stack allocated object
# and needs to pass by value. Also see comments in 'dispatch_lambda_return_str()'.
#
#   // aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
#   [](Tensor out, const Tensor & self, const Tensor & other, Scalar alpha) -> Tensor
#
# For multi-output case it can keep using reference type because the
# PythonArgParser output has been unpacked to local variables, e.g.:
#
#   // aten::max.names_dim_max(Tensor self, Dimname dim, bool keepdim=False, *,
#   //     Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
#   [](Tensor & max, Tensor & max_values, const Tensor & self, Dimname dim, bool keepdim) -> std::tuple<Tensor,Tensor>
#
# For deprecated python signature, it should follow deprecated python arg order.
# TODO: This is to keep same byte-for-byte result as the old codegen - maybe unnecessary?

def dispatch_lambda_args(ps: PythonSignature, f: NativeFunction) -> Tuple[DispatchLambdaArgument, ...]:
    # Start with cpp arguments - dispatch lambda signature always include 'self'
    cpp_args: Sequence[CppArgument] = _cpp_signature(f, method=False).arguments()

    # Special reorder logic for deprecated python signature
    if isinstance(ps, PythonSignatureDeprecated):
        m: Dict[str, CppArgument] = dict((a.name, a) for a in cpp_args)
        # reorder according to the deprecated signature
        # ignore 'out' argument when binding to non-output function.
        ordered_args = filter(lambda n: n != 'out' or f.func.is_out_fn(),
                              ps.deprecated_args_names)
        cpp_args = list(map(lambda n: m[n], ordered_args))

    out_args: Set[str] = set(a.name for a in f.func.out_arguments)

    # Convert from cpp argument to lambda argument
    def dispatch_lambda_arg(cpp_arg: CppArgument) -> DispatchLambdaArgument:
        type_str = cpp_arg.type
        is_out_arg = cpp_arg.name in out_args
        if ps.method and cpp_arg.name == 'self':
            # For method's 'self', we can use 'Tensor &' and simply ignore mutability!
            type_str = 'Tensor &'
        else:
            # For other cases we need prevent dangling refs to temps (unless it's
            # unpacked scattered output)
            # The reason is explained in the comments above and in 'dispatch_lambda_return_str()'.
            # TODO: avoid this special handling?
            ensure_temp_safe = len(out_args) <= 1 or not is_out_arg
            if ensure_temp_safe:
                type_str = {
                    'Tensor &': 'Tensor',
                }.get(type_str, type_str)
        return DispatchLambdaArgument(
            name=cpp_arg.name,
            type_str=type_str,
            is_out_arg=is_out_arg,
        )

    return tuple(map(dispatch_lambda_arg, cpp_args))

# [old codegen] XXX: if you got here because of an assertion failure, it doesn't mean
# it's enough to just extend the list here. Before you do this, make sure
# to add an appropriate wrap() overload in torch/csrc/autograd/utils/wrap_outputs.h.
SUPPORTED_RETURN_TYPES = {
    'Tensor',
    'std::tuple<Tensor,Tensor>',
    'std::tuple<Tensor,Tensor,Tensor>',
    'std::tuple<Tensor,Tensor,Tensor,Tensor>',
    'std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>',
    'std::tuple<Tensor,Tensor,Tensor,int64_t>',
    'std::tuple<Tensor,Tensor,double,int64_t>',
    'std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t>',
    'std::tuple<Tensor,Tensor,double,Tensor,int64_t>',
    'std::tuple<double,int64_t>',
    'std::vector<Tensor>',
    'Scalar', 'bool', 'int64_t', 'void*', 'void',
    'QScheme', 'double',
    'IntArrayRef',
    'ScalarType'
}

def dispatch_lambda_return_str(f: NativeFunction) -> str:
    # [old codegen] Remove type annotation (e.g. 'Tensor' rather than 'Tensor &')
    # because the dispatch lambdas take mutable arguments *by value*, not
    # by reference. If you then return a reference to such an argument, you
    # will now have a pointer to a dangling stack entry. Not good.
    #
    # You want:
    #
    #   auto dispatch_selu_ = [](Tensor self) -> Tensor { ...; return at::selu_(self); };
    #                                            ^^^^^^
    #
    # *not*
    #
    #   auto dispatch_selu_ = [](Tensor self) -> Tensor& { ...; return at::selu_(self); };
    #                                            ^^^^^^^
    #
    # (NB: We can't make dispatch_selu_ take Tensor&, because the enclosing
    # codegen looks like dispatch_selu_(_r.tensor(0)), and you can't take a
    # mutable reference to temporary.  Maybe we could assign it to a
    # variable itself.)
    returns_without_annotation = tuple(map(lambda r: Return(r.name, r.type, None), f.func.returns))
    return_str = cpp.returns_type(returns_without_annotation)
    if return_str not in SUPPORTED_RETURN_TYPES:
        raise RuntimeError(f'{f.func.name} returns unsupported type {return_str}')
    return return_str

def cpp_dispatch_target(f: NativeFunction) -> str:
    name = cpp.name(f.func)
    if Variant.method in f.variants:
        return f'self.{name}'
    if Variant.function in f.variants:
        if has_tensor_options(f) or f.func.name.name.base.endswith('_like'):
            namespace = 'torch'
        else:
            namespace = 'at'
        return f'{namespace}::{name}'
    raise RuntimeError(f'could not dispatch, neither function nor method: {f.func}')

def cpp_dispatch_exprs(f: NativeFunction, *,
                       python_signature: Optional[PythonSignature] = None,
                       ) -> Tuple[str, ...]:
    cpp_args: Sequence[CppArgument] = _cpp_signature(f, method=False).arguments()

    exprs: Tuple[str, ...] = tuple()
    if not isinstance(python_signature, PythonSignatureDeprecated):
        # By default the exprs are consistent with the C++ signature.
        exprs = tuple(map(lambda a: a.name, cpp_args))
    else:
        # For deprecated python signature we may need fill in some constants.
        exprs = tuple(filter(lambda n: n != 'out' or f.func.is_out_fn(),
                             python_signature.deprecated_args_exprs))

    if Variant.method in f.variants:
        exprs = tuple(filter('self'.__ne__, exprs))

    return exprs

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                     Python / C++ Args Binding
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# We explicitly enumerate the PythonArgParser unpacking methods for all
# supported types. This might be more verbose than necessary, partially
# because of the irregularity of unpacking method naming, partially
# because we want to mimic the old codegen behavior - to reject
# unexpected and/or unsupported cases which the old codegen rejects.
# For certain cases it is intentionally more restrictive than necessary,
# e.g.: it doesn't accepts doublelist with definite size.
def arg_parser_unpack_method(t: Type, has_default: bool) -> str:
    if has_default and str(t) not in ('ScalarType', 'Device', 'Layout?'):
        raise RuntimeError(f'type \'{t}\' does not supported unpacking with default')

    if isinstance(t, BaseType):
        if t.name in [BaseTy.Tensor, BaseTy.Stream, BaseTy.Storage,
                      BaseTy.Scalar, BaseTy.Dimname]:
            # These unpack methods line up with their schema names
            return t.name.name.lower()
        elif t.name == BaseTy.ScalarType:
            return 'scalartypeWithDefault' if has_default else 'scalartype'
        elif t.name == BaseTy.Device:
            return 'deviceWithDefault' if has_default else 'device'
        elif t.name == BaseTy.int:
            return 'toInt64'
        elif t.name == BaseTy.bool:
            return 'toBool'
        elif t.name == BaseTy.float:
            return 'toDouble'
        elif t.name == BaseTy.str:
            return 'string'

    elif isinstance(t, OptionalType):
        if str(t.elem) == 'Tensor':
            if local.use_c10_dispatcher().dispatcher_uses_new_style():
                return 'optionalTensor'
            else:
                return 'tensor'

        elif isinstance(t.elem, BaseType):
            if t.elem.name in [BaseTy.ScalarType, BaseTy.Scalar,
                               BaseTy.int, BaseTy.bool,
                               BaseTy.float, BaseTy.str]:
                # Regular cases: append 'Optional' to elem's unpacking method
                return arg_parser_unpack_method(t.elem, False) + 'Optional'
            elif t.elem.name == BaseTy.MemoryFormat:
                return 'memoryformatOptional'
            elif t.elem.name == BaseTy.Generator:
                return 'generator'
            elif t.elem.name == BaseTy.Layout:
                return 'layoutWithDefault' if has_default else 'layoutOptional'

        elif isinstance(t.elem, ListType):
            if str(t.elem.elem) == 'int':
                # accept definite size
                return 'intlistOptional'
            elif str(t.elem) == 'float[]':
                return 'doublelistOptional'
            elif str(t.elem) == 'Dimname[]':
                return 'toDimnameListOptional'

    elif isinstance(t, ListType):
        if str(t.elem) == 'Tensor' or str(t.elem) == 'Tensor?':
            # accept and use definite size
            if t.size is not None:
                return f'tensorlist_n<{t.size}>'
            else:
                return 'tensorlist'
        elif str(t.elem) == 'Dimname':
            # accept definite size
            return 'dimnamelist'
        elif str(t.elem) == 'int':
            # accept definite size
            return 'intlist'
        elif str(t) == 'float[]':
            return 'doublelist'

    raise RuntimeError(f'type \'{t}\' is not supported by PythonArgParser')

# Return RHS expression for python argument using PythonArgParser output.
# e.g. for arg name 'foo', arg type 'bool', arg_index = 2, returns '_r.toBool(2)'
def arg_parser_output_expr(
    arg_index: int, a: PythonArgument
) -> PythonArgParserOutputExpr:
    has_default = a.default_init is not None
    unpack_method = arg_parser_unpack_method(a.type, has_default)
    default = f', {a.default_init}' if has_default else ''
    expr = f'_r.{unpack_method}({arg_index}{default})'

    return PythonArgParserOutputExpr(
        name=a.name,
        expr=expr,
        index=arg_index,
        argument=a,
    )

# Returns a map with key = arg_name and value = PythonArgParserOutputExpr.
def arg_parser_output_exprs(
    ps: PythonSignature, f: NativeFunction
) -> Dict[str, PythonArgParserOutputExpr]:
    return {e.name: e for i, a in enumerate(ps.arguments())
            for e in (arg_parser_output_expr(i, a), )}

# argument name to type for scattered tensor options fields
TENSOR_OPTIONS_FIELDS = {
    'dtype': 'ScalarType',
    'device': 'Device',
    'layout': 'Layout?',
    'pin_memory': 'bool',
    'requires_grad': 'bool',
}

# bind arg parser outputs (python args) with dispatch lambda arguments (c++ args).
def dispatch_lambda_exprs(
    ps: PythonSignature, f: NativeFunction
) -> DispatchLambdaArgumentExprs:
    # This method is to bind 'arg_parser_outputs' and 'lambda_args' by producing
    # 'inits' and 'lambda_args_exprs' for each lambda argument using arg parser
    # outputs.
    arg_parser_outputs = arg_parser_output_exprs(ps, f)
    lambda_args = dispatch_lambda_args(ps, f)
    inits: List[str] = []
    lambda_args_exprs: Dict[str, str] = dict()

    has_toptions = has_tensor_options(f)

    # 1. special inits/unpacking to provide binding exprs for lambda arguments.
    for a in ps.arguments(skip_tensor_options=True):
        name = a.name
        arg_parser_expr = arg_parser_outputs[a.name].expr

        if has_toptions and name == 'self':
            # TODO: why this needs to be special case?
            inits.extend([
                f'auto self = {arg_parser_expr};',
            ])
            lambda_args_exprs[name] = name
        elif isinstance(a, PythonOutArgument) and len(a.outputs) > 1 and f.func.is_out_fn():
            inits.extend([
                f'auto out = {arg_parser_expr};',
            ])
            for i, out_arg in enumerate(a.outputs):
                lambda_args_exprs[out_arg.name] = f'out[{i}]'
        elif str(a.type) == 'Dimname[]?':
            # [old codegen]
            # TODO: make this part of something more general, or get rid of it.
            # optional<ArrayRef<T>> are special. The PythonArgParser returns an
            # optional<vector<T>>, which cannot be implicitly converted to
            # optional<ArrayRef<T>>. One needs to unwrap the optional and rewrap.
            inits.extend([
                f'auto __{name} = {arg_parser_expr};',
                f'c10::optional<DimnameList> {name} = __{name} ? c10::make_optional(DimnameList(__{name}.value())) : c10::nullopt;',
            ])
            lambda_args_exprs[name] = name
        else:
            # default case - directly using PythonArgParser output expr
            lambda_args_exprs[name] = arg_parser_expr

    # method's self is passed directly to python binding, rather than parsed
    if ps.method:
        lambda_args_exprs['self'] = 'self'

    # 2. special packing/checking for TensorOptions.
    tensor_options_args_names = list(map(lambda a: a.name, ps.tensor_options_args))
    if has_toptions:
        if f.func.is_out_fn():
            raise RuntimeError(f'{f.func}: tensor options with output arg')
        for a in ps.tensor_options_args:
            if a.name not in TENSOR_OPTIONS_FIELDS:
                raise RuntimeError(
                    f'{f.func}: unrecognized tensor options field \'{a.name}\' in python binding arguments')
            if str(a.type) != TENSOR_OPTIONS_FIELDS.get(a.name):
                raise RuntimeError(
                    f'{f.func}: unrecognized type \'{str(a.type)}\' for tensor options field \'{a.name}\'')
        if not all(map(lambda a: a in tensor_options_args_names, TENSOR_OPTIONS_FIELDS.keys())):
            raise RuntimeError(
                f'{f.func}: incomplete tensor options args: {tensor_options_args_names}')

        inits.append(f'''\
const auto options = TensorOptions()
    .dtype({arg_parser_outputs['dtype'].expr})
    .device({arg_parser_outputs['device'].expr})
    .layout({arg_parser_outputs['layout'].expr})
    .requires_grad({arg_parser_outputs['requires_grad'].expr})
    .pinned_memory({arg_parser_outputs['pin_memory'].expr});
torch::utils::maybe_initialize_cuda(options);
''')
        lambda_args_exprs['options'] = 'options'

    # 3. special case - access scattered TensorOptions fields without packing
    # TODO: maybe move to the generator side as it's not related to binding.
    if not has_toptions and tensor_options_args_names:
        if 'dtype' in tensor_options_args_names:
            # we're an output-arg variant, check these args against output tensor
            if not f.func.is_out_fn():
                raise RuntimeError(
                    f'{f.func}: dtype in tensor_options_args without output arg')
            if not all(map(lambda a: a in tensor_options_args_names, ('layout', 'device'))):
                raise RuntimeError(
                    f'{f.func}: incomplete tensor options for output check')

            inits.append(f"""\
check_out_type_matches({arg_parser_outputs['out'].expr}, {arg_parser_outputs['dtype'].expr},
                       {arg_parser_outputs['dtype'].is_none_expr}, {arg_parser_outputs['layout'].expr},
                       {arg_parser_outputs['device'].expr}, {arg_parser_outputs['device'].is_none_expr});
""")
        # we'll set requires_grad on outgoing tensor
        if 'requires_grad' not in tensor_options_args_names:
            raise RuntimeError(
                f'{f.func}: expected "requires_grad" in tensor_options_args absent, but found [{tensor_options_args_names}]')

    return DispatchLambdaArgumentExprs(
        exprs=tuple(map(lambda a: lambda_args_exprs[a.name], lambda_args)),
        inits=inits,
    )
