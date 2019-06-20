# HEY! Trying to understand what this file does?  Read
# "what has to be done to add a Operation ..." first!

import re
from code_template import CodeTemplate

try:
    import typing  # noqa: F401
except ImportError:
    raise RuntimeError(
        'Missing build dependency: Unable to import the `typing` module. '
        'Please install it via `conda install typing` or `pip install typing`')

# flake8 doesn't take into account usages in type annotations.
from typing import Union, Set  # noqa: F401
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

try:
    from mypy_extensions import TypedDict
except ImportError:
    # Avoid the dependency on the mypy_extensions package.
    # It is required, however, for type checking.
    def TypedDict(name, attrs, total=True):  # type: ignore
        return Dict[Any, Any]

import sys
if sys.version_info[0] == 3:
    string_type = str
else:
    string_type = basestring

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# what has to be done to add a Operation ...
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TH functions are generated into at::legacy::cpu and at::legacy::cuda,
# where they can be called directly by a native function, they can be wrapped
# by a native function that handles dispatch

# Handle broadcasting for TH functions that need it
LEGACY_TH_DECLARATION_BROADCAST = CodeTemplate("""\
${return_type} ${api_name}(${type_method_formals});
""")
LEGACY_TH_DEFINITION_BROADCAST = CodeTemplate("""\
${return_type} ${api_name}(${type_method_formals}) {
#ifdef NAMEDTENSOR_ENABLED
    ${named_guard_declaration}
#endif
    ${device_guard_declaration}
    Tensor ${broadcast_returns};
    std::tie(${broadcast_returns}) = ${broadcast_function}(${broadcast_actuals}, "${api_name}");
    return ${method_prefix_derived}${api_name}(${broadcast_modified_actuals});
}
""")

LEGACY_TH_DECLARATION = CodeTemplate("""\
${return_type} ${method_prefix_derived}${api_name}(${type_method_formals});
""")
LEGACY_TH_DEFINITION = CodeTemplate("""\
${return_type} ${method_prefix_derived}${api_name}(${type_method_formals}) {
#ifdef NAMEDTENSOR_ENABLED
    ${named_guard_declaration}
#endif
    ${device_guard_declaration}
    ${type_definition_body}
}
""")
LEGACY_TH_DEFINITION_SWITCH_STATEMENT = CodeTemplate("""\
${dispatch_scalar_type_declaration}
switch (dispatch_scalar_type) {
    ${cases}
    default:
        AT_ERROR("${api_name} not supported on ${Type} for ", dispatch_scalar_type);
}
""")
LEGACY_TH_DEFINITION_CASE = CodeTemplate("""\
case ScalarType::${ScalarName}: {
    ${case_body}
    break;
}
""")

# Native functions are generated and registered on the dispatcher. We register the
# function on Backend::Undefined if it does not have backend dependent dispatch.
# In this case, it will be called for all backends, but can be overwritten on a
# per backend basis.
NATIVE_DISPATCH_DECLARATION = CodeTemplate("""\
static ${return_type} ${api_name}(${type_method_formals});
""")

NATIVE_DISPATCH_DEFINITION_DEFAULT = CodeTemplate("""\
${return_type} TypeDefault::${api_name}(${type_method_formals}) {
#ifdef NAMEDTENSOR_ENABLED
    ${named_guard_declaration}
#endif
    ${device_guard_declaration}
    ${return_call} at::native::${native_type_method_dispatch}(${native_actuals});
}
""")

NATIVE_DISPATCH_DEFINITION_BACKEND = CodeTemplate("""\
${return_type} ${Type}::${api_name}(${type_method_formals}) {
#ifdef NAMEDTENSOR_ENABLED
    ${named_guard_declaration}
#endif
    ${device_guard_declaration}
    ${return_call} at::native::${native_type_method_dispatch}(${native_actuals});
}
""")

DEFAULT_FUNCTION_REGISTRATION = CodeTemplate("""\
.registerOp<${return_type} (${formals_types})>(Backend::Undefined, "${schema_string}", &TypeDefault::${api_name})
""")
BACKEND_FUNCTION_REGISTRATION = CodeTemplate("""\
.registerOp<${return_type} (${formals_types})>(Backend::${Backend}, "${schema_string}", &${Type}::${api_name})
""")

# Generate a file that lists all functions and their schema string. Used for XLA
REGISTRATION_DECLARATION = CodeTemplate("""\
${return_type} ${api_name}(${type_method_formals}); // ${schema_string}
""")

# Overrideable stubs to be used in user-extendable backends
NATIVE_DISPATCH_DEFINITION_EXTENSION_BACKEND = CodeTemplate("""\
${return_type} ${Type}::${api_name}(${type_method_formals}) {
    return ${Type}Dispatch::get_function<${return_type} (*)(${formals_types})>("${schema}")(${native_actuals});
}
""")

# add non-virtual declaration to Tensor.h
TENSOR_METHOD_DECLARATION = CodeTemplate("""\
${return_type} ${api_name}(${method_formals_with_defaults})${const_mark};
""")
# add non-virtual declaration to Tensor.cpp
TENSOR_METHOD_DEFINITION = CodeTemplate("""\
inline ${return_type} Tensor::${api_name}(${method_formals})${const_mark} {
    static auto table = globalATenDispatch().getOpTable("${schema_string}");
    return table->getOp<${return_type} (${formals_types})>(tensorTypeIdToBackend(type_id()), is_variable())(${method_actuals});
}
""")
# add a method declaration in Functions.h
FUNCTION_DECLARATION = CodeTemplate("""\
static inline ${return_type} ${api_name}(${formals_with_defaults});
""")
# add a method declaration in Functions.h
DEPRECATED_FUNCTION_DECLARATION = CodeTemplate("""\
C10_DEPRECATED static inline ${return_type} ${api_name}(${formals_with_defaults});
""")
# add method definition in Functions.h
FUNCTION_DEFINITION = CodeTemplate("""\
static inline ${return_type} ${api_name}(${formals}) {
    static auto table = globalATenDispatch().getOpTable("${schema_string}");
    return table->getOp<${return_type} (${formals_types})>(${inferred_backend}, ${inferred_is_variable})(${native_actuals});
}
""")
# add a native declaration for a native function
NATIVE_DECLARATION = CodeTemplate("""\
CAFFE2_API ${return_type} ${native_type_method_dispatch}(${formals_with_defaults});
""")

# special method definition for factory functions in Functions.h that initializes backends
FACTORY_DEFINITION = CodeTemplate("""\
static inline ${return_type} ${api_name}(${formals}) {
    globalLegacyTypeDispatch().initForBackend(${inferred_backend});
    static auto table = globalATenDispatch().getOpTable("${schema_string}");
    return table->getOp<${return_type} (${formals_types})>(${inferred_backend}, ${inferred_is_variable})(${native_actuals});
}
""")

ZERO_DIM_CHECK = CodeTemplate("""\
if (${check_name}.dim() == 0) {
    return ${api_name}(${zero_dim_actuals});
}""")

ZERO_DIM_ONLY = CodeTemplate("""\
AT_ERROR("${api_name} only supports a 0-dimensional ${check_name} tensor, but got tensor "
    "with ", ${check_name}.dim(), " dimension(s).");
""")

SPARSE_CHECK = CodeTemplate("""\
if(${check_name}.is_sparse()) {
    return static_cast<const TypeExtendedInterface*>(this)->${api_name}(${sparse_actuals});
}""")

CONDITIONAL_INITIALIZER = CodeTemplate("""\
if (${name}.defined()) {
    ${initializer}
}""")

CALL_TEMPLATE = CodeTemplate("${cname}(${actuals})")

# scalar_name, c_type, accreal, is_floating_type
scalar_types = [
    ('Bool', 'bool', 'BoolAccrealNotDefined', False),
    ('Byte', 'uint8_t', 'Long', False),
    ('Char', 'int8_t', 'Long', False),
    ('Double', 'double', 'Double', True),
    ('Float', 'float', 'Double', True),
    ('Int', 'int', 'Long', False),
    ('Long', 'int64_t', 'Long', False),
    ('Short', 'int16_t', 'Long', False),
    ('Half', 'Half', 'Double', True),
    ('BFloat16', 'BFloat16', 'BFloat16AccrealNotDefined', True),
]


class NYIError(Exception):
    """Indicates we don't support this declaration yet"""

    __slots__ = ['reason']

    def __init__(self, reason):
        self.reason = reason


TYPE_FORMAL_GENERIC = {
    'THTensor*': 'Tensor &',
    'THByteTensor*': 'Tensor &',
    'THIndexTensor*': 'Tensor &',
    'THBoolTensor*': 'Tensor &',
    'THIntegerTensor*': 'Tensor &',
    'THDenseTensor*': 'Tensor &',
    'THDenseIndexTensor*': 'Tensor &',
    'THStorage*': 'Storage',
    'THGenerator*': 'Generator *',
    'IntArrayRefSize': 'IntArrayRef',
    'accreal': 'Scalar',
    'real': 'Scalar',
    'long': 'int64_t',
}

DYNAMIC_TYPE = {
    'THTensor*': 'Tensor',
    'THByteTensor*': 'ByteTensor',
    'THBoolTensor*': 'BoolTensor',
    'THIndexTensor*': 'IndexTensor',
    'THIntegerTensor*': 'IntegerTensor',
    'THDenseTensor*': 'Tensor',
    'THDenseIndexTensor*': 'IndexTensor',
    'THStorage*': 'Storage',
    'THGenerator*': 'Generator*',
    'IntArrayRefSize': 'IntArrayRef',
    'accreal': 'accreal',
    'real': 'real',
    'long': 'int64_t',
}

NATIVE_DYNAMIC_TYPE = {
    'Tensor &': 'Tensor',
    'const Tensor &': 'Tensor',
}

TYPE_RETURN = {
    'THTensor*': 'Tensor',
    'THIndexTensor*': 'Tensor',
    'THByteTensor*': 'Tensor',
    'THBoolTensor*': 'Tensor',
    'THIntegerTensor*': 'Tensor',
    'THDenseTensor*': 'Tensor',
    'THDenseIndexTensor*': 'Tensor',
    'real': 'Tensor',
    'accreal': 'Tensor',
    'long': 'int64_t',
}

CHECKED_CAST = {
    'THTensor*':
        CodeTemplate(
            'checked_tensor_unwrap('
            '${arg_name},"${arg_name}",${arg_pos}, ${null_okay}, '
            'Backend::${Backend}, ScalarType::${ScalarName})'),
    'THByteTensor*':
        CodeTemplate(
            'checked_tensor_unwrap('
            '${arg_name},"${arg_name}",${arg_pos}, ${null_okay}, '
            'Backend::${Backend}, ScalarType::Byte)'),
    'THBoolTensor*':
        CodeTemplate(
            'checked_tensor_unwrap('
            '${arg_name},"${arg_name}",${arg_pos}, ${null_okay}, '
            'Backend::${Backend}, ScalarType::Bool)'),
    'THIndexTensor*':
        CodeTemplate(
            'checked_tensor_unwrap('
            '${arg_name},"${arg_name}",${arg_pos}, ${null_okay}, '
            'Backend::${Backend}, ScalarType::Long)'),
    'THIntegerTensor*':
        CodeTemplate(
            'checked_tensor_unwrap('
            '${arg_name},"${arg_name}",${arg_pos}, ${null_okay}, '
            'Backend::${Backend}, ScalarType::Int)'),
    'THDenseTensor*':
        CodeTemplate(
            'checked_tensor_unwrap('
            '${arg_name},"${arg_name}",${arg_pos}, ${null_okay}, '
            'Backend::${DenseBackend}, ScalarType::${ScalarName})'),
    'THDenseIndexTensor*':
        CodeTemplate(
            'checked_tensor_unwrap('
            '${arg_name},"${arg_name}",${arg_pos}, ${null_okay}, '
            'Backend::${DenseBackend}, ScalarType::Long)'),
    'THStorage*':
        CodeTemplate(
            'checked_storage('
            '${arg_name},"${arg_name}",${arg_pos}, '
            # We're punning here (Backend and DeviceType constructors coincide)
            # but DeviceType is the correct way to classify storages
            'DeviceType::${Backend}, at::scalarTypeToTypeMeta(ScalarType::${ScalarName}))'),
    # This is a cast done via direct-construction
    'IntArrayRefStride': CodeTemplate('at::IntArrayRef ${result_name} = get_intlist_stride_th(${arg_name});'),
    'real': CodeTemplate('${arg_name}.to${ScalarName}()'),
    'accreal': CodeTemplate('${arg_name}.to${AccScalarName}()'),
    'TensorList': CodeTemplate(
            'checked_tensor_list_unwrap(${arg_name},"${arg_name}",${arg_pos}, '
            'Backend::${Backend}, ScalarType::${ScalarName})'),
    'IntArrayRef': CodeTemplate('check_intlist<${size}>(${arg_name}, "${arg_name}", ${arg_pos}${,default_init})')
}

CHECKED_USE = {
    'THTensor*': '{}_',
    'THIndexTensor*': '{}_',
    'THByteTensor*': '{}_',
    'THBoolTensor*': '{}_',
    'THIntegerTensor*': '{}_',
    'THDenseTensor*': '{}_',
    'THDenseIndexTensor*': '{}_',
    'THStorage*': '{}_.unsafeGetStorageImpl()',
    'TensorList': "{0}_.data(), {0}_.size()",
}

CHECKED_USE_NULLABLE = CodeTemplate('${arg_name}_ ? ${usage} : NULL')

ALLOC_NOARGS_WRAP = {
    'THTensor*': 'c10::make_intrusive<TensorImpl, UndefinedTensorImpl>'
                 '(c10::Storage(caffe2::TypeMeta::Make<${ScalarType}>(), 0, allocator(), true),'
                 '${Backend}TensorId()).release()',
    'THByteTensor*': 'c10::make_intrusive<TensorImpl, UndefinedTensorImpl>'
                     '(c10::Storage(scalarTypeToTypeMeta(ScalarType::Byte), 0, allocator(), true),'
                     '${Backend}TensorId()).release()',
    'THBoolTensor*': 'c10::make_intrusive<TensorImpl, UndefinedTensorImpl>'
                     '(c10::Storage(scalarTypeToTypeMeta(ScalarType::Bool), 0, allocator(), true),'
                     '${Backend}TensorId()).release()',
    'THIndexTensor*': 'c10::make_intrusive<TensorImpl, UndefinedTensorImpl>'
                     '(c10::Storage(scalarTypeToTypeMeta(ScalarType::Long), 0, allocator(), true),'
                     '${Backend}TensorId()).release()',
    'THIntegerTensor*': 'c10::make_intrusive<TensorImpl, UndefinedTensorImpl>'
                        '(c10::Storage(scalarTypeToTypeMeta(ScalarType::Int), 0, allocator(), true),'
                        '${Backend}TensorId()).release()',
}

ALLOC_WRAP = {
    'THTensor*': '${arguments}',
    'THByteTensor*': '${arguments}',
    'THBoolTensor*': '${arguments}',
    'THIndexTensor*': '${arguments}',
    'THIntegerTensor*': '${arguments}',
    'THDenseTensor*': '${arguments}',
    'THDenseIndexTensor*': '${arguments}',
}

# Replacements for constants when calling into TH
CONSTANT_REPLACEMENTS = [
    ('AS_REAL', '${ScalarType}'),
    ('__last_dim', 'self.ndimension()-1'),
]

# Replacements for constants in header file function definitions
HEADER_CONSTANT_REPLACEMENTS = [
    (r'AS_REAL\((.*)\)', r'\1'),
    ('__last_dim', '-1'),
]


class nested_dict(object):
    def __init__(self, base, parent):
        self.base, self.parent = base, parent

    def __getitem__(self, x):
        r = self.base.get(x)
        if r is not None:
            return r
        return self.parent[x]


Environment = TypedDict('Environment', {
    'state': str,
    'ScalarType': str,
    'ScalarName': str,
    'THTensor': str,
    'THType': str,
    'Backend': str,
    'DeviceType': str,
    'AccScalarName': str,
})

TopEnvironment = TypedDict('TopEnvironment', {
    'type_registrations': List[str],
    'type_headers': List[str],
    'function_registrations': List[str],
    'type_method_declarations': List[str],
    'type_method_definitions': List[str],
    'tensor_method_declarations': List[str],
    'tensor_method_definitions': List[str],
    'function_declarations': List[str],
    'function_definitions': List[str],
    'type_ids': List[str],
    'native_function_declarations': List[str],
    'registration_declarations': List[str],
})

# A Declarations.cwrap formal argument
# type can contain THTensor* types
THFormal = TypedDict('THFormal', {
    'name': str,
    'type': str,
    'dynamic_type': str,
    'kwarg_only': bool,
    'is_nullable': bool,
    'default': str,
    'default_init': str,
    'output': bool,
    'size': int,
    'declared_type': str,
    'ignore_check': bool,
    'allocate': bool,
    'mask': bool,
    'if_true': bool,
    'if_false': bool,
    'wrap_dim': str,
    # Broadcast is originally a str but gets unwrapped to a List or Dict in-place
    'broadcast': Any,
    'resize': str,
    'cpu_zero': bool,
    'zero': bool,
}, total=False)

# Generic ATen formal or native_functions.yaml formal argument.
# type can contain Tensor& reference types.
AtFormal = TypedDict('AtFormal', {
    'name': str,
    'type': str,
    'dynamic_type': str,
    'kwarg_only': bool,
    'is_nullable': bool,
    'default': str,
    'default_init': str,
    'output': bool,
    'size': int,
}, total=False)

# Note [field_name versus name]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# What is the difference between "field_name" and "name"?
#
# Return values of ATen operators always have a name: if it is not
# explicitly assigned a name inside native_functions.yaml like func:
# myop() -> (Tensor indices, Tensor value), then the codegen will
# automatically assign it a name like result0, or name might be
# specified inside Declarations.cwrap.  We don't want these assigned
# names to become part of the public API when we return a namedtuple for
# any such multiple-return function.
#
# Thus field_name is like name, but it is defined only when there is a
# name specified in native_functions.yaml. If field_name is defined,
# then the codegen would generate code to return namedtuple. Otherwise,
# it would just return tuple.

ReturnType = TypedDict('ReturnType', {
    'name': str,
    # See Note [field_name versus name]
    'field_name': str,
    'type': str,
    'dynamic_type': str,
}, total=False)

ReturnDecl = TypedDict('ReturnDecl', {
    'kind': str,
    'type': str,
    'arguments': List[int],
}, total=False)

# Represents a buffer in nn.yaml
NNBuffer = TypedDict('NNBuffer', {
    'name': str,
})

FunctionOption = TypedDict('FunctionOption', {
    'actuals': List[str],
    'api_name': str,
    'arguments': List[THFormal],
    'aten_custom_call': str,
    'backend_types': Dict[str, List[str]],
    'backends': List[str],
    'broadcast_actuals': List[str],
    'broadcast_function': str,
    'broadcast_modified_actuals': List[str],
    'broadcast_returns': List[str],
    'buffers': List[NNBuffer],
    # cimpls is really a List[FunctionOption]
    'cimpls': List[Any],
    'cname': str,
    'condition': str,
    'const_mark': str,
    'device_guard': bool,
    'device_guard_declaration': str,
    'dispatch_scalar_type_declaration': str,
    'with_gil': bool,
    'cpu_half': bool,
    'cpu_bfloat16': bool,
    'deprecated': bool,
    'cpu_bool': bool,
    'cuda_bool': bool,
    # See Note [field_name versus name]
    'field_name': str,
    'formals_list': List[AtFormal],
    'formals_with_defaults': List[str],
    'formals': List[str],
    'formals_types': List[str],
    'inferred_backend': str,
    'inferred_is_variable': str,
    'inplace': bool,
    'matches_jit_signature': bool,
    # This controls whether or not we generate the interface in Type or
    # TypeExtendedInterface
    'extended_method': bool,
    'method_actuals': List[str],
    'method_formals_with_defaults': List[str],
    'method_formals': List[str],
    'method_prefix_derived': str,
    'named_guard_declaration': str,
    'mode': str,
    'python_module': str,
    'name': str,
    'native_actuals': List[str],
    'native_type_method_dispatch': str,
    # options should be List[FunctionOption]
    'options': Any,
    'schema_string': str,
    'requires_tensor': bool,
    'return_call': str,
    'return_type': str,
    'return': ReturnDecl,
    'returns': List[ReturnType],
    'scalar_check': str,
    # schema used for extension backend operator registration
    'schema': str,
    'sparse': bool,
    'type_definition_body': List[str],
    'type_method_actuals': List[str],
    'type_method_definition_dispatch': str,
    'type_method_formals': List[str],
    'variants': str,
    'when_spares_dispatch': str,
    'when_sparse_dispatch': str,
    'with_gil': bool,
    'zero_dim_dispatch_when_scalar': str,
    'zero_dim_tensor_only': bool,
})

OutputDeclaration = NamedTuple('OutputDeclaration', [
    ('name', str),
    ('matches_jit_signature', bool),
    ('schema_string', str),
    ('method_prefix_derived', str),
    ('arguments', List[AtFormal]),
    ('method_of', List[str]),
    ('mode', str),
    ('python_module', str),
    ('buffers', Optional[List[str]]),
    ('returns', List[ReturnType]),
    ('inplace', bool),
    ('is_factory_method', bool),
    ('abstract', bool),
    ('requires_tensor', bool),
    ('device_guard', bool),
    ('with_gil', bool),
    ('deprecated', bool),
])


def device_guard(option, dispatch_options, dispatch_tensor):
    # For factory methods the `DeviceGuard` is already in the template.
    if option.get('device_guard', True):
        if dispatch_options:
            return 'const DeviceGuard device_guard({}.device());'.format(dispatch_options['name'])
        if dispatch_tensor:
            return 'const OptionalDeviceGuard device_guard(device_of({}));'.format(dispatch_tensor)
    return '// DeviceGuard omitted'


def named_guard(option, tensors, tensorlists):
    if not option.get('named_guard', True) or (len(tensors) + len(tensorlists) == 0):
        return ''
    named_conditions = []
    for tensor in tensors:
        named_conditions.append('{}.is_named()'.format(tensor))
    for tensorlist in tensorlists:
        named_conditions.append('at::has_names({})'.format(tensorlist))
    return ("""\
if ({named_conditions}) {{
    AT_ERROR("{op}: no named inference rule implemented.");
}}""".format(named_conditions=' || '.join(named_conditions), op=option['name']))


def dispatch_scalar_type(option, dispatch_options, dispatch_tensor):
    if dispatch_options:
        return 'auto dispatch_scalar_type = typeMetaToScalarType({}.dtype());'.format(dispatch_options['name'])
    if dispatch_tensor:
        return 'auto dispatch_scalar_type = infer_scalar_type({});'.format(dispatch_tensor)
    return '// dispatch_scalar_type omitted'


def is_real_argument_to_wrapper(argument):
    # type: (THFormal) -> bool
    return not argument.get('output', False) and\
        argument['type'] != 'CONSTANT' and\
        argument['type'] != 'argument'


def is_mutable_formal_argument(argument, option):
    # type: (THFormal, FunctionOption) -> bool
    return argument.get('output') or option['inplace'] and argument['name'] == 'self'


def check_methods_do_not_start_with_underscore(name, is_method):
    if name in {'_values', '_indices', '_nnz', '_dimI', '_dimV', '_coalesced_'}:
        return
    if is_method and name.startswith('_') and not name.startswith('__') and not name.startswith('_th_'):
        message = "Function '{}' starts with a single underscore and is ".format(name)
        message += "configured to have a method on Tensor. Functions that start with "
        message += " a single underscore should only be functions in the at:: "
        message += "namespace and not methods on Tensor!"
        raise RuntimeError(message)


def to_return_type(arg, option):
    # type: (THFormal, FunctionOption) -> ReturnType
    t = arg['type']
    rt = TYPE_RETURN.get(t, t)
    if rt == 'Tensor' and not arg.get('allocate'):
        rt = rt + ' &'
        if not is_mutable_formal_argument(arg, option):
            rt = 'const ' + rt
    return {
        'name': arg['name'],
        'type': rt,
        'dynamic_type': DYNAMIC_TYPE.get(arg['type'], arg['type']),
    }


def create_generic(top_env, declarations):
    # type: (TopEnvironment, List[FunctionOption]) -> List[OutputDeclaration]
    # translates defaults from cwrap types to C++ values
    def translate_default(argument, type_str, default):
        # type: (THFormal, str, Any) -> Any
        if default is None:
            # cause the default constructor for the object to run
            return '{}'
        if 'if_true' in argument:
            return argument['default'] == argument['if_true']
        for pattern, replacement in HEADER_CONSTANT_REPLACEMENTS:
            default = re.sub(pattern, replacement, str(default))
        if type_str in {'Scalar', 'int64_t', 'double'}:
            try:
                return int(default)
            except Exception:
                try:
                    return float(default)
                except Exception:
                    return default
        elif type_str == 'bool':
            assert default.lower() in ['true', 'false']
            return default.lower() == 'true'
        else:
            return default

    # change from THTensor* to Tensor & so we get how it will appear
    # in the aten argument list...
    def translate_formal(argument, option):
        # type: (THFormal, FunctionOption) -> AtFormal
        type_str = TYPE_FORMAL_GENERIC.get(argument['type'], argument['type'])
        if type_str == 'Tensor &' and not is_mutable_formal_argument(argument, option):
            type_str = 'const ' + type_str
        translated = {
            'name': argument['name'],
            'type': type_str,
            'dynamic_type': DYNAMIC_TYPE.get(argument['type'], argument['type']),
        }  # type: AtFormal
        if 'kwarg_only' in argument:
            translated['kwarg_only'] = argument['kwarg_only']
        if 'default' in argument:
            default = translate_default(argument, type_str, argument['default'])
            translated['default'] = default
            translated['default_init'] = argument.get('default_init', default)
        if argument.get('output'):
            translated['output'] = True
        if argument.get('size'):
            translated['size'] = argument['size']
        if argument.get('is_nullable') is not None:
            translated['is_nullable'] = argument['is_nullable']
        return translated

    def get_formals(option, include_constants=False):
        # type: (FunctionOption, bool) -> List[AtFormal]
        seen = set()  # type: Set[str]
        pos_args = []  # type: List[THFormal]
        kwd_args = []  # type: List[THFormal]

        def insert(argument):
            # type: (THFormal) -> None
            if argument['name'] not in seen:
                seen.add(argument['name'])
                if argument.get('kwarg_only', False):
                    kwd_args.append(argument)
                else:
                    pos_args.append(argument)

        def has_output_mask(argument):
            # type: (THFormal) -> bool
            return argument.get('allocate', False) and argument.get('mask', False)

        for argument in option['arguments']:
            if argument.get('output') and not argument.get('allocate', False):
                insert(argument)
        for argument in option['arguments']:
            if include_constants and argument['type'] == 'CONSTANT':
                insert(argument)
            elif is_real_argument_to_wrapper(argument):
                insert(argument)
        if any(has_output_mask(arg) for arg in option['arguments']):
            mask_size = sum(has_output_mask(arg) for arg in option['arguments'])
            insert({
                'name': 'output_mask',
                # NB: Lack of space in comma works around parsing
                # problem in gen_variable_type.py
                'type': 'std::array<bool,{}>'.format(mask_size),
                'default': '{{' + ', '.join(['true'] * mask_size) + '}}',
            })

        result = pos_args + kwd_args
        return [translate_formal(argument, option) for argument in result]

    def get_return_types(option):
        # type: (FunctionOption) -> List[ReturnType]
        ret = option['return']
        if ret['kind'] == 'arguments':
            argument_indices = ret['arguments']
            if len(argument_indices) == 1:
                the_arg = option['arguments'][argument_indices[0]]
                return [to_return_type(the_arg, option)]
            else:
                return [to_return_type(option['arguments'][idx], option)
                        for idx in argument_indices]
        elif ret['kind'] == 'type':
            return [{
                'type': TYPE_RETURN.get(ret['type'], ret['type']),
                'dynamic_type': DYNAMIC_TYPE.get(ret['type'], ret['type']),
            }]
        else:
            raise Exception("format_return_type")

    def format_return_type(return_types):
        # type: (List[ReturnType]) -> str
        if len(return_types) == 1:
            return return_types[0]['type']
        return "std::tuple<{}>".format(','.join(r['type'] for r in return_types))

    def is_any_tensor_type(formal):
        return (formal['dynamic_type'] == 'Tensor' or formal['dynamic_type'] == 'ByteTensor'
                or formal['dynamic_type'] == 'IndexTensor' or formal['dynamic_type'] == 'BoolTensor')

    def find_tensors(formals):
        # type: (List[AtFormal]) -> List[str]
        return [formal['name'] for formal in formals if is_any_tensor_type(formal)]

    def find_tensorlists(formals):
        # type: (List[AtFormal]) -> List[str]
        return [formal['name'] for formal in formals if formal['dynamic_type'] == 'TensorList']

    def find_dispatch_tensor(formals):
        # type: (List[AtFormal]) -> Optional[str]
        # dispatch to self if it's a parameter

        for formal in formals:
            if formal['name'] == 'self' and is_any_tensor_type(formal) and not formal.get('is_nullable', False):
                return formal['name']
        # otherwise dispatch to the first Tensor or TensorList
        for formal in formals:
            if 'TensorList' == formal['dynamic_type'] or is_any_tensor_type(formal) and \
               not formal.get('is_nullable', False):
                return formal['name']

        return None

    def format_formal(f):
        # type: (AtFormal) -> str
        return '{} {}'.format(f['type'], f['name'])

    def formal_with_default(f):
        # type: (AtFormal) -> str
        s = format_formal(f)
        v = f.get('default')
        if v is None:
            return s
        if isinstance(v, bool):
            v = str(v).lower()
        return '{}={}'.format(s, v)

    def get_broadcast_argument(option):
        # type: (FunctionOption) -> Optional[THFormal]
        for argument in option['arguments']:
            if argument.get('broadcast'):
                return argument
        return None

    def get_broadcast_actuals(broadcast_arg, broadcast_inplace, broadcast_dims):
        # type: (THFormal, bool, bool) -> List[str]
        # Note: broadcast_dims can change type...
        # return the actuals that will be passed to the broadcast function.
        # 1) in the common case, this is the broadcasted argument (e.g. "self") followed by the tensors
        #    that it is broadcasted against (comma-separated) (e.g. "self, tensor1, tensor2").
        # 2) in the broadcast_dims case, this is the broadcasted argument (e.g. "self") followed by the sizes
        #    it is broadcasted to (as an initializer list), so e.g. the specification
        #    "mat1.dim0,mat2.dim1" gets transformed to "self, {mat1.size(0),mat2.size(1)}"
        if not broadcast_dims:
            broadcast_actuals = [broadcast_arg['name']] + broadcast_arg['broadcast'].split()[0].split(",")
        else:
            broadcast_dims_spec = broadcast_arg['broadcast'].split()[1].split(':')[1].split(',')
            # generate size call for each dimension
            broadcast_dims = ([x.split('.')[0] + '.size(' + x.split('.')[1].replace('dim', '') + ')'  # type: ignore
                              for x in broadcast_dims_spec])
            broadcast_dims_init_list = '{' + ','.join(broadcast_dims) + '}'  # type: ignore
            broadcast_actuals = [broadcast_arg['name'], broadcast_dims_init_list]

        return broadcast_actuals

    def emit_nn_body(option):
        # type: (FunctionOption) -> Union[str, List[str]]
        # Concrete definition on Type.cpp for NN functions. Delegates to the
        # xxx_forward variant variant after creating any necessary buffers.
        actuals = option['actuals']
        base_name = option['name'][:-1] if option['inplace'] else option['name']
        fwd_name = option['api_name'].replace(base_name, base_name + '_forward')

        if len(option['buffers']) == 0:
            return 'return {}({});'.format(fwd_name, ', '.join(actuals))

        body = []  # type: List[str]
        if option['api_name'].endswith('_out'):
            # _out variants must create buffers and insert them in the
            # arguments list between output and input arguments
            for buffer in option['buffers']:
                body.append('Tensor {} = at::empty({{0}}, this->options());'.format(buffer['name']))
            actuals = [arg['name'] for arg in option['arguments'] if arg.get('output')]
            actuals += [buffer['name'] for buffer in option['buffers']]
            actuals += [arg['name'] for arg in option['arguments'] if not arg.get('output')]

        body.append('return std::get<0>({}({}));'.format(fwd_name, ', '.join(actuals)))
        return body

    def process_option(option, output_options):
        # type: (FunctionOption, List[OutputDeclaration]) -> None
        option['inplace'] = re.search(
            '(^__i|[^_]_$)', option['api_name']) is not None

        # print(yaml.dump(option))
        formals = get_formals(option)
        option['formals_list'] = formals
        option['formals'] = [format_formal(f) for f in formals]
        option['formals_with_defaults'] = [formal_with_default(f) for f in formals]
        option['returns'] = get_return_types(option)
        option['return_type'] = format_return_type(option['returns'])
        option['return_call'] = 'return ' if option['return_type'] != 'void' else ''
        option['actuals'] = [f['name'] for f in formals]

        option['method_formals'] = [format_formal(f) for f in formals
                                    if f['name'] != 'self']
        option['method_formals_with_defaults'] = (
            [formal_with_default(f) for f in formals if f['name'] != 'self'])
        option['method_actuals'] = [
            f['name'] if f['name'] != 'self' else '*this' for f in formals]

        # There are no cases where these differ, but they do in native_functions
        option['type_method_formals'] = option['formals']
        option['type_method_actuals'] = option['actuals']

        option['const_mark'] = '' if option['inplace'] else ' const'

        assert 'method' not in option['variants'], 'TH functions cannot be methods'
        is_function = 'function' in option['variants']
        dispatch_tensor = find_dispatch_tensor(formals)
        is_namespace_function = is_function and dispatch_tensor is not None

        broadcast_arg = get_broadcast_argument(option)
        # "s_" for "same size".
        option['method_prefix_derived'] = '' if broadcast_arg is None else 's_'
        if option['mode'] == 'TH':
            option['device_guard'] = False
        option['device_guard_declaration'] = device_guard(option, False, dispatch_tensor)
        option['named_guard_declaration'] = named_guard(option, find_tensors(formals),
                                                        find_tensorlists(formals))
        option['dispatch_scalar_type_declaration'] = dispatch_scalar_type(option, False, dispatch_tensor)

        assert option['extended_method'], 'Expected legacy operator to be an extended method'

        if broadcast_arg is not None:
            broadcast_inplace = 'inplace' in broadcast_arg['broadcast']
            broadcast_dims = 'dims:' in broadcast_arg['broadcast']
            option['broadcast_actuals'] = get_broadcast_actuals(broadcast_arg, broadcast_inplace, broadcast_dims)
            if not broadcast_dims:
                option['broadcast_returns'] = (["b_" + x for x in option['broadcast_actuals']
                                               if x != broadcast_arg['name'] or not broadcast_inplace])
            else:
                option['broadcast_returns'] = ["b_" + broadcast_arg['name']]

            option['broadcast_function'] = 'expand_' + ('inplace' if broadcast_inplace
                                                        else 'size' if broadcast_dims else 'outplace')
            option['broadcast_modified_actuals'] = ['b_' + y if 'b_' + y in option['broadcast_returns'] else y
                                                    for y in option['actuals']]

    def native_get_formals(option, include_constants=False):
        # type: (FunctionOption, bool) -> List[AtFormal]
        seen = set()  # type: Set[str]
        pos_args = []
        kwd_args = []

        def insert(argument):
            # type: (AtFormal) -> None
            if argument['name'] not in seen:
                seen.add(argument['name'])
                if argument.get('kwarg_only', False):
                    kwd_args.append(argument)
                else:
                    pos_args.append(argument)

        for argument in option['arguments']:
            insert(argument)

        # not clear we need dynamic_type translation as we can specify the correct type
        # directly in native functions
        def add_dynamic_type(argument, option):
            # type: (AtFormal, FunctionOption) -> AtFormal
            argument['dynamic_type'] = NATIVE_DYNAMIC_TYPE.get(argument['type'], argument['type'])
            return argument

        result = pos_args + kwd_args
        result = [add_dynamic_type(argument, option) for argument in result]

        # ensure we get reference-type formals when appropriate
        def native_translate_formals(argument, option):
            # type: (AtFormal, FunctionOption) -> AtFormal
            def translate_map(const):
                # type: (bool) -> Dict[str, str]
                return {
                    'Tensor': 'const Tensor &' if const else 'Tensor &',
                    'Type': 'const Type &' if const else 'Type &',
                    'TensorOptions': 'const TensorOptions &' if const else 'TensorOptions &',
                    'TensorList': 'TensorList',
                }

            if argument.get('is_nullable') and argument['type'] not in translate_map(False).keys():
                argument['type'] = "c10::optional<{}>".format(argument['type'])

            if (option['inplace'] and argument['name'] == 'self') or argument.get('output', False):
                argument['type'] = translate_map(False).get(argument['type'], argument['type'])
            else:
                argument['type'] = translate_map(True).get(argument['type'], argument['type'])

            return argument

        result = [native_translate_formals(argument, option) for argument in result]
        return result

    # this can return multiple return types in a list, e.g. ['Tensor', 'Tensor']
    def native_get_return_types(option):
        # type: (FunctionOption) -> List[ReturnType]
        ret = option['return']

        return_types = []  # List[ReturnType]
        for t_raw in ret:
            # See Note [field_name versus name]
            field_name = None
            if isinstance(t_raw, string_type):
                t = t_raw
                name = None
            elif t_raw is None:
                t = 'void'
                name = None
            else:
                t = t_raw['type']
                name = t_raw['name']
                if 'field_name' in t_raw:
                    field_name = t_raw['field_name']

            # can't actually return a TensorList (since it's a reference object)
            actual_return_type = {'TensorList': 'std::vector<Tensor>'}.get(t, t)

            if actual_return_type == 'Tensor' and (option['inplace'] or option['api_name'].endswith('_out')):
                # follow normal ATen convention of returning Tensor & for inplace functions.
                actual_return_type = 'Tensor &'

            rtype = {
                'type': actual_return_type,
                'dynamic_type': NATIVE_DYNAMIC_TYPE.get(t, t),
            }  # type: ReturnType
            if name is not None:
                rtype['name'] = name
            if field_name is not None:
                rtype['field_name'] = field_name
            return_types.append(rtype)

        return return_types

    def process_native(option, output_options):
        # type: (FunctionOption, List[OutputDeclaration]) -> None
        assert option['python_module'] == '' or option['python_module'] == 'nn', \
            "Found python_module of {} for decl {}, but only \'\' string or \'nn\' are supported".format(
                option['python_module'], option['name'])

        formals = native_get_formals(option)
        option['formals_list'] = formals
        option['formals'] = [format_formal(f) for f in formals]
        option['formals_with_defaults'] = [formal_with_default(f) for f in formals]
        option['returns'] = native_get_return_types(option)
        option['return_type'] = format_return_type(option['returns'])
        option['return_call'] = 'return ' if option['return_type'] != 'void' else ''
        option['actuals'] = [f['name'] for f in formals]

        option['formals_types'] = [f['type'] for f in option['formals_list']]
        option['native_actuals'] = [f['name'] for f in option['formals_list']]

        option['method_formals'] = [format_formal(f) for f in formals
                                    if f['name'] != 'self']
        option['method_formals_with_defaults'] = (
            [formal_with_default(f) for f in formals if f['name'] != 'self'])
        option['method_actuals'] = [
            f['name'] if f['name'] != 'self' else '*this' for f in formals]

        def find_formal(formal_name, formals):
            for formal in formals:
                if formal_name == formal['dynamic_type']:
                    return formal
            return None

        assert find_formal('Type', formals) is None, \
            "Found Type argument in {}({}). Use TensorOptions instead.".format(
                option['name'], ", ".join(option['method_formals_with_defaults']))

        type_method_dispatch = option['type_method_definition_dispatch']

        dispatch_options = find_formal('TensorOptions', formals)
        # Only dispatch via tensor if there is no Options argument
        dispatch_tensor = None if dispatch_options else find_dispatch_tensor(formals)

        option['type_method_formals'] = [format_formal(f) for f in formals]
        option['type_method_actuals'] = [f['name'] for f in formals]
        option['native_actuals'] = [f['name'] for f in formals]

        option['const_mark'] = '' if option['inplace'] else ' const'

        is_method = 'method' in option['variants']
        is_namespace_function = 'function' in option['variants']
        # For method-only entries, the first argument should be self
        if is_method and not is_namespace_function:
            assert formals[0]['name'] == 'self'
        is_factory_method = find_formal('TensorOptions', formals) and 'method' not in option['variants']

        check_methods_do_not_start_with_underscore(option['name'], is_method)

        option['method_prefix_derived'] = ''
        option['device_guard_declaration'] = device_guard(option, dispatch_options, dispatch_tensor)
        option['named_guard_declaration'] = named_guard(option, find_tensors(formals),
                                                        find_tensorlists(formals))
        option['dispatch_scalar_type_declaration'] = dispatch_scalar_type(option, dispatch_options, dispatch_tensor)

        env = nested_dict(option, top_env)

        broadcast_arg = get_broadcast_argument(option)
        if broadcast_arg is not None:
            raise Exception("broadcasting is not yet supported for native functions, "
                            "but specified for function {}", option['name'])

        top_env['registration_declarations'].append(
            REGISTRATION_DECLARATION.substitute(env))
        option['native_type_method_dispatch'] = type_method_dispatch

        # Note [Abstract ATen methods]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # An abstract ATen method is one whose dispatch differs between
        # types.  These are implemented in derived types (with a
        # standard (throwing) definition in Type).  A concrete ATen
        # method is one which has the same dispatch for all types;
        # we just implement it in the base Type.  This is exposed
        # in Declarations.yaml via a field named 'abstract'.
        abstract = False
        if isinstance(type_method_dispatch, dict):
            abstract = True
        else:
            top_env['type_method_declarations'].append(
                NATIVE_DISPATCH_DECLARATION.substitute(env))
            top_env['type_method_definitions'].append(
                NATIVE_DISPATCH_DEFINITION_DEFAULT.substitute(env))
            top_env['function_registrations'].append(
                DEFAULT_FUNCTION_REGISTRATION.substitute(env))


        # generate the at::native function declarations (i.e. what the user will implement)
        if isinstance(type_method_dispatch, dict):
            generated_native_functions = []  # type: List[str]
            for key in sorted(type_method_dispatch.keys()):
                value = type_method_dispatch[key]
                # skip functions in different namespace, e.g. legacy::cpu
                if "::" in value:
                    continue
                if value not in generated_native_functions:
                    option['native_type_method_dispatch'] = value
                    top_env['native_function_declarations'].append(
                        NATIVE_DECLARATION.substitute(env))
                    generated_native_functions.append(value)
        else:
            top_env['native_function_declarations'].append(
                NATIVE_DECLARATION.substitute(env))

        method_of = ['Type']
        if is_method:
            top_env['tensor_method_declarations'].append(
                TENSOR_METHOD_DECLARATION.substitute(env))
            top_env['tensor_method_definitions'].append(
                TENSOR_METHOD_DEFINITION.substitute(env))
            method_of.append('Tensor')

        if is_namespace_function:
            if dispatch_tensor:
                option['inferred_backend'] = 'at::detail::infer_backend({})'.format(dispatch_tensor)
                option['inferred_is_variable'] = 'at::detail::infer_is_variable({})'.format(dispatch_tensor)
            elif dispatch_options:
                option['inferred_backend'] = '{}.backend()'.format(dispatch_options['name'])
                option['inferred_is_variable'] = '{}.is_variable()'.format(dispatch_options['name'])
            else:
                # doesn't depend on a specific backend, use CPU
                option['inferred_backend'] = 'Backend::CPU'
                option['inferred_is_variable'] = 'false'
            declaration = DEPRECATED_FUNCTION_DECLARATION if option['deprecated'] else FUNCTION_DECLARATION
            top_env['function_declarations'].append(declaration.substitute(env))
            if is_factory_method:
                top_env['function_definitions'].append(
                    FACTORY_DEFINITION.substitute(env))
            else:
                top_env['function_definitions'].append(
                    FUNCTION_DEFINITION.substitute(env))
            method_of.append('namespace')

        output_options.append(OutputDeclaration(
            name=option['api_name'],
            matches_jit_signature=option["matches_jit_signature"],
            schema_string=option["schema_string"],
            method_prefix_derived=option['method_prefix_derived'],
            arguments=formals,
            method_of=method_of,
            mode=option['mode'],
            python_module=option['python_module'],
            buffers=None,
            returns=option['returns'],
            inplace=option['inplace'],
            is_factory_method=is_factory_method,
            # See Note [Abstract ATen methods]
            abstract=abstract,
            requires_tensor=option.get('requires_tensor', False),
            device_guard=option.get('device_guard', True),
            with_gil=option.get('with_gil', False),
            deprecated=option['deprecated'],
        ))

    output_declarations = []  # type: List[OutputDeclaration]
    for declaration in declarations:
        output_options = []  # type: List[OutputDeclaration]
        for option in declaration['options']:
            option["matches_jit_signature"] = declaration["matches_jit_signature"]
            option["schema_string"] = declaration["schema_string"]
            try:
                if option['mode'] != 'native':
                    process_option(option, output_options)
                else:
                    process_native(option, output_options)
            except NYIError:
                option['skip'] = True
        output_declarations.extend(output_options)

    return output_declarations


def create_derived(backend_type_env, declarations):
    # type: (Environment, List[FunctionOption]) -> Tuple[List[str], List[str], List[str], List[str], List[str]]
    type_object_declarations = []  # type: List[str]
    type_object_definitions = []  # type: List[str]
    function_registrations = []  # type: List[str]
    legacy_th_declarations = []  # type: List[str]
    legacy_th_definitions = []  # type: List[str]
    is_cuda = 'CUDA' in backend_type_env['Backend']

    def requires_checked_cast(argument):
        # type: (THFormal) -> bool
        if argument['type'] == 'IntArrayRef':
            return 'size' in argument
        return argument['type'] in CHECKED_CAST

    def nullable_argument(argument):
        # type: (THFormal) -> bool
        return argument.get('is_nullable', False)

    def bool_option_is_string(argument):
        # type: (THFormal) -> bool
        return 'if_true' in argument and isinstance(argument['if_true'], string_type)

    def get_argument(env, argument, option):
        # type: (Environment, THFormal, FunctionOption) -> str
        if requires_checked_cast(argument):
            checked_use = CHECKED_USE.get(
                argument['type'], '{}_').format(argument['name'])
            if nullable_argument(argument):
                checked_use = CHECKED_USE_NULLABLE.substitute(
                    env={}, arg_name=argument['name'], usage=checked_use)
            return checked_use
        elif argument['type'] == 'bool' and 'if_true' in argument:
            if bool_option_is_string(argument):
                tpl = '({}) ? "{}" : "{}"'
            else:
                tpl = '({}) ? {} : {}'
            return tpl.format(argument['name'],
                              argument['if_true'], argument['if_false'])
        elif argument['type'] == 'CONSTANT':
            # this is a bool that is actually a string...
            if bool_option_is_string(argument):
                return '"{}"'.format(argument['name'])
            v = str(argument.get('default', argument['name']))
            for pattern, replacement in CONSTANT_REPLACEMENTS:
                v = re.sub(pattern, replacement, v)
            return CodeTemplate(v).substitute(env)
        # e.g. argument 0, i.e. repeat the 0th argument in this position...
        elif argument['type'] == 'argument':
            index = int(argument['name'])
            return get_argument(env, option['arguments'][index], option)
        else:
            return argument['name']

    def drop_argument(argument, option):
        # type: (THFormal, FunctionOption) -> bool
        # Devices are handled in the body of the function.
        if argument['name'] == 'device':
            return True
        return False

    def get_arguments(env, arguments, option):
        # type: (Environment, List[THFormal], FunctionOption) -> List[str]
        return [get_argument(env, argument, option)
                for argument in arguments if not drop_argument(argument, option)]

    def is_actual_return_long(env, ret):
        # type: (Environment, ReturnDecl) -> bool
        if ret['type'] == 'long':
            return True
        if ret['type'] == 'real':
            return env['ScalarName'] == 'Long'
        if ret['type'] == 'accreal':
            return env['AccScalarName'] == 'Long'
        return False

    def handle_zero_dim(env, option):
        # type: (Environment, FunctionOption) -> List[str]
        zero_dim_dispatch = option.get('zero_dim_dispatch_when_scalar', '')
        if not zero_dim_dispatch:
            return []
        broadcasts_arg = zero_dim_dispatch in option.get('broadcast_actuals', '')
        zero_dim_only = option.get('zero_dim_tensor_only', False)
        # this combination doesn't seem to make sense
        assert not (broadcasts_arg and zero_dim_only)
        # if the argument broadcasts, then this would only affect cases where all broadcasted
        # tensors were zero-dim, which is inconsistent with the scalar handling.
        if broadcasts_arg:
            return []
        zero_dim_actuals = [arg['name']
                            if arg['name'] != zero_dim_dispatch else "{}.item()".format(arg['name'])
                            for arg in option['formals_list']]
        return [ZERO_DIM_CHECK.substitute(env, check_name=zero_dim_dispatch, zero_dim_actuals=zero_dim_actuals)]

    def handle_only_zero_dim(env, option):
        # type: (Environment, FunctionOption) -> Optional[List[str]]
        if option.get('zero_dim_tensor_only', False):
            check_name = option['zero_dim_dispatch_when_scalar']
            return [ZERO_DIM_ONLY.substitute(env, check_name=check_name)]
        else:
            return None

    def allocate_arg(env, arg, output_count):
        # type: (Environment, THFormal, int) -> List[str]
        name = arg['name']
        allocation = CodeTemplate(ALLOC_NOARGS_WRAP[arg['type']]).substitute(env)
        tensor_arg = '{}_'.format(name)
        if arg.get('mask', False):
            allocation = 'output_mask[{}] ? {} : nullptr'.format(output_count, allocation)
            tensor_arg = ('{}_ == nullptr ? (TensorImpl*)UndefinedTensorImpl::singleton() : (TensorImpl*){}_'
                          .format(name, name))
        intrusive_ptr_type = 'c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>'
        return [
            'auto {}_ = {};'.format(name, allocation),
            'auto {} = Tensor({}::reclaim({}));'.format(name, intrusive_ptr_type, tensor_arg),
        ]

    def resize_arg(arg):
        # type: (THFormal) -> str
        resize = arg['resize']
        if isinstance(resize, str):
            return "{}.resize_({}.sizes());".format(arg['name'], resize)
        else:
            resize_scalar = arg.get('resize_scalar', False)
            if resize_scalar:
                dims = ['{}.dim() == 0 ? 1 : {}.size({})'.format(name, name, dim) for name, dim in resize]
            else:
                dims = ['{}.size({})'.format(name, dim) for name, dim in resize]
            return "{}.resize_({{ {} }});".format(arg['name'], ','.join(dims))

    def handle_call(env, option, cimpl):
        # type: (Environment, FunctionOption, FunctionOption) -> str
        is_nn = option['mode'] == 'NN'
        actuals = get_arguments(env, cimpl['arguments'], option)
        if is_cuda or is_nn:
            actuals = ['globalContext().getTHCState()'] + actuals

        cname = cimpl['cname']
        if option.get('sparse', False):
            if is_cuda:
                cname = 'THCS' + env['ScalarName'] + "Tensor_" + cname
            else:
                cname = env['THTensor'].replace('TH', 'THS') + '_' + cname
        elif is_nn:
            cname = 'THNN_{}'.format(env['THType']) + cname
        else:
            cname = env['THTensor'] + '_' + cname

        call = CALL_TEMPLATE.substitute(actuals=actuals, cname=cname)
        if cimpl.get('condition') is not None:
            call = 'if ({}) {}'.format(cimpl['condition'], call)
        return call

    def emit_body(env, option, scalar_type_cases):
        # type: (Environment, FunctionOption, List[str]) -> List[str]
        body = []  # type: List[str]
        body += handle_zero_dim(env, option)
        only_zero_dim_check = handle_only_zero_dim(env, option)
        if only_zero_dim_check is not None:
            #  code below only_zero_dim_check is unreachable so we do not need to generate the rest.
            body += only_zero_dim_check
            return body

        cases = []
        for scalar_name, c_type, accreal, _ in scalar_types:
            if scalar_name in scalar_type_cases:
                case_body = []
                # arguments are potentially duplicated because of one argument
                # referencing another
                seen_names = set()  # type: Set[str]
                seen_tensorlists = set()  # type: Set[str]
                count = 0
                output_count = 0

                case_env = {
                    'Backend': env['Backend'],
                    'DeviceType': env['DeviceType'],
                    'state': env['state'],
                    'ScalarType': c_type,
                    'ScalarName': scalar_name,
                    'AccScalarName': accreal,
                    'THType': scalar_name,
                    'THTensor': 'TH{}Tensor'.format(scalar_name)
                }  # type: Environment
                if case_env['Backend'] == 'CUDA':
                    sname = '' if scalar_name == "Float" else scalar_name
                    case_env['THType'] = 'Cuda{}'.format(sname)
                    case_env['THTensor'] = 'THCuda{}Tensor'.format(sname)

                # scalar_check is the heuristic conditions when a result may be a scalar_check
                # if there is a IntArrayRefSize argument, then its dimensions are used to determine scalar.
                # otherwise, it is true if all the input tensors are scalars,
                scalar_check_is_from_size = False
                scalar_check_is_from_option = False
                scalar_check = None
                scalar_check_opt = option.get('scalar_check')
                if scalar_check_opt is not None:
                    if isinstance(scalar_check_opt, bool):
                        scalar_check = str(scalar_check_opt).lower()
                    else:
                        scalar_check = scalar_check_opt
                    scalar_check_is_from_option = True

                for arg in option['arguments']:
                    if is_real_argument_to_wrapper(arg):
                        count += 1
                    if arg['type'] == 'IntArrayRefSize' and not scalar_check_is_from_option:
                        scalar_check_is_from_size = True
                        scalar_check = '{}.size() == 0'.format(arg['name'])
                    if arg['type'] == 'TensorList':
                        seen_tensorlists.add(arg['name'])

                    wrap_dim_target = arg.get('wrap_dim', None)
                    if wrap_dim_target is not None:
                        # for Tensors, "name_" is the TensorImpl, but for TensorLists, it is an
                        # std::vector of TH*s.  Since TH*s have different dimension rules, we used
                        # "name" instead, but keep "name_" for tensor to avoid an extra function call.
                        if wrap_dim_target not in seen_tensorlists:
                            wrap_dim_target = wrap_dim_target + "_"
                        case_body.append("{} = maybe_wrap_dim({}, {});".format(
                            arg['name'], arg['name'], wrap_dim_target))

                    # only generated checked casts the first time we see it
                    if arg['name'] not in seen_names and requires_checked_cast(arg):
                        seen_names.add(arg['name'])

                        # make a new allocation of TensorImpl, then wrap a Tensor around it.
                        if arg.get('allocate', False):
                            case_body += allocate_arg(case_env, arg, output_count)
                            output_count += 1
                        # extract the TensorImpl from an existing tensor (or Storage, etc.)
                        else:
                            # special case where we allow undefined Tensors, and thus
                            # the checked cast succeeds even if the Tensor is not
                            # defined
                            null_okay = 'true' if nullable_argument(arg) else 'false'
                            default_init = []
                            if 'default_init' in arg:
                                default_init.append(arg['default_init'])

                            check_cast = CHECKED_CAST[arg['type']].substitute(
                                case_env, arg_name=arg['name'], arg_pos=count,
                                null_okay=null_okay, default_init=default_init,
                                size=arg.get('size'))
                            case_body.append("auto {}_ = {};".format(
                                arg['name'], check_cast))
                        if drop_argument(arg, option):
                            case_body.append(
                                "(void) {}_; //silence unused warning".format(arg['name']))

                        initializers = []

                        # resize tensors for special ops that require it
                        if 'resize' in arg:
                            initializers.append(resize_arg(arg))

                        # also special handling where we zero some outputs.
                        if arg.get('zero', False) or (arg.get('cpu_zero', False) and not is_cuda):
                            initializers.append("{}.zero_();".format(arg['name']))

                        # only initialize non-null arguments
                        if nullable_argument(arg) and len(initializers) > 0:
                            case_body.append(CONDITIONAL_INITIALIZER.substitute({
                                'name': arg['name'],
                                'initializer': initializers
                            }))
                        else:
                            case_body += initializers

                        # for out-of-place: dim() == 0 for all input tensors is and'd to form
                        # the test for whether the output is also a scalar
                        # for in-place: dim() == 0 shouldn't change as a result of the operation
                        if (not arg.get('output') and 'Tensor' in arg['type'] and
                                'TensorList' not in arg['type'] and
                                'THS' not in arg['type'] and
                                not scalar_check_is_from_size and
                                not scalar_check_is_from_option and
                                not option['inplace']):
                            check = '{}->dim() == 0'.format(arg['name'] + '_')
                            if nullable_argument(arg):
                                check = '(!{} || {})'.format(arg['name'] + '_', check)
                            scalar_check = (check if scalar_check is None
                                            else scalar_check + ' && ' + check)

                # cimpls, if it exists, contains the underlying C function names and
                # arguments. Otherwise use option
                cimpls = option.get('cimpls', [option])
                calls = [handle_call(case_env, option, cimpl) for cimpl in cimpls]

                ret = option['return']

                if ret['kind'] == 'arguments':
                    if 'aten_custom_call' in option:
                        # all aten_custom_call bodies handle settings on their own.
                        scalar_check = None
                        case_body.append(CodeTemplate(
                            option['aten_custom_call']).substitute(case_env))
                    else:
                        case_body.extend([call + ';' for call in calls])
                    arguments_indices = ret['arguments']
                    arguments = [option['arguments'][argi]
                                 for argi in arguments_indices]
                    if scalar_check is not None:
                        if not isinstance(scalar_check, dict):
                            if len(arguments) > 1:
                                case_body.append("bool maybe_scalar = {};".format(scalar_check))
                                scalar_check = 'maybe_scalar'
                        for arg in arguments:
                            scalar_check_arg = (scalar_check if not isinstance(scalar_check, dict)
                                                else scalar_check.get(arg['name']))  # type: ignore
                            if scalar_check_arg is not None:
                                stmt = "{}_->maybe_zero_dim({});".format(arg['name'], scalar_check_arg)
                                if nullable_argument(arg):
                                    stmt = "if ({}_) {}".format(arg['name'], stmt)
                                case_body.append(stmt)
                    if len(arguments_indices) == 1:
                        arg = arguments[0]
                        case_body.append("return {};".format(arg['name']))
                    else:
                        types = [to_return_type(arg, option)['type']
                                 for arg in arguments]
                        # TODO: check for move semantics...
                        names = [arg['name'] for arg in arguments]
                        case_body.append(CodeTemplate("return std::tuple<${types}>(${names});").substitute(
                            types=types, names=names))
                elif ret['kind'] == 'type':
                    assert len(calls) == 1
                    call = calls[0]
                    if 'aten_custom_call' in option:
                        # all aten_custom_call bodies handle settings on their own.
                        scalar_check = None
                        case_body.append(CodeTemplate(
                            option['aten_custom_call']).substitute(case_env))

                    if ret['type'] in ALLOC_WRAP.keys():
                        maybe_scalar = "->maybe_zero_dim({})".format(scalar_check) \
                                       if scalar_check is not None \
                                       else ""
                        wrapped_tensor = CodeTemplate(ALLOC_WRAP[ret['type']]).substitute(
                            case_env, arguments=[call])
                        return_tensor = (
                            "return Tensor(" +
                            "c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(" +
                            "(${wrapped_tensor})${maybe_scalar}));")
                        case_body.append(CodeTemplate(return_tensor).substitute(
                            case_env, wrapped_tensor=wrapped_tensor, maybe_scalar=maybe_scalar))
                    # return the same underlying Tensor type for both real and accreal; this ensures
                    # e.g. x.sum(0) and x.sum() return the same type. We explicitly cast to the
                    # ScalarType before constructing the scalar_tensor to avoid overflow checking.
                    elif ret['type'] == 'accreal' or ret['type'] == 'real':
                        return_scalar = ('return at::scalar_tensor(convert<${ScalarType}>(${call}), '
                                         'options(ScalarType::${ScalarName}));')
                        case_body.append(CodeTemplate(return_scalar).substitute(case_env, call=call))
                    else:
                        # we using int64_t for long in the API, so correct it here...
                        if is_actual_return_long(case_env, ret):
                            call = "static_cast<int64_t>({})".format(call)
                        case_body.append("return {};".format(call))
                else:
                    raise Exception("NYI - return handling")

                cases.append(LEGACY_TH_DEFINITION_CASE.substitute(case_env, case_body=case_body))
        body.append(LEGACY_TH_DEFINITION_SWITCH_STATEMENT.substitute(env, cases=cases))
        return body

    def process_option(option):
        # type: (FunctionOption) -> None
        backend = backend_type_env['Backend']
        if backend in option['backend_types']:
            env = nested_dict(option, backend_type_env)
            body = emit_body(env, option, option['backend_types'][backend])  # type: ignore
            option['type_definition_body'] = body
            if option.get('broadcast_actuals', None):
                legacy_th_declarations.append(
                    LEGACY_TH_DECLARATION_BROADCAST.substitute(env))
                legacy_th_definitions.append(
                    LEGACY_TH_DEFINITION_BROADCAST.substitute(env))
            legacy_th_declarations.append(
                LEGACY_TH_DECLARATION.substitute(env))
            legacy_th_definitions.append(
                LEGACY_TH_DEFINITION.substitute(env))

    def process_native(option):
        # type: (FunctionOption) -> None
        dispatch = option['type_method_definition_dispatch']
        env = nested_dict(option, backend_type_env)

        if isinstance(dispatch, dict):
            backend = backend_type_env['Backend']
            if backend in option['backend_types']:
                native_dispatch = dispatch.get(backend)
                if native_dispatch:
                    type_object_declarations.append(
                        NATIVE_DISPATCH_DECLARATION.substitute(env))
                    option['native_type_method_dispatch'] = native_dispatch
                    type_object_definitions.append(
                        NATIVE_DISPATCH_DEFINITION_BACKEND.substitute(env))
                    function_registrations.append(
                        BACKEND_FUNCTION_REGISTRATION.substitute(env))

    for declaration in declarations:
        for option in declaration['options']:
            if not option.get('skip', False):
                try:
                    if option['mode'] == 'NN' and option.get('cimpls') is None:
                        continue
                    if option['mode'] != 'native':
                        process_option(option)
                    else:
                        process_native(option)
                except NYIError:
                    pass
    return (type_object_declarations, type_object_definitions, function_registrations, legacy_th_declarations,
            legacy_th_definitions)


def create_extension_backend(backend_type_env, declarations):
    # type: (Environment, List[FunctionOption]) -> Tuple[List[str], List[str], List[str]]
    type_object_declarations = []
    type_object_definitions = []
    function_registrations = []

    for declaration in declarations:
        for option in declaration['options']:
            if not option.get('skip', False) and option['mode'] == 'native':
                try:
                    schema_args = ", ".join(
                        ["{} {}".format(f['dynamic_type'], f['name']) for f in option['formals_list']])
                    return_type = NATIVE_DYNAMIC_TYPE.get(option['return_type'], option['return_type'])
                    option['schema'] = "{}({}) -> {}".format(option['api_name'], schema_args, return_type)
                    env = nested_dict(option, backend_type_env)
                    type_object_declarations.append(
                        NATIVE_DISPATCH_DECLARATION.substitute(env))
                    type_object_definitions.append(
                        NATIVE_DISPATCH_DEFINITION_EXTENSION_BACKEND.substitute(env))
                    function_registrations.append(
                        BACKEND_FUNCTION_REGISTRATION.substitute(env))
                except NYIError:
                    pass
    return type_object_declarations, type_object_definitions, function_registrations
