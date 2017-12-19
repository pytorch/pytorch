# HEY! Trying to understand what this file does?  Read
# "what has to be done to add a Operation ..." first!

import re
from collections import OrderedDict
from code_template import CodeTemplate

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
#
# 1. if broadcasting or without the full list of arguments, add a non-virtual
#    declaration under Type.h  (right now, we call this template
#    BROADCAST but it also handles default arguments)
TYPE_METHOD_DECLARATION_BROADCAST = CodeTemplate("""\
${return_type} ${api_name}(${formals_with_defaults}) const;
""")
# 2. broadcasting functions are implemented in Type.cpp
TYPE_METHOD_DEFINITION_BROADCAST = CodeTemplate("""\
${return_type} Type::${api_name}(${formals}) const {
    Tensor ${broadcast_returns};
    std::tie(${broadcast_returns}) = ${broadcast_function}(${broadcast_actuals}, "${api_name}");
    return ${method_prefix_derived}${api_name}(${broadcast_modified_actuals});
}
""")
# 3. add virtual dispatch declaration to Type.h and impl to Type.cpp; method_prefix_derived
#    is present for providing a base-class definition for a derived-type method with a prefix.
#
#    If the declaration is abstract, then the actual implementation will
#    be in a derived type; we put in a simple default "not implemented"
#    stub.  However, if the declaration is concrete, we dispatch to the
#    actual implementation.  At the moment, this situation *only* occurs
#    for 'native' declarations (so the native dispatch is hardcoded into
#    the template here.)
TYPE_METHOD_DECLARATION_ABSTRACT = CodeTemplate("""\
virtual ${return_type} ${method_prefix_derived}${api_name}(${formals_with_defaults}) const;
""")
TYPE_METHOD_DEFINITION_ABSTRACT = CodeTemplate("""\
${return_type} Type::${method_prefix_derived}${api_name}(${formals}) const {
    runtime_error("${method_prefix_derived}${api_name} is not implemented for type %s", toString());
}
""")
TYPE_METHOD_DECLARATION_CONCRETE = CodeTemplate("""\
virtual ${return_type} ${api_name}(${formals_with_defaults}) const;
""")
TYPE_METHOD_DEFINITION_CONCRETE = CodeTemplate("""\
${return_type} Type::${api_name}(${formals}) const {
    ${return_call} at::native::${native_type_method_dispatch}(${actuals});
}
""")
# 4. add virtual override to TypeDerived.h
TYPE_DERIVED_DECLARATION = CodeTemplate("""\
virtual ${return_type} ${method_prefix_derived}${api_name}(${formals}) const override;
""")
# 5. add override definition to TypeDerived.cpp
TYPE_DERIVED_DEFINITION = CodeTemplate("""\
${return_type} ${Type}::${method_prefix_derived}${api_name}(${formals}) const {
    ${type_definition_body}
}
""")
# NB: As far as ezyang can tell, we don't *have* to codegen this,
# because we will inherit it from the TYPE_METHOD_DEFINITION_CONCRETE in
# the superclass.  But it doesn't seem to be harmful.
TYPE_DERIVED_DEFINITION_NATIVE = CodeTemplate("""\
${return_type} ${Type}::${api_name}(${formals}) const {
    ${return_call} at::native::${native_type_method_dispatch}(${actuals});
}
""")

# 6. add non-virtual declaration to Tensor.h
TENSOR_METHOD_DECLARATION = CodeTemplate("""\
${return_type} ${api_name}(${method_formals_with_defaults})${const_mark};
""")
# 7. add non-virtual declaration to Tensor.cpp
TENSOR_METHOD_DEFINITION = CodeTemplate("""\
inline ${return_type} Tensor::${api_name}(${method_formals})${const_mark} {
    return type().${api_name}(${method_actuals});
}
""")
# 8. add a method declaration in Functions.h
FUNCTION_DECLARATION = CodeTemplate("""\
static inline ${return_type} ${api_name}(${formals_with_defaults});
""")
# 9. add method definition in Functions.h
FUNCTION_DEFINITION = CodeTemplate("""\
static inline ${return_type} ${api_name}(${formals}) {
    return ${inferred_type}.${api_name}(${actuals});
}
""")
# 10. add a native declaration for a native function
NATIVE_DECLARATION = CodeTemplate("""\
${return_type} ${native_type_method_dispatch}(${formals_with_defaults});
""")

# We need to cast to the base type because C++ may hide the base class
# implementation of ${api_name} if we have overloaded a function with
# the same name (but different signature) already
ZERO_DIM_CHECK = CodeTemplate("""\
if (${check_name}.dim() == 0) {
    return static_cast<const Type*>(this)->${api_name}(${zero_dim_actuals});
}""")

ZERO_DIM_ONLY = CodeTemplate("""\
runtime_error("${api_name} only supports a 0-dimensional ${check_name} tensor, but got tensor "
    "with %" PRId64 " dimension(s)", ${check_name}.dim());
""")

SPARSE_CHECK = CodeTemplate("""\
if(${check_name}.type().is_sparse()) {
    return static_cast<const Type*>(this)->${api_name}(${sparse_actuals});
}""")

BUFFER_DEFINITION = CodeTemplate("""\
auto ${name}_ = new ${Tensor}(context);
auto ${name} = Tensor(${name}_, false);""")

CONDITIONAL_INITIALIZER = CodeTemplate("""\
if (${name}.defined()) {
    ${initializer}
}""")

CALL_TEMPLATE = CodeTemplate("${cname}(${actuals})")


class NYIError(Exception):
    """Indicates we don't support this declaration yet"""

    def __init__(self, reason):
        self.reason = reason


TYPE_FORMAL_GENERIC = {
    'THTensor*': 'Tensor &',
    'THSTensor*': 'SparseTensor',
    'THBoolTensor*': 'Tensor &',
    'THIndexTensor*': 'Tensor &',
    'THIntegerTensor*': 'Tensor &',
    'THStorage*': 'Storage &',
    'THGenerator*': 'Generator *',
    'THSize*': 'IntList',
    'THStride*': 'IntList',
    'accreal': 'Scalar',
    'real': 'Scalar',
    'long': 'int64_t',
}

DYNAMIC_TYPE = {
    'THTensor*': 'Tensor',
    'THSTensor*': 'SparseTensor',
    'THBoolTensor*': 'BoolTensor',
    'THIndexTensor*': 'IndexTensor',
    'THIntegerTensor*': 'IntegerTensor',
    'THStorage*': 'Storage',
    'THGenerator*': 'Generator*',
    'THSize*': 'IntList',
    'THStride*': 'IntList',
    'accreal': 'accreal',
    'real': 'real',
    'long': 'int64_t',
}

TYPE_RETURN = {
    'THTensor*': 'Tensor',
    'THIndexTensor*': 'Tensor',
    'THBoolTensor*': 'Tensor',
    'THIntegerTensor*': 'Tensor',
    'THSTensor*': 'Tensor',
    'real': 'Tensor',
    'accreal': 'Tensor',
    'long': 'int64_t',
}

CHECKED_CAST = {
    'THTensor*':
        CodeTemplate(
            'checked_cast_tensor<${Tensor}>(${arg_name}.pImpl,"${arg_name}",${arg_pos}, ${null_okay})'),
    'THSTensor*':
    CodeTemplate(
        'checked_cast_tensor<Sparse${Tensor}>(${arg_name}.tref.pImpl,"${arg_name}",${arg_pos},false)'),
    'THBoolTensor*':
        CodeTemplate(
            'checked_cast_tensor<${Backend}ByteTensor>(${arg_name}.pImpl,"${arg_name}",${arg_pos}, ${null_okay})'),
    'THIndexTensor*':
        CodeTemplate(
            'checked_cast_tensor<${Backend}LongTensor>(${arg_name}.pImpl,"${arg_name}",${arg_pos}, ${null_okay})'),
    'THIntegerTensor*':
        CodeTemplate(
            'checked_cast_tensor<${Backend}IntTensor>(${arg_name}.pImpl,"${arg_name}",${arg_pos}, ${null_okay})'),
    'THStorage*': CodeTemplate('checked_cast_storage<${Storage}>(&${arg_name},"${arg_name}",${arg_pos})'),
    'THGenerator*':
        CodeTemplate(
            'check_generator<${Backend}Generator>(${arg_name}, &context->defaultGenerator(backend()))'),
    'THSize*': CodeTemplate('THLongStorageView::makeFromSize(${arg_name})'),
    'THStride*': CodeTemplate('THLongStorageView::makeFromStride(${arg_name}, ${noelem_to_empty})'),
    'real': CodeTemplate('${arg_name}.to${ScalarName}()'),
    'accreal': CodeTemplate('${arg_name}.to${AccScalarName}()'),
    'TensorList': CodeTemplate('tensor_list_checked_cast<${Tensor}, Tensor, '
                               '${THTensor}>(${arg_name},"${arg_name}",${arg_pos})'),
    'IntList': CodeTemplate('check_intlist<${size}>(${arg_name}, "${arg_name}", ${arg_pos}${,default_init})')
}

CHECKED_USE = {
    'THTensor*': '{}_->tensor',
    'THSTensor*': '{}_->tensor',
    'THIndexTensor*': '{}_->tensor',
    'THBoolTensor*': '{}_->tensor',
    'THIntegerTensor*': '{}_->tensor',
    'THStorage*': '{}_->storage',
    'THGenerator*': '{}_->generator',
    'TensorList': "{0}_.data(), {0}_.size()",
}

CHECKED_USE_NULLABLE = CodeTemplate('${arg_name}_ ? ${usage} : NULL')

ALLOC_WRAP = {
    'THTensor*': 'new ${Tensor}(context)',
    'THBoolTensor*': 'new ${Backend}ByteTensor(context)',
    'THIndexTensor*': 'new ${Backend}LongTensor(context)',
    'THIntegerTensor*': 'new ${Backend}IntTensor(context)',
    'THSTensor*': 'new Sparse${Tensor}(context)',
}

# Replacements for constants when calling into TH
CONSTANT_REPLACEMENTS = [
    ('AS_REAL', '${AS_REAL}'),
    ('THPGenerator_TH_CData(THPDefaultGenerator)',
     'dynamic_cast<${Generator}&>().generator'),
    ('__storage_size.get\\(\\)',
     'THLongStorageView::makeFromLength(static_cast<int64_t>(storage.size()))'),
    ('__last_dim', 'self.ndimension()-1'),
]

# Replacements for constants in header file function definitions
HEADER_CONSTANT_REPLACEMENTS = [
    (r'AS_REAL\((.*)\)', r'\1'),
    ('THPGenerator_TH_CData\(THPDefaultGenerator\)', 'nullptr'),
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


def is_real_argument_to_wrapper(argument):
    return not argument.get('output', False) and\
        argument['type'] != 'CONSTANT' and\
        argument['type'] != 'argument'


def is_mutable_formal_argument(argument, option):
    return argument.get('output') or option['inplace'] and argument['name'] == 'self'


def to_return_type(arg, option):
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
    # translates defaults from cwrap types to C++ values
    def translate_default(argument, type_str, default):
        if default is None:
            # cause the default constructor for the object to run
            return '{}'
        if 'if_true' in argument:
            return argument['default'] == argument['if_true']
        for pattern, replacement in HEADER_CONSTANT_REPLACEMENTS:
            default = re.sub(pattern, replacement, str(default))
        if type_str in {'Scalar', 'int64_t', 'double'}:
            return float(default) if '.' in default else int(default)
        elif type_str == 'bool':
            assert default.lower() in ['true', 'false']
            return default.lower() == 'true'
        else:
            return default

    # change from THTensor* to Tensor & so we get how it will appear
    # in the aten argument list...
    def translate_formal(argument, option):
        type_str = TYPE_FORMAL_GENERIC.get(argument['type'], argument['type'])
        if type_str == 'Tensor &' and not is_mutable_formal_argument(argument, option):
            type_str = 'const ' + type_str
        translated = {
            'name': argument['name'],
            'type': type_str,
            'dynamic_type': DYNAMIC_TYPE.get(argument['type'], argument['type']),
        }
        if 'kwarg_only' in argument:
            translated['kwarg_only'] = argument['kwarg_only']
        if 'default' in argument:
            default = translate_default(argument, type_str, argument['default'])
            translated['default'] = default
            translated['default_init'] = argument.get('default_init', default)
        if 'python_default_init' in argument:
            assert 'default' not in argument
            default = translate_default(argument, type_str, argument['python_default_init'])
            translated['python_default_init'] = default
        if argument.get('output'):
            translated['output'] = True
        if argument.get('size'):
            translated['size'] = argument['size']
        if argument.get('is_nullable') is not None:
            translated['is_nullable'] = argument['is_nullable']
        return translated

    def get_formals(option, include_constants=False):
        seen = set()
        pos_args = []
        kwd_args = []

        def insert(argument):
            if argument['name'] not in seen:
                seen.add(argument['name'])
                if argument.get('kwarg_only', False):
                    kwd_args.append(argument)
                else:
                    pos_args.append(argument)

        def has_output_mask(argument):
            return argument.get('allocate', False) and argument.get('mask', False)

        for argument in option['arguments']:
            if argument.get('output') and not argument.get('allocate', False):
                insert(argument)
        for argument in option['arguments']:
            if argument['type'] == 'THSTensor*':
                # only enable for a subset of Dense/Sparse ops
                if not (option.get('aten_dense_sparse', False)):
                    raise NYIError("Sparse Tensor")

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
        if len(return_types) == 1:
            return return_types[0]['type']
        return "std::tuple<{}>".format(','.join(r['type'] for r in return_types))

    def find_dispatch_tensor(formals):
        # dispatch to self if it's a parameter
        for formal in formals:
            if formal['name'] == 'self' and formal['dynamic_type'] == 'Tensor':
                return formal['name']
        # otherwise dispatch to the first Tensor or TensorList
        for formal in formals:
            if 'TensorList' == formal['dynamic_type'] or formal['dynamic_type'] == 'Tensor':
                return formal['name']
        return None

    def format_formal(f):
        return '{} {}'.format(f['type'], f['name'])

    def formal_with_default(f):
        s = format_formal(f)
        v = f.get('default')
        if v is None:
            return s
        if isinstance(v, bool):
            v = str(v).lower()
        return '{}={}'.format(s, v)

    def get_broadcast_argument(option):
        for argument in option['arguments']:
            if argument.get('broadcast'):
                return argument

    def get_broadcast_actuals(broadcast_arg, broadcast_inplace, broadcast_dims):
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
            broadcast_dims = ([x.split('.')[0] + '.size(' + x.split('.')[1].replace('dim', '') + ')'
                              for x in broadcast_dims_spec])
            broadcast_dims_init_list = '{' + ','.join(broadcast_dims) + '}'
            broadcast_actuals = [broadcast_arg['name'], broadcast_dims_init_list]

        return broadcast_actuals

    def process_option(option, output_options):
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

        option['const_mark'] = '' if option['inplace'] else ' const'

        is_method = 'method' in option['variants']
        is_function = 'function' in option['variants']
        dispatch_tensor = find_dispatch_tensor(formals)
        is_namespace_function = is_function and dispatch_tensor is not None

        broadcast_arg = get_broadcast_argument(option)
        # "s_" for "same size".
        option['method_prefix_derived'] = '' if broadcast_arg is None else 's_'
        env = nested_dict(option, top_env)

        abstract = True
        if broadcast_arg is None:
            top_env['type_method_declarations'].append(
                TYPE_METHOD_DECLARATION_ABSTRACT.substitute(env))
            top_env['type_method_definitions'].append(
                TYPE_METHOD_DEFINITION_ABSTRACT.substitute(env))
        else:
            top_env['type_method_declarations'].append(
                TYPE_METHOD_DECLARATION_BROADCAST.substitute(env))
            top_env['type_method_declarations'].append(
                TYPE_METHOD_DECLARATION_ABSTRACT.substitute(env))
            top_env['type_method_definitions'].append(
                TYPE_METHOD_DEFINITION_ABSTRACT.substitute(env))

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
            top_env['type_method_definitions'].append(
                TYPE_METHOD_DEFINITION_BROADCAST.substitute(env))

        method_of = ['Type']
        if is_method:
            top_env['tensor_method_declarations'].append(
                TENSOR_METHOD_DECLARATION.substitute(env))
            top_env['tensor_method_definitions'].append(
                TENSOR_METHOD_DEFINITION.substitute(env))
            method_of.append('Tensor')

        if is_namespace_function:
            option['inferred_type'] = 'infer_type({})'.format(dispatch_tensor)
            top_env['function_declarations'].append(
                FUNCTION_DECLARATION.substitute(env))
            top_env['function_definitions'].append(
                FUNCTION_DEFINITION.substitute(env))
            method_of.append('namespace')

        buffer_names = [buffer['name'] for buffer in option.get('buffers', [])]

        output_options.append(OrderedDict([
            ('name', option['api_name']),
            ('method_prefix_derived', option['method_prefix_derived']),
            ('arguments', formals),
            ('method_of', method_of),
            ('mode', option['mode']),
            ('buffers', buffer_names),
            ('returns', option['returns']),
            ('inplace', option['inplace']),
            # See Note [Abstract ATen methods]
            ('abstract', abstract),
        ]))

    def native_get_formals(option, include_constants=False):
        seen = set()
        pos_args = []
        kwd_args = []

        def insert(argument):
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
        def add_type_as_dynamic_type(argument, option):
            argument['dynamic_type'] = argument['type']
            return argument

        result = pos_args + kwd_args
        result = [add_type_as_dynamic_type(argument, option) for argument in result]

        # ensure we get reference-type formals when appropriate
        def native_translate_formals(argument, option):
            if (option['inplace'] and argument['name'] == 'self') or argument.get('output', False):
                argument['type'] = {'Tensor': 'Tensor &'}.get(argument['type'], argument['type'])
            else:
                argument['type'] = {'Tensor': 'const Tensor &'}.get(argument['type'], argument['type'])

            return argument

        result = [native_translate_formals(argument, option) for argument in result]
        return result

    # this can return multiple return types in a list, e.g. ['Tensor', 'Tensor']
    def native_get_return_types(option):
        ret = option['return']

        return_types = []
        for t_raw in ret:
            if isinstance(t_raw, string_type):
                t = t_raw
                name = None
            else:
                t = t_raw['type']
                name = t_raw['name']

            # can't actually return a TensorList (since it's a reference object)
            actual_return_type = {'TensorList': 'std::vector<Tensor>'}.get(t, t)

            if actual_return_type == 'Tensor' and option['inplace']:
                # follow normal ATen convention of returning Tensor & for inplace functions.
                actual_return_type = 'Tensor &'

            rtype = {
                'type': actual_return_type,
                'dynamic_type': t,
            }
            if name is not None:
                rtype['name'] = name
            return_types.append(rtype)

        return return_types

    def process_native(option, output_options):
        option['inplace'] = re.search(
            '(^__i|[^_]_$)', option['api_name']) is not None

        formals = native_get_formals(option)
        option['formals_list'] = formals
        option['formals'] = [format_formal(f) for f in formals]
        option['formals_with_defaults'] = [formal_with_default(f) for f in formals]
        option['returns'] = native_get_return_types(option)
        option['return_type'] = format_return_type(option['returns'])
        option['return_call'] = 'return ' if option['return_type'] != 'void' else ''
        option['actuals'] = [f['name'] for f in formals]

        option['method_formals'] = [format_formal(f) for f in formals
                                    if f['name'] != 'self']
        option['method_formals_with_defaults'] = (
            [formal_with_default(f) for f in formals if f['name'] != 'self'])
        option['method_actuals'] = [
            f['name'] if f['name'] != 'self' else '*this' for f in formals]

        option['const_mark'] = '' if option['inplace'] else ' const'

        is_method = 'method' in option['variants']
        is_function = 'function' in option['variants']
        dispatch_tensor = find_dispatch_tensor(formals)
        is_namespace_function = is_function and dispatch_tensor is not None

        option['method_prefix_derived'] = ''
        env = nested_dict(option, top_env)

        broadcast_arg = get_broadcast_argument(option)
        if broadcast_arg is not None:
            raise Exception("broadcasting is not yet supported for native functions, "
                            "but specified for function {}", option['name'])

        top_env['type_method_declarations'].append(
            TYPE_METHOD_DECLARATION_CONCRETE.substitute(env))
        dispatch = option['type_method_definition_dispatch']
        option['native_type_method_dispatch'] = dispatch

        # Note [Abstract ATen methods]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # An abstract ATen method is one whose dispatch differs between
        # types.  These are implemented in derived types (with a
        # standard (throwing) definition in Type).  A concrete ATen
        # method is one which has the same dispatch for all types;
        # we just implement it in the base Type.  This is exposed
        # in Declarations.yaml via a field named 'abstract'.
        if isinstance(dispatch, dict):
            abstract = True
            top_env['type_method_definitions'].append(
                TYPE_METHOD_DEFINITION_ABSTRACT.substitute(env))
        else:
            abstract = False
            top_env['type_method_definitions'].append(
                TYPE_METHOD_DEFINITION_CONCRETE.substitute(env))

        # generate the at::native function declarations (i.e. what the user will implement)
        if isinstance(dispatch, dict):
            generated_native_functions = []
            for key in sorted(dispatch.keys()):
                value = dispatch[key]
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
            option['inferred_type'] = 'infer_type({})'.format(dispatch_tensor)
            top_env['function_declarations'].append(
                FUNCTION_DECLARATION.substitute(env))
            top_env['function_definitions'].append(
                FUNCTION_DEFINITION.substitute(env))
            method_of.append('namespace')

        output_options.append(OrderedDict([
            ('name', option['api_name']),
            ('method_prefix_derived', option['method_prefix_derived']),
            ('arguments', formals),
            ('method_of', method_of),
            ('mode', option['mode']),
            ('returns', option['returns']),
            ('inplace', option['inplace']),
            # See Note [Abstract ATen methods]
            ('abstract', abstract),
        ]))

    output_declarations = []
    for declaration in declarations:
        output_options = []
        for option in declaration['options']:
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
    type_object_declarations = []
    type_object_definitions = []

    is_cuda = 'CUDA' in backend_type_env['Backend']

    def replace_with_null(argument):
        return (argument['type'] == 'THGenerator*' and
                backend_type_env['Backend'] == 'CUDA')

    def requires_checked_cast(argument):
        if argument['type'] == 'IntList':
            return 'size' in argument
        return argument['type'] in CHECKED_CAST

    def nullable_argument(argument):
        return argument.get('is_nullable', False)

    def bool_option_is_string(argument):
        return 'if_true' in argument and isinstance(argument['if_true'], string_type)

    def get_argument(argument, option):
        if replace_with_null(argument):
            return 'NULL'
        elif requires_checked_cast(argument):
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
            return CodeTemplate(v).substitute(backend_type_env)
        # e.g. argument 0, i.e. repeat the 0th argument in this position...
        elif argument['type'] == 'argument':
            index = int(argument['name'])
            return get_argument(option['arguments'][index], option)
        else:
            return argument['name']

    def drop_argument(argument, option):
        return 'CUDA' in backend_type_env['Backend'] and (
            (option['mode'] == 'TH' and argument['type'] == 'THGenerator*') or
            argument.get('default') == 'THPGenerator_TH_CData(THPDefaultGenerator)')

    def get_arguments(arguments, option):
        return [get_argument(argument, option)
                for argument in arguments if not drop_argument(argument, option)]

    def is_actual_return_long(ret):
        if ret['type'] == 'long':
            return True
        if ret['type'] == 'real':
            return backend_type_env['ScalarName'] == 'Long'
        if ret['type'] == 'accreal':
            return backend_type_env['AccScalarName'] == 'Long'
        return False

    def get_zero_dim_dispatch_when_scalar(option):
        return option.get('zero_dim_dispatch_when_scalar', False)

    def handle_zero_dim(env, option):
        zero_dim_dispatch = get_zero_dim_dispatch_when_scalar(option)
        if not zero_dim_dispatch:
            return []
        zero_dim_actuals = [arg['name']
                            if arg['name'] != zero_dim_dispatch else "Scalar({})".format(arg['name'])
                            for arg in option['formals_list']]
        return [ZERO_DIM_CHECK.substitute(env, check_name=zero_dim_dispatch, zero_dim_actuals=zero_dim_actuals)]

    def handle_only_zero_dim(env, option):
        if option.get('zero_dim_tensor_only', False):
            check_name = get_zero_dim_dispatch_when_scalar(option)
            return [ZERO_DIM_ONLY.substitute(env, check_name=check_name)]
        else:
            return None

    def handle_sparse(env, option):
        if 'when_sparse_dispatch' not in option or 'Sparse' in backend_type_env['Backend']:
            return []
        check_name = option['when_sparse_dispatch']
        sparse_actuals = [arg['name']
                          if arg['name'] != check_name else "SparseTensor({})".format(arg['name'])
                          for arg in option['formals_list']]
        return [SPARSE_CHECK.substitute(env, check_name=check_name, sparse_actuals=sparse_actuals)]

    def handle_buffers(env, option):
        if 'buffers' not in option:
            return []
        return [BUFFER_DEFINITION.substitute(env, name=b['name'])
                for b in option['buffers']]

    def allocate_arg(env, arg, output_count):
        name = arg['name']
        allocation = CodeTemplate(ALLOC_WRAP[arg['type']]).substitute(env)
        tensor_arg = '{}_'.format(name)
        if arg.get('mask', False):
            allocation = 'output_mask[{}] ? {} : nullptr'.format(output_count, allocation)
            tensor_arg = ('{}_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*){}_'
                          .format(name, name))
        return [
            'auto {}_ = {};'.format(name, allocation),
            'auto {} = Tensor({}, false);'.format(name, tensor_arg),
        ]

    def resize_arg(arg):
        resize = arg['resize']
        if isinstance(resize, str):
            return "{}.resize_({}.sizes());".format(arg['name'], resize)
        else:
            dims = ['{}.size({})'.format(name, dim) for name, dim in resize]
            return "{}.resize_({{ {} }});".format(arg['name'], ','.join(dims))

    def handle_call(env, option, cimpl):
        is_nn = option['mode'] == 'NN'
        actuals = get_arguments(cimpl['arguments'], option)
        if is_cuda or is_nn:
            actuals = ['context->thc_state'] + actuals

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

    def emit_body(env, option):
        body = []
        body += handle_sparse(env, option)
        body += handle_zero_dim(env, option)
        only_zero_dim_check = handle_only_zero_dim(env, option)
        if only_zero_dim_check is not None:
            #  code below only_zero_dim_check is unreachable so we do not need to generate the rest.
            body += only_zero_dim_check
            return body

        body += handle_buffers(env, option)
        # arguments are potentially duplicated because of one argument
        # referencing another
        seen_names = set()
        seen_tensorlists = set()
        count = 0
        output_count = 0

        # scalar_check is the heuristic conditions when a result may be a scalar_check
        # if there is a THSize* argument, then its dimensions are used to determine scalar.
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
            if arg['type'] == 'THSize*' and not scalar_check_is_from_option:
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
                body.append("{} = maybe_wrap_dim({}, {});"
                            .format(arg['name'], arg['name'], wrap_dim_target))

            # only generated checked casts the first time we see it
            if arg['name'] not in seen_names and requires_checked_cast(arg):
                seen_names.add(arg['name'])

                # make a new allocation of TensorImpl, then wrap a Tensor around it.
                if arg.get('allocate', False):
                    body += allocate_arg(env, arg, output_count)
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

                    noelem_to_empty = 'is_noelem_tensor_size(size)' if 'size' in seen_names else 'false'
                    check_cast = CHECKED_CAST[arg['type']].substitute(
                        env, arg_name=arg['name'], arg_pos=count,
                        null_okay=null_okay, default_init=default_init,
                        size=arg.get('size'),
                        noelem_to_empty=noelem_to_empty)
                    body.append("auto {}_ = {};".format(
                        arg['name'], check_cast))
                if drop_argument(arg, option) or replace_with_null(arg):
                    body.append(
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
                    body.append(CONDITIONAL_INITIALIZER.substitute({
                        'name': arg['name'],
                        'initializer': initializers
                    }))
                else:
                    body += initializers

                # for out-of-place: isScalar() for all input tensors is and'd to form
                # the test for whether the output is also a scalar
                # for in-place: isScalar() shouldn't change as a result of the operation
                if (not arg.get('output') and 'Tensor' in arg['type'] and
                        'TensorList' not in arg['type'] and
                        'THS' not in arg['type'] and
                        not scalar_check_is_from_size and
                        not scalar_check_is_from_option and
                        not option['inplace']):
                    check = '{}->isScalar()'.format(arg['name'] + '_')
                    if nullable_argument(arg):
                        check = '(!{} || {})'.format(arg['name'] + '_', check)
                    scalar_check = (check if scalar_check is None
                                    else scalar_check + ' && ' + check)

        # cimpls, if it exists, contains the underlying C function names and
        # arguments. Otherwise use option
        cimpls = option.get('cimpls', [option])
        calls = [handle_call(env, option, cimpl) for cimpl in cimpls]

        ret = option['return']

        if ret['kind'] == 'arguments':
            if 'aten_custom_call' in option:
                # all aten_custom_call bodies handle settings on their own.
                scalar_check = None
                body.append(CodeTemplate(
                    option['aten_custom_call']).substitute(env))
            else:
                body.extend([call + ';' for call in calls])
            arguments_indices = ret['arguments']
            arguments = [option['arguments'][argi]
                         for argi in arguments_indices]
            if scalar_check is not None:
                if len(arguments) > 1:
                    body.append("bool maybe_scalar = {};".format(scalar_check))
                    scalar_check = 'maybe_scalar'
                for arg in arguments:
                    stmt = "{}_->maybeScalar({});".format(arg['name'], scalar_check)
                    if nullable_argument(arg):
                        stmt = "if ({}_) {}".format(arg['name'], stmt)
                    body.append(stmt)
            if len(arguments_indices) == 1:
                arg = arguments[0]
                body.append("return {};".format(arg['name']))
            else:
                types = [to_return_type(arg, option)['type']
                         for arg in arguments]
                # TODO: check for move semantics...
                names = [arg['name'] for arg in arguments]
                body.append(CodeTemplate("return std::tuple<${types}>(${names});").substitute(
                    types=types, names=names))
        elif ret['kind'] == 'type':
            assert len(calls) == 1
            call = calls[0]
            if ret['type'] == 'THTensor*':
                maybe_scalar = "->maybeScalar({})".format(scalar_check) \
                               if scalar_check is not None \
                               else ""
                return_tensor = "return Tensor((new ${Tensor}(context,${arg_name}))${maybe_scalar},false);"
                body.append(CodeTemplate(return_tensor).substitute(
                    env, arg_name=call, maybe_scalar=maybe_scalar))
            # return the same underlying Tensor type for both real and accreal; this ensures
            # e.g. x.sum(0) and x.sum() return the same type.
            elif ret['type'] == 'accreal' or ret['type'] == 'real':
                body.append('return scalarTensor({});'.format(call))
            else:
                # we using int64_t for long in the API, so correct it here...
                if is_actual_return_long(ret):
                    call = "static_cast<int64_t>({})".format(call)
                body.append("return {};".format(call))
        else:
            raise Exception("NYI - return handling")
        return body

    def process_option(option):
        pair = (backend_type_env['Backend'],
                backend_type_env['ScalarName'])
        if pair in option['backend_type_pairs']:
            env = nested_dict(option, backend_type_env)
            body = emit_body(env, option)
            option['type_definition_body'] = body
            type_object_declarations.append(
                TYPE_DERIVED_DECLARATION.substitute(env))
            type_object_definitions.append(
                TYPE_DERIVED_DEFINITION.substitute(env))

    def process_native(option):
        dispatch = option['type_method_definition_dispatch']
        env = nested_dict(option, backend_type_env)

        if isinstance(dispatch, dict):
            pair = (backend_type_env['Backend'],
                    backend_type_env['ScalarName'])
            if pair in option['backend_type_pairs']:
                native_dispatch = dispatch.get(pair[0])
                if native_dispatch is None:
                    raise Exception('could not find backend {} in native function dispatch specification {}'
                                    .format(pair[0], dispatch))
                option['native_type_method_dispatch'] = native_dispatch
                type_object_declarations.append(
                    TYPE_DERIVED_DECLARATION.substitute(env))
                type_object_definitions.append(
                    TYPE_DERIVED_DEFINITION_NATIVE.substitute(env))

    for declaration in declarations:
        for option in declaration['options']:
            if not option.get('skip', False):
                try:
                    if option['mode'] != 'native':
                        process_option(option)
                    else:
                        process_native(option)
                except NYIError:
                    pass
    return type_object_declarations, type_object_definitions
