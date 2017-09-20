import re
from code_template import CodeTemplate

import sys
if sys.version_info[0] == 3:
    string_type = str
else:
    string_type = basestring

# temporary things we cannot handle
EXCLUDE_PATTERN = "bernoulli.*|normal.*|exponential.*|random.*|arange.*"
# what has to be done to add a Operation ...
# 1. if broadcasting or without the full list of arguments, add a non-virtual
#    declaration under Type.h
TYPE_METHOD_DECLARATION_NON_VIRTUAL = CodeTemplate("""\
${return_type} ${method_prefix}${api_name}(${formals}) ;
""")
# 2. broadcasting functions are implemented in Type.cpp
TYPE_METHOD_DEFINITION_BROADCAST = CodeTemplate("""\
${return_type} Type::${method_prefix}${api_name}(${formals}) {
    Tensor ${broadcast_returns};
    std::tie(${broadcast_returns}) = ${broadcast_function}(${broadcast_actuals});
    return ${method_prefix_derived}${api_name}(${broadcast_modified_actuals});
}
""")
# 3. functions without the full list of arguments are implemented in TypeMethods.h
TYPE_METHOD_INLINE = CodeTemplate("""\
inline ${return_type} Type::${method_prefix}${api_name}(${formals}) {
    ${return_call}${method_prefix}${api_name}(${actuals_with_constants});
}
""")
# 4. add virtual dispatch declaration to Type.h and default impl to Type.cpp
TYPE_METHOD_DECLARATION = CodeTemplate("""\
virtual ${return_type} ${method_prefix}${api_name}(${formals}) ;
""")
TYPE_METHOD_DEFINITION = CodeTemplate("""\
${return_type} Type::${method_prefix}${api_name}(${formals}) {
    throw std::runtime_error(std::string("${api_name} is not implemented for type ") + toString());
}
""")
# 5. add virtual override to TypeDerived.h
TYPE_DERIVED_DECLARATION = CodeTemplate("""\
virtual ${return_type} ${method_prefix_derived}${api_name}(${formals}) override;
""")
# 6. add override definition to TypeDerived.cpp
TYPE_DERIVED_DEFINITION = CodeTemplate("""\
${return_type} ${Type}::${method_prefix_derived}${api_name}(${formals}) {
    ${type_definition_body}
}
""")
# 7. add non-virtual declaration to Tensor.h
TENSOR_METHOD_DECLARATION = CodeTemplate("""\
${return_type} ${api_name}(${method_formals})${const_mark};
""")
# 8. add non-virtual declaration to Tensor.cpp
TENSOR_METHOD_DEFINITION = CodeTemplate("""\
inline ${return_type} Tensor::${api_name}(${method_formals})${const_mark} {
    return type().${method_prefix}${api_name}(${method_actuals});
}
""")
# 9. add a method declaration in Functions.h
FUNCTION_DECLARATION = CodeTemplate("""\
static inline ${return_type} ${api_name}(${formals});
""")
# 10. add a method definition in Functions.cpp
FUNCTION_DEFINITION = CodeTemplate("""\
static inline ${return_type} ${api_name}(${formals}) {
    return ${inferred_type}.${api_name}(${actuals});
}
""")

# We need to cast to the base type because C++ may hide the base class
# implementation of ${api_name} if we have overloaded a function with
# the same name (but different signature) already
ZERO_DIM_CHECK = CodeTemplate("""\
if(${check_name}.dim() == 0) {
    return static_cast<Type*>(this)->${method_prefix}${api_name}(${zero_dim_actuals});
}""")

SPARSE_CHECK = CodeTemplate("""\
if(${check_name}.type().isSparse()) {
    return static_cast<Type*>(this)->${method_prefix}${api_name}(${sparse_actuals});
}""")

SPARSE_CHECK = CodeTemplate("""\
if(${check_name}.type().isSparse()) {
    return static_cast<Type*>(this)->${method_prefix}${api_name}(${sparse_actuals});
}""")


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
    'THGenerator*': 'Generator &',
    'THSize*': 'IntList',
    'THStride*': 'IntList',
    'accreal': 'Scalar',
    'real': 'Scalar',
    'long': 'int64_t',
}

DYNAMIC_TYPE = {
    'THTensor*': 'Tensor',
    'THBoolTensor*': 'BoolTensor',
    'THIndexTensor*': 'IndexTensor',
    'THIntegerTensor*': 'IntegerTensor',
    'THStorage*': 'Storage',
    'THGenerator*': 'Generator',
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
    'real': 'Scalar',
    'accreal': 'Scalar',
    'long': 'int64_t',
}
CHECKED_CAST = {
    'THTensor*': CodeTemplate('checked_cast<${Tensor}>(${arg_name}.pImpl,"${arg_name}",${arg_pos}, ${null_okay})'),
    'THSTensor*':
    CodeTemplate(
        'checked_cast<Sparse${Tensor}>(${arg_name}.tref.pImpl,"${arg_name}",${arg_pos},false)'),
    'THBoolTensor*':
        CodeTemplate(
            'checked_cast<${Backend}ByteTensor>(${arg_name}.pImpl,"${arg_name}",${arg_pos}, ${null_okay})'),
    'THIndexTensor*':
        CodeTemplate(
            'checked_cast<${Backend}LongTensor>(${arg_name}.pImpl,"${arg_name}",${arg_pos}, ${null_okay})'),
    'THIntegerTensor*':
        CodeTemplate(
            'checked_cast<${Backend}IntTensor>(${arg_name}.pImpl,"${arg_name}",${arg_pos}, ${null_okay})'),
    'THStorage*': CodeTemplate('checked_cast<${Storage}>(&${arg_name},"${arg_name}",${arg_pos}, false)'),
    'THGenerator*': CodeTemplate('check_generator<${Backend}Generator>(&${arg_name})'),
    'THSize*': CodeTemplate('THLongStorageView::make(${arg_name}, true)'),
    'THStride*': CodeTemplate('THLongStorageView::make(${arg_name}, false, true)'),
    'real': CodeTemplate('${arg_name}.to${ScalarName}()'),
    'accreal': CodeTemplate('${arg_name}.to${AccScalarName}()'),
    'TensorList': CodeTemplate('tensor_list_checked_cast<${Tensor}, Tensor, '
                               '${THTensor}>(${arg_name},"${arg_name}",${arg_pos})'),
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
}

# Replacements for constants when calling into TH
CONSTANT_REPLACEMENTS = [
    ('AS_REAL', '${AS_REAL}'),
    ('THPDefaultGenerator->cdata',
     'dynamic_cast<${Generator}&>(context->defaultGenerator(backend())).generator'),
    ('__storage_size.get\\(\\)',
     'THLongStorageView::make(static_cast<int64_t>(storage.size()))'),
    ('__last_dim', 'self.ndimension()-1'),
]

# Replacements for constants when calling other ATen functions
INLINE_CONSTANT_REPLACEMENTS = [
    (r'AS_REAL\((.*)\)', r'\1'),
    ('THPDefaultGenerator->cdata', 'context->defaultGenerator(backend())'),
    ('__last_dim', 'self.ndimension()-1'),
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
        'type': rt,
        'dynamic_type': DYNAMIC_TYPE.get(arg['type'], arg['type']),
    }


def create_generic(top_env, declarations):

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
        if 'default' in argument:
            if 'if_true' in argument:
                val = argument['default'] == argument['if_true']
                translated['default'] = str(val).lower()
            else:
                translated['default'] = argument['default']
        if argument.get('output'):
            translated['output'] = True
        return translated

    def get_formals(option, include_constants=False):
        seen = set()
        result = []

        def insert(argument):
            if argument['name'] not in seen:
                seen.add(argument['name'])
                result.append(argument)
        for argument in option['arguments']:
            if argument['type'] == 'THSTensor*':
                # only enable for a subset of Dense/Sparse ops
                if not (option.get('aten_dense_sparse', False)):
                    raise NYIError("Sparse Tensor")

            if include_constants and argument['type'] == 'CONSTANT':
                insert(argument)
            elif is_real_argument_to_wrapper(argument):
                insert(argument)
        for argument in option['arguments']:
            if argument.get('output') and not argument.get('allocate', False):
                insert(argument)

        return [translate_formal(argument, option) for argument in result]

    def get_actuals_with_constants(option):
        actuals = []
        for arg in get_formals(option, include_constants=True):
            if arg['type'] != 'CONSTANT':
                actuals.append(arg['name'])
                continue
            v = str(arg.get('default', arg['name']))
            for pattern, replacement in INLINE_CONSTANT_REPLACEMENTS:
                v = re.sub(pattern, replacement, v)
            if v in {'NULL', 'nullptr'}:
                if arg['name'] == 'stride':
                    v = 'IntList()'
                else:
                    v = 'Tensor()'
            actuals.append(v)
        return actuals

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
        return return_types

    def find_first_tensor(formals):
        for formal in formals:
            if 'Tensor' == formal['dynamic_type'] or 'TensorList' == formal['dynamic_type']:
                return formal['name']
        return None

    def format_formal(f):
        return '{} {}'.format(f['type'], f['name'])

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

    excluded_names = set()

    def process_option(option, output_options):
        option['inplace'] = re.search(
            '(^__i|[^_]_$)', option['api_name']) is not None

        if re.match(EXCLUDE_PATTERN, option['name']):
            excluded_names.add(option['name'])
            raise NYIError("NYI")

        # print(yaml.dump(option))
        formals = get_formals(option)
        option['formals_list'] = formals
        option['formals'] = [format_formal(f) for f in formals]
        option['returns'] = get_return_types(option)
        option['return_type'] = format_return_type(option['returns'])
        option['return_call'] = 'return ' if option['return_type'] != 'void' else ''
        option['actuals'] = [f['name'] for f in formals]

        option['method_formals'] = [format_formal(f) for f in formals
                                    if f['name'] != 'self']
        option['method_actuals'] = [
            f['name'] if f['name'] != 'self' else '*this' for f in formals]

        option['const_mark'] = '' if option['inplace'] else ' const'

        is_method = 'method' in option['variants']
        is_function = 'function' in option['variants']
        first_tensor = find_first_tensor(formals)
        is_namespace_function = is_function and first_tensor is not None

        # method-only things are prefixed with m_ in Type so that
        # another function-only variant can exist without the name colliding
        option['method_prefix'] = 'm_' if is_method and not is_function else ''
        option['method_prefix_derived'] = option['method_prefix']
        env = nested_dict(option, top_env)

        broadcast_arg = get_broadcast_argument(option)
        if broadcast_arg is None and option['has_full_argument_list']:
            top_env['type_method_declarations'].append(
                TYPE_METHOD_DECLARATION.substitute(env))
            top_env['type_method_definitions'].append(
                TYPE_METHOD_DEFINITION.substitute(env))
        else:
            top_env['type_method_declarations'].append(
                TYPE_METHOD_DECLARATION_NON_VIRTUAL.substitute(env))

        if not option['has_full_argument_list']:
            # functions without the full list of arguments are implemented
            # inline in TypeMethods.h
            option['actuals_with_constants'] = get_actuals_with_constants(option)
            top_env['type_method_inline_definitions'].append(
                TYPE_METHOD_INLINE.substitute(env))
        elif broadcast_arg is not None:
            # "s_" for "same size".
            option['method_prefix_derived'] = 's_' + option['method_prefix']
            same_size_option = option.copy()
            same_size_option['method_prefix'] = option['method_prefix_derived']
            same_size_env = nested_dict(same_size_option, top_env)
            top_env['type_method_declarations'].append(
                TYPE_METHOD_DECLARATION.substitute(same_size_env))
            top_env['type_method_definitions'].append(
                TYPE_METHOD_DEFINITION.substitute(same_size_env))

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
            option['inferred_type'] = 'infer_type({})'.format(first_tensor)
            top_env['function_declarations'].append(
                FUNCTION_DECLARATION.substitute(env))
            top_env['function_definitions'].append(
                FUNCTION_DEFINITION.substitute(env))
            method_of.append('namespace')

        output_options.append({
            'name': option['api_name'],
            'method_prefix': option['method_prefix_derived'],
            'arguments': formals,
            'method_of': method_of,
            'returns': option['returns'],
            'inplace': option['inplace'],
            'has_full_argument_list': option['has_full_argument_list'],
        })

    output_declarations = []
    for declaration in declarations:
        output_options = []
        for option in declaration['options']:
            try:
                process_option(option, output_options)
            except NYIError:
                option['skip'] = True
        output_declarations.extend(output_options)
    print("ATen Excluded: {}".format(excluded_names))
    return output_declarations


def create_derived(backend_type_env, declarations):
    type_object_declarations = []
    type_object_definitions = []

    def replace_with_null(argument):
        return (argument['type'] == 'THGenerator*' and
                backend_type_env['Backend'] == 'CUDA')

    def requires_checked_cast(argument):
        return argument['type'] in CHECKED_CAST

    def nullable_argument(argument):
        return (argument['type'] in {'THIntegerTensor*', 'THTensor*'} and
                argument.get('default', '') in {'NULL', 'nullptr'})

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
            argument.get('default') == 'THPDefaultGenerator->cdata')

    def get_arguments(option):
        return [get_argument(argument, option)
                for argument in option['arguments'] if not drop_argument(argument, option)]

    def is_actual_return_long(ret):
        if ret['type'] == 'long':
            return True
        if ret['type'] == 'real':
            return backend_type_env['ScalarName'] == 'Long'
        if ret['type'] == 'accreal':
            return backend_type_env['AccScalarName'] == 'Long'
        return False

    def handle_zero_dim(env, option):
        if 'zero_dim_dispatch_when_scalar' not in option:
            return []
        check_name = option['zero_dim_dispatch_when_scalar']
        zero_dim_actuals = [arg['name']
                            if arg['name'] != check_name else "Scalar({})".format(arg['name'])
                            for arg in option['formals_list']]
        return [ZERO_DIM_CHECK.substitute(env, check_name=check_name, zero_dim_actuals=zero_dim_actuals)]

    def handle_sparse(env, option):
        if 'when_sparse_dispatch' not in option or 'Sparse' in backend_type_env['Backend']:
            return []
        check_name = option['when_sparse_dispatch']
        sparse_actuals = [arg['name']
                          if arg['name'] != check_name else "SparseTensor({})".format(arg['name'])
                          for arg in option['formals_list']]
        return [SPARSE_CHECK.substitute(env, check_name=check_name, sparse_actuals=sparse_actuals)]

    def emit_body(env, option):
        body = []
        body += handle_sparse(env, option)
        body += handle_zero_dim(env, option)
        # arguments are potentially duplicated because of one argument
        # referencing another
        seen_names = set()
        count = 0
        is_cuda = 'CUDA' in backend_type_env['Backend']

        # scalar_check is the heuristic conditions when a result may be a scalar_check
        # if there is a THSize* argument, then its dimensions are used to determine scalar.
        # otherwise, it is true if all the input tensors are scalars,
        scalar_check_is_from_size = False
        scalar_check = None
        for arg in option['arguments']:
            if is_real_argument_to_wrapper(arg):
                count += 1
            if arg['type'] == 'THSize*':
                scalar_check_is_from_size = True
                scalar_check = '{}.size() == 0'.format(arg['name'])

            wrap_dim_arg = arg.get('wrap_dim', None)
            if wrap_dim_arg is not None:
                # wrap_dim specification can have (add) expressions, e.g. self+1
                wrap_dim_params = wrap_dim_arg.split("+")
                wrap_dim_params[0] = wrap_dim_params[0] + "_"
                wrap_dim_target = wrap_dim_params[0]
                wrap_dim_params[0] = "{}->dim()".format(wrap_dim_target)
                wrap_dim_expr = "+".join(wrap_dim_params)
                body.append("{} = maybe_wrap_dim({}, {});".format(arg['name'], arg['name'], wrap_dim_expr))

            # only generated checked casts the first time we see it
            if not arg['name'] in seen_names and requires_checked_cast(arg):
                seen_names.add(arg['name'])

                # make a new allocation of TensorImpl, then wrap a Tensor around it.
                if arg.get('allocate', False):
                    allocation = CodeTemplate(
                        ALLOC_WRAP[arg['type']]).substitute(env)
                    body.append('auto {}_ = {};'.format(
                        arg['name'], allocation))
                    body.append('auto {} = Tensor({}_,false);'.format(
                        arg['name'], arg['name']))
                # extract the TensorImpl from an existing tensor (or Storage, etc.)
                else:
                    # special case where we allow undefined Tensors, and thus
                    # the checked cast succeeds even if the Tensor is not
                    # defined
                    null_okay = 'true' if nullable_argument(arg) else 'false'

                    check_cast = CHECKED_CAST[arg['type']].substitute(
                        env, arg_name=arg['name'], arg_pos=count,
                        null_okay=null_okay)
                    body.append("auto {}_ = {};".format(
                        arg['name'], check_cast))
                if drop_argument(arg, option) or replace_with_null(arg):
                    body.append(
                        "(void) {}_; //silence unused warning".format(arg['name']))
                # resize tensors for special ops that require it
                if 'resize' in arg:
                    resize = arg['resize']
                    if isinstance(resize, str):
                        body.append("{}.resize_({}.sizes());".format(
                            arg['name'], resize))
                    else:
                        dims = ['{}.size({})'.format(name, dim)
                                for name, dim in resize]
                        body.append("{}.resize_({{ {} }});".format(
                            arg['name'], ','.join(dims)))
                # also special handling where we zero some outputs.
                if arg.get('cpu_zero', False) and not is_cuda:
                    body.append("{}.zero_();".format(arg['name']))

                # isScalar() for all input tensors is and'd to form
                # the test for whether the output is also a scalar
                if (not arg.get('output') and 'Tensor' in arg['type'] and
                        'TensorList' not in arg['type'] and
                        'THS' not in arg['type'] and
                        not scalar_check_is_from_size):
                    check = '{}->isScalar()'.format(arg['name'] + '_')
                    scalar_check = (check if scalar_check is None
                                    else scalar_check + ' && ' + check)

        option['derived_actuals'] = get_arguments(option)
        is_nn = option['mode'] == 'NN'
        if is_cuda or is_nn:
            option['derived_actuals'] = [
                'context->thc_state'] + option['derived_actuals']

        if is_nn:
            prefix = 'THNN_{}'.format(env['THType'])
        elif option.get('sparse', False):
            if is_cuda:
                prefix = 'THCS' + env['ScalarName'] + "Tensor_"
            else:
                prefix = env['THTensor'].replace('TH', 'THS') + '_'
        else:
            prefix = env['THTensor'] + '_'

        call = prefix + \
            CodeTemplate("${cname}(${derived_actuals})").substitute(env)
        ret = option['return']

        if ret['kind'] == 'arguments':
            if 'aten_custom_call' in option:
                # all aten_custom_call bodies handle settings on their own.
                scalar_check = None
                body.append(CodeTemplate(
                    option['aten_custom_call']).substitute(env))
            else:
                body.append(call + ";")
            arguments_indices = ret['arguments']
            arguments = [option['arguments'][argi]
                         for argi in arguments_indices]
            if scalar_check is not None:
                if len(arguments) > 1:
                    body.append("bool maybe_scalar = {};".format(scalar_check))
                    scalar_check = 'maybe_scalar'
                for arg in arguments:
                    body.append(
                        "{}_->maybeScalar({});".format(arg['name'], scalar_check))
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
            if ret['type'] == 'THTensor*':
                maybe_scalar = "->maybeScalar({})".format(scalar_check) \
                               if scalar_check is not None \
                               else ""
                return_tensor = "return Tensor((new ${Tensor}(context,${arg_name}))${maybe_scalar},false);"
                body.append(CodeTemplate(return_tensor).substitute(
                    env, arg_name=call, maybe_scalar=maybe_scalar))
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

    for declaration in declarations:
        for option in declaration['options']:
            if not option.get('skip', False) and option['has_full_argument_list']:
                try:
                    process_option(option)
                except NYIError:
                    pass
    return type_object_declarations, type_object_definitions
