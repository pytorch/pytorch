import re
import yaml
from code_template import CodeTemplate

# temporary things we cannot handle
EXCLUDE_PATTERN = "bernoulli.*|normal.*|exponential.*|random.*"
# what has to be done to add a Operation ...
# 1. add virtual dispatch declaration to Type.h and default impl to Type.cpp
TYPE_METHOD_DECLARATION = CodeTemplate("""\
virtual ${return_type} ${api_name}(${formals}) ;
""")
TYPE_METHOD_DEFINITION = CodeTemplate("""\
${return_type} Type::${api_name}(${formals}) {
    throw std::runtime_error(std::string("${api_name} is not implemented for type ") + toString());
}
""")
# 2. add virtual override to TypeDerived.h
TYPE_DERIVED_DECLARATION = CodeTemplate("""\
virtual ${return_type} ${api_name}(${formals}) override;
""")
# 3. add override definition to TypeDerived.cpp
TYPE_DERIVED_DEFINITION = CodeTemplate("""\
${return_type} ${Type}::${api_name}(${formals}) {
    ${type_definition_body}
}
""")
# 4. add non-virtual declaration to Tensor.h
TENSOR_METHOD_DECLARATION = CodeTemplate("""\
${return_type} ${api_name}(${method_formals}) ${const_mark};
""")
# 5. add non-virtual declaration to Tensor.cpp
TENSOR_METHOD_DEFINITION = CodeTemplate("""\
inline ${return_type} Tensor::${api_name}(${method_formals}) ${const_mark} {
    return type().${api_name}(${method_actuals});
}
""")
# 6. add a method declaration in Functions.h
FUNCTION_DECLARATION = CodeTemplate("""\
static inline ${return_type} ${api_name}(${formals});
""")
# 7. add a method definition in Functions.cpp
FUNCTION_DEFINITION = CodeTemplate("""\
static inline ${return_type} ${api_name}(${formals}) {
    return ${inferred_type}.${api_name}(${actuals});
}
""")


class NYIError(Exception):
    """Indicates we don't support this declaration yet"""

    def __init__(self, reason):
        self.reason = reason


TYPE_FORMAL_GENERIC = {
    'THTensor*': 'Tensor &',
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
    'THTensor*': CodeTemplate('checked_cast<${Tensor}>(${arg_name}.pImpl,"${arg_name}",${arg_pos})'),
    'THBoolTensor*': CodeTemplate('checked_cast<${Backend}ByteTensor>(${arg_name}.pImpl,"${arg_name}",${arg_pos})'),
    'THIndexTensor*': CodeTemplate('checked_cast<${Backend}LongTensor>(${arg_name}.pImpl,"${arg_name}",${arg_pos})'),
    'THIntegerTensor*': CodeTemplate('checked_cast<${Backend}IntTensor>(${arg_name}.pImpl,"${arg_name}",${arg_pos})'),
    'THStorage*': CodeTemplate('checked_cast<${Storage}>(&${arg_name},"${arg_name}",${arg_pos})'),
    'THGenerator*': CodeTemplate('check_generator(&${arg_name})'),
    'THSize*': CodeTemplate('THLongStorageView::make(${arg_name})'),
    'THStride*': CodeTemplate('THLongStorageView::make(${arg_name})'),
    'real': CodeTemplate('${arg_name}.to${ScalarName}()'),
    'accreal': CodeTemplate('${arg_name}.to${AccScalarName}()'),

}
CHECKED_USE = {
    'THTensor*': '{}_->tensor',
    'THIndexTensor*': '{}_->tensor',
    'THBoolTensor*': '{}_->tensor',
    'THIntegerTensor*': '{}_->tensor',
    'THStorage*': '{}_->storage',
    'THGenerator*': '{}_->generator',
}

ALLOC_WRAP = {
    'THTensor*': 'new ${Tensor}(context)',
    'THBoolTensor*': 'new ${Backend}ByteTensor(context)',
    'THIndexTensor*': 'new ${Backend}LongTensor(context)',
    'THIntegerTensor*': 'new ${Backend}IntTensor(context)',
}

CONSTANT_REPLACEMENTS = [
    ('AS_REAL', '${AS_REAL}'),
    ('THPDefaultGenerator->cdata',
     'dynamic_cast<${Backend}Generator&>(context->defaultGenerator(backend())).generator'),
    ('__storage_size.get\\(\\)',
     'THLongStorageView::make(static_cast<int64_t>(storage.size()))'),
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


def to_return_type(t):
    return TYPE_RETURN.get(t, t)


def create_generic(top_env, declarations):

    def get_formals(option):
        seen = set()
        result = []

        def insert(argument):
            if argument['name'] not in seen:
                seen.add(argument['name'])
                result.append(argument)
        for argument in option['arguments']:
            if argument['type'] == 'THSTensor*':
                raise NYIError("Sparse Tensor")
            if is_real_argument_to_wrapper(argument):
                insert(argument)
        for argument in option['arguments']:
            if argument.get('output') and not argument.get('allocate', False):
                insert(argument)
        return result

    def format_formal(argument):
        type_str = TYPE_FORMAL_GENERIC.get(argument['type'], argument['type'])
        if type_str == 'Tensor &' and not argument.get('output'):
            type_str = 'const ' + type_str
        return '{} {}'.format(type_str, argument['name'])

    def format_return_type(option):
        ret = option['return']
        if ret['kind'] == 'arguments':
            argument_indices = ret['arguments']
            if len(argument_indices) == 1:
                the_type = option['arguments'][argument_indices[0]]['type']
                return to_return_type(the_type)
            else:
                types = [to_return_type(option['arguments'][idx]['type'])
                         for idx in argument_indices]
                return "std::tuple<{}>".format(','.join(types))

        elif ret['kind'] == 'type':
            return to_return_type(ret['type'])
        else:
            raise Exception("format_return_type")

    def find_first_tensor(formals):
        for argument in formals:
            if argument['type'] == "THTensor*":
                return argument['name']
        return None

    def process_option(option):
        if re.match(EXCLUDE_PATTERN, option['name']):
            print("Excluding {}".format(option['name']))
            raise NYIError("NYI")
        # print(yaml.dump(option))
        formals = get_formals(option)
        option['formals_list'] = formals
        option['formals'] = [format_formal(f) for f in formals]
        option['actuals'] = [f['name'] for f in formals]
        option['method_formals'] = [format_formal(f) for f in formals
                                    if f['name'] != 'self']
        option['method_actuals'] = [
            f['name'] if f['name'] != 'self' else '*this' for f in formals]
        option['return_type'] = format_return_type(option)
        option['const_mark'] = 'const' if option.get('const') else ''
        env = nested_dict(option, top_env)
        top_env['type_method_declarations'].append(
            TYPE_METHOD_DECLARATION.substitute(env))
        top_env['type_method_definitions'].append(
            TYPE_METHOD_DEFINITION.substitute(env))

        if 'method' in option['variants']:
            top_env['tensor_method_declarations'].append(
                TENSOR_METHOD_DECLARATION.substitute(env))
            top_env['tensor_method_definitions'].append(
                TENSOR_METHOD_DEFINITION.substitute(env))

        if 'function' in option['variants']:
            first_tensor = find_first_tensor(formals)
            if first_tensor is not None:
                option['inferred_type'] = '{}.type()'.format(first_tensor)
            else:
                option['inferred_type'] = 'globalContext()->defaultType()'
            top_env['function_declarations'].append(
                FUNCTION_DECLARATION.substitute(env))
            top_env['function_definitions'].append(
                FUNCTION_DEFINITION.substitute(env))

    for declaration in declarations:
        for option in declaration['options']:
            try:
                process_option(option)
            except NYIError as e:
                option['skip'] = True


def create_derived(backend_type_env, declarations):
    type_object_declarations = []
    type_object_definitions = []

    def requires_checked_cast(argument):
        return argument['type'] in CHECKED_CAST

    def get_argument(argument, option):
        if requires_checked_cast(argument):
            return CHECKED_USE.get(argument['type'], '{}_').format(argument['name'])
        elif argument['type'] == 'bool' and 'if_true' in argument:
            return '({}) ? "{}" : "{}"'.format(argument['name'],
                                               argument['if_true'], argument['if_false'])
        elif argument['type'] == "CONSTANT":
            if 'if_true' in argument:  # this was a bool that is actually a string...
                return '"{}"'.format(argument['name'])
            v = str(argument['name'])
            for pattern, replacement in CONSTANT_REPLACEMENTS:
                v = re.sub(pattern, replacement, v)
            return CodeTemplate(v).substitute(backend_type_env)
        # e.g. argument 0, i.e. repeat the 0th argument in this position...
        elif argument['type'] == 'argument':
            index = int(argument['name'])
            return get_argument(option['arguments'][index], option)
        else:
            return argument['name']

    def drop_argument(argument):
        return backend_type_env['Backend'] == 'CUDA' and (
            argument['type'] == 'THGenerator*' or
            argument['name'] == 'THPDefaultGenerator->cdata')

    def get_arguments(option):
        return [get_argument(argument, option)
                for argument in option['arguments'] if not drop_argument(argument)]

    def is_actual_return_long(ret):
        return ret['type'] == 'long' or (backend_type_env['ScalarName'] == 'Long'
                                         and ret['type'] == 'real' or ret['type'] == 'accreal')

    def emit_body(env, option):
        body = []
        # arguments are potentially duplicated because of one argument
        # referencing another
        seen_names = set()
        # only generated checked casts the first time we see it
        count = 0
        for arg in option['arguments']:
            if is_real_argument_to_wrapper(arg):
                count += 1
            if not arg['name'] in seen_names and requires_checked_cast(arg):
                seen_names.add(arg['name'])
                if arg.get('allocate', False):
                    allocation = CodeTemplate(
                        ALLOC_WRAP[arg['type']]).substitute(env)
                    body.append('auto {}_ = {};'.format(
                        arg['name'], allocation))
                    body.append('auto {} = Tensor({}_,false);'.format(
                        arg['name'], arg['name']))
                else:
                    check_cast = CHECKED_CAST[arg['type']].substitute(
                        env, arg_name=arg['name'], arg_pos=count)
                    body.append("auto {}_ = {};".format(
                        arg['name'], check_cast))
                if 'resize' in arg:
                    resize = arg['resize']
                    if type(resize) == str:
                        body.append("{}.resize_as_({});".format(
                            arg['name'], resize))
                    else:
                        dims = ['{}.size({})'.format(name, dim)
                                for name, dim in resize]
                        body.append("{}.resize_({{ {} }});".format(
                            arg['name'], ','.join(dims)))
                if arg.get('cpu_zero', False):
                    body.append("{}.zero_();".format(arg['name']))

        option['actuals'] = backend_type_env['state'] + get_arguments(option)
        call = CodeTemplate("${THTensor}_${cname}(${actuals})").substitute(env)
        ret = option['return']
        if ret['kind'] == 'arguments':
            body.append(call + ";")
            arguments_indices = ret['arguments']
            if len(arguments_indices) == 1:
                arg = option['arguments'][arguments_indices[0]]
                body.append("return {};".format(arg['name']))
            else:
                arguments = [option['arguments'][argi]
                             for argi in arguments_indices]
                types = [TYPE_RETURN[arg['type']] for arg in arguments]
                # TODO: check for move semantics...
                names = [arg['name'] for arg in arguments]
                body.append(CodeTemplate("return std::tuple<${types}>(${names});").substitute(
                    types=types, names=names))
        elif ret['kind'] == 'type':
            if ret['type'] == 'THTensor*':
                body.append(CodeTemplate(
                    "return Tensor(new ${Tensor}(context,${arg_name}),false);").substitute(env, arg_name=call))
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
            if not option.get('skip', False):
                try:
                    process_option(option)
                except NYIError as e:
                    pass
    return type_object_declarations, type_object_definitions
