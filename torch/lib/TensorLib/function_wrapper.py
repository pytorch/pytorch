import re
import yaml
from code_template import CodeTemplate

# what has to be done to add a Operation ...
# 1. add virtual dispatch declaration to Type.h and default impl to Type.cpp
TYPE_METHOD_DECLARATION = CodeTemplate("""\
virtual ${return_type} ${api_name}(${formals});
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
${return_type} ${api_name}(${method_formals});
""")
# 5. add non-virtual declaration to Tensor.cpp
TENSOR_METHOD_DEFINITION = CodeTemplate("""\
${return_type} Tensor::${api_name}(${method_formals}) {
    return type().${api_name}(${method_actuals});
}
""")
# 6. add a method declaration in Functions.h
FUNCTION_DECLARATION = CodeTemplate("""\
${return_type} ${api_name}(${formals});
""")
# 7. add a method definition in Functions.cpp
FUNCTION_DEFINITION = CodeTemplate("""\
${return_type} ${api_name}(${formals}) {
    return ${first_tensor}.type().${api_name}(${actuals});
}
""")

class NYIError(Exception):
    """Indicates we don't support this declaration yet"""
    def __init__(self,reason):
        self.reason = reason

TYPE_FORMAL_GENERIC = {
    'THTensor*' : 'Tensor &',
    'THBoolTensor*': 'Tensor &',
    'THStorage*' : 'Storage &',
    'THIndexTensor*' : 'Tensor &',
    'THGenerator*': 'Generator &',
    'THSize*': 'IntList',
    'THStride*': 'IntList',
    'accreal' : 'Scalar',
    'real' : 'Scalar',
}

TYPE_RETURN = {
    'THTensor*' : 'Tensor *',
    'THIndexTensor*' : 'Tensor *',
    'THBoolTensor*' : 'Tensor *',
    'real': 'Scalar',
    'accreal': 'Scalar',
}
CHECKED_CAST = {
    'THTensor*': CodeTemplate('checked_cast<${Tensor}>(&${arg_name})'),
    'THIndexTensor*' : CodeTemplate('checked_cast<${THIndexTensor}>(&${arg_name})'),
    'THSize*' : CodeTemplate('THStorageView::make(${arg_name})'),
    'THStride*' : CodeTemplate('THStorageView::make(${arg_name})'),
}
CHECKED_USE = {
    'THTensor*': '{}_->tensor',
    'THIndexTensor*' : '{}_->tensor',
    'THSize*' : '{}_',
    'THStride*' : '{}_',
}

RETURN_WRAP = {
'THTensor*': 'new ${Tensor}(context,${returned})'
}

CONSTANT_REPLACEMENTS = [
    ('AS_REAL','${ScalarType}'),
    ('THPDefaultGenerator->cdata','dynamic_cast<${Processor}Generator*>(context->defaultGenerator(processor())->generator'),
]

class nested_dict(object):
    def __init__(self,base,parent):
        self.base, self.parent = base,parent
    def __getitem__(self,x):
        r = self.base.get(x)
        if r is not None:
            return r
        return self.parent[x]

def create_generic(top_env, declarations):

    def is_real_argument_to_wrapper(argument):
        return not argument.get('output',False) and\
            argument['type'] != 'CONSTANT' and\
            argument['type'] != 'argument'
    def get_formals(option):
        seen = set()
        result = []
        def insert(argument):
            if argument['name'] not in seen:
                seen.add(argument['name'])
                result.append(argument)
        for argument in option['arguments']:
            if is_real_argument_to_wrapper(argument):
                insert(argument)
        for argument in option['arguments']:
            if argument.get('output') and not argument.get('allocate',False):
                insert(argument)
        return result
    def format_formal(argument):
        type_str = TYPE_FORMAL_GENERIC.get(argument['type'],argument['type'])
        return '{} {}'.format(type_str,argument['name'])

    def format_return_type(option):
        ret = option['return']
        if ret['kind'] == 'arguments':
            #TODO multiple returns
            index = ret['arguments'][0]
            the_type = option['arguments'][index]['type']
        elif ret['kind'] == 'type':
            the_type = ret['type']
        else:
            raise Exception("format_return_type")
        return TYPE_RETURN.get(the_type,the_type)

    def first_tensor(formals):
        for argument in formals:
            if argument['type'] == "THTensor*":
                return argument['name']
        return None
    def process_option(option):
        if option['name'] != 'zeros':
            raise NYIError("NYI")

        formals = get_formals(option)
        option['formals_list'] = formals
        option['formals'] = [format_formal(f) for f in formals]
        option['actuals'] = [ f['name'] for f in formals ]
        option['method_formals'] = [format_formal(f) for f in formals
             if f['name'] != 'self']
        option['method_actuals'] = [
            f['name'] if f['name'] != 'self' else '*this' for f in formals ]
        option['return_type'] = format_return_type(option)

        option['first_tensor'] = first_tensor(formals)
        env = nested_dict(option,top_env)
        top_env['type_method_declarations'].append(
            TYPE_METHOD_DECLARATION.substitute(env))
        top_env['type_method_definitions'].append(
            TYPE_METHOD_DEFINITION.substitute(env))

        if 'method' in option['variants']:
            top_env['tensor_method_declarations'].append(
                TENSOR_METHOD_DECLARATION.substitute(env))
            top_env['tensor_method_definitions'].append(
                TENSOR_METHOD_DEFINITION.substitute(env))

        if ('function' in option['variants'] and
            option['first_tensor'] is not None):
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

def create_derived(processor_type_env,declarations):
    type_object_declarations = []
    type_object_definitions = []
    def requires_checked_cast(argument):
        return argument['type'] in CHECKED_CAST
    def get_argument(argument,option):
        if requires_checked_cast(argument):
            return CHECKED_USE[argument['type']].format(argument['name'])
        elif argument['type'] == "CONSTANT":
            v = str(argument['name'])
            for pattern,replacement in CONSTANT_REPLACEMENTS:
                v = re.sub(pattern, replacement, v)
            return CodeTemplate(v).substitute(processor_type_env)
        # e.g. argument 0, i.e. repeat the 0th argument in this position...
        elif argument['type'] == 'argument':
            index = int(argument['name'])
            return get_argument(option['arguments'][index],option)
        else:
            return argument['name']
    def drop_argument(argument):
        return argument['type'] == 'THGenerator*' and processor_type_env['Processor'] == 'CUDA'
    def get_arguments(option):
        return [get_argument(argument,option)
            for argument in option['arguments'] if not drop_argument(argument)]

    def emit_body(env,option):
        body = []
        for arg in option['arguments']:
            if requires_checked_cast(arg):
                if arg.get('allocate',False):
                    body.append(
                        CodeTemplate('auto ${arg_name}_ = new ${Tensor}(context);').substitute(
                        env,arg_name=arg['name']))
                else:
                    check_cast = CHECKED_CAST[arg['type']].substitute(env,arg_name=arg['name'])
                    body.append("auto {}_ = {};".format(arg['name'],check_cast))

        option['actuals'] = processor_type_env['state'] + get_arguments(option)
        call = CodeTemplate("${THTensor}_${cname}(${actuals})").substitute(env)
        ret = option['return']
        if ret['kind'] == 'arguments':
            arg = option['arguments'][ret['arguments'][0]]
            body.append(call+";")
            body.append("return {}_;".format(arg['name']))
        elif ret['kind'] == 'type':
            if ret['type'] == 'THTensor*':
                body.append(CodeTemplate("return new ${Tensor}(context,${arg_name});").substitute(env,arg_name=call))
            else:
                body.append("return {};".format(call))
        else:
            raise Exception("NYI - return handling")
        return body

    def process_option(option):
        pair = (processor_type_env['Processor'],processor_type_env['ScalarName'])
        if pair in option['type_processor_pairs']:
            env = nested_dict(option, processor_type_env)
            body = emit_body(env,option)
            option['type_definition_body'] = body
            type_object_declarations.append(
                TYPE_DERIVED_DECLARATION.substitute(env))
            type_object_definitions.append(
                TYPE_DERIVED_DEFINITION.substitute(env))


    for declaration in declarations:
        for option in declaration['options']:
            if not option.get('skip',False):
                try:
                    process_option(option)
                except NYIError as e:
                    pass
    return type_object_declarations,type_object_definitions
