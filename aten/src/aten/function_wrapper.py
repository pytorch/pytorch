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
    'THTensor*' : 'Tensor &'
}

TYPE_RETURN = {
    'THTensor*' : 'Tensor *',
    'real': 'Scalar',
    'accreal': 'Scalar'
}
TYPE_ARGUMENT = {
    'THTensor*': CodeTemplate('checked_cast<${THTensor}>(${arg_name})'),
}

RETURN_WRAP = {
'THTensor*': 'new ${Tensor}(context,${returned})'
}

class nested_dict(object):
    def __init__(self,base,parent):
        self.base, self.parent = base,parent
    def __getitem__(self,x):
        r = self.base.get(x)
        if r is not None:
            return r
        return self.parent[x]

def create_generic(top_env, declarations):

    def get_formals(option):
        seen = set()
        result = []
        def insert(argument):
            if argument['name'] not in seen:
                seen.add(argument['name'])
                result.append(argument)
        for argument in option['arguments']:
            if not argument.get('output',False):
                insert(argument)
        for argument in option['arguments']:
            if not argument.get('allocate',False):
                insert(argument)
        return result
    def format_formal(argument):
        type_str = TYPE_FORMAL_GENERIC.get(argument['type'],"NYIType")
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

    def first_tensor(option):
        for argument in option['arguments']:
            if not argument.get('output',False) and argument['type'] == "THTensor*":
                return argument['name']
        return None
    def process_option(option):
        if option['cname'] != 'neg':
            raise NYIError("all not implemented")

        formals = get_formals(option)
        option['formals_list'] = formals
        option['formals'] = [format_formal(f) for f in formals]
        option['actuals'] = [ f['name'] for f in formals ]
        option['method_formals'] = [format_formal(f) for f in formals
             if f['name'] != 'self']
        option['method_actuals'] = [
            f['name'] if f['name'] != 'self' else '*this' for f in formals ]
        option['return_type'] = format_return_type(option)

        option['first_tensor'] = first_tensor(option)
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
        return argument['type'] == 'THTensor*'
    def get_argument(argument):
        if requires_checked_cast(argument):
            return "{}_->tensor".format(argument['name'])
        else:
            return argument['name']

    def get_arguments(option):
        return [get_argument(argument) for argument in option['arguments']]

    def emit_body(env,option):
        body = []
        for arg in option['formals_list']:
            if requires_checked_cast(arg):
                body.append(
                    CodeTemplate("auto ${arg_name}_ = checked_cast<${Tensor}>(&${arg_name});").substitute(
                    env,arg_name=arg['name']))
        for arg in option['arguments']:
            if arg.get('allocate',False):
                body.append(
                    CodeTemplate('auto ${arg_name}_ = new ${Tensor}(context);').substitute(
                    env,arg_name=arg['name']))

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
                body.append("return {};").format(call)
            assert(False and "NYI - return handling")
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
