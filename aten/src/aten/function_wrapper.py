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
    'THTensor*' : 'Tensor *'
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
            if argument.get('allocate',False):
                insert(argument)
        return result
    def format_formal(argument):
        return '{} {}'.format(TYPE_FORMAL_GENERIC[argument['type']],argument['name'])

    def format_return_type(option):
        ret = option['return']
        m = re.match('argument (\d+)',ret)
        if m is not None:
            argument = option['arguments'][int(m.group(1))]
        elif ret == 'self':
            argument = [x for x in option['arguments'] if x['name'] == 'self'][0]
        else:
            raise Exception("format_return_type")
        return TYPE_RETURN[argument['type']]

    def first_tensor(option):
        for argument in option['arguments']:
            if not argument.get('output',False) and argument['type'] == "THTensor*":
                return argument['name']
        return None
    def process_option(option):
        if option['cname'] != 'neg':
            raise NYIError("all not implemented")

        formals = get_formals(option)
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
    def emit_body(option):
        print(yaml.dump(option))
        return []
    def process_option(option):
        pair = (processor_type_env['Processor'],processor_type_env['ScalarName'])
        if pair in option['type_processor_pairs']:
            body = emit_body(option)
            option['type_definition_body'] = body
            env = nested_dict(option, processor_type_env)
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
