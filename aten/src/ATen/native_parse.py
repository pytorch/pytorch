def python_num(s):
    try:
        return int(s)
    except Exception:
        return float(s)

CPP_TO_ATEN_TYPE_MAP = {
  'std::vector<Tensor>': 'TensorList',
  'std::tuple<Tensor, Tensor>': 'Tensor, Tensor',
  'const Tensor&': 'Tensor',
  'Tensor&': 'Tensor',
}

def parse_arguments(args):
    arguments = []

    for arg in args.split(','):
        arg = arg.strip()
        t, name = arg.rsplit(' ', 1)
        default = None

        if '=' in name:
            ns = name.split('=', 1)
            name, default = ns[0], python_num(ns[1])

        argument_dict = {'type': CPP_TO_ATEN_TYPE_MAP.get(t, t), 'name': name}
        if default is not None:
            argument_dict['default'] = default

        arguments.append(argument_dict)
    return arguments

def parse(filename):
    with open(filename, 'r') as file:
        declarations = []
        in_declaration = False
        in_dispatch_table = False
        in_decl_parse = False
        for line in file.readlines():
            if '[NativeFunction]' in line:
                in_declaration = True
                arguments = []
                dispatch_level = 'base'
                dispatch = None
                decl_parse = ''
                declaration = {'mode': 'native'}
            elif '[/NativeFunction]' in line:
                in_declaration = False
                in_dispatch_table = False
                in_decl_parse = True
            elif in_decl_parse:
                if ';' in line:
                    decl_parse += line.split(';')[0]
                    in_decl_parse = False
                    return_and_name, arguments = decl_parse.split('(')
                    arguments = arguments.split(')')[0]
                    declaration['name'] = return_and_name.split(' ')[-1]
                    return_type_cpp = return_and_name.rsplit(maxsplit=1)[0]
                    declaration['return'] = CPP_TO_ATEN_TYPE_MAP.get(return_type_cpp, return_type_cpp)
                    declaration['arguments'] = parse_arguments(arguments)
                    declaration['type_method_definition_dispatch'] = dispatch
                    declaration['type_method_definition_level'] = dispatch_level
                    if declaration.get('variants', None) is None:
                        declaration['variants'] = ['method', 'function']
                    type_method_definition_level = declaration.get('type_method_definition_level')
                    if type_method_definition_level != 'base' and type_method_definition_level != 'backend':
                        raise RuntimeError("Native functions currently only support (and must be specified with) "
                                           "\'base\' or \'backend\' type_method_definition_level, got {}"
                                           .format(type_method_definition_level))
                    declarations.append(declaration)
                elif line == '*/\n':
                    pass
                else:
                    decl_parse += line
            elif in_declaration:
                ls = line.strip().split(':', 1)
                key = ls[0].strip()
                if key == "}":
                    assert in_dispatch_table
                    in_dispatch_table = False
                    continue

                value = ls[1].strip()
                if key == 'variants':
                    declaration[key] = [x.strip() for x in value.split(',')]
                elif key == 'type_method_definition_level':
                    dispatch_level = value
                elif key == 'type_method_definition_dispatch':
                    if value == '{':
                        dispatch = {}
                        in_dispatch_table = True
                    else:
                        dispatch = value
                elif in_dispatch_table:
                    key = key.split('-', 1)[1].strip()
                    dispatch[key] = value
                else:
                    declaration[key] = value
        return declarations
