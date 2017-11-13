def python_num(s):
    try:
        return int(s)
    except Exception:
        return float(s)

CPP_TO_ATEN_TYPE_MAP = {
  'std::vector<Tensor>': 'TensorList',
}

# change C++ types into ATen types (for output to Declarations.yaml);
# e.g. std::vector<Tensor> -> TensorList, const/reference modifiers on Tensor
# go away and are eventually restored to dynamic_type (this can probably be simplified
# to a single transformation).
def to_aten_type(typ):
    # split tuples into constituents
    if 'std::tuple<' in typ:
        template = typ.split('std::tuple<', 1)[1].rsplit('>', 1)[0]
        type_list = template.split(',')
    else:
        type_list = [typ]
    # remove const/references
    type_list = [t.replace('const ', '').replace('&', '').strip() for t in type_list]
    type_list = [CPP_TO_ATEN_TYPE_MAP.get(t, t) for t in type_list]
    return ','.join(type_list)

def parse_arguments(args):
    arguments = []

    for arg in args.split(','):
        arg = arg.strip()
        if '&' in arg:
            ref_split = arg.rsplit('&', 1)
            t, name = (ref_split[0] + '&').strip(), ref_split[1].strip()
        else:
            t, name = [a.strip() for a in arg.rsplit(' ', 1)]
        default = None

        if '=' in name:
            ns = name.split('=', 1)
            name, default = ns[0], python_num(ns[1])

        argument_dict = {'type': to_aten_type(t), 'name': name}
        if default is not None:
            argument_dict['default'] = default

        arguments.append(argument_dict)
    return arguments

def parse(filename):
    with open(filename, 'r') as file:
        declarations = []
        in_comment_native_decl = False
        in_cpp_native_decl = False
        for line in file.readlines():
            if '[NativeFunction]' in line:
                arguments = []
                dispatch_level = None
                dispatch = None
                decl_parse = ''
                declaration = {'mode': 'native'}
                in_dispatch_table = False
                if '//' == line.strip()[:2] or '*/' in line.split('[NativeFunction]')[1]:
                    in_comment_native_decl = False
                    in_cpp_native_decl = True
                else:
                    in_comment_native_decl = True
                    in_cpp_native_decl = False
            elif '[/NativeFunction]' in line:
                in_comment_native_decl = False
                in_dispatch_table = False
                in_cpp_native_decl = True
            elif in_cpp_native_decl:
                if ';' in line:
                    decl_parse += line.split(';')[0]
                    in_cpp_native_decl = False
                    return_and_name, arguments = decl_parse.split('(')
                    arguments = arguments.split(')')[0]
                    return_type_cpp, declaration['name'] = return_and_name.rsplit(None, 1)
                    declaration['return'] = to_aten_type(return_type_cpp)
                    declaration['arguments'] = parse_arguments(arguments)
                    if dispatch is None:
                        dispatch = 'at::native::' + declaration['name']
                    declaration['type_method_definition_dispatch'] = dispatch
                    if dispatch_level is None:
                        dispatch_level = 'backend' if isinstance(dispatch, dict) else 'base'
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
            elif in_comment_native_decl:
                ls = line.strip().split(':', 1)
                key = ls[0].strip()
                if key == '*/':
                    in_comment_native_decl = False
                    in_dispatch_table = False
                    in_cpp_native_decl = True
                    continue
                elif key == '}':
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
