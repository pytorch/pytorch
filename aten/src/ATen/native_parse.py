import yaml

try:
    # use faster C loader if available
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def python_num(s):
    try:
        return int(s)
    except Exception:
        return float(s)


def sanitize_types(typ):
    # split tuples into constituent list
    if typ[0] == '(' and typ[-1] == ')':
        type_list = [x.strip() for x in typ[1:-1].split(',')]
    else:
        type_list = [typ]
    return type_list


def parse_arguments(args):
    arguments = []

    for arg in args.split(','):
        t, name = [a.strip() for a in arg.rsplit(' ', 1)]
        default = None

        if '=' in name:
            ns = name.split('=', 1)
            name, default = ns[0], python_num(ns[1])

        typ = sanitize_types(t)
        assert len(typ) == 1
        argument_dict = {'type': typ[0], 'name': name}
        if default is not None:
            argument_dict['default'] = default

        arguments.append(argument_dict)
    return arguments


def parse_native_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=Loader)


def run(paths):
    declarations = []
    for path in paths:
        for func in parse_native_yaml(path):
            declaration = {'mode': 'native'}
            if '->' in func['func']:
                func_decl, return_type = [x.strip() for x in func['func'].split('->')]
                return_type = sanitize_types(return_type)
            else:
                func_decl = func['func']
                return_type = None
            fn_name, arguments = func_decl.split('(')
            arguments = arguments.split(')')[0]
            declaration['name'] = func.get('name', fn_name)
            declaration['return'] = list(func.get('return', return_type))
            declaration['variants'] = func.get('variants', ['method', 'function'])
            declaration['template_scalar'] = func.get('template_scalar')
            declaration['arguments'] = func.get('arguments', parse_arguments(arguments))
            declaration['type_method_definition_dispatch'] = func.get('dispatch', declaration['name'])
            declarations.append(declaration)

    return declarations
