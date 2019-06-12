from __future__ import print_function
import re
import yaml
import pprint
import sys

try:
    # use faster C loader if available
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def parse_default(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    elif s == 'nullptr':
        return s
    elif s == '{}':
        return '{}'
    elif re.match(r'{.*}', s):
        return s
    elif s == 'None':
        return 'c10::nullopt'
    try:
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            return s


def sanitize_types(typ):
    # split tuples into constituent list
    if typ[0] == '(' and typ[-1] == ')':
        return [x.strip() for x in typ[1:-1].split(',')]
    elif typ == 'Generator*':
        return ['Generator *']
    return [typ]


def parse_arguments(args, func_decl, func_name, func_return):
    arguments = []
    python_default_inits = func_decl.get('python_default_init', {})
    is_out_fn = func_name.endswith('_out')
    if is_out_fn and func_decl.get('variants', []) not in [[], 'function', ['function']]:
        raise RuntimeError("Native functions suffixed with _out MUST be declared with only the function variant; "
                           "e.g., variants: function; otherwise you will tickle a Python argument binding bug "
                           "(which usually manifests itself as the result variable being undefined.) "
                           "The culprit was: {}".format(func_name))
    kwarg_only = False

    if len(args.strip()) == 0:
        return arguments

    # TODO: Use a real parser here; this will get bamboozled
    # by signatures that contain things like std::array<bool, 2> (note the space)
    for arg_idx, arg in enumerate(args.split(', ')):
        type_and_name = [a.strip() for a in arg.rsplit(' ', 1)]
        if type_and_name == ['*']:
            assert not kwarg_only
            kwarg_only = True
            continue

        t, name = type_and_name
        default = None
        python_default_init = None

        if '=' in name:
            ns = name.split('=', 1)
            name, default = ns[0], parse_default(ns[1])

        if name in python_default_inits:
            assert default is None
            python_default_init = python_default_inits[name]

        typ = sanitize_types(t)
        assert len(typ) == 1
        argument_dict = {'type': typ[0].rstrip('?'), 'name': name, 'is_nullable': typ[0].endswith('?')}
        match = re.match(r'IntList\[(\d+)\]', argument_dict['type'])
        if match:
            argument_dict['type'] = 'IntList'
            argument_dict['size'] = int(match.group(1))
        if default is not None:
            argument_dict['default'] = default
        if python_default_init is not None:
            argument_dict['python_default_init'] = python_default_init
        # TODO: convention is that the ith-argument correspond to the i-th return, but it would
        # be better if we just named everything and matched by name.
        if is_out_fn and arg_idx < len(func_return):
            argument_dict['output'] = True
        if kwarg_only:
            argument_dict['kwarg_only'] = True

        arguments.append(argument_dict)
    return arguments


def has_sparse_dispatches(dispatches):
    for dispatch in dispatches:
        if 'Sparse' in dispatch:
            return True
    return False


def parse_native_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=Loader)


def run(paths):
    declarations = []
    for path in paths:
        for func in parse_native_yaml(path):
            declaration = {'mode': 'native'}
            try:
                if '->' in func['func']:
                    func_decl, return_type = [x.strip() for x in func['func'].split('->')]
                    return_type = sanitize_types(return_type)
                else:
                    func_decl = func['func']
                    return_type = [None]
                fn_name, arguments = func_decl.split('(')
                arguments = arguments.split(')')[0]
                declaration['name'] = func.get('name', fn_name)
                return_type = list(func.get('return', return_type))
                arguments = parse_arguments(arguments, func, declaration['name'], return_type)
                output_arguments = [x for x in arguments if x.get('output')]
                declaration['return'] = return_type if len(output_arguments) == 0 else output_arguments
                declaration['variants'] = func.get('variants', ['function'])
                declaration['requires_tensor'] = func.get('requires_tensor', False)
                declaration['cpu_half'] = func.get('cpu_half', False)
                declaration['deprecated'] = func.get('deprecated', False)
                declaration['device_guard'] = func.get('device_guard', True)
                declaration['arguments'] = func.get('arguments', arguments)
                declaration['type_method_definition_dispatch'] = func.get('dispatch', declaration['name'])
                declaration['aten_sparse'] = has_sparse_dispatches(
                    declaration['type_method_definition_dispatch'])
                declarations.append(declaration)
            except Exception as e:
                msg = '''Exception raised in processing function:
{func}
Generated partial declaration:
{decl}'''.format(func=pprint.pformat(func), decl=pprint.pformat(declaration))
                print(msg, file=sys.stderr)
                raise e

    return declarations
