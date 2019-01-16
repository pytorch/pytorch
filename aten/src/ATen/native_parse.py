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
    elif s == '[]':
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


def sanitize_type(typ):
    if typ == 'Generator*':
        return 'Generator *'
    return typ


def sanitize_types(types):
    # split tuples into constituent list
    if types[0] == '(' and types[-1] == ')':
        return [sanitize_type(x.strip()) for x in types[1:-1].split(',')]
    return [sanitize_type(types)]


def get_type_annotation(tensor_type):
    paren_start = tensor_type.find("(")
    annotation = None
    if paren_start > -1:
        annotation = tensor_type[paren_start:]
        if not annotation.endswith("!)"):
            raise RuntimeError("We only support writing alias anotations for now")
        annotation = annotation.strip()
        tensor_type = tensor_type[:paren_start]
    return tensor_type, annotation


def parse_arguments(args, func_decl, func_name, func_return):
    arguments = []
    kwarg_only = False
    is_out_fn = False

    if len(args.strip()) == 0:
        return arguments, func_name

    # needed in case this is an out function
    output_arguments = []

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

        if '=' in name:
            ns = name.split('=', 1)
            name, default = ns[0], parse_default(ns[1])

        typ = sanitize_types(t)
        assert len(typ) == 1
        argument_dict = {'type': typ[0].rstrip('?'), 'name': name, 'is_nullable': typ[0].endswith('?')}
        match = re.match(r'int\[(\d+)\]', argument_dict['type'])
        if match:
            argument_dict['type'] = 'IntList'
            argument_dict['size'] = int(match.group(1))
        if argument_dict['type'] == 'int[]':
            argument_dict['type'] = 'IntList'
        if default is not None:
            argument_dict['default'] = default
        # TODO: convention is that the ith-argument correspond to the i-th return, but it would
        # be better if we just named everything and matched by name.
        if is_out_fn and arg_idx < len(func_return):
            argument_dict['output'] = True
        argument_dict["type"], argument_dict["annotation"] = get_type_annotation(argument_dict["type"])
        # We assume that type annotations are only used for output tensors
        is_output_arg = False
        for ret in func_return:
            if ret["annotation"] and ret["annotation"] == argument_dict["annotation"]:
                argument_dict["output"] = True
                output_arguments.append(argument_dict)
                assert not is_output_arg
                is_output_arg = True
                break
        if is_output_arg:
            is_out_fn = True
            continue
        if kwarg_only:
            argument_dict['kwarg_only'] = True

        arguments.append(argument_dict)

    arguments = output_arguments + arguments

    if is_out_fn and func_decl.get('variants', []) not in [[], 'function', ['function']]:
        raise RuntimeError("Native functions suffixed with _out MUST be declared with only the function variant; "
                           "e.g., variants: function; otherwise you will tickle a Python argument binding bug "
                           "(which usually manifests itself as the result variable being undefined.) "
                           "The culprit was: {}".format(func_name))
    if is_out_fn:
        func_name += "_out"

    return arguments, func_name


def parse_return_arguments(return_decl, inplace):
    arguments = []
    # TODO: Use a real parser here; this will get bamboozled
    # by signatures that contain things like std::array<bool, 2> (note the space)
    if return_decl[0] == '(' and return_decl[-1] == ')':
        return_decl = return_decl[1:-1]
    multiple_args = len(return_decl.split(', ')) > 1

    for arg_idx, arg in enumerate(return_decl.split(', ')):
        type_and_maybe_name = [a.strip() for a in arg.rsplit(' ', 1)]
        if len(type_and_maybe_name) == 1:
            t = type_and_maybe_name[0]
            if inplace:
                name = 'self'
            else:
                name = 'result' if not multiple_args else 'result' + str(arg_idx)
        else:
            t, name = type_and_maybe_name

        typ = sanitize_type(t)
        argument_dict = {'type': typ, 'name': name}
        argument_dict['output'] = True
        argument_dict["type"], argument_dict["annotation"] = get_type_annotation(argument_dict["type"])

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
                declaration['schema_string'] = "aten::" + func['func']
                if '->' in func['func']:
                    func_decl, return_decl = [x.strip() for x in func['func'].split('->')]
                else:
                    raise Exception('Expected return declaration')
                func_decl = func_decl.strip()
                fn_name_start = func_decl.find("(")
                fn_name = func_decl[:fn_name_start]
                arguments = func_decl[fn_name_start + 1:-1]
                declaration['name'] = func.get('name', fn_name)
                declaration['inplace'] = re.search('(^__i|[^_]_$)', fn_name) is not None
                return_arguments = parse_return_arguments(return_decl, declaration['inplace'])
                arguments, declaration['name'] = parse_arguments(arguments, func, declaration['name'], return_arguments)
                output_arguments = [x for x in arguments if x.get('output')]
                declaration['return'] = return_arguments if len(output_arguments) == 0 else output_arguments
                declaration['variants'] = func.get('variants', ['function'])
                declaration['requires_tensor'] = func.get('requires_tensor', False)
                declaration['matches_jit_signature'] = func.get('matches_jit_signature', False)
                declaration['cpu_half'] = func.get('cpu_half', False)
                declaration['deprecated'] = func.get('deprecated', False)
                declaration['device_guard'] = func.get('device_guard', True)
                declaration['arguments'] = func.get('arguments', arguments)
                declaration['type_method_definition_dispatch'] = func.get('dispatch', declaration['name'])
                declaration['python_module'] = func.get('python_module', '')
                declaration['aten_sparse'] = has_sparse_dispatches(
                    declaration['type_method_definition_dispatch'])
                # The annotation field isn't required for further processing at this point
                # so we remove it to produce the exact same code gen as without it.
                for argument in declaration['arguments']:
                    if 'annotation' in argument:
                        del argument['annotation']
                declarations.append(declaration)
            except Exception as e:
                msg = '''Exception raised in processing function:
{func}
Generated partial declaration:
{decl}'''.format(func=pprint.pformat(func), decl=pprint.pformat(declaration))
                print(msg, file=sys.stderr)
                raise e

    return declarations
