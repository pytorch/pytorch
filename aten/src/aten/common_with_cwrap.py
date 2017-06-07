# this code should be common among cwrap and TensorLib preprocessing
# for now, I have put it in one place but right now is copied out of cwrap

from copy import deepcopy
from itertools import product

def parse_arguments(args):
    new_args = []
    for arg in args:
        # Simple arg declaration of form "<type> <name>"
        if isinstance(arg, str):
            t, _, name = arg.partition(' ')
            new_args.append({'type': t, 'name': name})
        elif isinstance(arg, dict):
            if 'arg' in arg:
                arg['type'], _, arg['name'] = arg['arg'].partition(' ')
                del arg['arg']
            new_args.append(arg)
        else:
            assert False
    return new_args

def set_declaration_defaults(declaration):
    declaration.setdefault('arguments', [])
    declaration.setdefault('return', 'void')
    if 'cname' not in declaration:
        declaration['cname'] = declaration['name']
    if 'api_name' not in declaration:
        declaration['api_name'] = (declaration['python_name']
            if 'python_name' in declaration else declaration['name'])
    # Simulate multiple dispatch, even if it's not necessary
    if 'options' not in declaration:
        declaration['options'] = [{'arguments': declaration['arguments']}]
        del declaration['arguments']
    # Parse arguments (some of them can be strings)
    for option in declaration['options']:
        option['arguments'] = parse_arguments(option['arguments'])
    # Propagate defaults from declaration to options
    for option in declaration['options']:
        for k, v in declaration.items():
            #TODO(zach): why does cwrap not propagate 'name'? I need it propagaged for TensorLib
            if k != 'options':
                option.setdefault(k, v)

#TODO(zach): added option to remove keyword handling for C++ which cannot support it.
def filter_unique_options(options,allow_kwarg):
    def signature(option, kwarg_only_count):
        if kwarg_only_count == 0:
            kwarg_only_count = None
        else:
            kwarg_only_count = -kwarg_only_count
        arg_signature = '#'.join(
            arg['type']
            for arg in option['arguments'][:kwarg_only_count]
            if not arg.get('ignore_check') and not arg['name'] == 'self')
        if kwarg_only_count is None:
            return arg_signature
        kwarg_only_signature = '#'.join(
            arg['name'] + '#' + arg['type']
            for arg in option['arguments'][kwarg_only_count:]
            if not arg.get('ignore_check'))
        return arg_signature + "#-#" + kwarg_only_signature
    seen_signatures = set()
    unique = []
    for option in options:
        # if only check num_kwarg_only == 0 if allow_kwarg == False
        limit = len(option['arguments']) if allow_kwarg else 0
        for num_kwarg_only in range(0, limit + 1):
            sig = signature(option, num_kwarg_only)
            if sig not in seen_signatures:
                if num_kwarg_only > 0:
                    for arg in option['arguments'][-num_kwarg_only:]:
                        arg['kwarg_only'] = True
                unique.append(option)
                seen_signatures.add(sig)
                break
    return unique

def enumerate_options_due_to_default(declaration,allow_kwarg=True):
    # TODO(zach): in cwrap this is shared among all declarations
    # but seems to assume that all declarations will have the same
    new_options = []
    for option in declaration['options']:
        optional_args = []
        for i, arg in enumerate(option['arguments']):
            if 'default' in arg:
                optional_args.append(i)
        for permutation in product((True, False), repeat=len(optional_args)):
            option_copy = deepcopy(option)
            for i, bit in zip(optional_args, permutation):
                arg = option_copy['arguments'][i]
                if not bit:
                    arg['type'] = 'CONSTANT'
                    arg['ignore_check'] = True
                    # PyYAML interprets NULL as None...
                    arg['name'] = 'NULL' if arg['default'] is None else arg['default']
            new_options.append(option_copy)
    declaration['options'] = filter_unique_options(new_options,allow_kwarg)

def sort_by_number_of_options(declaration):
    def num_checked_args(option):
        return sum(map(lambda a: not a.get('ignore_check', False), option['arguments']))
    declaration['options'].sort(key=num_checked_args, reverse=True)
