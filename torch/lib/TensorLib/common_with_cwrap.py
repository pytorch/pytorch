# this code should be common among cwrap and TensorLib preprocessing
# for now, I have put it in one place but right now is copied out of cwrap

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
            if k != 'name' and k != 'options':
                option.setdefault(k, v)
