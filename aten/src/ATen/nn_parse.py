import copy
import re
import common_with_cwrap
import yaml
from collections import OrderedDict, defaultdict

try:
    # use faster C loader if available
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


# matches `name`, `params` in `name(params)`
NAME_PARAM_REGEX = r'(\w+)\((.*)\)'


def argument_to_declaration(param, func=None):
    arg = {}
    arg['type'], name = param.split(' ')
    if (arg['type'].endswith('?')):
        arg['is_nullable'] = True
        arg['type'] = arg['type'].rstrip('?')
    if arg['type'] == 'Tensor':
        arg['type'] = 'THTensor*'
    elif arg['type'] == 'LongTensor':
        arg['type'] = 'THIndexTensor*'
    elif arg['type'] == 'Scalar':
        arg['type'] = 'accreal'
    elif arg['type'] == 'Generator*':
        arg['type'] = 'THGenerator*'

    match = re.match(r'IntArrayRef\[(\d+)\]', arg['type'])
    if match:
        arg['type'] = 'IntArrayRef'
        arg['size'] = int(match.group(1))

    if '=' in name:
        name, default = name.split('=')
        arg['optional'] = True
        arg['default'] = default
    arg['name'] = name

    if func is not None:
        default_inits = func.get('default_init', {})
        wrap_dims = func.get('wrap_dim', {})
        if name in default_inits:
            # non constexpr defaults
            arg['default_init'] = default_inits[name]
        if name in wrap_dims:
            arg['wrap_dim'] = wrap_dims[name]

    return arg


def output_arguments(thnn_function):
    cname = thnn_function.name
    output_args = []

    # function_wrapper expects everything in a declaration to be in
    # the base type (i.e. THTensor*), but if we pull a THCUNN only
    # implementation, it will have THCTensor* as the arg type. So we
    # strip the THC here before returning
    def map_to_th_type(t):
        if t.startswith('THC'):
            t = t.replace('THC', 'TH')
        return t

    def is_output_arg(arg_name, func_name):
        if arg_name == 'output' and 'updateOutput' in cname:
            return True
        if name in {'gradInput', 'gradWeight', 'gradBias', 'gradGrid'}:
            return True
        if arg_name == 'indices' and 'updateOutput' in cname and 'Unpool' not in cname:
            # indices is an output argument in pooling and an input in unpooling
            return True
        return False

    for arg in thnn_function.arguments:
        name = arg.name
        if is_output_arg(name, cname):
            desc = {
                'type': map_to_th_type(arg.type),
                'name': camel_to_snake(name),
                'output': True,
            }
            if name.startswith('grad_'):
                desc['is_nullable'] = True
            output_args.append(desc)
    return output_args


def get_return(args):
    indices = [str(idx) for idx, arg in enumerate(args) if arg.get('output')]
    return 'argument {}'.format(','.join(indices))


ARGUMENT_MAPPINGS = {
    'k': 'kernel_size',
    'd': 'stride',
    'pad': 'padding',
    'p': 'padding',
    'o': 'output_size',
    'osize': 'output_size',
    'output': 'output_size',  # as a prefix e.g. outputW
    'isize': 'input_size',
    'dilation': 'dilation',
    'adj': 'output_padding',
    'a': 'output_padding',
}

DIMENSION_OFFSET = {
    'width': -1,
    'height': -2,
    'B': 0,
    'C': 1,
    'W': -1,
    'H': -2,
    'T': -3,
    'left': 0,
    'right': 1,
    'top': 2,
    'bottom': 3,
    'front': 4,
    'back': 5,
}

SUBSTITUTIONS = {
    'input': 'self',
    'weights': 'weight',
    'train': 'training',
    'val': 'value',
    'lambda': 'lambd',
    'negval': 'negative_slope',
}


def camel_to_snake(name):
    # from https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def get_thnn_args(thnn_function, params, inplace):
    params_by_name = {p['name']: p for p in params}

    def arg_expr(prefix, suffix):
        # e.g kW, kH
        name = ARGUMENT_MAPPINGS[prefix]
        if name not in params_by_name:
            raise RuntimeError('missing arg "{}" in {}'.format(name, thnn_function.name))
        param = params_by_name[name]
        if param['type'] == 'IntArrayRef' and 'size' in param:
            name = name + '_'
        # NB: We calculate the dimension based on the name of
        # the argument, not its positional order.  This means
        # that we may reorder arguments to get them in
        # the right place; e.g., if a THNN implementation
        # has arguments in the order kernelW, kernelH, we
        # will generate a caller that is kernel[1], kernel[0]
        # to order them in the correct way.
        index = DIMENSION_OFFSET[suffix]
        if index < 0:
            index += param['size']
        expr = '{}[{}]'.format(name, index)
        return {'type': 'EXPRESSION', 'name': expr}

    thnn_args = []
    for arg in thnn_function.arguments:
        name = arg.name
        if name == 'state':
            continue
        if inplace and name == 'output':
            name = 'self'
        aten_name = camel_to_snake(SUBSTITUTIONS.get(name, name))
        parts = aten_name.split('_')
        if aten_name in params_by_name:
            param = params_by_name[aten_name]
            if arg.is_optional:
                param['is_nullable'] = True
            thnn_args.append(copy.deepcopy(param))
        elif len(parts) == 2 and parts[0] in ARGUMENT_MAPPINGS and parts[1] in DIMENSION_OFFSET:
            # e.g. pad_left
            thnn_args.append(arg_expr(parts[0], parts[1]))
        elif name[-1] in DIMENSION_OFFSET and name[:-1] in ARGUMENT_MAPPINGS:
            # e.g kW, kH
            thnn_args.append(arg_expr(name[:-1], name[-1]))
        elif name == 'owidth' or name == 'oheight':
            thnn_args.append(arg_expr(name[0], name[1:]))
        elif name == 'scale':
            thnn_args.append({'type': 'EXPRESSION', 'name': '1'})
        elif name == 'inplace':
            thnn_args.append({'type': 'EXPRESSION', 'name': str(inplace).lower()})
        else:
            raise RuntimeError("{}: can't find binding for '{}'"
                               .format(thnn_function.name, name))
    return thnn_args


def remove_unused_args(args, thnn_args):
    """Returns the subset of args whose name appears in thnn_args"""
    def clean_name(name):
        name = name[:name.index('[')] if '[' in name else name
        if name.endswith('_'):
            name = name[:-1]
        return name
    uses = set([clean_name(arg['name']) for arg in thnn_args])
    uses.add('output_mask')
    args = [arg for arg in args if arg['name'] in uses]
    for arg in args:
        if 'default' in arg:
            del arg['default']
    return args


def unique_args(argslist):
    result = []
    seen = set()
    for args in argslist:
        for arg in args:
            if arg['name'] in seen:
                continue
            seen.add(arg['name'])
            result.append(arg)
    return result


def function_info(name, arguments, cimpls, buffers, backends, inplace, scalar_check):
    """
    cimpls contains information use to call into THNN:
        cname: THNN function name
        arguments: arguments to functional call
        condition: [optional] guard around call
    """
    return {
        'mode': 'NN',
        'name': name,
        'types': ['Float', 'Double', 'Half'],  # Half will be stripped for CPU backend
        'arguments': arguments,
        'return': 'argument 0' if inplace else get_return(arguments),
        'buffers': buffers,
        'backends': backends,
        'cimpls': cimpls,
        'scalar_check': scalar_check,
        'variants': ['function'],
    }


def base_declaration(func, thnn_function, backends, inplace=False):
    """Creates the NN function without any buffers in it's signature"""
    name, params = re.match(NAME_PARAM_REGEX, func['name']).groups()
    if inplace:
        name += '_'
    params = params.split(', ')
    arguments = [argument_to_declaration(a, func) for a in params]
    if not inplace:
        arguments += output_arguments(thnn_function)
    buffers = [argument_to_declaration('Tensor ' + buf)
               for buf in func.get('buffers', [])]

    return function_info(name, arguments, None, buffers, backends, inplace, func.get('scalar_check'))


def forward_declaration(base, thnn_function, inplace=False):
    name = '{}_forward'.format(base['name'])
    if inplace:
        name += '_'

    arguments = [copy.deepcopy(arg) for arg in base['arguments']
                 if not arg.get('output')]

    arguments += output_arguments(thnn_function)
    for buffer in base['buffers']:
        buffer = copy.deepcopy(buffer)
        buffer['output'] = True
        arguments.append(buffer)

    thnn_args = get_thnn_args(thnn_function, arguments, inplace)
    arguments = remove_unused_args(arguments, thnn_args)
    cimpl = {'cname': thnn_function.name, 'arguments': thnn_args}

    scalar_check = base['scalar_check']
    if scalar_check is not None:
        output_arg_names = [arg['name'] for arg in arguments if arg.get('output', False)]
        scalar_check = {k: v for (k, v) in scalar_check.items() if k in output_arg_names}

    return function_info(name, arguments, [cimpl], [], base['backends'], inplace, scalar_check)


def backward_declaration(base, thnn_functions):
    name = '{}_backward'.format(base['name'])

    arguments = []
    arguments.append({'type': 'THTensor*', 'name': 'grad_output'})
    arguments += [copy.deepcopy(arg) for arg in base['arguments']
                  if arg['name'] != 'inplace']
    arguments += base['buffers']

    if 'upsample' in base['name']:
        # Add input_size as parameter to upsample backwards functions
        # Note that input_size is 4-dim for upsample_xxx2d
        size = 2 + int(re.search(r'(\d+)d', base['name']).group(1))
        input_size_arg = {'type': 'IntArrayRef', 'name': 'input_size', 'size': size}
        for output_size_idx, arg in enumerate(arguments):
            if arg['name'] == 'output_size':
                break
        arguments.insert(output_size_idx + 1, input_size_arg)

    if 'im2col' in base['name']:
        # Add input_size as parameter to im2col backwards function
        input_size_arg = {'type': 'IntArrayRef', 'name': 'input_size', 'size': 2}
        arguments.insert(2, input_size_arg)

    # outputs from the forward may be inputs to the backwards
    for arg in arguments:
        if 'output' in arg:
            del arg['output']

    arguments += unique_args([output_arguments(f) for f in thnn_functions])

    def initialize_output_arg(arg):
        # the mask array<bool, N> specifies which return values to compute
        arg['mask'] = True
        arg['is_nullable'] = True

        # grad_weight and grad_bias need to be resized and zeroed
        if arg['name'] == 'grad_weight':
            arg['resize'] = 'weight'
            arg['zero'] = True
        if arg['name'] == 'grad_bias':
            dim = 1 if 'transpose' in name else 0
            arg['resize'] = [('weight', dim)]
            arg['zero'] = True

    is_batch_norm_backward = '_backward' in thnn_functions[0].name
    grad_params = []
    if len(thnn_functions) > 1 or is_batch_norm_backward:
        for arg in arguments:
            if arg.get('output', False):
                initialize_output_arg(arg)
            if 'Tensor' in arg['type'] and arg['name'].startswith('grad_') and \
                    'input' not in arg['name'] and 'output' not in arg['name']:
                grad_params.append(arg['name'])

    thnn_args = [get_thnn_args(f, arguments, False) for f in thnn_functions]
    arguments = remove_unused_args(arguments, unique_args(thnn_args))
    cimpls = []

    def get_condition(func):
        # only call into the THNN functions if the output args are not null
        if '_updateGradInput' in func.name:
            return 'grad_input_'
        if '_accGradParameters' in func.name:
            return ' || '.join(p + '_' for p in grad_params)
        return None

    for func, args in zip(thnn_functions, thnn_args):
        cimpl = {'cname': func.name, 'arguments': args}
        if len(thnn_functions) > 1:
            cimpl['condition'] = get_condition(func)
        cimpls.append(cimpl)

    output_args = [arg for arg in arguments if arg.get('output', False)]
    scalar_check_arg = base['scalar_check'] if base['scalar_check'] is not None else dict()
    scalar_check = {k: v for (k, v) in scalar_check_arg.items() if k in [a['name'] for a in output_args]}
    for arg in output_args:
        # resize automatically sets scalar_check
        if scalar_check.get(arg['name']) is not None or arg.get('resize', False):
            pass
        else:
            base_name = arg['name'][len('grad_'):] if arg['name'] != 'grad_input' else 'self'
            if base_name in [a['name'] for a in arguments]:
                scalar_check[arg['name']] = base_name + '_->dim() == 0'
            else:
                raise ValueError(("Could not infer scalar_check for {} argument of func {} because {} "
                                  "does not exist.  Please explicitly specify scalar_check."
                                  .format(arg['name'], name, base_name)))

    return function_info(name, arguments, cimpls, [], base['backends'], False, scalar_check)


def parse_nn_yaml(filename):
    with open(filename, 'r') as f:
        return yaml.load(f, Loader=Loader)


include_only = '(updateOutput|updateGradInput|accGradParameters|backward)$'
exclude = 'LookupTable'


def run(paths):
    function_backends = defaultdict(list)
    header_functions = OrderedDict()

    headers = [p for p in paths if p.endswith('.h')]
    yamls = [p for p in paths if p.endswith('.yaml')]

    for path in headers:
        backend = 'CUDA' if re.search('THCU', path) else 'CPU'
        for func in common_with_cwrap.parse_header(path):
            if re.search(include_only, func.name) is None or re.search(exclude, func.name) is not None:
                continue
            function_backends[func.name].append(backend)
            if func.name not in header_functions:
                header_functions[func.name] = func

    bwd_suffixes = ['_updateGradInput', '_accGradParameters', '_backward']

    declarations = []
    for path in yamls:
        for func in parse_nn_yaml(path):
            cname = func['cname']
            backends = function_backends[cname + '_updateOutput']

            fwd_function = header_functions[cname + '_updateOutput']
            bwd_functions = []
            for suffix in bwd_suffixes:
                if cname + suffix in header_functions:
                    bwd_functions.append(header_functions[cname + suffix])

            base = base_declaration(func, fwd_function, backends)
            declarations.append(forward_declaration(base, fwd_function))
            declarations.append(backward_declaration(base, bwd_functions))

            if func.get('has_inplace', False):
                declarations.append(base_declaration(func, fwd_function, backends, True))
                declarations.append(forward_declaration(base, fwd_function, True))

    return declarations
