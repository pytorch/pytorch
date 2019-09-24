import re
from copy import deepcopy
from function_wrapper import TYPE_FORMAL_GENERIC
import common_with_cwrap

type_map = {
    'floating_point': [
        'Float',
        'Double',
        'Half',
        'BFloat16',
    ],
    'integral': [
        'Byte',
        'Char',
        'Short',
        'Int',
        'Long',
        'Bool',
    ],
    'quantized': [
        'QInt8',
        'QUInt8',
        'QInt32',
    ]
}

all_types = type_map['floating_point'] + type_map['integral'] + type_map['quantized']
type_map['all'] = all_types

all_backends = ['CPU', 'CUDA', 'SparseCPU', 'SparseCUDA', 'MkldnnCPU', 'QuantizedCPU']
default_backends = ['CPU', 'CUDA']


def process_types_and_backends(option):
    # if specific pairs were not listed, then enumerate them
    # based on the backend and type attributes
    # if backend or type is not defined, it is assumed to be all of them
    if 'backend_types' not in option:
        backends = option.get('backends', default_backends)
        if isinstance(option.get('type_method_definition_dispatch'), dict):
            backends = option.get('type_method_definition_dispatch').keys()
        backends = set(backends)

        backend_types = {}
        for backend in backends:
            if backend == 'QuantizedCPU':
                backend_types[backend] = type_map['quantized']
            else:
                backend_types[backend] = option.get('types', all_types)
    else:
        backend_types = option['backend_types']

    # expand type alias (integral, floating_point, all)
    def expand(types):
        ret = []
        for t in types:
            if t in type_map:
                ret.extend(type_map[t])
            else:
                assert(t in all_types)
                ret.append(t)
        return ret

    for backend in backend_types.keys():
        assert(backend in all_backends)
        backend_types[backend] = set(expand(backend_types[backend]))

    # special case remove Half for cpu unless it is explicitly enabled
    if not option.get('cpu_half', False):
        if 'CPU' in backend_types:
            backend_types['CPU'].discard('Half')

    # special case remove BFloat16 for cpu unless it is explicitly enabled
    if not option.get('cpu_bfloat16', False):
        if 'CPU' in backend_types:
            backend_types['CPU'].discard('BFloat16')

    # TODO: remove this hack once support for a bfloat16 tensor for CUDA is enabled
    if 'CUDA' in backend_types:
        backend_types['CUDA'].discard('BFloat16')

    # special cases remove bool for cpu and cuda unless it is explicitly enabled
    if not option.get('cpu_bool', False):
        if 'CPU' in backend_types:
            backend_types['CPU'].discard('Bool')

    if not option.get('cuda_bool', False):
        if 'CUDA' in backend_types:
            backend_types['CUDA'].discard('Bool')

    # sort the result for easy reading
    for backend in backend_types.keys():
        backend_types[backend] = sorted([type for type in backend_types[backend]])
    option['backend_types'] = backend_types


def exclude(declaration):
    return 'only_register' in declaration or declaration.get('name') == 'ndimension'


def add_variants(option):
    option.setdefault('variants', ['method'])

# if we have 'output' arguments, generate a variant where
# we mark oututs as allocate = True, and where the method variant
# is disabled...


def handle_outputs_taken_as_arguments(options):
    new_options = []

    def is_nullable(arg):
        return (arg['type'] in {'THIntegerTensor*', 'THTensor*'} and
                arg.get('default', '') in {None, 'NULL', 'nullptr'})

    def should_generate_out_variant(option):
        if 'function' in option['variants'] and option['mode'] != 'native':
            # don't generate _out variants for in-place functions
            return re.search('(^__i|[^_]_$)', option['api_name']) is None
        return False

    for option in options:
        for arg in option['arguments']:
            # mark arguments which can be null
            if is_nullable(arg):
                arg['is_nullable'] = True

        if any('output' in arg for arg in option['arguments']):
            allocate_option = deepcopy(option)
            # the allocating option needs to be marked
            for arg in allocate_option['arguments']:
                if 'output' in arg:
                    arg['allocate'] = True

            # the original option, which takes arguments for the results,
            # is no longer a method, and has _out added to indicte it takes
            # output arguments
            if should_generate_out_variant(option):
                if 'method' in option['variants']:
                    option['variants'].remove('method')
                option['api_name'] += '_out'
                new_options.append(option)

            new_options.append(allocate_option)
        else:
            new_options.append(option)
    return new_options


def sanitize_return(option):
    ret = option['return']
    m = re.match(r'argument (\d+(,\d+)*)', ret)
    if m is not None:
        arguments = [int(x) for x in m.group(1).split(',')]
        option['return'] = {'kind': 'arguments', 'arguments': arguments}
    elif ret == 'self':
        option['return'] = {'kind': 'arguments', 'arguments': []}
        for i, x in enumerate(option['arguments']):
            if x['name'] == 'self':
                option['return']['arguments'].append(i)
                break
    else:
        option['return'] = {'kind': 'type', 'type': option['return']}


def set_mode(option):
    option['mode'] = option.get('mode', 'TH')

# To enable 0-dim support in TH operations
# we find all places where a single Scalar replaced with a Tensor
# as an argument is still a valid function
# we then mark the tensor variant with a key zero_dim_dispatch_when_scalar: name
# where 'name' is the name of the argument that should be a scalar
# during dispatch, if that argument is marked internally as holding a scalar
# then the method will dispatch to that function.


def discover_zero_dim_tensor_operations(declaration):
    def signature(option, i=None, value=None):
        elements = [TYPE_FORMAL_GENERIC.get(arg['type'], arg['type'])
                    if i is None or j != i else value
                    for j, arg in enumerate(option['arguments'])]
        return '#'.join(elements)
    signature_to_option = {signature(option): option
                           for option in declaration['options']}

    for option in declaration['options']:
        for i, arg in enumerate(option['arguments']):
            if arg['type'] == 'real':
                signature_of_tensor_version = signature(option, i, 'Tensor &')
                if signature_of_tensor_version in signature_to_option:
                    tensor_version = \
                        signature_to_option[signature_of_tensor_version]
                    names = [arg['name'] for arg in tensor_version['arguments']]
                    tensor_version['zero_dim_dispatch_when_scalar'] = names[i]
                    # print("FOUND "+str(i)   )
                    # print("Scalar Version ===== ")
                    # print(yaml.dump(option))
                    # print("Tensor Version ===== ")
                    # print(yaml.dump(tensor_version))
                    # print("SHARED "+names[i])


def is_extended_method(option):
    if 'method' in option['variants']:
        return False
    else:
        return True


def run(declarations):
    declarations = [d for d in declarations if not exclude(d)]
    non_extended_methods = set()
    for declaration in declarations:
        common_with_cwrap.set_declaration_defaults(declaration)
        declaration['options'] = [deepcopy(o) for o in declaration['options']]
        declaration['options'] = common_with_cwrap.filter_unique_options(
            declaration['options'],
            allow_kwarg=False,
            type_to_signature=TYPE_FORMAL_GENERIC,
            remove_self=True)

        common_with_cwrap.sort_by_number_of_args(declaration)

        discover_zero_dim_tensor_operations(declaration)

        for option in declaration['options']:
            set_mode(option)
            if option['mode'] != 'native':
                sanitize_return(option)
            process_types_and_backends(option)
            add_variants(option)
            if not is_extended_method(option):
                non_extended_methods.add(option['api_name'])
        declaration['options'] = handle_outputs_taken_as_arguments(
            declaration['options'])
    # We (very unfortunately) have overloaded virtual methods. Because
    # of C++'s rules, we cannot move one overload without doing some
    # extra work to make sure that overload in a superclass and an
    # overload in a subclass resolve together. I've chosen to resolve
    # this problem simply by moving ALL overloads of a method which
    # occurs in Tensor to Type.  This is why we have to first compute
    # which methods *names* go on type, and then move ALL overloads
    # of this name to Type.
    for declaration in declarations:
        for option in declaration['options']:
            option['extended_method'] = option['api_name'] not in non_extended_methods
    return declarations
