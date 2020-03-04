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

# [temp translations]
# We're currently incrementally moving from the custom func schema to the
# JIT signature schema incrementally. This will reduce overall complexity
# and increase compliance between these components. So for now we do simple
# type translations to continue to emit the legacy func schema for further
# processing by downstream tools. This will helps us avoid having to prematurely
# change all downstream tools to detect these new types.
def type_argument_translations(arg):
    type_and_name = [a.strip() for a in arg.rsplit(' ', 1)]
    name = ''
    if len(type_and_name) > 1:
        name = type_and_name[1]
    t = type_and_name[0]
    name = name.split('=')
    default = None
    nullable = False
    size = None  # Only applies to int[\d+] and Tensor[\d+] arguments
    if len(name) > 1:
        default = name[1]
    name = name[0]

    match = re.match(r'(Tensor.*)\((.+)\)(.*)', t)
    annotation = None
    if match:
        t = match.group(1) + match.group(3)
        annotation = match.group(2)

    # XXX: is_nullable flag can only annotate entire type as optional type,
    # need to special case Generator? logic to make ? only available in jit
    # TODO: deprecate is_nullable global flag, and parse the type
    # to support annotating complicated types with optional annotation
    nullable = (t != 'Generator?' and '?' in t)

    # This enables "Generator? x = None and translates to legacy
    # "Generator* x = nullptr". See [temp translations].
    if t == 'Generator?' and default == 'None':
        t = 'Generator*'
        default = 'nullptr'
    # Enables Generator? by translating to legacy Generator*.
    elif t == "Generator?":
        t = 'Generator*'
    # Enables Tensor[] by translating to legacy TensorList.
    elif t == 'Tensor[]' or t == 'Tensor?[]':
        t = 'TensorList'
    # Enables int[] by translating to legacy IntArrayRef.
    elif t == 'int[]':
        t = 'IntArrayRef'
    # Enables int by translating to legacy int64_t.
    elif t == 'int':
        t = 'int64_t'
    elif t == 'int?':
        t = 'int64_t?'
    elif t == 'int64_t':
        raise RuntimeError("Please use int and not int64_t. "
                           "See [temp translations] for details.")
    elif t == 'int64_t?':
        raise RuntimeError("Please use int? and not int64_t?. "
                           "See [temp translations] for details.")
    # Enables Dimname[] by translating to legacy DimnameList.
    elif t == 'Dimname[]':
        t = 'DimnameList'
    elif t == 'Dimname[]?':
        t = 'DimnameList?'
    # Enables float by translating to legacy double.
    elif t == 'float':
        t = 'double'
    elif t == 'float?':
        t = 'double?'
    # Enables str by translating to legacy std::string.
    elif t == 'str':
        t = 'std::string'
    elif t == 'double':
        raise RuntimeError("Please use float and not double. "
                           "See [temp translations] for details.")
    # Enables int[x] by translating to legacy IntArrayRef[x]. See [temp translations]
    elif re.match(r'int\[(\d+)\]', t):
        match = re.match(r'int\[(\d+)\]', t)
        t = 'IntArrayRef'
        size = int(match.group(1))
    # Enables bool[x] by translating to legacy std::array<bool,x>. See [temp translations]
    elif re.match(r'bool\[(\d+)\]', t):
        match = re.match(r'bool\[(\d+)\]', t)
        t = 'std::array<bool,{}>'.format(match.group(1))
    elif re.match(r'std::array', t):
        raise RuntimeError("Please use array notation, e.g. bool[3] and not std::array."
                           "See [temp translations] for details.")
    # Enables Dimname[x] by translating to DimnameList[x]. See [temp translations]
    elif re.match(r'Dimname\[(\d+)\]', t):
        match = re.match(r'Dimname\[(\d+)\]', t)
        t = 'DimnameList'
        size = int(match.group(1))

    # Legacy type sanitization. TODO: Do we really need this?
    if t == 'Generator*':
        t = 'Generator *'

    if not default:
        pass
    # This enables Tensor? x=None and translates to legacy
    # "Tensor? x={}". See [temp translations].
    elif t.startswith('Tensor?') and default == 'None':
        default = "{}"
    elif default == 'True':
        default = True
    elif default == 'False':
        default = False
    elif default == 'true':
        raise RuntimeError("Please use True and not true. "
                           "See [temp translations] for details.")
    elif default == 'false':
        raise RuntimeError("Please use False and not false. "
                           "See [temp translations] for details.")
    # Enables default argument [] by translating to legacy {}.
    # See [temp translations]
    elif default == '[]':
        default = '{}'
    # Enables lists by translating to legacy {.*}.
    # See [temp translations]
    elif re.match(r'\[.*\]', default):
        default = "{" + default[1:-1] + "}"
    elif default == 'None':
        default = 'c10::nullopt'
    # The JIT signature schema uses Mean, but in particular C++ needs
    # the legacy at::Reduction::Mean. So we'll continue emiting that until
    # we change this at either a JIT schema or C++ level.
    elif default == 'Mean':
        default = 'at::Reduction::Mean'
    elif default == 'contiguous_format':
        default = 'MemoryFormat::Contiguous'
    elif default == 'per_tensor_affine':
        default = 'QScheme::PER_TENSOR_AFFINE'
    else:
        try:
            default = int(default)
        except ValueError:
            try:
                default = float(default)
            except ValueError:
                pass

    return t, name, default, nullable, size, annotation


def parse_arguments(name, args, func_variants, declaration, func_return):
    arguments = []
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

        t, name, default, nullable, size, annotation = type_argument_translations(arg)

        argument_dict = {'type': t.rstrip('?'), 'name': name, 'is_nullable': nullable, 'annotation': annotation}
        if size:
            argument_dict['size'] = size
        if default is not None:
            argument_dict['default'] = default
        if kwarg_only:
            argument_dict['kwarg_only'] = True
        arguments.append(argument_dict)

    is_out_fn = False
    arguments_out = []
    arguments_other = []
    for argument in arguments:
        if argument['type'] == "Tensor" and \
                argument['annotation'] and \
                re.match(r'^(.*!)$', argument['annotation']) and \
                argument.get('kwarg_only'):
            argument['output'] = True
            argument['kwarg_only'] = False
            arguments_out.append(argument)
            is_out_fn = True
        else:
            arguments_other.append(argument)

    arguments = arguments_out + arguments_other

    name = declaration['name']
    if is_out_fn:
        declaration['name'] += "_out"

    # Reverse splat of TensorOptions
    # As we move towards the JIT function schema for native_functions.yaml we need to support
    # the expanded version of TensorOptions. For now we discover whether there are three
    # types and names of keyword arguments: "ScalarType dtype", "Layout layout" and "Device device"
    # Each, if set, must have default arguments set to long or float, strided and "cpu" respectively.
    # They must appear in this order and in this order only in order for us to be able to process them.
    # In the future we will get rid of this specific processing as downstream consumers start relying
    # less on the content of Declarations.yaml. If you want to support more than this you'll
    # potentially have to extend the JIT.

    def make_topt_arg(name, ty):
        arg = {
            'name': name,
            'type': ty,
            'annotation': None,
            'kwarg_only': True,
            'is_nullable': True,
            'default': 'c10::nullopt',
        }
        return arg

    supported_topt_arguments = [
        make_topt_arg('dtype', 'ScalarType'),
        make_topt_arg('layout', 'Layout'),
        make_topt_arg('device', 'Device'),
        make_topt_arg('pin_memory', 'bool'),
    ]

    corresponding_topt = {
        'type': 'TensorOptions',
        'name': 'options',
        'is_nullable': False,
        'annotation': None,
        'kwarg_only': True,
        'default': '{}',
    }

    def check_topt_representation(topt_representation):
        matches = all(topt_representation[i] == topt for i, topt in enumerate(supported_topt_arguments))
        if matches:
            return corresponding_topt
        else:
            return None

    def is_tensor_option(argument):
        return argument['name'] in ['dtype', 'layout', 'device', 'pin_memory']

    new_arguments = []
    idx = 0
    while idx < len(arguments):
        argument = arguments[idx]
        number_of_arguments = len(supported_topt_arguments)
        if is_tensor_option(argument) and len(arguments) - idx >= number_of_arguments:
            topt_representation = []
            for i in range(number_of_arguments):
                argument = arguments[idx]
                if not is_tensor_option(argument):
                    break
                topt_representation.append(argument)
                idx += 1
            if len(topt_representation) == number_of_arguments:
                merged_argument = check_topt_representation(topt_representation)
                assert merged_argument, \
                    "Unsupported combination of TensorOptions in {}:\n{}\n\n"\
                    "The only currently supported combinations are:\n{}"\
                    .format(name,
                            pprint.pformat(topt_representation),
                            pprint.pformat(supported_topt_arguments))
                new_arguments.append(merged_argument)
            else:
                new_arguments += topt_representation
        else:
            new_arguments.append(argument)
            idx += 1

    arguments = new_arguments

    # Sanity checks

    # TODO: convention is that the ith-argument correspond to the i-th return, but it would
    # be better if we just named everything and matched by name.
    for arg_idx, argument in enumerate(arguments_out):
        assert argument['annotation'] == func_return[arg_idx]['annotation'], \
            "For func {} writeable keyword Tensor arguments need to have a matching return Tensor. Further, " \
            "the ith-argument needs to correspond to the i-th return.".format(name)

    assert len(arguments_out) <= len(func_return), "func {} must return at least as many Tensors " \
        "as can be passed as output.".format(name)

    if name.endswith('_out'):
        raise RuntimeError("Native function {} may not be suffixed with _out as we transition to a unified schema. "
                           "Otherwise you will cause confusion amongst consumers of native functions.".format(name))

    if is_out_fn and func_variants not in [[], 'function', ['function']]:
        raise RuntimeError("Native functions with output MUST be declared with only the function variant; "
                           "e.g., variants: function; otherwise you will tickle a Python argument binding bug "
                           "(which usually manifests itself as the result variable being undefined.) "
                           "The culprit was: {}".format(name))
    if not is_out_fn:
        assert len(arguments_out) == 0, "func {} is not marked as output yet contains output " \
            "keyword arguments".format(name)

    # TODO: Explicit checking for void is a hack and should disappear after a more
    # functionally complete implementation of Tensor aliases.
    if declaration['inplace'] and len(func_return) > 0:
        found_self = False
        for arg_idx, argument in enumerate(arguments):
            if argument['name'] == "self":
                assert argument['annotation'] and argument['annotation'].endswith("!"), \
                    "Inplace function \"{}\" needs to annotate Tensor argument named self " \
                    "as mutable.".format(name)
                found_self = True
                assert argument['annotation'] == func_return[arg_idx]['annotation'], \
                    "Inplace function annotations of function {} need to match between " \
                    "input and correponding output.".format(name)
                assert argument['name'] == func_return[arg_idx]['name'] or \
                    argument['name'] == func_return[arg_idx]['name'] + "_return"
                assert argument['type'] == func_return[arg_idx]['type']
        assert found_self, "Inplace function \"{}\" needs Tensor argument named self.".format(name)

    return arguments


def parse_return_arguments(return_decl, inplace, func_decl):
    arguments = []
    if return_decl == '()':
        return arguments

    # TODO: Use a real parser here; this will get bamboozled
    # by signatures that contain things like std::array<bool, 2> (note the space)
    if return_decl[0] == '(' and return_decl[-1] == ')':
        return_decl = return_decl[1:-1]

    multiple_args = len(return_decl.split(', ')) > 1
    for arg_idx, arg in enumerate(return_decl.split(', ')):
        t, name, default, nullable, size, annotation = type_argument_translations(arg)
        # name of arguments and name of return sometimes have collision
        # in this case, we rename the return name to <name>_return.
        return_name = name
        if name in func_decl['func'].split('->')[0]:
            return_name = name + "_return"
        argument_dict = {'type': t, 'name': return_name, 'annotation': annotation}
        if name:
            # See Note [field_name versus name]
            argument_dict['field_name'] = name
        else:
            if t == "Tensor" and inplace:
                assert annotation and annotation.endswith("!"), \
                    "Return Tensor of function \"{}\" flagged as inplace needs to be " \
                    "annotated as mutable".format(func_decl['func'])
                argument_dict['name'] = 'self'
            else:
                argument_dict['name'] = 'result' if not multiple_args else 'result' + str(arg_idx)
        argument_dict['output'] = True
        arguments.append(argument_dict)
    return arguments


def parse_native_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=Loader)


def propagate_field_names(output_arguments, return_arguments):
    if output_arguments:
        for i, r in enumerate(return_arguments):
            if 'field_name' in r:
                output_arguments[i]['field_name'] = r['field_name']

def is_named_tensor_only(declaration):
    return any(['Dimname' in arg['type'] for arg in declaration['arguments']])


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
                fn_name, arguments = func_decl.split('(', 1)
                if '.' in fn_name:
                    fn_name, overload_name = fn_name.split('.', 1)
                else:
                    overload_name = ''
                assert arguments[-1] == ")", "Expecting closing ) for {}".format(func['func'])
                arguments = arguments[:-1]  # Expect closing )
                declaration['name'] = func.get('name', fn_name)
                declaration['operator_name'] = func.get('name', fn_name)
                declaration['overload_name'] = func.get('overload_name', overload_name)
                declaration['inplace'] = re.search('(^__i|[^_]_$)', fn_name) is not None
                return_arguments = parse_return_arguments(return_decl, declaration['inplace'], func)
                arguments = parse_arguments(
                    declaration['name'], arguments, func.get('variants', []), declaration, return_arguments)
                output_arguments = [x for x in arguments if x.get('output')]
                propagate_field_names(output_arguments, return_arguments)
                declaration['return'] = return_arguments if len(output_arguments) == 0 else output_arguments
                declaration['variants'] = func.get('variants', ['function'])
                declaration['requires_tensor'] = func.get('requires_tensor', False)
                declaration['matches_jit_signature'] = func.get('matches_jit_signature', True)
                declaration['cpu_half'] = func.get('cpu_half', False)
                declaration['cpu_bfloat16'] = func.get('cpu_bfloat16', False)
                declaration['cuda_bfloat16'] = func.get('cuda_bfloat16', False)
                declaration['cpu_bool'] = func.get('cpu_bool', False)
                declaration['cuda_bool'] = func.get('cuda_bool', False)
                declaration['deprecated'] = func.get('deprecated', False)
                declaration['device_guard'] = func.get('device_guard', True)
                declaration['supports_named_tensor'] = func.get('supports_named_tensor', False)
                declaration['use_c10_dispatcher'] = func.get('use_c10_dispatcher', 'unboxed_only')
                assert declaration['use_c10_dispatcher'] in ['unboxed_only', 'full']
                declaration['manual_kernel_registration'] = func.get('manual_kernel_registration', False)
                declaration['category_override'] = func.get('category_override', '')
                declaration['arguments'] = func.get('arguments', arguments)
                declaration['type_method_definition_dispatch'] = func.get('dispatch', declaration['name'])
                declaration['python_module'] = func.get('python_module', '')
                declarations.append(declaration)
            except Exception as e:
                msg = '''Exception raised in processing function:
{func}
Generated partial declaration:
{decl}'''.format(func=pprint.pformat(func), decl=pprint.pformat(declaration))
                print(msg, file=sys.stderr)
                raise e

    return declarations
