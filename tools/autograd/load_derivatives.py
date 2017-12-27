# Parses derivatives.yaml into autograd functions
#
# Each autograd function is represented by dictionary containing a list of
# derivatives (also a dictionary). See `create_autograd_function` and
# `create_derivative` for the keys.
from collections import defaultdict
import copy
import re
import yaml
from .utils import YamlLoader
from .gen_variable_type import IDENT_REGEX, split_name_params


def load_derivatives(path, declarations_by_signature, declarations_by_name):
    with open(path, 'r') as f:
        definitions = yaml.load(f, Loader=YamlLoader)

    autograd_functions = [
        process_definition(defn, declarations_by_signature, declarations_by_name)
        for defn in definitions]
    ensure_unique_names(autograd_functions)
    return autograd_functions


def create_autograd_function(name, derivatives, num_inputs, buffers=None):
    return {
        'name': name,
        'op': to_camel_case(name) + 'Backward',
        'num_inputs': num_inputs,
        'derivatives': derivatives,
        'buffers': [] if buffers is None else buffers,
        'saved_inputs': all_saved_variables(derivatives, 'saved_inputs'),
        'saved_outputs': all_saved_variables(derivatives, 'saved_outputs'),
    }


def create_derivative(declaration, formula, output_indices, var_names):
    returns = [r for r in declaration['returns'] if r.get('name') != 'self']
    arguments = declaration['arguments']
    formula, saved_inputs = saved_variables(formula, arguments)
    formula, saved_outputs = saved_variables(formula, returns)

    return {
        'formula': formula,
        'output_indices': output_indices,
        'saved_inputs': saved_inputs,
        'saved_outputs': saved_outputs,
        'var_names': var_names,
    }


def process_definition(defn, declarations_by_signature, declarations_by_name):
    """Processes a single entry `defn` in derivatives.yaml"""

    def canonical_declaration(declarations, name):
        for declaration in declarations:
            if declaration['name'] == name:
                return declaration
        # some functions only have in-place variants
        assert name + '_' == declarations[0]['name']
        return declarations[0]

    def split_names(raw_names):
        """Given "foo, bar", return ["foo", "bar"]."""
        return [x.strip() for x in raw_names.split(',')]

    def lookup_pred(pred, xs):
        """Return the index of the first element of xs matching pred."""
        return next((i, x) for i, x in enumerate(xs) if pred(x))

    def set_up_derivatives(defn, declaration):
        # First, let us determine the set of inputs for which gradients
        # were specified in declarations.  We'll use this in layout
        # computation.
        args_with_gradients = set()
        for raw_names in defn:
            args_with_gradients |= set(split_names(raw_names))

        # Next, let us compute the layout of the grad_inputs we will
        # return.  In general this is not in one-to-one correspondence
        # with the inputs, because some will not have gradients, and we
        # will not bother allocating an undefined tensor for them.
        num_inputs = 0  # number of grad_inputs to return
        arg_name_to_output_index = {}
        for arg in declaration['arguments']:
            if arg['name'] not in args_with_gradients:
                continue
            if arg['type'] == 'TensorList':
                num_inputs = ''
                output_index = '*'  # variable length thing
            else:
                output_index = num_inputs  # the current index
                num_inputs += 1
            arg_name_to_output_index[arg['name']] = output_index

        # Finally, let us set up the derivative information
        derivatives = []
        for raw_names in sorted(defn.keys()):
            formula = defn[raw_names]
            names = split_names(raw_names)
            output_indices = []
            args = []
            for name in names:
                output_indices.append(arg_name_to_output_index[name])
                args.append(name)
            derivatives.append(create_derivative(declaration, formula, output_indices, args))

        return derivatives, num_inputs

    def is_nn_fwd(defn_name, declarations_by_name):
        """Return True if the definition is of an NN, non-double
           backward function, False otherwise"""

        if len(declarations_by_name[defn_name]) == 0:
            return False
        declaration = declarations_by_name[defn_name][0]
        base_name = defn_name if not declaration['inplace'] else defn_name[:-1]
        fwd_name = base_name + '_forward'
        if declaration['mode'] != 'NN' or fwd_name not in declarations_by_name:
            return False
        return True

    def preprocess_nn_function(defn_name, declarations_by_name):
        """Set up declaration and derivative information for NN,
           non-double backward functions"""

        declaration = declarations_by_name[defn_name][0]
        base_name = defn_name if not declaration['inplace'] else defn_name[:-1]
        fwd_name = base_name + ('_forward' if not declaration['inplace'] else '_forward_')

        assert len(declarations_by_name[fwd_name]) == 1

        declaration['base_name'] = fwd_name
        fwd = declarations_by_name[fwd_name][0]

        derivatives, num_inputs = set_up_derivatives(defn, fwd)
        buffers = declaration['buffers']

        func = create_autograd_function(defn_name, derivatives, num_inputs, buffers)
        declaration['derivative'] = func

        return func

    def unzip(xs):
        return zip(*xs)

    # NB: Removes 'name' from defn dictionary
    defn_name, params = split_name_params(defn.pop('name'))
    param_types, param_names = unzip([p.split(' ') for p in params if p != '*'])
    if 'grad_input_mask' in param_names:
        raise RuntimeError("Signature for {} has an argument named grad_input_mask, "
                           "but this name would be shadowed by our codegen. "
                           "Please use a different name in Declarations.cwrap."
                           .format(defn_name))
    signature = '{}({})'.format(defn_name, ', '.join(param_types))

    if is_nn_fwd(defn_name, declarations_by_name):
        return preprocess_nn_function(defn_name, declarations_by_name)

    declarations = declarations_by_signature[signature]
    if len(declarations) == 0:
        avail = [k for k, v in declarations_by_signature.items()
                 if k.startswith(defn_name + '(') and len(v) > 0]
        raise RuntimeError('no ATen declaration found for: {}.  '
                           'Available signatures: {}'.format(signature, ', '.join(avail)))
    canonical = canonical_declaration(declarations, defn_name)

    # TODO: Check the types line up
    if len(param_names) != len(canonical['args']):
        raise RuntimeError('Signature for {} has {} arguments ({}), but '
                           'Declarations.yaml records {} arguments ({})'
                           .format(defn_name,
                                   len(param_names),
                                   ', '.join(param_names),
                                   len(canonical['args']),
                                   ', '.join(canonical['args'])))
    for i, (x, y) in enumerate(zip(param_names, canonical['args'])):
        if x != y:
            raise RuntimeError('Argument {} of {} has different names in '
                               'derivatives.yaml ({}) and '
                               'Declarations.yaml ({})'
                               .format(i, defn_name, x, y))

    derivatives, num_inputs = set_up_derivatives(defn, canonical)
    buffers = canonical.get('buffers')

    func = create_autograd_function(defn_name, derivatives, num_inputs, buffers)
    for declaration in declarations:
        declaration['derivative'] = func
    return func


def ensure_unique_names(autograd_functions):
    # de-duplicate operation names
    # you end up with something like:
    #   AddBackward0
    #   AddBackward1
    # one for each overload
    functions_by_name = defaultdict(list)
    for func in autograd_functions:
        functions_by_name[func['op']].append(func)
    for op in functions_by_name.keys():
        overloads = functions_by_name[op]
        if len(overloads) > 1:
            for i, func in enumerate(overloads):
                func['op'] += str(i)


def saved_variables(formula, args):
    # find which arguments need to be saved
    saved = []

    REPLACEMENTS = [
        # replace self.sizes() with self_sizes
        (r'{}.sizes\(\)', {
            'suffix': '_sizes',
            'type': 'IntList',
        }),
        # replace zeros_like(self) with self_info
        (r'zeros_like\({}\)', {
            'suffix': '_info',
            'type': 'TypeAndSize',
            'expr': lambda name: name,  # at save-time
            'res': lambda name: name + '_info.zeros()',  # at eval-time
        }),
        # replace self.size(2) with self_size_2
        (r'{}.size\((\w+)\)', {
            'suffix': lambda m: '_argsize_{}'.format(*m.groups()),
            'type': 'int64_t',
        }),
        # replace to_arg_sizes(self, 2) with self_argsizes_2
        (r'to_arg_sizes\({}, (\w+)\)', {
            'suffix': lambda m: '_sizes_{}'.format(*m.groups()),
            'type': 'IntList',
        }),
        # replace TensorGeometry(self) with self_geometry
        (r'TensorGeometry\({}\)', {
            'suffix': '_geometry',
            'type': 'TensorGeometry',
        }),
    ]

    for arg in args:
        if 'name' not in arg:
            # some returned arguments do not have names
            continue

        name = arg['name']

        # First search the formula for expressions which can be evaluated
        # when the autograd Function is created to avoid saving variables
        for regex, info in REPLACEMENTS:
            def repl(m):
                suffix = info['suffix']
                suffix = suffix(m) if callable(suffix) else suffix
                expr = info['expr'](name) if 'expr' in info else m.group(0)
                saved.append({
                    'name': name + suffix,
                    'type': info['type'],
                    'expr': expr,
                })
                if 'res' in info:
                    return info['res'](name)
                return name + suffix

            formula = re.sub(regex.format(name), repl, formula)

        # Find any variables which remain in the formula and save them
        if re.search(IDENT_REGEX.format(name), formula):
            arg = copy.deepcopy(arg)
            arg['type'] = arg['type'].replace('const ', '').replace(' &', '')
            saved.append(arg)

    return formula, saved


def all_saved_variables(derivatives, key):
    seen = set()
    saved = []
    for d in derivatives:
        for saved_arg in d[key]:
            if saved_arg['name'] in seen:
                continue
            seen.add(saved_arg['name'])
            saved.append(saved_arg)
    return saved


def to_camel_case(name):
    return ''.join([p.title() for p in name.split('_')])
