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
from .utils import IDENT_REGEX, split_name_params


def load_derivatives(path, declarations):
    with open(path, 'r') as f:
        definitions = yaml.load(f, Loader=YamlLoader)

    declarations_by_signature = defaultdict(list)
    declarations_by_schema = dict()
    for declaration in declarations:
        declarations_by_signature[get_signature(declaration)].append(declaration)
        if declaration['schema_string']:
            assert declaration['schema_string'] not in declarations_by_schema
            declarations_by_schema[declaration['schema_string']] = declaration

    differentiability_infos = [
        process_definition(defn, declarations_by_signature, declarations_by_schema)
        for defn in definitions]

    autograd_functions = [d['autograd_fn'] for d in differentiability_infos if d['autograd_fn'] is not None]
    ensure_unique_names(autograd_functions)
    match_declarations_with_differentiability_info(declarations, differentiability_infos)

    return autograd_functions


def create_differentiability_info(signature, non_differentiable_arg_names,
                                  output_differentiability,
                                  autograd_fn):
    return {
        'signature': signature,
        'non_differentiable_arg_names': non_differentiable_arg_names,
        'output_differentiability': output_differentiability,
        'autograd_fn': autograd_fn,
    }


# How do you feel about pasting declaration inside autograd function...
def create_autograd_function(name, derivatives, args_with_derivatives,
                             declaration):
    op = to_camel_case(name) + 'Backward'
    op = op.replace('ForwardBackward', 'Backward')
    return {
        'name': name,
        'op': op,
        'declaration': declaration,
        'args_with_derivatives': args_with_derivatives,
        'derivatives': derivatives,
        'saved_inputs': all_saved_variables(derivatives, 'saved_inputs'),
        'saved_outputs': all_saved_variables(derivatives, 'saved_outputs'),
    }


def create_derivative(arguments, returns, name, formula, var_names):
    def transform_return(r):
        # In-place functions take in and return self. Call the modified version
        # "output" so that it can be referred to in derivative definitions.
        if r['name'] == 'self':
            r = copy.deepcopy(r)
            r['name'] = 'result'
        return r

    returns = [transform_return(r) for r in returns]
    formula, saved_inputs = saved_variables(formula, arguments)
    formula, saved_outputs = saved_variables(formula, returns)

    # Check that the referenced derivatives in the formula are in bounds
    for i in used_gradient_indices(formula):
        if i >= len(returns):
            raise RuntimeError(
                "Out of bounds grads access: derivative formula for {} "
                "used grads[{}], but the forward only returns {} outputs."
                .format(name, i, len(returns)))

    return {
        'formula': formula,
        'saved_inputs': saved_inputs,
        'saved_outputs': saved_outputs,
        'var_names': var_names,
    }


def process_definition(defn, declarations_by_signature, declarations_by_schema):
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

    def check_grad_usage(defn_name, declaration, derivatives):
        """
        Check for some subtle mistakes one might make when writing derivatives.
        These mistakes will compile, but will be latent until a function is
        used with double backwards.
        """

        used_grad = 0
        used_grads = 0
        fully_implemented = True
        used_grads_indices = []
        for d in derivatives:
            formula = d['formula']
            used_grad += len(re.findall(IDENT_REGEX.format('grad'), formula))
            used_grads += len(re.findall(IDENT_REGEX.format('grads'), formula))
            fully_implemented = \
                fully_implemented and \
                not re.search(IDENT_REGEX.format('not_implemented'), formula)
            used_grads_indices.extend(used_gradient_indices(formula))
        assert used_grads >= len(used_grads_indices)
        only_used_grads_indices = used_grads == len(used_grads_indices)

        if used_grad and used_grads:
            raise RuntimeError("Derivative definition of {} in derivatives.yaml illegally "
                               "mixes use of 'grad' and 'grads'. Consider replacing "
                               "occurrences of 'grad' with 'grads[0]'".format(defn_name))

        if only_used_grads_indices and set(used_grads_indices) == {0}:
            raise RuntimeError("Derivative definition of {} in derivatives.yaml solely "
                               "refers to 'grads[0]'.  If the first output is indeed the "
                               "only differentiable output, replace 'grads[0]' with 'grad'; "
                               "otherwise, there is a likely error in your derivatives "
                               "declaration.".format(defn_name))

    def set_up_derivatives(defn_name, defn, declaration):
        # Determine the set of inputs which have derivatives
        args_with_derivatives_set = set()
        for raw_names in defn:
            args_with_derivatives_set |= set(split_names(raw_names))

        # Next, let us determine the list of inputs in order.
        args_with_derivatives = []
        for arg in declaration['arguments']:
            if arg['name'] not in args_with_derivatives_set:
                continue
            args_with_derivatives.append(arg)

        # Set up the derivative information
        derivatives = []
        non_differentiable_arg_names = []
        for raw_names in sorted(defn.keys()):
            formula = defn[raw_names]
            names = split_names(raw_names)
            derivative = create_derivative(declaration['arguments'], declaration['returns'],
                                           declaration['name'], formula, names)
            if formula.lower().strip() == 'non_differentiable':
                assert not sum([type(var_name) == list
                                for var_name in derivative['var_names']]), \
                    "Variable names associated to a formula should be a flat list"
                non_differentiable_arg_names += derivative['var_names']
            else:
                derivatives.append(derivative)
        args_with_derivatives = list(filter(lambda x: x['name'] not in non_differentiable_arg_names,
                                            args_with_derivatives))

        # Test to see if the use of 'grads' makes sense.
        check_grad_usage(defn_name, declaration, derivatives)

        return derivatives, args_with_derivatives, non_differentiable_arg_names

    def unzip(xs):
        return zip(*xs)

    # NB: Removes 'name' from defn dictionary
    specification = defn.pop('name')
    defn_name, params = split_name_params(specification)
    # NB: Removes 'output_differentiability' from defn dictionary
    #     `None` means all differentiable.
    output_differentiability = defn.pop('output_differentiability', None)

    schema_declaration = declarations_by_schema.get('aten::' + specification)
    if not schema_declaration:
        avail = [k.replace('aten::', '') for k, v in declarations_by_schema.items()
                 if k.replace('aten::', '').startswith(defn_name + '(') and len(v) > 0]
        raise RuntimeError('could not find ATen declaration for schema: {} '
                           '.  Available signatures:\n{}'.format(specification, '\n'.join(avail)))

    # now map this to the legacy schema; this isn't technically necessary, but we'd need some logic here
    # to map in-place schemas to the out-of-place variants.
    signature = get_signature(schema_declaration)
    declarations = declarations_by_signature[signature]
    if len(declarations) == 0:
        avail = [k for k, v in declarations_by_signature.items()
                 if k.startswith(defn_name + '(') and len(v) > 0]
        raise RuntimeError('could not find ATen declaration for legacy signature: {} '
                           'corresponding to schema {}.  Please report a bug to PyTorch. '
                           'Available signatures: {}'.format(signature, specification, ', '.join(avail)))

    canonical = canonical_declaration(declarations, defn_name)
    if 'grad_input_mask' in (a['name'] for a in canonical['arguments']):
        raise RuntimeError("Schema for {} has an argument named grad_input_mask, "
                           "but this name would be shadowed by our codegen. "
                           "Please use a different name in native_functions.yaml."
                           .format(defn_name))

    derivatives, args_with_derivatives, non_differentiable_arg_names = set_up_derivatives(defn_name, defn, canonical)
    autograd_fn = None

    # only create an autograd function if we are actually going to calculate a derivative
    if len(args_with_derivatives) > 0:
        autograd_fn = create_autograd_function(defn_name, derivatives, args_with_derivatives,
                                               canonical)

    return create_differentiability_info(signature, non_differentiable_arg_names,
                                         output_differentiability, autograd_fn)


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


def get_signature(declaration, use_base_variant=False):
    name = declaration['name']
    arguments = declaration['arguments']
    if use_base_variant:
        if declaration['inplace']:
            assert name.endswith('_')
            name = name[:-1]
        elif name.endswith('_out'):
            name = name[:-4]
            arguments = [arg for arg in arguments if not arg.get('output', False)]
    simple_types = [arg['simple_type'] for arg in arguments]
    return '{}({})'.format(name, ', '.join(simple_types))


GRAD_INDEX_REGEX = r'(?:^|\W)grads\[(\d+)\]'


def used_gradient_indices(formula):
    """Determine a list of gradient indices (the i in grads[i]) that
    are used by the formula.

    >>> used_gradient_indices("foo(grads[0], grads[1])")
    [0, 1]
    """
    return [int(i) for i in re.findall(GRAD_INDEX_REGEX, formula)]


def saved_variables(formula, args):
    # find which arguments need to be saved
    saved = []

    REPLACEMENTS = [
        # replace self.sizes() with self_sizes
        (r'{}.sizes\(\)', {
            'suffix': '_sizes',
            'type': 'IntArrayRef',
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
        # replace self.numel() with self_numel
        (r'{}.numel\(\)', {
            'suffix': '_numel',
            'type': 'int64_t',
        }),
        # replace to_args_sizes(self) with self_args_sizes
        (r'to_args_sizes\({}\)', {
            'suffix': '_args_sizes',
            'type': 'std::vector<std::vector<int64_t>>',
        }),
        # replace TensorGeometry(self) with self_geometry
        (r'TensorGeometry\({}\)', {
            'suffix': '_geometry',
            'type': 'TensorGeometry',
        }),
        (r'{}.scalar_type\(\)', {
            'suffix': '_scalar_type',
            'type': 'ScalarType',
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


def match_declarations_with_differentiability_info(declarations, differentiability_infos):
    """Sets the "derivative" and "output_differentiability" key on declarations
    to matching differentiability info

    In-place functions will use the out-of-place derivative definition if there
    is no in-place specific derivative.
    """

    infos_by_signature = {f['signature']: f for f in differentiability_infos}

    def find_info(declaration):
        signature = get_signature(declaration)
        if signature in infos_by_signature:
            return infos_by_signature[signature]

        # if there is no exact match look for the out-of-place signature.
        # i.e mul() for mul_() or mul_out()
        signature = get_signature(declaration, use_base_variant=True)
        return infos_by_signature.get(signature)

    for declaration in declarations:
        info = find_info(declaration)
        declaration['derivative'] = info['autograd_fn'] if info else None
        declaration['non_differentiable_arg_names'] = info['non_differentiable_arg_names'] if info else []
        declaration['output_differentiability'] = info['output_differentiability'] if info else None
