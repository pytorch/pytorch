# Parses derivatives.yaml into autograd functions
#
# Each autograd function is represented by `DifferentiabilityInfo` containing
# a list of `Derivative`. See `tools.codegen.api.autograd` for the data models.
from collections import defaultdict, Counter
import re
from typing import Sequence, Any, Tuple, List, Set, Dict, Match, Optional
import yaml

from tools.codegen.api.autograd import *
from tools.codegen.api.types import *
from tools.codegen.api import cpp
from tools.codegen.gen import parse_native_yaml
from tools.codegen.context import with_native_function
from tools.codegen.model import *
from tools.codegen.utils import *

try:
    # use faster C loader if available
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader  # type: ignore

def load_derivatives(derivatives_yaml_path: str, native_yaml_path: str) -> Sequence[DifferentiabilityInfo]:
    with open(derivatives_yaml_path, 'r') as f:
        definitions = yaml.load(f, Loader=Loader)

    functions = parse_native_yaml(native_yaml_path)

    # What's the difference between function schema v.s. signature?
    # function schema is the complete declaration including mutability annotation / default value and etc.
    # signature is the canonical schema for a group of functions (in-place/out/functional variants)
    # that are semantically related.
    functions_by_signature: Dict[FunctionSchema, List[NativeFunction]] = defaultdict(list)
    functions_by_schema: Dict[str, NativeFunction] = dict()
    for function in functions:
        functions_by_signature[function.func.signature()].append(function)
        assert str(function.func) not in functions_by_schema
        functions_by_schema[str(function.func)] = function

    infos = [
        create_differentiability_info(defn, functions_by_signature, functions_by_schema)
        for defn in definitions]

    # To keep it byte-for-byte compatible with the old codegen, we assign op names as a separate
    # step. We only assign op names to those with differentiable args, and only append suffix to
    # duplicated op names. This can be simplified if the first of the duplicates can be named
    # 'XyzBackward' instead of 'XyzBackward0' or unconditionally append '0' to singletons.
    op_names = create_op_names(infos)
    return [
        DifferentiabilityInfo(
            name=info.name,
            func=info.func,
            op=op_name,
            derivatives=info.derivatives,
            all_saved_inputs=info.all_saved_inputs,
            all_saved_outputs=info.all_saved_outputs,
            args_with_derivatives=info.args_with_derivatives,
            non_differentiable_arg_names=info.non_differentiable_arg_names,
            output_differentiability=info.output_differentiability,
        )
        for info, op_name in zip(infos, op_names)]

@with_native_function
def cpp_arguments(f: NativeFunction) -> Sequence[Binding]:
    return CppSignatureGroup.from_native_function(f, method=False).signature.arguments()

def create_derivative(f: NativeFunction, formula: str, var_names: Tuple[str, ...]) -> Derivative:
    arguments = cpp_arguments(f)
    argument_names = tuple(a.name for a in arguments)
    argument_types = tuple(a.type for a in arguments)

    return_names = tuple(n if n != 'self' else 'result' for n in cpp.return_names(f))
    return_types = tuple(cpp.return_type(r) for r in f.func.returns)

    formula, saved_inputs = saved_variables(formula, argument_names, argument_types, var_names)
    formula, saved_outputs = saved_variables(formula, return_names, return_types, var_names)

    # Check that the referenced derivatives in the formula are in bounds
    for i in used_gradient_indices(formula):
        if i >= len(f.func.returns):
            raise RuntimeError(
                f'Out of bounds grads access: derivative formula for {cpp.name(f.func)} '
                f'used grads[{i}], but the forward only returns {len(f.func.returns)} outputs.'
            )

    return Derivative(
        formula=formula,
        var_names=var_names,
        saved_inputs=saved_inputs,
        saved_outputs=saved_outputs,
    )

def create_differentiability_info(
    defn: Dict[Any, Any],
    functions_by_signature: Dict[FunctionSchema, List[NativeFunction]],
    functions_by_schema: Dict[str, NativeFunction],
) -> DifferentiabilityInfo:
    """Processes a single entry `defn` in derivatives.yaml"""

    def canonical_function(functions: Sequence[NativeFunction], name: str) -> NativeFunction:
        for f in functions:
            if cpp.name(f.func) == name:
                return f
        # some functions only have in-place variants
        assert name + '_' == cpp.name(functions[0].func)
        return functions[0]

    def split_names(raw_names: str) -> Tuple[str, ...]:
        """Given "foo, bar", return ["foo", "bar"]."""
        return tuple(x.strip() for x in raw_names.split(','))

    def check_grad_usage(defn_name: str, derivatives: Sequence[Derivative]) -> None:
        """
        Check for some subtle mistakes one might make when writing derivatives.
        These mistakes will compile, but will be latent until a function is
        used with double backwards.
        """

        used_grad = 0
        used_grads = 0
        fully_implemented = True
        used_grads_indices: List[int] = []
        for d in derivatives:
            formula = d.formula
            used_grad += len(re.findall(IDENT_REGEX.format('grad'), formula))
            used_grads += len(re.findall(IDENT_REGEX.format('grads'), formula))
            fully_implemented = \
                fully_implemented and \
                not re.search(IDENT_REGEX.format('not_implemented'), formula)
            used_grads_indices.extend(used_gradient_indices(formula))
        assert used_grads >= len(used_grads_indices)
        only_used_grads_indices = used_grads == len(used_grads_indices)

        if used_grad and used_grads:
            raise RuntimeError(f"Derivative definition of {defn_name} in derivatives.yaml illegally "
                               "mixes use of 'grad' and 'grads'. Consider replacing "
                               "occurrences of 'grad' with 'grads[0]'")

        if only_used_grads_indices and set(used_grads_indices) == {0}:
            raise RuntimeError(f"Derivative definition of {defn_name} in derivatives.yaml solely "
                               "refers to 'grads[0]'.  If the first output is indeed the "
                               "only differentiable output, replace 'grads[0]' with 'grad'; "
                               "otherwise, there is a likely error in your derivatives "
                               "declaration.")

    @with_native_function
    def set_up_derivatives(f: NativeFunction) -> Tuple[
        Sequence[Derivative],
        Sequence[Binding],
        Sequence[str],
    ]:
        # Set up the derivative information
        derivatives: List[Derivative] = []
        non_differentiable_arg_names: List[str] = []
        args_with_derivatives_set: Set[str] = set()
        for raw_names in sorted(defn.keys()):
            formula = defn[raw_names]
            names = split_names(raw_names)
            if formula.lower().strip() == 'non_differentiable':
                non_differentiable_arg_names += names
            else:
                derivative = create_derivative(f, formula, names)
                derivatives.append(derivative)
                args_with_derivatives_set |= set(names)

        overlap = args_with_derivatives_set.intersection(non_differentiable_arg_names)
        if overlap:
            raise RuntimeError(f'derivatives definition for {defn} have overlapped non_differentiable '
                               f'and differentiable variables: {overlap}')

        # Next, let us determine the list of inputs in order.
        # TODO: do we need eagerly calculate and save it here? Can it be derived
        # from NativeFunction and `derivatives` on callsites instead?
        args_with_derivatives = list(filter(lambda a: a.name in args_with_derivatives_set, cpp_arguments(f)))

        # Test to see if the use of 'grads' makes sense.
        check_grad_usage(defn_name, derivatives)

        return derivatives, args_with_derivatives, non_differentiable_arg_names

    # NB: Removes 'name' from defn dictionary
    specification = defn.pop('name')
    defn_name, _ = split_name_params(specification)
    # NB: Removes 'output_differentiability' from defn dictionary
    #     `None` means all differentiable.
    output_differentiability = defn.pop('output_differentiability', None)

    schema_function = functions_by_schema.get(specification)
    if not schema_function:
        avail = '\n'.join(k for k, v in functions_by_schema.items() if cpp.name(v.func) == defn_name)
        raise RuntimeError(f'could not find ATen function for schema: {specification} '
                           f'.  Available signatures:\n{avail}')

    # now map this to the legacy schema; this isn't technically necessary, but we'd need some logic here
    # to map in-place schemas to the out-of-place variants.
    # TODO: maybe the logic to handle the legacy schema is no longer necessary?
    signature = schema_function.func.signature()
    functions = functions_by_signature[signature]
    if len(functions) == 0:
        avail = '\n'.join(str(k) for k, v in functions_by_signature.items() if cpp.name(k) == defn_name)
        raise RuntimeError(f'could not find ATen function for legacy signature: {signature} '
                           f'corresponding to schema {specification}.  Please report a bug to PyTorch. '
                           f'Available signatures:\n{avail}')

    canonical = canonical_function(functions, defn_name)
    if 'grad_input_mask' in (a.name for a in cpp_arguments(canonical)):
        raise RuntimeError(f"Schema for {defn_name} has an argument named grad_input_mask, "
                           "but this name would be shadowed by our codegen. "
                           "Please use a different name in native_functions.yaml.")

    derivatives, args_with_derivatives, non_differentiable_arg_names = set_up_derivatives(canonical)

    return DifferentiabilityInfo(
        name=defn_name,
        func=canonical,
        op=None,
        derivatives=derivatives,
        all_saved_inputs=dedup_vars([v for d in derivatives for v in d.saved_inputs]),
        all_saved_outputs=dedup_vars([v for d in derivatives for v in d.saved_outputs]),
        args_with_derivatives=args_with_derivatives,
        non_differentiable_arg_names=non_differentiable_arg_names,
        output_differentiability=output_differentiability,
    )

GRAD_INDEX_REGEX = r'(?:^|\W)grads\[(\d+)\]'

def used_gradient_indices(formula: str) -> List[int]:
    """Determine a list of gradient indices (the i in grads[i]) that
    are used by the formula.

    >>> used_gradient_indices("foo(grads[0], grads[1])")
    [0, 1]
    """
    return [int(i) for i in re.findall(GRAD_INDEX_REGEX, formula)]

def saved_variables(
    formula: str,
    arg_names: Tuple[str, ...],
    arg_types: Tuple[str, ...],
    var_names: Tuple[str, ...],
) -> Tuple[str, Tuple[SavedAttribute, ...]]:

    def stride_expr(name: str) -> str:
        assert var_names == (name,), (
            'Replacement for ".strides()" is currently only supported for single derivatives of the same tensor '
            'that ".strides()" is being called on.')
        return f'strides_or_error({name}, "{name}")'

    REPLACEMENTS: List[Tuple[str, Dict[str, Any]]] = [
        # replace self.sizes() with self_sizes
        (r'{}.sizes\(\)', {
            'suffix': '_sizes',
            'type': 'IntArrayRef',
        }),
        # replace self.options() with self_options
        (r'{}.options\(\)', {
            'suffix': '_options',
            'type': 'at::TensorOptions',
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
        # replace to_args_scalartypes(self) with self_args_scalartypes
        (r'to_args_scalartypes\({}\)', {
            'suffix': '_args_scalartypes',
            'type': 'std::vector<ScalarType>',
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
        # replace self.dim() with self_dim
        (r'{}.dim\(\)', {
            'suffix': '_dim',
            'type': 'int64_t',
        }),
        # replace self.strides() with self_strides
        (r'{}.strides\(\)', {
            'suffix': '_strides',
            'type': 'IntArrayRef',
            'expr': stride_expr,
        }),
    ]

    # find which arguments need to be saved
    saved: List[SavedAttribute] = []

    for name, type in zip(arg_names, arg_types):
        # First search the formula for expressions which can be evaluated
        # when the autograd Function is created to avoid saving variables
        for regex, info in REPLACEMENTS:
            def repl(m: Match[str]) -> str:
                suffix: str = info['suffix'](m) if callable(info['suffix']) else info['suffix']
                expr: str = info['expr'](name) if 'expr' in info else m.group(0)
                saved.append(SavedAttribute(
                    name=name + suffix,
                    type=info['type'],
                    expr=expr,
                ))
                if 'res' in info:
                    replacement: str = info['res'](name)
                    return replacement
                return name + suffix

            formula = re.sub(regex.format(name), repl, formula)

        # Find any variables which remain in the formula and save them
        if re.search(IDENT_REGEX.format(name), formula):
            saved.append(SavedAttribute(
                name=name,
                # TODO: change from string to type data model
                type=type.replace('const ', '').replace(' &', ''),
                expr=name,
            ))

    return formula, tuple(saved)

def create_op_name(info: DifferentiabilityInfo) -> Optional[str]:
    # only assign an op name if we are actually going to calculate a derivative
    if not info.args_with_derivatives:
        return None
    name = info.name
    camel_case = ''.join([p.title() for p in name.split('_')])
    return (camel_case + 'Backward').replace('ForwardBackward', 'Backward')

def create_op_names(infos: Sequence[DifferentiabilityInfo]) -> Sequence[Optional[str]]:
    names = list(map(create_op_name, infos))
    dups = set(item for item, count in Counter(names).items() if count > 1)

    # de-duplicate operation names
    # you end up with something like:
    #   AddBackward0
    #   AddBackward1
    # one for each overload
    counter: Dict[str, int] = Counter()
    dedup: List[Optional[str]] = []
    for name in names:
        if name is None:
            # Keep a placeholder
            dedup.append(None)
        elif name in dups:
            dedup.append(f'{name}{counter[name]}')
            counter[name] += 1
        else:
            dedup.append(name)
    return dedup

def dedup_vars(vars: Sequence[SavedAttribute]) -> Sequence[SavedAttribute]:
    seen: Set[str] = set()
    saved: List[SavedAttribute] = []
    for var in vars:
        if var.name in seen:
            continue
        seen.add(var.name)
        saved.append(var)
    return saved
