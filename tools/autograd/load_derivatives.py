# Parses derivatives.yaml into autograd functions
#
# Each autograd function is represented by `DifferentiabilityInfo` containing
# a list of `Derivative`. See `tools.codegen.api.autograd` for the data models.
from collections import defaultdict
import re
from typing import Counter, Sequence, Any, Tuple, List, Set, Dict, Match, Optional
import yaml

from tools.codegen.api.autograd import (Derivative, DifferentiabilityInfo,
                                        SavedAttribute, ForwardDerivative)
from tools.codegen.api.types import (Binding, CppSignatureGroup, NamedCType, BaseCType, VectorCType,
                                     intArrayRefT, tensorOptionsT, typeAndSizeT, longT, boolT,
                                     tensorGeometryT, scalarTypeT, SpecialArgName,
                                     OptionalCType, stringT)
from tools.codegen.api import cpp
from tools.codegen.gen import parse_native_yaml
from tools.codegen.context import with_native_function
from tools.codegen.model import FunctionSchema, NativeFunction, Variant, Type
from tools.codegen.utils import IDENT_REGEX, split_name_params, YamlLoader

_GLOBAL_LOAD_DERIVATIVE_CACHE = {}

def load_derivatives(derivatives_yaml_path: str, native_yaml_path: str) -> Sequence[DifferentiabilityInfo]:
    # Do some caching as this is a deterministic function
    global _GLOBAL_LOAD_DERIVATIVE_CACHE
    key = (derivatives_yaml_path, native_yaml_path)
    if key not in _GLOBAL_LOAD_DERIVATIVE_CACHE:

        with open(derivatives_yaml_path, 'r') as f:
            definitions = yaml.load(f, Loader=YamlLoader)

        functions = parse_native_yaml(native_yaml_path).native_functions

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

        # Keep track of how many of which ops we've seen so we can
        # disambiguate them with a numeric suffix.
        op_counter = Counter[str]()

        infos = [
            create_differentiability_info(defn, functions_by_signature, functions_by_schema, op_counter)
            for defn in definitions]

        _GLOBAL_LOAD_DERIVATIVE_CACHE[key] = infos

    return _GLOBAL_LOAD_DERIVATIVE_CACHE[key]

@with_native_function
def cpp_arguments(f: NativeFunction) -> Sequence[Binding]:
    return CppSignatureGroup.from_native_function(f, method=False).signature.arguments()

def create_derivative(f: NativeFunction, formula: str, var_names: Tuple[str, ...],
                      available_named_gradients: Sequence[str]) -> Derivative:
    original_formula = formula
    arguments: List[NamedCType] = [a.nctype.remove_const_ref() for a in cpp_arguments(f)]

    return_names = tuple(n if n != 'self' else 'result' for n in cpp.return_names(f))
    return_types = tuple(cpp.return_type(r).remove_const_ref() for r in f.func.returns)

    named_returns = [NamedCType(name, type) for name, type in zip(return_names, return_types)]

    formula, saved_inputs = saved_variables(formula, arguments, var_names)
    formula, saved_outputs = saved_variables(formula, named_returns, var_names)

    used_named_gradients = {name for name in available_named_gradients if re.search(IDENT_REGEX.format(name), formula)}

    # Check that the referenced derivatives in the formula are in bounds
    for i in used_gradient_indices(formula):
        if i >= len(f.func.returns):
            raise RuntimeError(
                f'Out of bounds grads access: derivative formula for {cpp.name(f.func)} '
                f'used grads[{i}], but the forward only returns {len(f.func.returns)} outputs.'
            )

    return Derivative(
        formula=formula,
        original_formula=original_formula,
        var_names=var_names,
        saved_inputs=saved_inputs,
        saved_outputs=saved_outputs,
        named_gradients=used_named_gradients,
    )

def create_forward_derivative(f: NativeFunction, formula: str, names: Tuple[str, ...]) -> ForwardDerivative:
    assert len(names) == 1, "Forward derivatives can define gradients for only one output at a time"
    var_name = names[0]
    var_type: Optional[Type] = None
    for r in f.func.returns:
        if r.name == var_name:
            var_type = r.type
            break
    # Handle default return names
    if var_type is None:
        if var_name == "result":
            assert len(f.func.returns) == 1
            var_type = f.func.returns[0].type
        else:
            res = re.findall(r"^result(\d+)$", var_name)
            if len(res) == 1:
                arg_idx = int(res[0])
                var_type = f.func.returns[arg_idx].type

    assert var_type is not None, "No matching output for forward derivative definition"
    return ForwardDerivative(
        formula=formula,
        var_name=var_name,
        var_type=var_type,
        required_inputs_fw_grad=None,
        required_inputs_primal=None,
        required_original_self_value=False,
        is_reusing_outplace_formula=False)

def postprocess_forward_derivatives(
    f: NativeFunction,
    defn_name: str,
    all_arg_names: List[str],
    derivatives: List[Derivative],
    forward_derivatives: List[ForwardDerivative],
    args_with_derivatives: Sequence[Binding]
) -> List[ForwardDerivative]:

    def find_required_inputs(formula: str, postfix: str) -> Tuple[str, ...]:
        required_inputs = set()
        for arg in args_with_derivatives:
            if arg.type in ('at::TensorList', 'const at::ITensorList &'):
                # The functions taking TensorList or ITensorList handle everything internally
                continue
            arg_name = arg.name

            found = re.search(IDENT_REGEX.format(arg_name), formula)
            if found:
                raise RuntimeError(f"The forward formula for {defn_name} is using the base name of the {arg_name} "
                                   f"argument which is ambiguous. You should use {arg_name}_p to access the primal "
                                   f"value and {arg_name}_t to access the tangent.")

            found = re.search(IDENT_REGEX.format(arg_name + postfix), formula)
            if found:
                required_inputs.add(arg_name)

        return tuple(required_inputs)

    updated_derivatives: List[ForwardDerivative] = []

    for defn in forward_derivatives:
        formula = defn.formula
        required_inputs_tangent = find_required_inputs(formula, "_t")
        if formula == "auto_element_wise":
            if (not len(args_with_derivatives) == 1) or len(forward_derivatives) > 1:
                raise RuntimeError(f"Derivative definition of {defn_name} in derivatives.yaml defines the "
                                   "forward definition of gradient as element_wise but this only "
                                   "works for functions with a single differentiable input and a "
                                   "single differentiable output.")
            if not len(derivatives) == 1:
                raise RuntimeError(f"Derivative definition of {defn_name} in derivatives.yaml defines the "
                                   "forward definition of gradient as element_wise but it does not "
                                   "defines the gradient formula for its argument which is required.")
            # This transformation is based on the observation that for element-wise functions, the Jacobian
            # matrix is diagonal and thus doing J * v is the same as (v^T J)^T (in practice, we ignore the transpositions)
            # For the complex case, we use hermitian transpose and get (v.conj() J).conj()
            # So here we are going to re-use the backward formula and replace two things:
            # 1) all occurrences of "grad" with "foo_t.conj()", where foo is the name of the unique differentiable input.
            # 2) all usage of an original input "foo" with its primal value "foo_p".
            # 3) conjugate the final result
            # For example, for abs, the backward formula is:
            #   grad * self.sgn()
            # And this function generates a forward formula that is:
            #   (self_t.conj() * self_p.sgn()).conj()

            backward_formula = derivatives[0].original_formula
            input_name = args_with_derivatives[0].name

            # Do replacement 1) of the grad
            def repl(m: Any) -> str:
                return f"{m.group(1)}{input_name}_t.conj(){m.group(2)}"
            fw_formula = re.sub(IDENT_REGEX.format("grad"), repl, backward_formula)

            # Do replacement 2) of the input variables
            for arg in args_with_derivatives:
                arg_name = arg.name

                def repl(m: Any) -> str:
                    return f"{m.group(1)}{arg_name}_p{m.group(2)}"
                fw_formula = re.sub(IDENT_REGEX.format(arg_name), repl, fw_formula)

            # Do the final conjugate 3)
            fw_formula = f"({fw_formula}).conj()"

            # Since there is a single differentiable inputs and we necessarily need its tangent we can
            # simply require all differentiable input's tangent.
            required_inputs_tangent = tuple(all_arg_names)
            formula = fw_formula
        elif formula == "auto_linear":
            if len(forward_derivatives) > 1:
                raise RuntimeError(f"Derivative definition of {defn_name} in derivatives.yaml defines the "
                                   "forward definition of gradient as linear but this only works "
                                   "for functions with a single differentiable output.")
            # This transformation is based on the observation that linear functions can be written as:
            #   y = f(x) = A * x
            # For some matrix A and the Jacobian of the function f is also A.
            # So doing J * v = A * v = f(v).
            # Hence to do the jvp, we simply need to evaluate the function at the point v instead of x.
            # We do this by calling the forward again by replacing any occurrence of the differentiable
            # input "foo" by it's tangent "foo_t".
            # Note that multiple inputs are not a problem as long as the function is truly linear wrt to
            # the vector where all the differentiable inputs are stacked.

            diff_arg_names = [arg.name for arg in args_with_derivatives]
            assert len(diff_arg_names) > 0

            # Do replacement of input variables
            new_args = []
            for arg_name in all_arg_names:
                if arg_name in diff_arg_names:
                    arg_name = arg_name + "_t"
                new_args.append(arg_name)

            # Call into the forward again. We need two cases here to handle both Tensor methods and at:: functions.
            if Variant.function in f.variants:
                fw_formula = "at::{}({})".format(defn_name, ", ".join(new_args))
            else:
                assert Variant.method in f.variants
                fw_formula = "{}.{}({})".format(new_args[0], defn_name, ", ".join(new_args[1:]))

            # All of the input tangents are always used so all of them are required here.
            required_inputs_tangent = tuple(diff_arg_names)
            formula = fw_formula

        # At this point, the formula is final and is not modified anymore.

        # During forward formula, we use the primal instead of the input Tensors.
        # This call inspects the formula to find for which input's primal are used.
        required_inputs_primal = find_required_inputs(formula, "_p")

        updated_derivatives.append(ForwardDerivative(
            formula=formula,
            var_name=defn.var_name,
            var_type=defn.var_type,
            required_inputs_fw_grad=required_inputs_tangent,
            required_inputs_primal=required_inputs_primal,
            required_original_self_value=False,
            is_reusing_outplace_formula=False))

    return updated_derivatives

def is_forward_derivative_definition(all_arg_names: List[str], names: Tuple[str, ...]) -> bool:
    if len(names) > 1:
        # Forward definition are always for a single output at a time
        return False
    name = names[0]
    if name not in all_arg_names:
        return True
    else:
        return False

def create_differentiability_info(
    defn: Dict[Any, Any],
    functions_by_signature: Dict[FunctionSchema, List[NativeFunction]],
    functions_by_schema: Dict[str, NativeFunction],
    op_counter: Counter[str],
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

        uses_grad = False                   # true if any derivative uses "grad"
        num_grads_uses = 0                  # count of uses of "grads" or "grads[INDEX]"
        uses_named_grads = False            # true if any derivative uses "grad_{name}"
        used_grads_indices: List[int] = []  # which indices of grads are used
        for d in derivatives:
            formula = d.formula
            uses_grad = uses_grad or bool(re.findall(IDENT_REGEX.format('grad'), formula))
            num_grads_uses += len(re.findall(IDENT_REGEX.format('grads'), formula))
            uses_named_grads = uses_named_grads or bool(d.named_gradients)
            used_grads_indices.extend(used_gradient_indices(formula))
        # This is a basic sanity check: the number of places we see
        # "grads" should be no fewer than the number of indices we see
        # inside "grads". They may not be equal because we may use
        # "grads" without an index.
        assert num_grads_uses >= len(used_grads_indices)
        # Thus if the number is equal, every use of grads is also
        # indexed.
        only_used_grads_indices = num_grads_uses == len(used_grads_indices)

        if uses_grad and num_grads_uses > 0:
            raise RuntimeError(f"Derivative definition of {defn_name} in derivatives.yaml illegally "
                               "mixes use of 'grad' and 'grads'. Consider replacing "
                               "occurrences of 'grad' with 'grads[0]'")

        if only_used_grads_indices and set(used_grads_indices) == {0}:
            raise RuntimeError(f"Derivative definition of {defn_name} in derivatives.yaml solely "
                               "refers to 'grads[0]'.  If the first output is indeed the "
                               "only differentiable output, replace 'grads[0]' with 'grad'; "
                               "otherwise, there is a likely error in your derivatives "
                               "declaration.")

        if uses_named_grads and (uses_grad or num_grads_uses > 0):
            raise RuntimeError(
                f'Derivative definition of {defn_name} in derivatives.yaml illegally '
                'mixes use of "grad_RETURN_NAME" and "grad" or "grads[x]". Use '
                'only one method for identifying gradients.')


    @with_native_function
    def set_up_derivatives(f: NativeFunction) -> Tuple[
        Sequence[Derivative],
        Sequence[ForwardDerivative],
        Sequence[Binding],
        Sequence[str],
        Sequence[str],
    ]:
        # Set up the derivative information
        derivatives: List[Derivative] = []
        forward_derivatives: List[ForwardDerivative] = []
        non_differentiable_arg_names: List[str] = []
        args_with_derivatives_set: Set[str] = set()

        all_arg_names = [a.name for a in cpp_arguments(f)]

        # output_differentiability is captured from the enclosed
        # scope. Don't modify it.
        #
        # If it is not present, then no output is explicitly
        # undifferentiable.
        #
        # It may be present and shorter than the length of return
        # values. If that's the case, any return value that does not
        # have a corresponding entry is considered not differentiable.
        differentiability = output_differentiability or [True] * len(f.func.returns)
        # A return is available as a named gradient ...
        available_named_gradients = [
            f'grad_{ret.name}' for ret, differentiable in zip(f.func.returns, differentiability)
            # if it has not been explicitly made undifferentiable
            if differentiable
            # and if it has a name
            and ret.name is not None
            # and if its type is differentiable
            and ret.type.is_tensor_like()]

        for raw_names in sorted(defn.keys()):
            formula = defn[raw_names]
            names = split_names(raw_names)

            if is_forward_derivative_definition(all_arg_names, names):
                forward_derivatives.append(create_forward_derivative(f, formula, names))
            else:
                if formula.lower().strip() == 'non_differentiable':
                    non_differentiable_arg_names += names
                else:
                    derivative = create_derivative(f, formula, names,
                                                   available_named_gradients)
                    derivatives.append(derivative)
                    args_with_derivatives_set |= set(names)

        overlap = args_with_derivatives_set.intersection(non_differentiable_arg_names)
        if overlap:
            raise RuntimeError(f'derivatives definition for {defn} have overlapped non_differentiable '
                               f'and differentiable variables: {overlap}')

        # Next, let us determine the list of inputs in order.
        # TODO: do we need eagerly calculate and save it here? Can it be derived
        # from NativeFunction and `derivatives` on callsites instead?
        args_with_derivatives = [a for a in cpp_arguments(f) if a.name in args_with_derivatives_set]

        # Postprocess forward derivatives definitions now that we know the differentiable arguments
        forward_derivatives = postprocess_forward_derivatives(f, defn_name, all_arg_names, derivatives,
                                                              forward_derivatives, args_with_derivatives)

        # Test to see if the use of 'grads' makes sense.
        check_grad_usage(defn_name, derivatives)

        return (derivatives, forward_derivatives, args_with_derivatives,
                non_differentiable_arg_names, available_named_gradients)

    # NB: Removes 'name' from defn dictionary
    specification = defn.pop('name')
    defn_name, _ = split_name_params(specification)
    # NB: Removes 'output_differentiability' from defn dictionary
    #     `None` means all differentiable.
    output_differentiability = defn.pop('output_differentiability', None)
    output_differentiability_conditions = None
    if output_differentiability and any([isinstance(diff, str) for diff in output_differentiability]):
        if len(output_differentiability) != 1:
            raise RuntimeError(f'Not supported: for {specification},'
                               f'output_differentiability must either be '
                               f'List[bool] or a List[str] where each str is a '
                               f'condition. In the case where it is a condition, '
                               f'we only support single-output functions. '
                               f'Please file us an issue. ')
        output_differentiability_conditions = output_differentiability
        output_differentiability = [True]

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

    if 'result' in (a.name for a in cpp_arguments(canonical)):
        raise RuntimeError(f"Schema for {defn_name} has an argument named result, "
                           "but this is only allowed for outputs."
                           "Please use a different name in native_functions.yaml.")

    (derivatives, forward_derivatives, args_with_derivatives,
     non_differentiable_arg_names, available_named_gradients) = set_up_derivatives(canonical)

    used_named_gradients: Set[str] = set()
    for d in derivatives:
        used_named_gradients |= d.named_gradients

    # only assign an op name if we are actually going to calculate a derivative
    op = None
    if args_with_derivatives:
        op_prefix = _create_op_prefix(defn_name)
        op = f'{op_prefix}{op_counter[op_prefix]}'
        op_counter[op_prefix] += 1

    return DifferentiabilityInfo(
        name=defn_name,
        func=canonical,
        op=op,
        derivatives=derivatives,
        forward_derivatives=forward_derivatives,
        all_saved_inputs=dedup_vars([v for d in derivatives for v in d.saved_inputs]),
        all_saved_outputs=dedup_vars([v for d in derivatives for v in d.saved_outputs]),
        available_named_gradients=available_named_gradients,
        used_named_gradients=used_named_gradients,
        args_with_derivatives=args_with_derivatives,
        non_differentiable_arg_names=non_differentiable_arg_names,
        output_differentiability=output_differentiability,
        output_differentiability_conditions=output_differentiability_conditions,
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
    nctypes: List[NamedCType],
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
            'nctype': lambda name: NamedCType(name, BaseCType(intArrayRefT)),
        }),
        # replace self->sizes() with self_sizes_opt
        (r'{}->sizes\(\)', {
            'suffix': '_sizes_opt',
            'nctype': lambda name: NamedCType(name, OptionalCType(BaseCType(intArrayRefT))),
            'expr': lambda name: f'{name}.has_value() ? c10::optional<IntArrayRef>({name}->sizes()) : c10::nullopt',
        }),
        # replace self.options() with self_options
        (r'{}.options\(\)', {
            'suffix': '_options',
            'nctype': lambda name: NamedCType(name, BaseCType(tensorOptionsT)),
        }),
        # replace zeros_like(self) with self_info
        (r'zeros_like\({}\)', {
            'suffix': '_info',
            'nctype': lambda name: NamedCType(name, BaseCType(typeAndSizeT)),
            'expr': lambda name: name,  # at save-time
            'res': lambda name: name + '_info.zeros()',  # at eval-time
        }),
        # replace self.size(2) with self_size_2
        (r'{}.size\((\w+)\)', {
            'suffix': lambda m: '_argsize_{}'.format(*m.groups()),
            'nctype': lambda name: NamedCType(name, BaseCType(longT)),
        }),
        # replace self.numel() with self_numel
        (r'{}.numel\(\)', {
            'suffix': '_numel',
            'nctype': lambda name: NamedCType(name, BaseCType(longT)),
        }),
        # replace to_args_sizes(self) with self_args_sizes
        (r'to_args_sizes\({}\)', {
            'suffix': '_args_sizes',
            'nctype': lambda name: NamedCType(name, VectorCType(VectorCType(BaseCType(longT)))),
        }),
        # replace to_args_scalartypes(self) with self_args_scalartypes
        (r'to_args_scalartypes\({}\)', {
            'suffix': '_args_scalartypes',
            'nctype': lambda name: NamedCType(name, VectorCType(BaseCType(scalarTypeT))),
        }),
        # replace TensorGeometry(self) with self_geometry
        (r'TensorGeometry\({}\)', {
            'suffix': '_geometry',
            'nctype': lambda name: NamedCType(name, BaseCType(tensorGeometryT)),
        }),
        (r'{}.scalar_type\(\)', {
            'suffix': '_scalar_type',
            'nctype': lambda name: NamedCType(name, BaseCType(scalarTypeT)),
        }),
        # replace self.dim() with self_dim
        (r'{}.dim\(\)', {
            'suffix': '_dim',
            'nctype': lambda name: NamedCType(name, BaseCType(longT)),
        }),
        # replace self.strides() with self_strides
        (r'{}.strides\(\)', {
            'suffix': '_strides',
            'nctype': lambda name: NamedCType(name, BaseCType(intArrayRefT)),
            'expr': stride_expr,
        }),
        # replace self.is_conj() with self_conjugate
        (r'{}.is_conj\(\)', {
            'suffix': '_conjugate',
            'nctype': lambda name: NamedCType(name, BaseCType(boolT)),
        })
    ]

    # find which arguments need to be saved
    saved: List[SavedAttribute] = []

    for nctype in nctypes:
        name = nctype.name.name if isinstance(nctype.name, SpecialArgName) else nctype.name
        # First search the formula for expressions which can be evaluated
        # when the autograd Function is created to avoid saving variables
        for regex, info in REPLACEMENTS:
            def repl(m: Match[str]) -> str:
                suffix: str = info['suffix'](m) if callable(info['suffix']) else info['suffix']
                expr: str = info['expr'](name) if 'expr' in info else m.group(0)
                saved.append(SavedAttribute(
                    nctype=info['nctype'](name + suffix),
                    expr=expr,
                ))
                if 'res' in info:
                    replacement: str = info['res'](name)
                    return replacement
                return name + suffix

            formula = re.sub(regex.format(name), repl, formula)

        # c10::optional<std::string> types stored in Backward nodes must be
        # converted to c10::optional<c10::string_view> before being passed into
        # the backward function
        if nctype.type == OptionalCType(BaseCType(stringT)):
            formula = re.sub(
                rf'\b{name}\b',
                f'{name}.has_value() ? c10::optional<c10::string_view>({name}.value()) : c10::nullopt',
                formula)

        # Find any variables which remain in the formula and save them
        if re.search(IDENT_REGEX.format(name), formula):
            saved.append(SavedAttribute(
                nctype=nctype,
                expr=name,
            ))

    return formula, tuple(saved)

def _create_op_prefix(name: str) -> str:
    """Takes a native function name converts to a op prefix name.

    Note that the "name" parameter must be the native function name
    without the optional variant suffix, so "add" instead of
    "add.out".

    OP names correspond to classes, hence the change to title case.

    Example::
    >>> _create_op_prefix('add')
    'AddBackward'
    """
    camel_case = ''.join([p.title() for p in name.split('_')])
    return (camel_case + 'Backward').replace('ForwardBackward', 'Backward')


def dedup_vars(vars: Sequence[SavedAttribute]) -> Sequence[SavedAttribute]:
    seen: Set[str] = set()
    saved: List[SavedAttribute] = []
    for var in vars:
        name = var.nctype.name.name if isinstance(var.nctype.name, SpecialArgName) else var.nctype.name
        if name in seen:
            continue
        seen.add(name)
        saved.append(var)
    return saved
