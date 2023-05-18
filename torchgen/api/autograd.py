import re
from dataclasses import dataclass
from typing import cast, Dict, List, Match, Optional, Sequence, Set, Tuple

from torchgen import local

from torchgen.api import cpp
from torchgen.api.types import BaseCType, Binding, NamedCType, tensorListT
from torchgen.model import (
    FunctionSchema,
    ListType,
    NativeFunction,
    NativeFunctionsViewGroup,
    SchemaKind,
    Type,
)
from torchgen.utils import IDENT_REGEX


# Represents a saved attribute involved in backward calculation.
# Note that it can be a derived property of an input argument, e.g.:
# we could save `other.scalar_type()` instead of the entire `other` tensor.
@dataclass(frozen=True)
class SavedAttribute:
    # The NamedCType holds the updated name and cpp type of the attribute
    # for the name, Suffix is appended if it's derived property, e.g.: `other_scalar_type`
    nctype: NamedCType

    # The expression to read the derived property at save time, e.g.:
    # `other.scalar_type()`.
    expr: str


# Represents a backward formula that calculates derivatives for one
# or more tensors.
@dataclass(frozen=True)
class Derivative:
    # The formula string (legit C++ expression).
    # Note that expressions against input arguments have been replaced with the
    # corresponding saved attributes.
    # E.g.:
    #  raw formula: `mul_tensor_backward(grad, self, other.scalar_type())`
    #         here: `mul_tensor_backward(grad, self, other_scalar_type)`
    formula: str

    # The formula string before input argument replacement
    original_formula: str

    # Names of the arguments for which this formula calculates derivatives.
    var_names: Tuple[str, ...]

    # Saved inputs that are referenced by the formula.
    saved_inputs: Tuple[SavedAttribute, ...]

    # Saved outputs that are referenced by the formula.
    saved_outputs: Tuple[SavedAttribute, ...]

    # Gradients that are referenced by name in the formula.
    named_gradients: Set[str]


# Represents a forward formula that calculates forward derivatives
# for one tensor.
@dataclass(frozen=True)
class ForwardDerivative:
    # The formula string (legit C++ expression).
    # Note that special keywords such as "linear" or "element_wise" have been
    # replaced by the automatically generated formula.
    formula: str

    # Name of the output arguments for which this formula calculates forward
    # derivatives
    var_names: Tuple[str, ...]

    # Type of the output arguments for which this formula calculates forward
    # derivatives
    var_types: Tuple[Type, ...]

    # Inputs for which the forward derivatives are required for this formula
    required_inputs_fw_grad: Optional[Tuple[str, ...]]

    # Inputs for which the primal is required for this formula
    required_inputs_primal: Optional[Tuple[str, ...]]

    # Flag to specify if this formula requires the original value of self
    # This is only used by inplace operations
    required_original_self_value: bool

    # If this formula is specified in derivatives.yaml or if we are re-using the
    # out of place formula for inplace
    is_reusing_outplace_formula: bool


# Represents differentiability info for a NativeFunction.
@dataclass(frozen=True)
class DifferentiabilityInfo:
    # The base name read from derivatives.yaml.
    name: str

    # The matching native function.
    #
    # There can be multiple NativeFunction having the same base name:
    #  - different overloads with different types of input arguments;
    #  - in-place/out/functional variants of the same function;
    #
    # We first use the schema string (under the 'name' key) in derivatives.yaml
    # to find the NativeFunction having the same schema string.
    # Then we find the in-place/out/functional variants of the matching function.
    # Among these variants, we choose the one having the same name as the
    # derivatives.yaml entry. If there is no exact match, then we choose the
    # in-place variant.
    # TODO: maybe the logic to search for all variants is no longer necessary?
    func: NativeFunction

    # The name of the generated autograd function.
    # It's set only if we will calculate a derivative, i.e.
    # 'args_with_derivatives' is not empty.
    op: Optional[str]

    # The derivatives formulae for this function.
    # Note that the length of this sequence is the number of differentiable inputs
    derivatives: Sequence[Derivative]

    # The forward derivatives formulae for this function.
    # Note that the length of this sequence is the number of differentiable outputs
    forward_derivatives: Sequence[ForwardDerivative]

    # The union of 'saved_inputs' of all 'derivatives'.
    all_saved_inputs: Sequence[SavedAttribute]

    # The union of 'saved_outputs' of all 'derivatives'.
    all_saved_outputs: Sequence[SavedAttribute]

    # All named gradients that are available for use, in the same
    # order as in the grads vector.
    available_named_gradients: Sequence[str]

    # The named gradients that are used in any of the derivatives.
    # Invariant: all(name in available_named_gradients for name in used_named_gradients)
    used_named_gradients: Set[str]

    # The function's input arguments for which it calculates derivatives.
    # It's the union of 'var_names' of all 'derivatives', sorted by the
    # argument order in the function schema.
    args_with_derivatives: Sequence[Binding]

    # Names of arguments whose derivative formula is 'non_differentiable'.
    non_differentiable_arg_names: Sequence[str]

    # Raw data read from derivatives.yaml.
    output_differentiability: Optional[List[bool]]

    # output_differentiability in derivatives.yaml can be a list of
    # conditions that express if the output is differentiable. In this case,
    # the number of conditions must match the number of outputs
    # (NB: we only support one condition right now).
    # output_differentiability gets populated with True for each condition,
    # while output_differentiability_conditions gets populated with the conditions
    output_differentiability_conditions: Optional[List[str]]

    @property
    def has_derivatives(self) -> bool:
        return len(self.args_with_derivatives) > 0

    # Generates a new DifferentiabilityInfo using the exact same set of derivative information,
    # but with a new operator name.
    # This is used when generating "copy" variants of view ops,
    # which are able to use the exact same derivative formula as the original view op
    # See Note [Codegen'd {view}_copy Operators]
    def create_view_copy_from_view_derivative(
        self, g: NativeFunctionsViewGroup
    ) -> Optional["DifferentiabilityInfo"]:
        if g.view_copy is None:
            return None
        f = g.view_copy

        name_split_by_period = self.name.split(".", maxsplit=2)
        # Append a "_copy" to the base name of the operator (but keep the overload name the same)
        view_copy_name = f"{name_split_by_period[0]}_copy." + ".".join(
            name_split_by_period[1:]
        )
        view_copy_op_name = None if self.op is None else f"{self.op}_copy"

        return DifferentiabilityInfo(
            # Use the "_copy" version of name/func/op
            name=view_copy_name,
            func=f,
            op=view_copy_op_name,
            # But keep all derivative info the same
            derivatives=self.derivatives,
            forward_derivatives=self.forward_derivatives,
            all_saved_inputs=self.all_saved_inputs,
            all_saved_outputs=self.all_saved_outputs,
            available_named_gradients=self.available_named_gradients,
            used_named_gradients=self.used_named_gradients,
            args_with_derivatives=self.args_with_derivatives,
            non_differentiable_arg_names=self.non_differentiable_arg_names,
            output_differentiability=self.output_differentiability,
            output_differentiability_conditions=self.output_differentiability_conditions,
        )


def uses_ident(info: Optional[DifferentiabilityInfo], ident: str) -> bool:
    if info is None:
        return False
    for derivative in info.derivatives:
        formula = derivative.formula
        if re.search(IDENT_REGEX.format(ident), formula):
            return True
    return False


def uses_retain_variables(info: Optional[DifferentiabilityInfo]) -> bool:
    return uses_ident(info, "retain_variables")


def uses_single_grad(info: Optional[DifferentiabilityInfo]) -> bool:
    return uses_ident(info, "grad")


# Represents a differentiable `Argument`.
# How is it different from the `Argument` type?
# - It's processed Arguments which are differentiable and only used in the
#   context of the autograd codegen;
# - It can represent SelfArgument or regular Argument but not TensorOptionsArgument;
@dataclass(frozen=True)
class DifferentiableInput:
    name: str
    type: Type

    # TODO: only to keep it byte-for-byte compatible with the old codegen, should remove.
    cpp_type: str


# Represents a differentiable `Return`.
# How it it different from the `Return` type?
# - The name in `Return` is optional. Here it is always populated using the same
#   `cpp.return_names()` method.
#   TODO: some cpp naming logic (e.g. resolving name conflict) might be irrelevant?
# - It's processed Returns which are differentiable, in compliance with the
#   `output_differentiability` field defined in derivatives.yaml (if specified),
#   and are only used in the context of the autograd codegen;
@dataclass(frozen=True)
class DifferentiableOutput:
    name: str
    type: Type

    # TODO: only to keep it byte-for-byte compatible with the old codegen, should remove.
    cpp_type: str


@dataclass(frozen=True)
class NativeFunctionWithDifferentiabilityInfo:
    func: NativeFunction
    info: Optional[Dict[str, DifferentiabilityInfo]]
    fw_derivatives: Optional[Dict[str, Sequence[ForwardDerivative]]]


# TODO: Update comment below since it is out of date.
def dispatch_strategy(fn: NativeFunctionWithDifferentiabilityInfo) -> str:
    """How are we going to call the underlying implementation of a
    declaration?  There are two strategies:
        - use_derived: we want to call the implementation on CPUDoubleType
          (or a similar, derived Type instance).  Because these derived
          instances deal in Tensors, not Variables (it's a completely different
          object, so it doesn't dispatch back to VariableType), code on
          this dispatch path needs to wrap/unwrap tensors.  If the
          derived implementation takes and returns tensors, the
          implementation is usually differentiable (although we also use
          the derived dispatch path for non-differentiable functions
          that we still want to dispatch on the derived Type instance;
          e.g., size())
        - use_type: we want to call the implementation on Type, because
          it is implemented concretely, and the functions it invokes will
          get dispatched back to VariableType (which will ensure that they
          are differentiable.)
    """
    # fn is derived as long as any of its per-key differentiability infos
    # has_derivatives. dispatch_strategy() is used to guard generation of fns in VariableType
    # and ADInplaceOrViewType. We want to generate these functions as long as a
    # derivative is defined for ANY dispatch key.
    if fn.func.is_abstract or (
        fn.info is not None and any(info.has_derivatives for info in fn.info.values())
    ):
        # If the function is abstract (not implemented on at::Type), we must
        # call the implementation on the derived type with unpacked tensors.

        # If the function has a derivative specified and is concrete, we could
        # call either implementation. We prefer the calling the derived
        # type's implementation with unpacked tensors because it is more
        # performant in some cases: any internal calls to other ATen functions
        # won't have the history tracked.

        # If the function has a type dispatched argument (i.e. is a factory),
        # we prefer calling the derived type's implementation both because it is
        # more performant and to ensure factory functions return tensors with _version
        # of 0 (probably not strictly necessary, but nice to have to keeps versions simple
        # to understand.

        return "use_derived"
    else:
        # If the function is concrete (we don't have to override it) and we
        # didn't declare it in derivatives.yaml, we'll assume that it is
        # actually implemented out of differentiable functions. (This
        # assumption might not hold, but then you'll see gradcheck fail.)
        return "use_type"


def is_foreach_func(f: NativeFunction) -> bool:
    return f.func.name.name.base.startswith("_foreach_")


# note(crcrpar): Most foreach functions can reference an out-place `torch` function whose schema kind
# is functional for their backward derivatives (and forward derivatives in the future), i.e.,
# they would find such one in `functional_info_by_signature`. There however are some exceptions:
_foreach_with_inplace_ref = {"_foreach_zero_"}


# Checks if `function_schema` is a native, non-foreach function which `f`, a foreach function
# reference to generate derivatives.
def is_reference_for_foreach(
    f: NativeFunction,
    function_schema: FunctionSchema,
) -> bool:
    return (
        f.func.name.name.base.split("_foreach_")[-1] == function_schema.name.name.base
        and (
            not function_schema.name.name.inplace
            or str(f.func.name) in _foreach_with_inplace_ref
        )
        and all(
            ref_arg.type in (arg.type, getattr(arg.type, "elem", None))
            for arg, ref_arg in zip(
                f.func.arguments.flat_non_out,
                function_schema.arguments.flat_non_out,
            )
        )
    )


# TODO(crcrpar): Avoid hard coding "Default" ideally.
def gen_foreach_derivativeinfo(
    foreach_function: NativeFunction,
    functional_info_by_signature: Dict[
        FunctionSchema, Dict[str, DifferentiabilityInfo]
    ],
    non_functional_info_by_signature: Dict[
        FunctionSchema, Dict[str, DifferentiabilityInfo]
    ],
    dispatch_key: str = "Default",
) -> Tuple[Optional[DifferentiabilityInfo], bool]:
    """Generate DifferentiabilityInfo for out-place foreach function, return the existing one for in-place.

    The second return value indicates whether the info is generated in this function.
    """
    ref_diff_info: Optional[DifferentiabilityInfo] = None

    for function_schema, diff_info in functional_info_by_signature.items():
        if not is_reference_for_foreach(foreach_function, function_schema):
            continue
        ref_diff_info = diff_info[dispatch_key]
        if ref_diff_info is not None:
            break
    # note(crcrpar): It seems like `zero`'s info isn't available in functional_info_by_signature
    # while the info of `zero_` is in non_functional_info_by_signature
    if (
        ref_diff_info is None
        and foreach_function.func.kind() == SchemaKind.inplace
        and str(foreach_function.func.name) in _foreach_with_inplace_ref
    ):
        for function_schema, diff_info in non_functional_info_by_signature.items():
            if not is_reference_for_foreach(foreach_function, function_schema):
                continue
            ref_diff_info = diff_info[dispatch_key]
            if ref_diff_info is not None:
                break
    if ref_diff_info is None:
        return None, False

    # non out-place uses the existing Derivative.
    if foreach_function.func.kind() == SchemaKind.inplace:
        return ref_diff_info, False

    map_refarg2foreacharg, map_name2arg = {}, {}
    for i, (arg, ref_arg) in enumerate(
        zip(
            foreach_function.func.arguments.flat_non_out,
            function_schema.arguments.flat_non_out,
        )
    ):
        map_refarg2foreacharg[ref_arg.name] = arg.name
        map_name2arg[arg.name] = arg

    all_saved_inputs, all_saved_outputs, all_var_names = [], [], []
    modified_derivative_formulas = []
    for i, derivative in enumerate(ref_diff_info.derivatives):
        modified_formula = derivative.formula.replace("grad", "grads[i]").replace(
            "result", "result[i]"
        )
        saved_inputs, saved_outputs = [], []
        # note(crcrpar): This context seems necessary to call `cpp.argument_type`
        with local.parametrize(
            use_const_ref_for_mutable_tensors=foreach_function.use_const_ref_for_mutable_tensors,
            use_ilistref_for_tensor_lists=foreach_function.part_of_structured_group,
        ):
            for ref_input in derivative.saved_inputs:
                ref_input_jit_name = ref_input.expr.split(".")[0]
                mapped_name = map_refarg2foreacharg[ref_input_jit_name]
                if isinstance(map_name2arg[mapped_name].type, ListType):
                    mapped_expr = mapped_name + "[i]"
                else:
                    mapped_expr = mapped_name
                new_expr = ref_input.expr.replace(ref_input_jit_name, mapped_expr)
                modified_formula = modified_formula.replace(
                    cast(str, ref_input.nctype.name), new_expr
                )

                nctype = cpp.argument_type(map_name2arg[mapped_name], binds=mapped_name)
                canonical_nctype = NamedCType(
                    nctype.name, nctype.type.remove_const_ref()
                )
                saved_inputs.append(
                    SavedAttribute(nctype=canonical_nctype, expr=mapped_name)
                )
            for ref_output in derivative.saved_outputs:
                if ref_output.nctype.name == "result":
                    saved_outputs.append(
                        SavedAttribute(
                            nctype=NamedCType(
                                name="result", type=BaseCType(tensorListT)
                            ),
                            expr="result",
                        )
                    )
                else:
                    raise RuntimeError("")
        var_names = [map_refarg2foreacharg[var] for var in derivative.var_names]
        all_var_names.extend(var_names)
        all_saved_inputs.extend(saved_inputs)
        all_saved_outputs.extend(saved_outputs)
        modified_derivative = Derivative(
            formula=modified_formula,
            original_formula=derivative.formula,
            var_names=tuple(var_names),
            saved_inputs=tuple(saved_inputs),
            saved_outputs=tuple(saved_outputs),
            named_gradients=set(),
        )
        modified_derivative_formulas.append(modified_derivative)

    with local.parametrize(
        use_const_ref_for_mutable_tensors=foreach_function.use_const_ref_for_mutable_tensors,
        use_ilistref_for_tensor_lists=foreach_function.part_of_structured_group,
    ):
        args_with_derivatives = [
            Binding(
                name=arg.name,
                nctype=cpp.argument_type(arg, binds=arg.name),
                argument=arg,
                default=None,
            )
            for arg in foreach_function.func.arguments.flat_non_out
            if arg.name in all_var_names
        ]
    return (
        DifferentiabilityInfo(
            name=foreach_function.func.name.name.base,
            func=foreach_function,
            op="Foreach{}{}".format(
                ref_diff_info.op, foreach_function.func.name.overload_name
            ),
            derivatives=modified_derivative_formulas,
            forward_derivatives=[],
            all_saved_inputs=tuple(set(all_saved_inputs)),
            all_saved_outputs=tuple(set(all_saved_outputs)),
            available_named_gradients=(),
            used_named_gradients=set(),
            args_with_derivatives=args_with_derivatives,
            non_differentiable_arg_names=[],
            output_differentiability=None,
            output_differentiability_conditions=None,
        ),
        True,
    )


def match_differentiability_info(
    native_functions: List[NativeFunction],
    differentiability_infos: Dict[FunctionSchema, Dict[str, DifferentiabilityInfo]],
) -> List[NativeFunctionWithDifferentiabilityInfo]:
    """Sets the "derivative" key on declarations to matching autograd function
    In-place functions will use the out-of-place derivative definition if there
    is no in-place specific derivative.
    """

    functional_info_by_signature = {
        schema.signature(strip_default=True): info_dict
        for schema, info_dict in differentiability_infos.items()
        if schema.kind() == SchemaKind.functional
    }
    non_functional_info_by_signature = {
        schema.signature(strip_default=True): info_dict
        for schema, info_dict in differentiability_infos.items()
        if schema.kind() != SchemaKind.functional
    }

    def find_info(
        f: NativeFunction,
    ) -> Tuple[Optional[Dict[str, DifferentiabilityInfo]], bool]:
        # Don't bother matching info to generated out= variants
        if "generated" in f.tags and f.func.kind() == SchemaKind.out:
            return None, False

        # (1) Check for an exact match
        if f.func in differentiability_infos:
            return differentiability_infos[f.func], True

        # (2) If no exact match, check if the out-of-place variant
        # of this operator has a match.
        # i.e mul() for mul_() or mul_out()
        # note(crcrpar): Check foreach or not because in-place foreach functions use backward defined for the existing
        # native functions instead of the out-place counterparts.
        f_sig = f.func.signature(strip_default=True)
        if f_sig in functional_info_by_signature and not is_foreach_func(f):
            return functional_info_by_signature[f_sig], False

        # (3) Some operators have a derivative explicitly defined for the mutable
        # variant, but get a code-generated out-of-place variant which does *not*
        # come with a derivative formula.
        # For the generated out-of-place variant, use the mutable variant's formula
        # if it exists.
        if "generated" in f.tags and f_sig in non_functional_info_by_signature:
            info_dict = non_functional_info_by_signature[f_sig]
            # See https://github.com/pytorch/pytorch/pull/76320/files#r874816389
            assert not any(
                any("self" in str(inpt.nctype.name) for inpt in info.all_saved_inputs)
                for info in info_dict.values()
            ), f"""\
Attempted to convert a derivative formula for a mutable operator
 to be used by automatically by its functional variant ("{str(f.func)}").
 this is not currently supported (we'd need to fix up the formula in the codegen)."""
            return info_dict, False

        # (4) Generate derivative information of foreach functions if none is defined in `derivatives.yaml`
        if is_foreach_func(f):
            assert f.func not in differentiability_infos
            diff_info, is_generated = gen_foreach_derivativeinfo(
                f,
                functional_info_by_signature,
                non_functional_info_by_signature,
            )
            if diff_info is None:
                return None, False
            # TODO(crcrpar): Avoid hard coding "Default" ideally.
            diff_info_dict = {"Default": diff_info}
            if is_generated:
                differentiability_infos[f.func] = diff_info_dict
                functional_info_by_signature[f.func] = diff_info_dict
            return diff_info_dict, is_generated

        return None, False

    result: List[NativeFunctionWithDifferentiabilityInfo] = []
    for f in native_functions:
        info_dict, is_exact_match = find_info(f)

        # Currently, the '.strides()' to 'strides_or_error' replacement does not support
        # 'self' derivatives of an inplace function, so we must check for this case.
        if f.func.kind() == SchemaKind.inplace and (info_dict is not None):
            for info in info_dict.values():
                for derivative in info.derivatives:
                    if "self" in derivative.var_names:
                        for saved_input in derivative.saved_inputs:
                            assert "strides_or_error" not in saved_input.expr, (
                                "Calling '.strides()' in the 'self' derivative formula of an "
                                f"in-place function is not supported: {f.func}"
                            )

        if not info_dict:
            result.append(
                NativeFunctionWithDifferentiabilityInfo(
                    func=f, info=None, fw_derivatives=None
                )
            )
            continue

        fw_derivative_dict: Dict[str, Sequence[ForwardDerivative]] = {}
        for key, info in info_dict.items():
            if not info.forward_derivatives:
                fw_derivative_dict[key] = []
                continue

            forward_derivatives = info.forward_derivatives

            # For functions that have a single def for out-of-place and inplace (like abs())
            if f.func.kind() == SchemaKind.inplace:
                # For inplace functions there is a little bit of work to do:
                #  1) Validate the formula and make sure the input that is modified in not used:
                #    - If there is a formula for the inplace variant of the function (is_exact_match == True) then
                #      we make sure that the original value of the input that is being modified inplace (self_p) is
                #      not used in the formula. Note that the formula can use "original_self_p" here and that would
                #      trigger a clone of the original input.
                #    - If we are re-using the out of place formula (is_exact_match == False) then we replace every
                #      occurrence of self_p and self_t by original_self_p and original_self_t. These will be
                #      populated by cloned version of the original input (either the clone done by the backward AD
                #      logic if self is also used in a backward formula or a special clone that we add).
                #  2) At this point, there cannot be a self_p in the formula.
                #  3) Change "result" into "self_p" as by design, in the inplace function codegen, the result is
                #     simply called self (as it is modified inplace).
                #  4) Update the required primals data in case it used to contain "result" but should now contain
                #     "self"
                #  5) If it is not an exact match, the user formula is not modifying the existing forward grad
                #     inplace as it should. So add some code that makes sure that we do so if the forward grad
                #     already exists.

                assert (
                    len(info.forward_derivatives) == 1
                )  # Only single output inplace should exist
                fw_info = info.forward_derivatives[0]
                formula = fw_info.formula

                def replace_self_with_original_self(formula: str, postfix: str) -> str:
                    def repl(m: Match[str]) -> str:
                        return f"{m.group(1)}original_self{postfix}{m.group(2)}"

                    return re.sub(IDENT_REGEX.format(f"self{postfix}"), repl, formula)

                if re.search(IDENT_REGEX.format("self_p"), formula):
                    if is_exact_match:
                        # For manually defined formulas, don't allow the original value to be used
                        raise RuntimeError(
                            f'The formula for "{f.func.name}" is using the original value of self '
                            "that is being modified inplace. This would lead to wrong forward gradients. "
                            'Please use "result" in the formula only.'
                        )
                    else:
                        # When the original formula is out of place, we save a clone of the primal
                        # value to be able to access this value if needed
                        # replace "self_p"/"self_t" from the formula by "original_self_p"/"original_self_t"
                        formula = replace_self_with_original_self(formula, "_p")
                        formula = replace_self_with_original_self(formula, "_t")

                # replace "result" from the formula by "self_p"
                def repl(m: Match[str]) -> str:
                    return f"{m.group(1)}self_p{m.group(2)}"

                formula = re.sub(IDENT_REGEX.format("result"), repl, formula)

                required_primals = fw_info.required_inputs_primal
                if re.search(IDENT_REGEX.format("self_p"), formula):
                    required_primals = (
                        required_primals + ("self",) if required_primals else ("self",)
                    )

                if not is_exact_match:
                    # NOTE [In-place forward AD formula Optimization]
                    #
                    # This optimization transforms the formula to directly do inplace, i.e.
                    # instead of self_t.copy_(self_t.op()) we do self_t.op_() when the following are met:
                    #
                    # 1) the formula satisfies the pattern: "self_t.op(*args)"
                    # 2) "op" in (1) needs to be the same as the op the derivative is for
                    #
                    # (2) may seem too strict, but currently the only ops that satisfy (1) also satisfy (2)
                    # If there is a need, we can relax (2) to allow any op that has an in-place variant
                    is_single_method_on_self_t = False
                    directly_do_inplace = False
                    op_name: Optional[str] = None
                    between_parens: Optional[str] = None
                    match = re.fullmatch(r"self_t.([\w]*)\((.*)\)", formula)
                    if match:
                        op_name, between_parens = match.group(1), match.group(2)

                        # We want to...
                        #   Match: self_t.op1(other_p.op2(arg))
                        #   Avoid: self_t.op1(args) + self_t.op2(args)
                        #   Avoid: self_t.op1(other_p.op2(arg)) + self_t.op2(args)
                        def check_parens_nest_level_gt_zero(s: str) -> bool:
                            level = 1
                            for ch in s:
                                if ch == ")":
                                    level -= 1
                                    if level == 0:
                                        return False
                                if ch == "(":
                                    level += 1
                            return True

                        is_single_method_on_self_t = check_parens_nest_level_gt_zero(
                            between_parens
                        )
                        directly_do_inplace = (
                            is_single_method_on_self_t and op_name == info.name
                        )

                    if directly_do_inplace:
                        assert op_name is not None
                        assert between_parens is not None
                        formula = f"self_t_raw.defined() ? self_t_raw.{op_name}_({between_parens}) : {formula}"
                    else:
                        # Make sure that the forward grad is modified inplace when the original formula
                        # is out of place
                        formula = f"self_t_raw.defined() ? self_t_raw.copy_({formula}) : {formula}"

                required_original_self_value = bool(
                    re.search(IDENT_REGEX.format("original_self_p"), formula)
                ) or bool(re.search(IDENT_REGEX.format("original_self_t"), formula))

                forward_derivatives = [
                    ForwardDerivative(
                        formula=formula,
                        var_names=("self",),
                        var_types=fw_info.var_types,
                        required_inputs_fw_grad=fw_info.required_inputs_fw_grad,
                        required_inputs_primal=required_primals,
                        required_original_self_value=required_original_self_value,
                        is_reusing_outplace_formula=not is_exact_match,
                    ),
                ]

            fw_derivative_dict[key] = forward_derivatives

        result.append(
            NativeFunctionWithDifferentiabilityInfo(
                func=f, info=info_dict, fw_derivatives=fw_derivative_dict
            )
        )

    return result


def is_differentiable(
    name: str, type: Type, info: Optional[DifferentiabilityInfo]
) -> bool:
    return type.is_tensor_like() and (
        info is None or name not in info.non_differentiable_arg_names
    )


def gen_differentiable_outputs(
    fn: NativeFunctionWithDifferentiabilityInfo, key: str = "Default"
) -> List[DifferentiableOutput]:
    f = fn.func
    info = fn.info[key] if fn.info else None
    outputs: List[DifferentiableOutput] = [
        DifferentiableOutput(
            name=name,
            type=ret.type,
            cpp_type=cpp.return_type(ret, symint=True).cpp_type(),
        )
        for name, ret in zip(cpp.return_names(f), f.func.returns)
    ]
    output_differentiability = info.output_differentiability if info else None
    if output_differentiability is not None:
        if len(output_differentiability) != len(outputs):
            raise RuntimeError(
                f"The length of output_differentiability ({len(output_differentiability)}), "
                f"does not match the number of outputs ({len(outputs)})."
            )
        differentiable_outputs: List[DifferentiableOutput] = []
        if False in output_differentiability and f.func.kind() == SchemaKind.inplace:
            raise RuntimeError(
                "output_differentiability=False for inplace operation (version_counter won't get updated)"
            )
        for differentiable, output in zip(output_differentiability, outputs):
            if differentiable:
                differentiable_outputs.append(output)
        return differentiable_outputs
    candidate_differentiable_outputs = list(
        filter(lambda r: is_differentiable(r.name, r.type, info), outputs)
    )
    if uses_single_grad(info):
        return candidate_differentiable_outputs[:1]
    else:
        return candidate_differentiable_outputs
