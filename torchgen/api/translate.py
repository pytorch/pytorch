from typing import Dict, List, NoReturn, Sequence, Union

from torchgen.api.types import (
    ArrayRefCType,
    BaseCType,
    Binding,
    boolT,
    ConstRefCType,
    deviceT,
    Expr,
    intArrayRefT,
    iOptTensorListRefT,
    layoutT,
    ListCType,
    longT,
    memoryFormatT,
    MutRefCType,
    NamedCType,
    opmath_t,
    OptionalCType,
    optionalIntArrayRefT,
    optionalScalarRefT,
    optionalSymIntArrayRefT,
    optionalTensorRefT,
    scalar_t,
    scalarT,
    scalarTypeT,
    SpecialArgName,
    symIntArrayRefT,
    SymIntT,
    tensorOptionsT,
    tensorT,
    VectorCType,
)

# This file implements a small program synthesis engine that implements
# conversions between one API to another.
#
# The key data type in this file in NamedCType, short for Named C++ semantic type.  A NamedCType
# represents a C++ type, plus semantic information about what it represents.
# For example, consider the argument "bool pin_memory"; its normal C++ type is
# "bool", but its C++ semantic type also keeps track that this represents a
# "pin_memory"; you can't just use a random other boolean in a context where you
# need a "pin_memory"!
#
# The translator takes a list of needed NamedCTypes, and then figures out how
# to construct expressions with these NamedCTypes from the given bindings.  Many
# of these expressions are trivial (I need a Tensor other; there's a Tensor
# other scope); others are more nontrivial and may require packing/unpacking.
# Some examples of non-trivial action:
#
#   - Need the "dtype" binding?  Well, maybe "dtype" isn't available
#     in the context, instead, "options" is, and you need to extract
#     it from there.  (Gather)
#
#   - Need the "context" binding?  Well, maybe "context" isn't available
#     in the context, and you need to construct it from "dtype", "device",
#     etc.  (Scatter)
#
#   - Need the "memory_format" binding?  Well, actually, it's available
#     from both "memory_format" and "options", so you had better make sure
#     they are consistent.  (Join)

options_ctype = NamedCType("options", ConstRefCType(BaseCType(tensorOptionsT)))

out_tensor_ctype = NamedCType("out", ConstRefCType(BaseCType(tensorT)))

longVec_ctype = VectorCType(BaseCType(longT))
longSymVec_ctype = VectorCType(BaseCType(SymIntT))
optionalLongVec_ctype = OptionalCType(VectorCType(BaseCType(longT)))
optionalScalar_ctype = OptionalCType(BaseCType(scalarT))
optionalTensor_ctype = OptionalCType(BaseCType(tensorT))


class UnsatError(RuntimeError):
    pass


# Given a set of in-scope bindings and a set of target bindings, synthesize
# a list of expressions that uses only the in-scope bindings (bindings) that
# have all of the types of goals.  You may want to use this function if
# you're generating code for a function like:
#
#   void f({args}) {
#     g({exprs}); // g is a different API
#   }
#
# and you need to generate "exprs".
#
# Typically, a list of Bindings is convenient to get (you usually call something
# like arguments() to get them); but technically you only need less information:
# for 'bindings' an (un-ordered) list of Exprs is sufficient; similarly, for
# 'goals', an (ordered) list of NamedCType goals is sufficient.  If you are doing
# something more complicated, e.g., tracking the set of bindings in a context,
# you may find using these smaller types more convenient.
def translate(
    bindings: Sequence[Union[Expr, Binding]],
    goals: Sequence[Union[NamedCType, Binding]],
    *,
    method: bool = False,
    allow_expensive_conversions: bool = False,
) -> List[Expr]:
    binding_exprs: List[Expr] = []
    for b in bindings:
        if isinstance(b, Binding):
            binding_exprs.append(
                Expr(
                    expr=b.name,
                    type=b.nctype,
                )
            )
        else:
            binding_exprs.append(b)

    goal_ctypes: List[NamedCType] = []
    for g in goals:
        if isinstance(g, Binding):
            goal_ctypes.append(g.nctype)
        else:
            goal_ctypes.append(g)

    # Add all the bindings to the context
    ctx: Dict[NamedCType, str] = {}
    for b in binding_exprs:
        ctx[b.type] = b.expr

        # While we're at it, do some simple forward inference, looking through
        # constructors.
        #
        # NB: When should you do forward inference versus backward inference?
        # The general idea:
        #
        #   - Backward inference WHEN the goal gets smaller
        #   - Forward inference WHEN the hypothesis gets smaller
        #
        # This helps ensure termination: backward inference starts with a goal
        # and tries to make it simpler and simpler until it's trivial; if the
        # goal can grow in size, we blow up to a really huge goal size.
        # Similarly, with forward inference we take hypotheses and decompose
        # them into simpler hypotheses; if hypotheses could expand in size,
        # we also have potential nontermination.  (In the code below, forward
        # inference is only ever carried out at a single step, but you could
        # imagine repeated application of forward inference being profitable.)
        #
        # A good starting point in the literature for exploring more about proof
        # search are these lecture notes
        # https://www.cs.cmu.edu/~fp/courses/oregon-m10/04-focusing.pdf
        #
        # TODO: My kingdom for a pattern matcher
        # https://www.python.org/dev/peps/pep-0634/
        #
        # TODO: This could get us in recomputation trouble if b.expr is nontrivial.
        # Fix this by implementing some sort of sharing so that if multiple
        # goals share the same expression, we only compute it once.  This seems
        # to matter in practice as compiler is often unwilling to CSE nontrivial
        # expressions like scalar.to<scalar_t>()
        t = b.type
        if (
            isinstance(t, ConstRefCType)
            and isinstance(t.elem, OptionalCType)
            and isinstance(t.elem.elem, BaseCType)
            and str(t.elem.elem.type) == "at::Tensor"
        ):
            ctx[
                NamedCType(t.elem.elem.name, ConstRefCType(BaseCType(tensorT)))
            ] = f"({b.expr}.has_value() ? *{b.expr} : at::Tensor())"

        if t.type == ConstRefCType(OptionalCType(BaseCType(tensorT))):
            ctx[
                NamedCType(t.name, BaseCType(optionalTensorRefT))
            ] = f"(({b.expr}.has_value() && (*{b.expr}).defined()) ? at::OptionalTensorRef(*{b.expr}) : at::OptionalTensorRef())"

        if t.type == ConstRefCType(BaseCType(scalarT)):
            ctx[NamedCType(t.name, BaseCType(opmath_t))] = f"({b.expr}).to<opmath_t>()"

        if t.type == ConstRefCType(OptionalCType(BaseCType(scalarT))):
            ctx[
                NamedCType(t.name, BaseCType(optionalScalarRefT))
            ] = f"({b.expr}.has_value() ? at::OptionalScalarRef(&({b.expr}.value())) : at::OptionalScalarRef())"

        if t.type == BaseCType(scalar_t):
            ctx[
                NamedCType(t.name, BaseCType(opmath_t))
            ] = f"static_cast<opmath_t>({b.expr})"

        # [Note: IOptTensorListRef]
        if t.type == ConstRefCType(ListCType(OptionalCType(BaseCType(tensorT)))):
            ctx[
                NamedCType(t.name, BaseCType(iOptTensorListRefT))
            ] = f"at::IOptTensorListRef({b.expr})"

    # Add implicit bindings if the generated code is inside a Tensor method
    if method:
        ctx[
            NamedCType("self", MutRefCType(BaseCType(tensorT)))
        ] = "const_cast<Tensor&>(*this)"
        ctx[
            NamedCType("self", ConstRefCType(BaseCType(tensorT)))
        ] = "const_cast<Tensor&>(*this)"
        # This is better!  Byte-for-byte compat
        # ctx[NamedCType("self", ConstRefCType(BaseCType(tensorT)))] = "*this"

    def unsat(goal: NamedCType) -> NoReturn:
        ctx_desc = "\n".join(
            f"  {t.cpp_type()} {t.name}; // {e}" for t, e in ctx.items()
        )
        raise UnsatError(
            f"""
Failed to synthesize the expression "{goal.cpp_type()} {goal.name}".
When I failed, the following bindings were available in the context:

{ctx_desc}

This probably means there is a missing rule in the rules of torchgen.api.translate.
Check this module for more information.
"""
        )

    # A shitty backtracking search implementation.  It's shitty because it
    # does backtracking via stack (bad idea!) and for the most part tries to
    # avoid backtracking.  In particular, if
    # direct=True, we won't try to do any fancy synthesis, just trivial
    # conversions (e.g., "T a" is OK for "const T& a").  So all of the
    # existing rules in this function simply try to solve immediately,
    # and bail if things don't work out.
    def solve(goal: NamedCType, *, direct: bool) -> str:
        def direct_solve(goal: NamedCType) -> str:
            return solve(goal, direct=True)

        if goal in ctx:
            # Trivial
            return ctx[goal]

        # const & is satisfied with mutable &
        if isinstance(goal.type, ConstRefCType):
            try:
                # WARNING: not strictly decreasing; be careful not
                # to add a direct conversion that goes satisfies
                # mutable& with const&
                return solve(
                    NamedCType(goal.name, MutRefCType(goal.type.elem)), direct=direct
                )
            except UnsatError:
                pass

        # mutable & is satisfied with value
        if isinstance(goal.type, MutRefCType):
            try:
                return solve(NamedCType(goal.name, goal.type.elem), direct=direct)
            except UnsatError:
                pass

        # TODO: These are referentially equal, shouldn't have to do this;
        # ensuring we don't use type synonym IntArrayRef in codegen would
        # help
        if goal.type == ArrayRefCType(BaseCType(longT)):
            return solve(NamedCType(goal.name, BaseCType(intArrayRefT)), direct=direct)

        if direct:
            unsat(goal)

        # For now, all of these rules are mutually exclusive.
        if goal == NamedCType("memory_format", OptionalCType(BaseCType(memoryFormatT))):
            memory_format = direct_solve(
                NamedCType(
                    SpecialArgName.possibly_redundant_memory_format,
                    OptionalCType(BaseCType(memoryFormatT)),
                )
            )
            # No need to join "memory_format" and "options" if the target API takes "options" directly.
            # Otherwise it will cause the redundant memory_format error.
            if options_ctype in goal_ctypes:
                return memory_format
            try:
                options = direct_solve(options_ctype)
                return f"c10::impl::check_tensor_options_and_extract_memory_format({options}, {memory_format})"
            except UnsatError:
                return memory_format
        elif goal == NamedCType("options", BaseCType(tensorOptionsT)):
            dtype = direct_solve(
                NamedCType("dtype", OptionalCType(BaseCType(scalarTypeT)))
            )
            pin_memory = direct_solve(
                NamedCType("pin_memory", OptionalCType(BaseCType(boolT)))
            )
            device = direct_solve(
                NamedCType("device", OptionalCType(BaseCType(deviceT)))
            )
            layout = direct_solve(
                NamedCType("layout", OptionalCType(BaseCType(layoutT)))
            )
            return f"TensorOptions().dtype({dtype}).layout({layout}).device({device}).pinned_memory({pin_memory})"

        elif goal == NamedCType("dtype", OptionalCType(BaseCType(scalarTypeT))):
            try:
                options = direct_solve(options_ctype)
                return f"c10::optTypeMetaToScalarType({options}.dtype_opt())"
            except UnsatError:
                out_tensor = direct_solve(out_tensor_ctype)
                return f"{out_tensor}.scalar_type()"

        elif goal == NamedCType("layout", OptionalCType(BaseCType(layoutT))):
            try:
                options = direct_solve(options_ctype)
                return f"{options}.layout_opt()"
            except UnsatError:
                out_tensor = direct_solve(out_tensor_ctype)
                return f"{out_tensor}.layout()"

        elif goal == NamedCType("device", OptionalCType(BaseCType(deviceT))):
            try:
                options = direct_solve(options_ctype)
                return f"{options}.device_opt()"
            except UnsatError:
                out_tensor = direct_solve(out_tensor_ctype)
                return f"{out_tensor}.device()"

        elif goal == NamedCType("pin_memory", OptionalCType(BaseCType(boolT))):
            try:
                options = direct_solve(options_ctype)
                return f"{options}.pinned_memory_opt()"
            except UnsatError:
                # If we're calling a factory op from its out= variant,
                # We don't actually care about the value of pin_memory.
                out_tensor = direct_solve(out_tensor_ctype)
                return "c10::nullopt"

        # We can always do translations from value types to reference types, like vector<int> -> IntArrayRef
        elif goal.type == BaseCType(intArrayRefT):
            try:
                return direct_solve(NamedCType(goal.name, longVec_ctype))
            except UnsatError:
                # We can also go SymIntArrayRef -> IntArrayRef
                symIntArrayRef_type = direct_solve(
                    NamedCType(goal.name, BaseCType(symIntArrayRefT))
                )
                return f"C10_AS_INTARRAYREF_SLOW({symIntArrayRef_type})"
        elif goal.type == BaseCType(symIntArrayRefT):
            try:
                r = direct_solve(NamedCType(goal.name, BaseCType(intArrayRefT)))
                return f"c10::fromIntArrayRefSlow({r})"
            except UnsatError:
                return direct_solve(NamedCType(goal.name, longSymVec_ctype))
        elif goal.type == BaseCType(SymIntT):
            return direct_solve(NamedCType(goal.name, BaseCType(longT)))
        elif goal.type == OptionalCType(BaseCType(SymIntT)):
            argname = direct_solve(
                NamedCType(goal.name, OptionalCType(BaseCType(longT)))
            )
            return f"{argname}.has_value() ? c10::make_optional(c10::SymInt(*{argname})) : c10::nullopt"
        elif goal.type == BaseCType(longT):
            symInt_type = direct_solve(NamedCType(goal.name, BaseCType(SymIntT)))
            return f"{symInt_type}.guard_int(__FILE__, __LINE__)"
        elif goal.type == OptionalCType(BaseCType(longT)):
            argname = direct_solve(
                NamedCType(goal.name, OptionalCType(BaseCType(SymIntT)))
            )
            return f"{argname}.has_value() ? c10::make_optional({argname}->guard_int(__FILE__, __LINE__)) : c10::nullopt"
        elif goal.type == BaseCType(optionalIntArrayRefT):
            try:
                return direct_solve(NamedCType(goal.name, optionalLongVec_ctype))
            except UnsatError:
                argname = direct_solve(
                    NamedCType(goal.name, BaseCType(optionalSymIntArrayRefT))
                )
                return f"{argname}.has_value() ? c10::make_optional(C10_AS_INTARRAYREF_SLOW(*{argname})) : c10::nullopt"
        elif goal.type == BaseCType(optionalSymIntArrayRefT):
            # TODO: You might also want to solve this from longSymVec_ctype or
            # an optional version of it
            argname = direct_solve(
                NamedCType(goal.name, BaseCType(optionalIntArrayRefT))
            )
            return f"{argname}.has_value() ? c10::make_optional(c10::fromIntArrayRefSlow(*{argname})) : c10::nullopt"
        elif goal.type == BaseCType(optionalScalarRefT):
            return direct_solve(NamedCType(goal.name, optionalScalar_ctype))
        elif goal.type == BaseCType(optionalTensorRefT):
            return direct_solve(NamedCType(goal.name, optionalTensor_ctype))

        # Note [translation from C++ reference to value types]
        # The below cases are all for when we have an argument with a reference type,
        # and a corresponding goal with a value type.
        # These are needed when we populate the inputs to a lambda capture and we need
        # to guarantee the lifetime of each captured argument.
        # We guard it with an explicit kwarg because converting to a value type is expensive
        # (O(n)) to convert from IntArrayRef to vector<int>),
        # so the caller of translate() should be explicit that they need it.
        if allow_expensive_conversions:
            if goal.type == VectorCType(BaseCType(longT)):
                intArrayRef_ctype = NamedCType(goal.name, BaseCType(intArrayRefT))
                argname = direct_solve(intArrayRef_ctype)
                return f"{argname}.vec()"
            if goal.type == VectorCType(BaseCType(SymIntT)):
                symIntArrayRef_ctype = NamedCType(goal.name, BaseCType(symIntArrayRefT))
                argname = direct_solve(symIntArrayRef_ctype)
                return f"{argname}.vec()"
            elif goal.type == OptionalCType(VectorCType(BaseCType(longT))):
                optionalIntArrayRef_ctype = NamedCType(
                    goal.name, BaseCType(optionalIntArrayRefT)
                )
                argname = direct_solve(optionalIntArrayRef_ctype)
                return f"{argname}.has_value() ? c10::make_optional({argname}->vec()) : c10::nullopt"
            elif goal.type == OptionalCType(BaseCType(scalarT)):
                optionalScalarRef_ctype = NamedCType(
                    goal.name, BaseCType(optionalScalarRefT)
                )
                argname = direct_solve(optionalScalarRef_ctype)
                return f"{argname}.has_value() ? c10::make_optional({argname}) : c10::nullopt"
            elif goal.type == OptionalCType(BaseCType(scalarT)):
                optionalTensorRef_ctype = NamedCType(
                    goal.name, BaseCType(optionalTensorRefT)
                )
                argname = direct_solve(optionalTensorRef_ctype)
                return f"{argname}.has_value() ? c10::make_optional({argname}) : c10::nullopt"
            # Technically, we also need to handle cases of C++ containers holding reference types.
            # But there currently aren't any ops that require lambda capture codegen
            # With arguments like std::vector<IntArrayRef>.
            # If that changes, we'll have to add the translation here.

        # We allow const casting on tensors, since const-correctness is a bit broken for at::Tensor.
        # We could probably generalize this to non-tensor types too.
        if goal.type == MutRefCType(BaseCType(tensorT)):
            const_ref_tensor_ctype = NamedCType(
                goal.name, ConstRefCType(BaseCType(tensorT))
            )
            argname = direct_solve(const_ref_tensor_ctype)
            return f"const_cast<Tensor&>({argname})"

        unsat(goal)

    return [Expr(solve(g, direct=False), g) for g in goal_ctypes]
