// slang-check-constraint.cpp
#include "slang-check-impl.h"

// This file provides the core services for creating
// and solving constraint systems during semantic checking.
//
// We currently use constraint systems primarily to solve
// for the implied values to use for generic parameters when a
// generic declaration is being applied without explicit
// generic arguments.
//
// Conceptually, our constraint-solving strategy starts by
// trying to "unify" the actual argument types to a call
// with the parameter types of the callee (which may mention
// generic parameters). E.g., if we have a situation like:
//
//      void doIt<T>(T a, vector<T,3> b);
//
//      int x, y;
//      ...
//      doIt(x, y);
//
// then an we would try to unify the type of the argument
// `x` (which is `int`) with the type of the parameter `a`
// (which is `T`). Attempting to unify a concrete type
// and a generic type parameter would (in the simplest case)
// give rise to a constraint that, e.g., `T` must be `int`.
//
// In our example, unifying `y` and `b` creates a more complex
// scenario, because we cannot ever unify `int` with `vector<T,3>`;
// there is no possible value of `T` for which those two types
// are equivalent.
//
// So instead of the simpler approach to unification (which
// works well for languages without implicit type conversion),
// our approach to unification recognizes that scalar types
// can be promoted to vectors, and thus tries to unify the
// type of `y` with the element type of `b`.
//
// When it comes time to actually solve the constraints, we
// might have seemingly conflicting constraints:
//
//      void another<U>(U a, U b);
//
//      float x; int y;
//      another(x, y);
//
// In this case we'd have constraints that `U` must be `int`,
// *and* that `U` must be `float`, which is clearly impossible
// to satisfy. Instead, our constraints are treated as a kind
// of "lower bound" on the type variable, and we combine
// those lower bounds using the "join" operation (in the
// sense of "meet" and "join" on lattices), which ideally
// gives us a type for `U` that all the argument types can
// convert to.

namespace Slang
{
Type* SemanticsVisitor::TryJoinVectorAndScalarType(
    ConstraintSystem* constraints,
    VectorExpressionType* vectorType,
    BasicExpressionType* scalarType)
{
    // Join( vector<T,N>, S ) -> vetor<Join(T,S), N>
    //
    // That is, the join of a vector and a scalar type is
    // a vector type with a joined element type.
    auto joinElementType = TryJoinTypes(constraints, vectorType->getElementType(), scalarType);
    if (!joinElementType)
        return nullptr;

    return createVectorType(joinElementType, vectorType->getElementCount());
}

Type* SemanticsVisitor::_tryJoinTypeWithInterface(
    ConstraintSystem* constraints,
    Type* type,
    Type* interfaceType)
{
    // The most basic test here should be: does the type declare conformance to the trait.

    if (constraints->subTypeForAdditionalWitnesses == type)
    {
        // If additional subtype witnesses are provided for `type` in `constraints`,
        // try to use them to see if the interface is satisfied.
        if (constraints->additionalSubtypeWitnesses->containsKey(interfaceType))
            return type;
    }
    else
    {
        if (isSubtype(
                type,
                interfaceType,
                constraints->additionalSubtypeWitnesses ? IsSubTypeOptions::NoCaching
                                                        : IsSubTypeOptions::None))
            return type;
    }

    // Just because `type` doesn't conform to the given `interfaceDeclRef`, that
    // doesn't necessarily indicate a failure. It is possible that we have a call
    // like `sqrt(2)` so that `type` is `int` and `interfaceDeclRef` is
    // `__BuiltinFloatingPointType`. The "obvious" answer is that we should infer
    // the type `float`, but it seems like the compiler would have to synthesize
    // that answer from thin air.
    //
    // A robsut/correct solution here might be to enumerate set of types types `S`
    // such that for each type `X` in `S`:
    //
    // * `type` is implicitly convertible to `X`
    // * `X` conforms to the interface named by `interfaceDeclRef`
    //
    // If the set `S` is non-empty then we would try to pick the "best" type from `S`.
    // The "best" type would be a type `Y` such that `Y` is implicitly convertible to
    // every other type in `S`.
    //
    // We are going to implement a much simpler strategy for now, where we only apply
    // the search process if `type` is a builtin scalar type, and then we only search
    // through types `X` that are also builtin scalar types.
    //
    Type* bestType = nullptr;
    ConversionCost bestCost = kConversionCost_Explicit;
    if (auto basicType = dynamicCast<BasicExpressionType>(type))
    {
        for (Int baseTypeFlavorIndex = 0; baseTypeFlavorIndex < Int(BaseType::CountOf);
             baseTypeFlavorIndex++)
        {
            // Don't consider `type`, since we already know it doesn't work.
            if (baseTypeFlavorIndex == Int(basicType->getBaseType()))
                continue;

            // Look up the type in our session.
            auto candidateType =
                getCurrentASTBuilder()->getBuiltinType(BaseType(baseTypeFlavorIndex));
            if (!candidateType)
                continue;

            // We only want to consider types that implement the target interface.
            if (!isSubtype(candidateType, interfaceType, IsSubTypeOptions::None))
                continue;

            // We only want to consider types where we can implicitly convert from `type`
            auto conversionCost = getConversionCost(candidateType, type);
            if (!canConvertImplicitly(conversionCost))
                continue;

            // At this point, we have a candidate type that is usable.
            //
            // If this is our first viable candidate, then it is our best one:
            //
            if (!bestType)
            {
                bestType = candidateType;
            }
            else
            {
                // Otherwise, we want to pick the "better" type between `candidateType`
                // and `bestType`.
                //
                // The candidate type that has lower conversion cost from `type` is better.
                //
                if (conversionCost < bestCost)
                {
                    // Our candidate can convert to the current "best" type, so
                    // it is logically a more specific type that satisfies our
                    // constraints, therefore we should keep it.
                    //
                    bestType = candidateType;
                    bestCost = conversionCost;
                }
            }
        }
        if (bestType)
            return bestType;
    }

    // If `interfaceType` represents some generic interface type, such as `IFoo<T>`, and `type`
    // conforms to some `IFoo<X>`, then we should attempt to unify the them to discover constraints
    // for `T`.
    if (auto interfaceDeclRef = isDeclRefTypeOf<InterfaceDecl>(interfaceType))
    {
        if (as<GenericAppDeclRef>(interfaceDeclRef.declRefBase))
        {
            auto inheritanceInfo = getShared()->getInheritanceInfo(type);
            for (auto facet : inheritanceInfo.facets)
            {
                if (facet->origin.declRef.getDecl() == interfaceDeclRef.getDecl())
                {
                    auto unificationResult = TryUnifyTypes(
                        *constraints,
                        ValUnificationContext(),
                        QualType(facet->getType()),
                        interfaceType);

                    if (unificationResult)
                        return type;
                }
            }
            if (constraints->subTypeForAdditionalWitnesses)
            {
                for (auto witnessKV : *constraints->additionalSubtypeWitnesses)
                {
                    auto unificationResult = TryUnifyTypes(
                        *constraints,
                        ValUnificationContext(),
                        QualType(witnessKV.first),
                        interfaceType);
                    if (unificationResult)
                        return type;
                }
            }
        }
    }

    // For all other cases, we will just bail out for now.
    //
    // TODO: In the future we should build some kind of side data structure
    // to accelerate either one or both of these queries:
    //
    // * Given a type `T`, what types `U` can it convert to implicitly?
    //
    // * Given an interface `I`, what types `U` conform to it?
    //
    // The intersection of the sets returned by these two queries is
    // the set of candidates we would like to consider here.

    return nullptr;
}

Type* SemanticsVisitor::TryJoinTypes(ConstraintSystem* constraints, QualType left, QualType right)
{
    // Easy case: they are the same type!
    if (left->equals(right))
        return left;

    // We can join two basic types by picking the "better" of the two
    if (auto leftBasic = as<BasicExpressionType>(left))
    {
        if (auto rightBasic = as<BasicExpressionType>(right))
        {
            auto costConvertRightToLeft = getConversionCost(leftBasic, right);
            auto costConvertLeftToRight = getConversionCost(rightBasic, left);

            // Return the one that had lower conversion cost.
            if (costConvertRightToLeft > costConvertLeftToRight)
                return right;
            else
            {
                return left;
            }
        }

        // We can also join a vector and a scalar
        if (auto rightVector = as<VectorExpressionType>(right))
        {
            return TryJoinVectorAndScalarType(constraints, rightVector, leftBasic);
        }
    }

    // We can join two vector types by joining their element types
    // (and also their sizes...)
    if (auto leftVector = as<VectorExpressionType>(left))
    {
        if (auto rightVector = as<VectorExpressionType>(right))
        {
            // Check if the vector sizes match
            if (!leftVector->getElementCount()->equals(rightVector->getElementCount()))
                return nullptr;

            // Try to join the element types
            auto joinElementType = TryJoinTypes(
                constraints,
                QualType(leftVector->getElementType(), left.isLeftValue),
                QualType(rightVector->getElementType(), right.isLeftValue));
            if (!joinElementType)
                return nullptr;

            return createVectorType(joinElementType, leftVector->getElementCount());
        }

        // We can also join a vector and a scalar
        if (auto rightBasic = as<BasicExpressionType>(right))
        {
            return TryJoinVectorAndScalarType(constraints, leftVector, rightBasic);
        }
    }

    // HACK: trying to work trait types in here...
    if (auto leftDeclRefType = as<DeclRefType>(left))
    {
        if (auto leftInterfaceRef = leftDeclRefType->getDeclRef().as<InterfaceDecl>())
        {
            //
            return _tryJoinTypeWithInterface(constraints, right, left);
        }
    }
    if (auto rightDeclRefType = as<DeclRefType>(right))
    {
        if (auto rightInterfaceRef = rightDeclRefType->getDeclRef().as<InterfaceDecl>())
        {
            //
            return _tryJoinTypeWithInterface(constraints, left, right);
        }
    }

    // We can recursively join two TypePacks.
    if (auto leftTypePack = as<ConcreteTypePack>(left))
    {
        if (auto rightTypePack = as<ConcreteTypePack>(right))
        {
            if (leftTypePack->getTypeCount() != rightTypePack->getTypeCount())
                return nullptr;
            ShortList<Type*> joinedTypes;
            for (Index i = 0; i < leftTypePack->getTypeCount(); ++i)
            {
                auto joinedType = TryJoinTypes(
                    constraints,
                    QualType(leftTypePack->getElementType(i), left.isLeftValue),
                    QualType(rightTypePack->getElementType(i), right.isLeftValue));
                if (!joinedType)
                    return nullptr;
                joinedTypes.add(joinedType);
            }
            return m_astBuilder->getTypePack(joinedTypes.getArrayView().arrayView);
        }
    }

    // TODO: all the cases for vectors apply to matrices too!

    // Default case is that we just fail.
    return nullptr;
}

DeclRef<Decl> SemanticsVisitor::trySolveConstraintSystem(
    ConstraintSystem* system,
    DeclRef<GenericDecl> genericDeclRef,
    ArrayView<Val*> knownGenericArgs,
    ConversionCost& outBaseCost)
{
    ensureDecl(genericDeclRef.getDecl(), DeclCheckState::ReadyForLookup);

    outBaseCost = kConversionCost_None;

    // For now the "solver" is going to be ridiculously simplistic.

    // The generic itself will have some constraints, and for now we add these
    // to the system of constrains we will use for solving for the type variables.
    //
    // TODO: we need to decide whether constraints are used like this to influence
    // how we solve for type/value variables, or whether constraints in the parameter
    // list just work as a validation step *after* we've solved for the types.
    //
    // That is, should we allow `<T : Int>` to be written, and cause us to "infer"
    // that `T` should be the type `Int`? That seems a little silly.
    //
    // Eventually, though, we may want to support type identity constraints, especially
    // on associated types, like `<C where C : IContainer && C.IndexType == Int>`
    // These seem more reasonable to have influence constraint solving, since it could
    // conceivably let us specialize a `X<T> : IContainer` to `X<Int>` if we find
    // that `X<T>.IndexType == T`.
    for (auto constraintDeclRef :
         getMembersOfType<GenericTypeConstraintDecl>(m_astBuilder, genericDeclRef))
    {
        if (!TryUnifyTypes(
                *system,
                ValUnificationContext(),
                getSub(m_astBuilder, constraintDeclRef),
                getSup(m_astBuilder, constraintDeclRef)))
            return DeclRef<Decl>();
    }

    // Once have built up the initial list of constraints we are trying to satisfy,
    // we will attempt to solve for each parameter in a way that satisfies all
    // the constraints that apply to that parameter.
    //
    // Note: this is a very limited kind of solver, in that it doesn't have a
    // way to make use of constraints between two or more parameters.
    //
    // As we go, we will build up a list of argument values for a possible
    // solution for how to assign the parameters in a way that satisfies all
    // the constraints.
    //
    ShortList<Val*> args;

    // If the context is such that some of the arguments are already specified
    // or known, we need to go ahead and use those arguments direclty (whether
    // or not they are compatible with the constraints).
    //
    Count knownGenericArgCount = 0;
    if (knownGenericArgs.getCount())
    {
        knownGenericArgCount = knownGenericArgs.getCount();
        for (auto arg : knownGenericArgs)
        {
            args.add(arg);
        }
    }

    // The state of currently solved arguments.
    struct SolvedArg
    {
        IntVal* val = nullptr;
        bool isOptional = true;
        ShortList<QualType, 8> types;
    };
    ShortList<SolvedArg> solvedArgs;

    // We will then iterate over the constraints trying to solve all generic parameters.
    // Note that we do not use ranged for here, because processing one constraint may lead to
    // new constraints being discovered.
    for (Index constraintIndex = 0; constraintIndex < system->constraints.getCount();
         constraintIndex++)
    {
        // Note: it is important to keep a copy of the constraint here instead of
        // using a reference, because the constraint list may be modified during the
        // loop as we discover new constraints.
        //
        auto c = system->constraints[constraintIndex];
        if (auto typeParam = as<GenericTypeParamDeclBase>(c.decl))
        {
            SLANG_ASSERT(typeParam->parameterIndex != -1);
            // If the parameter is one where we already know
            // the argument value to use, we don't bother with
            // trying to solve for it, and treat any constraints
            // on such a parameter as implicitly solved-for.
            //
            if (typeParam->parameterIndex < knownGenericArgCount)
            {
                system->constraints[constraintIndex].satisfied = true;
                continue;
            }

            // If the parameter is a type pack, then we may have
            // constraints that apply to invidual elements of the pack.
            // We will need to handle the type pack case slightly differently.
            //
            bool isPack = as<GenericTypePackParamDecl>(typeParam) != nullptr;

            // We will use a temporary list to hold the resolved types
            // for this generic parameter.
            // For normal type parameters, there should be only one type
            // in the list. For type pack parameters, there can be one type
            // for each element in the pack.
            //
            if (solvedArgs.getCount() <= typeParam->parameterIndex)
            {
                solvedArgs.setCount(typeParam->parameterIndex + 1);
            }
            auto& types = solvedArgs[typeParam->parameterIndex].types;
            if (!isPack)
                types.setCount(1);

            bool& typeConstraintOptional = solvedArgs[typeParam->parameterIndex].isOptional;

            QualType* ptype = nullptr;
            if (isPack)
            {
                types.setCount(Math::Max(types.getCount(), c.indexInPack + 1));
                ptype = &types[c.indexInPack];
            }
            else
                ptype = &types[0];
            QualType& type = *ptype;

            auto cType = QualType(as<Type>(c.val), c.isUsedAsLValue);
            SLANG_RELEASE_ASSERT(cType);

            if (!type || (typeConstraintOptional && !c.isOptional))
            {
                type = cType;
                typeConstraintOptional = c.isOptional;
            }
            else if (!typeConstraintOptional)
            {
                // If the type parameter is already constrained to a known type,
                // we need to make sure our resolved type can satisfy both constraints.
                // We do so by updating the resolved type to be the "join" of the current
                // solution and the type in the new constraint. If such join cannot be found,
                // it means it is not possible to have a compatible solution that meets all
                // constraints and we should fail.
                //
                // Another detail here is that during type joining, we may discover
                // new constraints from the base types of the types being joined.
                // We will pass the constraint system to `TryJoinTypes` which can
                // add new constraints to the system, and we will process the new constraints
                // in the next iteration.
                //
                auto joinType = TryJoinTypes(system, type, cType);
                if (!joinType)
                {
                    // failure!
                    return DeclRef<Decl>();
                }
                type = QualType(joinType, type.isLeftValue || cType.isLeftValue);
            }

            c.satisfied = true;
        }
        else if (auto valParam = as<GenericValueParamDecl>(c.decl))
        {
            SLANG_ASSERT(valParam->parameterIndex != -1);

            // If the parameter is one where we already know
            // the argument value to use, we don't bother with
            // trying to solve for it, and treat any constraints
            // on such a parameter as implicitly solved-for.
            //
            if (valParam->parameterIndex < knownGenericArgCount)
            {
                system->constraints[constraintIndex].satisfied = true;
                continue;
            }

            if (solvedArgs.getCount() <= valParam->parameterIndex)
                solvedArgs.setCount(valParam->parameterIndex + 1);
            IntVal*& val = solvedArgs[valParam->parameterIndex].val;
            bool& valOptional = solvedArgs[valParam->parameterIndex].isOptional;

            auto cVal = as<IntVal>(c.val);
            SLANG_RELEASE_ASSERT(cVal);

            if (!val || (valOptional && !c.isOptional))
            {
                val = cVal;
                valOptional = c.isOptional;
            }
            else
            {
                if (!valOptional && !val->equals(cVal))
                {
                    // failure!
                    return DeclRef<Decl>();
                }
            }

            c.satisfied = true;
        }
        system->constraints[constraintIndex].satisfied = c.satisfied;
    }

    // After we processed all constraints, `solvedTypes` and `solvedVals`
    // should have been filled with the resolved types and values for the
    // generic parameters. We can now verify if they are complete and consolidate
    // them into final argument list.
    for (auto member : genericDeclRef.getDecl()->members)
    {
        if (auto typeParam = as<GenericTypeParamDeclBase>(member))
        {
            SLANG_ASSERT(typeParam->parameterIndex != -1);

            if (typeParam->parameterIndex < knownGenericArgCount)
                continue;
            bool isPack = as<GenericTypePackParamDecl>(typeParam) != nullptr;
            if (typeParam->parameterIndex >= solvedArgs.getCount())
            {
                // If the parameter is not a type pack and we don't have a
                // resolved type for it, we should fail.
                if (!isPack)
                    return DeclRef<Decl>();
                // If the parameter is a type pack, we should add an empty
                // type list to solvedTypes.
                solvedArgs.setCount(typeParam->parameterIndex + 1);
            }
            auto& types = solvedArgs[typeParam->parameterIndex].types;
            // Fail if any of the resolved type element is empty.
            for (auto t : types)
            {
                if (!t)
                    return DeclRef<Decl>();
            }
            if (!isPack)
            {
                // If the generic parameter is not a pack, we can simply add the first type.
                if (types.getCount() != 1)
                    return DeclRef<Decl>();

                args.add(types[0]);
            }
            else
            {
                // If the generic parameter is a pack, and we are supplying one single pack
                // argument, we can use it as is.
                if (types.getCount() == 1 && isTypePack(types[0]))
                {
                    args.add(types[0]);
                }
                else
                {
                    // If we are supplying 0 or multiple arguments for the pack, we need to create a
                    // type pack and add it to the argument list.
                    ShortList<Type*> typeList;
                    bool isLVal = true;
                    for (auto t : types)
                    {
                        typeList.add(t);
                        isLVal = isLVal && t.isLeftValue;
                    }
                    args.add(QualType(
                        m_astBuilder->getTypePack(typeList.getArrayView().arrayView),
                        isLVal));
                }
            }
        }
        else if (auto valParam = as<GenericValueParamDecl>(member))
        {
            SLANG_ASSERT(valParam->parameterIndex != -1);

            if (valParam->parameterIndex < knownGenericArgCount)
                continue;

            if (valParam->parameterIndex >= solvedArgs.getCount())
                return DeclRef<Decl>();

            auto val = solvedArgs[valParam->parameterIndex].val;
            if (!val)
            {
                // failure!
                return DeclRef<Decl>();
            }
            args.add(val);
        }
    }

    // After we've solved for the explicit arguments, we need to
    // make a second pass and consider the implicit arguments,
    // based on what we've already determined to be the values
    // for the explicit arguments.

    // Before we begin, we are going to go ahead and create the
    // "solved" substitution that we will return if everything works.
    // This is because we are going to use this substitution,
    // partially filled in with the results we know so far,
    // in order to specialize any constraints on the generic.
    //
    // E.g., if the generic parameters were `<T : ISidekick>`, and
    // we've already decided that `T` is `Robin`, then we want to
    // search for a conformance `Robin : ISidekick`, which involved
    // apply the substitutions we already know...

    HashSet<Decl*> constrainedGenericParams;

    for (auto constraintDecl :
         genericDeclRef.getDecl()->getMembersOfType<GenericTypeConstraintDecl>())
    {
        DeclRef<GenericTypeConstraintDecl> constraintDeclRef =
            m_astBuilder
                ->getGenericAppDeclRef(
                    genericDeclRef,
                    args.getArrayView().arrayView,
                    constraintDecl)
                .as<GenericTypeConstraintDecl>();

        // Extract the (substituted) sub- and super-type from the constraint.
        auto sub = getSub(m_astBuilder, constraintDeclRef);
        auto sup = getSup(m_astBuilder, constraintDeclRef);

        // Mark sub type as constrained.
        if (auto subDeclRefType = as<DeclRefType>(constraintDeclRef.getDecl()->sub.type))
            constrainedGenericParams.add(subDeclRefType->getDeclRef().getDecl());
        else if (auto subEachType = as<EachType>(constraintDeclRef.getDecl()->sub.type))
            constrainedGenericParams.add(
                as<DeclRefType>(subEachType->getElementType())->getDeclRef().getDecl());

        if (sub->equals(sup) && isDeclRefTypeOf<InterfaceDecl>(sup))
        {
            // We are trying to use an interface type itself to conform to the
            // type constraint. We can reach this case when the user code does
            // not provide an explicit type parameter to specialize a generic
            // and the type parameter cannot be inferred from any arguments.
            // In this case, we should fail the constraint check.
            return DeclRef<Decl>();
        }

        // Search for a witness that shows the constraint is satisfied.
        SubtypeWitness* subTypeWitness = nullptr;
        if (sub == system->subTypeForAdditionalWitnesses)
        {
            // If we are trying to find the subtype info for a type whose inheritance info is
            // being calculated, use what we have already known about the type.
            system->additionalSubtypeWitnesses->tryGetValue(sup, subTypeWitness);
        }
        else
        {
            // The general case is to initiate a subtype query.
            subTypeWitness = isSubtype(
                sub,
                sup,
                system->additionalSubtypeWitnesses ? IsSubTypeOptions::NoCaching
                                                   : IsSubTypeOptions::None);
        }

        if (constraintDecl->isEqualityConstraint)
        {
            // If constraint is an equality constraint, we need to make sure
            // the witness is equality witness.
            if (!isTypeEqualityWitness(subTypeWitness))
                subTypeWitness = nullptr;
        }

        if (subTypeWitness)
        {
            // We found a witness, so it will become an (implicit) argument.
            args.add(subTypeWitness);
            outBaseCost += subTypeWitness->getOverloadResolutionCost();
        }
        else
        {
            // No witness was found, so the inference will now fail.
            //
            // TODO: Ideally we should print an error message in
            // this case, to let the user know why things failed.
            return DeclRef<Decl>();
        }

        // TODO: We may need to mark some constrains in our constraint
        // system as being solved now, as a result of the witness we found.
    }

    // Make sure we haven't constructed any spurious constraints
    // that we aren't able to satisfy:
    for (auto c : system->constraints)
    {
        if (!c.satisfied)
        {
            return DeclRef<Decl>();
        }
    }

    // Verify that all type coercion constraints can be satisfied.
    for (auto constraintDecl :
         genericDeclRef.getDecl()->getMembersOfType<TypeCoercionConstraintDecl>())
    {
        DeclRef<TypeCoercionConstraintDecl> constraintDeclRef =
            m_astBuilder
                ->getGenericAppDeclRef(
                    genericDeclRef,
                    args.getArrayView().arrayView,
                    constraintDecl)
                .as<TypeCoercionConstraintDecl>();
        auto fromType = constraintDeclRef.substitute(m_astBuilder, constraintDecl->fromType.Ptr());
        auto toType = constraintDeclRef.substitute(m_astBuilder, constraintDecl->toType.Ptr());
        auto conversionCost = getConversionCost(toType, fromType);
        if (constraintDecl->findModifier<ImplicitConversionModifier>())
        {
            if (conversionCost > kConversionCost_GeneralConversion)
            {
                // The type arguments are not implicitly convertible, return failure.
                return DeclRef<Decl>();
            }
        }
        else
        {
            if (conversionCost == kConversionCost_Impossible)
            {
                // The type arguments are not convertible, return failure.
                return DeclRef<Decl>();
            }
        }
        if (auto fromDecl = isDeclRefTypeOf<Decl>(constraintDecl->fromType))
        {
            constrainedGenericParams.add(fromDecl.getDecl());
        }
        if (auto toDecl = isDeclRefTypeOf<Decl>(constraintDecl->toType))
        {
            constrainedGenericParams.add(toDecl.getDecl());
        }
        // If we are to expand the support of type coercion constraint beyond simple builtin core
        // module functions, then the witness should be a reference to the conversion function. For
        // now, this isn't required, and it is not easy to get it from the coercion logic, so we
        // leave it empty.
        args.add(m_astBuilder->getTypeCoercionWitness(fromType, toType, DeclRef<Decl>()));
    }

    // Add a flat cost to all unconstrained generic params.
    for (auto typeParamDecl : genericDeclRef.getDecl()->getMembersOfType<GenericTypeParamDecl>())
    {
        if (!constrainedGenericParams.contains(typeParamDecl))
            outBaseCost += kConversionCost_UnconstraintGenericParam;
    }

    return m_astBuilder->getGenericAppDeclRef(genericDeclRef, args.getArrayView().arrayView);
}

bool SemanticsVisitor::TryUnifyVals(
    ConstraintSystem& constraints,
    ValUnificationContext unifyCtx,
    Val* fst,
    bool fstLVal,
    Val* snd,
    bool sndLVal)
{
    // if both values are types, then unify types
    if (auto fstType = as<Type>(fst))
    {
        if (auto sndType = as<Type>(snd))
        {
            return TryUnifyTypes(
                constraints,
                unifyCtx,
                QualType(fstType, fstLVal),
                QualType(sndType, sndLVal));
        }
    }

    // if both values are constant integers, then compare them
    if (auto fstIntVal = as<ConstantIntVal>(fst))
    {
        if (auto sndIntVal = as<ConstantIntVal>(snd))
        {
            return fstIntVal->getValue() == sndIntVal->getValue();
        }
    }

    // Check if both are integer values in general
    const auto fstInt = as<IntVal>(fst);
    const auto sndInt = as<IntVal>(snd);
    if (fstInt && sndInt)
    {
        const auto paramUnderCast = [](IntVal* i)
        {
            if (const auto c = as<TypeCastIntVal>(i))
                i = as<IntVal>(c->getBase());
            return as<GenericParamIntVal>(i);
        };
        auto fstParam = paramUnderCast(fstInt);
        auto sndParam = paramUnderCast(sndInt);

        bool okay = false;
        if (fstParam)
            okay |= TryUnifyIntParam(constraints, unifyCtx, fstParam->getDeclRef(), sndInt);
        if (sndParam)
            okay |= TryUnifyIntParam(constraints, unifyCtx, sndParam->getDeclRef(), fstInt);
        return okay;
    }

    if (auto fstWit = as<DeclaredSubtypeWitness>(fst))
    {
        if (auto sndWit = as<DeclaredSubtypeWitness>(snd))
        {
            auto constraintDecl1 = fstWit->getDeclRef().as<TypeConstraintDecl>();
            auto constraintDecl2 = sndWit->getDeclRef().as<TypeConstraintDecl>();
            SLANG_ASSERT(constraintDecl1);
            SLANG_ASSERT(constraintDecl2);
            return TryUnifyTypes(
                constraints,
                unifyCtx,
                getSup(m_astBuilder, constraintDecl1),
                getSup(m_astBuilder, constraintDecl2));
        }
    }

    // Two subtype witnesses can be unified if they exist (non-null) and
    // prove that some pair of types are subtypes of types that can be unified.
    //
    if (auto fstWit = as<SubtypeWitness>(fst))
    {
        if (auto sndWit = as<SubtypeWitness>(snd))
        {
            return TryUnifyTypes(constraints, unifyCtx, fstWit->getSup(), sndWit->getSup());
        }
    }

    SLANG_UNIMPLEMENTED_X("value unification case");

    // default: fail
    // return false;
}

bool SemanticsVisitor::tryUnifyDeclRef(
    ConstraintSystem& constraints,
    ValUnificationContext unifyCtx,
    DeclRefBase* fst,
    bool fstIsLVal,
    DeclRefBase* snd,
    bool sndIsLVal)
{
    if (fst == snd)
        return true;
    if (fst == nullptr || snd == nullptr)
        return false;
    auto fstGen = SubstitutionSet(fst).findGenericAppDeclRef();
    auto sndGen = SubstitutionSet(snd).findGenericAppDeclRef();
    if (fstGen == sndGen)
        return true;
    if (fstGen == nullptr || sndGen == nullptr)
        return false;
    return tryUnifyGenericAppDeclRef(constraints, unifyCtx, fstGen, fstIsLVal, sndGen, sndIsLVal);
}

bool SemanticsVisitor::tryUnifyGenericAppDeclRef(
    ConstraintSystem& constraints,
    ValUnificationContext unifyCtx,
    GenericAppDeclRef* fst,
    bool fstIsLVal,
    GenericAppDeclRef* snd,
    bool sndIsLVal)
{
    SLANG_ASSERT(fst);
    SLANG_ASSERT(snd);

    auto fstGen = fst;
    auto sndGen = snd;
    // They must be specializing the same generic
    if (fstGen->getGenericDecl() != sndGen->getGenericDecl())
        return false;

    // Their arguments must unify
    SLANG_RELEASE_ASSERT(fstGen->getArgs().getCount() == sndGen->getArgs().getCount());
    Index argCount = fstGen->getArgs().getCount();
    bool okay = true;
    for (Index aa = 0; aa < argCount; ++aa)
    {
        if (!TryUnifyVals(
                constraints,
                unifyCtx,
                fstGen->getArgs()[aa],
                fstIsLVal,
                sndGen->getArgs()[aa],
                sndIsLVal))
        {
            okay = false;
        }
    }

    // Their "base" specializations must unify
    auto fstBase = fst->getBase();
    auto sndBase = snd->getBase();

    if (!tryUnifyDeclRef(constraints, unifyCtx, fstBase, fstIsLVal, sndBase, sndIsLVal))
    {
        okay = false;
    }

    return okay;
}

bool SemanticsVisitor::TryUnifyTypeParam(
    ConstraintSystem& constraints,
    ValUnificationContext unificationContext,
    GenericTypeParamDeclBase* typeParamDecl,
    QualType type)
{
    // We want to constrain the given type parameter
    // to equal the given type.
    Constraint constraint;
    constraint.decl = typeParamDecl;
    constraint.indexInPack = unificationContext.indexInTypePack;
    constraint.val = type;
    constraint.isUsedAsLValue = type.isLeftValue;
    constraints.constraints.add(constraint);

    return true;
}

bool SemanticsVisitor::TryUnifyIntParam(
    ConstraintSystem& constraints,
    ValUnificationContext unifyCtx,
    GenericValueParamDecl* paramDecl,
    IntVal* val)
{
    SLANG_UNUSED(unifyCtx);

    // We only want to accumulate constraints on
    // the parameters of the declarations being
    // specialized (don't accidentially constrain
    // parameters of a generic function based on
    // calls in its body).
    if (paramDecl->parentDecl != constraints.genericDecl)
        return false;

    // We want to constrain the given parameter to equal the given value.
    Constraint constraint;
    constraint.decl = paramDecl;
    // If `val` is of different type than `paramDecl`, we want to insert a type cast.
    if (val->getType() != paramDecl->getType())
    {
        auto cast = m_astBuilder->getTypeCastIntVal(paramDecl->getType(), val);
        val = cast;
    }
    constraint.val = val;

    constraints.constraints.add(constraint);

    return true;
}

bool SemanticsVisitor::TryUnifyIntParam(
    ConstraintSystem& constraints,
    ValUnificationContext unifyCtx,
    DeclRef<VarDeclBase> const& varRef,
    IntVal* val)
{
    if (auto genericValueParamRef = varRef.as<GenericValueParamDecl>())
    {
        return TryUnifyIntParam(constraints, unifyCtx, genericValueParamRef.getDecl(), val);
    }
    else
    {
        return false;
    }
}

bool SemanticsVisitor::TryUnifyTypesByStructuralMatch(
    ConstraintSystem& constraints,
    ValUnificationContext unifyCtx,
    QualType fst,
    QualType snd)
{
    if (auto fstDeclRefType = as<DeclRefType>(fst))
    {
        auto fstDeclRef = fstDeclRefType->getDeclRef();

        if (auto typeParamDecl = as<GenericTypeParamDecl>(fstDeclRef.getDecl()))
            if (typeParamDecl->parentDecl == constraints.genericDecl)
                return TryUnifyTypeParam(constraints, unifyCtx, typeParamDecl, snd);

        if (auto sndDeclRefType = as<DeclRefType>(snd))
        {
            auto sndDeclRef = sndDeclRefType->getDeclRef();

            if (auto typeParamDecl = as<GenericTypeParamDecl>(sndDeclRef.getDecl()))
                if (typeParamDecl->parentDecl == constraints.genericDecl)
                    return TryUnifyTypeParam(constraints, unifyCtx, typeParamDecl, fst);

            // If they refer to different declarations, we need to check if one type's super type
            // matches the other type, if so we can unify them.
            if (fstDeclRef.getDecl() != sndDeclRef.getDecl())
            {
                {
                    auto fstTypeInheritanceInfo = getShared()->getInheritanceInfo(fstDeclRefType);
                    for (auto supType : fstTypeInheritanceInfo.facets)
                    {
                        if (supType->origin.declRef.getDecl() == sndDeclRef.getDecl())
                        {
                            fstDeclRef = supType->origin.declRef;
                            goto endMatch;
                        }
                    }
                }
                // try the other direction
                {
                    auto sndTypeInheritanceInfo = getShared()->getInheritanceInfo(sndDeclRefType);
                    for (auto supType : sndTypeInheritanceInfo.facets)
                    {
                        if (supType->origin.declRef.getDecl() == fstDeclRef.getDecl())
                        {
                            sndDeclRef = supType->origin.declRef;
                            goto endMatch;
                        }
                    }
                }
            endMatch:;
                // If they still refer to different decls, then we can't unify them.
                if (fstDeclRef.getDecl() != sndDeclRef.getDecl())
                    return false;
            }

            // next we need to unify the substitutions applied
            // to each declaration reference.
            if (!tryUnifyDeclRef(
                    constraints,
                    unifyCtx,
                    fstDeclRef,
                    fst.isLeftValue,
                    sndDeclRef,
                    snd.isLeftValue))
            {
                return false;
            }

            return true;
        }
    }
    else if (auto fstFunType = as<FuncType>(fst))
    {
        if (auto sndFunType = as<FuncType>(snd))
        {
            const Index numParams = fstFunType->getParamCount();
            if (numParams != sndFunType->getParamCount())
                return false;
            for (Index i = 0; i < numParams; ++i)
            {
                if (!TryUnifyTypes(
                        constraints,
                        unifyCtx,
                        fstFunType->getParamType(i),
                        sndFunType->getParamType(i)))
                    return false;
            }
            return TryUnifyTypes(
                constraints,
                unifyCtx,
                fstFunType->getResultType(),
                sndFunType->getResultType());
        }
    }
    else if (auto expandType = as<ExpandType>(fst))
    {
        if (auto sndExpandType = as<ExpandType>(snd))
        {
            return TryUnifyTypes(
                constraints,
                unifyCtx,
                expandType->getPatternType(),
                sndExpandType->getPatternType());
        }
    }
    else if (auto eachType = as<EachType>(fst))
    {
        if (auto sndEachType = as<EachType>(snd))
        {
            return TryUnifyTypes(
                constraints,
                unifyCtx,
                eachType->getElementType(),
                sndEachType->getElementType());
        }
    }
    else if (auto typePack = as<ConcreteTypePack>(fst))
    {
        if (auto sndTypePack = as<ConcreteTypePack>(snd))
        {
            if (typePack->getTypeCount() != sndTypePack->getTypeCount())
                return false;
            for (Index i = 0; i < typePack->getTypeCount(); ++i)
            {
                if (!TryUnifyTypes(
                        constraints,
                        unifyCtx,
                        QualType(typePack->getElementType(i), fst.isLeftValue),
                        QualType(sndTypePack->getElementType(i), snd.isLeftValue)))
                    return false;
            }
            return true;
        }
    }
    return false;
}

bool SemanticsVisitor::TryUnifyConjunctionType(
    ConstraintSystem& constraints,
    ValUnificationContext unifyCtx,
    QualType fst,
    QualType snd)
{
    // Unifying a type `A & B` with `T` amounts to unifying
    // `A` with `T` and also `B` with `T` while
    // unifying a type `T` with `A & B` amounts to either
    // unifying `T` with `A` or `T` with `B`
    //
    // If either unification is impossible, then the full
    // case is also impossible.
    //
    if (auto fstAndType = as<AndType>(fst))
    {
        return TryUnifyTypes(
                   constraints,
                   unifyCtx,
                   QualType(fstAndType->getLeft(), fst.isLeftValue),
                   snd) &&
               TryUnifyTypes(
                   constraints,
                   unifyCtx,
                   QualType(fstAndType->getRight(), fst.isLeftValue),
                   snd);
    }
    else if (auto sndAndType = as<AndType>(snd))
    {
        return TryUnifyTypes(
                   constraints,
                   unifyCtx,
                   fst,
                   QualType(sndAndType->getLeft(), snd.isLeftValue)) ||
               TryUnifyTypes(
                   constraints,
                   unifyCtx,
                   fst,
                   QualType(sndAndType->getRight(), snd.isLeftValue));
    }
    else
        return false;
}

void SemanticsVisitor::maybeUnifyUnconstraintIntParam(
    ConstraintSystem& constraints,
    ValUnificationContext unifyCtx,
    IntVal* param,
    IntVal* arg,
    bool paramIsLVal)
{
    SLANG_UNUSED(unifyCtx);

    // If `param` is an unconstrained integer val param, and `arg` is a const int val,
    // we add a constraint to the system that `param` must be equal to `arg`.
    // If `param` is already constrained, ignore and do nothing.
    if (auto typeCastParam = as<TypeCastIntVal>(param))
    {
        param = as<IntVal>(typeCastParam->getBase());
    }
    auto intParam = as<GenericParamIntVal>(param);
    if (!intParam)
        return;
    for (auto c : constraints.constraints)
        if (c.decl == intParam->getDeclRef().getDecl())
            return;
    Constraint c;
    c.decl = intParam->getDeclRef().getDecl();
    c.isUsedAsLValue = paramIsLVal;
    c.val = arg;
    c.isOptional = true;
    constraints.constraints.add(c);
}

bool SemanticsVisitor::TryUnifyTypes(
    ConstraintSystem& constraints,
    ValUnificationContext unifyCtx,
    QualType fst,
    QualType snd)
{
    if (!fst)
        return false;

    if (fst->equals(snd))
        return true;

    // An error type can unify with anything, just so we avoid cascading errors.

    if (const auto fstErrorType = as<ErrorType>(fst))
        return true;

    if (const auto sndErrorType = as<ErrorType>(snd))
        return true;

    // If one or the other of the types is a conjunction `X & Y`,
    // then we want to recurse on both `X` and `Y`.
    //
    // Note that we check this case *before* we check if one of
    // the types is a generic parameter below, so that we should
    // never end up trying to match up a type parameter with
    // a conjunction directly, and will instead find all of the
    // "leaf" types we need to constrain it to.
    //
    if (as<AndType>(fst) || as<AndType>(snd))
    {
        return TryUnifyConjunctionType(constraints, unifyCtx, fst, snd);
    }

    // If one of the types is a type pack, we need to recursively unify the element types.
    if (auto fstTypePack = as<ConcreteTypePack>(fst))
    {
        if (auto sndTypePack = as<ConcreteTypePack>(snd))
        {
            if (fstTypePack->getTypeCount() != sndTypePack->getTypeCount())
                return false;
            for (Index i = 0; i < fstTypePack->getTypeCount(); ++i)
            {
                if (!TryUnifyTypes(
                        constraints,
                        unifyCtx,
                        QualType(fstTypePack->getElementType(i), fst.isLeftValue),
                        QualType(sndTypePack->getElementType(i), snd.isLeftValue)))
                    return false;
            }
            return true;
        }
        else if (auto sndExpandType = as<ExpandType>(snd))
        {
            for (Index i = 0; i < fstTypePack->getTypeCount(); ++i)
            {
                ValUnificationContext subUnifyCtx = unifyCtx;
                subUnifyCtx.indexInTypePack = i;
                if (!TryUnifyTypes(
                        constraints,
                        subUnifyCtx,
                        QualType(fstTypePack->getElementType(i), fst.isLeftValue),
                        QualType(sndExpandType->getPatternType(), snd.isLeftValue)))
                    return false;
            }
            return true;
        }
    }

    if (auto sndTypePack = as<ConcreteTypePack>(snd))
    {
        if (auto fstExpandType = as<ExpandType>(fst))
        {
            for (Index i = 0; i < sndTypePack->getTypeCount(); ++i)
            {
                ValUnificationContext subUnifyCtx = unifyCtx;
                subUnifyCtx.indexInTypePack = i;
                if (!TryUnifyTypes(
                        constraints,
                        subUnifyCtx,
                        QualType(fstExpandType->getPatternType(), fst.isLeftValue),
                        QualType(sndTypePack->getElementType(i), snd.isLeftValue)))
                    return false;
            }
            return true;
        }
    }

    // A generic parameter type can unify with anything.
    // TODO: there actually needs to be some kind of "occurs check" sort
    // of thing here...

    if (auto fstDeclRefType = as<DeclRefType>(fst))
    {
        auto fstDeclRef = fstDeclRefType->getDeclRef();

        if (auto typeParamDecl = as<GenericTypeParamDecl>(fstDeclRef.getDecl()))
        {
            if (typeParamDecl->parentDecl == constraints.genericDecl)
                return TryUnifyTypeParam(constraints, unifyCtx, typeParamDecl, snd);
        }
        else if (auto typePackParamDecl = as<GenericTypePackParamDecl>(fstDeclRef.getDecl()))
        {
            if (typePackParamDecl->parentDecl == constraints.genericDecl && isTypePack(snd))
                return TryUnifyTypeParam(constraints, unifyCtx, typePackParamDecl, snd);
        }
    }

    if (auto sndDeclRefType = as<DeclRefType>(snd))
    {
        auto sndDeclRef = sndDeclRefType->getDeclRef();

        if (auto typeParamDecl = as<GenericTypeParamDeclBase>(sndDeclRef.getDecl()))
        {
            if (typeParamDecl->parentDecl == constraints.genericDecl)
                return TryUnifyTypeParam(constraints, unifyCtx, typeParamDecl, fst);
        }
        else if (auto typePackParamDecl = as<GenericTypePackParamDecl>(sndDeclRef.getDecl()))
        {
            if (typePackParamDecl->parentDecl == constraints.genericDecl && isTypePack(fst))
                return TryUnifyTypeParam(constraints, unifyCtx, typePackParamDecl, fst);
        }
    }

    // If we can unify the types structurally, then we are golden
    if (TryUnifyTypesByStructuralMatch(constraints, unifyCtx, fst, snd))
        return true;

    // Now we need to consider cases where coercion might
    // need to be applied. For now we can try to do this
    // in a completely ad hoc fashion, but eventually we'd
    // want to do it more formally.

    if (auto fstVectorType = as<VectorExpressionType>(fst))
    {
        if (auto sndScalarType = as<BasicExpressionType>(snd))
        {
            // Try unify the vector count param. In case the vector count is defined by a generic
            // value parameter, we want to be able to infer that parameter should be 1. However, we
            // don't want a failed unification to fail the entire generic argument inference,
            // because a scalar can still be casted into a vector of any length.

            maybeUnifyUnconstraintIntParam(
                constraints,
                unifyCtx,
                fstVectorType->getElementCount(),
                m_astBuilder->getIntVal(m_astBuilder->getIntType(), 1),
                fst.isLeftValue);
            return TryUnifyTypes(
                constraints,
                unifyCtx,
                QualType(fstVectorType->getElementType(), fst.isLeftValue),
                QualType(sndScalarType, snd.isLeftValue));
        }
    }

    if (auto fstScalarType = as<BasicExpressionType>(fst))
    {
        if (auto sndVectorType = as<VectorExpressionType>(snd))
        {
            maybeUnifyUnconstraintIntParam(
                constraints,
                unifyCtx,
                sndVectorType->getElementCount(),
                m_astBuilder->getIntVal(m_astBuilder->getIntType(), 1),
                snd.isLeftValue);
            return TryUnifyTypes(
                constraints,
                unifyCtx,
                QualType(fstScalarType, fst.isLeftValue),
                QualType(sndVectorType->getElementType(), snd.isLeftValue));
        }
    }

    if (auto fstUniformParamGroupType = as<UniformParameterGroupType>(fst))
        return TryUnifyTypes(
            constraints,
            unifyCtx,
            QualType(fstUniformParamGroupType->getElementType(), fst.isLeftValue),
            snd);
    if (auto sndUniformParamGroupType = as<UniformParameterGroupType>(snd))
        return TryUnifyTypes(
            constraints,
            unifyCtx,
            fst,
            QualType(sndUniformParamGroupType->getElementType(), snd.isLeftValue));

    // Each T can coerce with any DeclRefType.
    if (auto eachSnd = as<EachType>(snd))
    {
        if (auto innerSnd = eachSnd->getElementDeclRefType())
        {
            if (auto sndTypePackParamDecl =
                    as<GenericTypePackParamDecl>(innerSnd->getDeclRef().getDecl()))
            {
                if (innerSnd->getDeclRef().getDecl()->parentDecl == constraints.genericDecl)
                {
                    return TryUnifyTypeParam(constraints, unifyCtx, sndTypePackParamDecl, fst);
                }
            }
        }
    }
    if (auto eachFst = as<EachType>(fst))
    {
        if (auto innerFst = eachFst->getElementDeclRefType())
        {
            if (auto fstTypePackParamDecl =
                    as<GenericTypePackParamDecl>(innerFst->getDeclRef().getDecl()))
            {
                if (innerFst->getDeclRef().getDecl()->parentDecl == constraints.genericDecl)
                {
                    return TryUnifyTypeParam(constraints, unifyCtx, fstTypePackParamDecl, snd);
                }
            }
        }
    }
    return false;
}


} // namespace Slang
