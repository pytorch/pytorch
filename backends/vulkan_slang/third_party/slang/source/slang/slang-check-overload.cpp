// slang-check-overload.cpp

#include "slang-ast-base.h"
#include "slang-ast-print.h"
#include "slang-check-impl.h"
#include "slang-lookup.h"

// This file implements semantic checking logic related
// to resolving overloading call operations, by checking
// the applicability and relative priority of various candidates.

namespace Slang
{

bool isFreeFormTypePackParam(SemanticsVisitor* visitor, Type* type, ParamDecl* paramDecl)
{
    if (auto declRef = isDeclRefTypeOf<GenericTypePackParamDecl>(type))
    {
        return visitor->GetOuterGeneric(declRef.getDecl()) ==
               visitor->GetOuterGeneric(paramDecl->parentDecl);
    }
    return false;
}

SemanticsVisitor::ParamCounts SemanticsVisitor::CountParameters(
    FilteredMemberRefList<ParamDecl> params)
{
    ParamCounts counts = {0, 0};
    for (auto param : params)
    {
        Index allowedArgCountToAdd = 1;
        auto paramType = getParamType(m_astBuilder, param);
        if (isTypePack(paramType))
        {
            if (auto typePack = as<ConcreteTypePack>(paramType))
            {
                counts.required += typePack->getTypeCount();
                allowedArgCountToAdd = typePack->getTypeCount();
            }
            else if (isFreeFormTypePackParam(this, paramType, param.getDecl()))
            {
                counts.allowed = -1;
            }
            else
            {
                counts.required++;
                counts.allowed++;
            }
        }
        else if (!param.getDecl()->initExpr)
        {
            // No initializer means no default value
            //
            // TODO(tfoley): The logic here is currently broken in two ways:
            //
            // 1. We are assuming that once one parameter has a default, then all do.
            //    This can/should be validated earlier, so that we can assume it here.
            //
            // 2. We are not handling the possibility of multiple declarations for
            //    a single function, where we'd need to merge default parameters across
            //    all the declarations.
            counts.required++;
        }

        if (counts.allowed >= 0)
            counts.allowed += allowedArgCountToAdd;
    }
    return counts;
}

SemanticsVisitor::ParamCounts SemanticsVisitor::CountParameters(DeclRef<GenericDecl> genericRef)
{
    ParamCounts counts = {0, 0};
    for (auto m : genericRef.getDecl()->members)
    {
        if (auto typeParam = as<GenericTypeParamDecl>(m))
        {
            if (counts.allowed >= 0)
                counts.allowed++;
            if (!typeParam->initType.Ptr())
            {
                counts.required++;
            }
        }
        else if (auto valParam = as<GenericValueParamDecl>(m))
        {
            if (counts.allowed >= 0)
                counts.allowed++;
            if (!valParam->initExpr)
            {
                counts.required++;
            }
        }
        else if (as<GenericTypePackParamDecl>(m))
        {
            counts.allowed = -1;
        }
    }
    return counts;
}

bool SemanticsVisitor::TryCheckOverloadCandidateClassNewMatchUp(
    OverloadResolveContext& context,
    OverloadCandidate const& candidate)
{
    // Check that a constructor call to a class type must be in a `new` expr, and a `new` expr
    // is only used to construct a class.
    bool isClassType = false;
    bool isNewExpr = false;
    if (auto ctorDeclRef = candidate.item.declRef.as<ConstructorDecl>())
    {
        if (auto resultType = as<DeclRefType>(candidate.resultType))
        {
            if (resultType->getDeclRef().as<ClassDecl>())
            {
                isClassType = true;
            }
        }
    }
    if (as<NewExpr>(context.originalExpr))
    {
        isNewExpr = true;
    }

    if (isNewExpr && !isClassType)
    {
        getSink()->diagnose(context.originalExpr, Diagnostics::newCanOnlyBeUsedToInitializeAClass);
        return false;
    }
    if (!isNewExpr && isClassType && context.originalExpr)
    {
        getSink()->diagnose(context.originalExpr, Diagnostics::classCanOnlyBeInitializedWithNew);
        return false;
    }
    return true;
}

bool SemanticsVisitor::TryCheckOverloadCandidateArity(
    OverloadResolveContext& context,
    OverloadCandidate const& candidate)
{
    Count argCount = context.getArgCount();
    ParamCounts paramCounts = {0, 0};
    switch (candidate.flavor)
    {
    case OverloadCandidate::Flavor::Func:
        paramCounts =
            CountParameters(getParameters(m_astBuilder, candidate.item.declRef.as<CallableDecl>()));
        break;

    case OverloadCandidate::Flavor::Generic:
        paramCounts = CountParameters(candidate.item.declRef.as<GenericDecl>());

        // A generic can be applied to any number of arguments less
        // than or equal to the number of explicitly declared parameters.
        // When a program provides fewer arguments than their are parameters,
        // the rest will be inferred.
        //
        paramCounts.required = 0;
        break;

    case OverloadCandidate::Flavor::Expr:
        {
            auto paramCount = candidate.funcType->getParamCount();
            paramCounts.allowed = paramCount;
            paramCounts.required = paramCount;
        }
        break;

    default:
        SLANG_UNEXPECTED("unknown flavor of overload candidate");
        break;
    }

    if (argCount >= paramCounts.required &&
        (paramCounts.allowed == -1 || argCount <= paramCounts.allowed))
        return true;

    // Emit an error message if we are checking this call for real
    if (context.mode != OverloadResolveContext::Mode::JustTrying)
    {
        if (argCount < paramCounts.required)
        {
            getSink()->diagnose(
                context.loc,
                Diagnostics::notEnoughArguments,
                argCount,
                paramCounts.required);
        }
        else
        {
            SLANG_ASSERT(argCount > paramCounts.allowed);
            getSink()->diagnose(
                context.loc,
                Diagnostics::tooManyArguments,
                argCount,
                paramCounts.allowed);
        }
    }

    return false;
}

bool SemanticsVisitor::TryCheckOverloadCandidateFixity(
    OverloadResolveContext& context,
    OverloadCandidate const& candidate)
{
    auto expr = context.originalExpr;

    auto decl = candidate.item.declRef.getDecl();

    if (const auto prefixExpr = as<PrefixExpr>(expr))
    {
        if (decl->hasModifier<PrefixModifier>())
            return true;

        if (context.mode != OverloadResolveContext::Mode::JustTrying)
        {
            getSink()->diagnose(context.loc, Diagnostics::expectedPrefixOperator);
            getSink()->diagnose(decl, Diagnostics::seeDefinitionOf, decl->getName());
        }

        return false;
    }
    else if (const auto postfixExpr = as<PostfixExpr>(expr))
    {
        if (decl->hasModifier<PostfixModifier>())
            return true;

        if (context.mode != OverloadResolveContext::Mode::JustTrying)
        {
            getSink()->diagnose(context.loc, Diagnostics::expectedPostfixOperator);
            getSink()->diagnose(decl, Diagnostics::seeDefinitionOf, decl->getName());
        }

        return false;
    }
    else
    {
        return true;
    }
}

bool SemanticsVisitor::TryCheckOverloadCandidateVisibility(
    OverloadResolveContext& context,
    OverloadCandidate const& candidate)
{
    // Always succeeds when we are trying out constructors.
    if (context.mode == OverloadResolveContext::Mode::JustTrying)
    {
        if (as<ConstructorDecl>(candidate.item.declRef))
            return true;
    }

    if (!context.sourceScope)
        return true;

    if (!candidate.item.declRef)
        return true;

    if (!isDeclVisibleFromScope(candidate.item.declRef, context.sourceScope))
    {
        if (context.mode == OverloadResolveContext::Mode::ForReal)
        {
            getSink()->diagnose(context.loc, Diagnostics::declIsNotVisible, candidate.item.declRef);
        }
        return false;
    }

    return true;
}

bool SemanticsVisitor::TryCheckGenericOverloadCandidateTypes(
    OverloadResolveContext& context,
    OverloadCandidate& candidate)
{
    auto genericDeclRef = candidate.item.declRef.as<GenericDecl>();

    // Only allow constructing a PartialGenericAppExpr when referencing a callable decl.
    // Other types of generic decls must be fully specified.
    bool allowPartialGenericApp = false;
    if (as<CallableDecl>(genericDeclRef.getDecl()->inner))
    {
        allowPartialGenericApp = true;
    }

    // The basic idea here is that we need to check that the
    // arguments to a generic application (e.g., `F<A1, A2, ...>`)
    // have the right "type," which in this context means
    // checking that:
    //
    // * The argument for any generic type parameter is a (proper) type.
    //
    // * The argument for any generic value parameter is a
    //   specialization-time constant value of the appropriate type.
    //
    // Some additional checks are *not* handled at this point:
    //
    // * We don't check that a type argument actually conforms to
    //   the constraints on the parameter.
    //
    // Along the way we will build up a `GenericSubstitution`
    // to represent the arguments that have been coerced to
    // appropriate forms.
    //
    List<Val*> checkedArgs;

    // Rather than bail out as soon as we hit a problem,
    // we are going to process *all* of the parameters of the
    // generic and place suitable arguments into the `checkedArgs`
    // array. This is important so that we don't cause crashes
    // in cases where the arguments fail this step of checking,
    // but we decide to proceed with subsequent steps (e.g.,
    // because the candidate we are trying here is the *only*
    // candidate).
    //
    bool success = true;

    auto maybeReportGeneralError = [&]()
    {
        if (context.mode != OverloadResolveContext::Mode::JustTrying)
        {
            getSink()->diagnose(
                context.loc,
                Diagnostics::cannotSpecializeGeneric,
                candidate.item.declRef);
        }
    };
    List<QualType> paramTypes;
    for (auto memberRef : getMembers(m_astBuilder, genericDeclRef))
    {
        if (auto typeParamRef = memberRef.as<GenericTypeParamDecl>())
        {
            paramTypes.add(DeclRefType::create(m_astBuilder, typeParamRef));
        }
        else if (auto valParamRef = memberRef.as<GenericValueParamDecl>())
        {
            paramTypes.add(getType(m_astBuilder, valParamRef));
        }
        else if (auto typePackParam = memberRef.as<GenericTypePackParamDecl>())
        {
            paramTypes.add(DeclRefType::create(m_astBuilder, typePackParam));
        }
    }
    ShortList<OverloadResolveContext::MatchedArg> matchedArgs;
    if (!context.matchArgumentsToParams(this, paramTypes, false, matchedArgs))
    {
        maybeReportGeneralError();
        return false;
    }

    Index aa = 0;
    for (auto memberRef : getMembers(m_astBuilder, genericDeclRef))
    {
        if (auto typeParamRef = memberRef.as<GenericTypeParamDecl>())
        {
            if (aa >= matchedArgs.getCount())
            {
                if (allowPartialGenericApp)
                {
                    // If we have run out of arguments, and the referenced decl
                    // allows partially applied specialization (i.e. a callable
                    // decl) then we don't apply any more checks at this step.
                    // We will instead attempt to *infer* an argument at this
                    // position at a later stage.
                    //
                    candidate.flags |= OverloadCandidate::Flag::IsPartiallyAppliedGeneric;
                    break;
                }
                else
                {
                    // Otherwise, the generic decl had better provide a default value
                    // or this reference is ill-formed.
                    auto substType = typeParamRef.substitute(
                        m_astBuilder,
                        typeParamRef.getDecl()->initType.type);
                    if (!substType)
                    {
                        maybeReportGeneralError();
                        return false;
                    }
                    checkedArgs.add(substType);
                    continue;
                }
            }

            // We have a type parameter, and we expect to find
            // a type argument.
            //
            TypeExp typeArg;

            // Per the earlier check, we have at least one
            // argument left, so we will grab
            // it and try to coerce it to a proper type. The
            // manner in which we handle the coercion depends
            // on whether we are "just trying" the candidate
            // (so a failure would rule out the candidate, but
            // shouldn't be reported to the user), or are doing
            // the checking "for real" in which case any errors
            // we run into need to be reported.
            //
            auto arg = matchedArgs[aa++];
            if (context.mode == OverloadResolveContext::Mode::JustTrying)
            {
                typeArg = tryCoerceToProperType(TypeExp(arg.argExpr));
            }
            else
            {
                arg.argExpr = ExpectATypeRepr(arg.argExpr);
                typeArg = CoerceToProperType(TypeExp(arg.argExpr));
            }

            // If we failed to get a valid type (either because
            // there was no matching argument, or because the
            // "just trying" coercion failed), then we create
            // an error type to stand in for the argument
            //
            if (!typeArg.type)
            {
                typeArg.type = m_astBuilder->getErrorType();
                success = false;
            }

            checkedArgs.add(typeArg.type);
        }
        else if (auto valParamRef = memberRef.as<GenericValueParamDecl>())
        {
            if (aa >= matchedArgs.getCount())
            {
                if (allowPartialGenericApp)
                {
                    // If we have run out of arguments and the decl allows
                    // partial specialization, then we don't apply any more
                    // checks at this step. We will instead attempt to
                    // *infer* an argument at this position at a later
                    // stage.
                    //
                    candidate.flags |= OverloadCandidate::Flag::IsPartiallyAppliedGeneric;
                    break;
                }
                else
                {
                    // Otherwise, the generic decl had better provide a default value
                    // or this reference is ill-formed.
                    ensureDecl(valParamRef, DeclCheckState::DefinitionChecked);
                    ConstantFoldingCircularityInfo newCircularityInfo(
                        valParamRef.getDecl(),
                        nullptr);
                    auto defaultVal = tryConstantFoldExpr(
                        valParamRef.substitute(m_astBuilder, valParamRef.getDecl()->initExpr),
                        ConstantFoldingKind::CompileTime,
                        &newCircularityInfo);
                    if (!defaultVal)
                    {
                        maybeReportGeneralError();
                        return false;
                    }
                    checkedArgs.add(defaultVal);
                    continue;
                }
            }

            // The case for a generic value parameter is similar to that
            // for a generic type parameter.
            //
            Expr* arg = nullptr;

            // If we have an argument then we need to coerce it
            // to the type of the parameter (and fail if the
            // coercion is not possible)
            //
            arg = matchedArgs[aa++].argExpr;
            if (context.mode == OverloadResolveContext::Mode::JustTrying)
            {
                ConversionCost cost = kConversionCost_None;
                if (!canCoerce(getType(m_astBuilder, valParamRef), arg->type, arg, &cost))
                {
                    success = false;
                }
                candidate.conversionCostSum += cost;
            }
            else
            {
                arg = coerce(CoercionSite::Argument, getType(m_astBuilder, valParamRef), arg);
            }

            // If we have an argument to work with, then we will
            // try to extract its speicalization-time constant value.
            //
            Val* val = nullptr;
            if (arg)
            {
                val = ExtractGenericArgInteger(
                    arg,
                    getType(m_astBuilder, valParamRef),
                    context.mode == OverloadResolveContext::Mode::JustTrying ? nullptr : getSink());
            }

            // If any of the above checking steps fail and we don't
            // have a value to work with here, we will instead
            // use an "error" value to stand in for the argument.
            //
            if (!val)
            {
                val = m_astBuilder->getOrCreate<ErrorIntVal>(m_astBuilder->getIntType());
            }
            checkedArgs.add(val);
        }
        else if (auto typePackParam = memberRef.as<GenericTypePackParamDecl>())
        {
            Val* val = nullptr;
            if (aa >= matchedArgs.getCount())
            {
                if (allowPartialGenericApp)
                {
                    // If we have run out of arguments and the decl allows
                    // partial specialization, then we don't apply any more
                    // checks at this step. We will instead attempt to
                    // *infer* an argument at this position at a later
                    // stage.
                    //
                    candidate.flags |= OverloadCandidate::Flag::IsPartiallyAppliedGeneric;
                    break;
                }
                else
                {
                    // Otherwise, we will just create an empty pack.
                    val = m_astBuilder->getTypePack(ArrayView<Type*>());
                }
            }
            else
            {
                auto matchedArg = matchedArgs[aa++];
                if (auto packExpr = as<PackExpr>(matchedArg.argExpr))
                {
                    // We are providing a concrete pack of types as arguments to a type pack
                    // parameter. We need to create a `TypePack` type to serve as the argument.
                    ShortList<Type*> coercedProperTypes;

                    // Coerce all types in the pack to proper types.
                    for (Index i = 0; i < packExpr->args.getCount(); i++)
                    {
                        TypeExp typeArg;
                        auto elementTypeExpr = packExpr->args[i];
                        if (context.mode == OverloadResolveContext::Mode::JustTrying)
                        {
                            typeArg = tryCoerceToProperType(TypeExp(elementTypeExpr));
                            if (!typeArg.type)
                            {
                                typeArg.type = m_astBuilder->getErrorType();
                                success = false;
                            }
                        }
                        else
                        {
                            elementTypeExpr = ExpectATypeRepr(elementTypeExpr);
                            typeArg = CoerceToProperType(TypeExp(elementTypeExpr));
                        }
                        // If we failed to get a valid type (either because
                        // there was no matching argument, or because the
                        // "just trying" coercion failed), then we create
                        // an error type to stand in for the argument
                        //
                        if (!typeArg.type)
                        {
                            typeArg.type = m_astBuilder->getErrorType();
                            success = false;
                        }
                        coercedProperTypes.add(typeArg.type);
                    }
                    val = m_astBuilder->getTypePack(coercedProperTypes.getArrayView().arrayView);
                }
                else if (auto expandExpr = as<ExpandExpr>(matchedArg.argExpr))
                {
                    auto argType = expandExpr->type.type;
                    if (auto typeType = as<TypeType>(argType))
                        argType = typeType->getType();
                    val = argType;
                }
                else if (auto typeType = as<TypeType>(matchedArg.argType))
                {
                    if (isAbstractTypePack(typeType->getType()))
                    {
                        val = typeType->getType();
                    }
                }
            }
            if (val == nullptr)
            {
                maybeReportGeneralError();
                return false;
            }
            checkedArgs.add(val);
        }
        else
        {
            continue;
        }
    }

    auto genSubst = m_astBuilder->getGenericAppDeclRef(genericDeclRef, checkedArgs.getArrayView());
    candidate.subst = SubstitutionSet(genSubst);

    // Once we are done processing the parameters of the generic,
    // we will have build up a usable `checkedArgs` array and
    // can return to the caller a report of whether we
    // were successful or not.
    //
    return success;
}

static QualType getParamQualType(ASTBuilder* astBuilder, DeclRef<ParamDecl> param)
{
    auto paramType = getType(astBuilder, param);
    bool isLVal = false;
    switch (getParameterDirection(param.getDecl()))
    {
    case kParameterDirection_InOut:
    case kParameterDirection_Out:
    case kParameterDirection_Ref:
        isLVal = true;
        break;
    }
    return QualType(paramType, isLVal);
}

static QualType getParamQualType(Type* paramType)
{
    if (auto paramDirType = as<ParamDirectionType>(paramType))
    {
        if (as<OutTypeBase>(paramDirType) || as<RefType>(paramDirType))
            return QualType(paramDirType->getValueType(), true);
    }
    return paramType;
}

bool SemanticsVisitor::TryCheckOverloadCandidateTypes(
    OverloadResolveContext& context,
    OverloadCandidate& candidate)
{
    Index argCount = context.getArgCount();

    List<QualType> paramTypes;
    switch (candidate.flavor)
    {
    case OverloadCandidate::Flavor::Func:
        for (auto param : getParameters(m_astBuilder, candidate.item.declRef.as<CallableDecl>()))
        {
            paramTypes.add(getParamQualType(m_astBuilder, param));
        }
        break;

    case OverloadCandidate::Flavor::Expr:
        {
            auto funcType = candidate.funcType;
            Count paramCount = funcType->getParamCount();
            for (Index i = 0; i < paramCount; ++i)
            {
                auto paramType = getParamQualType(funcType->getParamType(i));
                paramTypes.add(paramType);
            }
        }
        break;

    case OverloadCandidate::Flavor::Generic:
        {
            return TryCheckGenericOverloadCandidateTypes(context, candidate);
        }
    default:
        SLANG_UNEXPECTED("unknown flavor of overload candidate");
        break;
    }

    Index paramIndex = 0;
    Index argIndex = 0;
    struct Arg
    {
        Expr* argExpr;
        Type* type;
    };
    auto readArg = [&]() -> Arg
    {
        if (argIndex >= argCount)
            return {nullptr, nullptr};
        auto arg = context.getArg(argIndex);
        Arg result = {arg, context.getArgType(argIndex)};
        argIndex++;
        return result;
    };

    auto coerceArgToParam = [&](Arg arg, QualType paramType) -> Arg
    {
        auto argType = QualType(arg.type, paramType.isLeftValue);
        if (!paramType)
            return {nullptr, nullptr};
        if (!argType)
            return {nullptr, nullptr};
        if (context.mode == OverloadResolveContext::Mode::JustTrying)
        {
            ConversionCost cost = kConversionCost_None;
            if (context.disallowNestedConversions)
            {
                // We need an exact match in this case.
                if (!paramType->equals(argType))
                    return {nullptr, nullptr};
            }
            else if (!canCoerce(paramType, argType, arg.argExpr, &cost))
            {
                return {nullptr, nullptr};
            }
            candidate.conversionCostSum += cost;
        }
        else
        {
            Expr* coercedExpr = coerce(CoercionSite::Argument, paramType, arg.argExpr);

            // Check if concrete-to-interface coercion caused loss of l-valueness.
            if (coercedExpr && !coercedExpr->type.isLeftValue && paramType.isLeftValue &&
                !isInterfaceType(arg.type) && isInterfaceType(paramType.type))
            {
                if (context.mode != OverloadResolveContext::Mode::JustTrying)
                {
                    String name;
                    if (candidate.flavor == OverloadCandidate::Flavor::Func)
                    {
                        auto decl = getParameters(
                            m_astBuilder,
                            candidate.item.declRef.as<CallableDecl>())[paramIndex];
                        name = getText(decl.getName());
                    }
                    else
                        name.append(paramIndex, 10);

                    getSink()->diagnose(
                        context.loc,
                        Diagnostics::concreteArgumentToOutputInterface,
                        name,
                        arg.type,
                        paramType.type);
                }
                return {nullptr, nullptr};
            }
            arg.argExpr = coercedExpr;
        }
        return arg;
    };
    ShortList<Expr*> resultArgs;

    while (paramIndex < paramTypes.getCount())
    {
        auto paramType = paramTypes[paramIndex];
        if (auto paramTypePack = as<ConcreteTypePack>(paramType))
        {
            ShortList<Expr*> innerArgs;
            for (Index i = 0; i < paramTypePack->getTypeCount(); i++)
            {
                auto arg = readArg();
                auto coercedArg = coerceArgToParam(
                    arg,
                    QualType(paramTypePack->getElementType(i), paramType.isLeftValue));
                if (!coercedArg.type)
                {
                    return false;
                }
                if (context.mode == OverloadResolveContext::Mode::ForReal)
                    innerArgs.add(coercedArg.argExpr);
            }
            if (context.mode == OverloadResolveContext::Mode::ForReal)
            {
                auto packArg = m_astBuilder->create<PackExpr>();
                for (auto aa : innerArgs)
                    packArg->args.add(aa);
                packArg->type = paramType;
                resultArgs.add(packArg);
            }

            // Always add a flat cost for using an argument pack,
            // so that we prefer non-pack overloads when possible.
            candidate.conversionCostSum += kConversionCost_ParameterPack;
        }
        else
        {
            auto arg = readArg();
            if (!arg.type)
            {
                // If we run out of arguments, we can exit the loop now.
                // Note that in this type we don't need to worry about
                // default arguments, because we already checked that
                // the number of arguments was correct in `TryCheckOverloadCandidateArity`.
                break;
            }
            auto coercedArg = coerceArgToParam(arg, paramType);
            if (!coercedArg.type)
            {
                return false;
            }
            if (context.mode == OverloadResolveContext::Mode::ForReal)
                resultArgs.add(coercedArg.argExpr);
        }
        paramIndex++;
    }
    if (context.mode == OverloadResolveContext::Mode::ForReal)
    {
        context.argCount = resultArgs.getCount();
        if (context.args)
        {
            context.args->setCount(context.argCount);
            for (Index i = 0; i < context.argCount; i++)
                (*context.args)[i] = resultArgs[i];
        }
    }
    return true;
}

bool isEffectivelyMutating(CallableDecl* decl)
{
    if (decl->hasModifier<MutatingAttribute>())
        return true;
    if (decl->hasModifier<RefAttribute>())
        return true;
    if (decl->hasModifier<NonmutatingAttribute>())
        return false;

    if (as<SetterDecl>(decl))
        return true;

    return false;
}

ParamDecl* SemanticsVisitor::isReferenceIntoFunctionInputParameter(Expr* inExpr)
{
    auto expr = inExpr;
    for (;;)
    {
        if (auto declRefExpr = as<DeclRefExpr>(expr))
        {
            auto declRef = declRefExpr->declRef;
            if (auto paramDeclRef = declRef.as<ParamDecl>())
            {
                if (paramDeclRef.as<ModernParamDecl>())
                {
                    // functions declared in our "modern" style (using
                    // the `func` keyword) never have mutable `in`
                    // parameters.
                    //
                    return nullptr;
                }

                if (paramDeclRef.getDecl()->findModifier<OutModifier>() ||
                    paramDeclRef.getDecl()->findModifier<RefModifier>())
                {
                    // Function parameters marked with `out`, `inout`,
                    // `in out` or `ref` are all mutable in a way where
                    // the result of mutations will be visible to the
                    // caller.
                    //
                    return nullptr;
                }

                // At this point we have an l-value decl-ref to a
                // function parameter that is (implicitly or
                // explicitly) declared `in`.
                //
                return paramDeclRef.getDecl();
            }
        }
        else if (auto memberExpr = as<MemberExpr>(expr))
        {
            expr = memberExpr->baseExpression;
            continue;
        }
        else if (auto indexExpr = as<IndexExpr>(expr))
        {
            expr = indexExpr->baseExpression;
            continue;
        }

        return nullptr;
    }
}

bool SemanticsVisitor::TryCheckOverloadCandidateDirections(
    OverloadResolveContext& context,
    OverloadCandidate const& candidate)
{
    if (candidate.flavor != OverloadCandidate::Flavor::Func)
        return true;

    auto funcDeclRef = candidate.item.declRef.as<CallableDecl>();
    SLANG_ASSERT(funcDeclRef);

    // Note: This operation was originally introduced as
    // a place to add checking around l-value-ness of arguments
    // and parameters, but currently that checking is being
    // done in other places.
    //
    // For now we will only use this step to check the
    // mutability of the `this` parameter where necessary.
    //
    if (!isEffectivelyStatic(funcDeclRef.getDecl()))
    {
        if (isEffectivelyMutating(funcDeclRef.getDecl()))
        {
            if (context.baseExpr && !context.baseExpr->type.isLeftValue)
            {
                if (context.mode == OverloadResolveContext::Mode::ForReal)
                {
                    getSink()->diagnose(
                        context.loc,
                        Diagnostics::mutatingMethodOnImmutableValue,
                        funcDeclRef.getName());
                    maybeDiagnoseThisNotLValue(context.baseExpr);
                }
                return false;
            }

            // The parameters of functions declared using traditional/legacy
            // syntax are currently exposed as mutable locals within the body
            // of the relevant function. As such, it is legal to call `[mutating]`
            // methods on such a function parameter. However, doing so is typically
            // indicative of an error on the programmer's part.
            //
            // We will detect such cases here and issue a diagnostic that explains
            // the situation.
            //
            if (context.baseExpr && context.mode == OverloadResolveContext::Mode::ForReal)
            {
                if (auto paramDecl = isReferenceIntoFunctionInputParameter(context.baseExpr))
                {
                    const bool isNonCopyable = isNonCopyableType(paramDecl->getType());

                    const auto& diagnotic =
                        isNonCopyable ? Diagnostics::mutatingMethodOnFunctionInputParameterError
                                      : Diagnostics::mutatingMethodOnFunctionInputParameterWarning;

                    getSink()->diagnose(
                        context.loc,
                        diagnotic,
                        funcDeclRef.getName(),
                        paramDecl->getName());
                }
            }
        }
    }

    return true;
}

bool SemanticsVisitor::TryCheckOverloadCandidateConstraints(
    OverloadResolveContext& context,
    OverloadCandidate& candidate)
{
    // We only need this step for generics, so always succeed on
    // everything else.
    if (candidate.flavor != OverloadCandidate::Flavor::Generic)
        return true;

    // It is possible that the overload candidate was only partially
    // applied (the number of arguments was not equal to the number
    // of explicit parameters). In that case, we want to defer
    // final checking of things like constraints until later, in
    // case a subsequent pass of overload resolution (like applying
    // an overloaded generic function to arguments) will give us
    // the missing information to enable inference.
    //
    if (candidate.flags & OverloadCandidate::Flag::IsPartiallyAppliedGeneric)
        return true;

    auto genericDeclRef = candidate.item.declRef.as<GenericDecl>();
    SLANG_ASSERT(genericDeclRef); // otherwise we wouldn't be a generic candidate...

    // We should have the existing arguments to the generic
    // handy, so that we can construct a substitution list.
    auto substArgs = tryGetGenericArguments(candidate.subst, genericDeclRef.getDecl());
    SLANG_ASSERT(substArgs.getCount());

    List<Val*> newArgs;
    for (auto arg : substArgs)
        newArgs.add(arg);

    for (auto constraintDecl :
         genericDeclRef.getDecl()->getMembersOfType<GenericTypeConstraintDecl>())
    {
        DeclRef<GenericTypeConstraintDecl> constraintDeclRef =
            m_astBuilder
                ->getGenericAppDeclRef(genericDeclRef, newArgs.getArrayView(), constraintDecl)
                .as<GenericTypeConstraintDecl>();

        auto sub = getSub(m_astBuilder, constraintDeclRef);
        auto sup = getSup(m_astBuilder, constraintDeclRef);

        auto subTypeWitness = tryGetSubtypeWitness(sub, sup);
        if (subTypeWitness)
        {
            newArgs.add(subTypeWitness);
        }
        else
        {
            if (context.mode != OverloadResolveContext::Mode::JustTrying)
            {
                subTypeWitness = isSubtype(sub, sup, IsSubTypeOptions::None);
                getSink()->diagnose(
                    context.loc,
                    Diagnostics::typeArgumentDoesNotConformToInterface,
                    sub,
                    sup);
            }
            return false;
        }
    }

    candidate.subst =
        SubstitutionSet(m_astBuilder->getGenericAppDeclRef(genericDeclRef, newArgs.getArrayView()));

    // Done checking all the constraints, hooray.
    return true;
}

void SemanticsVisitor::TryCheckOverloadCandidate(
    OverloadResolveContext& context,
    OverloadCandidate& candidate)
{
    if (!TryCheckOverloadCandidateArity(context, candidate))
        return;

    candidate.status = OverloadCandidate::Status::ArityChecked;
    if (!TryCheckOverloadCandidateFixity(context, candidate))
        return;

    candidate.status = OverloadCandidate::Status::FixityChecked;
    if (!TryCheckOverloadCandidateTypes(context, candidate))
        return;

    candidate.status = OverloadCandidate::Status::TypeChecked;
    if (!TryCheckOverloadCandidateDirections(context, candidate))
        return;

    candidate.status = OverloadCandidate::Status::DirectionChecked;
    if (!TryCheckOverloadCandidateConstraints(context, candidate))
        return;

    candidate.status = OverloadCandidate::Status::VisibilityChecked;
    if (!TryCheckOverloadCandidateVisibility(context, candidate))
        return;

    candidate.status = OverloadCandidate::Status::Applicable;
}

Expr* SemanticsVisitor::createGenericDeclRef(
    Expr* baseExpr,
    Expr* originalExpr,
    SubstitutionSet substArgs)
{
    auto baseDeclRefExpr = as<DeclRefExpr>(baseExpr);
    if (!baseDeclRefExpr)
    {
        SLANG_DIAGNOSE_UNEXPECTED(
            getSink(),
            baseExpr,
            "expected a reference to a generic declaration");
        return CreateErrorExpr(originalExpr);
    }
    auto baseGenericRef = baseDeclRefExpr->declRef.as<GenericDecl>();
    if (!baseGenericRef)
    {
        SLANG_DIAGNOSE_UNEXPECTED(
            getSink(),
            baseExpr,
            "expected a reference to a generic declaration");
        return CreateErrorExpr(originalExpr);
    }
    auto genSubst = substArgs.findGenericAppDeclRef(baseGenericRef.getDecl());
    SLANG_ASSERT(genSubst);
    DeclRef<Decl> innerDeclRef =
        m_astBuilder->getGenericAppDeclRef(baseGenericRef, genSubst->getArgs());

    Expr* base = nullptr;
    if (auto mbrExpr = as<MemberExpr>(baseExpr))
        base = mbrExpr->baseExpression;

    return ConstructDeclRefExpr(
        innerDeclRef,
        base,
        innerDeclRef.getName(),
        originalExpr->loc,
        originalExpr);
}

Expr* SemanticsVisitor::CompleteOverloadCandidate(
    OverloadResolveContext& context,
    OverloadCandidate& candidate)
{
    // special case for generic argument inference failure
    if (candidate.status == OverloadCandidate::Status::GenericArgumentInferenceFailed)
    {
        String callString = getCallSignatureString(context);
        getSink()->diagnose(context.loc, Diagnostics::genericArgumentInferenceFailed, callString);

        String declString = ASTPrinter::getDeclSignatureString(candidate.item, m_astBuilder);
        getSink()->diagnose(candidate.item.declRef, Diagnostics::genericSignatureTried, declString);
        goto error;
    }

    context.mode = OverloadResolveContext::Mode::ForReal;

    if (!TryCheckOverloadCandidateClassNewMatchUp(context, candidate))
        goto error;

    if (!TryCheckOverloadCandidateArity(context, candidate))
        goto error;

    if (!TryCheckOverloadCandidateFixity(context, candidate))
        goto error;

    if (!TryCheckOverloadCandidateTypes(context, candidate))
        goto error;

    if (!TryCheckOverloadCandidateDirections(context, candidate))
        goto error;

    if (!TryCheckOverloadCandidateConstraints(context, candidate))
        goto error;

    if (!TryCheckOverloadCandidateVisibility(context, candidate))
        goto error;

    {
        Expr* baseExpr;
        switch (candidate.flavor)
        {
        case OverloadCandidate::Flavor::Func:
        case OverloadCandidate::Flavor::Generic:
            baseExpr = ConstructLookupResultExpr(
                candidate.item,
                context.baseExpr,
                candidate.item.declRef.getName(),
                context.funcLoc,
                context.originalExpr);
            break;
        case OverloadCandidate::Flavor::Expr:
        default:
            baseExpr = nullptr;
            break;
        }

        switch (candidate.flavor)
        {
        case OverloadCandidate::Flavor::Func:
            {
                AppExprBase* callExpr = as<InvokeExpr>(context.originalExpr);
                if (!callExpr)
                {
                    callExpr = m_astBuilder->create<InvokeExpr>();
                    callExpr->loc = context.loc;
                    for (Index aa = 0; aa < context.argCount; ++aa)
                        callExpr->arguments.add(context.getArg(aa));
                }

                callExpr->originalFunctionExpr = callExpr->functionExpr;
                callExpr->functionExpr = baseExpr;
                callExpr->type = QualType(candidate.resultType);

                // A call may yield an l-value, and we should take a look at the candidate to be
                // sure
                if (auto subscriptDeclRef = candidate.item.declRef.as<SubscriptDecl>())
                {
                    const auto& decl = subscriptDeclRef.getDecl();
                    for (auto member : decl->members)
                    {
                        if (as<SetterDecl>(member) || as<RefAccessorDecl>(member))
                        {
                            // If the subscript decl has a setter,
                            // then the call is an l-value if base is l-value.
                            if (auto base = GetBaseExpr(baseExpr))
                            {
                                if (base->type.isLeftValue)
                                {
                                    callExpr->type.isLeftValue = true;
                                    break;
                                }
                            }
                            // Otherwise, if the accessor is [nonmutating], we can
                            // also consider the result of the subscript call as l-value
                            // regardless of the base.
                            if (member->findModifier<NonmutatingAttribute>())
                            {
                                callExpr->type.isLeftValue = true;
                                break;
                            }
                        }
                    }
                }

                // TODO: there may be other cases that confer l-value-ness

                return callExpr;
            }

            break;

        case OverloadCandidate::Flavor::Expr:
            {
                AppExprBase* callExpr = as<InvokeExpr>(context.originalExpr);
                if (!callExpr)
                {
                    callExpr = m_astBuilder->create<InvokeExpr>();
                    callExpr->loc = context.loc;
                    for (Index aa = 0; aa < context.argCount; ++aa)
                        callExpr->arguments.add(context.getArg(aa));
                }

                callExpr->originalFunctionExpr = callExpr->functionExpr;
                callExpr->type = QualType(candidate.resultType);
                callExpr->functionExpr = candidate.exprVal;
                return callExpr;
            }
            break;

        case OverloadCandidate::Flavor::Generic:
            // We allow a generic to be applied to fewer arguments than its number
            // of parameters, and defer the process of inferring the remaining
            // arguments until later.
            //
            if (candidate.flags & OverloadCandidate::Flag::IsPartiallyAppliedGeneric)
            {
                auto expr = m_astBuilder->create<PartiallyAppliedGenericExpr>();
                expr->loc = context.loc;
                expr->originalExpr = baseExpr;
                expr->baseGenericDeclRef = as<DeclRefExpr>(baseExpr)->declRef.as<GenericDecl>();
                auto args =
                    tryGetGenericArguments(candidate.subst, expr->baseGenericDeclRef.getDecl());
                for (auto arg : args)
                    expr->knownGenericArgs.add(arg);
                return expr;
            }

            return createGenericDeclRef(baseExpr, context.originalExpr, candidate.subst);
            break;

        default:
            SLANG_DIAGNOSE_UNEXPECTED(getSink(), context.loc, "unknown overload candidate flavor");
            break;
        }
    }


error:

    if (context.originalExpr)
    {
        return CreateErrorExpr(context.originalExpr);
    }
    else
    {
        return nullptr;
    }
}

/// Does the given `declRef` represent an interface requirement?
bool isInterfaceRequirement(ASTBuilder* builder, DeclRef<Decl> const& declRef)
{
    SLANG_UNUSED(builder);

    if (!declRef)
        return false;

    auto parent = declRef.getParent();
    if (parent.as<GenericDecl>())
        parent = parent.getParent();

    if (parent.as<InterfaceDecl>())
        return true;

    return false;
}

/// If `declRef` representations a specialization of a generic, returns the number of specialized
/// generic arguments. Otherwise, returns zero.
///
Int SemanticsVisitor::getSpecializedParamCount(DeclRef<Decl> const& declRef)
{
    if (!declRef)
        return 0;

    // A specialization of a generic must point at the
    // "inner" declaration of a generic. That means that
    // the parent of the decl ref must be a generic.
    //
    auto parentGeneric = declRef.getParent().as<GenericDecl>();
    if (!parentGeneric)
        return 0;
    //
    // Furthermore, the declaration we are considering
    // must be the single "inner" declaration of the
    // parent generic (and not somthing like a generic
    // parameter).
    //
    if (parentGeneric.getDecl()->inner != declRef.getDecl())
        return 0;

    return CountParameters(parentGeneric).required;
}

DeclRef<Decl> getParentDeclRef(DeclRef<Decl> declRef)
{
    auto parent = declRef.getParent();
    while (parent.as<GenericDecl>())
    {
        parent = parent.getParent();
    }
    return parent;
}

// Returns -1 if left is preferred, 1 if right is preferred, and 0 if they are equal.
//
int SemanticsVisitor::CompareLookupResultItems(
    LookupResultItem const& left,
    LookupResultItem const& right)
{
    // It is possible for lookup to return both an interface requirement
    // and the concrete function that satisfies that requirement.
    // We always want to favor a concrete method over an interface
    // requirement it might override.
    //
    // TODO: This should turn into a more detailed check such that
    // a candidate for declaration A is always better than a candidate
    // for declaration B if A is an override of B. We can't
    // easily make that check right now because we aren't tracking
    // this kind of "is an override of ..." information on declarations
    // directly (it is only visible through the requirement witness
    // information for inheritance declarations).
    //
    auto leftDeclRefParent = getParentDeclRef(left.declRef);
    auto rightDeclRefParent = getParentDeclRef(right.declRef);
    bool leftIsInterfaceRequirement = isInterfaceRequirement(left.declRef.getDecl());
    bool rightIsInterfaceRequirement = isInterfaceRequirement(right.declRef.getDecl());
    if (leftIsInterfaceRequirement != rightIsInterfaceRequirement)
        return int(leftIsInterfaceRequirement) - int(rightIsInterfaceRequirement);

    // Prefer non-extension declarations over extension declarations.
    bool leftIsExtension = as<ExtensionDecl>(leftDeclRefParent.getDecl()) != nullptr;
    bool rightIsExtension = as<ExtensionDecl>(rightDeclRefParent.getDecl()) != nullptr;
    if (leftIsExtension != rightIsExtension)
    {
        // Add a special case for constructors, where we prefer the one that is not synthesized,
        if (auto leftCtor = as<ConstructorDecl>(left.declRef.getDecl()))
        {
            if (auto rightCtor = as<ConstructorDecl>(right.declRef.getDecl()))
            {
                bool leftIsSynthesized = leftCtor->containsFlavor(
                    ConstructorDecl::ConstructorFlavor::SynthesizedDefault);
                bool rightIsSynthesized = rightCtor->containsFlavor(
                    ConstructorDecl::ConstructorFlavor::SynthesizedDefault);

                if (leftIsSynthesized != rightIsSynthesized)
                {
                    return int(leftIsSynthesized) - int(rightIsSynthesized);
                }
            }
        }

        return int(leftIsExtension) - int(rightIsExtension);
    }
    else if (leftIsExtension)
    {
        // If both are declared in extensions, prefer the one that is least generic.
        bool leftIsGeneric = leftDeclRefParent.getParent().as<GenericDecl>() != nullptr;
        bool rightIsGeneric = rightDeclRefParent.getParent().as<GenericDecl>() != nullptr;
        if (leftIsGeneric != rightIsGeneric)
        {
            return int(leftIsGeneric) - int(rightIsGeneric);
        }
    }

    // Any decl is strictly better than a module decl.
    bool leftIsModule = (as<ModuleDeclarationDecl>(left.declRef) != nullptr);
    bool rightIsModule = (as<ModuleDeclarationDecl>(right.declRef) != nullptr);
    if (leftIsModule != rightIsModule)
        return int(leftIsModule) - int(rightIsModule);

    // If both are interface requirements, prefer the more derived interface.
    if (leftIsInterfaceRequirement && rightIsInterfaceRequirement)
    {
        auto leftType = DeclRefType::create(m_astBuilder, leftDeclRefParent);
        auto rightType = DeclRefType::create(m_astBuilder, rightDeclRefParent);

        if (!leftType->equals(rightType))
        {
            if (isSubtype(leftType, rightType, IsSubTypeOptions::None))
                return -1;
            if (isSubtype(rightType, leftType, IsSubTypeOptions::None))
                return 1;
        }
    }

    // If both parents are the same we have ambiguity
    if (left.declRef.getParent() == right.declRef.getParent())
        return 0;

    auto leftAggType = leftDeclRefParent.as<AggTypeDeclBase>();
    auto rightAggType = rightDeclRefParent.as<AggTypeDeclBase>();
    if (leftAggType && rightAggType)
    {
        auto leftType = DeclRefType::create(m_astBuilder, leftDeclRefParent);
        auto rightType = DeclRefType::create(m_astBuilder, rightDeclRefParent);

        auto inheritanceInfo = getShared()->getInheritanceInfo(rightType);
        for (auto facet : inheritanceInfo.facets)
            if (facet.getImpl()->getDeclRef().equals(leftDeclRefParent))
                return 1;
        inheritanceInfo = getShared()->getInheritanceInfo(leftType);
        for (auto facet : inheritanceInfo.facets)
            if (facet.getImpl()->getDeclRef().equals(rightDeclRefParent))
                return -1;
    }

    // If both are subscript decls, prefer the one that provides more
    // accessors.
    if (auto leftSubscriptDecl = left.declRef.as<SubscriptDecl>())
    {
        if (auto rightSubscriptDecl = right.declRef.as<SubscriptDecl>())
        {
            auto leftAccessorCount =
                leftSubscriptDecl.getDecl()->getMembersOfType<AccessorDecl>().getCount();
            auto rightAccessorCount =
                rightSubscriptDecl.getDecl()->getMembersOfType<AccessorDecl>().getCount();
            auto decl1IsSubsetOfDecl2 = [=](SubscriptDecl* decl1, SubscriptDecl* decl2)
            {
                for (auto accessorDecl1 : decl1->getMembersOfType<AccessorDecl>())
                {
                    bool found = false;
                    for (auto accessorDecl2 : decl2->getMembersOfType<AccessorDecl>())
                    {
                        if (accessorDecl1->astNodeType == accessorDecl2->astNodeType)
                        {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                        return false;
                }
                return true;
            };
            if (leftAccessorCount > rightAccessorCount &&
                decl1IsSubsetOfDecl2(rightSubscriptDecl.getDecl(), leftSubscriptDecl.getDecl()))
            {
                return -1;
            }
            else if (
                rightAccessorCount > leftAccessorCount &&
                decl1IsSubsetOfDecl2(leftSubscriptDecl.getDecl(), rightSubscriptDecl.getDecl()))
            {
                return 1;
            }
        }
    }


    // TODO: We should generalize above rules such that in a tie a declaration
    // A::m is better than B::m when all other factors are equal and
    // A inherits from B.

    // TODO: There are other cases like this we need to add in terms
    // of ranking/prioritizing overloads, around things like
    // "transparent" members, or when lookup proceeds from an "inner"
    // to an "outer" scope. In many cases the right way to proceed
    // could involve attaching a distance/cost/rank to things directly
    // as part of lookup, and in other cases it might be best handled
    // as a semantic check based on the actual declarations found.

    return 0;
}

int SemanticsVisitor::compareOverloadCandidateSpecificity(
    LookupResultItem const& left,
    LookupResultItem const& right)
{
    // HACK: if both items refer to the same declaration,
    // then arbitrarily pick one.
    if (left.declRef.equals(right.declRef))
        return -1;

    // There is a very general rule that we would like to enforce
    // in principle:
    //
    // Given candidates A and B, if A being applicable to some
    // arguments implies that B is also applicable, but not vice versa,
    // then A is a more specific/specialized candidate than B.
    //
    // A number of conclusions follow from this general rule.
    // For example, a non-generic declaration will always be
    // more specific than a generic declaration that was specialized
    // to matching types:
    //
    //      int doThing(int a);
    //      T doThing<T>(T a);
    //
    // It is clear that if the non-generic `doThing` is applicable
    // to an argument `x`, then `doThing<int>` is also applicable to
    // `x`. However, knowing that the generic `doThing` was applicable
    // to some `y` doesn't tell us that the non-generic `doThing` can
    // be called on `y`, because `y` could have some type that can't
    // convert to `int`.
    //
    // Similarly, a generic declaration with a subset of the parameters
    // of another generic is always more specialized:
    //
    //      int doThing<T>(vector<T,3> value);
    //      int doThing<T, let N : int>(vector<T,N> value);
    //
    // Here we know that both overloads can apply to `float3`, but only
    // one can apply to `float4`, so the first overload is more
    // specialized/specific.
    //
    // As a final example, a generic which places more constraints
    // on its generic parameters is more specific, all other things
    // being equal:
    //
    //      int doThing<T : IFoo>( T value );
    //      int doThing<T>(T value);
    //
    // In this case we know that the first overload is applicable
    // to a strict subset of the types that the second overload can
    // apply to.
    //
    // The above rules represent the idealized principles we want
    // to implement, but actually implementing that full check here
    // could make overload resolution far more expensive.
    //
    // For now we are going to do something far simpler and hackier,
    // which is to say that a candidate with more generic parameters
    // is always preferred over one with fewer.
    //
    // TODO: We could extend this definition to account for constraints
    // on generic parameters in the count, which would handle the
    // need to prefer a more-constrained generic when possible.
    //
    // TODO: In the long run we should clearly replace this with
    // the more general "does A being applicable imply B being applicable"
    // test.
    //
    // TODO: The principle stated here doesn't take the actual
    // arguments or their types into account, and it might be that
    // in some cases disambiguation of which declaration should be
    // preferred will depend on knowing the actual arguments.
    //
    auto leftSpecCount = getSpecializedParamCount(left.declRef);
    auto rightSpecCount = getSpecializedParamCount(right.declRef);
    if (leftSpecCount != rightSpecCount)
        return int(leftSpecCount - rightSpecCount);

    return 0;
}

int getOverloadRank(DeclRef<Decl> declRef)
{
    if (!declRef.getDecl())
        return 0;
    if (auto attr = declRef.getDecl()->findModifier<OverloadRankAttribute>())
        return attr->rank;
    return 0;
}

int getExportRank(DeclRef<Decl> left, DeclRef<Decl> right)
{
    if (left.getDecl() && left.getDecl()->hasModifier<ExternModifier>())
    {
        return (right.getDecl() && right.getDecl()->hasModifier<HLSLExportModifier>()) ? -1 : 0;
    }
    return 0;
}

int getScopeRank(
    DeclRef<Decl> const& left,
    DeclRef<Decl> const& right,
    Slang::Scope* referenceSiteScope)
{
    if (!referenceSiteScope)
        return 0;

    DeclRef<Decl> prefixDecl = referenceSiteScope->containerDecl;

    // Hold the path from reference site to the root
    // key: Decl node, value: distance from reference site
    Dictionary<Decl*, uint32_t> refPath;
    for (auto node = prefixDecl; node != nullptr; node = node.getParent())
    {
        Decl* key = node.getDecl();
        uint32_t value = (uint32_t)refPath.getCount();
        refPath.add(key, value);
    }

    // find the common prefix decl of reference site and left
    int leftDistance = 0;
    int rightDistance = 0;
    auto distanceToCommonPrefix = [](DeclRef<Decl> const& candidate,
                                     Dictionary<Decl*, uint32_t> refPath) -> int
    {
        uint32_t distanceToReferenceSite = 0;
        uint32_t distanceToCandidate = 0;

        // Sanity check
        if (candidate.getDecl() == nullptr)
            return -1;

        // search from candidate to root, once we found the first node in the reference path, that
        // is the first common prefix, and we can stop searching.
        for (auto node = candidate; node != nullptr; node = node.getParent())
        {
            Decl* key = node.getDecl();
            if (refPath.tryGetValue(key, distanceToReferenceSite))
            {
                break;
            }
            distanceToCandidate++;
        }

        // If we don't find the common prefix, there must be something wrong, return the max value.
        if (distanceToReferenceSite == 0)
            return -1;

        return distanceToReferenceSite + distanceToCandidate;
    };

    leftDistance = distanceToCommonPrefix(left, refPath);
    rightDistance = distanceToCommonPrefix(right, refPath);

    if (leftDistance == rightDistance)
        return 0;

    if (leftDistance == -1)
        return 1;

    if (rightDistance == -1)
        return -1;

    return leftDistance < rightDistance ? -1 : 1;
}

int SemanticsVisitor::CompareOverloadCandidates(OverloadCandidate* left, OverloadCandidate* right)
{
    // If one candidate got further along in validation, pick it
    if (left->status != right->status)
        return int(right->status) - int(left->status);

    // If both candidates are applicable, then we need to compare
    // the costs of their type conversion sequences
    if (left->status == OverloadCandidate::Status::Applicable)
    {
        // If one candidate incurred less cost related to
        // implicit conversion of arguments to matching
        // parameter types, then we should prefer that
        // candidate.
        //
        // TODO: This eventually should be refined into
        // a test that checks conversion cost per-argument,
        // and only considers a candidate "better" if it
        // has lower cost for at least one argument, and
        // does not have higher cost for any.
        //
        if (left->conversionCostSum != right->conversionCostSum)
            return left->conversionCostSum - right->conversionCostSum;

        // If both candidates appear to be equally good when it
        // comes to the per-argument conversions required,
        // then we have two other categories of criteria we
        // can look at to disambiguate things:
        //
        // 1. We can look at how the lookup process found `left` and `right`
        //    do decide which is a better match based purely on how "far away"
        //    they are for lookup purposes. A canonincal example here would
        //    be if one declaration shadows or overrides the other.
        //
        // 2. We can look at parameter lists of `left` and `right`, their types, etc.
        //    do decide which is a better match based purely on structure.
        //    Canonical examples in this case would be preferring a non-generic
        //    candidate over a generic one, preferring a non-variadic candidate
        //    over a variadic one, and preferring a candidate with fewer
        //    default parameters over one with more.
        //
        // Deciding how to order/interleave these two categories of criteria
        // is an important design decision.
        //
        // For example, consider:
        //
        //      float f(float x);
        //
        //      struct S
        //      {
        //          int f<T>(T x);
        //
        //          float g(float y) { return f(y); }
        //      }
        //
        // In terms of structural/type matching, the global `f` is a more specialized
        // candidate at the call site, while in terms of lookup/lexical crieteria
        // the `S.f` declaration is better.
        //
        // For now we are considering lookup/overriding concerns first (so
        // we would bias in favor of selecting `S.f` in the above example), and then
        // structural/type concerns, but a more nuanced approach may be
        // required in the future to better match programmer intuition.
        //
        auto itemDiff = CompareLookupResultItems(left->item, right->item);
        if (itemDiff)
            return itemDiff;

        // If one candidate is an implicit conversion, and other candidate is not,
        // then we should prefer the implicit conversion.
        int leftIsImplicitConversion =
            left->item.declRef.getDecl()->findModifier<ImplicitConversionModifier>() ? 1 : 0;
        int rightIsImplicitConversion =
            right->item.declRef.getDecl()->findModifier<ImplicitConversionModifier>() ? 1 : 0;
        if (leftIsImplicitConversion != rightIsImplicitConversion)
            return rightIsImplicitConversion - leftIsImplicitConversion;

        auto specificityDiff = compareOverloadCandidateSpecificity(left->item, right->item);
        if (specificityDiff)
            return specificityDiff;

        // `export` function is more flavored than `extern` function. But other modifiers are not
        // considered.
        auto externExportDiff = getExportRank(left->item.declRef, right->item.declRef);
        if (externExportDiff)
            return externExportDiff;

        // We need to consider the distance of the declarations to the global scope to resolve this
        // case:
        //      float f(float x);
        //      struct S
        //      {
        //          float f(float x);
        //          float g(float y) { return f(y); }   // will call S::f() instead of ::f()
        //      }
        //  we will count the distance from the reference site to the declaration in the scope tree.

        //  NOTE: We CAN'T do this for the generic function, because generic lookup is little bit
        //  complicated. It will go through multiple passes of candidates compare. In the first
        //  pass, it will lookup all the generic candidates that matches the generic parameter only,
        //  e.g., the following generic functions are totally different, but they will be selected
        //  as candidates because the function name and the generic parameters are the same: void
        //  func<let Z0 : uint, let Z1 : uint>(Z0 a, Z1 b); void func<let Z0 : uint, let Z1 :
        //  uint>(Z0 a, Z1 b, Z0 c); void func<let Z0 : uint, let Z1 : uint>(Z0 a, Z1 b, Z0 c, Z1
        //  d);
        //
        //  So in this case, we should not consider the scope rank and overload rank at all, because
        //  there is only one of above candidates is valid, and the rank calculation doesn't
        //  consider the correctness of the candidates, so it could select the wrong candidate.
        //
        //  In the next pass, the lookup system will match the input parameters in those candidates
        //  to find out the valid match, the "flavor" field will become "Func" or "Expr". So the
        //  rank calculation can be applied.
        if (left->flavor == OverloadCandidate::Flavor::Generic ||
            left->flavor == OverloadCandidate::Flavor::UnspecializedGeneric ||
            right->flavor == OverloadCandidate::Flavor::Generic ||
            right->flavor == OverloadCandidate::Flavor::UnspecializedGeneric)
        {
            return 0;
        }

        auto scopeRank = getScopeRank(left->item.declRef, right->item.declRef, this->m_outerScope);
        if (scopeRank)
            return scopeRank;

        // If we reach here, we will attempt to use overload rank to break the ties.
        auto overloadRankDiff =
            getOverloadRank(right->item.declRef) - getOverloadRank(left->item.declRef);
        if (overloadRankDiff)
            return overloadRankDiff;
    }

    return 0;
}

void SemanticsVisitor::AddOverloadCandidateInner(
    OverloadResolveContext& context,
    OverloadCandidate& candidate)
{
    // Filter our existing candidates, to remove any that are worse than our new one

    bool keepThisCandidate = true; // should this candidate be kept?

    if (context.bestCandidates.getCount() != 0)
    {
        // We have multiple candidates right now, so filter them.
        // This is only used in an assert in debug builds
        [[maybe_unused]] bool anyFiltered = false;
        // Note that we are querying the list length on every iteration,
        // because we might remove things.
        for (Index cc = 0; cc < context.bestCandidates.getCount(); ++cc)
        {
            int cmp = CompareOverloadCandidates(&candidate, &context.bestCandidates[cc]);
            if (cmp < 0)
            {
                // our new candidate is better!

                // remove it from the list (by swapping in a later one)
                context.bestCandidates.fastRemoveAt(cc);
                // and then reduce our index so that we re-visit the same index
                --cc;

                anyFiltered = true;
            }
            else if (cmp > 0)
            {
                // our candidate is worse!
                keepThisCandidate = false;
            }
        }
        // It should not be possible that we removed some existing candidate *and*
        // chose not to keep this candidate (otherwise the better-ness relation
        // isn't transitive). Therefore we confirm that we either chose to keep
        // this candidate (in which case filtering is okay), or we didn't filter
        // anything.
        SLANG_ASSERT(keepThisCandidate || !anyFiltered);
    }
    else if (context.bestCandidate)
    {
        // There's only one candidate so far
        int cmp = CompareOverloadCandidates(&candidate, context.bestCandidate);
        if (cmp < 0)
        {
            // our new candidate is better!
            context.bestCandidate = nullptr;
        }
        else if (cmp > 0)
        {
            // our candidate is worse!
            keepThisCandidate = false;
        }
    }

    // If our candidate isn't good enough, then drop it
    if (!keepThisCandidate)
        return;

    // Otherwise we want to keep the candidate
    if (context.bestCandidates.getCount() > 0)
    {
        // There were already multiple candidates, and we are adding one more
        context.bestCandidates.add(candidate);
    }
    else if (context.bestCandidate)
    {
        // There was a unique best candidate, but now we are ambiguous
        context.bestCandidates.add(*context.bestCandidate);
        context.bestCandidates.add(candidate);
        context.bestCandidate = nullptr;
    }
    else
    {
        // This is the only candidate worth keeping track of right now
        context.bestCandidateStorage = candidate;
        context.bestCandidate = &context.bestCandidateStorage;
    }
}

void SemanticsVisitor::AddOverloadCandidate(
    OverloadResolveContext& context,
    OverloadCandidate& candidate,
    ConversionCost baseCost)
{
    // Try the candidate out, to see if it is applicable at all.
    TryCheckOverloadCandidate(context, candidate);

    candidate.conversionCostSum += baseCost;

    // Now (potentially) add it to the set of candidate overloads to consider.
    AddOverloadCandidateInner(context, candidate);
}

void SemanticsVisitor::AddFuncOverloadCandidate(
    LookupResultItem item,
    DeclRef<CallableDecl> funcDeclRef,
    OverloadResolveContext& context,
    ConversionCost baseCost)
{
    auto funcDecl = funcDeclRef.getDecl();
    ensureDecl(funcDecl, DeclCheckState::CanUseFuncSignature);

    // If this function is a redeclaration,
    // then we don't want to include it multiple times,
    // and mistakenly think we have an ambiguous call.
    //
    // Instead, we will carefully consider only the
    // "primary" declaration of any callable.
    if (auto primaryDecl = funcDecl->primaryDecl)
    {
        if (funcDecl != primaryDecl)
        {
            // This is a redeclaration, so we don't
            // want to consider it. The primary
            // declaration should also get considered
            // for the call site and it will match
            // anything this declaration would have
            // matched.
            return;
        }
    }

    OverloadCandidate candidate;
    candidate.flavor = OverloadCandidate::Flavor::Func;
    candidate.item = item;
    candidate.resultType = getResultType(m_astBuilder, funcDeclRef);

    AddOverloadCandidate(context, candidate, baseCost);
}

void SemanticsVisitor::AddFuncOverloadCandidate(
    FuncType* funcType,
    OverloadResolveContext& context,
    ConversionCost baseCost)
{
    OverloadCandidate candidate;
    candidate.flavor = OverloadCandidate::Flavor::Expr;
    candidate.funcType = funcType;
    candidate.resultType = funcType->getResultType();

    AddOverloadCandidate(context, candidate, baseCost);
}

void SemanticsVisitor::AddFuncExprOverloadCandidate(
    FuncType* funcType,
    OverloadResolveContext& context,
    Expr* expr,
    ConversionCost baseCost)
{
    SLANG_ASSERT(expr);
    OverloadCandidate candidate;
    candidate.flavor = OverloadCandidate::Flavor::Expr;
    candidate.funcType = funcType;
    candidate.resultType = funcType->getResultType();
    candidate.exprVal = expr;

    AddOverloadCandidate(context, candidate, baseCost);
}

void SemanticsVisitor::AddCtorOverloadCandidate(
    LookupResultItem typeItem,
    Type* type,
    DeclRef<ConstructorDecl> ctorDeclRef,
    OverloadResolveContext& context,
    Type* resultType,
    ConversionCost baseCost)
{
    SLANG_UNUSED(type)

    ensureDecl(ctorDeclRef, DeclCheckState::CanUseFuncSignature);

    // `typeItem` refers to the type being constructed (the thing
    // that was applied as a function) so we need to construct
    // a `LookupResultItem` that refers to the constructor instead

    LookupResultItem ctorItem;
    ctorItem.declRef = ctorDeclRef;
    ctorItem.breadcrumbs = new LookupResultItem::Breadcrumb(
        LookupResultItem::Breadcrumb::Kind::Member,
        typeItem.declRef,
        nullptr,
        typeItem.breadcrumbs);

    OverloadCandidate candidate;
    candidate.flavor = OverloadCandidate::Flavor::Func;
    candidate.item = ctorItem;
    candidate.resultType = resultType;

    AddOverloadCandidate(context, candidate, baseCost);
}

bool SemanticsVisitor::OverloadResolveContext::matchArgumentsToParams(
    SemanticsVisitor* semantics,
    const List<QualType>& params,
    bool computeTypes,
    ShortList<MatchedArg>& outMatchedArgs)
{
    // We allow params to end with one or more variadic packs.
    // We will first find out how many type packs there are.
    Index typePackCount = 0;
    for (Index i = params.getCount() - 1; i >= 0; --i)
    {
        if (isTypePack(params[i].type))
            typePackCount++;
        else
            break;
    }
    auto fixedParamCount = params.getCount() - typePackCount;

    auto remainingArgCount = getArgCount() - fixedParamCount;

    // If there are remaining arguments after matching all fixed parameters,
    // we'd better have at least one type pack.
    if (remainingArgCount > 0 && typePackCount == 0)
        return false;

    // Now we can match the arguments to the parameters.

    // The fixed part comes first.
    for (Index i = 0; i < Math::Min(getArgCount(), fixedParamCount); ++i)
    {
        MatchedArg arg;
        arg.argExpr = getArg(i);
        arg.argType = getArgType(i);
        outMatchedArgs.add(arg);
    }

    // Try to match the variadic part.
    // Is the corresponding argument a expand expr? If so it will map 1:1 to the type pack param.
    auto astBuilder = semantics->getASTBuilder();

    if (remainingArgCount <= 0)
        return true;
    if (typePackCount == 0)
        return false;

    // If the number of type packs can't evenly divide the remaining arguments,
    // there isn't a match.
    if (remainingArgCount % typePackCount != 0)
        return false;

    // The default case is to group the remaining arguments into evenly divided PackExprs.
    Index typePackSize = remainingArgCount / typePackCount;
    for (Index i = 0; i < typePackCount; ++i)
    {
        // If type pack size is 1, we may not need to wrap things in a PackExpr,
        // if the argument is already a pack.
        if (typePackSize == 1)
        {
            auto argType = getArgType(fixedParamCount + i);
            if (auto typeType = as<TypeType>(argType))
            {
                argType = typeType->getType();
            }
            if (isTypePack(argType))
            {
                MatchedArg arg;
                arg.argExpr = getArg(fixedParamCount + i);
                arg.argType = getArgType(fixedParamCount + i);
                outMatchedArgs.add(arg);
                continue;
            }
        }
        PackExpr* packExpr = nullptr;
        if (mode == Mode::ForReal)
        {
            packExpr = astBuilder->create<PackExpr>();
            packExpr->loc = loc;
        }
        ShortList<Type*> types;
        for (Index j = 0; j < typePackSize; ++j)
        {
            if (packExpr)
            {
                auto arg = getArg(fixedParamCount + i * typePackSize + j);
                packExpr->args.add(arg);
            }
            if (computeTypes)
                types.add(
                    getArgTypeForInference(fixedParamCount + i * typePackSize + j, semantics));
        }
        MatchedArg matchedArg;
        matchedArg.argExpr = packExpr;
        if (computeTypes)
        {
            matchedArg.argType = astBuilder->getTypePack(types.getArrayView().arrayView);
            if (packExpr)
                packExpr->type = matchedArg.argType;
        }
        outMatchedArgs.add(matchedArg);
    }
    return true;
}

DeclRef<Decl> SemanticsVisitor::inferGenericArguments(
    DeclRef<GenericDecl> genericDeclRef,
    OverloadResolveContext& context,
    ArrayView<Val*> knownGenericArgs,
    ConversionCost& outBaseCost,
    List<QualType>* innerParameterTypes)
{
    // We have been asked to infer zero or more arguments to
    // `genericDeclRef`, in a context where it is being applied
    // to value-level arguments in `context`.
    //
    // It is possible that the call site included one or more
    // explicit arguments, in which case `substWithKnownGenericArgs`
    // will have been filled in and contain those. Otherwise,
    // that parameter will be null, and we are expected to
    // infer all arguments.

    // The declaration of the generic must be checked up to a point
    // where we can attempt to form specializations of it (which in
    // practice means that the declarations of its parameters and
    // their constraints must have been checked).
    //
    ensureDecl(genericDeclRef, DeclCheckState::CanSpecializeGeneric);

    // Conceptually, we are going to be trying to infer any unspecified
    // generic arguments by forming a system of constraints on those arguments
    // and then attempting to solve the constraint system.
    //
    // While the constraint solver we have implemented today is not especially
    // clever, we follow a flow that should in principle allow us to plug in
    // something more clever down the line.
    //
    ConstraintSystem constraints;
    constraints.loc = context.loc;
    constraints.genericDecl = genericDeclRef.getDecl();

    // In order to perform matching between the types passed in at the
    // call site represented by `context` and the parameters of the
    // declaraiton being applied, we want to form a reference to
    // the "inner" declaration of the generic (e.g., the `FuncitonDecl`
    // under the `GenericDecl`).
    //
    // Check what type of declaration we are dealing with, and then try
    // to match it up with the arguments accordingly...

    if (auto funcDeclRef = as<CallableDecl>(genericDeclRef.getDecl()->inner))
    {
        List<QualType> paramTypes;
        if (!innerParameterTypes)
        {
            auto params = getParameters(m_astBuilder, funcDeclRef).toArray();
            for (auto param : params)
            {
                paramTypes.add(getParamQualType(m_astBuilder, param));
            }
            innerParameterTypes = &paramTypes;
        }

        ShortList<OverloadResolveContext::MatchedArg> matchedArgs;

        // We now try to match arguments to parameters.
        //
        // Note that if there are *too few* arguments, we might still have
        // a match, because the other arguments might have default values
        // that can be used.
        //
        if (!context.matchArgumentsToParams(this, *innerParameterTypes, true, matchedArgs))
        {
            return DeclRef<Decl>();
        }

        // Perform type unification between arguments and parameters, so
        // we can populate the resolve system with inital constraints.
        //
        for (Index aa = 0; aa < matchedArgs.getCount(); ++aa)
        {
            // The question here is whether failure to "unify" an argument
            // and parameter should lead to immediate failure.
            //
            // The case that is interesting is if we want to unify, say:
            // `vector<float,N>` and `vector<int,3>`
            //
            // It is clear that we should solve with `N = 3`, and then
            // a later step may find that the resulting types aren't
            // actually a match.
            //
            // A more refined approach to "unification" could of course
            // see that `int` can convert to `float` and use that fact.
            // (and indeed we already use something like this to unify
            // `float` and `vector<T,3>`)
            //
            // So the question is then whether a mismatch during the
            // unification step should be taken as an immediate failure...
            auto argType = matchedArgs[aa].argType;
            auto paramType = (*innerParameterTypes)[aa];
            auto canUnify = TryUnifyTypes(
                constraints,
                ValUnificationContext(),
                QualType(argType, paramType.isLeftValue),
                paramType);

            // It is an error if we can't unify the argument with a type pack parameter.
            if (!canUnify && isTypePack(paramType))
            {
                return DeclRef<Decl>();
            }
        }
    }
    else
    {
        // TODO(tfoley): any other cases needed here?
        return DeclRef<Decl>();
    }

    // Once we have added all the appropriate constraints to the system, we
    // will try to solve for a set of arguments to the generic that satisfy
    // those constraints.
    //
    // Note that this step *also* attempts to infer arguments for all the
    // implicit parameters of a generic. Notably, this means inferring
    // witnesses for interface conformance constraints.
    //
    // TODO(tfoley): We probably need to pass along the explicit arguments here,
    // so that the solver knows to accept those arguments as-is.
    //
    return trySolveConstraintSystem(&constraints, genericDeclRef, knownGenericArgs, outBaseCost);
}

void SemanticsVisitor::AddTypeOverloadCandidates(Type* type, OverloadResolveContext& context)
{
    // The code being checked is trying to apply `type` like a function.
    // Semantically, the operations `T(args...)` is equivalent to
    // `T.__init(args...)` if we had a surface syntax that supported
    // looking up `__init` declarations by that name.
    //
    // Internally, all `__init` declarations are stored with the name
    // `$init`, to avoid potential conflicts if a user decided to name
    // a field/method `__init`.
    //
    // We will look up all the initializers on `type` by looking up
    // its members named `$init`, and then proceed to perform overload
    // resolution with what we find.
    //
    // TODO: One wrinkle here is single-argument constructor syntax.
    // An operation like `(T) oneArg` or `T(oneArg)` is currently
    // treated as a call expression, but we might want such cases
    // to go through the type coercion logic first/instead, because
    // by doing so we could weed out cases where a type is "constructed"
    // from a value of the same type. There is no need in Slang for
    // "copy constructors" but the core module currently has to define
    // some just to make code that does, e.g., `float(1.0f)` work.)
    LookupOptions options =
        LookupOptions(uint8_t(LookupOptions::IgnoreInheritance) | uint8_t(LookupOptions::NoDeref));
    LookupResult initializers = lookUpMember(
        m_astBuilder,
        this,
        getName("$init"),
        type,
        context.sourceScope,
        LookupMask::Default,
        options);
    AddOverloadCandidates(initializers, context);
}

void SemanticsVisitor::addOverloadCandidatesForCallToGeneric(
    LookupResultItem genericItem,
    OverloadResolveContext& context,
    ArrayView<Val*> knownGenericArgs)
{
    auto genericDeclRef = genericItem.declRef.as<GenericDecl>();
    SLANG_ASSERT(genericDeclRef);

    ConversionCost baseCost = kConversionCost_None;

    // Try to infer generic arguments, based on the context
    DeclRef<Decl> innerRef =
        inferGenericArguments(genericDeclRef, context, knownGenericArgs, baseCost);

    if (innerRef)
    {
        // If inference works, then we've now got a
        // specialized declaration reference we can apply.

        LookupResultItem innerItem;
        innerItem.breadcrumbs = genericItem.breadcrumbs;
        innerItem.declRef = innerRef;
        AddDeclRefOverloadCandidates(innerItem, context, baseCost);
    }
    else
    {
        // If inference failed, then we need to create
        // a candidate that can be used to reflect that fact
        // (so we can report a good error)
        OverloadCandidate candidate;
        candidate.item = genericItem;
        candidate.flavor = OverloadCandidate::Flavor::UnspecializedGeneric;
        candidate.status = OverloadCandidate::Status::GenericArgumentInferenceFailed;

        AddOverloadCandidateInner(context, candidate);
    }
}

void SemanticsVisitor::AddDeclRefOverloadCandidates(
    LookupResultItem item,
    OverloadResolveContext& context,
    ConversionCost baseCost)
{
    if (auto funcDeclRef = item.declRef.as<CallableDecl>())
    {
        AddFuncOverloadCandidate(item, funcDeclRef, context, baseCost);
    }
    else if (auto aggTypeDeclRef = item.declRef.as<AggTypeDecl>())
    {
        auto type = DeclRefType::create(m_astBuilder, aggTypeDeclRef);
        AddTypeOverloadCandidates(type, context);
    }
    else if (auto genericDeclRef = item.declRef.as<GenericDecl>())
    {
        LookupResultItem innerItem;
        innerItem.breadcrumbs = item.breadcrumbs;
        innerItem.declRef = genericDeclRef;
        addOverloadCandidatesForCallToGeneric(innerItem, context, ArrayView<Val*>());
    }
    else if (auto typeDefDeclRef = item.declRef.as<TypeDefDecl>())
    {
        auto type = getNamedType(m_astBuilder, typeDefDeclRef);
        AddTypeOverloadCandidates(type, context);
    }
    else if (auto genericTypeParamDeclRef = item.declRef.as<GenericTypeParamDecl>())
    {
        auto type = DeclRefType::create(m_astBuilder, genericTypeParamDeclRef);
        AddTypeOverloadCandidates(type, context);
    }
    else if (auto localDeclRef = item.declRef.as<ParamDecl>())
    {
        // We could probably be broader than just parameters here
        // eventually.
        // Limit it for now though to make the specialization easier
        // TODO: why can't this use DeclCheckState::CanUseFuncSignature
        ensureDecl(localDeclRef, DeclCheckState::TypesFullyResolved);
        const auto type = localDeclRef.getDecl()->getType();
        // We can only add overload candidates if this is known to be a function
        if (const auto funType = as<FuncType>(type))
            AddFuncExprOverloadCandidate(
                funType,
                context,
                context.originalExpr->functionExpr,
                baseCost);
        else
            return;
    }
    else
    {
        // TODO(tfoley): any other cases needed here?
        return;
    }
}

void SemanticsVisitor::AddOverloadCandidates(
    LookupResult const& result,
    OverloadResolveContext& context)
{
    if (result.isOverloaded())
    {
        for (auto item : result.items)
        {
            AddDeclRefOverloadCandidates(item, context, kConversionCost_None);
        }
    }
    else
    {
        AddDeclRefOverloadCandidates(result.item, context, kConversionCost_None);
    }
}

void SemanticsVisitor::AddOverloadCandidates(Expr* funcExpr, OverloadResolveContext& context)
{
    // A call of the form `(<something>)(<args>)` should be
    // resolved as if the user wrote `<something>(<args>)`,
    // so that we avoid introducing intermediate expressions
    // of function type in cases where they are not needed.
    //
    while (auto parenExpr = as<ParenExpr>(funcExpr))
    {
        funcExpr = parenExpr->base;
    }

    auto funcExprType = funcExpr->type;

    if (auto declRefExpr = as<DeclRefExpr>(funcExpr))
    {
        // The expression directly referenced a declaration,
        // so we can use that declaration directly to look
        // for anything applicable.
        AddDeclRefOverloadCandidates(
            LookupResultItem(declRefExpr->declRef),
            context,
            kConversionCost_None);
    }
    else if (auto higherOrderExpr = as<HigherOrderInvokeExpr>(funcExpr))
    {
        // The expression is the result of a higher order function application.
        AddHigherOrderOverloadCandidates(higherOrderExpr, context, kConversionCost_None);
    }
    else if (auto funcType = as<FuncType>(funcExprType))
    {
        // TODO(tfoley): deprecate this path...
        AddFuncOverloadCandidate(funcType, context, kConversionCost_None);
    }
    else if (auto overloadedExpr = as<OverloadedExpr>(funcExpr))
    {
        AddOverloadCandidates(overloadedExpr->lookupResult2, context);
    }
    else if (auto overloadedExpr2 = as<OverloadedExpr2>(funcExpr))
    {
        for (auto item : overloadedExpr2->candidiateExprs)
        {
            AddOverloadCandidates(item, context);
        }
    }
    else if (auto partiallyAppliedGenericExpr = as<PartiallyAppliedGenericExpr>(funcExpr))
    {
        // A partially-applied generic is allowed as an overload candidate,
        // and carries along an (incomplete) substitution that can be used
        // to carry the arguments known so far.
        //
        addOverloadCandidatesForCallToGeneric(
            LookupResultItem(partiallyAppliedGenericExpr->baseGenericDeclRef),
            context,
            partiallyAppliedGenericExpr->knownGenericArgs.getArrayView());
    }
    else if (auto typeType = as<TypeType>(funcExprType))
    {
        // If none of the above cases matched, but we are
        // looking at a type, then I suppose we have
        // a constructor call on our hands.
        //
        // TODO(tfoley): are there any meaningful types left
        // that aren't declaration references?
        auto type = typeType->getType();
        AddTypeOverloadCandidates(type, context);
        return;
    }
}

void SemanticsVisitor::AddHigherOrderOverloadCandidates(
    Expr* funcExpr,
    OverloadResolveContext& context,
    ConversionCost baseCost)
{
    // Lookup the higher order function and process types accordingly. In the future,
    // if there are enough varieties, we can have dispatch logic instead of an
    // if-else ladder.
    if (auto expr = as<HigherOrderInvokeExpr>(funcExpr))
    {
        auto funcDeclRefExpr =
            as<DeclRefExpr>(getInnerMostExprFromHigherOrderExpr(expr->baseFunction));
        if (!funcDeclRefExpr)
            return;
        if (auto baseFuncDeclRef = funcDeclRefExpr->declRef.as<CallableDecl>())
        {
            // Base is a normal or fully specialized generic function.
            OverloadCandidate candidate;
            candidate.flavor = OverloadCandidate::Flavor::Expr;
            if (auto diffExpr = as<HigherOrderInvokeExpr>(expr))
            {
                candidate.funcType = as<FuncType>(diffExpr->type.type);
            }
            candidate.resultType = candidate.funcType->getResultType();
            candidate.item = LookupResultItem(baseFuncDeclRef);
            candidate.exprVal = expr;
            AddOverloadCandidate(context, candidate, baseCost);
        }
        else if (auto baseFuncGenericDeclRef = funcDeclRefExpr->declRef.as<GenericDecl>())
        {
            // Process func type to generate JVP func type.
            auto diffFuncType = as<FuncType>(expr->type.type);
            SLANG_ASSERT(diffFuncType);

            // Extract parameter list from processed type.
            List<QualType> paramTypes;

            for (Index ii = 0; ii < diffFuncType->getParamCount(); ii++)
                paramTypes.add(getParamQualType(diffFuncType->getParamType(ii)));

            // Try to infer generic arguments, based on the updated context.
            OverloadResolveContext subContext = context;
            ConversionCost baseCost1 = kConversionCost_None;
            DeclRef<Decl> innerRef = inferGenericArguments(
                baseFuncGenericDeclRef,
                context,
                ArrayView<Val*>(),
                baseCost1,
                &paramTypes);

            if (!innerRef)
                return;

            OverloadCandidate candidate;
            candidate.flavor = OverloadCandidate::Flavor::Expr;
            if (innerRef)
            {
                diffFuncType = as<FuncType>(innerRef.substitute(m_astBuilder, diffFuncType));
                candidate.item = LookupResultItem(innerRef);
            }
            else
            {
                candidate.item = LookupResultItem(funcDeclRefExpr->declRef);
            }
            candidate.funcType = as<FuncType>(diffFuncType);
            candidate.resultType = candidate.funcType->getResultType();

            // Substitute all types in the high-order expression chain.
            Expr* inner = expr;
            HigherOrderInvokeExpr* lastInner = nullptr;
            while (auto hoInner = as<HigherOrderInvokeExpr>(inner))
            {
                lastInner = hoInner;
                if (innerRef)
                    hoInner->type = innerRef.substitute(m_astBuilder, hoInner->type.type);
                inner = hoInner->baseFunction;
            }
            // Set inner expression to resolved declref expr.
            if (lastInner)
            {
                auto baseExpr = GetBaseExpr(funcDeclRefExpr);
                lastInner->baseFunction = ConstructLookupResultExpr(
                    candidate.item,
                    baseExpr,
                    funcDeclRefExpr->name,
                    funcDeclRefExpr->loc,
                    funcDeclRefExpr);
            }
            candidate.exprVal = expr;
            expr->type.type = diffFuncType;
            AddOverloadCandidate(context, candidate, baseCost + baseCost1);
        }
        else
        {
            // Unhandled case for the inner expr.
            getSink()->diagnose(funcExpr->loc, Diagnostics::expectedFunction, funcExpr->type);
            funcExpr->type = this->getASTBuilder()->getErrorType();
        }
    }
}

String SemanticsVisitor::getCallSignatureString(OverloadResolveContext& context)
{
    StringBuilder argsListBuilder;
    argsListBuilder << "(";

    UInt argCount = context.getArgCount();
    for (UInt aa = 0; aa < argCount; ++aa)
    {
        if (aa != 0)
            argsListBuilder << ", ";
        auto argType = context.getArgType(aa);
        if (argType)
            context.getArgType(aa)->toText(argsListBuilder);
        else
            argsListBuilder << "error";
    }
    argsListBuilder << ")";
    return argsListBuilder.produceString();
}

Expr* SemanticsVisitor::ResolveInvoke(InvokeExpr* expr)
{
    OverloadResolveContext context;

    // Look at the base expression for the call, and figure out how to invoke it.
    auto funcExpr = expr->functionExpr;

    // If we are trying to apply an erroneous expression, then just bail out now.
    if (IsErrorExpr(funcExpr))
    {
        return CreateErrorExpr(expr);
    }
    // If any of the arguments is an error, then we should bail out, to avoid
    // cascading errors where we successfully pick an overload, but not the one
    // the user meant.
    for (auto arg : expr->arguments)
    {
        if (IsErrorExpr(arg))
            return CreateErrorExpr(expr);

        // If this argument is itself an overloaded value without a type
        // then we can't sensibly continue
        if (!arg->type && (as<OverloadedExpr>(arg) || as<OverloadedExpr2>(arg)))
        {
            getSink()->diagnose(expr->loc, Diagnostics::overloadedParameterToHigherOrderFunction);
            return CreateErrorExpr(expr);
        }
    }

    for (auto& arg : expr->arguments)
    {
        arg = maybeOpenRef(arg);
        arg = maybeOpenExistential(arg);
    }

    context.originalExpr = expr;
    context.funcLoc = funcExpr->loc;
    context.argCount = expr->arguments.getCount();
    context.args = &expr->arguments;
    context.loc = expr->loc;
    context.sourceScope = m_outerScope;
    context.baseExpr = GetBaseExpr(funcExpr);

    // check if this is a core module operator call, if so we want to use cached results
    // to speed up compilation
    bool shouldAddToCache = false;
    OperatorOverloadCacheKey key;
    TypeCheckingCache* typeCheckingCache = getLinkage()->getTypeCheckingCache();
    if (auto opExpr = as<OperatorExpr>(expr))
    {
        if (key.fromOperatorExpr(opExpr))
        {
            key.isGLSLMode = getShared()->glslModuleDecl != nullptr;
            ResolvedOperatorOverload candidate;
            if (typeCheckingCache->resolvedOperatorOverloadCache.tryGetValue(key, candidate))
            {
                // We should only use the cached candidate if it is persistent direct declref
                // created from GlobalSession's ASTBuilder, or it is created in the current Linkage.
                if (candidate.cacheVersion == typeCheckingCache->version ||
                    findNextOuterGeneric(candidate.decl) == nullptr)
                {
                    context.bestCandidateStorage = candidate.candidate;
                    context.bestCandidate = &context.bestCandidateStorage;
                }
                else
                {
                    LookupResultItem overloadCandidate = {};
                    overloadCandidate.declRef = getOuterGenericOrSelf(candidate.decl);
                    AddDeclRefOverloadCandidates(overloadCandidate, context, 0);
                    shouldAddToCache = true;
                }
            }
            else
            {
                shouldAddToCache = true;
            }
        }
    }

    // We run a special case here where an `InvokeExpr`
    // with a single argument where the base/func expression names
    // a type should always be treated as an explicit type coercion
    // (and hence bottleneck through `coerce()`) instead of just
    // as a constructor call.
    //
    // Such a special-case would help us handle cases of identity
    // casts (casting an expression to the type it already has),
    // without needing dummy initializer/constructor declarations.
    //
    // Handling that special casing here (rather than in, say,
    // that `(T) expr` and `T(expr)` continue to be semantically
    // `visitTypeCastExpr`) would allow us to continue to ensure
    // equivalent in (almost) all cases.
    // If callee is a type, and we are calling with one argument, then treat it as a
    // type coercion.
    bool typeOverloadChecked = false;

    if (expr->arguments.getCount() == 1 && !as<ExplicitCtorInvokeExpr>(expr))
    {
        if (const auto typeType = as<TypeType>(funcExpr->type))
        {
            if (isDeclRefTypeOf<AggTypeDeclBase>(typeType->getType()))
            {
                Expr* resultExpr = nullptr;
                DiagnosticSink tempSink(getSourceManager(), nullptr);
                ConversionCost conversionCost = kConversionCost_None;
                auto coerceResult = SemanticsVisitor(withSink(&tempSink))
                                        ._coerce(
                                            CoercionSite::ExplicitCoercion,
                                            typeType->getType(),
                                            &resultExpr,
                                            expr->arguments[0]->type,
                                            expr->arguments[0],
                                            &conversionCost);
                if (coerceResult)
                    return resultExpr;
                typeOverloadChecked = true;
            }
        }
    }
    if (!context.bestCandidate && !typeOverloadChecked)
    {
        AddOverloadCandidates(funcExpr, context);
    }

    if (context.bestCandidates.getCount() > 0)
    {
        // Things were ambiguous.

        // It might be that things were only ambiguous because
        // one of the argument expressions had an error, and
        // so a bunch of candidates could match at that position.
        //
        // If any argument was an error, we skip out on printing
        // another message, to avoid cascading errors.
        for (auto arg : expr->arguments)
        {
            if (IsErrorExpr(arg))
            {
                return CreateErrorExpr(expr);
            }
        }

        Name* funcName = nullptr;
        {
            Expr* baseExpr = funcExpr;

            if (auto baseGenericApp = as<GenericAppExpr>(baseExpr))
                baseExpr = baseGenericApp->functionExpr;

            if (auto baseVar = as<VarExpr>(baseExpr))
                funcName = baseVar->name;
            else if (auto baseMemberRef = as<MemberExpr>(baseExpr))
                funcName = baseMemberRef->name;
            else if (auto baseOverloaded = as<OverloadedExpr>(baseExpr))
                funcName = baseOverloaded->name;
        }

        String argsList = getCallSignatureString(context);

        if (context.bestCandidates[0].status != OverloadCandidate::Status::Applicable)
        {
            // There were multiple equally-good candidates, but none actually usable.
            // We will construct a diagnostic message to help out.

            if (funcName)
            {
                getSink()->diagnose(
                    expr,
                    Diagnostics::noApplicableOverloadForNameWithArgs,
                    funcName,
                    argsList);
            }
            else
            {
                getSink()->diagnose(expr, Diagnostics::noApplicableWithArgs, argsList);
            }
        }
        else
        {
            // There were multiple applicable candidates, so we need to report them.

            if (funcName)
            {
                getSink()->diagnose(
                    expr,
                    Diagnostics::ambiguousOverloadForNameWithArgs,
                    funcName,
                    argsList);
            }
            else
            {
                getSink()->diagnose(expr, Diagnostics::ambiguousOverloadWithArgs, argsList);
            }
        }

        {
            Index candidateCount = context.bestCandidates.getCount();
            Index maxCandidatesToPrint = 10; // don't show too many candidates at once...
            Index candidateIndex = 0;
            context.bestCandidates.sort([](const OverloadCandidate& c1, const OverloadCandidate& c2)
                                        { return c1.status < c2.status; });

            for (auto candidate : context.bestCandidates)
            {
                String declString =
                    ASTPrinter::getDeclSignatureString(candidate.item, m_astBuilder);

                if (candidate.status == OverloadCandidate::Status::VisibilityChecked)
                    getSink()->diagnose(
                        candidate.item.declRef,
                        Diagnostics::invisibleOverloadCandidate,
                        declString);
                else
                    getSink()->diagnose(
                        candidate.item.declRef,
                        Diagnostics::overloadCandidate,
                        declString);

                candidateIndex++;
                if (candidateIndex == maxCandidatesToPrint)
                    break;
            }
            if (candidateIndex != candidateCount)
            {
                getSink()->diagnose(
                    expr,
                    Diagnostics::moreOverloadCandidates,
                    candidateCount - candidateIndex);
            }
        }

        return CreateErrorExpr(expr);
    }
    else if (context.bestCandidate)
    {
        // There was one best candidate, even if it might not have been
        // applicable in the end.
        // We will report errors for this one candidate, then, to give
        // the user the most help we can.
        if (shouldAddToCache)
        {
            if (isFromCoreModule(context.bestCandidate->item.declRef.getDecl()) ||
                getShared()->glslModuleDecl ==
                    getModuleDecl(context.bestCandidate->item.declRef.getDecl()))
            {
                ResolvedOperatorOverload overloadResult;
                overloadResult.candidate = *context.bestCandidate;
                overloadResult.decl = context.bestCandidate->item.declRef.getDecl();
                overloadResult.cacheVersion = typeCheckingCache->version;
                typeCheckingCache->resolvedOperatorOverloadCache[key] = overloadResult;
            }
        }

        // Now that we have resolved the overload candidate, we need to undo an `openExistential`
        // operation that was applied to `out` arguments.
        //
        auto funcType = context.bestCandidate->funcType;
        ShortList<ParameterDirection> paramDirections;
        if (funcType)
        {
            for (Index i = 0; i < funcType->getParamCount(); i++)
            {
                paramDirections.add(funcType->getParamDirection(i));
            }
        }
        else if (auto callableDeclRef = context.bestCandidate->item.declRef.as<CallableDecl>())
        {
            for (auto param : callableDeclRef.getDecl()->getParameters())
            {
                paramDirections.add(getParameterDirection(param));
            }
        }
        for (Index i = 0; i < expr->arguments.getCount(); i++)
        {
            auto& arg = expr->arguments[i];
            if (i < paramDirections.getCount())
            {
                switch (paramDirections[i])
                {
                case kParameterDirection_Out:
                case kParameterDirection_InOut:
                case kParameterDirection_Ref:
                case kParameterDirection_ConstRef:
                    break;
                default:
                    continue;
                }
            }
            if (auto extractExistentialExpr = as<ExtractExistentialValueExpr>(arg))
                arg = extractExistentialExpr->originalExpr;
        }
        return CompleteOverloadCandidate(context, *context.bestCandidate);
    }

    // If absolutely no viable candidates were extracted from the overloaded expression,
    // we may be dealing with a composite type or an overloaded expression with composite types.
    //

    auto typeExpr = funcExpr;
    if (auto overloadedExpr = as<OverloadedExpr>(funcExpr))
    {
        if (overloadedExpr->lookupResult2.isValid() && overloadedExpr->lookupResult2.isOverloaded())
        {
            typeExpr = maybeResolveOverloadedExpr(overloadedExpr, LookupMask::type, nullptr);
        }
    }

    if (auto typetype = as<TypeType>(typeExpr->type))
    {
        // We allow a special case when `funcExpr` represents a composite type,
        // in which case we will try to construct the type via memberwise assignment from the
        // arguments.
        //
        auto initListExpr = m_astBuilder->create<InitializerListExpr>();
        initListExpr->loc = expr->loc;
        initListExpr->args.addRange(expr->arguments);
        initListExpr->type = m_astBuilder->getInitializerListType();
        Expr* outExpr = nullptr;
        if (_coerceInitializerList(typetype->getType(), &outExpr, initListExpr))
            return outExpr;
    }

    // Nothing at all was found that we could even consider invoking.
    // In all other cases, this is an error.
    getSink()->diagnose(expr->functionExpr, Diagnostics::expectedFunction, funcExpr->type);
    expr->type = QualType(m_astBuilder->getErrorType());
    return expr;
}

void SemanticsVisitor::AddGenericOverloadCandidate(
    LookupResultItem baseItem,
    OverloadResolveContext& context)
{
    if (auto genericDeclRef = baseItem.declRef.as<GenericDecl>())
    {
        ensureDecl(genericDeclRef, DeclCheckState::CanSpecializeGeneric);

        OverloadCandidate candidate;
        candidate.flavor = OverloadCandidate::Flavor::Generic;
        candidate.item = baseItem;
        candidate.resultType = nullptr;

        AddOverloadCandidate(context, candidate, kConversionCost_None);
    }
}

void SemanticsVisitor::AddGenericOverloadCandidates(Expr* baseExpr, OverloadResolveContext& context)
{
    if (auto baseDeclRefExpr = as<DeclRefExpr>(baseExpr))
    {
        auto declRef = baseDeclRefExpr->declRef;
        AddGenericOverloadCandidate(LookupResultItem(declRef), context);
    }
    else if (auto overloadedExpr = as<OverloadedExpr>(baseExpr))
    {
        // We are referring to a bunch of declarations, each of which might be generic
        for (auto item : overloadedExpr->lookupResult2)
        {
            AddGenericOverloadCandidate(item, context);
        }
    }
    else
    {
        // any other cases?
    }
}

Expr* SemanticsExprVisitor::visitGenericAppExpr(GenericAppExpr* genericAppExpr)
{
    // Start by checking the base expression and arguments.

    // Disable the short-circuiting logic expression when the experssion is in
    // the generic parameter.
    if (this->m_shouldShortCircuitLogicExpr)
    {
        auto subContext = disableShortCircuitLogicalExpr();
        return dispatchExpr(genericAppExpr, subContext);
    }

    auto& baseExpr = genericAppExpr->functionExpr;
    baseExpr = CheckTerm(baseExpr);
    auto& args = genericAppExpr->arguments;
    for (auto& arg : args)
    {
        arg = CheckTerm(arg);
    }

    return checkGenericAppWithCheckedArgs(genericAppExpr);
}

/// Check a generic application where the operands have already been checked.
Expr* SemanticsVisitor::checkGenericAppWithCheckedArgs(GenericAppExpr* genericAppExpr)
{
    // We are applying a generic to arguments, but there might be multiple generic
    // declarations with the same name, so this becomes a specialized case of
    // overload resolution.

    auto& baseExpr = genericAppExpr->functionExpr;
    auto& args = genericAppExpr->arguments;

    // If there was an error in the base expression,  or in any of
    // the arguments, then just bail.
    if (IsErrorExpr(baseExpr))
    {
        return CreateErrorExpr(genericAppExpr);
    }
    for (auto argExpr : args)
    {
        if (IsErrorExpr(argExpr))
        {
            return CreateErrorExpr(genericAppExpr);
        }
    }

    // Otherwise, let's start looking at how to find an overload...

    OverloadResolveContext context;
    context.originalExpr = genericAppExpr;
    context.funcLoc = baseExpr->loc;
    context.argCount = args.getCount();
    context.args = &args;
    context.loc = genericAppExpr->loc;
    context.sourceScope = m_outerScope;
    context.baseExpr = GetBaseExpr(baseExpr);

    AddGenericOverloadCandidates(baseExpr, context);

    if (context.bestCandidates.getCount() > 0)
    {
        // Things were ambiguous.
        if (context.bestCandidates[0].status != OverloadCandidate::Status::Applicable)
        {
            // There were multiple equally-good candidates, but none actually usable.
            // We will construct a diagnostic message to help out.

            // TODO(tfoley): print a reasonable message here...

            getSink()->diagnose(
                genericAppExpr,
                Diagnostics::unimplemented,
                "no applicable generic");

            return CreateErrorExpr(genericAppExpr);
        }
        else
        {
            // There were multiple viable candidates, but that isn't an error: we just need
            // to complete all of them and create an overloaded expression as a result.

            auto overloadedExpr = m_astBuilder->create<OverloadedExpr2>();
            overloadedExpr->base = context.baseExpr;
            for (auto candidate : context.bestCandidates)
            {
                auto candidateExpr = CompleteOverloadCandidate(context, candidate);
                overloadedExpr->candidiateExprs.add(candidateExpr);
            }
            return overloadedExpr;
        }
    }
    else if (context.bestCandidate)
    {
        // There was one best candidate, even if it might not have been
        // applicable in the end.
        // We will report errors for this one candidate, then, to give
        // the user the most help we can.
        return CompleteOverloadCandidate(context, *context.bestCandidate);
    }
    else
    {
        // Nothing at all was found that we could even consider invoking
        getSink()->diagnose(genericAppExpr, Diagnostics::expectedAGeneric, baseExpr->type);
        return CreateErrorExpr(genericAppExpr);
    }
}

} // namespace Slang
