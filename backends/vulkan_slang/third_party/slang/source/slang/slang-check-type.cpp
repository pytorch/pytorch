// slang-check-type.cpp
#include "slang-check-impl.h"

// This file implements semantic checking logic related to types
// and type expressions (aka `TypeRepr`).

namespace Slang
{
Type* checkProperType(Linkage* linkage, TypeExp typeExp, DiagnosticSink* sink)
{
    SharedSemanticsContext sharedSemanticsContext(linkage, nullptr, sink);
    SemanticsVisitor visitor(&sharedSemanticsContext);

    SLANG_AST_BUILDER_RAII(linkage->getASTBuilder());

    auto typeOut = visitor.CheckProperType(typeExp);
    return typeOut.type;
}

Type* getPointedToTypeIfCanImplicitDeref(Type* type)
{
    if (auto ptrLike = as<PointerLikeType>(type))
    {
        return ptrLike->getElementType();
    }
    else if (auto ptrType = as<PtrType>(type))
    {
        return ptrType->getValueType();
    }
    else if (auto refType = as<RefType>(type))
    {
        return refType->getValueType();
    }
    return nullptr;
}

Expr* SemanticsVisitor::TranslateTypeNodeImpl(Expr* node)
{
    if (!node)
        return nullptr;

    auto expr = CheckTerm(node);
    expr = ExpectATypeRepr(expr);
    return expr;
}

Type* SemanticsVisitor::ExtractTypeFromTypeRepr(Expr* typeRepr)
{
    if (!typeRepr)
        return nullptr;
    if (auto typeType = as<TypeType>(typeRepr->type))
    {
        return typeType->getType();
    }
    return m_astBuilder->getErrorType();
}

Type* SemanticsVisitor::TranslateTypeNode(Expr* node)
{
    if (!node)
        return nullptr;
    auto typeRepr = TranslateTypeNodeImpl(node);
    return ExtractTypeFromTypeRepr(typeRepr);
}

TypeExp SemanticsVisitor::TranslateTypeNodeForced(TypeExp const& typeExp)
{
    auto typeRepr = TranslateTypeNodeImpl(typeExp.exp);

    TypeExp result;
    result.exp = typeRepr;
    result.type = ExtractTypeFromTypeRepr(typeRepr);
    return result;
}

TypeExp SemanticsVisitor::TranslateTypeNode(TypeExp const& typeExp)
{
    // HACK(tfoley): It seems that in some cases we end up re-checking
    // syntax that we've already checked. We need to root-cause that
    // issue, but for now a quick fix in this case is to early
    // exist if we've already got a type associated here:
    if (typeExp.type)
    {
        return typeExp;
    }
    return TranslateTypeNodeForced(typeExp);
}

Type* SemanticsVisitor::getRemovedModifierType(ModifiedType* modifiedType, ModifierVal* modifier)
{
    if (modifiedType->getModifierCount() == 1)
        return modifiedType->getBase();
    List<Val*> newModifiers;
    for (Index i = 0; i < modifiedType->getModifierCount(); i++)
    {
        auto m = modifiedType->getModifier(i);
        if (m == modifier)
            continue;
        newModifiers.add(m);
    }
    return m_astBuilder->getModifiedType(modifiedType->getBase(), newModifiers);
}

Type* SemanticsVisitor::getConstantBufferType(Type* elementType, Type* layoutType)
{
    auto iBufferDataLayoutType = m_astBuilder->getSharedASTBuilder()->getIBufferDataLayoutType();
    auto witness = isSubtype(layoutType, iBufferDataLayoutType, IsSubTypeOptions());
    return m_astBuilder->getConstantBufferType(elementType, layoutType, witness);
}

Expr* SemanticsVisitor::ExpectATypeRepr(Expr* expr)
{
    if (auto overloadedExpr = as<OverloadedExpr>(expr))
    {
        expr = resolveOverloadedExpr(overloadedExpr, LookupMask::type);
    }

    if (const auto typeType = as<TypeType>(expr->type))
    {
        return expr;
    }
    else if (const auto errorType = as<ErrorType>(expr->type))
    {
        return expr;
    }

    getSink()->diagnose(expr, Diagnostics::expectedAType, expr->type);
    return CreateErrorExpr(expr);
}

Type* SemanticsVisitor::ExpectAType(Expr* expr)
{
    auto typeRepr = ExpectATypeRepr(expr);
    if (auto typeType = as<TypeType>(typeRepr->type))
    {
        return typeType->getType();
    }
    return m_astBuilder->getErrorType();
}

Type* SemanticsVisitor::ExtractGenericArgType(Expr* exp)
{
    return ExpectAType(exp);
}

IntVal* SemanticsVisitor::ExtractGenericArgInteger(
    Expr* exp,
    Type* genericParamType,
    DiagnosticSink* sink)
{
    IntVal* val = CheckIntegerConstantExpression(
        exp,
        genericParamType ? IntegerConstantExpressionCoercionType::SpecificType
                         : IntegerConstantExpressionCoercionType::AnyInteger,
        genericParamType,
        ConstantFoldingKind::LinkTime,
        sink);
    if (val)
        return val;

    // If the argument expression could not be coerced to an integer
    // constant expression in context, then we will instead construct
    // a dummy "error" value to represent the result.
    //
    val = m_astBuilder->getOrCreate<ErrorIntVal>(m_astBuilder->getIntType());
    return val;
}

IntVal* SemanticsVisitor::ExtractGenericArgInteger(Expr* exp, Type* genericParamType)
{
    return ExtractGenericArgInteger(exp, genericParamType, getSink());
}

Val* SemanticsVisitor::ExtractGenericArgVal(Expr* exp)
{
    if (auto overloadedExpr = as<OverloadedExpr>(exp))
    {
        // assume that if it is overloaded, we want a type
        exp = resolveOverloadedExpr(overloadedExpr, LookupMask::type);
    }
    if (auto typeType = as<TypeType>(exp->type))
    {
        return typeType->getType();
    }
    else if (const auto errorType = as<ErrorType>(exp->type))
    {
        return exp->type.type;
    }
    else
    {
        if (!exp->type.type)
        {
            CheckExpr(exp);
        }
        return ExtractGenericArgInteger(exp, nullptr);
    }
}

Type* SemanticsVisitor::InstantiateGenericType(
    DeclRef<GenericDecl> genericDeclRef,
    List<Expr*> const& args)
{
    List<Val*> evaledArgs;

    for (auto argExpr : args)
    {
        evaledArgs.add(ExtractGenericArgVal(argExpr));
    }

    DeclRef<Decl> innerDeclRef =
        m_astBuilder->getGenericAppDeclRef(genericDeclRef, evaledArgs.getArrayView());
    return DeclRefType::create(m_astBuilder, innerDeclRef);
}

bool isManagedType(Type* type)
{
    if (auto declRefValueType = as<DeclRefType>(type))
    {
        auto decl = declRefValueType->getDeclRef().getDecl();
        if (as<ClassDecl>(decl))
            return true;
        if (as<InterfaceDecl>(decl) && decl->findModifier<ComInterfaceAttribute>())
            return true;
    }
    return false;
}

bool SemanticsVisitor::CoerceToProperTypeImpl(
    TypeExp const& typeExp,
    Type** outProperType,
    DiagnosticSink* diagSink)
{
    Type* result = nullptr;
    Type* type = typeExp.type;
    auto originalExpr = typeExp.exp;
    auto expr = originalExpr;
    if (!type && expr)
    {
        expr = maybeResolveOverloadedExpr(expr, LookupMask::type, diagSink);

        if (auto typeType = as<TypeType>(expr->type))
        {
            type = typeType->getType();
        }
    }

    if (!type)
    {
        // Only output diagnostic if we have a sink.
        if (diagSink)
        {
            // This function *can* be called with typeExp with both exp and type = nullptr.
            // Previous behavior didn't output a diagnostic if originalExpr was null, so this keeps
            // that behavior.
            //
            // Additional we check for ErrorType on expr, because if it's set a diagnostic has
            // already been output via previous code or via maybeResolveOverloadedExpr.
            if (originalExpr && (expr == nullptr || as<ErrorType>(expr->type) == nullptr))
            {
                // The diagnostic for expectedAType wants to say what it 'got'.
                // The solution given here, currently is to just use the node name.
                // How useful that might be could depend, and perhaps some other mechanism
                // that catagorized 'what' the wrong thing was is. For now this seems sufficient.
                //
                // Note that use originalExpr (not expr) because we want original expr for
                // diagnostic.

                // Get the AST node type info, so we can output a 'got' name
                diagSink->diagnose(
                    originalExpr,
                    Diagnostics::expectedAType,
                    originalExpr->getClass().getName());
            }
        }

        if (outProperType)
        {
            *outProperType = nullptr;
        }
        return false;
    }

    if (auto genericDeclRefType = as<GenericDeclRefType>(type))
    {
        // We are using a reference to a generic declaration as a concrete
        // type. This means we should substitute in any default parameter values
        // if they are available.
        //
        // TODO(tfoley): A more expressive type system would substitute in
        // "fresh" variables and then solve for their values...
        //

        auto genericDeclRef = genericDeclRefType->getDeclRef();
        ensureDecl(genericDeclRef, DeclCheckState::CanSpecializeGeneric);
        List<Val*> args;
        List<Val*> witnessArgs;
        for (Decl* member : genericDeclRef.getDecl()->members)
        {
            if (auto typeParam = as<GenericTypeParamDecl>(member))
            {
                if (auto defaultArg = typeParam->initType.type)
                {
                    if (outProperType)
                        args.add(defaultArg);
                }
                else
                {
                    if (diagSink)
                    {
                        diagSink->diagnose(typeExp.exp, Diagnostics::genericTypeNeedsArgs, typeExp);
                        *outProperType = m_astBuilder->getErrorType();
                    }
                    return false;
                }
            }
            else if (auto valParam = as<GenericValueParamDecl>(member))
            {
                if (!valParam->initExpr)
                {
                    if (diagSink)
                    {
                        diagSink->diagnose(
                            typeExp.exp,
                            Diagnostics::unimplemented,
                            "can't fill in default for generic type parameter");
                        *outProperType = m_astBuilder->getErrorType();
                    }
                    return false;
                }
                // TODO: this is one place where syntax should get cloned!
                if (outProperType)
                    args.add(ExtractGenericArgVal(valParam->initExpr));
            }
            else if (auto constraintParam = as<GenericTypeConstraintDecl>(member))
            {
                auto genericParam = as<DeclRefType>(constraintParam->sub.type)->getDeclRef();
                if (!genericParam)
                    return false;
                auto genericTypeParamDecl = as<GenericTypeParamDecl>(genericParam.getDecl());
                if (!genericTypeParamDecl)
                    return false;
                auto defaultType = CheckProperType(genericTypeParamDecl->initType);
                if (!defaultType)
                    return false;
                auto witness =
                    tryGetSubtypeWitness(defaultType, CheckProperType(constraintParam->sup));
                if (!witness)
                {
                    // diagnose
                    getSink()->diagnose(
                        genericTypeParamDecl->initType.exp,
                        Diagnostics::typeArgumentDoesNotConformToInterface,
                        defaultType,
                        constraintParam->sup);
                    return false;
                }
                witnessArgs.add(witness);
            }
            else
            {
                // ignore non-parameter members
            }
        }
        // Combine args and witnessArgs
        args.addRange(witnessArgs);

        result = DeclRefType::create(
            getASTBuilder(),
            getASTBuilder()->getGenericAppDeclRef(genericDeclRef, args.getArrayView()));
    }

    // default case: we expect this to already be a proper type
    if (!result)
    {
        result = type;
    }

    // Check for invalid types.
    // We don't allow pointers to managed types.
    if (auto ptrType = as<PtrType>(result))
    {
        if (isManagedType(ptrType->getValueType()))
        {
            getSink()->diagnose(typeExp.exp, Diagnostics::cannotDefinePtrTypeToManagedResource);
        }
    }

    *outProperType = result;
    return true;
}

TypeExp SemanticsVisitor::CoerceToProperType(TypeExp const& typeExp)
{
    TypeExp result = typeExp;
    CoerceToProperTypeImpl(typeExp, &result.type, getSink());
    return result;
}

TypeExp SemanticsVisitor::tryCoerceToProperType(TypeExp const& typeExp)
{
    TypeExp result = typeExp;
    if (!CoerceToProperTypeImpl(typeExp, &result.type, nullptr))
        return TypeExp();
    return result;
}

TypeExp SemanticsVisitor::CheckProperType(TypeExp typeExp)
{
    return CoerceToProperType(TranslateTypeNode(typeExp));
}

TypeExp SemanticsVisitor::CoerceToUsableType(TypeExp const& typeExp, Decl* decl)
{
    TypeExp result = CoerceToProperType(typeExp);
    Type* type = result.type;
    if (auto basicType = as<BasicExpressionType>(type))
    {
        // TODO: `void` shouldn't be a basic type, to make this easier to avoid
        if (basicType->getBaseType() == BaseType::Void)
        {
            // TODO(tfoley): pick the right diagnostic message
            getSink()->diagnose(result.exp, Diagnostics::invalidTypeVoid);
            result.type = m_astBuilder->getErrorType();
            return result;
        }
    }

    // A type pack is not a usable type other than for defining parameters.
    if (!as<ParamDecl>(decl) && isTypePack(type))
    {
        getSink()->diagnose(typeExp.exp, Diagnostics::improperUseOfType, typeExp.type);
        result.type = m_astBuilder->getErrorType();
        return result;
    }
    return result;
}

TypeExp SemanticsVisitor::CheckUsableType(TypeExp typeExp, Decl* decl)
{
    return CoerceToUsableType(TranslateTypeNode(typeExp), decl);
}

bool SemanticsVisitor::ValuesAreEqual(IntVal* left, IntVal* right)
{
    if (left == right)
        return true;

    if (auto leftConst = as<ConstantIntVal>(left))
    {
        if (auto rightConst = as<ConstantIntVal>(right))
        {
            return leftConst->getValue() == rightConst->getValue();
        }
    }

    if (auto leftVar = as<GenericParamIntVal>(left))
    {
        if (auto rightVar = as<GenericParamIntVal>(right))
        {
            return leftVar->getDeclRef().equals(rightVar->getDeclRef());
        }
        else if (const auto rightPoly = as<PolynomialIntVal>(right))
        {
            return right->equals(leftVar);
        }
    }
    if (auto leftVar = as<PolynomialIntVal>(left))
    {
        return leftVar->equals(right);
    }
    return false;
}

VectorExpressionType* SemanticsVisitor::createVectorType(Type* elementType, IntVal* elementCount)
{
    return m_astBuilder->getVectorType(elementType, elementCount);
}

Expr* SemanticsExprVisitor::visitSharedTypeExpr(SharedTypeExpr* expr)
{
    if (!expr->type.Ptr())
    {
        expr->base = CheckProperType(expr->base);
        expr->type = expr->base.exp->type;
    }
    return expr;
}

} // namespace Slang
