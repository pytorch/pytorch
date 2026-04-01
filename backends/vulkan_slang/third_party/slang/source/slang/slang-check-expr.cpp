// slang-check-expr.cpp
#include "slang-check-impl.h"

// This file contains semantic-checking logic for the various
// expression types in the AST.
//
// Note that some cases of expression checking are split
// of into their own files. Notably:
//
// * `slang-check-overload.cpp` is responsible for the logic of resolving overloaded calls
//
// * `slang-check-conversion.cpp` is responsible for the logic of handling type conversion/coercion

#include "core/slang-char-util.h"
#include "slang-ast-decl.h"
#include "slang-ast-natural-layout.h"
#include "slang-ast-print.h"
#include "slang-ast-synthesis.h"
#include "slang-lookup-spirv.h"
#include "slang-lookup.h"

namespace Slang
{
DeclRefType* SemanticsVisitor::getExprDeclRefType(Expr* expr)
{
    if (auto typetype = as<TypeType>(expr->type))
        return dynamicCast<DeclRefType>(typetype->getType());
    else
        return as<DeclRefType>(expr->type);
}

void SemanticsContext::ExprLocalScope::addBinding(LetExpr* binding)
{
    if (!m_innerMostBinding)
    {
        SLANG_ASSERT(!m_outerMostBinding);

        // If we haven't added any bindings, then `binding`
        // becomes both the inner-most and outer most.
        //
        m_innerMostBinding = binding;
        m_outerMostBinding = binding;
    }
    else
    {
        SLANG_ASSERT(m_outerMostBinding);

        // If we already have bindings, then `binding`
        // will become the new inner-most binding.
        //
        m_innerMostBinding->body = binding;
        m_innerMostBinding = binding;
    }
}


/// Move `expr` into a temporary variable and execute `func` on that variable.
///
/// Returns an expression that wraps both the creation and initialization of
/// the temporary, and the computation created by `func`.
///
template<typename F>
Expr* SemanticsVisitor::moveTemp(Expr* const& expr, F const& func)
{
    VarDecl* varDecl = m_astBuilder->create<VarDecl>();
    varDecl->parentDecl = nullptr;
    if (m_outerScope && m_outerScope->containerDecl)
        m_outerScope->containerDecl->addMember(varDecl);
    addModifier(varDecl, m_astBuilder->create<LocalTempVarModifier>());
    varDecl->checkState = DeclCheckState::DefinitionChecked;
    varDecl->nameAndLoc.loc = expr->loc;
    varDecl->initExpr = expr;
    varDecl->type.type = expr->type.type;

    auto varDeclRef = makeDeclRef(varDecl);

    LetExpr* letExpr = m_astBuilder->create<LetExpr>();
    letExpr->decl = varDecl;

    auto body = func(varDeclRef);
    Expr* result = body;
    if (auto exprLocalScope = getExprLocalScope())
    {
        // We want to add the `LetExpr` to the set of such expressions
        // in the local scope, so that it can be emitted properly.
        //
        exprLocalScope->addBinding(letExpr);
    }
    else
    {
        // If we somehow got in here and there wasn't an expression-local
        // scope established yet, it almost certainly represents an error.
        //
        SLANG_ASSERT(exprLocalScope);

        // As a fallback, though, we will try to wire up the `letExpr`
        // to surround the body directly and return that.
        //
        letExpr->body = body;
        letExpr->type = body->type;

        result = letExpr;
    }
    return result;
}

/// Execute `func` on a variable with the value of `expr`.
///
/// If `expr` is just a reference to an immutable (e.g., `let`) variable
/// then this might use the existing variable. Otherwise it will create
/// a new variable to hold `expr`, using `moveTemp()`.
///
template<typename F>
Expr* SemanticsVisitor::maybeMoveTemp(Expr* const& expr, F const& func)
{
    // TODO: Eventually this operation could consider any case where the
    // input `expr` names an immutable "path": one that starts at an
    // immutable binding and follows a (possibly empty) chain of accesses
    // to immutable members.

    if (auto varExpr = as<VarExpr>(expr))
    {
        auto declRef = varExpr->declRef;
        if (auto varDeclRef = declRef.as<LetDecl>())
            return func(varDeclRef);
    }

    return moveTemp(expr, func);
}

/// Return an expression that represents "opening" the existential `expr`.
///
/// The type of `expr` must be an interface type, matching `interfaceDeclRef`.
///
/// If we scope down the PL theory to just the case that Slang cares about,
/// a value of an existential type like `IMover` is a tuple of:
///
///  * a concrete type `X`
///  * a witness `w` of the fact that `X` implements `IMover`
///  * a value `v` of type `X`
///
/// "Opening" an existential value is the process of decomposing a single
/// value `e : IMover` into the pieces `X`, `w`, and `v`.
///
/// Rather than return all those pieces individually, this operation
/// returns an expression that logically corresponds to `v`: an expression
/// of type `X`, where the type carries the knowledge that `X` implements `IMover`.
///
Expr* SemanticsVisitor::openExistential(Expr* expr, DeclRef<InterfaceDecl> interfaceDeclRef)
{
    // If `expr` refers to an immutable binding,
    // then we can use it directly. If it refers
    // to an arbitrary expression or a mutable
    // binding, we will move its value into an
    // immutable temporary so that we can use
    // it directly.
    //
    return maybeMoveTemp(
        expr,
        [&](DeclRef<VarDeclBase> varDeclRef)
        {
            ExtractExistentialType* openedType = m_astBuilder->getOrCreate<ExtractExistentialType>(
                varDeclRef,
                expr->type.type,
                interfaceDeclRef);

            ExtractExistentialValueExpr* openedValue =
                m_astBuilder->create<ExtractExistentialValueExpr>();
            openedValue->declRef = varDeclRef;
            openedValue->type = QualType(openedType);
            openedValue->originalExpr = expr;
            openedValue->checked = true;
            // The result of opening an existential is an l-value
            // if the original existential is an l-value.
            //
            if (expr->type.isLeftValue)
            {
                // Marking the opened value as an l-value is the easy part.
                //
                openedValue->type.isLeftValue = true;

                // The more challenging bit is that in this case the `maybeMoveTemp()`
                // operation will have copied the original existential value into
                // a temporary.
                //
                // If this expression is used in an l-value context, then we need
                // to be able to generate code to "write back" the modified value
                // (which will be of `openedType`) to the original location named
                // by `expr` (an existential for `interfaceDeclRef`).
                //
            }

            return openedValue;
        });
}

/// If `expr` has existential type, then open it.
///
/// Returns an expression that opens `expr` if it had existential type.
/// Otherwise returns `expr` itself.
///
/// See `openExistential` for a discussion of what "opening" an
/// existential-type value means.
///
Expr* SemanticsVisitor::maybeOpenExistential(Expr* expr)
{
    auto exprType = expr->type.type;

    if (auto declRefType = as<DeclRefType>(exprType))
    {
        if (auto interfaceDeclRef = declRefType->getDeclRef().as<InterfaceDecl>())
        {
            return openExistential(expr, interfaceDeclRef);
        }
    }

    // Default: apply the callback to the original expression;
    return expr;
}

Expr* SemanticsVisitor::maybeOpenRef(Expr* expr)
{
    auto exprType = expr->type.type;

    if (auto refType = as<RefTypeBase>(exprType))
    {
        auto openRef = m_astBuilder->create<OpenRefExpr>();
        openRef->innerExpr = expr;
        openRef->type.isLeftValue = (as<RefType>(exprType) != nullptr);
        openRef->type.type = refType->getValueType();
        openRef->checked = true;
        return openRef;
    }
    return expr;
}

Scope* SemanticsVisitor::getScope(SyntaxNode* node)
{
    while (auto declBase = as<Decl>(node))
    {
        if (auto container = as<ContainerDecl>(node))
        {
            if (container->ownedScope)
                return container->ownedScope;
        }
        node = declBase->parentDecl;
    }
    return nullptr;
}

static SourceLoc _getMemberOpLoc(Expr* expr)
{
    if (auto m = as<MemberExpr>(expr))
        return m->memberOperatorLoc;
    if (auto m = as<StaticMemberExpr>(expr))
        return m->memberOperatorLoc;
    return SourceLoc();
}

void addSiblingScopeForContainerDecl(
    ASTBuilder* builder,
    ContainerDecl* dest,
    ContainerDecl* source)
{
    addSiblingScopeForContainerDecl(builder, dest->ownedScope, source);
}

void addSiblingScopeForContainerDecl(ASTBuilder* builder, Scope* destScope, ContainerDecl* source)
{
    auto subScope = builder->create<Scope>();
    subScope->containerDecl = source;

    subScope->nextSibling = destScope->nextSibling;
    destScope->nextSibling = subScope;
}

void SemanticsVisitor::diagnoseDeprecatedDeclRefUsage(
    DeclRef<Decl> declRef,
    SourceLoc loc,
    Expr* originalExpr)
{
    // This is slightly subtle, because we don't want to warn more than
    // once for the same occurrence, however in some cases this function is
    // called more than once for the same declref (specifically in the case
    // of a non-overloaded function, once when the function is identified at
    // first, and again when it's checked from
    // CheckInvokeExprWithCheckedOperands).
    //
    // The correct fix is probably to make
    // CheckInvokeExprWithCheckedOperands reuse the original declref,
    // however that doesn't appear to be a simple change.
    //
    // What we do instead is see if there's already been a declRef
    // constructed for this expression and rest assured that it's already
    // had a diagnostic emitted.
    auto originalAppExpr = as<AppExprBase>(originalExpr);
    auto originalAppFunDecl =
        originalAppExpr ? as<DeclRefExpr>(originalAppExpr->functionExpr) : nullptr;
    if (originalAppFunDecl && originalAppFunDecl->declRef)
    {
        return;
    }
    if (auto deprecatedAttr = declRef.getDecl()->findModifier<DeprecatedAttribute>())
    {
        getSink()->diagnose(
            loc,
            Diagnostics::deprecatedUsage,
            declRef.getName(),
            deprecatedAttr->message);
    }
}

static bool isMutableGLSLBufferBlockVarExpr(Expr* expr)
{
    const auto derefExpr = as<DerefExpr>(expr);
    if (!derefExpr)
        return false;

    // For SSBO arrays, derefExpr is expected to be IndexExpr instead of VarExpr
    const auto indexExpr = as<IndexExpr>(derefExpr->base);

    const auto varExpr =
        indexExpr ? as<VarExpr>(indexExpr->baseExpression) : as<VarExpr>(derefExpr->base);
    // Check the declaration type
    if (!varExpr)
        return false;

    const auto varExprType = (indexExpr ? indexExpr->type : varExpr->type)->getCanonicalType();
    const auto ssbt = as<GLSLShaderStorageBufferType>(varExprType);
    if (!ssbt)
        return false;

    // Check the modifiers on the declaration
    const auto d = varExpr->declRef.getDecl();
    auto collection = d->findModifier<MemoryQualifierSetModifier>();
    if (collection &&
        collection->getMemoryQualifierBit() & MemoryQualifierSetModifier::Flags::kReadOnly)
        return false;

    return true;
}

DeclRefExpr* SemanticsVisitor::ConstructDeclRefExpr(
    DeclRef<Decl> declRef,
    Expr* baseExpr,
    Name* name,
    SourceLoc loc,
    Expr* originalExpr)
{
    // Compute the type that this declaration reference will have in context.
    //
    auto type = GetTypeForDeclRef(declRef, loc);

    // This is the bottleneck for using declarations which might be
    // deprecated, diagnose here.
    diagnoseDeprecatedDeclRefUsage(declRef, loc, originalExpr);

    // Construct an appropriate expression based on the structured of
    // the declaration reference.
    //
    if (baseExpr)
    {
        // If there was a base expression, we will have some kind of
        // member expression.

        // We want to check for the case where the base "expression"
        // actually names a type, because in that case we are doing
        // a static member reference.
        //
        if (auto typeType = as<TypeType>(baseExpr->type->getCanonicalType()))
        {
            // Before forming the reference, we will check if the
            // member being referenced can even be used as a static
            // member, and if not we will diagnose an error.
            //
            // TODO: It is conceptually possible to allow static
            // references to many instance members, provided we
            // change the exposed type/signature.
            //
            // E.g., if we have:
            //
            //      struct Test { float getVal() { ... } }
            //
            // Then a reference to `Test.getVal` could be allowed,
            // and given a type of `(Test) -> float` to indicate
            // that it is an "unbound" instance method.
            //
            if (!isDeclUsableAsStaticMember(declRef.getDecl()))
            {
                getSink()->diagnose(
                    loc,
                    Diagnostics::staticRefToNonStaticMember,
                    typeType->getType(),
                    declRef.getName());
            }

            auto expr = m_astBuilder->create<StaticMemberExpr>();
            expr->loc = loc;
            expr->type = type;
            expr->baseExpression = baseExpr;
            expr->name = name;
            expr->declRef = declRef;
            expr->memberOperatorLoc = _getMemberOpLoc(originalExpr);
            return expr;
        }
        else if (isEffectivelyStatic(declRef.getDecl()))
        {
            // Extract the type of the baseExpr
            auto baseExprType = baseExpr->type.type;
            SharedTypeExpr* baseTypeExpr = m_astBuilder->create<SharedTypeExpr>();
            baseTypeExpr->base.type = baseExprType;
            baseTypeExpr->type.type = m_astBuilder->getTypeType(baseExprType);

            auto expr = m_astBuilder->create<StaticMemberExpr>();
            expr->loc = loc;
            expr->type = type;
            expr->baseExpression = baseTypeExpr;
            expr->name = name;
            expr->declRef = declRef;
            expr->memberOperatorLoc = _getMemberOpLoc(originalExpr);
            return expr;
        }
        else
        {
            // If the base expression wasn't a type, then this
            // is a normal member expression.
            //
            auto expr = m_astBuilder->create<MemberExpr>();
            expr->loc = loc;
            expr->type = type;
            expr->baseExpression = baseExpr;
            expr->name = name;
            expr->declRef = declRef;
            expr->memberOperatorLoc = _getMemberOpLoc(originalExpr);

            // If any member declares the following value is a
            // write only, we must declare the parent as a write
            // only to avoid modifying the child
            expr->type.isWriteOnly = baseExpr->type.isWriteOnly || expr->type.isWriteOnly;

            // When referring to a member through an expression,
            // the result is only an l-value if both the base
            // expression and the member agree that it should be.
            //
            // We have already used the `QualType` from the member
            // above (that is `type`), so we need to take the
            // l-value status of the base expression into account now.
            if (!baseExpr->type.isLeftValue)
            {
                // One exception to this is if we're reading the contents
                // of a GLSL buffer interface block which isn't marked as
                // read_only
                expr->type.isLeftValue = isMutableGLSLBufferBlockVarExpr(baseExpr) &&
                                         (expr->type.hasReadOnlyOnTarget == false);

                // Another exception is if we are accessing a property
                // that provides a [nonmutating] setter.
                if (!expr->type.isLeftValue && as<PropertyDecl>(declRef.getDecl()))
                {
                    bool isLValue = false;
                    for (auto member : as<ContainerDecl>(declRef.getDecl())->members)
                    {
                        if (as<SetterDecl>(member) || as<RefAccessorDecl>(member))
                        {
                            if (member->findModifier<NonmutatingAttribute>())
                            {
                                isLValue = true;
                            }
                            break;
                        }
                    }
                    expr->type.isLeftValue = isLValue;
                }
            }
            else
            {
                // If we are accessing a readonly property, then the result
                // is not an l-value.
                if (auto propertyDecl = as<PropertyDecl>(declRef.getDecl()))
                {
                    bool isLValue = false;
                    for (auto member : propertyDecl->members)
                    {
                        if (as<SetterDecl>(member) || as<RefAccessorDecl>(member))
                        {
                            isLValue = true;
                            break;
                        }
                    }
                    expr->type.isLeftValue = isLValue;
                }
            }
            return expr;
        }
    }
    else
    {
        // If there is no base expression, then the result must
        // be an ordinary variable expression.
        //
        auto expr = m_astBuilder->create<VarExpr>();
        expr->loc = loc;
        expr->name = name;
        expr->type = type;
        expr->declRef = declRef;
        // Keep a reference to the original expr if it was a genericApp/member.
        // This is needed by the language server to locate the original tokens.
        if (as<GenericAppExpr>(originalExpr) || as<MemberExpr>(originalExpr) ||
            as<StaticMemberExpr>(originalExpr))
        {
            expr->originalExpr = originalExpr;
        }
        return expr;
    }
}

Expr* SemanticsVisitor::constructDerefExpr(Expr* base, QualType elementType, SourceLoc loc)
{
    if (auto resPtrType = as<DescriptorHandleType>(base->type))
    {
        return coerce(CoercionSite::ExplicitCoercion, resPtrType->getElementType(), base);
    }

    auto derefExpr = m_astBuilder->create<DerefExpr>();
    derefExpr->loc = loc;
    derefExpr->base = base;
    derefExpr->type = QualType(elementType);
    derefExpr->checked = true;

    if (as<PtrType>(base->type) || as<RefType>(base->type))
    {
        derefExpr->type.isLeftValue = true;
    }
    else
    {
        derefExpr->type.isLeftValue = base->type.isLeftValue;
        derefExpr->type.isLeftValue = base->type.isLeftValue;
        derefExpr->type.hasReadOnlyOnTarget = base->type.hasReadOnlyOnTarget;
        derefExpr->type.isWriteOnly = base->type.isWriteOnly;
    }

    return derefExpr;
}

Expr* SemanticsVisitor::ConstructDerefExpr(Expr* base, SourceLoc loc)
{
    auto elementType = getPointedToTypeIfCanImplicitDeref(base->type);
    SLANG_ASSERT(elementType);

    return constructDerefExpr(base, elementType, loc);
}

InvokeExpr* SemanticsVisitor::constructUncheckedInvokeExpr(
    Expr* callee,
    const List<Expr*>& arguments)
{
    auto result = m_astBuilder->create<InvokeExpr>();
    result->loc = callee->loc;
    result->functionExpr = callee;
    result->arguments.addRange(arguments);
    return result;
}

Expr* SemanticsVisitor::maybeUseSynthesizedDeclForLookupResult(
    LookupResultItem const& item,
    Expr* originalExpr)
{
    // If the only result from lookup is an entry in an interface decl, it could be that
    // the user is leaving out an explicit definition for the requirement and depending on
    // the compiler to synthesis the definition.
    // In this case, if the lookup is triggered from a location such that the satisfying
    // definition should be returned should it existed, we should create a placeholder decl for
    // the definition and return a reference to to newly created decl instead of the requirement
    // decl in the interface.
    switch (item.declRef.getDecl()->astNodeType)
    {
    case ASTNodeType::AssocTypeDecl:
        break;
    case ASTNodeType::FuncDecl:
        // We don't need to intercept lookup results with synthesized decls for methods,
        // because function lookups will only take place when we are checking the decl bodies.
        // At that point conformance check and synthesis is already done so they will always
        // resolve to the synthesized method.
        return nullptr;
    default:
        return nullptr;
    }

    // We need to check if the lookup should resolve to a definition in an implementation type
    // if it existed.
    // This will be the case when the lookup is initiated from the concrete implementation type
    // instead of directly from the Interface decl. The breadcrumbs of the lookup should provide
    // this information.

    // If no breadcrumbs existed, then the lookup should just resolve to the interface requirement.

    if (!item.breadcrumbs)
        return nullptr;

    // We will only ever need to synthesis a type to satisfy an associatedtype requirement.
    // In this case the lookup should have resolved to a known associatedtype decl.
    auto builtinAssocTypeAttr = item.declRef.getDecl()->findModifier<BuiltinRequirementModifier>();
    if (!builtinAssocTypeAttr)
        return nullptr;

    DeclRefType* subType = nullptr;

    // Check if we are reaching the associated type decl through inheritance from a concrete type.
    for (auto breadcrumb = item.breadcrumbs; breadcrumb; breadcrumb = breadcrumb->next)
    {
        switch (breadcrumb->kind)
        {
        case LookupResultItem::Breadcrumb::Kind::SuperType:
            {
                auto witness = as<SubtypeWitness>(breadcrumb->val);
                if (auto subDeclRefType = as<DeclRefType>(witness->getSub()))
                {
                    if (!as<InterfaceDecl>(subDeclRefType->getDeclRef().getDecl()))
                    {
                        // Store the inner most concrete super type.
                        subType = subDeclRefType;
                    }
                }
            }
            break;
        default:
            break;
        }
    }
    if (!subType)
        return nullptr;

    subType = as<DeclRefType>(subType->getCanonicalType());
    if (!subType)
        return nullptr;

    // Don't synthesize for generic parameters.
    auto parent = as<AggTypeDecl>(subType->getDeclRef().getDecl());
    if (!parent)
        return nullptr;

    // Don't synthesize for ThisType.
    if (as<ThisTypeDecl>(subType->getDeclRef().getDecl()))
        return nullptr;

    // If the inner most subtype is itself an associated type, then we're dealing
    // with an abstract type. There's not need to synthesize anythin at this point.
    //
    if (as<AssocTypeDecl>(subType->getDeclRef().getDecl()))
        return nullptr;

    // If we reach here, we are expecting a synthesized decl defined in `subType`.
    // Instead of returning a DeclRefExpr to the requirement decl, we synthesize a placeholder decl
    // in `subType` and return a DeclRefExpr to the synthesized decl.

    Decl* synthesizedDecl = nullptr;
    switch (builtinAssocTypeAttr->kind)
    {
    case BuiltinRequirementKind::DifferentialType:
        {
            if (!canStructBeUsedAsSelfDifferentialType(parent))
            {
                // Need to create a new struct type for the differential.
                //
                auto structDecl = m_astBuilder->create<StructDecl>();
                auto conformanceDecl = m_astBuilder->create<InheritanceDecl>();
                conformanceDecl->base.type = m_astBuilder->getDiffInterfaceType();
                structDecl->addMember(conformanceDecl);
                structDecl->parentDecl = parent;

                synthesizedDecl = structDecl;
                auto typeDef = m_astBuilder->create<TypeAliasDecl>();
                typeDef->nameAndLoc.name = getName("Differential");
                typeDef->parentDecl = structDecl;

                auto synthDeclRef =
                    createDefaultSubstitutionsIfNeeded(m_astBuilder, this, makeDeclRef(structDecl));

                typeDef->type.type = DeclRefType::create(m_astBuilder, synthDeclRef);
                structDecl->members.add(typeDef);

                synthesizedDecl->nameAndLoc.name = item.declRef.getName();
                synthesizedDecl->loc = parent->loc;
                parent->addMember(synthesizedDecl);
                parent->invalidateMemberDictionary();

                // Mark the newly synthesized decl as `ToBeSynthesized` so future checking can
                // differentiate it from user-provided definitions, and proceed to fill in its
                // definition.
                auto toBeSynthesized = m_astBuilder->create<ToBeSynthesizedModifier>();
                addModifier(synthesizedDecl, toBeSynthesized);
            }
            else
            {
                // There's no need for a new struct decl.
                // We can simply add a typealias to the existing concrete type.
                //
                auto typeDef = m_astBuilder->create<TypeAliasDecl>();
                typeDef->nameAndLoc.name = item.declRef.getName();

                // Compute the decl's type as if it is referred to from itself. This is important
                // because subType may have substitutions from the context it is used in, while this
                // synthesis step is local to the decl.
                //
                typeDef->type.type =
                    calcThisType(subType->getDeclRef().getDecl()->getDefaultDeclRef());

                synthesizedDecl = parent;

                parent->addMember(typeDef);
                parent->invalidateMemberDictionary();

                markSelfDifferentialMembersOfType(parent, subType);
            }
        }
        break;
    default:
        return nullptr;
    }

    auto synthDeclMemberRef =
        m_astBuilder->getMemberDeclRef(subType->getDeclRef(), synthesizedDecl);
    return ConstructDeclRefExpr(
        synthDeclMemberRef,
        nullptr,
        item.declRef.getName(),
        originalExpr ? originalExpr->loc : SourceLoc(),
        originalExpr);
}

Expr* SemanticsVisitor::ConstructLookupResultExpr(
    LookupResultItem const& item,
    Expr* baseExpr,
    Name* name,
    SourceLoc loc,
    Expr* originalExpr)
{
    if (!item.declRef)
    {
        originalExpr->type = QualType(m_astBuilder->getErrorType());
        return originalExpr;
    }

    // We could be referencing a decl that will be synthesized. If so create a placeholder
    // and return a DeclRefExpr to it.
    if (auto lookupResultExpr = maybeUseSynthesizedDeclForLookupResult(item, originalExpr))
        return lookupResultExpr;

    // If we collected any breadcrumbs, then these represent
    // additional segments of the lookup path that we need
    // to expand here.
    auto bb = baseExpr;
    for (auto breadcrumb = item.breadcrumbs; breadcrumb; breadcrumb = breadcrumb->next)
    {
        switch (breadcrumb->kind)
        {
        case LookupResultItem::Breadcrumb::Kind::Member:
            bb = ConstructDeclRefExpr(breadcrumb->declRef, bb, name, loc, originalExpr);
            break;

        case LookupResultItem::Breadcrumb::Kind::Deref:
            bb = ConstructDerefExpr(bb, loc);
            break;

        case LookupResultItem::Breadcrumb::Kind::SuperType:
            {
                // Note: a lookup through a super-type can
                // occur even in the case of a `static` member,
                // so we only modify the base expression here
                // if there is one.
                //
                if (bb)
                {
                    // We know that the breadcrumb reprsents a
                    // cast of the base expression to a super type,
                    // so we construct that cast explicitly here.
                    //
                    auto witness = as<SubtypeWitness>(breadcrumb->val);
                    SLANG_ASSERT(witness);
                    auto expr = createCastToSuperTypeExpr(witness->getSup(), bb, witness);

                    // Note that we allow a cast of an l-value to
                    // be used as an l-value here because it enables
                    // `[mutating]` methods to be called, and
                    // mutable properties to be modified, but this
                    // is probably not *technically* correct, since
                    // treating an l-value of type `Derived` as
                    // an l-value of type `Base` implies that we
                    // can assign an arbitrary value of type `Base`
                    // to that l-value (which would be an error).
                    //
                    // TODO: make sure we believe there are no
                    // issues here.
                    //
                    if (bb && bb->type.isLeftValue)
                    {
                        expr->type.isLeftValue = true;
                    }

                    bb = expr;
                }
            }
            break;

        case LookupResultItem::Breadcrumb::Kind::This:
            {
                // We expect a `this` to always come
                // at the start of a chain.
                SLANG_ASSERT(bb == nullptr);

                // We will compute the type to use for `This` using
                // the same logic that a direct reference to `This`
                // uses.
                //
                auto thisType = calcThisType(breadcrumb->declRef);

                // Next we construct an appropriate expression to
                // stand in for the implicit `this` or `This` reference.
                //
                // The lookup process will have computed the appropriate
                // "mode" to use for the implicit `this` or `This`.
                //
                auto thisParameterMode = breadcrumb->thisParameterMode;
                if (thisParameterMode == LookupResultItem::Breadcrumb::ThisParameterMode::Type)
                {
                    // If we are in a static context, then we do not
                    // have implicit `this` expression, and the expression
                    // we construct will need to start with the `This`
                    // type.
                    //
                    // Because we are constrained to yield an expression
                    // here, we must construct an expression that
                    // references `This`, and the *type* of that expression
                    // will be `typeof(This)`, which conceptually
                    // `typeof(typeof(this))`
                    //
                    auto thisTypeType = m_astBuilder->getTypeType(thisType);

                    auto typeExpr = m_astBuilder->create<SharedTypeExpr>();
                    typeExpr->type.type = thisTypeType;
                    typeExpr->base.type = thisType;

                    bb = typeExpr;
                }
                else
                {
                    // In a context where both static and instance members can
                    // be referenced, we will construct a reference to `this`,
                    // and then rely on downstream logic to ensure that a
                    // refernece to `this.someStaticMember` will be translated
                    // over to `This.someStaticMember`.
                    //
                    ThisExpr* expr = m_astBuilder->create<ThisExpr>();
                    expr->type.type = thisType;
                    expr->loc = loc;
                    if (auto declRefExpr = as<DeclRefExpr>(originalExpr))
                        expr->scope = declRefExpr->scope;
                    else if (auto invokeExpr = as<InvokeExpr>(originalExpr))
                    {
                        if (auto calleeDeclRefExpr =
                                as<DeclRefExpr>(invokeExpr->originalFunctionExpr))
                            expr->scope = calleeDeclRefExpr->scope;
                    }
                    // Whether or not the implicit `this` is mutable depends
                    // on the context in which it is used, and the lookup
                    // logic will have computed an appropriate "mode" based
                    // on the context during lookup.
                    //
                    expr->type.isLeftValue =
                        thisParameterMode ==
                        LookupResultItem::Breadcrumb::ThisParameterMode::MutableValue;

                    bb = expr;
                }
            }
            break;

        default:
            SLANG_UNREACHABLE("all cases handle");
        }
        if (getShared()->isInLanguageServer())
        {
            // Don't make breadcrumb nodes carry any source loc info,
            // as they may confuse language server functionalities.
            if (bb)
            {
                bb->loc = SourceLoc();
            }
        }
    }

    return ConstructDeclRefExpr(item.declRef, bb, name, loc, originalExpr);
}

void SemanticsVisitor::suggestCompletionItems(
    CompletionSuggestions::ScopeKind scopeKind,
    LookupResult const& lookupResult)
{
    auto& suggestions = getLinkage()->contentAssistInfo.completionSuggestions;
    suggestions.clear();
    suggestions.scopeKind = scopeKind;
    for (auto item : lookupResult)
    {
        suggestions.candidateItems.add(item);
    }
}


Expr* SemanticsVisitor::createLookupResultExpr(
    Name* name,
    LookupResult const& lookupResult,
    Expr* baseExpr,
    SourceLoc loc,
    Expr* originalExpr)
{
    if (lookupResult.isOverloaded())
    {
        auto overloadedExpr = m_astBuilder->create<OverloadedExpr>();
        overloadedExpr->name = name;
        overloadedExpr->loc = loc;
        overloadedExpr->type = QualType(m_astBuilder->getOverloadedType());
        overloadedExpr->base = baseExpr;
        overloadedExpr->lookupResult2 = lookupResult;
        overloadedExpr->originalExpr = originalExpr;
        return overloadedExpr;
    }
    else
    {
        return ConstructLookupResultExpr(lookupResult.item, baseExpr, name, loc, originalExpr);
    }
}

DeclVisibility SemanticsVisitor::getTypeVisibility(Type* type)
{
    if (auto declRefType = as<DeclRefType>(type))
    {
        auto v = getDeclVisibility(declRefType->getDeclRef().getDecl());
        auto args = findInnerMostGenericArgs(SubstitutionSet(declRefType->getDeclRef()));
        for (auto arg : args)
        {
            if (auto typeArg = as<DeclRefType>(arg))
                v = Math::Min(v, getTypeVisibility(typeArg));
        }
        return v;
    }
    return DeclVisibility::Public;
}

bool SemanticsVisitor::isDeclVisibleFromScope(DeclRef<Decl> declRef, Scope* scope)
{
    auto visibility = getDeclVisibility(declRef.getDecl());
    if (visibility == DeclVisibility::Public)
        return true;
    if (visibility == DeclVisibility::Internal)
    {
        // Check that the decl is in the same module as the scope.
        auto declModule = getModuleDecl(declRef.getDecl());
        if (declModule == getModuleDecl(scope))
            return true;
    }
    if (visibility == DeclVisibility::Private)
    {
        // Check that the decl is in the same or parent container decl as scope.
        Decl* parentContainer = declRef.getDecl();
        for (; parentContainer; parentContainer = parentContainer->parentDecl)
        {
            if (as<AggTypeDeclBase>(parentContainer))
                break;
            if (as<NamespaceDeclBase>(parentContainer))
                break;
        }

        for (auto s = scope; s; s = s->parent)
        {
            if (s->containerDecl == parentContainer)
                return true;
        }
        return false;
    }
    return false;
}

LookupResult SemanticsVisitor::filterLookupResultByVisibility(const LookupResult& lookupResult)
{
    if (!m_outerScope)
        return lookupResult;
    LookupResult filteredResult;
    for (auto item : lookupResult)
    {
        if (isDeclVisibleFromScope(item.declRef, m_outerScope))
            AddToLookupResult(filteredResult, item);
    }
    return filteredResult;
}

LookupResult SemanticsVisitor::filterLookupResultByVisibilityAndDiagnose(
    const LookupResult& lookupResult,
    SourceLoc loc,
    bool& outDiagnosed)
{
    outDiagnosed = false;
    auto result = filterLookupResultByVisibility(lookupResult);
    if (lookupResult.isValid() && !result.isValid())
    {
        getSink()->diagnose(loc, Diagnostics::declIsNotVisible, lookupResult.item.declRef);
        outDiagnosed = true;

        if (getShared()->isInLanguageServer())
        {
            // When in language server mode, return the unfiltered result so we can still
            // provide language service around it.
            return lookupResult;
        }
    }
    return result;
}

LookupResult SemanticsVisitor::resolveOverloadedLookup(LookupResult const& inResult)
{
    // If the result isn't actually overloaded, it is fine as-is
    if (!inResult.isValid())
        return inResult;
    if (!inResult.isOverloaded())
        return inResult;

    // We are going to build up a list of items to return.
    List<LookupResultItem> items;
    for (auto item : inResult.items)
    {
        // For each item we consider adding, we will compare it
        // to those items we've already added.
        //
        // If any of the existing items is "better" than `item`,
        // then we will skip adding `item`.
        //
        // If `item` is "better" than any of the existing items,
        // we will remove those from `items`.
        //
        bool shouldAdd = true;
        for (Index ii = 0; ii < items.getCount(); ++ii)
        {
            int cmp = CompareLookupResultItems(item, items[ii]);
            if (cmp < 0)
            {
                // The new `item` is strictly better
                items.fastRemoveAt(ii);
                --ii;
            }
            else if (cmp > 0)
            {
                // The existing item is strictly better
                shouldAdd = false;
            }
        }
        if (shouldAdd)
        {
            items.add(item);
        }
    }

    // The resulting `items` list should be all those items
    // that were neither better nor worse than one another.
    //
    // There should always be at least one such item.
    //
    SLANG_ASSERT(items.getCount() != 0);

    LookupResult result;
    for (auto item : items)
    {
        AddToLookupResult(result, item);
    }
    return result;
}

void SemanticsVisitor::diagnoseAmbiguousReference(
    OverloadedExpr* overloadedExpr,
    LookupResult const& lookupResult)
{
    getSink()->diagnose(
        overloadedExpr,
        Diagnostics::ambiguousReference,
        lookupResult.items[0].declRef.getName());

    for (auto item : lookupResult.items)
    {
        String declString = ASTPrinter::getDeclSignatureString(item, m_astBuilder);
        getSink()->diagnose(item.declRef, Diagnostics::overloadCandidate, declString);
    }
}

void SemanticsVisitor::diagnoseAmbiguousReference(Expr* expr)
{
    if (auto overloadedExpr = as<OverloadedExpr>(expr))
    {
        diagnoseAmbiguousReference(overloadedExpr, overloadedExpr->lookupResult2);
    }
    else
    {
        getSink()->diagnose(expr, Diagnostics::ambiguousExpression);
    }
}

Expr* SemanticsVisitor::_resolveOverloadedExprImpl(
    OverloadedExpr* overloadedExpr,
    LookupMask mask,
    DiagnosticSink* diagSink)
{
    auto lookupResult = overloadedExpr->lookupResult2;
    SLANG_RELEASE_ASSERT(lookupResult.isValid() && lookupResult.isOverloaded());

    // Take the lookup result we had, and refine it based on what is expected in context.
    //
    // E.g., if there is both a type and a variable named `Foo`, but in context we know
    // that a type is expected, then we can disambiguate by assuming the type is intended.
    //
    lookupResult = refineLookup(lookupResult, mask);

    // Try to filter out overload candidates based on which ones are "better" than one another.
    lookupResult = resolveOverloadedLookup(lookupResult);

    if (!lookupResult.isValid())
    {
        // If we didn't find any symbols after filtering, then just
        // use the original and report errors that way
        return overloadedExpr;
    }

    if (!lookupResult.isOverloaded())
    {
        // If there is only a single item left in the lookup result,
        // then we can proceed to use that item alone as the resolved
        // expression.
        //
        return ConstructLookupResultExpr(
            lookupResult.item,
            overloadedExpr->base,
            overloadedExpr->name,
            overloadedExpr->loc,
            overloadedExpr);
    }

    // Otherwise, we weren't able to resolve the overloading given
    // the information available in context.
    //
    // If the client is asking for us to emit diagnostics about
    // this fact, we should do so here:
    //
    if (diagSink)
    {
        diagnoseAmbiguousReference(overloadedExpr, lookupResult);

        // TODO(tfoley): should we construct a new ErrorExpr here?
        return CreateErrorExpr(overloadedExpr);
    }
    else
    {
        // If the client isn't trying to *force* overload resolution
        // to complete just yet (e.g., they are just trying out one
        // candidate for an overloaded call site), then we return
        // the input expression as-is.
        //
        return overloadedExpr;
    }
}

Expr* SemanticsVisitor::maybeResolveOverloadedExpr(
    Expr* expr,
    LookupMask mask,
    DiagnosticSink* diagSink)
{
    if (IsErrorExpr(expr))
        return expr;

    if (auto overloadedExpr = as<OverloadedExpr>(expr))
    {
        return _resolveOverloadedExprImpl(overloadedExpr, mask, diagSink);
    }
    else
    {
        return expr;
    }
}

Expr* SemanticsVisitor::resolveOverloadedExpr(OverloadedExpr* overloadedExpr, LookupMask mask)
{
    return _resolveOverloadedExprImpl(overloadedExpr, mask, getSink());
}

Type* SemanticsVisitor::tryGetDifferentialType(ASTBuilder* builder, Type* type)
{
    if (auto ptrType = as<PtrTypeBase>(type))
    {
        auto baseDiffType = tryGetDifferentialType(builder, ptrType->getValueType());
        if (!baseDiffType)
            return nullptr;
        return builder->getPtrType(baseDiffType, ptrType->getClass().getName());
    }
    else if (auto arrayType = as<ArrayExpressionType>(type))
    {
        auto baseDiffType = tryGetDifferentialType(builder, arrayType->getElementType());
        if (!baseDiffType)
            return nullptr;
        return builder->getArrayType(baseDiffType, arrayType->getElementCount());
    }

    if (auto declRefType = as<DeclRefType>(type))
    {
        if (auto builtinRequirement =
                declRefType->getDeclRef().getDecl()->findModifier<BuiltinRequirementModifier>())
        {
            if (builtinRequirement->kind == BuiltinRequirementKind::DifferentialType ||
                builtinRequirement->kind == BuiltinRequirementKind::DifferentialPtrType)
            {
                // We are trying to get differential type from a differential type.
                // The result is itself.
                return type;
            }
        }
        type = resolveType(type);
        auto witness = as<SubtypeWitness>(
            tryGetInterfaceConformanceWitness(type, builder->getDifferentiableInterfaceType()));
        if (!witness)
            witness = as<SubtypeWitness>(tryGetInterfaceConformanceWitness(
                type,
                builder->getDifferentiableRefInterfaceType()));
        if (witness)
        {
            auto diffTypeLookupResult = lookUpMember(
                getASTBuilder(),
                this,
                getName("Differential"),
                type,
                nullptr,
                Slang::LookupMask::type,
                Slang::LookupOptions::None);

            diffTypeLookupResult = resolveOverloadedLookup(diffTypeLookupResult);

            if (!diffTypeLookupResult.isValid())
            {
                return nullptr;
            }
            else if (diffTypeLookupResult.isOverloaded())
            {
                return nullptr;
            }
            else
            {
                SharedTypeExpr* baseTypeExpr = m_astBuilder->create<SharedTypeExpr>();
                baseTypeExpr->base.type = type;
                baseTypeExpr->type.type = m_astBuilder->getTypeType(type);

                auto diffTypeExpr = ConstructLookupResultExpr(
                    diffTypeLookupResult.item,
                    baseTypeExpr,
                    declRefType->getDeclRef().getName(),
                    declRefType->getDeclRef().getLoc(),
                    baseTypeExpr);

                return resolveType(ExtractTypeFromTypeRepr(diffTypeExpr));
            }
        }
    }

    if (auto typePack = as<ConcreteTypePack>(type))
    {
        bool anyDifferentiableElement = false;
        List<Type*> diffTypes;
        for (Index i = 0; i < typePack->getTypeCount(); i++)
        {
            auto t = typePack->getElementType(i);
            auto diffType = tryGetDifferentialType(builder, t);
            if (!diffType)
                diffType = m_astBuilder->getVoidType();
            else
                anyDifferentiableElement = true;
            diffTypes.add(diffType);
        }
        if (anyDifferentiableElement)
            return builder->getTypePack(diffTypes.getArrayView());
    }
    return nullptr;
}

bool SemanticsVisitor::canStructBeUsedAsSelfDifferentialType(AggTypeDecl* aggTypeDecl)
{
    // A struct can be used as its own differential type if all its members are differentiable
    // and their differential types are the same as the original types.
    //
    bool canBeUsed = true;
    for (auto member : aggTypeDecl->members)
    {
        if (auto varDecl = as<VarDecl>(member))
        {
            // Try to get the differential type of the member.
            Type* diffType = tryGetDifferentialType(getASTBuilder(), varDecl->getType());
            if (!diffType || !diffType->equals(varDecl->getType()))
            {
                canBeUsed = false;
                break;
            }
        }
    }
    return canBeUsed;
}

void SemanticsVisitor::markSelfDifferentialMembersOfType(AggTypeDecl* parent, Type* type)
{
    // TODO: Handle extensions.
    // Add derivative member attributes to all the fields pointing to themselves.
    for (auto member : parent->getMembersOfType<VarDeclBase>())
    {
        auto derivativeMemberModifier = m_astBuilder->create<DerivativeMemberAttribute>();
        auto fieldLookupExpr = m_astBuilder->create<StaticMemberExpr>();
        fieldLookupExpr->type.type = member->getType();

        auto baseTypeExpr = m_astBuilder->create<SharedTypeExpr>();
        baseTypeExpr->base.type = type;
        auto baseTypeType = m_astBuilder->getOrCreate<TypeType>(type);
        baseTypeExpr->type.type = baseTypeType;
        fieldLookupExpr->baseExpression = baseTypeExpr;

        fieldLookupExpr->declRef = makeDeclRef(member);

        derivativeMemberModifier->memberDeclRef = fieldLookupExpr;
        addModifier(member, derivativeMemberModifier);
    }
}

void SemanticsVisitor::checkDerivativeMemberAttributeReferences(
    VarDeclBase* varDecl,
    DerivativeMemberAttribute* derivativeMemberAttr)
{
    if (derivativeMemberAttr->memberDeclRef)
    {
        // Already checked! This usually happens if this attribute is synthesized by the compiler.
        return;
    }

    SLANG_ASSERT(derivativeMemberAttr->args.getCount() == 1);
    auto checkedExpr =
        dispatchExpr(derivativeMemberAttr->args[0], allowStaticReferenceToNonStaticMember());

    auto memberType = varDecl->type.type; // All types must be fully checked by now.
    auto diffType = getDifferentialType(m_astBuilder, memberType, varDecl->loc);
    auto thisType = calcThisType(makeDeclRef(varDecl->parentDecl));
    if (!thisType)
        return; // Diagnostic should have been emitted previously.

    auto diffThisType = getDifferentialType(m_astBuilder, thisType, derivativeMemberAttr->loc);
    if (!diffThisType)
        return; // Diagnostic should have been emitted previously.

    if (auto declRefExpr = as<DeclRefExpr>(checkedExpr))
    {
        derivativeMemberAttr->memberDeclRef = declRefExpr;
        if (!diffType->equals(declRefExpr->type))
        {
            getSink()->diagnose(
                derivativeMemberAttr,
                Diagnostics::typeMismatch,
                diffType,
                declRefExpr->type);
        }
        if (!varDecl->parentDecl)
        {
            getSink()->diagnose(
                derivativeMemberAttr,
                Diagnostics::attributeNotApplicable,
                diffType,
                declRefExpr->type);
        }
        if (auto memberExpr = as<StaticMemberExpr>(declRefExpr))
        {
            auto baseExprType = memberExpr->baseExpression->type.type;
            if (auto typeType = as<TypeType>(baseExprType))
            {
                if (diffThisType->equals(typeType->getType()))
                {
                    return;
                }
            }
        }
    }
    getSink()->diagnose(
        derivativeMemberAttr,
        Diagnostics::derivativeMemberAttributeMustNameAMemberInExpectedDifferentialType,
        diffThisType);
}

Type* SemanticsVisitor::getDifferentialType(ASTBuilder* builder, Type* type, SourceLoc loc)
{
    auto result = tryGetDifferentialType(builder, type);
    if (!result)
    {
        getSink()->diagnose(
            loc,
            Diagnostics::typeDoesntImplementInterfaceRequirement,
            type,
            getName("Differential"));
        return m_astBuilder->getErrorType();
    }
    return result;
}

void SemanticsVisitor::addDifferentiableTypeToDiffTypeRegistry(Type* type, SubtypeWitness* witness)
{
    SLANG_RELEASE_ASSERT(m_parentDifferentiableAttr);
    if (witness)
    {
        m_parentDifferentiableAttr->addType(type, witness);
    }
}

void SemanticsVisitor::maybeRegisterDifferentiableType(ASTBuilder* builder, Type* type)
{
    if (!builder->isDifferentiableInterfaceAvailable())
    {
        return;
    }

    if (!m_parentDifferentiableAttr)
    {
        return;
    }

    maybeRegisterDifferentiableTypeImplRecursive(builder, type);
}

void SemanticsVisitor::maybeRegisterDifferentiableTypeImplRecursive(ASTBuilder* builder, Type* type)
{
    // Recursively visit the tree of type and register all differentiable types along the way.

    if (as<TypeType>(type))
        return;
    if (!type)
        return;

    // Have we already registered this type? If so we can exit now.
    if (m_parentDifferentiableAttr->m_typeRegistrationWorkingSet.contains(type))
        return;

    m_parentDifferentiableAttr->m_typeRegistrationWorkingSet.add(type);

    // Check for special cases such as PtrTypeBase<T> or Array<T>
    // This could potentially be handled later by simply defining extensions
    // for Ptr<T:IDifferentiable> etc..
    //
    if (auto ptrType = as<PtrTypeBase>(type))
    {
        maybeRegisterDifferentiableTypeImplRecursive(builder, ptrType->getValueType());
        return;
    }

    if (auto arrayType = as<ArrayExpressionType>(type))
    {
        maybeRegisterDifferentiableTypeImplRecursive(builder, arrayType->getElementType());
        // Fall through to register the array type itself.
    }

    if (auto declRefType = as<DeclRefType>(type))
    {
        if (auto subtypeWitness = as<SubtypeWitness>(tryGetInterfaceConformanceWitness(
                type,
                getASTBuilder()->getDifferentiableInterfaceType())))
        {
            addDifferentiableTypeToDiffTypeRegistry(type, subtypeWitness);
        }

        if (auto subtypeWitness = as<SubtypeWitness>(tryGetInterfaceConformanceWitness(
                type,
                getASTBuilder()->getDifferentiableRefInterfaceType())))
        {
            addDifferentiableTypeToDiffTypeRegistry(type, subtypeWitness);
        }

        if (auto aggTypeDeclRef = declRefType->getDeclRef().as<AggTypeDecl>())
        {
            foreachDirectOrExtensionMemberOfType<InheritanceDecl>(
                this,
                aggTypeDeclRef,
                [&](DeclRef<InheritanceDecl> member)
                {
                    auto subType = DeclRefType::create(m_astBuilder, member);
                    maybeRegisterDifferentiableTypeImplRecursive(m_astBuilder, subType);
                });
            foreachDirectOrExtensionMemberOfType<VarDeclBase>(
                this,
                aggTypeDeclRef,
                [&](DeclRef<VarDeclBase> member)
                {
                    auto fieldType = getType(m_astBuilder, member);
                    maybeRegisterDifferentiableTypeImplRecursive(m_astBuilder, fieldType);
                });
        }
        SubstitutionSet(declRefType->getDeclRef())
            .forEachSubstitutionArg(
                [&](Val* arg)
                {
                    if (auto typeArg = as<Type>(arg))
                    {
                        maybeRegisterDifferentiableTypeImplRecursive(m_astBuilder, typeArg);
                    }
                });
        return;
    }

    if (auto typePack = as<ConcreteTypePack>(type))
    {
        for (Index i = 0; i < typePack->getTypeCount(); i++)
            maybeRegisterDifferentiableTypeImplRecursive(builder, typePack->getElementType(i));
        return;
    }

    // General check for types that may not be decl-ref-type, but still have some conformance to
    // IDifferentiable/IDifferentiablePtrType
    if (auto subtypeWitness = as<SubtypeWitness>(tryGetInterfaceConformanceWitness(
            type,
            getASTBuilder()->getDifferentiableInterfaceType())))
    {
        addDifferentiableTypeToDiffTypeRegistry(type, subtypeWitness);
    }
}


Expr* SemanticsVisitor::CheckTerm(Expr* term)
{
    // If we have already checked the expr, don't check again.
    if (term->checked)
    {
        return term;
    }

    auto checkedTerm = _CheckTerm(term);
    checkedTerm->checked = true;

    // Differentiable type checking.
    // TODO: This can be super slow.
    if (this->m_parentFunc && this->m_parentFunc->findModifier<DifferentiableAttribute>())
    {
        maybeRegisterDifferentiableType(getASTBuilder(), checkedTerm->type.type);
    }
    return checkedTerm;
}

Expr* SemanticsVisitor::_CheckTerm(Expr* term)
{
    if (!term)
        return nullptr;

    // The process of checking a term/expression can end up introducing
    // temporaries that need to be added to an outer scope. When jumping
    // into expression checking, we want to check if we already have such
    // a scope in place. If we do, we will re-use it for any sub-expressions.
    // If not, we need to create one.
    //
    if (getExprLocalScope())
    {
        return dispatchExpr(term, *this);
    }

    ExprLocalScope exprLocalScope;

    Expr* checkedTerm = dispatchExpr(term, withExprLocalScope(&exprLocalScope));

    if (IsErrorExpr(checkedTerm))
        return checkedTerm;

    LetExpr* outerMostBinding = exprLocalScope.getOuterMostBinding();
    if (!outerMostBinding)
    {
        return checkedTerm;
    }

    LetExpr* binding = outerMostBinding;
    auto type = checkedTerm->type;
    while (binding)
    {
        binding->type = type;

        if (const auto body = binding->body)
        {
            binding = as<LetExpr>(binding->body);
            SLANG_ASSERT(binding);
            continue;
        }
        else
        {
            binding->body = checkedTerm;
            break;
        }
    }

    return outerMostBinding;
}

Expr* SemanticsVisitor::CreateErrorExpr(Expr* expr)
{
    if (!expr)
    {
        expr = m_astBuilder->create<IncompleteExpr>();
    }
    expr->type = QualType(m_astBuilder->getErrorType());
    return expr;
}

bool SemanticsVisitor::IsErrorExpr(Expr* expr)
{
    // TODO: we may want other cases here...

    if (const auto errorType = as<ErrorType>(expr->type))
        return true;

    return false;
}

Expr* SemanticsVisitor::GetBaseExpr(Expr* expr)
{
    if (auto memberExpr = as<MemberExpr>(expr))
    {
        return memberExpr->baseExpression;
    }
    else if (auto overloadedExpr = as<OverloadedExpr>(expr))
    {
        return overloadedExpr->base;
    }
    else if (auto overloadedExpr2 = as<OverloadedExpr2>(expr))
    {
        return overloadedExpr2->base;
    }
    else if (auto genApp = as<GenericAppExpr>(expr))
    {
        return GetBaseExpr(genApp->functionExpr);
    }
    else if (auto partiallyApplied = as<PartiallyAppliedGenericExpr>(expr))
    {
        return GetBaseExpr(partiallyApplied->originalExpr);
    }
    return nullptr;
}

Expr* SemanticsExprVisitor::visitIncompleteExpr(IncompleteExpr* expr)
{
    expr->type = m_astBuilder->getErrorType();
    return expr;
}

Expr* SemanticsExprVisitor::visitBoolLiteralExpr(BoolLiteralExpr* expr)
{
    expr->type = m_astBuilder->getBoolType();
    return expr;
}

Expr* SemanticsExprVisitor::visitNullPtrLiteralExpr(NullPtrLiteralExpr* expr)
{
    expr->type = m_astBuilder->getNullPtrType();
    return expr;
}

Expr* SemanticsExprVisitor::visitNoneLiteralExpr(NoneLiteralExpr* expr)
{
    expr->type = m_astBuilder->getNoneType();
    return expr;
}

Expr* SemanticsExprVisitor::visitIntegerLiteralExpr(IntegerLiteralExpr* expr)
{
    // The expression might already have a type, determined by its suffix.
    // It it doesn't, we will give it a default type.
    //
    // TODO: We should be careful to pick a "big enough" type
    // based on the size of the value (e.g., don't try to stuff
    // a constant in an `int` if it requires 64 or more bits).
    //
    // The long-term solution here is to give a type to a literal
    // based on the context where it is used, but that requires
    // a more sophisticated type system than we have today.
    //
    if (!expr->type.type)
    {
        expr->type = m_astBuilder->getBuiltinType(expr->suffixType);
    }
    return expr;
}

Expr* SemanticsExprVisitor::visitFloatingPointLiteralExpr(FloatingPointLiteralExpr* expr)
{
    if (!expr->type.type)
    {
        expr->type = m_astBuilder->getBuiltinType(expr->suffixType);
    }
    return expr;
}

Expr* SemanticsExprVisitor::visitStringLiteralExpr(StringLiteralExpr* expr)
{
    expr->type = m_astBuilder->getStringType();
    return expr;
}

IntVal* SemanticsVisitor::getIntVal(IntegerLiteralExpr* expr)
{
    return m_astBuilder->getIntVal(expr->type.type, expr->value);
}

IntVal* SemanticsVisitor::tryConstantFoldExpr(
    SubstExpr<InvokeExpr> invokeExpr,
    ConstantFoldingKind kind,
    ConstantFoldingCircularityInfo* circularityInfo)
{
    // We need all the operands to the expression

    // Check if the callee is an operation that is amenable to constant-folding.
    //
    // For right now we will look for calls to intrinsic functions, and then inspect
    // their names (this is bad and slow).
    auto funcDeclRefExpr = getBaseExpr(invokeExpr).as<DeclRefExpr>();
    if (!funcDeclRefExpr)
        return nullptr;

    auto funcDeclRef = getDeclRef(m_astBuilder, funcDeclRefExpr);
    if (!funcDeclRef)
        return nullptr;
    auto intrinsicMod = funcDeclRef.getDecl()->findModifier<IntrinsicOpModifier>();
    auto implicitCast = funcDeclRef.getDecl()->findModifier<ImplicitConversionModifier>();
    if (!intrinsicMod && !implicitCast)
    {
        // We can't constant fold anything that doesn't map to a builtin
        // operation right now.
        //
        // TODO: we should really allow constant-folding for anything
        // that can be lowered to our bytecode...
        return nullptr;
    }

    // Let's not constant-fold operations with more than a certain number of arguments, for
    // simplicity
    static const int kMaxArgs = 8;
    auto argCount = getArgCount(invokeExpr);
    if (argCount > kMaxArgs)
        return nullptr;

    // Before checking the operation name, let's look at the arguments
    IntVal* argVals[kMaxArgs];
    IntegerLiteralValue constArgVals[kMaxArgs];
    bool allConst = true;
    for (Index a = 0; a < argCount; ++a)
    {
        auto argExpr = getArg(invokeExpr, a);
        auto argVal = tryFoldIntegerConstantExpression(argExpr, kind, circularityInfo);
        if (!argVal)
            return nullptr;

        argVals[a] = argVal;

        if (auto constArgVal = as<ConstantIntVal>(argVal))
        {
            constArgVals[a] = constArgVal->getValue();
        }
        else
        {
            allConst = false;
        }
    }

    if (!allConst)
    {
        // We support a very limited number of operations
        // on "constants" that aren't actually known, to be able to handle a generic
        // that takes an integer `N` but then constructs a vector of size `N+1`.
        //
        // The hard part there is implementing the rules for value unification in the
        // presence of more complicated `IntVal` subclasses, like `SumIntVal`. You'd
        // need inference to be smart enough to know that `2 + N` and `N + 2` are the
        // same value, as are `N + M + 1 + 1` and `M + 2 + N`.
        //
        // This is done by constructing a 'PolynomialIntVal' and rely on its
        // `canonicalize` operation.
        if (implicitCast)
        {
            // We cannot support casting in this case.
            return nullptr;
        }

        auto opName = funcDeclRef.getName();

        // handle binary operators
        if (opName == getName("-"))
        {
            if (argCount == 1)
            {
                return PolynomialIntVal::neg(m_astBuilder, argVals[0]);
            }
            else if (argCount == 2)
            {
                return PolynomialIntVal::sub(m_astBuilder, argVals[0], argVals[1]);
            }
        }
        else if (opName == getName("+"))
        {
            if (argCount == 1)
            {
                return argVals[0];
            }
            else if (argCount == 2)
            {
                return PolynomialIntVal::add(m_astBuilder, argVals[0], argVals[1]);
            }
        }
        else if (opName == getName("*"))
        {
            if (argCount == 2)
            {
                return PolynomialIntVal::mul(m_astBuilder, argVals[0], argVals[1]);
            }
        }
        else if (
            opName == getName("/") || opName == getName("==") || opName == getName(">=") ||
            opName == getName("<=") || opName == getName("!=") || opName == getName(">") ||
            opName == getName("<") || opName == getName("&&") || opName == getName("||") ||
            opName == getName("!") || opName == getName("|") || opName == getName("&") ||
            opName == getName("^") || opName == getName("~") || opName == getName("%") ||
            opName == getName("?:") || opName == getName("<<") || opName == getName(">>"))
        {
            auto result = m_astBuilder->getOrCreate<FuncCallIntVal>(
                invokeExpr.getExpr()->type.type,
                funcDeclRef,
                as<Type>(funcDeclRefExpr.getExpr()->type->substitute(
                    m_astBuilder,
                    funcDeclRefExpr.getSubsts())),
                makeArrayView(argVals, argCount));
            SLANG_RELEASE_ASSERT(result->getFuncType());
            return result;
        }
        return nullptr;
    }

    // At this point, all the operands had simple integer values, so we are golden.
    IntegerLiteralValue resultValue = 0;
    // If this is an implicit cast, we can try to fold.
    if (implicitCast)
    {
        auto targetBasicType = as<BasicExpressionType>(invokeExpr.getExpr()->type.type);
        if (!targetBasicType)
            return nullptr;
        auto foldVal = as<IntVal>(
            TypeCastIntVal::tryFoldImpl(m_astBuilder, targetBasicType, argVals[0], getSink()));
        if (foldVal)
            return foldVal;
        auto result = m_astBuilder->getTypeCastIntVal(targetBasicType, argVals[0]);
        return result;
    }
    else
    {
        auto opName = funcDeclRef.getName();

        // handle binary operators
        if (opName == getName("-"))
        {
            if (argCount == 1)
            {
                resultValue = -constArgVals[0];
            }
            else if (argCount == 2)
            {
                resultValue = constArgVals[0] - constArgVals[1];
            }
        }
        else if (opName == getName("!"))
        {
            resultValue = constArgVals[0] != 0;
        }
        else if (opName == getName("~"))
        {
            resultValue = ~constArgVals[0];
        }

        // simple binary operators
#define CASE(OP)                                          \
    else if (opName == getName(#OP)) do                   \
    {                                                     \
        if (argCount != 2)                                \
            return nullptr;                               \
        resultValue = constArgVals[0] OP constArgVals[1]; \
    }                                                     \
    while (0)

        CASE(+); // TODO: this can also be unary...
        CASE(*);
        CASE(<<);
        CASE(>>);
        CASE(&);
        CASE(|);
        CASE(^);
        CASE(!=);
        CASE(==);
        CASE(>=);
        CASE(<=);
        CASE(<);
        CASE(>);
#undef CASE
        // binary operators with chance of divide-by-zero
        // TODO: issue a suitable error in that case
#define CASE(OP)                                          \
    else if (opName == getName(#OP)) do                   \
    {                                                     \
        if (argCount != 2)                                \
            return nullptr;                               \
        if (!constArgVals[1])                             \
            return nullptr;                               \
        resultValue = constArgVals[0] OP constArgVals[1]; \
    }                                                     \
    while (0)
        CASE(/);
        CASE(%);
#undef CASE
        else if (opName == getName("?:"))
        {
            if (argCount != 3)
                return nullptr;
            if (constArgVals[0] != 0)
                resultValue = constArgVals[1];
            else
                resultValue = constArgVals[2];
        }
        // TODO(tfoley): more cases
        else
        {
            return nullptr;
        }
    }

    IntVal* result = m_astBuilder->getIntVal(invokeExpr.getExpr()->type.type, resultValue);
    return result;
}

bool SemanticsVisitor::_checkForCircularityInConstantFolding(
    Decl* decl,
    ConstantFoldingCircularityInfo* circularityInfo)
{
    // TODO: If the `decl` is already on the chain of `circularityInfo`,
    // then we know that we are trying to recursively fold the
    // same declaration as part of its own definition, and we need
    // to diagnose that as an error.
    //
    for (auto info = circularityInfo; info; info = info->next)
    {
        if (decl == info->decl)
        {
            getSink()->diagnose(decl, Diagnostics::variableUsedInItsOwnDefinition, decl);
            return true;
        }
    }

    return false;
}

IntVal* SemanticsVisitor::tryConstantFoldDeclRef(
    DeclRef<VarDeclBase> const& declRef,
    ConstantFoldingKind kind,
    ConstantFoldingCircularityInfo* circularityInfo)
{
    auto decl = declRef.getDecl();

    if (_checkForCircularityInConstantFolding(decl, circularityInfo))
        return nullptr;

    // In HLSL, `const` is used to mark compile-time constant expressions.
    if (!decl->hasModifier<ConstModifier>())
        return nullptr;

    // The values of specialization constants aren't known at compile time even
    // if they're marked `const`.
    if (decl->hasModifier<SpecializationConstantAttribute>() ||
        decl->hasModifier<VkConstantIdAttribute>())
        return nullptr;

    if (decl->hasModifier<ExternModifier>())
    {
        // Extern const is not considered compile-time constant by the front-end.
        if (kind == ConstantFoldingKind::CompileTime)
            return nullptr;
        // But if we are OK with link-time constants, we can still fold it into a val.
        auto rs = m_astBuilder->getOrCreate<GenericParamIntVal>(
            declRef.substitute(m_astBuilder, declRef.getDecl()->getType()),
            declRef);
        return rs;
    }

    if (isInterfaceRequirement(decl))
    {
        auto witness =
            findThisTypeWitness(SubstitutionSet(declRef), as<InterfaceDecl>(decl->parentDecl));

        auto val = WitnessLookupIntVal::tryFold(
            m_astBuilder,
            witness,
            decl,
            declRef.substitute(m_astBuilder, decl->type.type));
        return as<IntVal>(val);
    }

    if (!getInitExpr(m_astBuilder, declRef))
        return nullptr;

    ensureDecl(declRef.getDecl(), DeclCheckState::DefinitionChecked);
    ConstantFoldingCircularityInfo newCircularityInfo(decl, circularityInfo);
    return tryConstantFoldExpr(getInitExpr(m_astBuilder, declRef), kind, &newCircularityInfo);
}

IntVal* SemanticsVisitor::tryConstantFoldExpr(
    SubstExpr<Expr> expr,
    ConstantFoldingKind kind,
    ConstantFoldingCircularityInfo* circularityInfo)
{

    // Unwrap any "identity" expressions
    while (auto parenExpr = expr.as<ParenExpr>())
    {
        expr = getBaseExpr(parenExpr);
    }

    if (auto intLitExpr = expr.as<IntegerLiteralExpr>())
    {
        return getIntVal(intLitExpr);
    }

    if (auto boolLitExpr = expr.as<BoolLiteralExpr>())
    {
        // If it's a boolean, we allow promotion to int.
        const IntegerLiteralValue value = IntegerLiteralValue(boolLitExpr.getExpr()->value);
        return m_astBuilder->getIntVal(m_astBuilder->getBoolType(), value);
    }

    if (auto arrayLengthExpr = expr.as<GetArrayLengthExpr>())
    {
        if (arrayLengthExpr.getExpr()->arrayExpr && arrayLengthExpr.getExpr()->arrayExpr->type)
        {
            auto type = arrayLengthExpr.getExpr()->arrayExpr->type.type->substitute(
                m_astBuilder,
                expr.getSubsts());
            if (auto arrayType = as<ArrayExpressionType>(type))
            {
                if (!arrayType->isUnsized())
                {
                    if (auto val = as<IntVal>(arrayType->getElementCount()))
                        return val;
                }
            }
        }
    }

    if (auto countOfExpr = expr.as<CountOfExpr>())
    {
        auto type =
            as<Type>(countOfExpr.getExpr()->sizedType->substitute(m_astBuilder, expr.getSubsts()));
        if (type)
            return as<IntVal>(
                CountOfIntVal::tryFold(m_astBuilder, expr.getExpr()->type.type, type));
    }

    // it is possible that we are referring to a generic value param
    if (auto declRefExpr = expr.as<DeclRefExpr>())
    {
        auto declRef = getDeclRef(m_astBuilder, declRefExpr);

        if (auto genericValParamRef = declRef.as<GenericValueParamDecl>())
        {
            Val* valResult = m_astBuilder->getOrCreate<GenericParamIntVal>(
                declRef.substitute(m_astBuilder, genericValParamRef.getDecl()->getType()),
                genericValParamRef);
            valResult = valResult->substitute(m_astBuilder, expr.getSubsts());
            return as<IntVal>(valResult);
        }

        // We may also need to check for references to variables that
        // are defined in a way that can be used as a constant expression:
        if (auto varRef = declRef.as<VarDeclBase>())
        {
            return tryConstantFoldDeclRef(varRef, kind, circularityInfo);
        }
        else if (auto enumRef = declRef.as<EnumCaseDecl>())
        {
            auto enumTypeDecl = enumRef.getParent().getDecl();
            if (enumTypeDecl && !enumTypeDecl->checkState.isBeingChecked())
            {
                ensureDecl(enumRef.getParent(), DeclCheckState::DefinitionChecked);
            }

            // The cases in an `enum` declaration can also be used as constant expressions,
            if (auto tagExpr = getTagExpr(m_astBuilder, enumRef))
            {
                auto enumCaseDecl = enumRef.getDecl();
                if (_checkForCircularityInConstantFolding(enumCaseDecl, circularityInfo))
                    return nullptr;

                ConstantFoldingCircularityInfo newCircularityInfo(enumCaseDecl, circularityInfo);
                auto intVal = as<IntVal>(tryConstantFoldExpr(tagExpr, kind, &newCircularityInfo));
                if (!intVal)
                    return nullptr;
                return as<IntVal>(
                    m_astBuilder->getTypeCastIntVal(enumCaseDecl->getType(), intVal)->resolve());
            }
        }
    }

    SubstExpr<Expr> typeCastOperand;
    if (auto typeCastExpr = expr.as<TypeCastExpr>())
        typeCastOperand = getArg(typeCastExpr, 0);
    else if (auto builtinCastExpr = expr.as<BuiltinCastExpr>())
        typeCastOperand = getBaseExpr(builtinCastExpr);

    if (typeCastOperand)
    {
        auto substType = getType(m_astBuilder, expr);
        if (!substType)
            return nullptr;
        if (!isValidCompileTimeConstantType(substType))
            return nullptr;

        IntVal* val = tryConstantFoldExpr(typeCastOperand, kind, circularityInfo);
        if (!val)
        {
            if (auto floatLitExpr = typeCastOperand.as<FloatingPointLiteralExpr>())
            {
                // When explicitly casting from float type to integer type, let's fold it as
                // an integer value.
                const IntegerLiteralValue value =
                    IntegerLiteralValue(floatLitExpr.getExpr()->value);
                val = m_astBuilder->getIntVal(substType, value);
            }
        }

        if (val)
        {
            if (!expr.getExpr()->type)
                return nullptr;
            auto foldVal =
                as<IntVal>(TypeCastIntVal::tryFoldImpl(m_astBuilder, substType, val, getSink()));
            if (foldVal)
                return foldVal;
            auto result = m_astBuilder->getTypeCastIntVal(substType, val);
            return result;
        }
    }
    else if (auto invokeExpr = expr.as<InvokeExpr>())
    {
        auto val = tryConstantFoldExpr(invokeExpr, kind, circularityInfo);
        if (val)
            return val;
    }
    else if (auto sizeOfLikeExpr = as<SizeOfLikeExpr>(expr.getExpr()))
    {
        ASTNaturalLayoutContext context(getASTBuilder(), nullptr);
        const auto size = context.calcSize(sizeOfLikeExpr->sizedType);
        if (!size)
        {
            return nullptr;
        }

        auto value = as<AlignOfExpr>(sizeOfLikeExpr) ? size.alignment : size.size;

        // We can return as an IntVal
        return getASTBuilder()->getIntVal(expr.getExpr()->type, value);
    }
    else if (auto indexExpr = expr.as<IndexExpr>())
    {
        return tryFoldIndexExpr(indexExpr.getExpr(), kind, circularityInfo);
    }
    return nullptr;
}

IntVal* SemanticsVisitor::tryFoldIndexExpr(
    SubstExpr<IndexExpr> expr,
    ConstantFoldingKind kind,
    ConstantFoldingCircularityInfo* circularityInfo)
{
    // Ad-hoc constant folding for index expressions.
    // TOOD: we should generalize this by extending `Val` to support compile-time constants that are
    // not just integers, but also arrays and structs etc, so that we can independently fold
    // the base expression and the index expression, and then form a ElementExtractVal() from an
    // index expr.
    // For now we just specialize case for array expression that is an initialization list.
    // And this won't work if the array is a link-time constant.
    //
    auto declRefExpr = as<DeclRefExpr>(expr.getExpr()->baseExpression);
    if (!declRefExpr)
        return nullptr;
    auto varDecl = as<VarDecl>(declRefExpr->declRef.getDecl());
    if (!varDecl)
        return nullptr;
    auto type = varDecl->getType();
    if (!type)
        return nullptr;
    auto arrayType = as<ArrayExpressionType>(type);
    if (!arrayType)
        return nullptr;
    if (!varDecl->hasModifier<ConstModifier>())
        return nullptr;
    if (isGlobalDecl(varDecl) && !varDecl->hasModifier<HLSLStaticModifier>())
        return nullptr;
    if (!varDecl->initExpr)
        return nullptr;
    auto arrayContentExpr = as<InitializerListExpr>(varDecl->initExpr);
    if (!arrayContentExpr)
        return nullptr;
    if (expr.getExpr()->indexExprs.getCount() != 1)
        return nullptr;
    auto indexVal = as<ConstantIntVal>(
        tryFoldIntegerConstantExpression(expr.getExpr()->indexExprs[0], kind, circularityInfo));
    if (!indexVal)
        return nullptr;
    auto index = indexVal->getValue();
    if (index < 0 || index >= arrayContentExpr->args.getCount())
        return nullptr;
    auto elementExpr = arrayContentExpr->args[Index(index)];
    return tryFoldIntegerConstantExpression(elementExpr, kind, circularityInfo);
}

IntVal* SemanticsVisitor::tryFoldIntegerConstantExpression(
    SubstExpr<Expr> expr,
    ConstantFoldingKind kind,
    ConstantFoldingCircularityInfo* circularityInfo)
{
    // Check if type is acceptable for an integer constant expression
    //
    if (!isValidCompileTimeConstantType(getType(m_astBuilder, expr)))
        return nullptr;

    // Consider operations that we might be able to constant-fold...
    //
    return tryConstantFoldExpr(expr, kind, circularityInfo);
}

IntVal* SemanticsVisitor::CheckIntegerConstantExpression(
    Expr* inExpr,
    IntegerConstantExpressionCoercionType coercionType,
    Type* expectedType,
    ConstantFoldingKind kind,
    DiagnosticSink* sink)
{
    // No need to issue further errors if the expression didn't even type-check.
    if (IsErrorExpr(inExpr))
        return nullptr;

    // First coerce the expression to the expected type
    Expr* expr = nullptr;
    switch (coercionType)
    {
    case IntegerConstantExpressionCoercionType::SpecificType:
        expr = coerce(CoercionSite::General, expectedType, inExpr);
        break;
    case IntegerConstantExpressionCoercionType::AnyInteger:
        if (isScalarIntegerType(inExpr->type))
            expr = inExpr;
        else if (isEnumType(inExpr->type))
            expr = inExpr;
        else
            expr = coerce(CoercionSite::General, m_astBuilder->getIntType(), inExpr);
        break;
    default:
        break;
    }

    // No need to issue further errors if the type coercion failed.
    if (IsErrorExpr(expr))
        return nullptr;

    auto result = tryFoldIntegerConstantExpression(expr, kind, nullptr);
    if (!result && sink)
    {
        sink->diagnose(expr, Diagnostics::expectedIntegerConstantNotConstant);
    }
    return result;
}

IntVal* SemanticsVisitor::CheckIntegerConstantExpression(
    Expr* inExpr,
    IntegerConstantExpressionCoercionType coercionType,
    Type* expectedType,
    ConstantFoldingKind kind)
{
    return CheckIntegerConstantExpression(inExpr, coercionType, expectedType, kind, getSink());
}

IntVal* SemanticsVisitor::CheckEnumConstantExpression(Expr* expr, ConstantFoldingKind kind)
{
    // No need to issue further errors if the expression didn't even type-check.
    if (IsErrorExpr(expr))
        return nullptr;

    // No need to issue further errors if the type coercion failed.
    if (IsErrorExpr(expr))
        return nullptr;

    auto result = tryConstantFoldExpr(expr, kind, nullptr);
    if (!result)
    {
        getSink()->diagnose(expr, Diagnostics::expectedIntegerConstantNotConstant);
    }
    return result;
}

Expr* SemanticsVisitor::CheckSimpleSubscriptExpr(IndexExpr* subscriptExpr, Type* elementType)
{
    auto baseExpr = subscriptExpr->baseExpression;
    if (subscriptExpr->indexExprs.getCount() < 1)
    {
        getSink()->diagnose(
            subscriptExpr,
            Diagnostics::notEnoughArguments,
            subscriptExpr->indexExprs.getCount(),
            1);
        return CreateErrorExpr(subscriptExpr);
    }
    else if (subscriptExpr->indexExprs.getCount() > 1)
    {
        getSink()->diagnose(
            subscriptExpr,
            Diagnostics::tooManyArguments,
            subscriptExpr->indexExprs.getCount(),
            1);
        return CreateErrorExpr(subscriptExpr);
    }

    auto indexExpr = subscriptExpr->indexExprs[0];

    if (!isScalarIntegerType(indexExpr->type.type))
    {
        getSink()->diagnose(indexExpr, Diagnostics::subscriptIndexNonInteger);
        return CreateErrorExpr(subscriptExpr);
    }

    subscriptExpr->type = QualType(elementType);

    // TODO(tfoley): need to be more careful about this stuff
    subscriptExpr->type.isLeftValue = baseExpr->type.isLeftValue;

    return subscriptExpr;
}

Expr* SemanticsExprVisitor::visitIndexExpr(IndexExpr* subscriptExpr)
{
    bool needDeref = false;
    auto baseExpr = checkBaseForMemberExpr(
        subscriptExpr->baseExpression,
        CheckBaseContext::Subscript,
        needDeref);

    // If the base expression is a type, it means that this is an array declaration,
    // then we should disable short-circuit in case there is logical expression in
    // the subscript
    auto baseType = baseExpr->type.Ptr();
    auto baseTypeType = as<TypeType>(baseType);
    auto subVisitor = (baseTypeType && m_shouldShortCircuitLogicExpr)
                          ? SemanticsVisitor(disableShortCircuitLogicalExpr())
                          : *this;

    for (auto& arg : subscriptExpr->indexExprs)
    {
        arg = subVisitor.CheckTerm(arg);
    }

    // If anything went wrong in the base expression,
    // then just move along...
    if (IsErrorExpr(baseExpr))
        return CreateErrorExpr(subscriptExpr);

    subscriptExpr->baseExpression = baseExpr;

    // Otherwise, we need to look at the type of the base expression,
    // to figure out how subscripting should work.
    if (baseTypeType)
    {
        // We are trying to "index" into a type, so we have an expression like `float[2]`
        // which should be interpreted as resolving to an array type.

        IntVal* elementCount = nullptr;
        if (subscriptExpr->indexExprs.getCount() == 1)
        {
            elementCount = CheckIntegerConstantExpression(
                subscriptExpr->indexExprs[0],
                IntegerConstantExpressionCoercionType::AnyInteger,
                nullptr,
                ConstantFoldingKind::LinkTime);

            // Validate that array size is greater than zero
            if (auto constElementCount = as<ConstantIntVal>(elementCount))
            {
                if (constElementCount->getValue() <= 0)
                {
                    getSink()->diagnose(
                        subscriptExpr->indexExprs[0],
                        Diagnostics::invalidArraySize);
                    return CreateErrorExpr(subscriptExpr);
                }
            }
        }
        else if (subscriptExpr->indexExprs.getCount() != 0)
        {
            getSink()->diagnose(subscriptExpr, Diagnostics::multiDimensionalArrayNotSupported);
        }

        auto elementType = CoerceToUsableType(TypeExp(baseExpr, baseTypeType->getType()), nullptr);
        auto arrayType = getArrayType(m_astBuilder, elementType, elementCount);

        subscriptExpr->type = QualType(m_astBuilder->getTypeType(arrayType));
        return subscriptExpr;
    }
    else if (auto baseArrayType = as<ArrayExpressionType>(baseType))
    {
        return CheckSimpleSubscriptExpr(subscriptExpr, baseArrayType->getElementType());
    }
    else if (auto vecType = as<VectorExpressionType>(baseType))
    {
        return CheckSimpleSubscriptExpr(subscriptExpr, vecType->getElementType());
    }
    else if (auto matType = as<MatrixExpressionType>(baseType))
    {
        // TODO(tfoley): We shouldn't go and recompute
        // row types over and over like this... :(
        auto rowType = createVectorType(matType->getElementType(), matType->getColumnCount());

        return CheckSimpleSubscriptExpr(subscriptExpr, rowType);
    }

    // Default behavior is to look at all available `__subscript`
    // declarations on the type and try to call one of them.

    auto operatorName = getName("operator[]");

    LookupResult lookupResult = lookUpMember(
        m_astBuilder,
        this,
        operatorName,
        baseType,
        m_outerScope,
        LookupMask::Default,
        LookupOptions::NoDeref);
    bool diagnosed = false;
    lookupResult =
        filterLookupResultByVisibilityAndDiagnose(lookupResult, subscriptExpr->loc, diagnosed);
    if (!lookupResult.isValid())
    {
        if (!diagnosed)
            getSink()->diagnose(subscriptExpr, Diagnostics::subscriptNonArray, baseType);
        return CreateErrorExpr(subscriptExpr);
    }
    auto subscriptFuncExpr = createLookupResultExpr(
        operatorName,
        lookupResult,
        subscriptExpr->baseExpression,
        subscriptExpr->loc,
        subscriptExpr);

    InvokeExpr* subscriptCallExpr = m_astBuilder->create<InvokeExpr>();
    subscriptCallExpr->loc = subscriptExpr->loc;
    subscriptCallExpr->functionExpr = subscriptFuncExpr;
    subscriptCallExpr->arguments.addRange(subscriptExpr->indexExprs);
    subscriptCallExpr->argumentDelimeterLocs.addRange(subscriptExpr->argumentDelimeterLocs);

    return CheckInvokeExprWithCheckedOperands(subscriptCallExpr);
}

Expr* SemanticsExprVisitor::visitParenExpr(ParenExpr* expr)
{
    auto base = expr->base;
    base = CheckTerm(base);

    expr->base = base;
    expr->type = base->type;
    return expr;
}

void SemanticsVisitor::maybeDiagnoseThisNotLValue(Expr* expr)
{
    // We will try to handle expressions of the form:
    //
    //      e ::= "this"
    //          | e . name
    //          | e [ expr ]
    //
    // We will unwrap the `e.name` and `e[expr]` cases in a loop.
    Expr* e = expr;
    for (;;)
    {
        if (auto memberExpr = as<MemberExpr>(e))
        {
            e = memberExpr->baseExpression;
        }
        else if (auto subscriptExpr = as<IndexExpr>(e))
        {
            e = subscriptExpr->baseExpression;
        }
        else
        {
            break;
        }
    }
    //
    // Now we check to see if we have a `this` expression,
    // and if it is immutable.
    if (auto thisExpr = as<ThisExpr>(e))
    {
        if (!thisExpr->type.isLeftValue)
        {
            getSink()->diagnoseWithoutSourceView(thisExpr, Diagnostics::thisIsImmutableByDefault);
        }
    }
}

Expr* SemanticsVisitor::checkAssignWithCheckedOperands(AssignExpr* expr)
{
    if (expr->right->type.isWriteOnly)
        getSink()->diagnose(expr, Diagnostics::readingFromWriteOnly);

    expr->left = maybeOpenRef(expr->left);
    auto type = expr->left->type;
    if (auto atomicType = as<AtomicType>(type))
    {
        type = atomicType->getElementType();
    }
    auto right = maybeOpenRef(expr->right);
    expr->right = coerce(CoercionSite::Assignment, type, right);

    if (!expr->left->type.isLeftValue)
    {
        if (as<ErrorType>(type))
        {
            // Don't report an l-value issue on an erroneous expression
        }
        else
        {
            getSink()->diagnose(expr, Diagnostics::assignNonLValue);

            // As a special case, check if the LHS expression is derived
            // from a `this` parameter (implicitly or explicitly), which
            // is immutable. We can give the user a bit more context into
            // what is going on.
            //
            maybeDiagnoseThisNotLValue(expr->left);
        }
    }
    expr->type = type;
    return expr;
}

Expr* SemanticsExprVisitor::visitAssignExpr(AssignExpr* expr)
{
    expr->left = CheckExpr(expr->left);
    expr->right = CheckTerm(expr->right);

    return checkAssignWithCheckedOperands(expr);
}

Expr* SemanticsVisitor::CheckExpr(Expr* uncheckedExpr)
{
    auto checkedTerm = CheckTerm(uncheckedExpr);

    // First, we want to do any disambiguation that is needed in order
    // to turn the `term` into an expression that names a single
    // value (and not something overloaded).
    //
    auto checkedExpr = maybeResolveOverloadedExpr(checkedTerm, LookupMask::Default, getSink());

    // Next, we want to ensure that the `expr` actually has a type
    // that is allowable in an expression context (e.g., make sure
    // that `expr` names a value and not a type).
    //
    // TODO: Implement this step.

    return checkedExpr;
}

static bool _canLValueCoerceScalarType(Type* a, Type* b)
{
    auto basicTypeA = as<BasicExpressionType>(a);
    auto basicTypeB = as<BasicExpressionType>(b);

    if (basicTypeA && basicTypeB)
    {
        const auto& infoA = BaseTypeInfo::getInfo(basicTypeA->getBaseType());
        const auto& infoB = BaseTypeInfo::getInfo(basicTypeB->getBaseType());

        // TODO(JS): Initially this tries to limit where LValueImplict casts happen.
        // We could in principal allow different sizes, as long as we converted to a temprorary
        // and back again.
        //
        // For now we just stick with the simple case.
        // // We only allow on integer types for now. In effect just allowing any size uint/int
        // conversions
        if (infoA.sizeInBytes == infoB.sizeInBytes &&
            (infoA.flags & infoB.flags & BaseTypeInfo::Flag::Integer))
        {
            return true;
        }
    }
    return false;
}

static bool _canLValueCoerce(Type* a, Type* b)
{
    // We can *assume* here that if they are coercable, that dimensions of vectors
    // and matrices match. We might want to assert to be sure...
    SLANG_ASSERT(a != b);
    if (a->astNodeType == b->astNodeType)
    {
        if (auto matA = as<MatrixExpressionType>(a))
        {
            return _canLValueCoerceScalarType(
                matA->getElementType(),
                static_cast<MatrixExpressionType*>(b)->getElementType());
        }
        else if (auto vecA = as<VectorExpressionType>(a))
        {
            return _canLValueCoerceScalarType(
                vecA->getScalarType(),
                static_cast<VectorExpressionType*>(b)->getScalarType());
        }
    }
    return _canLValueCoerceScalarType(a, b);
}


void SemanticsVisitor::compareMemoryQualifierOfParamToArgument(ParamDecl* paramIn, Expr* argIn)
{
    auto arg = as<VarExpr>(argIn);
    if (!paramIn || !arg)
        return;

    auto argDeclRef = arg->declRef;
    if (!argDeclRef)
        return;
    auto argDecl = argDeclRef.getDecl();
    auto argMemMods = argDecl->findModifier<MemoryQualifierSetModifier>();
    if (!argMemMods)
        return;
    uint32_t argQualifiers = argMemMods->getMemoryQualifierBit();

    uint32_t paramQualifiers = 0;
    auto paramMemMods = paramIn->findModifier<MemoryQualifierSetModifier>();
    if (paramMemMods)
        paramQualifiers = paramMemMods->getMemoryQualifierBit();

    if (argQualifiers & MemoryQualifierSetModifier::Flags::kCoherent &&
        !(paramQualifiers & MemoryQualifierSetModifier::Flags::kCoherent))
        getSink()->diagnose(arg, Diagnostics::argumentHasMoreMemoryQualifiersThanParam, "coherent");
    if (argQualifiers & MemoryQualifierSetModifier::Flags::kReadOnly &&
        !(paramQualifiers & MemoryQualifierSetModifier::Flags::kReadOnly))
        getSink()->diagnose(arg, Diagnostics::argumentHasMoreMemoryQualifiersThanParam, "readonly");
    if (argQualifiers & MemoryQualifierSetModifier::Flags::kWriteOnly &&
        !(paramQualifiers & MemoryQualifierSetModifier::Flags::kWriteOnly))
        getSink()->diagnose(
            arg,
            Diagnostics::argumentHasMoreMemoryQualifiersThanParam,
            "writeonly");
    if (argQualifiers & MemoryQualifierSetModifier::Flags::kVolatile &&
        !(paramQualifiers & MemoryQualifierSetModifier::Flags::kVolatile))
        getSink()->diagnose(arg, Diagnostics::argumentHasMoreMemoryQualifiersThanParam, "volatile");
    // dropping a `restrict` qualifier from arguments is allowed in GLSL with memory qualifiers
}

Expr* SemanticsVisitor::CheckInvokeExprWithCheckedOperands(InvokeExpr* expr)
{
    auto rs = ResolveInvoke(expr);
    if (auto invoke = as<InvokeExpr>(rs))
    {
        // if this is still an invoke expression, test arguments passed to inout/out parameter are
        // LValues
        if (auto funcType = as<FuncType>(invoke->functionExpr->type))
        {
            if (!funcType->getErrorType()->equals(m_astBuilder->getBottomType()))
            {
                // If the callee throws, make sure we are inside a try clause.
                if (m_enclosingTryClauseType == TryClauseType::None)
                {
                    getSink()->diagnose(invoke, Diagnostics::mustUseTryClauseToCallAThrowFunc);
                }
            }

            auto funcDeclRefExpr = as<DeclRefExpr>(invoke->functionExpr);
            FunctionDeclBase* funcDeclBase = nullptr;
            if (funcDeclRefExpr)
                funcDeclBase = as<FunctionDeclBase>(funcDeclRefExpr->declRef.getDecl());

            Index paramCount = funcType->getParamCount();
            for (Index pp = 0; pp < paramCount; ++pp)
            {
                auto paramType = funcType->getParamType(pp);
                Expr* argExpr = nullptr;
                ParamDecl* paramDecl = nullptr;
                if (pp < expr->arguments.getCount())
                {
                    argExpr = expr->arguments[pp];
                    if (funcDeclBase)
                        paramDecl = funcDeclBase->getParameters()[pp];
                }
                compareMemoryQualifierOfParamToArgument(paramDecl, argExpr);

                if (as<OutTypeBase>(paramType) || as<RefType>(paramType))
                {
                    // `out`, `inout`, and `ref` parameters currently require
                    // an *exact* match on the type of the argument.
                    //
                    // TODO: relax this requirement by allowing an argument
                    // for an `inout` parameter to be converted in both
                    // directions.
                    //
                    if (argExpr)
                    {
                        if (!argExpr->type.isLeftValue)
                        {
                            auto implicitCastExpr = as<ImplicitCastExpr>(argExpr);

                            // NOTE:
                            // This is currently only enabled for in/inout based scenarios. Ie NOT
                            // ref.
                            //
                            // Depending on the target there can be an issue around atomics.
                            // The fall back transformation with InOut/OutImplicitCast is to
                            // introduce a temporary, and do the work on that and copy back.
                            //
                            // This doesn't work with an atomic. So the work around is to not enable
                            // the transformation with ref types, which atomics are defined on.
                            //
                            // An argument can be made that transformation shouldn't apply to the
                            // ref scenario in general.
                            if (implicitCastExpr && as<OutTypeBase>(paramType) &&
                                _canLValueCoerce(
                                    implicitCastExpr->arguments[0]->type,
                                    implicitCastExpr->type))
                            {
                                // This is to work around issues like
                                //
                                // ```
                                // int a = 0;
                                // uint b = 1;
                                // a += b;
                                // ```
                                // That strictly speaking it's not allowed, but we are going to
                                // allow it for now for situations were the types are uint/int and
                                // vector/matrix varieties of those types
                                //
                                // Then in lowering we are going to insert code to do something like
                                // ```
                                // var OutType: tmp = arg;
                                // f(... tmp);
                                // arg = tmp;
                                // ```

                                TypeCastExpr* lValueImplicitCast;

                                // We want to record if the cast is being used for `out` or
                                // `inout`/`ref` as if it's just `out` we won't need to convert
                                // before passing in.
                                if (as<OutType>(paramType))
                                {
                                    lValueImplicitCast =
                                        getASTBuilder()->create<OutImplicitCastExpr>(
                                            *implicitCastExpr);
                                }
                                else
                                {
                                    lValueImplicitCast =
                                        getASTBuilder()->create<InOutImplicitCastExpr>(
                                            *implicitCastExpr);
                                }

                                // Replace the expression. This should make this situation easier to
                                // detect.
                                expr->arguments[pp] = lValueImplicitCast;
                            }
                            else if (!as<ErrorType>(argExpr->type))
                            {
                                getSink()->diagnose(
                                    argExpr,
                                    Diagnostics::argumentExpectedLValue,
                                    pp);


                                if (implicitCastExpr)
                                {
                                    const DiagnosticInfo* diagnostic = nullptr;

                                    // Try and determine reason for failure
                                    if (as<RefType>(paramType))
                                    {
                                        // Ref types are not allowed to use this mechanism because
                                        // it breaks atomics
                                        diagnostic = &Diagnostics::implicitCastUsedAsLValueRef;
                                    }
                                    else if (!_canLValueCoerce(
                                                 implicitCastExpr->arguments[0]->type,
                                                 implicitCastExpr->type))
                                    {
                                        // We restict what types can use this mechanism - currently
                                        // int/uint and same sized matrix/vectors of those types.
                                        diagnostic = &Diagnostics::implicitCastUsedAsLValueType;
                                    }
                                    else
                                    {
                                        // Fall back, in case there are other reasons...
                                        diagnostic = &Diagnostics::implicitCastUsedAsLValue;
                                    }
                                    getSink()->diagnoseWithoutSourceView(
                                        argExpr,
                                        *diagnostic,
                                        implicitCastExpr->arguments[0]->type,
                                        implicitCastExpr->type);
                                }

                                maybeDiagnoseThisNotLValue(argExpr);
                            }
                        }
                    }
                    else
                    {
                        // There are two ways we could get here, both involving
                        // a call where the number of argument expressions is
                        // less than the number of parameters on the callee:
                        //
                        // 1. There might be fewer arguments than parameters
                        // because the trailing parameters should be defaulted
                        //
                        // 2. There might be fewer arguments than parameters
                        // because the call is incorrect.
                        //
                        // In case (2) an error would have already been diagnosed,
                        // and we don't want to emit another cascading error here.
                        //
                        // In case (1) this implies the user declared an `out`
                        // or `inout` parameter with a default argument expression.
                        // That should be an error, but it should be detected
                        // on the declaration instead of here at the use site.
                        //
                        // Thus, it makes sense to ignore this case here.
                    }
                }
            }

            if (auto higherOrderInvoke = as<DifferentiateExpr>(invoke->functionExpr))
            {
                FunctionDifferentiableLevel requiredLevel;
                if (auto funcDeclExpr = as<DeclRefExpr>(
                        getInnerMostExprFromHigherOrderExpr(higherOrderInvoke, requiredLevel)))
                {
                    auto funcDecl = as<FunctionDeclBase>(funcDeclExpr->declRef.getDecl());
                    if (funcDecl)
                    {
                        if (requiredLevel == FunctionDifferentiableLevel::Forward &&
                            !getShared()->isDifferentiableFunc(funcDecl))
                        {
                            getSink()->diagnose(
                                funcDeclExpr,
                                Diagnostics::functionNotMarkedAsDifferentiable,
                                funcDecl,
                                "forward");
                        }
                        if (requiredLevel == FunctionDifferentiableLevel::Backward &&
                            !getShared()->isBackwardDifferentiableFunc(funcDecl))
                        {
                            getSink()->diagnose(
                                funcDeclExpr,
                                Diagnostics::functionNotMarkedAsDifferentiable,
                                funcDecl,
                                "backward");
                        }
                        if (!isEffectivelyStatic(funcDecl) && !isGlobalDecl(funcDecl))
                        {
                            getSink()->diagnose(
                                invoke->functionExpr,
                                Diagnostics::nonStaticMemberFunctionNotAllowedAsDiffOperand,
                                funcDecl);
                        }
                    }
                }
            }
        }
    }
    return rs;
}


Expr* SemanticsExprVisitor::visitSelectExpr(SelectExpr* expr)
{
    auto result = visitInvokeExpr(expr);
    if (as<ErrorType>(result->type.type))
        return result;
    auto invokeExpr = as<InvokeExpr>(result);
    if (!result)
        return result;
    if (invokeExpr->arguments.getCount() != 3)
        return result;

    if (as<BasicExpressionType>(invokeExpr->arguments[0]->type.type))
    {
        auto newArgs = invokeExpr->arguments;
        expr->arguments.clear();
        expr->arguments = newArgs;
        expr->type = invokeExpr->type;
        return expr;
    }

    if (getParentDifferentiableAttribute())
    {
        // If we are in a differentiable func, issue
        // a diagnostic on use of non short-circuiting select.
        getSink()->diagnose(expr->loc, Diagnostics::useOfNonShortCircuitingOperatorInDiffFunc);
    }
    else
    {
        // For all other functions, we issue a warning for deprecation of vector-typed ?: operator.
        getSink()->diagnose(expr->loc, Diagnostics::useOfNonShortCircuitingOperator);
    }
    return result;
}

Expr* SemanticsExprVisitor::convertToLogicOperatorExpr(InvokeExpr* expr)
{
    LogicOperatorShortCircuitExpr* newExpr = nullptr;

    // If the logic expression is inside the generic parameter list, it cannot support short-circuit
    // which will generate the ifelse branch.
    if (!m_shouldShortCircuitLogicExpr)
    {
        return nullptr;
    }

    if (auto varExpr = as<VarExpr>(expr->functionExpr))
    {
        if ((getText(varExpr->name) == "&&") || (getText(varExpr->name) == "||"))
        {
            // We only use short-circuiting in scalar input, will fall back
            // to non-short-circuiting in vector input.
            bool shortCircuitSupport = true;
            for (auto& arg : expr->arguments)
            {
                if (!as<BasicExpressionType>(arg->type.type))
                {
                    shortCircuitSupport = false;
                }
            }

            if (!shortCircuitSupport)
            {
                return nullptr;
            }

            // We do the cast in the 2nd pass because we want to leave it for 'visitInvokeExpr'
            // to handle if this expression doesn't support short-circuiting.
            for (auto& arg : expr->arguments)
            {
                arg = coerce(CoercionSite::Argument, m_astBuilder->getBoolType(), arg);
            }

            expr->functionExpr = CheckTerm(expr->functionExpr);
            newExpr = m_astBuilder->create<LogicOperatorShortCircuitExpr>();
            if (varExpr->name->text == "&&")
            {
                newExpr->flavor = LogicOperatorShortCircuitExpr::Flavor::And;
            }
            else
            {
                newExpr->flavor = LogicOperatorShortCircuitExpr::Flavor::Or;
            }
            newExpr->loc = expr->loc;
            newExpr->functionExpr = expr->functionExpr;
            newExpr->type = m_astBuilder->getBoolType();
            newExpr->arguments = expr->arguments;
        }
    }

    return newExpr;
}

Expr* SemanticsExprVisitor::visitInvokeExpr(InvokeExpr* expr)
{
    // check the base expression first
    if (!expr->originalFunctionExpr)
        expr->originalFunctionExpr = expr->functionExpr;
    auto treatAsDifferentiableExpr = m_treatAsDifferentiableExpr;
    m_treatAsDifferentiableExpr = nullptr;
    // Next check the argument expressions
    for (auto& arg : expr->arguments)
    {
        arg = CheckExpr(arg);
    }

    // if the expression is '&&' or '||', we will convert it
    // to use short-circuit evaluation.
    if (auto newExpr = convertToLogicOperatorExpr(expr))
        return newExpr;

    expr->functionExpr = CheckTerm(expr->functionExpr);

    if (auto baseType = as<DeclRefType>(expr->functionExpr->type))
    {
        // If callee is a value of DeclRefType, then it is a functor.
        // We need to look for `operator()` member within the type and
        // call that instead.
        auto operatorName = getName("()");

        bool needDeref = false;
        expr->functionExpr = maybeInsertImplicitOpForMemberBase(
            expr->functionExpr,
            CheckBaseContext::Member,
            needDeref);

        LookupResult lookupResult = lookUpMember(
            m_astBuilder,
            this,
            operatorName,
            expr->functionExpr->type,
            m_outerScope,
            LookupMask::Default,
            LookupOptions::NoDeref);
        bool diagnosed = false;
        lookupResult =
            filterLookupResultByVisibilityAndDiagnose(lookupResult, expr->loc, diagnosed);
        if (!lookupResult.isValid())
        {
            if (!diagnosed)
                getSink()->diagnose(expr, Diagnostics::callOperatorNotFound, baseType);
            return CreateErrorExpr(expr);
        }
        auto callFuncExpr = createLookupResultExpr(
            operatorName,
            lookupResult,
            expr->functionExpr,
            expr->loc,
            expr->functionExpr);
        expr->functionExpr = callFuncExpr;
    }

    m_treatAsDifferentiableExpr = treatAsDifferentiableExpr;

    // If we are in a differentiable function, register differential witness tables involved in
    // this call.
    if (m_parentFunc && m_parentFunc->hasModifier<DifferentiableAttribute>())
    {
        for (auto& arg : expr->arguments)
        {
            maybeRegisterDifferentiableType(m_astBuilder, arg->type.type);
        }
    }

    auto checkedExpr = CheckInvokeExprWithCheckedOperands(expr);

    // Perform additional validation for known built-in functions.
    maybeCheckKnownBuiltinInvocation(checkedExpr);

    if (m_parentDifferentiableAttr)
    {
        FunctionDifferentiableLevel callerDiffLevel = FunctionDifferentiableLevel::None;
        if (m_parentFunc)
            callerDiffLevel = getShared()->getFuncDifferentiableLevel(m_parentFunc);

        if (auto checkedInvokeExpr = as<InvokeExpr>(checkedExpr))
        {
            // Register types for final resolved invoke arguments again.
            for (auto& arg : expr->arguments)
            {
                maybeRegisterDifferentiableType(m_astBuilder, arg->type.type);
            }

            if (auto calleeExpr = as<DeclRefExpr>(checkedInvokeExpr->functionExpr))
            {
                if (auto calleeDecl = as<FunctionDeclBase>(calleeExpr->declRef.getDecl()))
                {
                    auto calleeDiffLevel = getShared()->getFuncDifferentiableLevel(calleeDecl);
                    if (calleeDiffLevel >= callerDiffLevel)
                    {
                        if (!m_treatAsDifferentiableExpr)
                        {
                            auto newFuncExpr = getASTBuilder()->create<TreatAsDifferentiableExpr>();
                            newFuncExpr->type = checkedInvokeExpr->type;
                            newFuncExpr->innerExpr = checkedInvokeExpr;
                            newFuncExpr->loc = checkedInvokeExpr->loc;
                            newFuncExpr->flavor = TreatAsDifferentiableExpr::Flavor::Differentiable;
                            checkedExpr = newFuncExpr;
                        }
                        else
                        {
                            getSink()->diagnose(
                                m_treatAsDifferentiableExpr,
                                Diagnostics::useOfNoDiffOnDifferentiableFunc);
                        }
                    }
                }
            }
        }
        maybeRegisterDifferentiableType(m_astBuilder, checkedExpr->type.type);
    }
    return checkedExpr;
}

Expr* SemanticsExprVisitor::visitVarExpr(VarExpr* expr)
{
    // If we've already resolved this expression, don't try again.
    if (expr->declRef)
    {
        if (!expr->type)
            expr->type = GetTypeForDeclRef(expr->declRef, expr->loc);
        return expr;
    }
    expr->type = QualType(m_astBuilder->getErrorType());
    auto lookupResult = lookUp(
        m_astBuilder,
        this,
        expr->name,
        expr->scope,
        LookupMask::Default,
        false,
        getDeclToExcludeFromLookup(),
        getExcludeTransparentMembersFromLookup());

    bool diagnosed = false;
    lookupResult = filterLookupResultByVisibilityAndDiagnose(lookupResult, expr->loc, diagnosed);

    if (expr->name == getSession()->getCompletionRequestTokenName())
    {
        auto scopeKind = CompletionSuggestions::ScopeKind::Expr;
        if (!m_parentFunc)
            scopeKind = CompletionSuggestions::ScopeKind::Decl;
        suggestCompletionItems(scopeKind, lookupResult);
        return expr;
    }

    Expr* resultExpr = expr;

    if (lookupResult.isValid())
    {
        auto lookupResultExpr =
            createLookupResultExpr(expr->name, lookupResult, nullptr, expr->loc, expr);
        if (m_parentLambdaExpr)
            return maybeRegisterLambdaCapture(lookupResultExpr);
        return lookupResultExpr;
    }

    if (!diagnosed)
        getSink()->diagnose(expr, Diagnostics::undefinedIdentifier2, expr->name);

    return resultExpr;
}

Expr* SemanticsExprVisitor::maybeRegisterLambdaCapture(Expr* exprIn)
{
    if (auto memberExpr = as<MemberExpr>(exprIn))
    {
        memberExpr->baseExpression = maybeRegisterLambdaCapture(memberExpr->baseExpression);
        return memberExpr;
    }
    else if (auto subscriptExpr = as<IndexExpr>(exprIn))
    {
        subscriptExpr->baseExpression = maybeRegisterLambdaCapture(subscriptExpr->baseExpression);
        return subscriptExpr;
    }
    auto thisExpr = as<ThisExpr>(exprIn);
    auto varExpr = as<VarExpr>(exprIn);
    if (!thisExpr && !varExpr)
        return exprIn;

    Decl* srcDecl = nullptr;
    if (varExpr)
        srcDecl = as<VarDeclBase>(varExpr->declRef.getDecl());
    else
    {
        // If we see a `this` expression inside a lambda, it is referencing the
        // `this` value of the parent type of the outer function, not the lambda struct
        // itself. Since we don't have a VarDecl representing `this`, we will just use
        // the AggTypeDecl as the key to register in the lambda capture map.
        auto thisTypeDecl = isDeclRefTypeOf<Decl>(thisExpr->type.type);
        if (!thisTypeDecl)
            return exprIn;
        srcDecl = thisTypeDecl.getDecl();
    }

    if (!srcDecl)
        return exprIn;

    if (as<VarDeclBase>(srcDecl) && isGlobalDecl(srcDecl))
        return exprIn;

    auto lambdaScope = m_parentLambdaExpr->paramScopeDecl;
    bool isDefinedInLambdaScope = false;
    for (auto parentDecl = srcDecl->parentDecl; parentDecl; parentDecl = parentDecl->parentDecl)
    {
        if (parentDecl == lambdaScope)
        {
            isDefinedInLambdaScope = true;
            break;
        }
    }
    if (isDefinedInLambdaScope)
        return exprIn;

    // We are referencing something that doesn't belong to the lambda scope, we need to
    // capture it in the current lambda function.

    // If we have already captured the variable, just return the captured variable.
    VarDeclBase* capturedVarDecl = nullptr;
    if (!m_mapSrcDeclToCapturedLambdaDecl->tryGetValue(srcDecl, capturedVarDecl))
    {
        // If not already captured, create a captured variable in the lambda struct decl.
        capturedVarDecl = m_astBuilder->create<VarDecl>();
        capturedVarDecl->nameAndLoc = srcDecl->nameAndLoc;
        SLANG_ASSERT(exprIn->type.type);
        capturedVarDecl->type.type = exprIn->type.type;
        m_mapSrcDeclToCapturedLambdaDecl->add(srcDecl, capturedVarDecl);
        m_parentLambdaDecl->addMember(capturedVarDecl);

        // Is captured value NonCopyable? If so, it needs to be an error.
        if (isNonCopyableType(capturedVarDecl->type.type))
        {
            getSink()->diagnose(
                exprIn,
                Diagnostics::nonCopyableTypeCapturedInLambda,
                capturedVarDecl->type.type);
        }
    }

    // Return a VarExpr referencing the capturedVarDecl.
    auto thisLambdaExpr = m_astBuilder->create<ThisExpr>();
    thisLambdaExpr->scope = m_parentLambdaDecl->ownedScope;
    thisLambdaExpr->type = QualType(DeclRefType::create(m_astBuilder, m_parentLambdaDecl));
    thisLambdaExpr->checked = true;

    auto resultMemberExpr = m_astBuilder->create<MemberExpr>();
    resultMemberExpr->declRef = capturedVarDecl;
    resultMemberExpr->baseExpression = thisLambdaExpr;
    resultMemberExpr->type = exprIn->type;
    resultMemberExpr->loc = exprIn->loc;

    // For captured variables, we need to set the type to be a non-lvalue to prevent
    // lambda expression body from mutating their values.
    resultMemberExpr->type.isLeftValue = false;
    resultMemberExpr->checked = true;
    return resultMemberExpr;
}

Type* SemanticsVisitor::_toDifferentialParamType(Type* primalType)
{
    // Check for type modifiers like 'out' and 'inout'. We need to differentiate the
    // nested type.
    //
    if (auto primalOutType = as<OutType>(primalType))
    {
        return m_astBuilder->getOutType(_toDifferentialParamType(primalOutType->getValueType()));
    }
    else if (auto primalInOutType = as<InOutType>(primalType))
    {
        return m_astBuilder->getInOutType(
            _toDifferentialParamType(primalInOutType->getValueType()));
    }
    return getDifferentialPairType(primalType);
}

Type* SemanticsVisitor::getDifferentialPairType(Type* primalType)
{
    if (auto modifiedType = as<ModifiedType>(primalType))
    {
        if (modifiedType->findModifier<NoDiffModifierVal>())
            return modifiedType->getBase();
    }

    if (auto typePack = as<ConcreteTypePack>(primalType))
    {
        // The differential pair of a type pack should be a type pack of differential pairs.
        List<Type*> diffTypes;
        for (Index i = 0; i < typePack->getTypeCount(); i++)
        {
            auto t = typePack->getElementType(i);
            diffTypes.add(getDifferentialPairType(t));
        }
        return m_astBuilder->getTypePack(diffTypes.getArrayView());
    }
    else if (isAbstractTypePack(primalType))
    {
        // The differential pair of an abstract type pack P should be `expand DifferentialPair<each
        // P>`.
        auto eachType = m_astBuilder->getEachType(primalType);
        auto diffPairEachType = getDifferentialPairType(eachType);
        if (auto expandType = as<ExpandType>(primalType))
        {
            List<Type*> capturedTypePacks;
            for (Index i = 0; i < expandType->getCapturedTypePackCount(); i++)
            {
                capturedTypePacks.add(expandType->getCapturedTypePack(i));
            }
            return m_astBuilder->getExpandType(diffPairEachType, capturedTypePacks.getArrayView());
        }
        else
        {
            return m_astBuilder->getExpandType(diffPairEachType, makeArrayViewSingle(primalType));
        }
    }

    // Get a reference to the builtin 'IDifferentiable' interface
    auto differentiableInterface = getASTBuilder()->getDifferentiableInterfaceType();
    auto differentiableRefInterface = getASTBuilder()->getDifferentiableRefInterfaceType();

    // Check if the provided type inherits from IDifferentiable.
    // If not, return the original type.
    if (auto conformanceWitness = isTypeDifferentiable(primalType))
    {
        if (conformanceWitness->getSup() == differentiableInterface)
        {
            return m_astBuilder->getDifferentialPairType(primalType, conformanceWitness);
        }
        else if (conformanceWitness->getSup() == differentiableRefInterface)
        {
            return m_astBuilder->getDifferentialPtrPairType(primalType, conformanceWitness);
        }
    }
    return primalType;
}

Type* SemanticsVisitor::getForwardDiffFuncType(FuncType* originalType)
{
    // Resolve JVP type here.
    // Note that this type checking needs to be in sync with
    // the auto-generation logic in slang-ir-jvp-diff.cpp
    List<Type*> paramTypes;

    // The JVP return type is float if primal return type is float
    // void otherwise.
    //
    auto resultType = getDifferentialPairType(originalType->getResultType());

    // No support for differentiating function that throw errors, for now.
    SLANG_ASSERT(originalType->getErrorType()->equals(m_astBuilder->getBottomType()));
    auto errorType = originalType->getErrorType();

    for (Index i = 0; i < originalType->getParamCount(); i++)
    {
        if (auto jvpParamType = _toDifferentialParamType(originalType->getParamType(i)))
            paramTypes.add(jvpParamType);
    }
    FuncType* jvpType =
        m_astBuilder->getOrCreate<FuncType>(paramTypes.getArrayView(), resultType, errorType);

    return jvpType;
}

Type* SemanticsVisitor::getBackwardDiffFuncType(FuncType* originalType)
{
    // Resolve backward diff type here.
    // Note that this type checking needs to be in sync with
    // the auto-generation logic in slang-ir-jvp-diff.cpp
    List<Type*> paramTypes;

    // The backward diff return type is void
    //
    auto resultType = m_astBuilder->getVoidType();

    // No support for differentiating function that throw errors, for now.
    SLANG_ASSERT(originalType->getErrorType()->equals(m_astBuilder->getBottomType()));
    auto errorType = originalType->getErrorType();

    for (Index i = 0; i < originalType->getParamCount(); i++)
    {
        if (auto outType = as<OutType>(originalType->getParamType(i)))
        {
            auto diffElementType = tryGetDifferentialType(m_astBuilder, outType->getValueType());
            if (diffElementType)
            {
                paramTypes.add(diffElementType);
            }
            else
            {
                continue;
            }
        }
        else if (auto derivType = _toDifferentialParamType(originalType->getParamType(i)))
        {
            if (as<DifferentialPairType>(derivType))
            {
                // An `in` differentiable parameter becomes an `inout` parameter.
                derivType = m_astBuilder->getInOutType(derivType);
            }
            else if (auto inoutType = as<InOutType>(derivType))
            {
                if (!as<DifferentialPairType>(inoutType->getValueType()))
                {
                    // An `inout` non differentiable parameter becomes an `in` parameter
                    // (removing `out`).
                    derivType = inoutType->getValueType();
                }
            }
            paramTypes.add(derivType);
        }
    }

    // Last parameter is the initial derivative of the original return type
    auto dOutType = tryGetDifferentialType(m_astBuilder, originalType->getResultType());
    if (dOutType)
        paramTypes.add(dOutType);

    return m_astBuilder->getOrCreate<FuncType>(paramTypes.getArrayView(), resultType, errorType);
}

struct HigherOrderInvokeExprCheckingActions
{
    virtual HigherOrderInvokeExpr* createHigherOrderInvokeExpr(SemanticsVisitor* semantics) = 0;
    virtual void fillHigherOrderInvokeExpr(
        HigherOrderInvokeExpr* resultDiffExpr,
        SemanticsVisitor* semantics,
        Expr* funcExpr) = 0;
    FuncType* getBaseFunctionType(SemanticsVisitor* semantics, Expr* funcExpr)
    {
        if (auto funcType = as<FuncType>(funcExpr->type.type))
            return funcType;
        auto astBuilder = semantics->getASTBuilder();
        if (auto declRefExpr = as<DeclRefExpr>(funcExpr))
        {
            if (auto baseFuncGenericDeclRef = declRefExpr->declRef.as<GenericDecl>())
            {
                // Get inner function
                DeclRef<Decl> unspecializedInnerRef = createDefaultSubstitutionsIfNeeded(
                    astBuilder,
                    semantics,
                    astBuilder->getMemberDeclRef(
                        baseFuncGenericDeclRef,
                        getInner(baseFuncGenericDeclRef)));
                auto callableDeclRef = unspecializedInnerRef.as<CallableDecl>();
                if (!callableDeclRef)
                    return nullptr;
                auto funcType = getFuncType(astBuilder, callableDeclRef);
                return funcType;
            }
        }
        return nullptr;
    }
};

struct ForwardDifferentiateExprCheckingActions : HigherOrderInvokeExprCheckingActions
{
    virtual HigherOrderInvokeExpr* createHigherOrderInvokeExpr(SemanticsVisitor* semantics) override
    {
        return semantics->getASTBuilder()->create<ForwardDifferentiateExpr>();
    }
    void fillHigherOrderInvokeExpr(
        HigherOrderInvokeExpr* resultDiffExpr,
        SemanticsVisitor* semantics,
        Expr* funcExpr) override
    {
        resultDiffExpr->baseFunction = funcExpr;
        auto baseFuncType = getBaseFunctionType(semantics, funcExpr);
        if (!baseFuncType)
        {
            resultDiffExpr->type = semantics->getASTBuilder()->getErrorType();
            semantics->getSink()->diagnose(
                funcExpr,
                Diagnostics::expectedFunction,
                funcExpr->type.type);
            return;
        }
        resultDiffExpr->type = semantics->getForwardDiffFuncType(baseFuncType);
        if (auto declRefExpr = as<DeclRefExpr>(getInnerMostExprFromHigherOrderExpr(funcExpr)))
        {
            auto funcDecl = declRefExpr->declRef.as<CallableDecl>().getDecl();
            if (auto genDecl = as<GenericDecl>(declRefExpr->declRef.getDecl()))
            {
                funcDecl = as<CallableDecl>(genDecl->inner);
            }
            if (funcDecl)
            {
                for (auto param : funcDecl->getParameters())
                {
                    resultDiffExpr->newParameterNames.add(param->getName());
                }
            }
        }
    }
};

struct BackwardDifferentiateExprCheckingActions : HigherOrderInvokeExprCheckingActions
{
    virtual HigherOrderInvokeExpr* createHigherOrderInvokeExpr(SemanticsVisitor* semantics) override
    {
        return semantics->getASTBuilder()->create<BackwardDifferentiateExpr>();
    }
    void fillHigherOrderInvokeExpr(
        HigherOrderInvokeExpr* resultDiffExpr,
        SemanticsVisitor* semantics,
        Expr* funcExpr) override
    {
        resultDiffExpr->baseFunction = funcExpr;
        auto baseFuncType = getBaseFunctionType(semantics, funcExpr);
        if (!baseFuncType)
        {
            resultDiffExpr->type = semantics->getASTBuilder()->getErrorType();
            semantics->getSink()->diagnose(
                funcExpr,
                Diagnostics::expectedFunction,
                funcExpr->type.type);
            return;
        }
        resultDiffExpr->type = semantics->getBackwardDiffFuncType(baseFuncType);
        if (auto declRefExpr = as<DeclRefExpr>(getInnerMostExprFromHigherOrderExpr(funcExpr)))
        {
            auto funcDecl = declRefExpr->declRef.as<CallableDecl>().getDecl();
            if (auto genDecl = as<GenericDecl>(declRefExpr->declRef.getDecl()))
            {
                funcDecl = as<CallableDecl>(genDecl->inner);
            }
            if (funcDecl)
            {
                for (auto param : funcDecl->getParameters())
                {
                    if (param->findModifier<NoDiffModifier>())
                    {
                        if (param->findModifier<OutModifier>() &&
                            !param->findModifier<InModifier>() &&
                            !param->findModifier<InOutModifier>())
                            continue;
                    }
                    resultDiffExpr->newParameterNames.add(param->getName());
                }
                resultDiffExpr->newParameterNames.add(semantics->getName("resultGradient"));
            }
        }
    }
};

template<typename ExprASTType>
struct PassthroughHighOrderExprCheckingActionsBase : HigherOrderInvokeExprCheckingActions
{
    virtual HigherOrderInvokeExpr* createHigherOrderInvokeExpr(SemanticsVisitor* semantics) override
    {
        return semantics->getASTBuilder()->create<ExprASTType>();
    }
    void fillHigherOrderInvokeExpr(
        HigherOrderInvokeExpr* resultDiffExpr,
        SemanticsVisitor* semantics,
        Expr* funcExpr) override
    {
        resultDiffExpr->baseFunction = funcExpr;
        auto baseFuncType = getBaseFunctionType(semantics, funcExpr);
        if (!baseFuncType)
        {
            resultDiffExpr->type = semantics->getASTBuilder()->getErrorType();
            semantics->getSink()->diagnose(
                funcExpr,
                Diagnostics::expectedFunction,
                funcExpr->type.type);
            return;
        }
        resultDiffExpr->type = baseFuncType;
        if (auto declRefExpr = as<DeclRefExpr>(getInnerMostExprFromHigherOrderExpr(funcExpr)))
        {
            auto funcDecl = declRefExpr->declRef.as<CallableDecl>().getDecl();
            if (auto genDecl = as<GenericDecl>(declRefExpr->declRef.getDecl()))
            {
                funcDecl = as<CallableDecl>(genDecl->inner);
            }
            if (funcDecl)
            {
                for (auto param : funcDecl->getParameters())
                {
                    resultDiffExpr->newParameterNames.add(param->getName());
                }
            }
        }
    }
};

static Expr* _checkHigherOrderInvokeExpr(
    SemanticsVisitor* semantics,
    HigherOrderInvokeExpr* expr,
    HigherOrderInvokeExprCheckingActions* actions)
{
    // Check/Resolve inner function declaration.
    expr->baseFunction = semantics->CheckTerm(expr->baseFunction);

    auto astBuilder = semantics->getASTBuilder();

    // If base is overloaded expr, we want to return an overloaded expr as check result.
    // This is done by pushing the `differentiate` operator to each item in the overloaded expr.
    if (auto overloadedExpr = as<OverloadedExpr>(expr->baseFunction))
    {
        OverloadedExpr2* result = astBuilder->create<OverloadedExpr2>();
        for (auto item : overloadedExpr->lookupResult2)
        {
            auto lookupResultExpr = semantics->ConstructLookupResultExpr(
                item,
                nullptr,
                overloadedExpr->name,
                overloadedExpr->loc,
                nullptr);
            auto candidateExpr = actions->createHigherOrderInvokeExpr(semantics);
            actions->fillHigherOrderInvokeExpr(candidateExpr, semantics, lookupResultExpr);
            candidateExpr->loc = expr->loc;
            result->candidiateExprs.add(candidateExpr);
        }
        result->type.type = astBuilder->getOverloadedType();
        result->loc = expr->loc;
        return result;
    }
    else if (auto overloadedExpr2 = as<OverloadedExpr2>(expr->baseFunction))
    {
        OverloadedExpr2* result = astBuilder->create<OverloadedExpr2>();
        for (auto item : overloadedExpr2->candidiateExprs)
        {
            auto candidateExpr = actions->createHigherOrderInvokeExpr(semantics);
            actions->fillHigherOrderInvokeExpr(candidateExpr, semantics, item);
            candidateExpr->loc = expr->loc;
            result->candidiateExprs.add(candidateExpr);
        }
        result->type.type = astBuilder->getOverloadedType();
        result->loc = expr->loc;
        return result;
    }

    actions->fillHigherOrderInvokeExpr(expr, semantics, expr->baseFunction);
    return expr;
}

Expr* SemanticsExprVisitor::visitForwardDifferentiateExpr(ForwardDifferentiateExpr* expr)
{
    ForwardDifferentiateExprCheckingActions actions;
    return _checkHigherOrderInvokeExpr(this, expr, &actions);
}

Expr* SemanticsExprVisitor::visitBackwardDifferentiateExpr(BackwardDifferentiateExpr* expr)
{
    BackwardDifferentiateExprCheckingActions actions;
    return _checkHigherOrderInvokeExpr(this, expr, &actions);
}

Expr* SemanticsExprVisitor::visitPrimalSubstituteExpr(PrimalSubstituteExpr* expr)
{
    PassthroughHighOrderExprCheckingActionsBase<PrimalSubstituteExpr> actions;
    return _checkHigherOrderInvokeExpr(this, expr, &actions);
}

Expr* SemanticsExprVisitor::visitDispatchKernelExpr(DispatchKernelExpr* expr)
{
    auto isInt3Type = [this](Type* type)
    {
        auto vectorType = as<VectorExpressionType>(type);
        if (!vectorType)
            return false;
        if (!isIntegerBaseType(getVectorBaseType(vectorType)))
            return false;
        auto constElementCount = as<ConstantIntVal>(vectorType->getElementCount());
        if (!constElementCount)
            return false;
        return constElementCount->getValue() == 3;
    };
    expr->threadGroupSize = dispatchExpr(expr->threadGroupSize, *this);
    if (!isInt3Type(expr->threadGroupSize->type.type))
    {
        getSink()->diagnose(
            expr->threadGroupSize,
            Diagnostics::typeMismatch,
            "uint3",
            expr->threadGroupSize->type);
    }
    expr->dispatchSize = dispatchExpr(expr->dispatchSize, *this);
    if (!isInt3Type(expr->dispatchSize->type.type))
    {
        getSink()->diagnose(
            expr->dispatchSize,
            Diagnostics::typeMismatch,
            "uint3",
            expr->dispatchSize->type);
    }
    PassthroughHighOrderExprCheckingActionsBase<DispatchKernelExpr> actions;
    return _checkHigherOrderInvokeExpr(this, expr, &actions);
}

Expr* SemanticsExprVisitor::visitTreatAsDifferentiableExpr(TreatAsDifferentiableExpr* expr)
{
    auto subContext = withTreatAsDifferentiable(expr);
    expr->innerExpr = dispatchExpr(expr->innerExpr, subContext);
    expr->type = expr->innerExpr->type;
    auto innerExpr = expr->innerExpr;
    while (auto parenExpr = as<ParenExpr>(innerExpr))
    {
        innerExpr = parenExpr->base;
    }
    if (!as<InvokeExpr>(innerExpr) && !as<IndexExpr>(innerExpr))
    {
        getSink()->diagnose(expr, Diagnostics::invalidUseOfNoDiff);
    }
    else if (!m_parentDifferentiableAttr)
    {
        getSink()->diagnose(expr, Diagnostics::cannotUseNoDiffInNonDifferentiableFunc);
    }
    return expr;
}

Expr* SemanticsExprVisitor::visitGetArrayLengthExpr(GetArrayLengthExpr* expr)
{
    expr->arrayExpr = CheckTerm(expr->arrayExpr);
    if (auto arrType = as<ArrayExpressionType>(expr->arrayExpr->type))
    {
        expr->type = m_astBuilder->getIntType();
        if (arrType->isUnsized())
        {
            getSink()->diagnose(expr, Diagnostics::invalidArraySize);
        }
    }
    else
    {
        if (!as<ErrorType>(expr->arrayExpr->type))
        {
            getSink()->diagnose(expr, Diagnostics::typeMismatch, "array", expr->arrayExpr->type);
        }
        expr->type = m_astBuilder->getErrorType();
    }
    return expr;
}

Expr* SemanticsExprVisitor::visitDefaultConstructExpr(DefaultConstructExpr* expr)
{
    return expr;
}

Expr* SemanticsExprVisitor::visitDetachExpr(DetachExpr* expr)
{
    expr->inner = CheckTerm(expr->inner);
    expr->type = expr->inner->type;
    return expr;
}


static bool _isSizeOfType(Type* type)
{
    if (!type)
    {
        return false;
    }

    if (as<ArithmeticExpressionType>(type) || as<ArrayExpressionType>(type) ||
        as<PtrTypeBase>(type) || as<TupleType>(type) || as<GenericDeclRefType>(type))
    {
        return true;
    }

    if (as<DeclRefType>(type))
    {
        return true;
    }

    return false;
}

static bool _isCountOfType(Type* type)
{
    if (!type)
    {
        return false;
    }

    if (isTypePack(type))
    {
        return true;
    }

    if (as<TupleType>(type))
    {
        return true;
    }

    if (as<ArrayExpressionType>(type))
    {
        return true;
    }

    return false;
}

Expr* SemanticsExprVisitor::visitSizeOfLikeExpr(SizeOfLikeExpr* sizeOfLikeExpr)
{
    auto valueExpr = dispatch(sizeOfLikeExpr->value);
    sizeOfLikeExpr->type = m_astBuilder->getIntType();

    Type* type = nullptr;

    if (as<TypeType>(valueExpr->type))
    {
        TypeExp typeExp;
        typeExp.exp = valueExpr;

        auto properTypeExpr = CoerceToProperType(typeExp);

        type = properTypeExpr.type;
    }
    else
    {
        // Is this a proper type?
        TypeExp typeExp(valueExpr->type);
        TypeExp properType = tryCoerceToProperType(typeExp);

        type = properType.type;
    }

    if (as<CountOfExpr>(sizeOfLikeExpr))
    {
        if (!_isCountOfType(type))
        {
            getSink()->diagnose(sizeOfLikeExpr, Diagnostics::countOfArgumentIsInvalid);

            sizeOfLikeExpr->type = m_astBuilder->getErrorType();
            return sizeOfLikeExpr;
        }
    }
    else
    {
        if (!_isSizeOfType(type))
        {
            getSink()->diagnose(sizeOfLikeExpr, Diagnostics::sizeOfArgumentIsInvalid);

            sizeOfLikeExpr->type = m_astBuilder->getErrorType();
            return sizeOfLikeExpr;
        }
    }

    sizeOfLikeExpr->sizedType = type;

    return sizeOfLikeExpr;
}

Expr* SemanticsExprVisitor::visitBuiltinCastExpr(BuiltinCastExpr* expr)
{
    // All builtin cast exprs should already be checked.
    return expr;
}

Expr* SemanticsExprVisitor::visitTypeCastExpr(TypeCastExpr* expr)
{
    if (expr->type)
        return expr;

    // Check the term we are applying first
    auto funcExpr = expr->functionExpr;
    funcExpr = CheckTerm(funcExpr);

    // Now ensure that the term represents a (proper) type.
    TypeExp typeExp;
    typeExp.exp = funcExpr;
    typeExp = CheckProperType(typeExp);

    expr->functionExpr = typeExp.exp;
    expr->type.type = typeExp.type;

    // Next check the argument expression (there should be only one)
    for (auto& arg : expr->arguments)
    {
        arg = CheckTerm(arg);
    }

    // LEGACY FEATURE: As a backwards-compatibility feature
    // for HLSL, we will allow for a cast to a `struct` type
    // from a literal zero, with the semantics of default
    // initialization.
    //
    if (auto declRefType = as<DeclRefType>(typeExp.type))
    {
        if (const auto structDeclRef = as<StructDecl>(declRefType->getDeclRef()))
        {
            if (expr->arguments.getCount() == 1)
            {
                auto arg = expr->arguments[0];
                if (auto intLitArg = as<IntegerLiteralExpr>(arg))
                {
                    if (getIntegerLiteralValue(intLitArg->token) == 0)
                    {
                        // At this point we have confirmed that the cast
                        // has the right form, so we want to apply our special case.
                        //
                        // TODO: If/when we allow for user-defined initializer/constructor
                        // definitions we would have to be careful here because it is
                        // possible that the target type has defined an initializer/constructor
                        // that takes a single `int` parmaeter and means to call that instead.
                        //
                        // For now that should be a non-issue, and in a pinch such a user
                        // could use `T(0)` instead of `(T) 0` to get around this special
                        // HLSL legacy feature.

                        // We will type-check code like:
                        //
                        //      MyStruct s = (MyStruct) 0;
                        //
                        // the same as:
                        //
                        //      MyStruct s = {};
                        //
                        // That is, we construct an empty initializer list, and then coerce
                        // that initializer list expression to the desired type (letting
                        // the code for handling initializer lists work out all of the
                        // details of what is/isn't valid). This choice means we get
                        // to benefit from the existing codegen support for initializer
                        // lists, rather than needing the `(MyStruct) 0` idiom to be
                        // special-cased in later stages of the compiler.
                        //
                        // Note: we use an empty initializer list `{}` instead of an
                        // initializer list with a single zero `{0}`, which is semantically
                        // significant if the first field of `MyStruct` had its own
                        // default initializer defined as part of the `struct` definition.
                        // Basically we have chosen to interpret the "cast from zero" syntax
                        // as sugar for default initialization, and *not* specifically
                        // for zero-initialization. That choice could be revisited if
                        // users express displeasure. For now there isn't enough usage
                        // of explicit default initializers for `struct` fields to
                        // make this a major concern (since they aren't supported in HLSL).
                        //
                        InitializerListExpr* initListExpr =
                            m_astBuilder->create<InitializerListExpr>();
                        initListExpr->loc = expr->loc;
                        initListExpr->useCStyleInitialization = false;
                        auto checkedInitListExpr = visitInitializerListExpr(initListExpr);


                        return coerce(CoercionSite::General, typeExp.type, checkedInitListExpr);
                    }
                }
            }
        }
    }


    // Now process this like any other explicit call (so casts
    // and constructor calls are semantically equivalent).
    return CheckInvokeExprWithCheckedOperands(expr);
}

Expr* SemanticsExprVisitor::visitTryExpr(TryExpr* expr)
{
    auto prevTryClauseType = m_enclosingTryClauseType;
    m_enclosingTryClauseType = expr->tryClauseType;
    expr->base = CheckTerm(expr->base);
    m_enclosingTryClauseType = prevTryClauseType;
    expr->type = expr->base->type;
    if (as<ErrorType>(expr->type))
        return expr;

    auto parentFunc = this->m_parentFunc;
    // TODO: check if the try clause is caught.
    // For now we assume all `try`s are not caught (because we don't have catch yet).
    if (!parentFunc)
    {
        getSink()->diagnose(expr, Diagnostics::uncaughtTryCallInNonThrowFunc);
        return expr;
    }
    if (parentFunc->errorType->equals(m_astBuilder->getBottomType()))
    {
        getSink()->diagnose(expr, Diagnostics::uncaughtTryCallInNonThrowFunc);
        return expr;
    }
    if (!as<InvokeExpr>(expr->base))
    {
        getSink()->diagnose(expr, Diagnostics::tryClauseMustApplyToInvokeExpr);
        return expr;
    }
    auto base = as<InvokeExpr>(expr->base);
    if (auto callee = as<DeclRefExpr>(base->functionExpr))
    {
        if (auto funcCallee = as<FuncDecl>(callee->declRef.getDecl()))
        {
            if (funcCallee->errorType->equals(m_astBuilder->getBottomType()))
            {
                getSink()->diagnose(expr, Diagnostics::tryInvokeCalleeShouldThrow, callee->declRef);
            }
            if (!parentFunc->errorType->equals(funcCallee->errorType))
            {
                getSink()->diagnose(
                    expr,
                    Diagnostics::errorTypeOfCalleeIncompatibleWithCaller,
                    callee->declRef,
                    funcCallee->errorType,
                    parentFunc->errorType);
            }
            return expr;
        }
    }
    getSink()->diagnose(expr, Diagnostics::calleeOfTryCallMustBeFunc);
    return expr;
}

Expr* SemanticsExprVisitor::visitIsTypeExpr(IsTypeExpr* expr)
{
    expr->typeExpr = CheckProperType(expr->typeExpr);
    auto originalVal = CheckTerm(expr->value);
    expr->type = m_astBuilder->getBoolType();
    expr->value = originalVal;

    auto valueType = expr->value->type.type;
    if (auto typeType = as<TypeType>(valueType))
        valueType = typeType->getType();

    // If value is a subtype of `type`, then this expr is always true.
    if (isSubtype(valueType, expr->typeExpr.type, IsSubTypeOptions::None))
    {
        // Instead of returning a BoolLiteralExpr, we use a field to indicate this scenario,
        // so that the language server can still see the original syntax tree.
        expr->constantVal = m_astBuilder->create<BoolLiteralExpr>();
        expr->constantVal->type = m_astBuilder->getBoolType();
        expr->constantVal->value = true;
        expr->constantVal->loc = expr->loc;
        return expr;
    }

    // Otherwise, if the target type is a subtype of value->type, we need to grab the
    // subtype witness for runtime checks.

    expr->value = maybeOpenExistential(originalVal);
    expr->witnessArg = tryGetSubtypeWitness(expr->typeExpr.type, valueType);
    if (expr->witnessArg)
    {
        // For now we can only support the scenario where `expr->value` is an interface type.
        if (!isInterfaceType(originalVal->type))
        {
            getSink()->diagnose(expr, Diagnostics::isOperatorValueMustBeInterfaceType);
        }
        return expr;
    }
    return expr;
}

Expr* SemanticsExprVisitor::visitAsTypeExpr(AsTypeExpr* expr)
{
    TypeExp typeExpr;
    typeExpr.exp = expr->typeExpr;
    typeExpr = CheckProperType(typeExpr);
    expr->value = CheckTerm(expr->value);
    auto optType = m_astBuilder->getOptionalType(typeExpr.type);
    expr->type = optType;

    // If value is a subtype of `type`, then this expr is equivalent to a CastToSuperTypeExpr.
    if (auto witness = tryGetSubtypeWitness(expr->value->type.type, typeExpr.type))
    {
        auto castToSuperType = createCastToSuperTypeExpr(typeExpr.type, expr->value, witness);
        auto makeOptional = m_astBuilder->create<MakeOptionalExpr>();
        makeOptional->loc = expr->loc;
        makeOptional->type = optType;
        makeOptional->value = castToSuperType;
        makeOptional->typeExpr = typeExpr.exp;
        makeOptional->checked = true;
        return makeOptional;
    }

    // If target type is an interface type, we will obtain the witness here for
    // runtime casting.
    expr->witnessArg = tryGetSubtypeWitness(typeExpr.type, expr->value->type.type);
    if (expr->witnessArg)
    {
        // For now we can only support the scenario where `expr->value` is an interface type.
        if (!isInterfaceType(expr->value->type.type))
        {
            getSink()->diagnose(expr, Diagnostics::isOperatorValueMustBeInterfaceType);
        }
        expr->value = maybeOpenExistential(expr->value);
        return expr;
    }

    expr->typeExpr = typeExpr.exp;
    return expr;
}


Expr* SemanticsExprVisitor::visitExpandExpr(ExpandExpr* expr)
{
    OrderedHashSet<Type*> capturedTypePackSet;
    auto subContext = this->withParentExpandExpr(expr, &capturedTypePackSet);
    expr->baseExpr = dispatchExpr(expr->baseExpr, subContext);

    Type* patternType = nullptr;
    bool isTypeExpr = false;
    if (auto typeType = as<TypeType>(expr->baseExpr->type))
    {
        patternType = typeType->getType();
        isTypeExpr = true;
    }
    else
    {
        patternType = expr->baseExpr->type;
    }
    if (as<ErrorType>(patternType))
    {
        expr->type = m_astBuilder->getErrorType();
        return expr;
    }
    if (subContext.getCapturedTypePacks()->getCount() == 0)
    {
        getSink()->diagnose(expr, Diagnostics::expandTermCapturesNoTypePacks);
    }
    List<Type*> capturedTypePacks;
    for (auto capturedType : capturedTypePackSet)
    {
        capturedTypePacks.add(capturedType);
    }
    auto expandType = m_astBuilder->getExpandType(patternType, capturedTypePacks.getArrayView());
    if (isTypeExpr)
        expr->type = m_astBuilder->getTypeType(expandType);
    else
        expr->type = QualType(expandType);
    return expr;
}

Expr* SemanticsExprVisitor::visitEachExpr(EachExpr* expr)
{
    if (!m_parentExpandExpr)
    {
        getSink()->diagnose(expr, Diagnostics::eachExprMustBeInsideExpandExpr);
        expr->type = m_astBuilder->getErrorType();
        return expr;
    }

    expr->baseExpr = CheckTerm(expr->baseExpr);
    bool isTypeNode = false;
    Type* baseType = nullptr;
    if (auto typeType = as<TypeType>(expr->baseExpr->type))
    {
        isTypeNode = true;
        baseType = typeType->getType();
    }
    else
    {
        baseType = expr->baseExpr->type;
    }
    if (as<ErrorType>(baseType))
    {
        expr->type = m_astBuilder->getErrorType();
        return expr;
    }
    if (isTypeNode)
    {
        auto declRefType = as<DeclRefType>(baseType);
        if (!declRefType)
        {
            goto error;
        }
        if (!declRefType->getDeclRef().as<GenericTypePackParamDecl>())
        {
            goto error;
        }
    }
    else
    {
        if (!isTypePack(baseType) && !as<TupleType>(baseType))
            goto error;
    }

    if (auto tupleType = as<TupleType>(baseType))
        baseType = tupleType->getTypePack();

    {
        SLANG_ASSERT(m_capturedTypePacks);
        if (auto baseExpandType = as<ExpandType>(baseType))
        {
            for (Index i = 0; i < baseExpandType->getCapturedTypePackCount(); i++)
            {
                auto capturedType = baseExpandType->getCapturedTypePack(i);
                m_capturedTypePacks->add(capturedType);
            }
        }
        else
        {
            m_capturedTypePacks->add(baseType);
        }
        auto eachType = m_astBuilder->getEachType(baseType);
        if (isTypeNode)
            expr->type = m_astBuilder->getTypeType(eachType);
        else
            expr->type = QualType(eachType);
        return expr;
    }
error:;
    expr->type = m_astBuilder->getErrorType();
    if (!as<ErrorType>(baseType))
    {
        getSink()->diagnose(expr, Diagnostics::expectTypePackAfterEach);
    }
    return expr;
}

Expr* SemanticsExprVisitor::visitLambdaExpr(LambdaExpr* lambdaExpr)
{
    ASTSynthesizer synthesizer = ASTSynthesizer(m_astBuilder, getNamePool());
    synthesizer.pushContainerScope(m_outerScope->containerDecl);

    Dictionary<Decl*, VarDeclBase*> mapSrcDeclToCapturedDecl;
    ensureAllDeclsRec(lambdaExpr->paramScopeDecl, DeclCheckState::DefinitionChecked);
    LambdaDecl* lambdaStructDecl = m_astBuilder->create<LambdaDecl>();
    auto subContext = withParentLambdaExpr(lambdaExpr, lambdaStructDecl, &mapSrcDeclToCapturedDecl);
    addModifier(lambdaStructDecl, m_astBuilder->create<SynthesizedModifier>());
    m_parentFunc->addMember(lambdaStructDecl);
    synthesizer.pushScopeForContainer(lambdaStructDecl);
    lambdaStructDecl->loc = lambdaExpr->loc;
    StringBuilder nameBuilder;
    nameBuilder << "_slang_Lambda_";
    if (m_parentFunc)
    {
        nameBuilder << getText(m_parentFunc->getName());
    }
    nameBuilder << "_";
    nameBuilder << m_parentFunc->members.getCount();
    auto name = getName(nameBuilder.getBuffer());
    lambdaStructDecl->nameAndLoc.name = name;
    lambdaStructDecl->nameAndLoc.loc = lambdaExpr->loc;

    auto funcDecl = m_astBuilder->create<FuncDecl>();
    synthesizer.pushScopeForContainer(funcDecl);
    funcDecl->loc = lambdaExpr->loc;
    funcDecl->nameAndLoc.name = getName("()");
    lambdaStructDecl->addMember(funcDecl);
    lambdaStructDecl->funcDecl = funcDecl;
    addModifier(funcDecl, m_astBuilder->create<SynthesizedModifier>());

    // As we check the body, we will fill in the result type when we visit `ReturnStmt`.
    dispatchStmt(lambdaExpr->bodyStmt, subContext);

    // If the lambda has no return type, we will set it to `void`.
    if (!funcDecl->returnType.type)
        funcDecl->returnType.type = m_astBuilder->getVoidType();

    synthesizer.popScope();
    synthesizer.popScope();

    funcDecl->body = lambdaExpr->bodyStmt;
    for (auto param : lambdaExpr->paramScopeDecl->members)
    {
        funcDecl->addMember(param);
    }

    // LambdaDecl should inherit from `IFunc<>`.
    if (funcDecl->returnType.type)
    {
        auto genApp = m_astBuilder->create<GenericAppExpr>();
        genApp->functionExpr = synthesizer.emitVarExpr(getName("IFunc"));
        auto returnTypeExp = synthesizer.emitStaticTypeExpr(funcDecl->returnType.type);
        genApp->arguments.add(returnTypeExp);
        for (auto param : getMembersOfType<ParamDecl>(m_astBuilder, lambdaExpr->paramScopeDecl))
        {
            auto paramType = getParamTypeWithDirectionWrapper(m_astBuilder, param);
            auto paramTypeExp = synthesizer.emitStaticTypeExpr(paramType);
            genApp->arguments.add(paramTypeExp);
        }
        auto inheritanceDecl = m_astBuilder->create<InheritanceDecl>();
        inheritanceDecl->base.exp = genApp;
        lambdaStructDecl->addMember(inheritanceDecl);
    }

    // Synthesizer the ctor signature, and `IFunc` witness.
    ensureDecl(lambdaStructDecl, DeclCheckState::AttributesChecked);

    // Return an expr that represents `SynthesizedLambdaStruct.__init(captured_args...)`.
    List<Expr*> args;
    Dictionary<VarDeclBase*, Decl*> mapCapturedDeclToSrcDecl;
    for (auto kv : mapSrcDeclToCapturedDecl)
    {
        mapCapturedDeclToSrcDecl[kv.second] = kv.first;
    }
    for (auto capturedField : getMembersOfType<VarDecl>(m_astBuilder, lambdaStructDecl))
    {
        auto src = mapCapturedDeclToSrcDecl[capturedField.getDecl()];
        if (auto srcVarDecl = as<VarDeclBase>(src))
        {
            args.add(synthesizer.emitVarExpr(srcVarDecl));
        }
        else
        {
            args.add(synthesizer.emitThisExpr());
        }
    }
    auto resultLambdaObj = synthesizer.emitCtorInvokeExpr(
        synthesizer.emitStaticTypeExpr(DeclRefType::create(m_astBuilder, lambdaStructDecl)),
        _Move(args));
    auto checkedResultExpr = dispatchExpr(resultLambdaObj, *this);
    return checkedResultExpr;
}

void SemanticsExprVisitor::maybeCheckKnownBuiltinInvocation(Expr* invokeExpr)
{
    auto checkedInvokeExpr = as<InvokeExpr>(invokeExpr);
    if (!checkedInvokeExpr)
        return;
    auto declRefFuncExpr = as<DeclRefExpr>(checkedInvokeExpr->functionExpr);
    if (!declRefFuncExpr)
        return;
    auto callee = declRefFuncExpr->declRef.getDecl();
    if (!callee)
        return;
    auto knownBuiltinAttr = callee->findModifier<KnownBuiltinAttribute>();
    if (!knownBuiltinAttr)
        return;
    if (knownBuiltinAttr->name == "GetAttributeAtVertex")
    {
        if (checkedInvokeExpr->arguments.getCount() != 2)
            return;
        auto vertexAttributeArg = checkedInvokeExpr->arguments[0];
        auto vertexAttributeArgDeclRefExpr = as<DeclRefExpr>(vertexAttributeArg);
        if (!vertexAttributeArgDeclRefExpr)
        {
            getSink()->diagnose(
                invokeExpr,
                Diagnostics::getAttributeAtVertexMustReferToPerVertexInput);
            return;
        }
        auto vertexAttributeArgDecl = vertexAttributeArgDeclRefExpr->declRef.getDecl();
        if (!vertexAttributeArgDecl)
            return;
        if (!vertexAttributeArgDecl->findModifier<PerVertexModifier>() &&
            !vertexAttributeArgDecl->findModifier<HLSLNoInterpolationModifier>())
        {
            getSink()->diagnose(
                vertexAttributeArgDeclRefExpr,
                Diagnostics::getAttributeAtVertexMustReferToPerVertexInput);
            return;
        }
    }
}

Expr* SemanticsVisitor::maybeDereference(Expr* inExpr, CheckBaseContext checkBaseContext)
{
    Expr* expr = inExpr;
    for (;;)
    {
        auto baseType = expr->type;
        if (as<PtrType>(baseType))
        {
            if (checkBaseContext == CheckBaseContext::Subscript)
                return expr;
        }
        auto elementType = getPointedToTypeIfCanImplicitDeref(baseType);
        if (!elementType)
            return expr;
        expr = constructDerefExpr(expr, elementType, inExpr->loc);
    }
}

Expr* SemanticsVisitor::CheckMatrixSwizzleExpr(
    MemberExpr* memberRefExpr,
    Type* baseElementType,
    IntegerLiteralValue baseElementRowCount,
    IntegerLiteralValue baseElementColCount)
{
    // We can have up to 4 swizzles of two elements each
    MatrixCoord elementCoords[4];
    int elementCount = 0;

    bool anyDuplicates = false;
    int zeroIndexOffset = -1;

    if (memberRefExpr->name == getSession()->getCompletionRequestTokenName())
    {
        auto& suggestions = getLinkage()->contentAssistInfo.completionSuggestions;
        suggestions.clear();
        suggestions.scopeKind = CompletionSuggestions::ScopeKind::Swizzle;
        suggestions.swizzleBaseType =
            memberRefExpr->baseExpression ? memberRefExpr->baseExpression->type : nullptr;
        suggestions.elementCount[0] = baseElementRowCount;
        suggestions.elementCount[1] = baseElementColCount;
    }

    String swizzleText = getText(memberRefExpr->name);
    auto cursor = swizzleText.begin();

    // The contents of the string are 0-terminated
    // Every update to cursor corresponds to a check against 0-termination
    while (*cursor)
    {
        // Throw out swizzling with more than 4 output elements
        if (elementCount >= 4)
        {
            return nullptr;
        }
        MatrixCoord elementCoord = {0, 0};

        // Check for the preceding underscore
        if (*cursor++ != '_')
        {
            return nullptr;
        }

        // Check for one or zero indexing
        if (*cursor == 'm')
        {
            // Can't mix one and zero indexing
            if (zeroIndexOffset == 1)
            {
                return nullptr;
            }
            zeroIndexOffset = 0;
            // Increment the index since we saw 'm'
            cursor++;
        }
        else
        {
            // Can't mix one and zero indexing
            if (zeroIndexOffset == 0)
            {
                return nullptr;
            }
            zeroIndexOffset = 1;
        }

        // Check for the ij components
        for (Index j = 0; j < 2; j++)
        {
            auto ch = *cursor++;

            if (ch < '0' || ch > '4')
            {
                return nullptr;
            }
            const int subIndex = ch - '0' - zeroIndexOffset;

            // Check the limit for either the row or column, depending on the step
            IntegerLiteralValue elementLimit;
            if (j == 0)
            {
                elementLimit = baseElementRowCount;
                elementCoord.row = subIndex;
            }
            else
            {
                elementLimit = baseElementColCount;
                elementCoord.col = subIndex;
            }
            // Make sure the index is in range for the source type
            // Account for off-by-one and reject 0 if oneIndexed
            if (subIndex >= elementLimit || subIndex < 0)
            {
                return nullptr;
            }
        }
        // Check if we've seen this index before
        for (int ee = 0; ee < elementCount; ee++)
        {
            if (elementCoords[ee] == elementCoord)
                anyDuplicates = true;
        }

        // add to our list...
        elementCoords[elementCount] = elementCoord;
        elementCount++;
    }

    MatrixSwizzleExpr* swizExpr = m_astBuilder->create<MatrixSwizzleExpr>();
    swizExpr->loc = memberRefExpr->loc;
    swizExpr->base = memberRefExpr->baseExpression;
    swizExpr->memberOpLoc = memberRefExpr->memberOperatorLoc;
    swizExpr->checked = true;

    // Store our list in the actual AST node
    for (int ee = 0; ee < elementCount; ++ee)
    {
        swizExpr->elementCoords[ee] = elementCoords[ee];
    }
    swizExpr->elementCount = elementCount;

    if (elementCount == 1)
    {
        // single-component swizzle produces a scalar
        //
        // Note(tfoley): the official HLSL rules seem to be that it produces
        // a one-component vector, which is then implicitly convertible to
        // a scalar, but that seems like it just adds complexity.
        swizExpr->type = QualType(baseElementType);
    }
    else
    {
        // TODO(tfoley): would be nice to "re-sugar" type
        // here if the input type had a sugared name...
        swizExpr->type = QualType(createVectorType(
            baseElementType,
            m_astBuilder->getIntVal(m_astBuilder->getIntType(), elementCount)));
    }

    // A swizzle can be used as an l-value as long as there
    // were no duplicates in the list of components
    swizExpr->type.isLeftValue = !anyDuplicates;

    return swizExpr;
}

Expr* SemanticsVisitor::CheckMatrixSwizzleExpr(
    MemberExpr* memberRefExpr,
    Type* baseElementType,
    IntVal* baseRowCount,
    IntVal* baseColCount)
{
    if (auto constantRowCount = as<ConstantIntVal>(baseRowCount))
    {
        if (auto constantColCount = as<ConstantIntVal>(baseColCount))
        {
            return CheckMatrixSwizzleExpr(
                memberRefExpr,
                baseElementType,
                constantRowCount->getValue(),
                constantColCount->getValue());
        }
    }
    return nullptr;
}

Expr* SemanticsVisitor::checkTupleSwizzleExpr(MemberExpr* memberExpr, TupleType* baseTupleType)
{
    UInt tupleElementCount = (UInt)baseTupleType->getMemberCount();
    if (tupleElementCount == 0)
        return checkGeneralMemberLookupExpr(memberExpr, baseTupleType);

    if (memberExpr->name == getSession()->getCompletionRequestTokenName())
    {
        auto& suggestions = getLinkage()->contentAssistInfo.completionSuggestions;
        suggestions.clear();
        suggestions.scopeKind = CompletionSuggestions::ScopeKind::Swizzle;
        suggestions.swizzleBaseType =
            memberExpr->baseExpression ? memberExpr->baseExpression->type : nullptr;
        suggestions.elementCount[0] = (Index)tupleElementCount;
        suggestions.elementCount[1] = 0;
        return memberExpr;
    }

    String swizzleText = getText(memberExpr->name);
    auto span = swizzleText.getUnownedSlice();
    Index pos = 0;

    ShortList<uint32_t> elementCoords;

    bool anyDuplicates = false;

    // The contents of the string are 0-terminated
    // Every update to cursor corresponds to a check against 0-termination
    while (pos < span.getLength())
    {
        uint32_t elementCoord;

        // Check for the preceding underscore
        if (span[pos] != '_')
        {
            return checkGeneralMemberLookupExpr(memberExpr, baseTupleType);
        }
        pos++;

        // Parse index.
        if (pos >= span.getLength())
        {
            // Unexpected end of swizzle string, fallback to
            // member lookup.
            return checkGeneralMemberLookupExpr(memberExpr, baseTupleType);
        }

        auto ch = span[pos];

        if (!CharUtil::isDigit(ch))
        {
            // An invalid character in the swizzle is an error, fallback to
            // member lookup.
            return checkGeneralMemberLookupExpr(memberExpr, baseTupleType);
        }
        elementCoord = (uint32_t)StringUtil::parseIntAndAdvancePos(span, pos);

        if (elementCoord >= tupleElementCount)
        {
            getSink()
                ->diagnose(memberExpr, Diagnostics::invalidSwizzleExpr, swizzleText, baseTupleType);
            return CreateErrorExpr(memberExpr);
        }

        // Check if we've seen this index before
        for (int ee = 0; ee < elementCoords.getCount(); ee++)
        {
            if (elementCoords[ee] == elementCoord)
                anyDuplicates = true;
        }

        // add to our list...
        elementCoords.add(elementCoord);
    }

    SwizzleExpr* swizExpr = m_astBuilder->create<SwizzleExpr>();
    swizExpr->loc = memberExpr->loc;
    swizExpr->base = memberExpr->baseExpression;
    swizExpr->elementIndices = _Move(elementCoords);
    swizExpr->memberOpLoc = memberExpr->memberOperatorLoc;

    if (swizExpr->elementIndices.getCount() == 1)
    {
        // single-component swizzle produces a scalar
        //
        swizExpr->type = QualType(baseTupleType->getMember(swizExpr->elementIndices[0]));
    }
    else
    {
        List<Type*> types;
        for (auto index : swizExpr->elementIndices)
        {
            types.add(baseTupleType->getMember(index));
        }
        swizExpr->type = QualType(m_astBuilder->getTupleType(types.getArrayView()));
    }

    // A swizzle can be used as an l-value as long as there
    // were no duplicates in the list of components
    swizExpr->type.isLeftValue = !anyDuplicates;
    return swizExpr;
}

Expr* SemanticsVisitor::CheckSwizzleExpr(
    MemberExpr* memberRefExpr,
    Type* baseElementType,
    IntegerLiteralValue baseElementCount)
{
    IntegerLiteralValue limitElement = baseElementCount;

    ShortList<uint32_t, 4> elementIndices;

    bool anyDuplicates = false;
    bool anyError = false;

    auto swizzleText = getText(memberRefExpr->name);

    for (Index i = 0; i < swizzleText.getLength(); i++)
    {
        auto ch = swizzleText[i];
        int elementIndex = -1;
        switch (ch)
        {
        case 'x':
        case 'r':
            elementIndex = 0;
            break;
        case 'y':
        case 'g':
            elementIndex = 1;
            break;
        case 'z':
        case 'b':
            elementIndex = 2;
            break;
        case 'w':
        case 'a':
            elementIndex = 3;
            break;
        default:
            // An invalid character in the swizzle is an error
            anyError = true;
            break;
        }

        // TODO(tfoley): GLSL requires that all component names
        // come from the same "family"...

        // Make sure the index is in range for the source type
        if (elementIndex >= limitElement)
        {
            anyError = true;
            break;
        }

        // If elementCount is already at 4 stop trying to assign a swizzle element and send an
        // error, we cannot have more valid swizzle elements than 4.
        if (elementIndices.getCount() >= 4)
        {
            anyError = true;
            break;
        }

        // Check if we've seen this index before
        for (int ee = 0; ee < elementIndices.getCount(); ee++)
        {
            if (elementIndices[ee] == (UInt)elementIndex)
                anyDuplicates = true;
        }

        // add to our list...
        elementIndices.add(elementIndex);
    }

    if (anyError)
    {
        return nullptr;
    }

    SwizzleExpr* swizExpr = m_astBuilder->create<SwizzleExpr>();
    swizExpr->loc = memberRefExpr->loc;
    swizExpr->base = memberRefExpr->baseExpression;
    swizExpr->memberOpLoc = memberRefExpr->memberOperatorLoc;
    swizExpr->elementIndices = _Move(elementIndices);

    if (swizExpr->elementIndices.getCount() == 1)
    {
        // single-component swizzle produces a scalar
        //
        // Note(tfoley): the official HLSL rules seem to be that it produces
        // a one-component vector, which is then implicitly convertible to
        // a scalar, but that seems like it just adds complexity.
        swizExpr->type = QualType(baseElementType);
    }
    else
    {
        // TODO(tfoley): would be nice to "re-sugar" type
        // here if the input type had a sugared name...
        swizExpr->type = QualType(createVectorType(
            baseElementType,
            m_astBuilder->getIntVal(
                m_astBuilder->getIntType(),
                swizExpr->elementIndices.getCount())));
    }

    // A swizzle can be used as an l-value as long as there
    // were no duplicates in the list of components
    swizExpr->type.isLeftValue = !anyDuplicates && swizExpr->base && swizExpr->base->type &&
                                 swizExpr->base->type.isLeftValue;

    return swizExpr;
}

Expr* SemanticsVisitor::CheckSwizzleExpr(
    MemberExpr* memberRefExpr,
    Type* baseElementType,
    IntVal* baseElementCount)
{
    if (auto constantElementCount = as<ConstantIntVal>(baseElementCount))
    {
        return CheckSwizzleExpr(memberRefExpr, baseElementType, constantElementCount->getValue());
    }
    else
    {
        return nullptr;
    }
}

Expr* SemanticsVisitor::_lookupStaticMember(DeclRefExpr* expr, Expr* baseExpression)
{
    LookupResult globalLookupResult;
    bool hasErrors = false;
    Expr* base = nullptr;

    // Keep track of namespace scopes we've already looked up in to avoid producing
    // duplicates.
    HashSet<ContainerDecl*> processedNamespaceScopes;

    auto handleLeafCase = [&](DeclRef<Decl> baseDeclRef, Type* type)
    {
        auto aggTypeDeclRef = as<AggTypeDeclBase>(baseDeclRef);

        if (auto namespaceDeclRef = as<NamespaceDeclBase>(baseDeclRef))
        {
            // We are looking up a namespace member.
            //
            // We should lookup in all sibling scopes of the namespace.
            // Another detail here is that we need to skip scopes that are transitively imported.
            // For example, given:
            // ```
            //     module a;
            //     namespace ns { int f_a(); }
            //
            //     module b;
            //     namespace ns { int f_b(); } // will have a sibling scope that refers to a::ns.
            //
            //     module c;
            //     import b;
            //     void test() {ns.f_a(); // should not be valid, because c does not import a. }
            // ```
            // Note that this logic doesn't work nicely with __exported import, but we should
            // consider deprecate this feature anyway.
            //
            auto namespaceModule = getModuleDecl(namespaceDeclRef.getDecl());
            auto thisModule =
                m_outerScope ? getModuleDecl(m_outerScope->containerDecl) : namespaceModule;

            for (auto scope = namespaceDeclRef.getDecl()->ownedScope; scope;
                 scope = scope->nextSibling)
            {
                auto namespaceDecl = as<NamespaceDeclBase>(scope->containerDecl);
                if (!namespaceDecl)
                    continue;
                if (thisModule != namespaceModule &&
                    namespaceModule != getModuleDecl(namespaceDecl))
                    continue;
                if (processedNamespaceScopes.add(scope->containerDecl))
                {
                    LookupResult nsLookupResult = lookUpDirectAndTransparentMembers(
                        m_astBuilder,
                        this,
                        expr->name,
                        namespaceDecl,
                        DeclRef(namespaceDecl),
                        LookupMask::Default,
                        getDeclToExcludeFromLookup());
                    AddToLookupResult(globalLookupResult, nsLookupResult);
                }
            }
        }
        else if (aggTypeDeclRef || type)
        {
            // We are looking up a member inside a type.
            // We want to be careful here because we should only find members
            // that are implicitly or explicitly `static`.
            //
            if (type == nullptr)
                type = DeclRefType::create(m_astBuilder, aggTypeDeclRef);

            if (as<ErrorType>(type))
            {
                return;
            }

            LookupResult lookupResult = lookUpMember(
                m_astBuilder,
                this,
                expr->name,
                type,
                m_outerScope,
                LookupMask::Default,
                LookupOptions::NoDeref);

            // We need to confirm that whatever member we
            // are trying to refer to is usable via static reference.
            //
            // TODO: eventually we might allow a non-static
            // member to be adapted by turning it into something
            // like a closure that takes the missing `this` parameter.
            //
            // E.g., a static reference to a method could be treated
            // as a value with a function type, where the first parameter
            // is `type`.
            //
            // The biggest challenge there is that we'd need to arrange
            // to generate "dispatcher" functions that could be used
            // to implement that function, in the case where we are
            // making a static reference to some kind of polymorphic declaration.
            //
            // (Also, static references to fields/properties would get even
            // harder, because you'd have to know whether a getter/setter/ref-er
            // is needed).
            //
            // For now let's just be expedient and disallow all of that, because
            // we can always add it back in later.

            // If the lookup result is valid, then we want to filter
            // it to just those candidates that can be referenced statically,
            // and ignore any that would only be allowed as instance members.
            //
            if (lookupResult.isValid())
            {
                // We track both the usable items, and whether or
                // not there were any non-static items that need
                // to be ignored.
                //
                bool anyNonStatic = false;
                List<LookupResultItem> staticItems;
                for (auto item : lookupResult)
                {
                    // Is this item usable as a static member?
                    if (isUsableAsStaticMember(item))
                    {
                        // If yes, then it will be part of the output.
                        staticItems.add(item);
                    }
                    else
                    {
                        // If no, then we might need to output an error.
                        anyNonStatic = true;
                    }
                }

                // Was there anything non-static in the list?
                if (anyNonStatic)
                {
                    // If we had some static items, then that's okay,
                    // we just want to use our newly-filtered list.
                    if (staticItems.getCount())
                    {
                        lookupResult.items = staticItems;
                        lookupResult.item = staticItems[0];
                    }
                    else
                    {
                        // Otherwise, it is time to report an error.
                        getSink()->diagnose(
                            expr->loc,
                            Diagnostics::staticRefToNonStaticMember,
                            type,
                            expr->name);
                        hasErrors = true;
                        return;
                    }
                }
                // If there were no non-static items, then the `items`
                // array already represents what we'd get by filtering...

                AddToLookupResult(globalLookupResult, lookupResult);
                base = baseExpression;
            }
        }
    };

    auto handleLeafExpr = [&](Expr* e)
    {
        if (auto nsType = as<NamespaceType>(e->type))
            handleLeafCase(nsType->getDeclRef(), nsType);
        else if (auto aggType = as<DeclRefType>(e->type))
            handleLeafCase(aggType->getDeclRef(), aggType);
        else if (as<TypeType>(e->type))
        {
            auto properType = CoerceToProperType(TypeExp(e));
            if (properType.type)
                handleLeafCase(DeclRef<Decl>(), properType.type);
        }
    };

    auto& baseType = baseExpression->type;
    if (as<ErrorType>(baseType))
    {
        return CreateErrorExpr(expr);
    }

    if (auto overloaded = as<OverloadedExpr>(baseExpression))
    {
        for (auto candidate : overloaded->lookupResult2.items)
            handleLeafCase(candidate.declRef, nullptr);
    }
    else if (auto overloaded2 = as<OverloadedExpr2>(baseExpression))
    {
        for (auto candidate : overloaded2->candidiateExprs)
        {
            handleLeafExpr(candidate);
        }
    }
    else
    {
        handleLeafExpr(baseExpression);
    }

    bool diagnosed = false;
    globalLookupResult =
        filterLookupResultByVisibilityAndDiagnose(globalLookupResult, expr->loc, diagnosed);
    diagnosed |= hasErrors;
    if (!globalLookupResult.isValid())
    {
        return lookupMemberResultFailure(expr, baseType, diagnosed);
    }

    if (expr->name == getSession()->getCompletionRequestTokenName())
    {
        suggestCompletionItems(CompletionSuggestions::ScopeKind::Member, globalLookupResult);
    }
    return createLookupResultExpr(expr->name, globalLookupResult, base, expr->loc, expr);
}

Expr* SemanticsExprVisitor::visitStaticMemberExpr(StaticMemberExpr* expr)
{
    expr->baseExpression = CheckTerm(expr->baseExpression);

    // Not sure this is needed -> but guess someone could do
    expr->baseExpression = maybeDereference(expr->baseExpression, CheckBaseContext::Member);

    // If the base of the member lookup has an interface type
    // *without* a suitable this-type substitution, then we are
    // trying to perform lookup on a value of existential type,
    // and we should "open" the existential here so that we
    // can expose its structure.
    //

    expr->baseExpression = maybeOpenExistential(expr->baseExpression);
    // Do a static lookup
    return _lookupStaticMember(expr, expr->baseExpression);
}

Expr* SemanticsVisitor::lookupMemberResultFailure(
    DeclRefExpr* expr,
    QualType const& baseType,
    bool supressDiagnostic)
{
    // Check it's a member expression
    SLANG_ASSERT(as<StaticMemberExpr>(expr) || as<MemberExpr>(expr));

    if (!supressDiagnostic)
        getSink()->diagnose(expr, Diagnostics::noMemberOfNameInType, expr->name, baseType);
    expr->type = QualType(m_astBuilder->getErrorType());
    return expr;
}

Expr* SemanticsVisitor::maybeInsertImplicitOpForMemberBase(
    Expr* baseExpr,
    CheckBaseContext checkBaseContext,
    bool& outNeedDeref)
{
    auto derefExpr = maybeDereference(baseExpr, checkBaseContext);

    if (derefExpr != baseExpr)
        outNeedDeref = true;

    baseExpr = derefExpr;

    // If the base of the member lookup has an interface type
    // *without* a suitable this-type substitution, then we are
    // trying to perform lookup on a value of existential type,
    // and we should "open" the existential here so that we
    // can expose its structure.
    //
    baseExpr = maybeOpenExistential(baseExpr);

    // In case our base expressin is still overloaded, we can perform
    // some more refinement.
    //
    // Handle the case of an overloaded base expression
    // here, in case we can use the name of the member to
    // disambiguate which of the candidates is meant, or if
    // we can return an overloaded result.
    //
    if (auto overloadedExpr = as<OverloadedExpr>(baseExpr))
    {
        // If a member (dynamic or static) lookup result contains both the actual definition
        // and the interface definition obtained from inheritance, we want to filter out
        // the interface definitions.
        LookupResult filteredLookupResult;
        for (auto lookupResult : overloadedExpr->lookupResult2.items)
        {
            bool shouldRemove = false;
            if (lookupResult.declRef.getParent().as<InterfaceDecl>())
            {
                shouldRemove = true;
            }
            if (lookupResult.declRef.getDecl()->hasModifier<ExtensionExternVarModifier>())
                shouldRemove = true;
            if (!shouldRemove)
            {
                filteredLookupResult.items.add(lookupResult);
            }
        }
        if (filteredLookupResult.items.getCount() == 1)
            filteredLookupResult.item = filteredLookupResult.items.getFirst();
        baseExpr = createLookupResultExpr(
            overloadedExpr->name,
            filteredLookupResult,
            overloadedExpr->base,
            overloadedExpr->loc,
            overloadedExpr);
        // TODO: handle other cases of OverloadedExpr that need filtering.
    }

    return baseExpr;
}

Expr* SemanticsVisitor::checkBaseForMemberExpr(
    Expr* inBaseExpr,
    CheckBaseContext checkBaseContext,
    bool& outNeedDeref)
{
    auto baseExpr = inBaseExpr;
    baseExpr = CheckTerm(baseExpr);

    auto resultBaseExpr =
        maybeInsertImplicitOpForMemberBase(baseExpr, checkBaseContext, outNeedDeref);

    // We might want to register differentiability on any implicit ops that we add in.
    if (this->m_parentFunc && this->m_parentFunc->findModifier<DifferentiableAttribute>())
        maybeRegisterDifferentiableType(getASTBuilder(), resultBaseExpr->type.type);

    return resultBaseExpr;
}

Expr* SemanticsVisitor::checkGeneralMemberLookupExpr(MemberExpr* expr, Type* baseType)
{
    LookupResult lookupResult =
        lookUpMember(m_astBuilder, this, expr->name, baseType, m_outerScope);
    bool diagnosed = false;
    lookupResult = filterLookupResultByVisibilityAndDiagnose(lookupResult, expr->loc, diagnosed);
    if (!lookupResult.isValid())
    {
        return lookupMemberResultFailure(expr, baseType, diagnosed);
    }
    if (expr->name == getSession()->getCompletionRequestTokenName())
    {
        suggestCompletionItems(CompletionSuggestions::ScopeKind::Member, lookupResult);
        if (expr->baseExpression)
        {
            if (auto vectorType = as<VectorExpressionType>(expr->baseExpression->type))
            {
                auto& suggestions = getLinkage()->contentAssistInfo.completionSuggestions;
                suggestions.scopeKind = CompletionSuggestions::ScopeKind::Swizzle;
                suggestions.elementCount[1] = 0;
                suggestions.swizzleBaseType = vectorType;
                if (auto elementCount = as<ConstantIntVal>(vectorType->getElementCount()))
                    suggestions.elementCount[0] = elementCount->getValue();
                else
                    suggestions.elementCount[0] = 1;
            }
            else if (auto scalarType = as<BasicExpressionType>(expr->baseExpression->type))
            {
                auto& suggestions = getLinkage()->contentAssistInfo.completionSuggestions;
                suggestions.scopeKind = CompletionSuggestions::ScopeKind::Swizzle;
                suggestions.elementCount[1] = 0;
                suggestions.elementCount[0] = 1;
                suggestions.swizzleBaseType = scalarType;
            }
        }
    }
    return createLookupResultExpr(expr->name, lookupResult, expr->baseExpression, expr->loc, expr);
}

Expr* SemanticsExprVisitor::visitMemberExpr(MemberExpr* expr)
{
    bool needDeref = false;
    expr->baseExpression =
        checkBaseForMemberExpr(expr->baseExpression, CheckBaseContext::Member, needDeref);

    if (!needDeref && as<DerefMemberExpr>(expr) && !as<PtrType>(expr->baseExpression->type))
    {
        // The user is trying to use the `->` operator on something that can't be
        // dereferenced, so we should diagnose that.
        if (!as<ErrorType>(expr->baseExpression->type))
            getSink()->diagnose(
                expr->memberOperatorLoc,
                Diagnostics::cannotDereferenceType,
                expr->baseExpression->type);
    }

    auto baseType = expr->baseExpression->type;

    // If we are looking up through a modified type, just pass straight
    // through the inner type.
    if (auto modifiedType = as<ModifiedType>(baseType))
        baseType = modifiedType->getBase();

    // Try handle swizzle-able types (scalar,vector,matrix) first.
    // If checking as a swizzle failed for these types,
    // we will fallback to normal member lookup.
    //
    if (auto baseScalarType = as<BasicExpressionType>(baseType))
    {
        // Treat scalar like a 1-element vector when swizzling
        auto swizzle = CheckSwizzleExpr(expr, baseScalarType, 1);
        if (swizzle)
            return swizzle;
    }
    else if (auto baseVecType = as<VectorExpressionType>(baseType))
    {
        auto swizzle =
            CheckSwizzleExpr(expr, baseVecType->getElementType(), baseVecType->getElementCount());
        if (swizzle)
            return swizzle;
    }
    else if (auto baseMatrixType = as<MatrixExpressionType>(baseType))
    {
        auto swizzle = CheckMatrixSwizzleExpr(
            expr,
            baseMatrixType->getElementType(),
            baseMatrixType->getRowCount(),
            baseMatrixType->getColumnCount());
        if (swizzle)
            return swizzle;
    }

    if (as<NamespaceType>(baseType))
    {
        return _lookupStaticMember(expr, expr->baseExpression);
    }
    else if (const auto typeType = as<TypeType>(baseType))
    {
        return _lookupStaticMember(expr, expr->baseExpression);
    }
    else if (as<OverloadedExpr>(expr->baseExpression))
    {
        return _lookupStaticMember(expr, expr->baseExpression);
    }
    else if (as<OverloadedExpr2>(expr->baseExpression))
    {
        return _lookupStaticMember(expr, expr->baseExpression);
    }
    else if (auto baseTupleType = as<TupleType>(baseType))
    {
        return checkTupleSwizzleExpr(expr, baseTupleType);
    }
    else if (as<ErrorType>(baseType))
    {
        return CreateErrorExpr(expr);
    }
    else
    {
        return checkGeneralMemberLookupExpr(expr, baseType);
    }
}

Expr* SemanticsExprVisitor::visitInitializerListExpr(InitializerListExpr* expr)
{
    // If we are assigned a type, expr has already been legalized
    if (expr->type)
        return expr;

    // When faced with an initializer list, we first just check the sub-expressions blindly.
    // Actually making them conform to a desired type will wait for when we know the desired
    // type based on context.

    for (auto& arg : expr->args)
    {
        arg = CheckTerm(arg);
    }

    expr->type = m_astBuilder->getInitializerListType();

    return expr;
}

// Perform semantic checking of an object-oriented `this`
// expression.
Expr* SemanticsExprVisitor::visitThisExpr(ThisExpr* expr)
{
    // A `this` expression will default to immutable.
    expr->type.isLeftValue = false;

    // We will do an upwards search starting in the current
    // scope, looking for a surrounding type (or `extension`)
    // declaration that could be the referrant of the expression.
    auto scope = expr->scope;
    while (scope)
    {
        auto containerDecl = scope->containerDecl;

        if (const auto ctorDecl = as<ConstructorDecl>(containerDecl))
        {
            expr->type.isLeftValue = true;
        }
        else if (const auto setterDecl = as<SetterDecl>(containerDecl))
        {
            expr->type.isLeftValue = true;
        }
        else if (auto funcDeclBase = as<FunctionDeclBase>(containerDecl))
        {
            if (funcDeclBase->hasModifier<MutatingAttribute>())
            {
                expr->type.isLeftValue = true;
            }
            else if (funcDeclBase->hasModifier<RefAttribute>())
            {
                expr->type.isLeftValue = true;
            }
        }
        else if (auto typeOrExtensionDecl = as<AggTypeDeclBase>(containerDecl))
        {
            expr->type.type = calcThisType(makeDeclRef(typeOrExtensionDecl));
            if (m_parentLambdaExpr)
            {
                return maybeRegisterLambdaCapture(expr);
            }
            return expr;
        }
#if 0
            else if (auto aggTypeDecl = as<AggTypeDecl>(containerDecl))
            {
                ensureDecl(aggTypeDecl, DeclCheckState::CanUseAsType);

                // Okay, we are using `this` in the context of an
                // aggregate type, so the expression should be
                // of the corresponding type.
                expr->type.type = DeclRefType::Create(
                    getSession(),
                    makeDeclRef(aggTypeDecl));
                return expr;
            }
            else if (auto extensionDecl = as<ExtensionDecl>(containerDecl))
            {
                ensureDecl(extensionDecl, DeclCheckState::CanUseExtensionTargetType);

                // When `this` is used in the context of an `extension`
                // declaration, then it should refer to an instance of
                // the type being extended.
                //
                // TODO: There is potentially a small gotcha here that
                // lookup through such a `this` expression should probably
                // prioritize members declared in the current extension
                // if there are multiple extensions in scope that add
                // members with the same name...
                //
                expr->type.type = extensionDecl->targetType.type;
                return expr;
            }
#endif

        scope = scope->parent;
    }

    if (auto sink = getSink())
        sink->diagnose(expr, Diagnostics::thisExpressionOutsideOfTypeDecl);

    return CreateErrorExpr(expr);
}

Expr* SemanticsExprVisitor::visitThisTypeExpr(ThisTypeExpr* expr)
{
    auto scope = expr->scope;
    while (scope)
    {
        auto containerDecl = scope->containerDecl;
        if (auto typeOrExtensionDecl = as<AggTypeDeclBase>(containerDecl))
        {
            auto thisType = calcThisType(makeDeclRef(typeOrExtensionDecl));
            auto thisTypeType = m_astBuilder->getTypeType(thisType);

            expr->type.type = thisTypeType;
            return expr;
        }

        scope = scope->parent;
    }

    getSink()->diagnose(expr, Diagnostics::thisTypeOutsideOfTypeDecl);
    return CreateErrorExpr(expr);
}

Expr* SemanticsExprVisitor::visitCastToSuperTypeExpr(CastToSuperTypeExpr* expr)
{
    // CastToSuperType is effectively a struct field.
    // As long as the type is not readonly tagged we
    // can use CastToSuperType as an L-value
    if (!expr->type.hasReadOnlyOnTarget)
        expr->type.isLeftValue = true;
    return expr;
}

Expr* SemanticsExprVisitor::visitReturnValExpr(ReturnValExpr* expr)
{
    auto scope = expr->scope;
    if (scope)
    {
        auto parentFunc = as<CallableDecl>(getParentFunc(scope->containerDecl));
        if (parentFunc)
        {
            if (as<ErrorType>(parentFunc->returnType.type))
            {
                expr->type = parentFunc->returnType.type;
                return expr;
            }
            if (isNonCopyableType(parentFunc->returnType.type))
            {
                expr->type.isLeftValue = true;
                expr->type.type = parentFunc->returnType.type;
                return expr;
            }
        }
    }
    getSink()->diagnose(expr, Diagnostics::returnValNotAvailable);
    expr->type = getASTBuilder()->getErrorType();
    return expr;
}

Expr* SemanticsExprVisitor::visitAndTypeExpr(AndTypeExpr* expr)
{
    // The left and right sides of an `&` for types must both be types.
    //
    expr->left = CheckProperType(expr->left);
    expr->right = CheckProperType(expr->right);

    // TODO: We should enforce some rules here about what is allowed
    // for the `left` and `right` types.
    //
    // For now, the right rule is that they probably need to either
    // be interfaces, or conjunctions thereof.
    //
    // Eventually it may be valuable to support more flexible
    // types in conjunctions, especialy in cases where inheritance
    // gets involved.

    // The result of this expression is an `AndType`, which we need
    // to wrap in a `TypeType` to indicate that the result is the type
    // itself and not a value of  that type.
    //
    auto andType = m_astBuilder->getAndType(expr->left.type, expr->right.type);
    expr->type = m_astBuilder->getTypeType(andType);

    return expr;
}

Expr* SemanticsExprVisitor::visitPointerTypeExpr(PointerTypeExpr* expr)
{
    expr->base = CheckProperType(expr->base);
    if (as<ErrorType>(expr->base.type))
        expr->type = expr->base.type;
    auto ptrType = m_astBuilder->getPtrType(expr->base.type, AddressSpace::UserPointer);
    expr->type = m_astBuilder->getTypeType(ptrType);
    return expr;
}

Expr* SemanticsExprVisitor::visitModifiedTypeExpr(ModifiedTypeExpr* expr)
{
    // The base type should be a proper type (not an expression, generic, etc.)
    //
    expr->base = CheckProperType(expr->base);
    auto baseType = expr->base.type;

    // We will check the modifiers that were applied to the type expression
    // one by one, and collect a list of the ones that should modify the
    // resulting `Type`.
    //
    List<Val*> modifierVals;
    for (auto modifier : expr->modifiers)
    {
        if (auto matrixLayoutModifier = as<MatrixLayoutModifier>(modifier))
        {
            if (auto matrixType = as<MatrixExpressionType>(baseType))
            {
                if (as<ColumnMajorLayoutModifier>(matrixLayoutModifier))
                {
                    baseType = m_astBuilder->getMatrixType(
                        matrixType->getElementType(),
                        matrixType->getRowCount(),
                        matrixType->getColumnCount(),
                        m_astBuilder->getIntVal(
                            m_astBuilder->getIntType(),
                            kMatrixLayoutMode_ColumnMajor));
                }
                else
                {
                    baseType = m_astBuilder->getMatrixType(
                        matrixType->getElementType(),
                        matrixType->getRowCount(),
                        matrixType->getColumnCount(),
                        m_astBuilder->getIntVal(
                            m_astBuilder->getIntType(),
                            kMatrixLayoutMode_RowMajor));
                }
                expr->type = m_astBuilder->getTypeType(baseType);
            }
            else
            {
                getSink()->diagnose(
                    matrixLayoutModifier,
                    Diagnostics::matrixLayoutModifierOnNonMatrixType,
                    baseType);
            }
            continue;
        }
        auto modifierVal = checkTypeModifier(modifier, baseType);
        if (!modifierVal)
            continue;
        modifierVals.add(modifierVal);
    }

    if (modifierVals.getCount())
    {
        auto modifiedType = m_astBuilder->getModifiedType(baseType, modifierVals);
        expr->type = m_astBuilder->getTypeType(modifiedType);
    }
    return expr;
}

Val* SemanticsExprVisitor::checkTypeModifier(Modifier* modifier, Type* type)
{
    SLANG_UNUSED(type);

    if (const auto unormModifier = as<UNormModifier>(modifier))
    {
        // TODO: validate that `type` is either `float` or a vector of `float`s
        return m_astBuilder->getUNormModifierVal();
    }
    else if (const auto snormModifier = as<SNormModifier>(modifier))
    {
        // TODO: validate that `type` is either `float` or a vector of `float`s
        return m_astBuilder->getSNormModifierVal();
    }
    else if (const auto noDiffModifier = as<NoDiffModifier>(modifier))
    {
        return m_astBuilder->getNoDiffModifierVal();
    }
    else
    {
        // TODO: more complete error message here
        getSink()->diagnose(
            modifier,
            Diagnostics::unexpected,
            "unknown type modifier in semantic checking");
        return nullptr;
    }
}

Expr* SemanticsExprVisitor::visitFuncTypeExpr(FuncTypeExpr* expr)
{
    // The input and output to a function type must both be types
    for (auto& t : expr->parameters)
        t = CheckProperType(t);
    expr->result = CheckProperType(expr->result);

    // TODO: Kind checking? Where are we stopping someone passing
    // constraints around as value-inhabitable types

    // The result of this expression is a `FuncType`, which we need
    // to wrap in a `TypeType` to indicate that the result is the type
    // itself and not a value of that type.
    List<Type*> types;
    types.reserve(expr->parameters.getCount());
    for (const auto& t : expr->parameters)
        types.add(t.type);
    auto funcType = m_astBuilder->getFuncType(types.getArrayView(), expr->result.type);
    expr->type = m_astBuilder->getTypeType(funcType);

    return expr;
}

Expr* SemanticsExprVisitor::visitTupleTypeExpr(TupleTypeExpr* expr)
{
    // All tuple members must be types
    for (auto& t : expr->members)
        t = CheckProperType(t);

    // As in the other cases above, wrap in TypeType
    List<Type*> types;
    types.reserve(expr->members.getCount());
    for (auto t : expr->members)
        types.add(t.type);
    auto tupleType = m_astBuilder->getTupleType(types.getArrayView());
    expr->type = m_astBuilder->getTypeType(tupleType);

    return expr;
}

Expr* SemanticsExprVisitor::visitSPIRVAsmExpr(SPIRVAsmExpr* expr)
{
    //
    // Firstly, get the info for this op, the opcode has already been
    // discovered by the parser
    //
    const auto& spirvInfo = getSession()->spirvCoreGrammarInfo;

    // We will iterate over all the operands in all the insts and check
    // them
    bool failed = false;

    // Track %id's that have been defined in this asm block.
    HashSet<Name*> definedIds;

    for (auto& inst : expr->insts)
    {
        // It's not automatically a failure to not have info, we just won't
        // be able to deduce types for operands
        const auto opInfo = spirvInfo->opInfos.lookup(SpvOp(inst.opcode.knownValue));

        if (opInfo && opInfo->numOperandTypes == 0 && inst.operands.getCount())
        {
            failed = true;
            getSink()->diagnose(
                inst.opcode.token,
                Diagnostics::spirvInstructionWithTooManyOperands,
                inst.opcode.token,
                0);
            continue;
        }
        int resultIdIndex = -1;
        if (opInfo)
        {
            resultIdIndex = opInfo->resultIdIndex;
        }
        else if (inst.opcode.flavor == SPIRVAsmOperand::TruncateMarker)
        {
            // If this is __truncate, register the result id in the third operand.
            resultIdIndex = 1;
        }
        else
        {
            // If there is no opInfo, just register all Ids as defined.
            for (auto& operand : inst.operands)
            {
                if (operand.flavor == SPIRVAsmOperand::Id)
                {
                    definedIds.add(operand.token.getName());
                }
            }
        }

        // Register result ID.
        if (resultIdIndex != -1)
        {
            if (inst.operands.getCount() <= resultIdIndex)
            {
                failed = true;
                getSink()->diagnose(
                    inst.opcode.token,
                    Diagnostics::spirvInstructionWithNotEnoughOperands,
                    inst.opcode.token);
                continue;
            }
            auto& resultIdOperand = inst.operands[resultIdIndex];

            if (!definedIds.add(resultIdOperand.token.getName()))
            {
                failed = true;
                getSink()->diagnose(
                    inst.opcode.token,
                    Diagnostics::spirvIdRedefinition,
                    inst.opcode.token);
                continue;
            }
        }

        const bool isLast = &inst == &expr->insts.getLast();
        for (Index operandIndex = 0; operandIndex < inst.operands.getCount(); ++operandIndex)
        {
            // Clamp to the end of the type info array, because the last one will be any variable
            // operands
            const auto invalidOperandKind = SPIRVCoreGrammarInfo::OperandKind{0xff};
            const auto operandType =
                opInfo.has_value()
                    ? opInfo
                          ->operandTypes[std::min(operandIndex, Index(opInfo->numOperandTypes) - 1)]
                    : invalidOperandKind;
            const auto baseOperandType =
                spirvInfo->operandKindUnderneathIds.lookup(operandType).value_or(operandType);
            const auto needsIdWrapper = baseOperandType != operandType;

            const auto check = [&](const auto& go, auto& operand) -> void
            {
                if (operand.flavor == SPIRVAsmOperand::SlangType ||
                    operand.flavor == SPIRVAsmOperand::SampledType)
                {
                    // This is a $$type operand or __sampledType(T)
                    // operand, fill in its TypeExp member.
                    TypeExp& typeExpr = operand.type;
                    typeExpr.exp = operand.expr;
                    typeExpr = CheckProperType(typeExpr);
                    operand.expr = typeExpr.exp;
                }
                else if (
                    operand.flavor == SPIRVAsmOperand::SlangValue ||
                    operand.flavor == SPIRVAsmOperand::SlangImmediateValue ||
                    operand.flavor == SPIRVAsmOperand::SlangValueAddr ||
                    operand.flavor == SPIRVAsmOperand::ImageType ||
                    operand.flavor == SPIRVAsmOperand::SampledImageType ||
                    operand.flavor == SPIRVAsmOperand::ConvertTexel ||
                    operand.flavor == SPIRVAsmOperand::RayPayloadFromLocation ||
                    operand.flavor == SPIRVAsmOperand::RayAttributeFromLocation ||
                    operand.flavor == SPIRVAsmOperand::RayCallableFromLocation)
                {
                    // This is a $expr operand, check the expr
                    operand.expr = dispatch(operand.expr);
                }
                else if (operand.flavor == SPIRVAsmOperand::ResultMarker)
                {
                    // This is the <result-id> marker, check that it only
                    // appears in the last instruction.

                    // TODO: We could consider relaxing this, because SPIR-V
                    // does have forward references for decorations and such
                    if (!isLast)
                    {
                        getSink()->diagnose(operand.token, Diagnostics::misplacedResultIdMarker);
                        getSink()->diagnoseWithoutSourceView(
                            expr,
                            Diagnostics::considerOpCopyObject);
                    }
                }
                else if (operand.flavor == SPIRVAsmOperand::NamedValue)
                {
                    // First try and look it up with the knowledge of this operand's type
                    auto enumValue =
                        spirvInfo->allEnums.lookup({baseOperandType, operand.token.getContent()});
                    // Then fall back to with the type prefix
                    if (!enumValue)
                        enumValue =
                            spirvInfo->allEnumsWithTypePrefix.lookup(operand.token.getContent());
                    // Then see if it's an opcode (for OpSpecialize)
                    if (!enumValue)
                        enumValue = spirvInfo->opcodes.lookup(operand.token.getContent());
                    if (inst.opcode.knownValue == SpvOpExtInst)
                    {
                        if (!enumValue)
                        {
                            GLSLstd450 val;
                            if (lookupGLSLstd450(operand.token.getContent(), val))
                            {
                                enumValue = (SpvWord)val;
                            }
                        }
                    }
                    if (!enumValue)
                    {
                        failed = true;
                        getSink()->diagnose(
                            operand.token,
                            Diagnostics::spirvUnableToResolveName,
                            operand.token.getContent());
                        return;
                    }

                    operand.knownValue = *enumValue;
                    operand.wrapInId = needsIdWrapper;
                }
                else if (operand.flavor == SPIRVAsmOperand::BuiltinVar)
                {
                    operand.type = CheckProperType(operand.type);
                    auto builtinVarKind =
                        spirvInfo->allEnums.lookup(SPIRVCoreGrammarInfo::QualifiedEnumName{
                            spirvInfo->operandKinds.lookup(UnownedStringSlice("BuiltIn")).value(),
                            operand.token.getContent()});
                    if (!builtinVarKind)
                    {
                        failed = true;
                        getSink()->diagnose(
                            operand.token,
                            Diagnostics::spirvUnableToResolveName,
                            operand.token.getContent());
                        return;
                    }
                    operand.knownValue = builtinVarKind.value();
                }
                else if (operand.flavor == SPIRVAsmOperand::Id)
                {
                    if (!definedIds.contains(operand.token.getName()))
                    {
                        failed = true;
                        getSink()->diagnose(
                            operand.token,
                            Diagnostics::spirvUndefinedId,
                            operand.token);
                        return;
                    }
                }
                if (operand.bitwiseOrWith.getCount() &&
                    operand.flavor != SPIRVAsmOperand::Literal &&
                    operand.flavor != SPIRVAsmOperand::NamedValue)
                {
                    failed = true;
                    getSink()->diagnose(operand.token, Diagnostics::spirvNonConstantBitwiseOr);
                }
                for (auto& o : operand.bitwiseOrWith)
                {
                    if (o.flavor != SPIRVAsmOperand::Literal &&
                        o.flavor != SPIRVAsmOperand::NamedValue)
                    {
                        failed = true;
                        getSink()->diagnose(operand.token, Diagnostics::spirvNonConstantBitwiseOr);
                    }
                    go(go, o);
                    operand.knownValue |= o.knownValue;
                }
            };

            check(check, inst.operands[operandIndex]);
        }
    }

    if (failed)
        return CreateErrorExpr(expr);

    // Assign the type of this expression from the type of the last
    // instruction, otherwise void
    if (expr->insts.getCount())
    {
        // TODO: we trust that this is correct, but could should verify
        const auto lastOperands = expr->insts.getLast().operands;
        if (lastOperands.getCount() >= 2 && lastOperands[0].flavor == SPIRVAsmOperand::SlangType &&
            lastOperands[1].flavor == SPIRVAsmOperand::ResultMarker)
        {
            expr->type = lastOperands[0].type.type;
        }
    }
    if (!expr->type)
        expr->type = m_astBuilder->getVoidType();

    return expr;
}
} // namespace Slang
