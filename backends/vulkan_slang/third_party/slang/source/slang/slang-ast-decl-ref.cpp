// slang-ast-decl-ref.cpp

#include "slang-ast-builder.h"
#include "slang-ast-dispatch.h"
#include "slang-ast-forward-declarations.h"
#include "slang-check-impl.h"

namespace Slang
{

DeclRefBase* DirectDeclRef::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    SLANG_UNUSED(astBuilder);
    SLANG_UNUSED(subst);
    SLANG_UNUSED(ioDiff);
    return this;
}

void DirectDeclRef::_toTextOverride(StringBuilder& out)
{
    if (getDecl()->getName() && getDecl()->getName()->text.getLength() != 0)
    {
        out << getDecl()->getName()->text;
    }
}

Val* DirectDeclRef::_resolveImplOverride()
{
    return this;
}

DeclRefBase* DirectDeclRef::_getBaseOverride()
{
    return nullptr;
}

DeclRefBase* _getDeclRefFromVal(Val* val)
{
    if (auto declRefType = as<DeclRefType>(val))
        return declRefType->getDeclRef();
    else if (auto genParamIntVal = as<GenericParamIntVal>(val))
        return genParamIntVal->getDeclRef();
    else if (auto declaredSubtypeWitness = as<DeclaredSubtypeWitness>(val))
        return declaredSubtypeWitness->getDeclRef();
    else if (auto declRef = as<DeclRefBase>(val))
        return declRef;
    return nullptr;
}

DeclRefBase* _resolveAsDeclRef(DeclRefBase* declRefToResolve)
{
    if (auto rs = _getDeclRefFromVal(declRefToResolve->resolve()))
        return rs;
    return declRefToResolve;
}

DeclRefBase* MemberDeclRef::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    int diff = 0;
    auto substParent = getParentOperand()->substituteImpl(astBuilder, subst, &diff);
    if (diff)
    {
        (*ioDiff)++;
        return astBuilder->getMemberDeclRef(substParent, getDecl());
    }
    return this;
}

void MemberDeclRef::_toTextOverride(StringBuilder& out)
{
    getParentOperand()->toText(out);
    if (out.getLength() && !out.endsWith("."))
        out << ".";
    if (getDecl()->getName() && getDecl()->getName()->text.getLength() != 0)
    {
        out << getDecl()->getName()->text;
    }
}

Val* MemberDeclRef::_resolveImplOverride()
{
    auto resolvedParent = _resolveAsDeclRef(getParentOperand());
    if (resolvedParent != getParentOperand())
    {
        return getCurrentASTBuilder()->getMemberDeclRef(resolvedParent, getDecl());
    }
    return this;
}

DeclRefBase* MemberDeclRef::_getBaseOverride()
{
    return getParentOperand();
}

Decl* LookupDeclRef::getSupDecl()
{
    if (auto supType = as<DeclRefType>(getWitness()->getSup()))
    {
        return supType->getDeclRef().getDecl();
    }
    // If we reach here, something is wrong.
    SLANG_UNEXPECTED("Invalid lookup declref");
}

DeclRefBase* LookupDeclRef::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    int diff = 0;

    auto substWitness = as<SubtypeWitness>(getWitness()->substituteImpl(astBuilder, subst, &diff));
    if (diff == 0)
        return this;
    (*ioDiff)++;

    auto substSource = as<Type>(getLookupSource()->substituteImpl(astBuilder, subst, &diff));
    SLANG_ASSERT(substSource);

    if (auto resolved = _getDeclRefFromVal(tryResolve(substWitness, substSource)))
        return resolved;

    return astBuilder->getLookupDeclRef(substSource, substWitness, getDecl());
}

void LookupDeclRef::_toTextOverride(StringBuilder& out)
{
    getLookupSource()->toText(out);
    if (out.getLength() && !out.endsWith("."))
        out << ".";
    if (getDecl()->getName() && getDecl()->getName()->text.getLength() != 0)
    {
        out << getDecl()->getName()->text;
    }
}

Val* LookupDeclRef::_resolveImplOverride()
{
    auto astBuilder = getCurrentASTBuilder();
    Val* resolved = this;

    auto newLookupSource = as<Type>(getLookupSource()->resolve());
    SLANG_ASSERT(newLookupSource);

    auto newWitness = as<SubtypeWitness>(getWitness()->resolve());
    SLANG_ASSERT(newWitness);

    if (auto resolvedVal = tryResolve(newWitness, newLookupSource))
        return resolvedVal;
    if (newLookupSource != getLookupSource() || newWitness != getWitness())
        resolved = astBuilder->getLookupDeclRef(newLookupSource, newWitness, getDecl());
    return resolved;
}

DeclRefBase* LookupDeclRef::_getBaseOverride()
{
    auto supType = getWitness()->getSup();
    if (auto declRefType = as<DeclRefType>(supType))
    {
        return declRefType->getDeclRef();
    }
    return nullptr;
}

Val* LookupDeclRef::tryResolve(SubtypeWitness* newWitness, Type* newLookupSource)
{
    auto astBuilder = getCurrentASTBuilder();
    Decl* requirementKey = getDecl();
    RequirementWitness requirementWitness =
        tryLookUpRequirementWitness(astBuilder, newWitness, requirementKey);
    switch (requirementWitness.getFlavor())
    {
    default:
        // No usable value was found, so there is nothing we can do.
        break;

    case RequirementWitness::Flavor::val:
        {
            auto satisfyingVal = requirementWitness.getVal()->resolve();
            return satisfyingVal;
        }
        break;
    }

    // Hard code implementation of T.Differential.Differential == T.Differential rule.
    auto builtinReq = requirementKey->findModifier<BuiltinRequirementModifier>();
    bool isConstraint = false;
    if (!builtinReq)
    {
        if (auto parentAssocType = as<AssocTypeDecl>(requirementKey->parentDecl))
        {
            builtinReq = parentAssocType->findModifier<BuiltinRequirementModifier>();
            isConstraint = true;
        }
        if (!builtinReq)
            return nullptr;
    }
    if (builtinReq->kind != BuiltinRequirementKind::DifferentialType)
        return nullptr;
    // Is the concrete type a Differential associated type?
    auto innerDeclRefType = as<DeclRefType>(newLookupSource);
    if (!innerDeclRefType)
        return nullptr;
    auto innerBuiltinReq =
        innerDeclRefType->getDeclRef().getDecl()->findModifier<BuiltinRequirementModifier>();
    if (!innerBuiltinReq)
        return nullptr;
    if (innerBuiltinReq->kind != BuiltinRequirementKind::DifferentialType)
        return nullptr;
    if (isConstraint)
        return newWitness;
    if (innerDeclRefType->getDeclRef() != this)
    {
        auto result = innerDeclRefType->getDeclRef().declRefBase->resolve();
        if (result)
            return result;
    }
    return innerDeclRefType;
}

DeclRefBase* GenericAppDeclRef::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    int diff = 0;
    auto substGenericDeclRef = getGenericDeclRef()->substituteImpl(astBuilder, subst, &diff);
    List<Val*> substArgs;
    for (auto arg : getArgs())
    {
        substArgs.add(arg->substituteImpl(astBuilder, subst, &diff));
    }
    if (diff == 0)
        return this;
    (*ioDiff)++;
    return astBuilder->getGenericAppDeclRef(
        substGenericDeclRef,
        substArgs.getArrayView(),
        getDecl());
}

GenericDecl* GenericAppDeclRef::getGenericDecl()
{
    return as<GenericDecl>(getGenericDeclRef()->getDecl());
}


void GenericAppDeclRef::_toTextOverride(StringBuilder& out)
{
    auto genericDecl = as<GenericDecl>(getGenericDeclRef()->getDecl());
    Index paramCount = 0;
    for (auto member : genericDecl->members)
        if (as<GenericTypeParamDeclBase>(member) || as<GenericValueParamDecl>(member))
            paramCount++;
    getGenericDeclRef()->toText(out);
    out << "<";
    auto args = getArgs();
    Index argCount = args.getCount();
    for (Index aa = 0; aa < Math::Min(paramCount, argCount); ++aa)
    {
        if (aa != 0)
            out << ", ";
        args[aa]->toText(out);
    }
    out << ">";
}

Val* GenericAppDeclRef::_resolveImplOverride()
{
    auto astBuilder = getCurrentASTBuilder();
    Val* resolvedVal = this;
    auto resolvedGenericDeclRef = _resolveAsDeclRef(getGenericDeclRef());
    bool diff = false;
    if (resolvedGenericDeclRef != getGenericDeclRef())
        diff = true;
    List<Val*> resolvedArgs;
    for (auto arg : getArgs())
    {
        auto resolvedArg = arg->resolve();
        resolvedArgs.add(resolvedArg);
        if (resolvedArg != arg)
            diff = true;
    }
    if (diff)
        resolvedVal = astBuilder->getGenericAppDeclRef(
            resolvedGenericDeclRef,
            resolvedArgs.getArrayView(),
            getDecl());
    return resolvedVal;
}

DeclRefBase* GenericAppDeclRef::_getBaseOverride()
{
    return getGenericDeclRef();
}

// Convenience accessors for common properties of declarations

DeclRefBase* DeclRefBase::substituteImpl(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff)
{
    SLANG_AST_NODE_VIRTUAL_CALL(DeclRefBase, substituteImpl, (astBuilder, subst, ioDiff));
}

DeclRefBase* DeclRefBase::getBase()
{
    SLANG_AST_NODE_VIRTUAL_CALL(DeclRefBase, getBase, ());
}
void DeclRefBase::toText(StringBuilder& out)
{
    if (auto lookupDeclRef = as<LookupDeclRef>(this))
    {
        lookupDeclRef->_toTextOverride(out);
        return;
    }

    if (as<GenericTypeParamDeclBase>(this->getDecl()))
    {
        SLANG_ASSERT(as<DirectDeclRef>(this));
        out << this->getDecl()->getName()->text;
        return;
    }
    else if (as<GenericValueParamDecl>(this->getDecl()))
    {
        SLANG_ASSERT(as<DirectDeclRef>(this));
        out << this->getDecl()->getName()->text;
        return;
    }

    SubstitutionSet substSet(this);

    // Build a list of parent DeclRefs instead of just Decls
    List<DeclRefBase*> declRefs;

    for (DeclRefBase* dr = this; dr; dr = dr->getParent())
    {
        auto dd = dr->getDecl();

        // If this declaration is an extension, add it and then stop gathering parents
        if (as<ExtensionDecl>(dd))
        {
            declRefs.add(dr);
            break; // Stop gathering parent DeclRefs to exclude namespace
        }

        // Skip the module, file & include decls since their names are
        // considered "transparent"
        if (as<ModuleDecl>(dd) || as<FileDecl>(dd) || as<IncludeDecl>(dd))
            continue;

        // Skip base decls in generic containers. We will handle them when we handle the generic
        // decl.
        if (dd->parentDecl && as<GenericDecl>(dd->parentDecl))
            continue;

        declRefs.add(dr);
    }

    declRefs.reverse();

    bool first = true;
    for (auto declRef : declRefs)
    {
        auto decl = declRef->getDecl();
        if (!first)
            out << ".";
        first = false;

        if (auto name = decl->getName())
        {
            out << name->text;

            // If there are any specializations for this decl, emit them here:
            if (auto genericDecl = as<GenericDecl>(decl))
            {
                if (auto genericAppDeclRef = substSet.findGenericAppDeclRef(genericDecl))
                {
                    Index paramCount = 0;
                    for (auto member : genericDecl->members)
                        if (as<GenericTypeParamDeclBase>(member) ||
                            as<GenericValueParamDecl>(member))
                            paramCount++;
                    out << "<";
                    auto args = genericAppDeclRef->getArgs();
                    Index argCount = args.getCount();
                    for (Index aa = 0; aa < Math::Min(paramCount, argCount); ++aa)
                    {
                        if (aa != 0)
                            out << ", ";
                        args[aa]->toText(out);
                    }
                    out << ">";
                }
            }
        }
        else if (auto extDecl = as<ExtensionDecl>(decl))
        {
            if (extDecl->targetType)
            {
                getTargetType(getCurrentASTBuilder(), getParent())->toText(out);
            }
        }
    }
}

Name* DeclRefBase::getName() const
{
    return getDecl()->nameAndLoc.name;
}

SourceLoc DeclRefBase::getNameLoc() const
{
    return getDecl()->nameAndLoc.loc;
}

SourceLoc DeclRefBase::getLoc() const
{
    return getDecl()->loc;
}

DeclRefBase* DeclRefBase::getParent()
{
    auto astBuilder = getCurrentASTBuilder();
    if (!getDecl()->parentDecl)
        return nullptr;
    auto parentDecl = getDecl()->parentDecl;
    for (auto base = getBase(); base; base = base->getBase())
    {
        if (base->getDecl() == parentDecl)
            return base;
        bool parentIsChildOfBase = false;
        for (auto dd = parentDecl->parentDecl; dd; dd = dd->parentDecl)
        {
            if (dd == base->getDecl())
            {
                parentIsChildOfBase = true;
                break;
            }
        }
        if (parentIsChildOfBase)
            return astBuilder->getMemberDeclRef(base, parentDecl);
    }
    return astBuilder->getDirectDeclRef(parentDecl);
}

SubstitutionSet::operator bool() const
{
    return declRef != nullptr && !as<DirectDeclRef>(declRef);
}

Val::OperandView<Val> tryGetGenericArguments(SubstitutionSet substSet, Decl* genericDecl)
{
    if (!substSet.declRef)
        return Val::OperandView<Val>();

    DeclRefBase* currentDeclRef = substSet.declRef;
    // search for a substitution that might apply to us
    for (auto s = currentDeclRef; s; s = s->getBase())
    {
        auto genericAppDeclRef = as<GenericAppDeclRef>(s);
        if (!genericAppDeclRef)
            continue;

        // the generic decl associated with the substitution list must be
        // the generic decl that declared this parameter
        auto parentGeneric = genericAppDeclRef->getGenericDecl();
        if (parentGeneric != genericDecl)
            continue;

        return genericAppDeclRef->getArgs();
    }
    return Val::OperandView<Val>();
}

Type* SubstitutionSet::applyToType(ASTBuilder* astBuilder, Type* type) const
{
    if (!type)
        return nullptr;
    int diff = 0;
    auto newType = as<Type>(type->substituteImpl(astBuilder, *this, &diff));
    if (diff && newType)
        return newType;
    return type;
}

SubstExpr<Expr> applySubstitutionToExpr(SubstitutionSet substSet, Expr* expr)
{
    return SubstExpr<Expr>(expr, substSet);
}


DeclRefBase* SubstitutionSet::applyToDeclRef(ASTBuilder* astBuilder, DeclRefBase* otherDeclRef)
    const
{
    int diff = 0;
    return otherDeclRef->substituteImpl(astBuilder, *this, &diff);
}

LookupDeclRef* SubstitutionSet::findLookupDeclRef() const
{
    for (auto s = declRef; s; s = s->getBase())
    {
        if (auto lookupDeclRef = as<LookupDeclRef>(s))
            return lookupDeclRef;
    }
    return nullptr;
}

DeclRefBase* SubstitutionSet::getInnerMostNodeWithSubstInfo() const
{
    for (auto s = declRef; s; s = s->getBase())
    {
        if (as<LookupDeclRef>(s) || as<GenericAppDeclRef>(s))
            return s;
    }
    return nullptr;
}


GenericAppDeclRef* SubstitutionSet::findGenericAppDeclRef(GenericDecl* genericDecl) const
{
    for (auto s = declRef; s; s = s->getBase())
    {
        if (auto genApp = as<GenericAppDeclRef>(s))
        {
            if (genApp->getGenericDecl() == genericDecl)
                return genApp;
        }
    }
    return nullptr;
}

GenericAppDeclRef* SubstitutionSet::findGenericAppDeclRef() const
{
    for (auto s = declRef; s; s = s->getBase())
    {
        if (auto genApp = as<GenericAppDeclRef>(s))
        {
            return genApp;
        }
        else if (as<LookupDeclRef>(s))
        {
            return nullptr;
        }
    }
    return nullptr;
}

DeclRef<Decl> createDefaultSubstitutionsIfNeeded(
    ASTBuilder* astBuilder,
    SemanticsVisitor* semantics,
    DeclRef<Decl> declRef)
{
    if (declRef.as<GenericTypeParamDeclBase>())
        return declRef;
    if (declRef.as<GenericValueParamDecl>())
        return declRef;
    if (declRef.as<GenericTypeConstraintDecl>())
        return declRef;
    ShortList<GenericDecl*> genericParentDecls;
    auto lastSubstNode = SubstitutionSet(declRef).getInnerMostNodeWithSubstInfo();
    auto lastGenApp = as<GenericAppDeclRef>(lastSubstNode);
    auto lastLookup = as<LookupDeclRef>(lastSubstNode);
    for (auto dd = declRef.getDecl()->parentDecl; dd; dd = dd->parentDecl)
    {
        if (lastGenApp && dd == lastGenApp->getGenericDecl())
            break;
        if (lastLookup && lastLookup->getDecl()->isChildOf(dd))
            break;
        if (auto gen = as<GenericDecl>(dd))
            genericParentDecls.add(gen);
    }
    DeclRef<Decl> parentDeclRef = lastSubstNode;
    for (auto i = genericParentDecls.getCount() - 1; i >= 0; i--)
    {
        auto current = genericParentDecls[i];
        auto args = getDefaultSubstitutionArgs(astBuilder, semantics, current);
        if (parentDeclRef)
        {
            parentDeclRef = astBuilder->getMemberDeclRef(parentDeclRef, current);
        }
        else
        {
            parentDeclRef = astBuilder->getDirectDeclRef(current);
        }
        parentDeclRef =
            astBuilder->getGenericAppDeclRef(parentDeclRef.as<GenericDecl>(), args.getArrayView());
    }
    if (!parentDeclRef)
        return declRef;
    if (parentDeclRef.getDecl() == declRef.getDecl())
        return parentDeclRef;
    return astBuilder->getMemberDeclRef(parentDeclRef, declRef.getDecl());
}
} // namespace Slang
