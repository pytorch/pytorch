// slang-ast-type.cpp
#include "slang-ast-val.h"

#include "slang-ast-builder.h"
#include "slang-ast-dispatch.h"
#include "slang-check-impl.h"
#include "slang-diagnostics.h"
#include "slang-mangle.h"
#include "slang-syntax.h"

#include <assert.h>
#include <typeinfo>

namespace Slang
{

void ValNodeDesc::init()
{
    Hasher hasher;
    hasher.hashValue(type.getTag());
    for (Index i = 0; i < operands.getCount(); ++i)
    {
        // Note: we are hashing the raw pointer value rather
        // than the content of the value node. This is done
        // to match the semantics implemented for `==` on
        // `NodeDesc`.
        //
        hasher.hashValue(operands[i].values.intOperand);
    }
    hashCode = hasher.getResult();
}

Val* Val::substitute(ASTBuilder* astBuilder, SubstitutionSet subst)
{
    if (!subst)
        return this;
    int diff = 0;
    return substituteImpl(astBuilder, subst, &diff);
}

Val* Val::substituteImpl(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff)
{
    SLANG_AST_NODE_VIRTUAL_CALL(Val, substituteImpl, (astBuilder, subst, ioDiff))
}

void Val::toText(StringBuilder& out){SLANG_AST_NODE_VIRTUAL_CALL(Val, toText, (out))}

Val* Val::_resolveImplOverride()
{
    SLANG_UNEXPECTED("Val::_resolveImplOverride not overridden");
}

Val* Val::resolveImpl()
{
    SLANG_AST_NODE_VIRTUAL_CALL(Val, resolveImpl, ());
}

Val* Val::resolve()
{
    auto astBuilder = getCurrentASTBuilder();
    // If we are not in a proper checking context, just return the previously resolved val.
    if (!astBuilder)
        return m_resolvedVal ? m_resolvedVal : this;
    if (m_resolvedVal && m_resolvedValEpoch == astBuilder->getEpoch())
    {
        SLANG_ASSERT(as<Val>(m_resolvedVal));
        return m_resolvedVal;
    }
    // Update epoch now to avoid infinite recursion.
    m_resolvedValEpoch = astBuilder->getEpoch();
    m_resolvedVal = resolveImpl();
#ifdef _DEBUG
    if (m_resolvedVal->_debugUID > 0 && this->_debugUID < 0)
    {
        SLANG_ASSERT_FAILURE(
            "should not be modifying the core module vals outside of the core module checking.");
    }
#endif
    return m_resolvedVal;
}

void Val::_setUnique()
{
    m_resolvedVal = this;
    m_resolvedValEpoch = getCurrentASTBuilder()->getEpoch();
}

Val* Val::defaultResolveImpl()
{
    // Default resolve implementation is to recursively resolve all operands, and lookup in
    // deduplication cache.
    ValNodeDesc newDesc;
    newDesc.type = SyntaxClass<NodeBase>(astNodeType);
    bool diff = false;
    for (auto operand : m_operands)
    {
        if (operand.kind == ValNodeOperandKind::ValNode)
        {
            auto valOperand = as<Val>(operand.values.nodeOperand);
            if (valOperand)
            {
                auto newOperand = valOperand->resolve();
                if (newOperand != valOperand)
                {
                    diff = true;
                    operand.values.nodeOperand = newOperand;
                }
            }
        }
        newDesc.operands.add(operand);
    }

    if (!diff)
        return this;

    newDesc.init();
    auto astBuilder = getCurrentASTBuilder();
    return astBuilder->_getOrCreateImpl(_Move(newDesc));
}

String Val::toString()
{
    StringBuilder builder;
    toText(builder);
    return builder;
}

HashCode Val::getHashCode()
{
    return Slang::getHashCode(resolve());
}

Val* Val::_substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff)
{
    SLANG_UNUSED(astBuilder);
    SLANG_UNUSED(subst);
    SLANG_UNUSED(ioDiff);
    // Default behavior is to not substitute at all
    return this;
}

void Val::_toTextOverride(StringBuilder& out)
{
    SLANG_UNUSED(out);
    SLANG_UNEXPECTED("Val::_toStringOverride not overridden");
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ConstantIntVal !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

void ConstantIntVal::_toTextOverride(StringBuilder& out)
{
    if (auto enumTypeDecl = isDeclRefTypeOf<EnumDecl>(getType()))
    {
        // If this is an enum type, then we want to print the name of the
        // corresponding enum case, instead of the raw integer value, if possible.
        //
        // We will look up the enum case that corresponds to the value, and
        // print its name if we can find one.
        //
        for (auto enumCase : enumTypeDecl.getDecl()->getMembersOfType<EnumCaseDecl>())
        {
            if (auto constVal = as<ConstantIntVal>(enumCase->tagVal))
            {
                if (constVal->getValue() == getValue())
                {
                    out << DeclRef(enumCase);
                    return;
                }
            }
        }

        // Fallback to explicit cast to the enum type.
        out << getType() << "(" << getValue() << ")";
        return;
    }
    out << getValue();
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! GenericParamIntVal !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

void GenericParamIntVal::_toTextOverride(StringBuilder& out)
{
    Name* name = getDeclRef().getName();
    if (name)
    {
        out << name->text;
    }
}

Val* maybeSubstituteGenericParam(Val* paramVal, Decl* paramDecl, SubstitutionSet subst, int* ioDiff)
{
    // search for a substitution that might apply to us
    auto outerGeneric = as<GenericDecl>(paramDecl->parentDecl);
    if (!outerGeneric)
        return paramVal;

    GenericAppDeclRef* genAppArgs = subst.findGenericAppDeclRef(outerGeneric);
    if (!genAppArgs)
    {
        return paramVal;
    }

    auto args = genAppArgs->getArgs();

    // In some cases, we construct a `DeclRef` to a `GenericDecl`
    // (or a declaration under one) that only includes argument
    // values for a prefix of the parameters of the generic.
    //
    // If we aren't careful, we could end up indexing into the
    // argument list past the available range.
    //
    Count argCount = args.getCount();

    Count argIndex = 0;
    for (auto m : outerGeneric->members)
    {
        // If we have run out of arguments, then we can stop
        // iterating over the parameters, because `this`
        // parameter will not be replaced with anything by
        // the substituion.
        //
        if (argIndex >= argCount)
        {
            return paramVal;
        }


        if (m == paramDecl)
        {
            // We've found it, so return the corresponding specialization argument
            (*ioDiff)++;
            return args[argIndex];
        }
        else if (const auto typeParam = as<GenericTypeParamDeclBase>(m))
        {
            argIndex++;
        }
        else if (const auto valParam = as<GenericValueParamDecl>(m))
        {
            argIndex++;
        }
        else
        {
        }
    }

    // Nothing found: don't substitute.
    return paramVal;
}

Val* GenericParamIntVal::_substituteImplOverride(
    ASTBuilder* /* astBuilder */,
    SubstitutionSet subst,
    int* ioDiff)
{
    if (auto result = maybeSubstituteGenericParam(this, getDeclRef().getDecl(), subst, ioDiff))
        return result;

    return this;
}

bool GenericParamIntVal::_isLinkTimeValOverride()
{
    return getDeclRef().getDecl()->hasModifier<ExternModifier>();
}

Val* GenericParamIntVal::_linkTimeResolveOverride(Dictionary<String, IntVal*>& map)
{
    auto name = getMangledName(getCurrentASTBuilder(), getDeclRef().declRefBase);
    IntVal* v;
    if (map.tryGetValue(name, v))
        return v;
    return this;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ErrorIntVal !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

void ErrorIntVal::_toTextOverride(StringBuilder& out)
{
    out << toSlice("<error>");
}

Val* ErrorIntVal::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    SLANG_UNUSED(astBuilder);
    SLANG_UNUSED(subst);
    SLANG_UNUSED(ioDiff);
    return this;
}

Val* TypeEqualityWitness::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    auto type = as<Type>(getSub()->substituteImpl(astBuilder, subst, ioDiff));
    TypeEqualityWitness* rs = astBuilder->getOrCreate<TypeEqualityWitness>(type, type);
    return rs;
}

void TypeEqualityWitness::_toTextOverride(StringBuilder& out)
{
    out << toSlice("TypeEqualityWitness(") << getSub() << toSlice(")");
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TypePackSubtypeWitness !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Val* TypePackSubtypeWitness::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    int diff = 0;
    ShortList<SubtypeWitness*> newWitnesses;
    for (Index i = 0; i < getCount(); i++)
    {
        auto witness = getWitness(i);
        auto newWitness = as<SubtypeWitness>(witness->substituteImpl(astBuilder, subst, &diff));
        newWitnesses.add(newWitness);
    }
    auto newSub = as<Type>(getSub()->substituteImpl(astBuilder, subst, &diff));
    auto newSup = as<Type>(getSup()->substituteImpl(astBuilder, subst, &diff));
    if (!diff)
        return this;
    (*ioDiff)++;
    return getCurrentASTBuilder()->getSubtypeWitnessPack(
        newSub,
        newSup,
        newWitnesses.getArrayView().arrayView);
}

Val* TypePackSubtypeWitness::_resolveImplOverride()
{
    int diff = 0;
    ShortList<SubtypeWitness*> newWitnesses;
    for (Index i = 0; i < getCount(); i++)
    {
        auto witness = getWitness(i);
        auto newWitness = as<SubtypeWitness>(witness->resolve());
        if (witness != newWitness)
            diff++;
        newWitnesses.add(newWitness);
    }
    auto newSub = as<Type>(getSub()->resolve());
    if (newSub != getSub())
        diff++;
    auto newSup = as<Type>(getSup()->resolve());
    if (newSup != getSup())
        diff++;

    if (!diff)
        return this;
    return getCurrentASTBuilder()->getSubtypeWitnessPack(
        newSub,
        newSup,
        newWitnesses.getArrayView().arrayView);
}

void TypePackSubtypeWitness::_toTextOverride(StringBuilder& out)
{
    out << toSlice("Pack(");
    for (Index i = 0; i < getCount(); i++)
    {
        if (i != 0)
            out << toSlice(", ");
        getWitness(i)->toText(out);
    }
    out << toSlice(")");
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ExpandSubtypeWitness !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Val* ExpandSubtypeWitness::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    int diff = 0;
    auto newSub = as<Type>(getSub()->substituteImpl(astBuilder, subst, &diff));
    auto newSup = as<Type>(getSup()->substituteImpl(astBuilder, subst, &diff));
    if (!diff)
        return this;
    if (auto subTypePack = as<ConcreteTypePack>(newSub))
    {
        // If sub is substituted into a concrete type pack, we should return a
        // TypePackSubtypeWitness.
        ShortList<SubtypeWitness*> newWitnesses;
        for (int i = 0; i < (int)subTypePack->getTypeCount(); i++)
        {
            auto elementType = subTypePack->getElementType(i);
            subst.packExpansionIndex = i;
            auto elementWitness = as<SubtypeWitness>(
                getPatternTypeWitness()->substituteImpl(astBuilder, subst, &diff));
            auto newWitness = getCurrentASTBuilder()->getExpandSubtypeWitness(
                elementType,
                newSup,
                elementWitness);
            newWitnesses.add(as<SubtypeWitness>(newWitness));
        }
        (*ioDiff)++;
        return getCurrentASTBuilder()->getSubtypeWitnessPack(
            newSub,
            newSup,
            newWitnesses.getArrayView().arrayView);
    }

    (*ioDiff)++;
    auto newPatternWitness =
        as<SubtypeWitness>(getPatternTypeWitness()->substituteImpl(astBuilder, subst, &diff));
    return getCurrentASTBuilder()->getExpandSubtypeWitness(newSub, newSup, newPatternWitness);
}

Val* ExpandSubtypeWitness::_resolveImplOverride()
{
    int diff = 0;
    auto newPatternWitness = as<SubtypeWitness>(getPatternTypeWitness()->resolve());
    if (newPatternWitness != getPatternTypeWitness())
        diff++;
    auto newSub = as<Type>(getSub()->resolve());
    if (newSub != getSub())
        diff++;
    auto newSup = as<Type>(getSup()->resolve());
    if (newSup != getSup())
        diff++;
    if (!diff)
        return this;
    return getCurrentASTBuilder()->getExpandSubtypeWitness(newSub, newSup, newPatternWitness);
}

void ExpandSubtypeWitness::_toTextOverride(StringBuilder& out)
{
    out << toSlice("ExpandWitness(");
    getPatternTypeWitness()->toText(out);
    out << toSlice(")");
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! EachSubtypeWitness !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Val* EachSubtypeWitness::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    int diff = 0;
    auto newPatternWitness =
        as<SubtypeWitness>(getPatternTypeWitness()->substituteImpl(astBuilder, subst, &diff));
    if (auto witnessPack = as<TypePackSubtypeWitness>(newPatternWitness))
    {
        if (subst.packExpansionIndex >= 0 && subst.packExpansionIndex < witnessPack->getCount())
        {
            auto newWitness = witnessPack->getWitness(subst.packExpansionIndex);
            (*ioDiff)++;
            return newWitness;
        }
    }
    auto newSub = as<Type>(getSub()->substituteImpl(astBuilder, subst, &diff));
    auto newSup = as<Type>(getSup()->substituteImpl(astBuilder, subst, &diff));
    if (!diff)
        return this;
    return getCurrentASTBuilder()->getEachSubtypeWitness(newSub, newSup, newPatternWitness);
}

Val* EachSubtypeWitness::_resolveImplOverride()
{
    int diff = 0;
    auto newPatternWitness = as<SubtypeWitness>(getPatternTypeWitness()->resolve());
    if (newPatternWitness != getPatternTypeWitness())
        diff++;
    auto newSub = as<Type>(getSub()->resolve());
    if (newSub != getSub())
        diff++;
    auto newSup = as<Type>(getSup()->resolve());
    if (newSup != getSup())
        diff++;
    if (!diff)
        return this;
    return getCurrentASTBuilder()->getEachSubtypeWitness(newSub, newSup, newPatternWitness);
}

void EachSubtypeWitness::_toTextOverride(StringBuilder& out)
{
    out << toSlice("EachWitness(");
    getPatternTypeWitness()->toText(out);
    out << toSlice(")");
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DeclaredSubtypeWitness !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Val* DeclaredSubtypeWitness::_resolveImplOverride()
{
    auto resolvedDeclRef = getDeclRef().declRefBase->resolve();
    if (auto resolvedVal = as<SubtypeWitness>(resolvedDeclRef))
        return resolvedVal;

    auto newSub = as<Type>(getSub()->resolve());
    auto newSup = as<Type>(getSup()->resolve());

    // If we are trying to lookup for a witness that A<:B from a witness(A<:B), we
    // can just return the witness itself.
    if (auto lookupDeclRef = as<LookupDeclRef>(resolvedDeclRef))
    {
        auto witnessToLookupFrom = lookupDeclRef->getWitness();
        if (witnessToLookupFrom->getSub()->equals(newSub) &&
            witnessToLookupFrom->getSup()->equals(newSup))
            return witnessToLookupFrom;
    }
    auto newDeclRef = as<DeclRefBase>(resolvedDeclRef);
    if (!newDeclRef)
        newDeclRef = getDeclRef().declRefBase;
    if (newSub != getSub() || newSup != getSup() || newDeclRef != getDeclRef())
    {
        return getCurrentASTBuilder()->getDeclaredSubtypeWitness(newSub, newSup, newDeclRef);
    }
    return this;
}

ConversionCost DeclaredSubtypeWitness::_getOverloadResolutionCostOverride()
{
    return kConversionCost_None;
}

Val* DeclaredSubtypeWitness::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    if (auto genConstraintDeclRef = getDeclRef().as<GenericTypeConstraintDecl>())
    {
        auto genericDecl = as<GenericDecl>(getDeclRef().getDecl()->parentDecl);
        if (!genericDecl)
            goto breakLabel;

        // search for a substitution that might apply to us
        auto args = tryGetGenericArguments(subst, genericDecl);
        if (args.getCount() == 0)
            goto breakLabel;

        bool found = false;
        Index index = 0;
        for (auto m : genericDecl->members)
        {
            if (auto constraintParam = as<GenericTypeConstraintDecl>(m))
            {
                if (constraintParam == getDeclRef().getDecl())
                {
                    found = true;
                    break;
                }
                index++;
            }
        }
        if (found)
        {
            auto ordinaryParamCount =
                genericDecl->getMembersOfType<GenericTypeParamDeclBase>().getCount() +
                genericDecl->getMembersOfType<GenericValueParamDecl>().getCount();
            if (index + ordinaryParamCount < args.getCount())
            {
                (*ioDiff)++;
                return args[index + ordinaryParamCount];
            }
            else
            {
                // When the `subst` represents a partial substitution, we may not have a
                // corresponding argument. In this case we just return the original witness.
                //
                goto breakLabel;
            }
        }
    }
    else if (auto thisTypeConstraintDeclRef = getDeclRef().as<ThisTypeConstraintDecl>())
    {
        auto lookupSubst = subst.findLookupDeclRef();
        if (lookupSubst &&
            lookupSubst->getSupDecl() == thisTypeConstraintDeclRef.getDecl()->getInterfaceDecl())
        {
            (*ioDiff)++;
            return lookupSubst->getWitness();
        }
    }

breakLabel:;

    // Perform substitution on the constituent elements.
    int diff = 0;
    auto substSub = as<Type>(getSub()->substituteImpl(astBuilder, subst, &diff));
    auto substSup = as<Type>(getSup()->substituteImpl(astBuilder, subst, &diff));

    if (!diff)
        return this;

    (*ioDiff)++;

    // If we have a reference to a type constraint for an
    // associated type declaration, then we can replace it
    // with the concrete conformance witness for a concrete
    // type implementing the outer interface.
    //
    // TODO: It is a bit gross that we use `GenericTypeConstraintDecl` for
    // associated types, when they aren't really generic type *parameters*,
    // so we'll need to change this location in the code if we ever clean
    // up the hierarchy.
    //
    if (auto substTypeConstraintDecl = as<GenericTypeConstraintDecl>(getDeclRef().getDecl()))
    {
        if (auto substAssocTypeDecl = as<AssocTypeDecl>(substTypeConstraintDecl->parentDecl))
        {
            if (auto interfaceDecl = as<InterfaceDecl>(substAssocTypeDecl->parentDecl))
            {
                // At this point we have a constraint decl for an associated type,
                // and we nee to see if we are dealing with a concrete substitution
                // for the interface around that associated type.
                if (auto thisTypeWitness = findThisTypeWitness(subst, interfaceDecl))
                {
                    // We need to look up the declaration that satisfies
                    // the requirement named by the associated type.
                    Decl* requirementKey = substTypeConstraintDecl;
                    RequirementWitness requirementWitness =
                        tryLookUpRequirementWitness(astBuilder, thisTypeWitness, requirementKey);
                    switch (requirementWitness.getFlavor())
                    {
                    default:
                        break;

                    case RequirementWitness::Flavor::val:
                        {
                            auto satisfyingVal = requirementWitness.getVal();
                            return satisfyingVal;
                        }
                    }
                }
            }
        }
    }

    auto substDeclRef = getDeclRef().substituteImpl(astBuilder, subst, &diff);
    auto rs = astBuilder->getDeclaredSubtypeWitness(substSub, substSup, substDeclRef);
    return rs;
}

void DeclaredSubtypeWitness::_toTextOverride(StringBuilder& out)
{
    out << toSlice("DeclaredSubtypeWitness(") << getSub() << toSlice(", ") << getSup()
        << toSlice(", ") << getDeclRef() << toSlice(")");
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TransitiveSubtypeWitness !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Val* TransitiveSubtypeWitness::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    int diff = 0;

    SubtypeWitness* substSubToMid =
        as<SubtypeWitness>(getSubToMid()->substituteImpl(astBuilder, subst, &diff));
    SubtypeWitness* substMidToSup =
        as<SubtypeWitness>(getMidToSup()->substituteImpl(astBuilder, subst, &diff));

    // If nothing changed, then we can bail out early.
    if (!diff)
        return this;

    // Something changes, so let the caller know.
    (*ioDiff)++;

    // If it possible that substitution could have led to either of the
    // constituent witnesses being simplified, and such simplification could
    // (in principle) lead to opportunities to simplify this transitive witness.
    // As such, we do not simply create a fresh `TransitiveSubtypeWitness` here,
    // and instead go through a bottleneck routine in the `ASTBuilder` that will
    // detect and handle any possible simplifications.
    //
    return astBuilder->getTransitiveSubtypeWitness(substSubToMid, substMidToSup);
}

ConversionCost TransitiveSubtypeWitness::_getOverloadResolutionCostOverride()
{
    return getSubToMid()->getOverloadResolutionCost() + getMidToSup()->getOverloadResolutionCost() +
           kConversionCost_GenericParamUpcast;
}

void TransitiveSubtypeWitness::_toTextOverride(StringBuilder& out)
{
    // Note: we only print the constituent
    // witnesses, and rely on them to print
    // the starting and ending types.

    out << toSlice("TransitiveSubtypeWitness(") << getSubToMid() << toSlice(", ") << getMidToSup()
        << toSlice(")");
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ExtractFromConjunctionSubtypeWitness
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Val* ExtractFromConjunctionSubtypeWitness::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    int diff = 0;

    auto substSub = as<Type>(getSub()->substituteImpl(astBuilder, subst, &diff));
    auto substSup = as<Type>(getSup()->substituteImpl(astBuilder, subst, &diff));
    auto substWitness =
        as<SubtypeWitness>(getConjunctionWitness()->substituteImpl(astBuilder, subst, &diff));

    // If nothing changed, then we can bail out early.
    if (!diff)
        return this;

    // Something changes, so let the caller know.
    (*ioDiff)++;

    // Substitution into the constituent pieces of this witness could
    // have created opportunities for simplification. For example,
    // the `substWitness` might be a `ConjunctionSubtypeWitness`,
    // such that we could directly use one of its components in
    // place of the extraction.
    //
    // We use the factory function on the AST builder to create
    // the result witness, so that it can perform all of the
    // simplification logic as needed.
    //
    return astBuilder->getExtractFromConjunctionSubtypeWitness(
        substSub,
        substSup,
        substWitness,
        getIndexInConjunction());
}

ConversionCost ExtractFromConjunctionSubtypeWitness::_getOverloadResolutionCostOverride()
{
    auto witness = as<ConjunctionSubtypeWitness>(getConjunctionWitness());
    if (!witness)
        return kConversionCost_None;
    auto index = getIndexInConjunction();
    if (index < witness->getComponentCount())
        return witness->getComponentWitness(index)->getOverloadResolutionCost();
    return kConversionCost_None;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ExtractExistentialSubtypeWitness
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

void ExtractExistentialSubtypeWitness::_toTextOverride(StringBuilder& out)
{
    out << toSlice("extractExistentialValue(") << getDeclRef() << toSlice(")");
}

Val* ExtractExistentialSubtypeWitness::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    int diff = 0;

    auto substDeclRef = getDeclRef().substituteImpl(astBuilder, subst, &diff);
    auto substSub = as<Type>(getSub()->substituteImpl(astBuilder, subst, &diff));
    auto substSup = as<Type>(getSup()->substituteImpl(astBuilder, subst, &diff));

    if (!diff)
        return this;

    (*ioDiff)++;

    ExtractExistentialSubtypeWitness* substValue =
        astBuilder->getOrCreate<ExtractExistentialSubtypeWitness>(substSub, substSup, substDeclRef);
    return substValue;
}

void ConjunctionSubtypeWitness::_toTextOverride(StringBuilder& out)
{
    out << "ConjunctionSubtypeWitness(";
    for (Index i = 0; i < kComponentCount; ++i)
    {
        if (i != 0)
            out << ",";

        auto w = getComponentWitness(i);
        if (w)
            out << w;
    }
    out << ")";
}

Val* ConjunctionSubtypeWitness::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    int diff = 0;
    Val* substComponentWitnesses[kComponentCount];

    auto substSub = as<Type>(getSub()->substituteImpl(astBuilder, subst, &diff));
    auto substSup = as<Type>(getSup()->substituteImpl(astBuilder, subst, &diff));

    for (Index i = 0; i < kComponentCount; ++i)
    {
        auto w = getComponentWitness(i);
        substComponentWitnesses[i] = w ? w->substituteImpl(astBuilder, subst, &diff) : nullptr;
    }

    if (!diff)
        return this;

    *ioDiff += diff;

    // We use the factory function on the AST builder rather than
    // directly construct a new `ConjunctionSubtypeWitness`, because
    // the substitution process might have created further opportunities
    // for simplification.
    //
    auto result = astBuilder->getConjunctionSubtypeWitness(
        substSub,
        substSup,
        as<SubtypeWitness>(substComponentWitnesses[0]),
        as<SubtypeWitness>(substComponentWitnesses[1]));
    return result;
}

ConversionCost ConjunctionSubtypeWitness::_getOverloadResolutionCostOverride()
{
    ConversionCost result = kConversionCost_None;
    for (Index i = 0; i < getComponentCount(); i++)
        result += getComponentWitness(i)->getOverloadResolutionCost();
    return result;
}

void ExtractFromConjunctionSubtypeWitness::_toTextOverride(StringBuilder& out)
{
    out << "ExtractFromConjunctionSubtypeWitness(";
    if (getConjunctionWitness())
        out << getConjunctionWitness();
    if (getSub())
        out << getSub();
    out << ",";
    if (getSup())
        out << getSup();
    out << "," << getIndexInConjunction();
    out << ")";
}

void TypeCoercionWitness::_toTextOverride(StringBuilder& out)
{
    out << "TypeCoercionWitness(";
    if (getFromType())
        out << getFromType();
    if (getToType())
        out << getToType();
    out << ")";
}

Val* TypeCoercionWitness::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    int diff = 0;

    auto substDeclRef = getDeclRef().substituteImpl(astBuilder, subst, &diff);
    auto substFrom = as<Type>(getFromType()->substituteImpl(astBuilder, subst, &diff));
    auto substTo = as<Type>(getToType()->substituteImpl(astBuilder, subst, &diff));

    if (!diff)
        return this;

    (*ioDiff)++;

    TypeCoercionWitness* substValue =
        astBuilder->getTypeCoercionWitness(substFrom, substTo, substDeclRef);
    return substValue;
}

Val* TypeCoercionWitness::_resolveImplOverride()
{
    Val* resolvedDeclRef = nullptr;
    if (getDeclRef())
        resolvedDeclRef = getDeclRef().declRefBase->resolve();
    if (auto resolvedVal = as<Witness>(resolvedDeclRef))
        return resolvedVal;

    auto newFrom = as<Type>(getFromType()->resolve());
    auto newTo = as<Type>(getToType()->resolve());

    auto newDeclRef = as<DeclRefBase>(resolvedDeclRef);
    if (!newDeclRef)
        newDeclRef = getDeclRef().declRefBase;
    if (newFrom != getFromType() || newTo != getToType() || newDeclRef != getDeclRef())
    {
        return getCurrentASTBuilder()->getTypeCoercionWitness(newFrom, newTo, newDeclRef);
    }
    return this;
}

// UNormModifierVal

void UNormModifierVal::_toTextOverride(StringBuilder& out)
{
    out.append("unorm");
}

Val* UNormModifierVal::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    SLANG_UNUSED(astBuilder);
    SLANG_UNUSED(subst);
    SLANG_UNUSED(ioDiff);
    return this;
}

// SNormModifierVal

void SNormModifierVal::_toTextOverride(StringBuilder& out)
{
    out.append("snorm");
}

Val* SNormModifierVal::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    SLANG_UNUSED(astBuilder);
    SLANG_UNUSED(subst);
    SLANG_UNUSED(ioDiff);
    return this;
}

// NoDiffModifierVal
void NoDiffModifierVal::_toTextOverride(StringBuilder& out)
{
    out.append("no_diff");
}

Val* NoDiffModifierVal::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    SLANG_UNUSED(astBuilder);
    SLANG_UNUSED(subst);
    SLANG_UNUSED(ioDiff);
    return this;
}

// PolynomialIntVal

void PolynomialIntVal::_toTextOverride(StringBuilder& out)
{
    auto constantTerm = getConstantTerm();
    auto terms = getTerms();
    for (Index i = 0; i < terms.getCount(); i++)
    {
        auto& term = *(terms[i]);
        if (term.getConstFactor() > 0)
        {
            if (i > 0)
                out << "+";
        }
        else
            out << "-";
        bool isFirstFactor = true;
        if (abs(term.getConstFactor()) != 1 || term.getParamFactors().getCount() == 0)
        {
            out << abs(term.getConstFactor());
            isFirstFactor = false;
        }
        for (Index j = 0; j < term.getParamFactors().getCount(); j++)
        {
            auto factor = term.getParamFactors()[j];
            if (isFirstFactor)
            {
                isFirstFactor = false;
            }
            else
            {
                out << "*";
            }
            factor->getParam()->toText(out);
            if (factor->getPower() != 1)
            {
                out << "^^" << factor->getPower();
            }
        }
    }
    if (constantTerm > 0)
    {
        if (terms.getCount() > 0)
        {
            out << "+";
        }
        out << constantTerm;
    }
    else if (constantTerm < 0)
    {
        out << constantTerm;
    }
}

struct PolynomialIntValBuilder
{
    ASTBuilder* astBuilder;

    IntegerLiteralValue constantTerm = 0;
    List<PolynomialIntValTerm*> terms;

    PolynomialIntValBuilder(ASTBuilder* inAstBuilder)
        : astBuilder(inAstBuilder)
    {
    }

    // compute val += opreand*multiplier;
    bool addToPolynomialTerm(IntVal* operand, IntegerLiteralValue multiplier)
    {
        if (auto c = as<ConstantIntVal>(operand))
        {
            constantTerm += c->getValue() * multiplier;
            return true;
        }
        else if (auto poly = as<PolynomialIntVal>(operand))
        {
            constantTerm += poly->getConstantTerm() * multiplier;
            for (auto term : poly->getTerms())
            {
                auto newTerm = astBuilder->getOrCreate<PolynomialIntValTerm>(
                    multiplier * term->getConstFactor(),
                    term->getParamFactors());
                terms.add(newTerm);
            }
            return true;
        }
        else if (auto genVal = as<IntVal>(operand))
        {
            auto factor = astBuilder->getOrCreate<PolynomialIntValFactor>(genVal, 1);
            auto term = astBuilder->getOrCreate<PolynomialIntValTerm>(
                multiplier,
                makeArrayViewSingle(factor));
            terms.add(term);
            return true;
        }
        return false;
    }

    IntVal* canonicalize(Type* type)
    {
        List<PolynomialIntValTerm*> newTerms;
        IntegerLiteralValue newConstantTerm = constantTerm;
        auto addTerm = [&](PolynomialIntValTerm* newTerm)
        {
            for (auto& term : newTerms)
            {
                if (term->canCombineWith(*newTerm))
                {
                    term = astBuilder->getOrCreate<PolynomialIntValTerm>(
                        term->getConstFactor() + newTerm->getConstFactor(),
                        term->getParamFactors());
                    return;
                }
            }
            newTerms.add(newTerm);
        };
        for (auto term : terms)
        {
            if (term->getConstFactor() == 0)
                continue;
            List<PolynomialIntValFactor*> newFactors;
            List<bool> factorIsDifferent;
            for (Index i = 0; i < term->getParamFactors().getCount(); i++)
            {
                auto factor = term->getParamFactors()[i];
                bool factorFound = false;
                for (Index j = 0; j < newFactors.getCount(); j++)
                {
                    auto& newFactor = newFactors[j];
                    if (factor->getParam()->equals(newFactor->getParam()))
                    {
                        if (!factorIsDifferent[j])
                        {
                            factorIsDifferent[j] = true;
                            auto clonedFactor = astBuilder->getOrCreate<PolynomialIntValFactor>(
                                newFactor->getParam(),
                                newFactor->getPower());
                            newFactor = clonedFactor;
                        }
                        newFactor = astBuilder->getOrCreate<PolynomialIntValFactor>(
                            newFactor->getParam(),
                            newFactor->getPower() + factor->getPower());
                        factorFound = true;
                        break;
                    }
                }
                if (!factorFound)
                {
                    newFactors.add(factor);
                    factorIsDifferent.add(false);
                }
            }
            List<PolynomialIntValFactor*> newFactors2;
            // Remove zero-powered factors.
            for (auto factor : newFactors)
            {
                if (factor->getPower() != 0)
                    newFactors2.add(factor);
            }
            if (newFactors2.getCount() == 0)
            {
                newConstantTerm += term->getConstFactor();
                continue;
            }
            newFactors2.sort([](PolynomialIntValFactor* t1, PolynomialIntValFactor* t2)
                             { return *t1 < *t2; });
            bool isDifferent = false;
            if (newFactors2.getCount() != term->getParamFactors().getCount())
                isDifferent = true;
            if (!isDifferent)
            {
                for (Index i = 0; i < term->getParamFactors().getCount(); i++)
                    if (term->getParamFactors()[i] != newFactors2[i])
                    {
                        isDifferent = true;
                        break;
                    }
            }
            if (!isDifferent)
            {
                addTerm(term);
            }
            else
            {
                auto newTerm = astBuilder->getOrCreate<PolynomialIntValTerm>(
                    term->getConstFactor(),
                    newFactors2.getArrayView());
                addTerm(newTerm);
            }
        }
        List<PolynomialIntValTerm*> newTerms2;
        for (auto term : newTerms)
        {
            if (term->getConstFactor() == 0)
                continue;
            newTerms2.add(term);
        }
        newTerms2.sort([](PolynomialIntValTerm* t1, PolynomialIntValTerm* t2)
                       { return *t1 < *t2; });
        terms = _Move(newTerms2);
        constantTerm = newConstantTerm;
        if (terms.getCount() == 1 && constantTerm == 0 && terms[0]->getConstFactor() == 1 &&
            terms[0]->getParamFactors().getCount() == 1 &&
            terms[0]->getParamFactors()[0]->getPower() == 1)
        {
            return terms[0]->getParamFactors()[0]->getParam();
        }
        if (terms.getCount() == 0)
            return astBuilder->getIntVal(type, constantTerm);
        return nullptr;
    }

    IntVal* getIntVal(Type* type)
    {
        if (auto canVal = canonicalize(type))
            return canVal;
        return astBuilder->getOrCreate<PolynomialIntVal>(type, constantTerm, terms.getArrayView());
    }
};

Val* PolynomialIntVal::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    int diff = 0;
    PolynomialIntValBuilder builder(astBuilder);
    builder.constantTerm = getConstantTerm();
    for (auto& term : getTerms())
    {
        IntegerLiteralValue evaluatedTermConstFactor;
        List<PolynomialIntValFactor*> evaluatedTermParamFactors;
        evaluatedTermConstFactor = term->getConstFactor();
        for (auto& factor : term->getParamFactors())
        {
            auto substResult = factor->getParam()->substituteImpl(astBuilder, subst, &diff);

            if (auto constantVal = as<ConstantIntVal>(substResult))
            {
                evaluatedTermConstFactor *= constantVal->getValue();
            }
            else if (auto intResult = as<IntVal>(substResult))
            {
                auto newFactor =
                    astBuilder->getOrCreate<PolynomialIntValFactor>(intResult, factor->getPower());
                evaluatedTermParamFactors.add(newFactor);
            }
        }
        if (evaluatedTermParamFactors.getCount() == 0)
        {
            builder.constantTerm += evaluatedTermConstFactor;
        }
        else
        {
            if (evaluatedTermParamFactors.getCount() == 1 &&
                evaluatedTermParamFactors[0]->getPower() == 1)
            {
                if (auto polyTerm = as<PolynomialIntVal>(evaluatedTermParamFactors[0]->getParam()))
                {
                    builder.addToPolynomialTerm(polyTerm, evaluatedTermConstFactor);
                    continue;
                }
            }
            auto newTerm = astBuilder->getOrCreate<PolynomialIntValTerm>(
                evaluatedTermConstFactor,
                evaluatedTermParamFactors.getArrayView());
            builder.terms.add(newTerm);
        }
    }

    *ioDiff += diff;

    if (builder.terms.getCount() == 0)
        return astBuilder->getIntVal(getType(), builder.constantTerm);
    if (diff != 0)
    {
        return builder.getIntVal(getType());
    }
    return this;
}

IntVal* PolynomialIntVal::neg(ASTBuilder* astBuilder, IntVal* base)
{
    PolynomialIntValBuilder builder(astBuilder);
    builder.addToPolynomialTerm(base, -1);
    return builder.getIntVal(base->getType());
}

IntVal* PolynomialIntVal::sub(ASTBuilder* astBuilder, IntVal* op0, IntVal* op1)
{
    PolynomialIntValBuilder builder(astBuilder);
    builder.addToPolynomialTerm(op0, 1);
    builder.addToPolynomialTerm(op1, -1);
    return builder.getIntVal(op0->getType());
}

IntVal* PolynomialIntVal::add(ASTBuilder* astBuilder, IntVal* op0, IntVal* op1)
{
    PolynomialIntValBuilder builder(astBuilder);
    builder.addToPolynomialTerm(op0, 1);
    builder.addToPolynomialTerm(op1, 1);
    return builder.getIntVal(op0->getType());
}

IntVal* PolynomialIntVal::mul(ASTBuilder* astBuilder, IntVal* op0, IntVal* op1)
{
    if (auto poly0 = as<PolynomialIntVal>(op0))
    {
        if (auto poly1 = as<PolynomialIntVal>(op1))
        {
            PolynomialIntValBuilder builder(astBuilder);
            // add poly0.constant * poly1.constant
            builder.constantTerm = poly0->getConstantTerm() * poly1->getConstantTerm();
            // add poly0.constant * poly1.terms
            if (poly0->getConstantTerm() != 0)
            {
                for (auto term : poly1->getTerms())
                {
                    auto newTerm = astBuilder->getOrCreate<PolynomialIntValTerm>(
                        poly0->getConstantTerm() * term->getConstFactor(),
                        term->getParamFactors());
                    builder.terms.add(newTerm);
                }
            }
            // add poly1.constant * poly0.terms
            if (poly1->getConstantTerm() != 0)
            {
                for (auto term : poly0->getTerms())
                {
                    auto newTerm = astBuilder->getOrCreate<PolynomialIntValTerm>(
                        poly1->getConstantTerm() * term->getConstFactor(),
                        term->getParamFactors());
                    builder.terms.add(newTerm);
                }
            }
            // add poly1.terms * poly0.terms
            for (auto term0 : poly0->getTerms())
            {
                for (auto term1 : poly1->getTerms())
                {
                    List<PolynomialIntValFactor*> newFactors;
                    for (auto f : term0->getParamFactors())
                        newFactors.add(f);
                    for (auto f : term1->getParamFactors())
                        newFactors.add(f);
                    auto newTerm = astBuilder->getOrCreate<PolynomialIntValTerm>(
                        term0->getConstFactor() * term1->getConstFactor(),
                        newFactors.getArrayView());
                    builder.terms.add(newTerm);
                }
            }
            return builder.getIntVal(op0->getType());
        }
        else if (auto cVal1 = as<ConstantIntVal>(op1))
        {
            PolynomialIntValBuilder builder(astBuilder);
            builder.constantTerm = poly0->getConstantTerm() * cVal1->getValue();
            for (auto term : poly0->getTerms())
            {
                auto newTerm = astBuilder->getOrCreate<PolynomialIntValTerm>(
                    term->getConstFactor() * cVal1->getValue(),
                    term->getParamFactors());
                builder.terms.add(newTerm);
            }
            return builder.getIntVal(poly0->getType());
        }
        else if (auto val1 = as<IntVal>(op1))
        {
            PolynomialIntValBuilder builder(astBuilder);
            auto factor1 = astBuilder->getOrCreate<PolynomialIntValFactor>(val1, 1);
            if (poly0->getConstantTerm() != 0)
            {
                auto term0 = astBuilder->getOrCreate<PolynomialIntValTerm>(
                    poly0->getConstantTerm(),
                    makeArrayViewSingle(factor1));
                builder.terms.add(term0);
            }
            for (auto term : poly0->getTerms())
            {
                List<PolynomialIntValFactor*> newFactors;
                for (auto f : term->getParamFactors())
                    newFactors.add(f);
                newFactors.add(factor1);
                auto newTerm = astBuilder->getOrCreate<PolynomialIntValTerm>(
                    term->getConstFactor(),
                    newFactors.getArrayView());
                builder.terms.add(newTerm);
            }
            return builder.getIntVal(poly0->getType());
        }
        else
            return nullptr;
    }
    else if (as<ConstantIntVal>(op0))
    {
        return mul(astBuilder, op1, op0);
    }
    else if (auto val0 = as<IntVal>(op0))
    {
        if (const auto poly1 = as<PolynomialIntVal>(op1))
        {
            return mul(astBuilder, op1, op0);
        }
        else if (auto cVal1 = as<ConstantIntVal>(op1))
        {
            PolynomialIntValBuilder builder(astBuilder);
            auto factor0 = astBuilder->getOrCreate<PolynomialIntValFactor>(val0, 1);
            auto term = astBuilder->getOrCreate<PolynomialIntValTerm>(
                cVal1->getValue(),
                makeArrayView(&factor0, 1));
            builder.terms.add(term);
            return builder.getIntVal(val0->getType());
        }
        else if (auto val1 = as<IntVal>(op1))
        {
            PolynomialIntValBuilder builder(astBuilder);
            auto factor0 = astBuilder->getOrCreate<PolynomialIntValFactor>(val0, 1);
            auto factor1 = astBuilder->getOrCreate<PolynomialIntValFactor>(val1, 1);
            PolynomialIntValFactor* newFactors[] = {factor0, factor1};
            auto term = astBuilder->getOrCreate<PolynomialIntValTerm>(1, makeArrayView(newFactors));
            builder.terms.add(term);
            return builder.getIntVal(val0->getType());
        }
    }
    return nullptr;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TypeCastIntVal !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

void TypeCastIntVal::_toTextOverride(StringBuilder& out)
{
    getType()->toText(out);
    out << "(";
    getBase()->toText(out);
    out << ")";
}

Val* TypeCastIntVal::tryFoldImpl(
    ASTBuilder* astBuilder,
    Type* resultType,
    Val* base,
    DiagnosticSink* sink)
{
    SLANG_UNUSED(sink);
    auto convertValue = [&](BasicExpressionType* baseType, IntegerLiteralValue& resultValue) -> bool
    {
        switch (baseType->getBaseType())
        {
        case BaseType::Int:
            resultValue = (int)resultValue;
            return true;
        case BaseType::UInt:
            resultValue = (unsigned int)resultValue;
            return true;
        case BaseType::Int64:
        case BaseType::IntPtr:
            resultValue = (Int64)resultValue;
            return true;
        case BaseType::UInt64:
        case BaseType::UIntPtr:
            resultValue = (UInt64)resultValue;
            return true;
        case BaseType::Int16:
            resultValue = (int16_t)resultValue;
            return true;
        case BaseType::UInt16:
            resultValue = (uint16_t)resultValue;
            return true;
        case BaseType::Int8:
            resultValue = (int8_t)resultValue;
            return true;
        case BaseType::UInt8:
            resultValue = (uint8_t)resultValue;
            return true;
        default:
            return false;
        }
    };
    if (auto c = as<ConstantIntVal>(base))
    {
        IntegerLiteralValue resultValue = c->getValue();
        auto baseType = as<BasicExpressionType>(resultType);
        if (baseType)
        {
            if (!convertValue(baseType, resultValue))
                return nullptr;
        }
        else if (auto enumDecl = isEnumType(resultType))
        {
            baseType = as<BasicExpressionType>(enumDecl->tagType);
            if (!baseType || !convertValue(baseType, resultValue))
                return nullptr;
        }
        return astBuilder->getIntVal(resultType, resultValue);
    }
    return nullptr;
}

Val* TypeCastIntVal::_linkTimeResolveOverride(Dictionary<String, IntVal*>& map)
{
    auto intValBase = as<IntVal>(getBase());
    if (!intValBase)
        return this;
    auto resolvedBase = intValBase->linkTimeResolve(map);
    return tryFoldImpl(getCurrentASTBuilder(), getType(), resolvedBase, nullptr);
}

Val* TypeCastIntVal::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    int diff = 0;
    auto substBase = getBase()->substituteImpl(astBuilder, subst, &diff);
    if (substBase != getBase())
        diff++;
    auto substType = as<Type>(getType()->substituteImpl(astBuilder, subst, &diff));
    if (substType != getType())
        diff++;
    *ioDiff += diff;
    if (diff)
    {
        auto newVal = tryFoldImpl(astBuilder, substType, substBase, nullptr);
        if (newVal)
            return newVal;
        else
        {
            auto result = astBuilder->getTypeCastIntVal(substType, substBase);
            return result;
        }
    }
    // Nothing found: don't substitute.
    return this;
}

Val* TypeCastIntVal::_resolveImplOverride()
{
    if (auto resolved = tryFoldImpl(getCurrentASTBuilder(), getType(), getBase(), nullptr))
        return resolved;
    return this;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! FuncCallIntVal !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

void FuncCallIntVal::_toTextOverride(StringBuilder& out)
{
    auto args = getArgs();
    auto funcDeclRef = getFuncDeclRef();

    auto argToText = [&](int index)
    {
        if (as<PolynomialIntVal>(args[index]) || as<FuncCallIntVal>(args[index]))
        {
            out << "(";
            args[index]->toText(out);
            out << ")";
        }
        else
        {
            args[index]->toText(out);
        }
    };
    Name* name = funcDeclRef.getName();
    if (args.getCount() == 2)
    {
        argToText(0);
        out << (name ? name->text : "");
        argToText(1);
        ;
    }
    else if (args.getCount() == 1)
    {
        out << (name ? name->text : "");
        argToText(0);
    }
    else if (name && name->text == "?:")
    {
        argToText(0);
        out << "?";
        argToText(1);
        out << ":";
        argToText(2);
    }
    else
    {
        if (name)
        {
            out << name->text;
        }
        out << "(";
        for (Index i = 0; i < args.getCount(); i++)
        {
            if (i > 0)
                out << ", ";
            args[i]->toText(out);
        }
        out << ")";
    }
}

Val* FuncCallIntVal::_resolveImplOverride()
{
    auto astBuilder = getCurrentASTBuilder();
    auto args = getArgs();
    auto funcDeclRef = getFuncDeclRef();
    auto funcType = getFuncType();

    Val* resolvedVal = this;

    auto newFuncDeclRef = as<DeclRefBase>(funcDeclRef.declRefBase->resolve());
    if (!newFuncDeclRef)
        return this;
    bool diff = false;
    List<IntVal*> newArgs;
    for (auto arg : args)
    {
        auto newArg = as<IntVal>(arg->resolve());
        if (!newArg)
            return this;
        newArgs.add(newArg);
        if (newArg != arg)
            diff = true;
    }

    if (auto resolved = tryFoldImpl(astBuilder, getType(), newFuncDeclRef, newArgs, nullptr))
        resolvedVal = resolved;
    else if (diff)
    {
        resolvedVal = astBuilder->getOrCreate<FuncCallIntVal>(
            getType(),
            newFuncDeclRef,
            funcType,
            newArgs.getArrayView());
    }
    return resolvedVal;
}

Val* FuncCallIntVal::tryFoldImpl(
    ASTBuilder* astBuilder,
    Type* resultType,
    DeclRef<Decl> newFuncDecl,
    List<IntVal*>& newArgs,
    DiagnosticSink* sink)
{
    // Are all args const now?
    List<ConstantIntVal*> constArgs;
    bool allConst = true;
    for (auto arg : newArgs)
    {
        if (auto c = as<ConstantIntVal>(arg))
        {
            constArgs.add(c);
        }
        else
        {
            allConst = false;
            break;
        }
    }
    if (allConst)
    {
        // Evaluate the function.
        auto opName = newFuncDecl.getName();
        SLANG_ASSERT(opName);

        const auto opNameSlice = opName->text.getUnownedSlice();

        IntegerLiteralValue resultValue = 0;

        // Define convenience macros.
        // The last macro used in the list *must* be
        // TERMINATING_CASE, as this handles the closing else, and matches if nothing else does.

#define BINARY_OPERATOR_CASE(op)                                            \
    if (opNameSlice == toSlice(#op))                                        \
    {                                                                       \
        resultValue = constArgs[0]->getValue() op constArgs[1]->getValue(); \
    }                                                                       \
    else

#define DIV_OPERATOR_CASE(op)                                                    \
    if (opNameSlice == toSlice(#op))                                             \
    {                                                                            \
        if (constArgs[1]->getValue() == 0)                                       \
        {                                                                        \
            if (sink)                                                            \
                sink->diagnose(newFuncDecl.getLoc(), Diagnostics::divideByZero); \
            return nullptr;                                                      \
        }                                                                        \
        resultValue = constArgs[0]->getValue() op constArgs[1]->getValue();      \
    }                                                                            \
    else

#define LOGICAL_OPERATOR_CASE(op)                                                          \
    if (opNameSlice == toSlice(#op))                                                       \
    {                                                                                      \
        resultValue =                                                                      \
            (((constArgs[0]->getValue() != 0) op(constArgs[1]->getValue() != 0)) ? 1 : 0); \
    }                                                                                      \
    else


#define SPECIAL_OPERATOR_CASE(op, IF_MATCH) \
    if (opNameSlice == toSlice(op))         \
    {                                       \
        IF_MATCH                            \
    }                                       \
    else

#define TERMINATING_CASE(MATCH) {MATCH}

        // Handle the cases using the macros
        BINARY_OPERATOR_CASE(>=)
        BINARY_OPERATOR_CASE(<=)
        BINARY_OPERATOR_CASE(>)
        BINARY_OPERATOR_CASE(<)
        BINARY_OPERATOR_CASE(!=)
        BINARY_OPERATOR_CASE(==)
        BINARY_OPERATOR_CASE(<<)
        BINARY_OPERATOR_CASE(>>)
        BINARY_OPERATOR_CASE(&)
        BINARY_OPERATOR_CASE(|)
        BINARY_OPERATOR_CASE(^)
        DIV_OPERATOR_CASE(/)
        DIV_OPERATOR_CASE(%)
        LOGICAL_OPERATOR_CASE(&&)
        LOGICAL_OPERATOR_CASE(||)
        // Special cases need their "operator" names quoted.
        SPECIAL_OPERATOR_CASE("!", resultValue = ((constArgs[0]->getValue() != 0) ? 1 : 0);)
        SPECIAL_OPERATOR_CASE("~", resultValue = ~constArgs[0]->getValue();)
        SPECIAL_OPERATOR_CASE("?:",
                              resultValue = constArgs[0]->getValue() != 0
                                                ? constArgs[1]->getValue()
                                                : constArgs[2]->getValue();)
        TERMINATING_CASE(SLANG_UNREACHABLE("constant folding of FuncCallIntVal");)

        return astBuilder->getIntVal(resultType, resultValue);

        // The macros for the cases are no longer needed so undef them all.
#undef BINARY_OPERATOR_CASE
#undef DIV_OPERATOR_CASE
#undef LOGICAL_OPERATOR_CASE
#undef SPECIAL_OPERATOR_CASE
#undef TERMINATING_CASE
    }
    return nullptr;
}

Val* FuncCallIntVal::_linkTimeResolveOverride(Dictionary<String, IntVal*>& map)
{
    List<IntVal*> newArgs;
    for (auto arg : getArgs())
        newArgs.add(as<IntVal>(arg->linkTimeResolve(map)));
    return tryFoldImpl(getCurrentASTBuilder(), getType(), getFuncDeclRef(), newArgs, nullptr);
}

Val* FuncCallIntVal::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    int diff = 0;
    auto newFuncDeclRef = getFuncDeclRef().substituteImpl(astBuilder, subst, &diff);
    List<IntVal*> newArgs;
    for (auto& arg : getArgs())
    {
        auto substArg = arg->substituteImpl(astBuilder, subst, &diff);
        if (substArg != arg)
            diff++;
        newArgs.add(as<IntVal>(substArg));
    }
    *ioDiff += diff;
    if (diff)
    {
        // TODO: report diagnostics back.
        auto newVal = tryFoldImpl(astBuilder, getType(), newFuncDeclRef, newArgs, nullptr);
        if (newVal)
            return newVal;
        else
        {
            auto result = astBuilder->getOrCreate<FuncCallIntVal>(
                getType(),
                newFuncDeclRef,
                getFuncType(),
                newArgs.getArrayView());
            return result;
        }
    }
    // Nothing found: don't substitute.
    return this;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CountOfIntVal !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

void CountOfIntVal::_toTextOverride(StringBuilder& out)
{
    out << "countof(";
    getTypeArg()->toText(out);
    out << ")";
}

Val* CountOfIntVal::tryFoldOrNull(ASTBuilder* astBuilder, Type* intType, Type* newType)
{
    if (auto typePack = as<ConcreteTypePack>(newType))
    {
        bool anyAbstract = false;
        for (Index i = 0; i < typePack->getTypeCount(); i++)
        {
            if (isAbstractTypePack(typePack->getElementType(i)))
            {
                anyAbstract = true;
                break;
            }
        }
        if (!anyAbstract)
        {
            auto result = astBuilder->getIntVal(intType, typePack->getTypeCount());
            return result;
        }
    }
    else if (auto tupleType = as<TupleType>(newType))
    {
        bool anyAbstract = false;
        for (Index i = 0; i < tupleType->getMemberCount(); i++)
        {
            if (isAbstractTypePack(tupleType->getMember(i)))
            {
                anyAbstract = true;
                break;
            }
        }
        if (!anyAbstract)
        {
            auto result = astBuilder->getIntVal(intType, tupleType->getMemberCount());
            return result;
        }
    }
    return nullptr;
}

Val* CountOfIntVal::tryFold(ASTBuilder* astBuilder, Type* intType, Type* newType)
{
    if (auto result = tryFoldOrNull(astBuilder, intType, newType))
        return result;
    auto result = astBuilder->getOrCreate<CountOfIntVal>(intType, newType);
    return result;
}

Val* CountOfIntVal::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    int diff = 0;
    auto newType = as<Type>(getTypeArg()->substituteImpl(astBuilder, subst, &diff));
    if (!diff)
        return this;

    (*ioDiff)++;
    return tryFold(astBuilder, getType(), newType);
}

Val* CountOfIntVal::_resolveImplOverride()
{
    auto resolvedTypeArg = getTypeArg()->resolve();
    if (resolvedTypeArg == getTypeArg())
        return this;
    return tryFold(getCurrentASTBuilder(), getType(), as<Type>(resolvedTypeArg));
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WitnessLookupIntVal !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

void WitnessLookupIntVal::_toTextOverride(StringBuilder& out)
{
    getWitness()->getSub()->toText(out);
    out << ".";
    out << (getKey()->getName() ? getKey()->getName()->text : "??");
}

Val* WitnessLookupIntVal::_resolveImplOverride()
{
    auto astBuilder = getCurrentASTBuilder();

    auto newWitness = as<SubtypeWitness>(getWitness()->resolve());
    if (!newWitness)
        return this;

    auto witnessVal = tryLookUpRequirementWitness(astBuilder, newWitness, getKey());
    if (witnessVal.getFlavor() == RequirementWitness::Flavor::val)
    {
        return witnessVal.getVal();
    }

    auto newType = as<Type>(getType()->resolve());
    if (!newType)
        return this;

    if (newWitness != getWitness() || newType != getType())
    {
        return astBuilder->getOrCreate<WitnessLookupIntVal>(newType, newWitness, getKey());
    }

    return this;
}

Val* WitnessLookupIntVal::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    int diff = 0;
    auto newWitness = getWitness()->substituteImpl(astBuilder, subst, &diff);
    if (diff)
    {
        *ioDiff += diff;
        auto witnessEntry = tryFoldOrNull(astBuilder, as<SubtypeWitness>(newWitness), getKey());
        if (witnessEntry)
        {
            return witnessEntry;
        }
        else
        {
            return astBuilder->getOrCreate<WitnessLookupIntVal>(getType(), newWitness, getKey());
        }
    }
    return this;
}

Val* WitnessLookupIntVal::tryFoldOrNull(ASTBuilder* astBuilder, SubtypeWitness* witness, Decl* key)
{
    auto witnessEntry = tryLookUpRequirementWitness(astBuilder, witness, key);
    switch (witnessEntry.getFlavor())
    {
    case RequirementWitness::Flavor::val:
        return witnessEntry.getVal();
        break;
    default:
        break;
    }
    return nullptr;
}

Val* WitnessLookupIntVal::tryFold(
    ASTBuilder* astBuilder,
    SubtypeWitness* witness,
    Decl* key,
    Type* type)
{
    if (auto result = tryFoldOrNull(astBuilder, witness, key))
        return result;
    auto witnessResult = astBuilder->getOrCreate<WitnessLookupIntVal>(type, witness, key);
    return witnessResult;
}

void DifferentiateVal::_toTextOverride(StringBuilder& out)
{
    out << "DifferentiateVal(";
    out << getFunc();
    out << ")";
}

Val* DifferentiateVal::_substituteImplOverride(
    ASTBuilder* astBuilder,
    SubstitutionSet subst,
    int* ioDiff)
{
    int diff = 0;
    auto newFunc = getFunc().substituteImpl(astBuilder, subst, &diff);
    *ioDiff += diff;
    if (diff)
    {
        auto result = as<DifferentiateVal>(astBuilder->createByNodeType(astNodeType));
        result->getFunc() = newFunc;
        return result;
    }
    // Nothing found: don't substitute.
    return this;
}

Val* DifferentiateVal::_resolveImplOverride()
{
    return this;
}

Val* PolynomialIntValFactor::_resolveImplOverride()
{
    auto astBuilder = getCurrentASTBuilder();

    auto newParam = as<IntVal>(getParam()->resolve());
    if (newParam && newParam != getParam())
        return astBuilder->getOrCreate<PolynomialIntValFactor>(newParam, getPower());

    return this;
}

Val* PolynomialIntValTerm::_resolveImplOverride()
{
    auto astBuilder = getCurrentASTBuilder();

    bool diff = false;
    List<PolynomialIntValFactor*> newFactors;
    for (auto factor : getParamFactors())
    {
        auto newFactor = as<PolynomialIntValFactor>(factor->resolve());
        if (!newFactor)
            return this;

        if (newFactor != factor)
            diff = true;
        newFactors.add(newFactor);
    }

    if (diff)
        return astBuilder->getOrCreate<PolynomialIntValTerm>(
            getConstFactor(),
            newFactors.getArrayView());

    return this;
}

Val* PolynomialIntVal::_resolveImplOverride()
{
    auto astBuilder = getCurrentASTBuilder();

    bool diff = false;
    PolynomialIntValBuilder builder(astBuilder);
    builder.constantTerm = getConstantTerm();
    for (auto term : getTerms())
    {
        auto newTerm = as<PolynomialIntValTerm>(term->resolve());
        if (!newTerm)
            return this;

        if (newTerm != term)
            diff = true;
        builder.terms.add(newTerm);
    }

    if (diff)
        return builder.getIntVal(getType());

    return this;
}

bool IntVal::isLinkTimeVal()
{
    SLANG_AST_NODE_VIRTUAL_CALL(IntVal, isLinkTimeVal, ());
}

Val* IntVal::linkTimeResolve(Dictionary<String, IntVal*>& mapMangledNameToVal)
{
    SLANG_AST_NODE_VIRTUAL_CALL(IntVal, linkTimeResolve, (mapMangledNameToVal));
}

} // namespace Slang
