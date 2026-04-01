// slang-check-conformance.cpp
#include "slang-check-impl.h"

// This file provides semantic checking services related
// to checking and representing the conformance of types
// to interfaces, as well as other subtype relationships.

namespace Slang
{
bool SemanticsVisitor::isInterfaceSafeForTaggedUnion(DeclRef<InterfaceDecl> interfaceDeclRef)
{
    for (auto memberDeclRef : getMembers(m_astBuilder, interfaceDeclRef))
    {
        if (!isInterfaceRequirementSafeForTaggedUnion(interfaceDeclRef, memberDeclRef))
            return false;
    }

    return true;
}

bool SemanticsVisitor::isInterfaceRequirementSafeForTaggedUnion(
    DeclRef<InterfaceDecl> interfaceDeclRef,
    DeclRef<Decl> requirementDeclRef)
{
    SLANG_UNUSED(interfaceDeclRef);

    if (auto callableDeclRef = requirementDeclRef.as<CallableDecl>())
    {
        // A `static` method requirement can't be satisfied by a
        // tagged union, because there is no tag to dispatch on.
        //
        if (requirementDeclRef.getDecl()->hasModifier<HLSLStaticModifier>())
            return false;

        // TODO: We will eventually want to check that any callable
        // requirements do not use the `This` type or any associated
        // types in ways that could lead to errors.
        //
        // For now we are disallowing interfaces that have associated
        // types completely, and we haven't implemented the `This`
        // type, so we should be safe.

        return true;
    }
    else
    {
        return false;
    }
}

SubtypeWitness* SemanticsVisitor::isSubtype(
    Type* subType,
    Type* superType,
    IsSubTypeOptions isSubTypeOptions)
{
    SubtypeWitness* result = nullptr;
    if (getShared()->tryGetSubtypeWitnessFromCache(subType, superType, result))
        return result;
    result = checkAndConstructSubtypeWitness(subType, superType, isSubTypeOptions);

    if (!result && (int(isSubTypeOptions) & int(IsSubTypeOptions::NoCaching)))
        return result;

    getShared()->cacheSubtypeWitness(subType, superType, result);
    return result;
}

SubtypeWitness* SemanticsVisitor::checkAndConstructSubtypeWitness(
    Type* subType,
    Type* superType,
    IsSubTypeOptions isSubTypeOptions)
{
    // TODO: The Slang codebase is currently being quite slippery by conflating
    // multiple concepts, all under the banner of a "subtype" test:
    //
    // * Struct/class inheritance: When concrete type `A` inherits from concrete
    //   type `B`, we can directly convert any value of type `A` into a value of type `B`
    //
    // * Derived interfaces: When interface `X` derives from interface `Y`, we know
    //   that any concrete type conforming to `X` must also conform to `Y`, so we can
    //   derive a witness that `A : Y` from a witness tbale that `A : X` for some concrete `A`
    //
    // * Conformance: When concrete type `A` conforms to interface `X`, we know that there exists
    //   a witness table for that conformance.
    //
    // The problem is that these relationships mean different things. If we use the same
    // `isSubtype()` test for all of the above cases, then we risk determining that `IFoo`
    // *conforms* to `IBar` just because it was declared as `interface IFoo : IBar`. Or
    // even more simply that `IFoo` conforms to `IFoo`.
    //
    // It is dangerous to start treating an interface type like it conforms to itself:
    //
    //      interface IFoo { static int getValue(); }
    //      int get< T : IFoo >() { return T.getValue(); }
    //
    //      int x = get<IFoo>(); // This needs to be an error!!!
    //
    // We will eventually need to clarify the distinction between the different kinds of
    // subtype-ish relationships, *or* we will need to ensure that `interface`s are not
    // treated as proper types (such that they can be passed as generic arguments, etc.)
    //
    // Note that there is one more case of a subtype-ish relationship that is not covered
    // by this function, but that is relevant if/when we do more serious type inference:
    //
    // * Convertibility: When any value of type `A` can be converted to a value of type
    //   `B` (even if that conversion might involve computation or a change of representation),
    //   and that conversion is one that the compiler considers "okay" to do implicitly.
    //
    // For now we are continuing to conflate all the subtype-ish relationships but not
    // tangling convertibility into it.

    // First, make sure both sub type and super type decl are ready for lookup.
    if (!(int(isSubTypeOptions) & int(IsSubTypeOptions::NoCaching)))
    {
        if (auto subDeclRefType = as<DeclRefType>(subType))
        {
            ensureDecl(subDeclRefType->getDeclRef().getDecl(), DeclCheckState::ReadyForLookup);
        }
    }
    if (auto superDeclRefType = as<DeclRefType>(superType))
    {
        ensureDecl(superDeclRefType->getDeclRef().getDecl(), DeclCheckState::ReadyForLookup);
    }

    // In the common case, we can use the pre-computed inheritance information for `subType`
    // to enumerate all the types it transitively inherits from.
    //
    auto inheritanceInfo = getShared()->getInheritanceInfo(subType);
    for (auto facet : inheritanceInfo.facets)
    {
        // The `subType` will have a `facet` for each type
        // that it transitively inherits from, as well as
        // for each `extension` that was found to apply to it.
        //
        // For subtype testing, we are only interested in
        // the facets that represent supertypes, and those
        // will be the ones that store a type on the facet.
        //
        auto facetType = facet->getType();
        if (!facetType)
            continue;

        // We will scan until we find a facet that corresponds
        // to `superType`, or fail to find such a facet.
        //
        if (!facetType->equals(superType))
            continue;

        // If the `superType` appears in the flattened inheritance list
        // for the `subType`, then we know that the subtype relationship
        // holds. Conveniently, the `facet` stores a pre-computed witness
        // for the subtype relationship, which we can return here.
        //
        return facet->subtypeWitness;
    }
    //
    // TODO: We could expand upon the test using the facet list above
    // by taking the facet lists of both `subType` and `superType`
    // and then checking if all of the facets that appear in `superType`'s
    // linearization also appear in the linearization for `subType`
    // (and occur in the same order).
    //
    // That test could potentially handle certain cases of interface
    // conjunctions that the simpler algorithm above can't, but it wouldn't
    // seem to be a complete algorithm unless we ensured that interfaces
    // have a canonical sorting order for how they appear in linearizations.
    //
    // One of the main reasons why we don't implement such a test right now
    // is that it isn't obvious how to directly produce a witness value
    // as collateral from the test.

    // We expect the logic above to cover the vast majority of subtype
    // tests, but there are a few remaining cases of subtype testing
    // that cannot be folded into the type linearizations above.
    //
    // A few of these cases case if the `superType` is a `DeclRefType`
    // and, if so, want to compare its `DeclRef` against others. As
    // such, we will extract the `DeclRef` here, if it exists,
    // as a convienience.
    //
    DeclRef<Decl> superTypeDeclRef;
    if (auto superDeclRefType = as<DeclRefType>(superType))
    {
        superTypeDeclRef = superDeclRefType->getDeclRef();
    }

    if (as<DynamicType>(subType))
    {
        // A __Dynamic type always conforms to the interface via its witness table.
        auto witness = m_astBuilder->getOrCreate<DynamicSubtypeWitness>(subType, superType);
        return witness;
    }
    else if (auto conjunctionSuperType = as<AndType>(superType))
    {
        // We know that `T <: L & R` if `T <: L` and `T <: R`.
        //
        // We therefore simply recursively test both `T <: L`
        // and `T <: R`.
        //
        auto leftWitness =
            isSubtype(subType, conjunctionSuperType->getLeft(), IsSubTypeOptions::None);
        if (!leftWitness)
            return nullptr;
        //
        auto rightWitness =
            isSubtype(subType, conjunctionSuperType->getRight(), IsSubTypeOptions::None);
        if (!rightWitness)
            return nullptr;

        // If both of the sub-relationships hold, we can construct
        // a conjunction of those witnesses to witness `T <: L&R`
        //
        return m_astBuilder->getConjunctionSubtypeWitness(
            subType,
            conjunctionSuperType,
            leftWitness,
            rightWitness);
    }
    else if (auto extractExistentialType = as<ExtractExistentialType>(subType))
    {
        // An ExtractExistentialType from an existential value of type I
        // is a subtype of I.
        // We need to check and make sure the interface type of the `ExtractExistentialType`
        // is equal to `superType`.
        //
        // TODO(tfoley): We could add support for `ExtractExistentialType` to
        // the inheritance linearization logic, and eliminate this case.
        //
        auto interfaceDeclRef = extractExistentialType->getOriginalInterfaceDeclRef();
        if (interfaceDeclRef.equals(superTypeDeclRef))
        {
            auto witness = extractExistentialType->getSubtypeWitness();
            return witness;
        }
        return nullptr;
    }
    else if (auto subTypePack = as<ConcreteTypePack>(subType))
    {
        // A type pack (T0, T1, ...) is a subtype of supType, if each of its elements
        // is a subtype of the supType.
        ShortList<SubtypeWitness*> elementWitnesses;
        for (Index i = 0; i < subTypePack->getTypeCount(); i++)
        {
            auto elementWitness =
                isSubtype(subTypePack->getElementType(i), superType, IsSubTypeOptions::None);
            if (!elementWitness)
                return nullptr;
            elementWitnesses.add(elementWitness);
        }
        return m_astBuilder->getSubtypeWitnessPack(
            subType,
            superType,
            elementWitnesses.getArrayView().arrayView);
    }
    else if (auto expandType = as<ExpandType>(subType))
    {
        // A expand type `expand patternType, captureList` is a subtype of supType, if patternType
        // is a subtype of supType.
        auto elementWitness =
            isSubtype(expandType->getPatternType(), superType, IsSubTypeOptions::None);
        if (!elementWitness)
            return nullptr;
        return m_astBuilder->getExpandSubtypeWitness(subType, superType, elementWitness);
    }
    else if (auto eachType = as<EachType>(subType))
    {
        auto elementWitness =
            isSubtype(eachType->getElementType(), superType, IsSubTypeOptions::None);
        if (!elementWitness)
            return nullptr;
        return m_astBuilder->getEachSubtypeWitness(subType, superType, elementWitness);
    }
    // default is failure
    return nullptr;
}

bool SemanticsVisitor::isValidGenericConstraintType(Type* type)
{
    if (auto andType = as<AndType>(type))
    {
        return isValidGenericConstraintType(andType->getLeft()) &&
               isValidGenericConstraintType(andType->getRight());
    }
    return isInterfaceType(type);
}

SubtypeWitness* SemanticsVisitor::isTypeDifferentiable(Type* type)
{
    if (auto valueWitness =
            isSubtype(type, m_astBuilder->getDiffInterfaceType(), IsSubTypeOptions::None))
        return valueWitness;
    else if (
        auto ptrWitness = isSubtype(
            type,
            m_astBuilder->getDifferentiableRefInterfaceType(),
            IsSubTypeOptions::None))
        return ptrWitness;

    return nullptr;
}

bool SemanticsVisitor::doesTypeHaveTag(Type* type, TypeTag tag)
{
    if (auto arrayType = as<ArrayExpressionType>(type))
    {
        return doesTypeHaveTag(arrayType->getElementType(), tag);
    }
    if (auto modifiedType = as<ModifiedType>(type))
    {
        return doesTypeHaveTag(modifiedType->getBase(), tag);
    }
    if (auto declRefType = as<DeclRefType>(type))
    {
        if (auto aggTypeDecl = as<AggTypeDecl>(declRefType->getDeclRef()))
            return aggTypeDecl.getDecl()->hasTag(tag);
    }
    return false;
}

TypeTag SemanticsVisitor::getTypeTags(Type* type)
{
    if (auto arrayType = as<ArrayExpressionType>(type))
    {
        auto typeTag = getTypeTags(arrayType->getElementType());
        bool sized = false;
        if (auto cint = as<ConstantIntVal>(arrayType->getElementCount()))
        {
            if (cint->getValue() != kUnsizedArrayMagicLength)
            {
                sized = true;
            }
        }
        else if (arrayType->getElementCount())
        {
            sized = true;
            typeTag = (TypeTag)((int)typeTag | (int)TypeTag::LinkTimeSized);
        }
        if (!sized)
            typeTag = (TypeTag)((int)typeTag | (int)TypeTag::Unsized);

        return typeTag;
    }
    if (auto modifiedType = as<ModifiedType>(type))
    {
        return getTypeTags(modifiedType->getBase());
    }
    if (auto parameterGroupType = as<UniformParameterGroupType>(type))
    {
        auto elementTags = getTypeTags(parameterGroupType->getElementType());
        elementTags = (TypeTag)(((int)elementTags & ~(int)TypeTag::Unsized) | (int)TypeTag::Opaque);
        return elementTags;
    }
    else if (
        as<UntypedBufferResourceType>(type) || as<ResourceType>(type) ||
        as<SamplerStateType>(type) || as<HLSLStructuredBufferTypeBase>(type) ||
        as<DynamicResourceType>(type))
    {
        return TypeTag::Opaque;
    }
    else if (auto declRefType = as<DeclRefType>(type))
    {
        if (auto aggTypeDecl = as<AggTypeDecl>(declRefType->getDeclRef()))
            return aggTypeDecl.getDecl()->typeTags;
    }
    return TypeTag::None;
}


Type* SemanticsVisitor::getConstantBufferElementType(Type* type)
{
    if (auto arrType = as<ArrayExpressionType>(type))
        return getConstantBufferElementType(arrType->getElementType());
    if (auto modifiedType = as<ModifiedType>(type))
        return getConstantBufferElementType(modifiedType->getBase());
    if (auto constantBuffer = as<ConstantBufferType>(type))
        return constantBuffer->getElementType();
    if (auto parameterBlock = as<ParameterBlockType>(type))
        return parameterBlock->getElementType();
    return nullptr;
}


SubtypeWitness* SemanticsVisitor::tryGetInterfaceConformanceWitness(Type* type, Type* interfaceType)
{
    return isSubtype(type, interfaceType, IsSubTypeOptions::None);
}

TypeEqualityWitness* SemanticsVisitor::createTypeEqualityWitness(Type* type)
{
    return m_astBuilder->getTypeEqualityWitness(type);
}
} // namespace Slang
