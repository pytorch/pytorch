// slang-check-inheritance.cpp
#include "slang-check-impl.h"

// This file implements the semantic checking logic
// related to computing linearized inheritance
// information for types and decalrations.

namespace Slang
{
InheritanceInfo SharedSemanticsContext::getInheritanceInfo(
    Type* type,
    InheritanceCircularityInfo* circularityInfo)
{
    // We cache the computed inheritance information for types,
    // and re-use that information whenever possible.

    // DeclRefTypes will have their inheritance info cached in m_mapDeclRefToInheritanceInfo.
    if (auto declRefType = as<DeclRefType>(type))
        return _getInheritanceInfo(declRefType->getDeclRef(), declRefType, circularityInfo);

    // Non ordinary types are cached on m_mapTypeToInheritanceInfo.
    if (auto found = m_mapTypeToInheritanceInfo.tryGetValue(type))
        return *found;

    // Note: we install a null pointer into the dictionary to act
    // as a sentinel during the processing of calculating the inheritnace
    // info. If we encounter this sentinel value during the calcuation,
    // it means that there was some kind of circular dependency in the
    // inheritance graph, and we need to avoid crashing or going
    // into an infinite loop in such cases.
    //
    m_mapTypeToInheritanceInfo[type] = InheritanceInfo();

    auto info = _calcInheritanceInfo(type, circularityInfo);
    m_mapTypeToInheritanceInfo[type] = info;

    return info;
}

InheritanceInfo SharedSemanticsContext::getInheritanceInfo(
    DeclRef<ExtensionDecl> const& extension,
    InheritanceCircularityInfo* circularityInfo)
{
    if (_checkForCircularityInExtensionTargetType(extension.getDecl(), circularityInfo))
    {
        // If we detect a circularity in the inheritance graph,
        // we will return an empty `InheritanceInfo` to avoid
        // infinite recursion.
        //
        return InheritanceInfo();
    }

    // We bottleneck the calculation of inheritance information
    // for type and `extension` `DeclRef`s through a single
    // routine with an optional `Type` parameter.
    //
    InheritanceCircularityInfo newCircularityInfo(extension.getDecl(), circularityInfo);
    return _getInheritanceInfo(extension, nullptr, &newCircularityInfo);
}

bool SharedSemanticsContext::_checkForCircularityInExtensionTargetType(
    Decl* decl,
    InheritanceCircularityInfo* circularityInfo)
{
    for (auto info = circularityInfo; info; info = info->next)
    {
        if (decl == info->decl)
        {
            getSink()->diagnose(decl, Diagnostics::circularityInExtension, decl);
            return true;
        }
    }

    return false;
}

InheritanceInfo SharedSemanticsContext::_getInheritanceInfo(
    DeclRef<Decl> declRef,
    Type* selfType,
    InheritanceCircularityInfo* circularityInfo)
{
    // Just as with `Type`s, we cache and re-use the inheritance
    // information that has been computed for a `DeclRef` whenever
    // possible.

    if (auto found = m_mapDeclRefToInheritanceInfo.tryGetValue(declRef))
        return *found;

    // Note: we install a null pointer into the dictionary to act
    // as a sentinel during the processing of calculating the inheritnace
    // info. If we encounter this sentinel value during the calcuation,
    // it means that there was some kind of circular dependency in the
    // inheritance graph, and we need to avoid crashing or going
    // into an infinite loop in such cases.
    //
    m_mapDeclRefToInheritanceInfo[declRef] = InheritanceInfo();

    auto info = _calcInheritanceInfo(declRef, selfType, circularityInfo);
    m_mapDeclRefToInheritanceInfo[declRef] = info;

    getSession()->m_typeDictionarySize = Math::Max(
        getSession()->m_typeDictionarySize,
        (int)m_mapDeclRefToInheritanceInfo.getCount());

    return info;
}

void SharedSemanticsContext::getDependentGenericParentImpl(
    DeclRef<GenericDecl>& genericParent,
    DeclRef<Decl> declRef)
{
    auto mergeParent = [](DeclRef<GenericDecl>& currentParent, DeclRef<GenericDecl> newParent)
    {
        if (!currentParent)
        {
            currentParent = newParent;
            return;
        }
        if (currentParent == newParent)
            return;
        if (newParent.getDecl()->isChildOf(currentParent.getDecl()))
            currentParent = newParent;
    };

    if (declRef.as<GenericTypeParamDeclBase>())
    {
        if (!genericParent)
            mergeParent(genericParent, declRef.getParent().as<GenericDecl>());
        return;
    }
    else if (auto lookupDeclRef = as<LookupDeclRef>(declRef.declRefBase))
    {
        if (auto lookupSourceDeclRef = isDeclRefTypeOf<Decl>(lookupDeclRef->getLookupSource()))
            getDependentGenericParentImpl(genericParent, lookupSourceDeclRef);
    }
    else if (auto genericAppDeclRef = as<GenericAppDeclRef>(declRef.declRefBase))
    {
        for (Index i = 0; i < genericAppDeclRef->getArgCount(); i++)
        {
            if (auto argDeclRef = isDeclRefTypeOf<Decl>(genericAppDeclRef->getArg(i)))
            {
                getDependentGenericParentImpl(genericParent, argDeclRef);
            }
        }
    }
}

DeclRef<GenericDecl> SharedSemanticsContext::getDependentGenericParent(DeclRef<Decl> declRef)
{
    DeclRef<GenericDecl> genericParent;
    getDependentGenericParentImpl(genericParent, declRef);
    return genericParent;
}

InheritanceInfo SharedSemanticsContext::_calcInheritanceInfo(
    DeclRef<Decl> declRef,
    Type* selfType,
    InheritanceCircularityInfo* circularityInfo)
{
    // This method is the main engine for computing linearized inheritance
    // lists for types and `extension` declarations.
    //
    // The approach we use for linearization of an inheritance graph is based on
    // what is the most broadly-accepted solution to the problem, presented in
    // "A Monotonic Superclass Linearization for Dylan" by Barret et al.
    // The algorithm recommended in that paper is also called the "C3 linearization
    // algorithm." Many developers are most familiar with C3 linearization because
    // it is used to compute the method resolution order (MRO) for a class in Python.
    //
    // The basic idea is that given a type declaration like:
    //
    //      class A : B, C { ... }
    //
    // we can construct a linearization of the transitive bases of `A`
    // by merging the linearizations for `B` and `C`. Any transitive
    // base of `A` should appear in the linearization for `B` and/or `C`,
    // so the main tasks are to remove duplicates (when a base type appears
    // in both the linearization of `B` *and* `C`), and to ensure that
    // the ordering is reasonable.
    //
    // What makes an ordering "reasonable" is a little subtle, especially
    // in the context of Slang. In the original use case, the order of types
    // in the linearization would determine which methods would override
    // which other ones, so different orderings could have large semantic
    // impact. Slang currently has less support for overriding, but is
    // likely to add more over time.
    //
    // At the very least, we require that if `S <: T` for types `S` and `T`,
    // then `S` should appear *before* `T` in the linearization. This, e.g.,
    // guarantees that a concrete type that implements an `interface` will
    // be listed before that interface and thus during lookup the members
    // of the concrete type will be found before those of the `interface`.
    //
    // We will revisit the question of "reasonable" orderings later, as
    // we get more into the core of the algorithm.

    // Our linearization approach will construct a list of *facets* for
    // the `declRef` in question, where each facet corresponds to a
    // transitive base type, or an applicable `extension`.
    //
    FacetList::Builder allFacets;

    Facet::Kind selfFacetKind = Facet::Kind::Type;

    auto astBuilder = _getASTBuilder();
    auto& arena = astBuilder->getArena();
    SemanticsVisitor visitor(this);
    if (auto extensionDeclRef = declRef.as<ExtensionDecl>())
    {
        auto extendedType = getTargetType(astBuilder, extensionDeclRef);
        selfType = extendedType;
        selfFacetKind = Facet::Kind::Extension;
    }

    // Because we are dealing with entities that have declarations, the
    // first item in our linearization will always be a facet for
    // the declaration itself.
    //
    TypeEqualityWitness* selfIsSelf =
        selfType ? visitor.createTypeEqualityWitness(selfType) : nullptr;
    Facet selfFacet = new (arena)
        Facet::Impl(selfFacetKind, Facet::Directness::Self, declRef, selfType, selfIsSelf);
    allFacets.add(selfFacet);

    // After the self facet will come a list of facets formed
    // by merging the facet lists of each of the direct/declared
    // bases of the type/declaration in question.
    //
    // We will first traverse the structure of `declRef` to
    // accumulate the list of bases, and then perform the merge
    // when we are done.
    //
    DirectBaseList::Builder directBases;
    FacetList::Builder directBaseFacets;

    // We start with a simple operation to add an entry
    // into the list of direct bases, for the case where
    // we already have all of the relevant information
    // about that base.
    //
    auto addDirectBaseFacet = [&](Facet::Kind kind,
                                  Type* baseType,
                                  SubtypeWitness* selfIsBaseWitness,
                                  DeclRef<Decl> const& baseDeclRef,
                                  InheritanceInfo const& baseInheritanceInfo)
    {
        auto baseInfo = new (arena) DirectBaseInfo();

        // The information we store for each direct
        // base comprises two main things.
        //
        // First, we have a `Facet` that will represent
        // the base in the linearized inheritance list
        // we are building.
        //
        baseInfo->facetImpl =
            FacetImpl(kind, Facet::Directness::Direct, baseDeclRef, baseType, selfIsBaseWitness);
        Facet baseFacet(&baseInfo->facetImpl);
        //
        // Second, we have a list of the facets in the
        // linearization of the base itself.
        //
        baseInfo->facets = baseInheritanceInfo.facets;

        directBaseFacets.add(baseFacet);
        directBases.add(baseInfo);
    };

    // In the case where we know that the base being added
    // represents a direct base *type* (and not an `extension`)
    // we can derive some of the information needed by
    // `addDirectBaseFacet`.
    //
    auto addDirectBaseType = [&](Type* baseType, SubtypeWitness* selfIsBaseWitness)
    {
        // If we are representing inheritance from a type,
        // then we should have a witness that the type
        // in question (either the type being declared by
        // `declRef`, or the type being *extended* by
        // `declRef`) inherits from that base.
        //
        SLANG_ASSERT(selfIsBaseWitness);

        auto baseInheritanceInfo = getInheritanceInfo(baseType, circularityInfo);

        DeclRef<Decl> baseDeclRef;
        if (auto baseDeclRefType = as<DeclRefType>(baseType))
        {
            baseDeclRef = baseDeclRefType->getDeclRef();
        }

        addDirectBaseFacet(
            Facet::Kind::Type,
            baseType,
            selfIsBaseWitness,
            baseDeclRef,
            baseInheritanceInfo);
    };

    // If we know the type has a facet represented by `extensionTargetDeclRef`, we can consider
    // all extensions on this decl to see if they apply to the type.
    //
    auto considerExtension = [&](DeclRef<AggTypeDecl> extensionTargetDeclRef,
                                 Dictionary<Type*, SubtypeWitness*>* additionalSubtypeWitness)
    {
        bool result = false;
        auto candidateExtensions = getCandidateExtensions(extensionTargetDeclRef, &visitor);
        for (auto extDecl : candidateExtensions)
        {
            // The list of *candidate* extensions is computed and
            // cached based on the identity of the declaration alone,
            // and does not take into account any generic arguments
            // of either the type or the `extension`.
            //
            // For example, we might have an `extension` that applies
            // to `vector<float,N>` for any `N`, but the `selfType`
            // that we are working with could be `<vector<int,2>` so
            // that the extension doesn't match.
            //
            // In order to make sure that we don't enumerate members
            // that don't make sense in context, we must apply
            // the extension to the type and see if we succeed in
            // making a match.
            //
            auto extDeclRef =
                applyExtensionToType(&visitor, extDecl, selfType, additionalSubtypeWitness);
            if (!extDeclRef)
                continue;

            // In the case where we *do* find an extension that
            // applies to the type, we add a declared base to
            // represent the `extension`, knowing that its
            // own linearized inheritance list will include
            // any transitive based declared on the `extension`.
            //
            auto extInheritanceInfo = getInheritanceInfo(extDeclRef, circularityInfo);
            addDirectBaseFacet(
                Facet::Kind::Extension,
                selfType,
                selfIsSelf,
                extDeclRef,
                extInheritanceInfo);
            result = true;
        }
        return result;
    };

    // We now look at the structure of the declaration itself
    // to help us enumerate the direct bases.
    //
    auto currentDeclRef = declRef;
    for (; currentDeclRef;)
    {
        if (auto aggTypeDeclBaseRef = currentDeclRef.as<AggTypeDeclBase>())
        {
            // In the case where we have an aggregate type or `extension`
            // declaration, we can use the explicit list of direct bases.
            //
            for (auto typeConstraintDeclRef :
                 getMembersOfType<TypeConstraintDecl>(_getASTBuilder(), aggTypeDeclBaseRef))
            {
                // Note: In certain cases something takes the *syntactic* form of an inheritance
                // clause, but it is not actually something that should be treated as implying
                // a subtype relationship. For example, an `enum` declaration can use what looks
                // like an inheritance clause to indicate its underlying "tag type."
                //
                // We skip such pseudo-inheritance relationships for the purposes of determining
                // the linearized list of bases.
                //
                if (typeConstraintDeclRef.getDecl()->hasModifier<IgnoreForLookupModifier>())
                    continue;

                // The only case we will ever see a GenericTypeConstraintDecl inside a AggTypeDecl
                // is when AggTypeDecl is a associatedtype decl. In this case, we will only lookup
                // the type constraint if the constraint is on the associated type itself.
                //
                auto genericTypeConstraintDeclRef =
                    typeConstraintDeclRef.as<GenericTypeConstraintDecl>();
                if (genericTypeConstraintDeclRef)
                {
                    // If the base expr on the constraint isn't even a `VarExpr`, then it can't be
                    // referencing the associated type itself and we can skip this constraint.
                    if (!genericTypeConstraintDeclRef.getDecl()->sub.type &&
                        !as<VarExpr>(genericTypeConstraintDeclRef.getDecl()->sub.exp))
                        continue;
                }

                visitor.ensureDecl(
                    typeConstraintDeclRef,
                    DeclCheckState::CanUseBaseOfInheritanceDecl);

                // For generic type constraint decls, always make sure it is about the type being
                // checked.
                //
                if (genericTypeConstraintDeclRef)
                {
                    auto subType = getSub(astBuilder, genericTypeConstraintDeclRef);
                    if (subType != selfType)
                        continue;
                }
                else if (currentDeclRef != declRef)
                {
                    continue;
                }
                // The base type and subtype witness can easily be determined
                // using the `InheritanceDecl`.
                //
                auto baseType = getSup(astBuilder, typeConstraintDeclRef);
                auto satisfyingWitness = astBuilder->getDeclaredSubtypeWitness(
                    selfType,
                    baseType,
                    typeConstraintDeclRef);

                addDirectBaseType(baseType, satisfyingWitness);
            }
        }
        if (currentDeclRef.as<AssocTypeDecl>())
        {
            // If the current type is an associated type, continue inspecting the base/parent of the
            // associatedtype to discover additional constraints defined on the parent
            // associatedtype decls.
            //
            if (auto lookupDeclRef = as<LookupDeclRef>(currentDeclRef.declRefBase))
            {
                currentDeclRef =
                    isDeclRefTypeOf<Decl>(lookupDeclRef->getLookupSource()).as<AssocTypeDecl>();
                continue;
            }
        }
        break;
    }

    if (auto genericDeclRef = getDependentGenericParent(declRef))
    {
        // The constraints placed on a generic type parameter are siblings of that
        // parameter in its parent `GenericDecl`, so we need to enumerate all of
        // the constraints of the parent declaration to find those that constrain
        // this parameter.
        //
        // TODO(tfoley): We might consider adding a cached representation of the
        // constraint information that is keyed on a per-parameter basis. Such a
        // representation would need to take into account canonicalization of
        // constraints.

        if (auto extensionDecl = as<ExtensionDecl>(genericDeclRef.getDecl()->inner))
        {
            if (isDeclRefTypeOf<GenericTypeParamDecl>(extensionDecl->targetType.type) == declRef)
            {
                // If `T` is a generic parameter where the same generic is an extension on `T`,
                // then we need to add the extension itself as a facet.
                //
                auto extDeclRef =
                    createDefaultSubstitutionsIfNeeded(astBuilder, &visitor, extensionDecl);
                auto selfExtFacet = new (arena) Facet::Impl(
                    Facet::Kind::Extension,
                    Facet::Directness::Direct,
                    extDeclRef,
                    selfType,
                    astBuilder->getTypeEqualityWitness(selfType));
                allFacets.add(selfExtFacet);
            }
        }

        bool selfIsGenericParamType =
            isDeclRefTypeOf<GenericTypeParamDeclBase>(selfType) != nullptr;

        for (auto constraintDeclRef :
             getMembersOfType<GenericTypeConstraintDecl>(astBuilder, genericDeclRef))
        {
            if (constraintDeclRef.getDecl()->checkState.isBeingChecked())
                continue;

            ensureDecl(&visitor, constraintDeclRef.getDecl(), DeclCheckState::ScopesWired);

            // Check only the sub-type.
            visitor.CheckConstraintSubType(constraintDeclRef.getDecl()->sub);
            auto sub = constraintDeclRef.getDecl()->sub;

            // If the sub-type part of the generic constraint is a member expression, it can't
            // possibly be defining a constraint for a generic type parameter, so we skip it
            // to avoid circular checking on the generic param type.
            if (selfIsGenericParamType && as<MemberExpr>(sub.exp))
                continue;

            if (!sub.type)
                sub = visitor.TranslateTypeNodeForced(sub);
            auto subType = constraintDeclRef.substitute(astBuilder, sub.type);

            // We only consider constraints where the type represented
            // by `declRef` is the subtype, since those
            // constraints are the ones that give us information about
            // the declared supertypes.
            //
            auto subDeclRefType = as<DeclRefType>(subType);
            if (!subDeclRefType)
            {
                if (auto subEachType = as<EachType>(subType))
                {
                    subDeclRefType = as<DeclRefType>(subEachType->getElementType());
                }
                if (!subDeclRefType)
                    continue;
            }
            if (subDeclRefType->getDeclRef() != declRef)
                continue;

            // Further check the constraint, since we now need the sup-type.
            ensureDecl(&visitor, constraintDeclRef.getDecl(), DeclCheckState::CanSpecializeGeneric);

            auto superType = getSup(astBuilder, constraintDeclRef);

            // Because the constraint is a declared inheritance relationship,
            // adding the base to our list of direct bases is as straightforward
            // as in all the preceding cases.
            //
            auto satisfyingWitness =
                _getASTBuilder()->getDeclaredSubtypeWitness(selfType, superType, constraintDeclRef);
            addDirectBaseType(superType, satisfyingWitness);
        }
    }

    // At this point we have enumerated all of the bases that can be
    // gleaned by looking at the `declRef` itself. The next step is
    // to consider any `extension` declarations that might apply to
    // a type being delared.
    //
    // An `extension` may apply to our type, if it directly extends
    // the type, or extends a generic `T` type that are constrained
    // on one of the interfaces that our type conforms to.
    //
    if (auto directAggTypeDeclRef = declRef.as<AggTypeDecl>())
    {
        considerExtension(directAggTypeDeclRef, nullptr);
    }
    if (!declRef.as<ExtensionDecl>())
    {
        HashSet<Type*> supTypesConsideredForExtensionApplication;
        Dictionary<Type*, SubtypeWitness*> additionalSubtypeWitnesses;
        for (;;)
        {
            // After we flatten the list of bases, we may discover additional opportunities
            // to apply extensions.
            List<DeclRef<AggTypeDecl>> supTypeWorkList;
            auto base = directBases.begin();
            for (auto baseFacet = directBaseFacets.getHead(); baseFacet.getImpl();
                 baseFacet = baseFacet->next)
            {
                for (auto facet : (*base)->facets)
                {
                    if (auto interfaceDeclRef = facet->origin.declRef.as<InterfaceDecl>())
                    {
                        SubtypeWitness* transitiveWitness = baseFacet->subtypeWitness;
                        transitiveWitness = astBuilder->getTransitiveSubtypeWitness(
                            baseFacet->subtypeWitness,
                            facet->subtypeWitness);
                        additionalSubtypeWitnesses.addIfNotExists(
                            facet->origin.type,
                            transitiveWitness);
                        if (supTypesConsideredForExtensionApplication.add(facet->origin.type))
                        {
                            supTypeWorkList.add(interfaceDeclRef);
                        }
                    }
                }
                ++base;
            }
            bool canExit = true;
            for (auto baseItem : supTypeWorkList)
            {
                if (considerExtension(baseItem, &additionalSubtypeWitnesses))
                    canExit = false;
            }
            if (canExit)
                break;
        }
    }

    // At this point, the list of direct bases (each with its own linearization)
    // is complete.
    //
    // At this point we could scan through the list of bases and perform
    // consistency checks on it. For example, when two types in the list of direct
    // bases have a subtype relationship between them, it is possible that the
    // programmer made some kind of mistake, and we could emit a diagnostic
    // about it.
    //
    // The published C3 algorithm actually considers the declared list of bases
    // as one of the inputs to its merge, and is very strict about ordering.
    // As such, it would be an error for strict C3 if direct bases were declared
    // in an order that is inconsitent with the partial order determined by
    // the subtype relationship. Our implementation of linearization is relaxed
    // compared to C3 so that it is robust to such ordering issues.
    //
    // Note: This step takes quadratic time in the number of direct bases, but
    // there's really no other way to easily detect these issues.
    //
    for (auto leftBase = directBaseFacets.getHead(); leftBase.getImpl(); leftBase = leftBase->next)
    {
        // Note: all of the direct base facets with a `Type` kind will
        // precede all of those with `Extension` kind, so we can bail
        // out of the outer loop as soon as we find a non-`Type`
        // facet.
        //
        if (leftBase->kind != Facet::Kind::Type)
            break;
        auto leftBaseType = leftBase->origin.type;

        // For the inner loop we scan only the facets that appear *after*
        // the `leftBase` in the list of direct bases.
        //
        for (auto rightBase = leftBase->next; rightBase.getImpl(); rightBase = rightBase->next)
        {
            if (rightBase->kind != Facet::Kind::Type)
                break;
            auto rightBaseType = rightBase->origin.type;

            if (visitor.isSubtype(leftBaseType, rightBaseType, IsSubTypeOptions::None))
            {
                // If a type earlier in the list of bases is a subtype of
                // one later in the list, then the ordering is consistent
                // with the linearization that will be produced, but it
                // might represent a mistake on the programmer's part,
                // since they listed a base type that is redundant.
                //
                // TODO: decide whether to diagnose this case.
            }
            else if (visitor.isSubtype(rightBaseType, leftBaseType, IsSubTypeOptions::None))
            {
                // If a type later in the list is a subtype of a type earlier
                // in the list, then the declared list of bases is inconsistent
                // with the ordering that will (indeed *must*) appear in the
                // linearization we generate.
                //
                // If we end up implementing a strict version of the C3 algorithm,
                // we would need to treat such situations as an error, or at least
                // emit a warning and then remove the subtype from the list of
                // bases.
                //
                // TODO: decide whether to diagnose this case.
            }
        }
    }

    // Now that we've built up the list of direct bases and their
    // respective linearizations, we can apply the core merge algorithm
    // to those lists to produce the rest of the linearization for
    // the declaration in question.
    //
    _mergeFacetLists(directBases, directBaseFacets, allFacets);

    InheritanceInfo info;
    info.facets = allFacets;
    return info;
}

void SharedSemanticsContext::_mergeFacetLists(
    DirectBaseList bases,
    FacetList baseFacets,
    FacetList::Builder& ioMergedFacets)
{
    // Our task here is to take the list of direct/declared `bases`,
    // each of which holds a linearized list of `Facet`s, and produce
    // a single linearized list of facets in `ioMergedFacets`.
    //
    // The `Facet`s in the lists referenced by `bases` are always
    // relative to the base type/extension itself, and not to
    // the type or declaration for which we are computing
    // a linearization.
    //
    // The `baseFacets` list provides one `Facet` for each direct
    // base that are relative to the type/declaration we are
    // computing a linearization for. These facets will be used
    // directly, instead of those from `bases`, where possible.
    //
    auto astBuilder = _getASTBuilder();
    auto& arena = astBuilder->getArena();
    for (;;)
    {
        // The basic logic here is that on each iteration we
        // will look at the first item on each list in `bases`
        // and pick one that we will append to the merged output
        // (after removing it from the relevant input(s)).

        // If we have run out of lists that need merging, then we are done.
        //
        if (bases.isEmpty())
            break;

        // Otherwise, we will look at the remaining non-empty lists,
        // and see if one of them starts with an facet that can
        // be appended to our merged output.
        //
        // If multiple such facets are viable, we will always take
        // the one from the earliest list in `bases`. Doing so favors
        // the types that appear earlier in a list of bases.
        //
        Facet foundFacet;
        DirectBaseInfo* foundBase = nullptr;
        for (auto base : bases)
        {
            Facet headFacet = base->facets.getHead();

            // If the head facet of the `base` list appears at a non-head
            // position in any of the other lists, we cannot append this
            // element without risking inverting the order of some facets
            // relative to those other lists.
            //
            if (bases.doesAnyTailContainMatchFor(headFacet))
                continue;

            // Otherwise, we are safe to add the `headFacet` to our
            // merged list, because it only ever appears as the head
            // of one or more of the lists in `bases`.
            //
            foundFacet = headFacet;
            foundBase = base;
            break;
        }

        if (!foundFacet)
        {
            // If we could not identify a facet that could be safely
            // removed from any of the base lists, then it means that
            // we must have a cycle in the ordering constraints implied
            // by the `bases` lists.
            //
            // The simplest example of such a cycle would be if we
            // had two lists, `A` and `B`, such that:
            //
            //      A = { X, Y }
            //      B = { Y, X }
            //
            // In this case, producing output in the order `X, Y` *or*
            // `Y, X` will always invalidate the ordering constraints
            // implied by either `A` or `B`.
            //
            // In the C3 algorithm as published, such a situation is an
            // error, and the algorithm fails to produce a linearization.
            // The reason for this decision is that allowing this case
            // means that a base type and a derived type could disagree
            // on the relative priority of method overrides, and thus
            // a subclass could possible break semantic assumptions of
            // a superclass.
            //
            // In a more static language like Slang, it seems better to
            // allow more flexible inheritance, *especially* when dealing
            // with things like `interface`s and `extension`s, where the
            // relative ordering of things will often be immaterial.
            //
            // In a case like this, we would like to arbitrarily pick
            // one or the other of `X` and `Y`, and given our default
            // policy to favor the earlier list in `bases` where possible,
            // we would select `X` from `A`.
            //
            // One thing worth noting is that when a case like the above
            // arises, it is not possible that `X <: Y` or `Y <: X`.
            // If a subtype relationship existed between the two, then
            // they would be consistently ordered in *both* lists.
            // We thus do not have to worry about violating the most
            // important requirement for a "reasonable" linearization.
            //
            foundBase = *bases.begin();
            foundFacet = foundBase->facets.getHead();

            // Note: because we are grabbing a facet that might appear
            // in a non-head position in one or more of our lists,
            // we need to have a plan for what to do when we see
            // that same facet come to the front of one of our lists
            // later.
        }

        // If we still cannot find a facet, then there is a true cycle in
        // the inheritance graph, which is an error in the user code.
        if (!foundFacet.getImpl())
        {
            if (!bases.isEmpty())
            {
                auto baseDecl = (*bases.begin())->facetImpl.origin.declRef.getDecl();
                getSink()->diagnose(baseDecl, Diagnostics::cyclicReferenceInInheritance, baseDecl);
            }
            return;
        }

        // At this point we definitely have a facet we'd like to
        // add to the output, whether it was found via the true
        // C3 approach, or our relaxed rule above.
        //
        SLANG_ASSERT(foundFacet.getImpl());

        // If the facet we want to append to the output is the same as the front-most
        // facet on the list of bases, then we want to use that facet as-is (since we
        // have already allocated storage for it).
        //
        // TODO: in cases where the strict C3 algorithm would fail, and we choose a
        // `foundFacet` that is at a non-head position in at least some lists, it
        // might be possible that we have a facet that matches ones of the `baseFacets`,
        // but not the head one. We should confirm what happens in that case.
        //
        if (originsMatch(foundFacet, baseFacets.getHead()))
        {
            auto directBaseFacet = baseFacets.popHead();
            ioMergedFacets.add(directBaseFacet);
        }
        else
        {
            // This facet is seemingly *not* a facet that represents one of the direct
            // bases for the type/declaration being processed.
            //
            // As such, we need to allocate a fresh facet to represent it in the
            // linearization we are creating, since the `foundFacet` already belongs
            // to the linearization of one of the bases, and shouldn't be repurposed.
            //
            auto indirectFacet = new (arena) Facet::Impl();

            // We will initialize the fresh facet to a copy of the state of the
            // `foundFacet`, albeit with a higher level of indirection.
            //
            // TODO: In principle we could search through  all of the lists to
            // find the one with a facet matching `foundFacet` with minimum
            // indirection, so that our measure of indirection is always
            // as small as possible for any given facet.
            //
            *indirectFacet = *(foundFacet.getImpl());
            indirectFacet->next = nullptr;
            indirectFacet->directness =
                Facet::Directness(Facet::DirectnessVal(indirectFacet->directness) + 1);

            // When using this facet for subtype tests, or when looking
            // up member through this facet, we will need a witness
            // to show that the self type of the declaration being
            // linearized (the type being declared or extended) is a
            // subtype of the type for this facet.
            //
            // We can construct the appropriate witness transitively,
            // by noting that:
            //
            // * The self type is known to be a subtype of the direct
            //   base represented by `foundBase`, and the facet for
            //   that base stores a witness to that fact.
            //
            SubtypeWitness* selfIsSubtypeOfBase = foundBase->facetImpl.subtypeWitness;
            //
            // * The direct base type must be a subtype of the type
            //   for any facet found in its own linearization, and
            //   the `foundFacet` that came from the relevant base
            //   stores a witness to that fact.
            //
            SubtypeWitness* baseIsSubtypeOfFacet = foundFacet->subtypeWitness;

            auto selfIsSubtypeOfFacet = _getASTBuilder()->getTransitiveSubtypeWitness(
                selfIsSubtypeOfBase,
                baseIsSubtypeOfFacet);

            indirectFacet->subtypeWitness = selfIsSubtypeOfFacet;

            ioMergedFacets.add(indirectFacet);
        }

        // We picked one `foundFacet` above to be added to the merged
        // output list, and we now need to ensure that we won't ever
        // emit a matching facet again.
        //
        // In the case of the strict/standard C3 algorithm, any facets
        // matching `foundFacet` would need to appear at a head position
        // in one of the base lists. As such, it is sufficient to run
        // through the base lists, check for a match at the head of each,
        // and remove any matching facets we find.
        //
        for (auto base : bases)
        {
            if (originsMatch(foundFacet, base->facets.getHead()))
            {
                base->facets.advanceHead();
                continue;
            }
        }
        //
        // Because we are not implementing the C3 algorithm strictly,
        // we need a solution for the case where `foundFacet` is
        // in a non-head position in one or more of the base lists.
        //
        // Proactively filtering `foundFacet` out of all of the lists
        // is possible, but given that these are singly-linked lists
        // we cannot easily filter them without either allocation
        // or mutation.
        //
        // Instead, we will filter out facets that have already been
        // added to the merged list as needed, when such facets come
        // to the front of the relevant list.
        //
        for (auto base : bases)
        {
            for (;;)
            {
                // For each base list, we will check if its
                // head facet is one that has already been
                // emitted to the output.
                //
                // If the head facet has not been emitted
                // already, we don't need to perform any
                // filtering on the base list at this time.
                //
                auto head = base->facets.getHead();
                if (!ioMergedFacets.containsMatchFor(head))
                    break;

                // Otherwise, we remove the head facet from
                // the given base list and test again, unless
                // the list is now empty.
                //
                base->facets.advanceHead();
                if (base->facets.isEmpty())
                    break;
            }
        }

        // The filtering step might have led to one or more
        // of the `bases` lists becomming empty. Our merge
        // algorithm really only wants to consider non-empty
        // lists, so we go ahead and remove the empty lists
        // here.
        //
        bases.removeEmptyLists();

        // At this point all of the lists have been appropriately filtered,
        // and we are ready to circle back around again to the step
        // where select a facet to add to the merged list.
    }

    // At this point, all of the input lists in `bases` should be empty,
    // and all of the facets in those lists should have found their way
    // over to `ioMergedFacets`.
}

// The mering algorithm needs to be able to test if two potentially-distinct
// `Facet`s represent the same underlying type or declaration.
//
bool originsMatch(Facet left, Facet right)
{
    if (left.getImpl() == right.getImpl())
        return true;
    if (!left.getImpl() || !right.getImpl())
        return false;

    // If both of the facets are non-null, and not
    // identical, we check if their origins match,
    // meaning that they represent the same type
    // or declaration.
    //
    return left->origin == right->origin;
}

bool operator==(Facet::Origin left, Facet::Origin right)
{
    // If either facet represents a declaration, then
    // the origins only match if they both represent
    // the *same* declaration.
    //
    if (left.declRef.getDecl() || right.declRef.getDecl())
    {
        return left.declRef.getDecl() && right.declRef.getDecl() &&
               left.declRef.equals(right.declRef);
    }

    // Otherwise, if they both represent types, then the
    // origins match if they are the same type.
    //
    // Note: an `extension` facet will always have a non-null
    // `declRef`, so there is no risk here of an `extension`
    // and a type facet being matched by this step; they
    // would always land in the case above.
    //
    if (left.type || right.type)
    {
        return left.type && right.type && left.type->equals(right.type);
    }

    // TODO: The rules we are using for matching here
    // would need to be revisited and overhauled significantly
    // if we start supporting generic type declarations
    // with covariant/contravariant type parameters.
    //
    // In such cases we would need to treat two facets as
    // matching if their declarations or types are an exact
    // matching modulo type arguments, and the relationship
    // between pairwise type arguments is consistent with
    // the variance of the corresponding parameter.
    //
    // E.g., we would need to treat facets for `IEnumerable<Derived>`
    // and `IEnumerable<Base>` as matching, and ensure that a
    // merged output list for a type/declaration could only
    // include the more specific of the two (`IEnumerable<Derived>`).

    return false;
}

// The remaining list-related operations that relate to the merging
// process are relatively simple to follow once the definition of
// matching is clear.

bool SharedSemanticsContext::DirectBaseList::doesAnyTailContainMatchFor(Facet facet) const
{
    for (auto base : *this)
    {
        if (base->facets.isEmpty())
            continue;
        if (base->facets.getTail().containsMatchFor(facet))
            return true;
    }
    return false;
}

void SharedSemanticsContext::DirectBaseList::removeEmptyLists()
{
    DirectBaseInfo** link = &_head;
    while (auto base = *link)
    {
        if (base->facets.isEmpty())
        {
            *link = base->next;
        }
        else
        {
            link = &base->next;
        }
    }
}

bool FacetList::containsMatchFor(Facet facet) const
{
    for (auto f : *this)
    {
        if (originsMatch(f, facet))
            return true;
    }
    return false;
}

InheritanceInfo SharedSemanticsContext::_calcInheritanceInfo(
    Type* type,
    InheritanceCircularityInfo* circularityInfo)
{
    // The majority of the interesting for for computing linearized
    // inheritance information arises for `DeclRef`s, but we still
    // need a way to compute the relevant information for types
    // that might or might not be defined using `Decl`s.

    auto astBuilder = _getASTBuilder();
    auto& arena = astBuilder->getArena();
    if (auto declRefType = as<DeclRefType>(type))
    {
        // The `DeclRef` case is the easy one, since we can
        // bottleneck through the logic that gets shared between
        // type and `extension` declarations.
        //
        return _getInheritanceInfo(declRefType->getDeclRef(), declRefType, circularityInfo);
    }
    else if (auto extractExistentialType = as<ExtractExistentialType>(type))
    {
        return _getInheritanceInfo(
            extractExistentialType->getThisTypeDeclRef(),
            extractExistentialType,
            circularityInfo);
    }
    else if (auto conjunctionType = as<AndType>(type))
    {
        // In this case, we have a type of the form `L & R`,
        // such that it is a subtype of both `L` and `R`.
        //
        auto leftType = conjunctionType->getLeft();
        auto rightType = conjunctionType->getRight();

        // The linearized inheritance list for the conjunction
        // must include all the facets from the lists for `L`
        // and `R`, respectively.
        //
        auto leftInfo = getInheritanceInfo(leftType, circularityInfo);
        auto rightInfo = getInheritanceInfo(rightType, circularityInfo);

        // We have a case of subtype witness that can show that
        // `T : L` or `T : R` based on `T : L&R`. In this case,
        // though, the type `T` is actually `L&R` itself, so
        // we need to construct an identity witness for `L&R : L&R`
        // to give it something to start from.
        //
        auto selfIsSelf = astBuilder->getTypeEqualityWitness(conjunctionType);
        auto selfIsSubtypeOfLeft = _getASTBuilder()->getExtractFromConjunctionSubtypeWitness(
            type,
            leftType,
            selfIsSelf,
            0);
        auto selfIsSubtypeOfRight = _getASTBuilder()->getExtractFromConjunctionSubtypeWitness(
            type,
            rightType,
            selfIsSelf,
            1);

        // We will set up to perform a merge between the facet
        // lists for the two "bases" `L` and `R`. Note that  the
        // information we write into the `facetImpl` in each case
        // is largely just for completeness and debugging, since
        // we are *not* going to add those facets into a list
        // of direct base facets to be merged.
        //
        DirectBaseInfo leftBaseInfo;
        leftBaseInfo.facetImpl = FacetImpl(
            Facet::Kind::Type,
            Facet::Directness::Direct,
            DeclRef<Decl>(),
            leftType,
            selfIsSubtypeOfLeft);
        leftBaseInfo.facets = leftInfo.facets;

        DirectBaseInfo rightBaseInfo;
        rightBaseInfo.facetImpl = FacetImpl(
            Facet::Kind::Type,
            Facet::Directness::Direct,
            DeclRef<Decl>(),
            rightType,
            selfIsSubtypeOfRight);
        rightBaseInfo.facets = rightInfo.facets;

        DirectBaseList::Builder directBases;
        directBases.add(&leftBaseInfo);
        directBases.add(&rightBaseInfo);

        // The merging step is then the same as for the more "standard" case,
        // with the only detail that we are not passing in a list of facets
        // to represent the directly-declared bases (since there are none;
        // this is a structural rather than nominal type).
        //
        FacetList::Builder mergedFacets;
        _mergeFacetLists(directBases, FacetList(), mergedFacets);

        InheritanceInfo info;
        info.facets = mergedFacets;
        return info;
    }
    else if (auto eachType = as<EachType>(type))
    {
        auto elementInheritanceInfo =
            getInheritanceInfo(eachType->getElementType(), circularityInfo);
        SemanticsVisitor visitor(this);
        auto directFacet = new (arena) Facet::Impl(
            Facet::Kind::Type,
            Facet::Directness::Self,
            DeclRef<Decl>(),
            type,
            visitor.createTypeEqualityWitness(type));
        Facet tail = directFacet;
        for (auto facet : elementInheritanceInfo.facets)
        {
            if (facet->directness == Facet::Directness::Direct)
            {
                auto eachFacet = new (arena) Facet::Impl(
                    Facet::Kind::Type,
                    Facet::Directness::Direct,
                    facet->origin.declRef,
                    facet->origin.type,
                    astBuilder->getEachSubtypeWitness(
                        type,
                        facet->subtypeWitness->getSup(),
                        facet->subtypeWitness));
                tail->next = eachFacet;
                tail = eachFacet;
            }
        }
        InheritanceInfo info;
        info.facets = FacetList(directFacet);
        return info;
    }
    else if (auto modifiedType = as<ModifiedType>(type))
    {
        return _calcInheritanceInfo(modifiedType->getBase(), circularityInfo);
    }
    else
    {
        // As a fallback, any type not covered by the above cases will
        // get a trivial linearization that consists of a single facet
        // corresponding to that type itself.
        //
        SemanticsVisitor visitor(this);
        auto directFacet = new (arena) Facet::Impl(
            Facet::Kind::Type,
            Facet::Directness::Self,
            DeclRef<Decl>(),
            type,
            visitor.createTypeEqualityWitness(type));

        InheritanceInfo info;
        info.facets = FacetList(directFacet);
        return info;
    }
}
} // namespace Slang
