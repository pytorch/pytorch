// slang-lookup.cpp
#include "slang-lookup.h"

#include "../compiler-core/slang-name.h"
#include "slang-check-impl.h"

// TODO(tfoley): The implementation of lookup still involves
// recursion over the structure of a type/declaration, but
// it should be possible for it to use the flattened/linearized
// inheritance information that is being computed in
// `slang-check-inheritance.cpp`.

namespace Slang
{

void ensureDecl(SemanticsVisitor* visitor, Decl* decl, DeclCheckState state);

//

DeclRef<ExtensionDecl> applyExtensionToType(
    SemanticsVisitor* semantics,
    ExtensionDecl* extDecl,
    Type* type);

//


// Helper for constructing breadcrumb trails during lookup, without unnecessary heap allocaiton
struct BreadcrumbInfo
{
    LookupResultItem::Breadcrumb::Kind kind;
    LookupResultItem::Breadcrumb::ThisParameterMode thisParameterMode =
        LookupResultItem::Breadcrumb::ThisParameterMode::Default;
    DeclRef<Decl> declRef;
    Val* val = nullptr;
    BreadcrumbInfo* prev = nullptr;
};

//

bool DeclPassesLookupMask(Decl* decl, LookupMask mask)
{
    // Always exclude extern members from lookup result.
    if (decl->hasModifier<ExtensionExternVarModifier>())
    {
        return false;
    }
    else if (decl->hasModifier<ExternModifier>())
    {
        if (as<ExtensionDecl>(decl->parentDecl))
        {
            return false;
        }
    }
    // type declarations
    if (const auto aggTypeDecl = as<AggTypeDecl>(decl))
    {
        return int(mask) & int(LookupMask::type);
    }
    else if (const auto simpleTypeDecl = as<SimpleTypeDecl>(decl))
    {
        return int(mask) & int(LookupMask::type);
    }
    // function declarations
    else if (const auto funcDecl = as<FunctionDeclBase>(decl))
    {
        return (int(mask) & int(LookupMask::Function)) != 0;
    }
    // attribute declaration
    else if (const auto attrDecl = as<AttributeDecl>(decl))
    {
        return (int(mask) & int(LookupMask::Attribute)) != 0;
    }
    // syntax declaration
    else if (const auto syntaxDecl = as<SyntaxDecl>(decl))
    {
        return (int(mask) & int(LookupMask::SyntaxDecl)) != 0;
    }
    // default behavior is to assume a value declaration
    // (no overloading allowed)

    return (int(mask) & int(LookupMask::Value)) != 0;
}

void AddToLookupResult(LookupResult& result, LookupResultItem item)
{
    if (!result.isValid())
    {
        // If we hadn't found a hit before, we have one now
        result.item = item;
    }
    else if (!result.isOverloaded())
    {
        // We are about to make this overloaded
        result.items.add(result.item);
        result.items.add(item);
    }
    else
    {
        // The result was already overloaded, so we pile on
        result.items.add(item);
    }
}

void AddToLookupResult(LookupResult& result, const LookupResult& items)
{
    if (items.isOverloaded())
    {
        for (auto item : items.items)
            AddToLookupResult(result, item);
    }
    else if (items.isValid())
    {
        AddToLookupResult(result, items.item);
    }
}

LookupResult refineLookup(LookupResult const& inResult, LookupMask mask)
{
    if (!inResult.isValid())
        return inResult;
    if (!inResult.isOverloaded())
        return inResult;

    LookupResult result;
    for (auto item : inResult.items)
    {
        if (!DeclPassesLookupMask(item.declRef.getDecl(), mask))
            continue;

        AddToLookupResult(result, item);
    }
    return result;
}

LookupResultItem CreateLookupResultItem(DeclRef<Decl> declRef, BreadcrumbInfo* breadcrumbInfos)
{
    LookupResultItem item;
    item.declRef = declRef;

    // breadcrumbs were constructed "backwards" on the stack, so we
    // reverse them here by building a linked list the other way
    RefPtr<LookupResultItem::Breadcrumb> breadcrumbs;
    for (auto bb = breadcrumbInfos; bb; bb = bb->prev)
    {
        breadcrumbs = new LookupResultItem::Breadcrumb(
            bb->kind,
            bb->declRef,
            bb->val,
            breadcrumbs,
            bb->thisParameterMode);
    }
    item.breadcrumbs = breadcrumbs;
    return item;
}

static void _lookUpMembersInValue(
    ASTBuilder* astBuilder,
    Name* name,
    DeclRef<Decl> valueDeclRef,
    LookupRequest const& request,
    LookupResult& ioResult,
    BreadcrumbInfo* breadcrumbs);

static bool _isUncheckedLocalVar(const Decl* decl)
{
    auto checkStateExt = decl->checkState;
    auto isUnchecked = checkStateExt.getState() == DeclCheckState::Unchecked ||
                       checkStateExt.isBeingChecked() || decl->hiddenFromLookup;
    return isUnchecked && isLocalVar(decl);
}

/// Look up direct members (those declared in `containerDeclRef` itself, as well
/// as transitively through any direct members that are marked "transparent."
///
/// This function does *not* deal with looking up through `extension`s,
/// inheritance clauses, etc.
///
static void _lookUpDirectAndTransparentMembers(
    ASTBuilder* astBuilder,
    Name* name,
    ContainerDecl* containerDecl, // The container decl to find member with `name`.
    DeclRef<Decl> parentDeclRef,  // The parent of the resulting declref.
    LookupRequest const& request,
    LookupResult& result,
    BreadcrumbInfo* inBreadcrumbs)
{
    if (request.isCompletionRequest())
    {
        // If we are looking up for completion suggestions,
        // return all the members that are available.
        for (auto member : containerDecl->members)
        {
            if (!request.shouldConsiderAllLocalNames() && _isUncheckedLocalVar(member))
                continue;
            if (!DeclPassesLookupMask(member, request.mask))
                continue;
            AddToLookupResult(
                result,
                CreateLookupResultItem(
                    astBuilder->getMemberDeclRef<Decl>(parentDeclRef, member),
                    inBreadcrumbs));
        }
    }
    else
    {
        // Look up the declarations with the chosen name in the container.
        Decl* firstDecl = nullptr;
        containerDecl->getMemberDictionary().tryGetValue(name, firstDecl);

        // Now iterate over those declarations (if any) and see if
        // we find any that meet our filtering criteria.
        // For example, we might be filtering so that we only consider
        // type declarations.
        for (auto m = firstDecl; m; m = m->nextInContainerWithSameName)
        {
            // Skip this declaration if we are checking and this hasn't been
            // checked yet. Because we traverse block statements in order, if
            // it's unchecked or being checked then it isn't declared yet.
            if (!request.shouldConsiderAllLocalNames() && request.semantics &&
                _isUncheckedLocalVar(m))
                continue;
            if (m == request.declToExclude)
                continue;

            if (!DeclPassesLookupMask(m, request.mask))
                continue;

            // The declaration passed the test, so add it!
            AddToLookupResult(
                result,
                CreateLookupResultItem(
                    astBuilder->getMemberDeclRef<Decl>(parentDeclRef, m),
                    inBreadcrumbs));
        }
    }

    // Don't look up transparent members if we are looking for attributes, since
    // they are always defined at global scope in the core module. Trying to lookup transparent
    // members during attribute lookup can lead to infinite recursion on transparent types.
    if ((int)request.mask & (int)LookupMask::Attribute)
        return;

    // Also skip transparent members if they're explicitly excluded by the
    // request. This prevents cyclic lookups e.g. when looking up UnscopedEnum's
    // underlying types.
    if (((int)request.options & (int)LookupOptions::IgnoreTransparentMembers) != 0)
        return;

    for (auto transparentInfo : containerDecl->getTransparentMembers())
    {
        // The reference to the transparent member should use the same
        // path as we used in referring to its parent.
        DeclRef<Decl> transparentMemberDeclRef =
            astBuilder->getMemberDeclRef(parentDeclRef, transparentInfo.decl);
        if (transparentMemberDeclRef.getDecl() == request.declToExclude)
            continue;

        // We need to leave a breadcrumb so that we know that the result
        // of lookup involves a member lookup step here

        BreadcrumbInfo memberRefBreadcrumb;
        memberRefBreadcrumb.kind = LookupResultItem::Breadcrumb::Kind::Member;
        memberRefBreadcrumb.declRef = transparentMemberDeclRef;
        memberRefBreadcrumb.prev = inBreadcrumbs;

        _lookUpMembersInValue(
            astBuilder,
            name,
            transparentMemberDeclRef,
            request,
            result,
            &memberRefBreadcrumb);
    }
}

LookupRequest initLookupRequest(
    SemanticsVisitor* semantics,
    Name* name,
    LookupMask mask,
    LookupOptions options,
    Scope* scope,
    Decl* declToExclude)
{
    LookupRequest request;
    request.semantics = semantics;
    request.mask = mask;
    request.options = options;
    request.scope = scope;
    request.declToExclude = declToExclude;

    if (semantics && semantics->getSession() &&
        name == semantics->getSession()->getCompletionRequestTokenName())
        request.options = (LookupOptions)((int)request.options | (int)LookupOptions::Completion);

    return request;
}

/// Perform "direct" lookup in a container declaration
LookupResult lookUpDirectAndTransparentMembers(
    ASTBuilder* astBuilder,
    SemanticsVisitor* semantics,
    Name* name,
    ContainerDecl* containerDecl,
    DeclRef<Decl> parentDeclRef,
    LookupMask mask,
    Decl* declToExclude)
{
    LookupRequest request =
        initLookupRequest(semantics, name, mask, LookupOptions::None, nullptr, declToExclude);
    LookupResult result;
    _lookUpDirectAndTransparentMembers(
        astBuilder,
        name,
        containerDecl,
        parentDeclRef,
        request,
        result,
        nullptr);
    return result;
}

// Specialize `declRefToSpecialize` with ThisType info if `superType` is an interface type.
DeclRef<Decl> _maybeSpecializeSuperTypeDeclRef(
    ASTBuilder* astBuilder,
    DeclRef<Decl> declRefToSpecialize,
    Type* superType,
    SubtypeWitness* subIsSuperWitness)
{
    if (auto superDeclRefType = as<DeclRefType>(superType))
    {
        if (auto superInterfaceDeclRef = superDeclRefType->getDeclRef().as<InterfaceDecl>())
        {
            ThisTypeDecl* thisTypeDecl = superInterfaceDeclRef.getDecl()->getThisTypeDecl();
            auto specializedDeclRef = astBuilder->getLookupDeclRef(subIsSuperWitness, thisTypeDecl);

            return specializedDeclRef;
        }
    }
    return declRefToSpecialize;
}

// Same as the above, but we are specializing a type instead of a decl-ref
static Type* _maybeSpecializeSuperType(
    ASTBuilder* astBuilder,
    Type* superType,
    SubtypeWitness* subIsSuperWitness)
{
    if (auto superDeclRefType = as<DeclRefType>(superType))
    {
        auto specializedDeclRef = _maybeSpecializeSuperTypeDeclRef(
            astBuilder,
            superDeclRefType->getDeclRef(),
            superType,
            subIsSuperWitness);
        return DeclRefType::create(astBuilder, specializedDeclRef);
    }

    return superType;
}

static void _lookUpMembersInType(
    ASTBuilder* astBuilder,
    Name* name,
    Type* type,
    LookupRequest const& request,
    LookupResult& ioResult,
    BreadcrumbInfo* breadcrumbs);

static void _lookUpMembersInSuperTypeImpl(
    ASTBuilder* astBuilder,
    Name* name,
    Type* leafType,
    Type* superType,
    SubtypeWitness* leafIsSuperWitness,
    LookupRequest const& request,
    LookupResult& ioResult,
    BreadcrumbInfo* inBreadcrumbs);

static void _lookUpMembersInSuperType(
    ASTBuilder* astBuilder,
    Name* name,
    Type* leafType,
    Type* superType,
    SubtypeWitness* leafIsSuperWitness,
    LookupRequest const& request,
    LookupResult& ioResult,
    BreadcrumbInfo* inBreadcrumbs)
{
    // If we are looking up through an interface type, then
    // we need to be sure that we add an appropriate
    // "this type" substitution here, since that needs to
    // be applied to any members we look up.
    //
    superType = _maybeSpecializeSuperType(astBuilder, superType, leafIsSuperWitness);

    // We need to track the indirection we took in lookup,
    // so that we can construct an appropriate AST on the other
    // side that includes the "upcast" from sub-type to super-type.
    //
    BreadcrumbInfo breadcrumb;
    breadcrumb.prev = inBreadcrumbs;
    breadcrumb.kind = LookupResultItem::Breadcrumb::Kind::SuperType;
    breadcrumb.val = leafIsSuperWitness;
    breadcrumb.prev = inBreadcrumbs;

    _lookUpMembersInSuperTypeImpl(
        astBuilder,
        name,
        leafType,
        superType,
        leafIsSuperWitness,
        request,
        ioResult,
        &breadcrumb);
}

static void _lookupMembersInSuperTypeFacets(
    ASTBuilder* astBuilder,
    Name* name,
    Type* selfType,
    InheritanceInfo const& inheritanceInfo,
    LookupRequest const& request,
    LookupResult& ioResult,
    BreadcrumbInfo* inBreadcrumbs)
{


    for (auto facet : inheritanceInfo.facets)
    {
        auto containerDeclRef = facet->getDeclRef().as<ContainerDecl>();
        if (!containerDeclRef)
            continue;

        // Check for cases where we should skip this facet for lookup.
        //
        // If the facet doesn't correspond to a type, we can't lookup.
        if (!facet->getType() || !facet->subtypeWitness)
        {
            continue;
        }


        // If we are looking up in an interface, and the lookup request told us
        // to skip interfaces, we should do so here.
        if (auto baseInterfaceDeclRef = containerDeclRef.as<InterfaceDecl>())
        {
            if (int(request.options) & int(LookupOptions::IgnoreBaseInterfaces))
                continue;
        }
        // If we are looking up only immediate members, ignore non "Self" facets or extension to
        // "Self"
        else if (
            int(request.options) & int(LookupOptions::IgnoreInheritance) &&
            (facet.getImpl()->directness != Facet::Directness::Self))
        {
            if (auto extensionDeclRef = facet.getImpl()->getDeclRef().as<ExtensionDecl>())
            {
                if (auto targetType = getTargetType(astBuilder, extensionDeclRef))
                {
                    if (!targetType->equals(selfType))
                    {
                        // If the extension is to the same type as the one we are looking up in, we
                        // should include it in the lookup.
                        continue;
                    }
                }
            }
            else
                continue;
        }

        // Some things that are syntactically `InheritanceDecl`s don't actually
        // represent a subtype/supertype relationship, and thus we shouldn't
        // include members from the base type when doing lookup in the
        // derived type.
        //
        // TODO: this check currently only works when the facet is a direct
        // basee type of the type we are looking up in. This is OK because the
        // only case where we use `IgnoreForLookupModifier` is for skipping the
        // underlying int type of an enum type. We should either makes this
        // check more general, or just explicitly detect this case here without
        // relying on the modifier.
        if (auto declaredSubtypeWitness = as<DeclaredSubtypeWitness>(facet->subtypeWitness))
        {
            auto inheritanceDeclRef = declaredSubtypeWitness->getDeclRef();
            if (inheritanceDeclRef.getDecl()->hasModifier<IgnoreForLookupModifier>())
                continue;
        }

        // We are now going to lookup in the facet.

        BreadcrumbInfo* newBreadcrumbs = inBreadcrumbs;
        BreadcrumbInfo subtypeInfo;
        auto parentDeclRef = containerDeclRef;
        if (facet->directness != Facet::Directness::Self)
        {
            // Depending on the type of the facet, we may want to specialize the
            // declRef that we are going to lookup in. If the facet represents
            // an extension, we should just lookup in the extension decl.
            //
            // If the facet is an extension to an interface type, we should
            // specialize the interface declRef to the concrete type that this
            // extension applied to.
            //
            // If the facet represents an implementation of interface type,
            // we should also specialize the interface declRef with the concrete
            // type info.
            //
            parentDeclRef = _maybeSpecializeSuperTypeDeclRef(
                                astBuilder,
                                containerDeclRef,
                                facet->getType(),
                                facet->subtypeWitness)
                                .as<ContainerDecl>();
            if (as<ThisTypeDecl>(parentDeclRef.getDecl()) && getText(name) == "This")
            {
                // If we are going looking for `This` in a `ThisType`, we just need to return the
                // declRef itself.
                AddToLookupResult(ioResult, CreateLookupResultItem(parentDeclRef, inBreadcrumbs));
                continue;
            }

            // If we are looking up in a base type, we also need to make sure
            // to create a breadcrumb to track the sub to super indirection.
            if (facet->kind == Facet::Kind::Type)
            {
                subtypeInfo.kind = LookupResultItem_Breadcrumb::Kind::SuperType;
                subtypeInfo.val = facet->subtypeWitness;
                subtypeInfo.prev = inBreadcrumbs;
                subtypeInfo.declRef = facet->getDeclRef();
                newBreadcrumbs = &subtypeInfo;
            }
        }
        _lookUpDirectAndTransparentMembers(
            astBuilder,
            name,
            containerDeclRef.getDecl(),
            parentDeclRef,
            request,
            ioResult,
            newBreadcrumbs);
    }
}

static void _lookUpMembersInSuperTypeDeclImpl(
    ASTBuilder* astBuilder,
    Name* name,
    DeclRef<Decl> declRef,
    LookupRequest const& request,
    LookupResult& ioResult,
    BreadcrumbInfo* inBreadcrumbs)
{
    auto semantics = request.semantics;
    if (!as<InterfaceDecl>(declRef.getDecl()) &&
        name == astBuilder->getSharedASTBuilder()->getThisTypeName())
    {
        // If we are looking for `This` in anything other than an InterfaceDecl,
        // we just need to return the declRef itself.
        AddToLookupResult(ioResult, CreateLookupResultItem(declRef, inBreadcrumbs));
        return;
    }

    // If the semantics context hasn't been established yet (e.g. when looking up during parsing),
    // we simply do a direct lookup without considering subtypes or extensions.
    //
    if (!semantics)
    {
        // In this case we can only lookup in an aggregate type.
        if (auto aggTypeDeclBaseRef = declRef.as<AggTypeDeclBase>())
        {
            _lookUpDirectAndTransparentMembers(
                astBuilder,
                name,
                aggTypeDeclBaseRef.getDecl(),
                aggTypeDeclBaseRef,
                request,
                ioResult,
                inBreadcrumbs);
        }
        return;
    }

    ensureDecl(semantics, declRef.getDecl(), DeclCheckState::ReadyForLookup);

    // With semantics context, we can do a comprehensive lookup by scanning through
    // the linearized inheritance list.

    auto selfType = DeclRefType::create(astBuilder, declRef);
    InheritanceInfo inheritanceInfo;
    if (auto extDeclRef = declRef.as<ExtensionDecl>())
    {
        inheritanceInfo = semantics->getShared()->getInheritanceInfo(extDeclRef);
    }
    else
    {
        selfType = selfType->getCanonicalType();
        inheritanceInfo = semantics->getShared()->getInheritanceInfo(selfType);
    }

    _lookupMembersInSuperTypeFacets(
        astBuilder,
        name,
        selfType,
        inheritanceInfo,
        request,
        ioResult,
        inBreadcrumbs);
}

static void _lookUpMembersInSuperTypeImpl(
    ASTBuilder* astBuilder,
    Name* name,
    Type* leafType,
    Type* superType,
    SubtypeWitness* leafIsSuperWitness,
    LookupRequest const& request,
    LookupResult& ioResult,
    BreadcrumbInfo* inBreadcrumbs)
{
    // If the type was pointer-like, then dereference it
    // automatically here.
    if (((uint32_t)request.options & (uint32_t)LookupOptions::NoDeref) == 0)
    {
        if (auto pointerElementType = getPointedToTypeIfCanImplicitDeref(superType))
        {
            // Need to leave a breadcrumb to indicate that we
            // did an implicit dereference here
            BreadcrumbInfo derefBreacrumb;
            derefBreacrumb.kind = LookupResultItem::Breadcrumb::Kind::Deref;
            derefBreacrumb.prev = inBreadcrumbs;

            // Recursively perform lookup on the result of deref
            _lookUpMembersInType(
                astBuilder,
                name,
                pointerElementType,
                request,
                ioResult,
                &derefBreacrumb);
            if (ioResult.isValid())
                return;
        }
    }

    // Default case: no dereference needed

    if (auto declRefType = as<DeclRefType>(superType))
    {
        auto declRef = declRefType->getDeclRef();

        _lookUpMembersInSuperTypeDeclImpl(
            astBuilder,
            name,
            declRef,
            request,
            ioResult,
            inBreadcrumbs);
    }
    else if (auto eachType = as<EachType>(superType))
    {
        auto canEachType = eachType->getCanonicalType();
        InheritanceInfo inheritanceInfo =
            request.semantics->getShared()->getInheritanceInfo(canEachType);
        _lookupMembersInSuperTypeFacets(
            astBuilder,
            name,
            canEachType,
            inheritanceInfo,
            request,
            ioResult,
            inBreadcrumbs);
    }
    else if (auto extractExistentialType = as<ExtractExistentialType>(superType))
    {
        // We want lookup to be performed on the underlying interface type of the existential,
        // but we need to have a this-type substitution applied to ensure that the result of
        // lookup will have a comparable substitution applied (allowing things like associated
        // types, etc. used in the signature of a method to resolve correctly).
        //
        auto thisTypeDeclRef = extractExistentialType->getThisTypeDeclRef();
        _lookUpMembersInSuperTypeDeclImpl(
            astBuilder,
            name,
            thisTypeDeclRef,
            request,
            ioResult,
            inBreadcrumbs);
    }
    else if (auto andType = as<AndType>(superType))
    {
        // We have a type of the form `leftType & rightType` and we need to perform
        // lookup in both `leftType` and `rightType`.
        //
        auto leftType = andType->getLeft();
        auto rightType = andType->getRight();

        // Operationally, we are in a situation where we have a witness
        // that the `leafType` we are doing lookup on is an subtype
        // of `superType` (which is `leftType & rightType`) and now we need
        // to construct a witness that `leafType` is a subtype of
        // the `Left` type.
        //
        // Effectively, we have a witness that `T : X & Y` and we
        // need to extract from it a witness that `T : X`.
        //
        //
        auto leafIsLeftWitness = astBuilder->getExtractFromConjunctionSubtypeWitness(
            leafType,
            leftType,
            leafIsSuperWitness,
            0);


        // The witness for the fact that `leafType : rightType` is the
        // same as for the left case, just with a different index into
        // the conjunction.
        //
        auto leafIsRightWitness = astBuilder->getExtractFromConjunctionSubtypeWitness(
            leafType,
            rightType,
            leafIsSuperWitness,
            1);

        // We then perform lookup on both sides of the conjunction, and
        // accumulate whatever items are found on either/both sides.
        //
        // For each recursive lookup, we pass the appropriate pair of
        // the type to look up in and the witness of the subtype
        // relationship.
        //
        _lookUpMembersInSuperType(
            astBuilder,
            name,
            leafType,
            leftType,
            leafIsLeftWitness,
            request,
            ioResult,
            inBreadcrumbs);
        _lookUpMembersInSuperType(
            astBuilder,
            name,
            leafType,
            rightType,
            leafIsRightWitness,
            request,
            ioResult,
            inBreadcrumbs);
    }
}

/// Perform lookup for `name` in the context of `type`.
///
/// This operation does the kind of lookup we'd expect if `name`
/// was used inside of a member function on `type`, or if the
/// user wrote `obj.<name>` for a variable `obj` of the given
/// `type`.
///
/// Looking up members in `type` includes lookup through any
/// constraints or inheritance relationships that expand the
/// set of members visible on `type`.
///
static void _lookUpMembersInType(
    ASTBuilder* astBuilder,
    Name* name,
    Type* type,
    LookupRequest const& request,
    LookupResult& ioResult,
    BreadcrumbInfo* breadcrumbs)
{
    if (!type)
    {
        return;
    }

    _lookUpMembersInSuperTypeImpl(
        astBuilder,
        name,
        type,
        type,
        nullptr,
        request,
        ioResult,
        breadcrumbs);
}

/// Look up members by `name` in the given `valueDeclRef`.
///
/// If `valueDeclRef` represents a reference to a variable
/// or other named and typed value, then this performs the
/// kind of lookup we'd expect for `valueDeclRef.<name>`.
///
static void _lookUpMembersInValue(
    ASTBuilder* astBuilder,
    Name* name,
    DeclRef<Decl> valueDeclRef,
    LookupRequest const& request,
    LookupResult& ioResult,
    BreadcrumbInfo* breadcrumbs)
{
    // Looking up `name` in the context of a value can
    // be reduced to the problem of looking up `name`
    // in the *type* of that value.
    //
    auto valueType = getTypeForDeclRef(astBuilder, valueDeclRef, SourceLoc());
    if (auto typeType = as<TypeType>(valueType))
        valueType = typeType->getType();
    return _lookUpMembersInType(astBuilder, name, valueType, request, ioResult, breadcrumbs);
}

// True if the declaration is of an overloadable variety
// (ie can have multiple definitions with the same name)
//
// For example functions are overloadable, but variables are (typically) not.
static bool _isDeclOverloadable(Decl* decl)
{
    // If it's a generic strip off, to get to inner decl type
    while (auto genericDecl = as<GenericDecl>(decl))
    {
        decl = genericDecl->inner;
    }

    // TODO(JS): Do we need to special case around ConstructorDecl? or AccessorDecl?
    // It seems not as they are both function-like and potentially overloadable

    // If it's callable, it's a function-like and so overloadable
    if (auto callableDecl = as<CallableDecl>(decl))
    {
        SLANG_UNUSED(callableDecl);
        return true;
    }

    return false;
}

static void _lookUpInScopes(
    ASTBuilder* astBuilder,
    Name* name,
    LookupRequest const& request,
    LookupResult& result)
{
    auto thisParameterMode = LookupResultItem::Breadcrumb::ThisParameterMode::Default;

    auto scope = request.scope;

    auto endScope = request.endScope;

    // The file decl that this scope is in.
    FileDecl* thisFileDecl = nullptr;

    for (; scope != endScope; scope = scope->parent)
    {
        // Note that we consider all "peer" scopes together,
        // so that a hit in one of them does not preclude
        // also finding a hit in another
        for (auto link = scope; link; link = link->nextSibling)
        {
            auto containerDecl = link->containerDecl;

            // It is possible for the first scope in a list of
            // siblings to be a "dummy" scope that only exists
            // to combine the siblings; in that case it will
            // have a null `containerDecl` and needs to be
            // skipped over.
            //
            if (!containerDecl)
                continue;

            if (auto fileDecl = as<FileDecl>(containerDecl))
            {
                if (!thisFileDecl)
                    thisFileDecl = fileDecl;
                else if (fileDecl == thisFileDecl)
                {
                    // If we have already looked up in this file decl,
                    // we don't want to do so again.
                    continue;
                }
            }

            // TODO: If we need default substitutions to be applied to
            // the `containerDecl`, then it might make sense to have
            // each `link` in the scope store a decl-ref instead of
            // just a decl.
            //
            DeclRef<ContainerDecl> containerDeclRef = createDefaultSubstitutionsIfNeeded(
                                                          astBuilder,
                                                          request.semantics,
                                                          makeDeclRef(containerDecl))
                                                          .as<ContainerDecl>();

            // If the container we are looking into represents a type
            // or an `extension` of a type, then we need to treat
            // this step as lookup into the `this` variable (or the
            // `This` type), which means including any `extension`s
            // or inheritance clauses in the lookup process.
            //
            // Note: The `AggTypeDeclBase` class is the common superclass
            // between `AggTypeDecl` and `ExtensionDecl`.
            //
            if (auto aggTypeDeclBaseRef = containerDeclRef.as<AggTypeDeclBase>())
            {
                // When reconstructing the final expression for a result
                // looked up through the type or extension, we will need
                // a `this` expression (or a `This` type expression) to
                // mark the base of the member reference, so we create
                // a "breadcrumb" here to track that fact.
                //
                BreadcrumbInfo breadcrumb;
                breadcrumb.kind = LookupResultItem::Breadcrumb::Kind::This;
                breadcrumb.thisParameterMode = thisParameterMode;
                breadcrumb.declRef = aggTypeDeclBaseRef;
                breadcrumb.prev = nullptr;
                BreadcrumbInfo* breadcrumbPtr = &breadcrumb;
                Type* type = nullptr;
                if (auto extDeclRef = aggTypeDeclBaseRef.as<ExtensionDecl>())
                {
                    if (request.semantics)
                    {
                        ensureDecl(
                            request.semantics,
                            extDeclRef.getDecl(),
                            DeclCheckState::CanUseExtensionTargetType);
                    }

                    // If we are doing lookup from inside an `extension`
                    // declaration, then the `this` expression will have
                    // a type that uses the "target type" of the `extension`.
                    //
                    type = getTargetType(astBuilder, extDeclRef);
                    if (name == astBuilder->getSharedASTBuilder()->getThisTypeName())
                    {
                        breadcrumbPtr = nullptr;
                    }
                }
                else
                {
                    assert(aggTypeDeclBaseRef.as<AggTypeDecl>());
                    if (auto interfaceBase = as<InterfaceDecl>(aggTypeDeclBaseRef.getDecl()))
                    {
                        // When looking up inside an interface type, we are actually looking up
                        // through ThisType.
                        if (name != interfaceBase->getThisTypeDecl()->getName())
                        {
                            type = DeclRefType::create(
                                astBuilder,
                                astBuilder->getMemberDeclRef(
                                    aggTypeDeclBaseRef,
                                    interfaceBase->getThisTypeDecl()));
                            // Don't need any breadcrumb for looking up through ThisType, since we
                            // have already created the base type reference in the new `type`'s
                            // declref.
                            breadcrumbPtr = nullptr;
                        }
                    }

                    if (!type)
                    {
                        type = DeclRefType::create(astBuilder, aggTypeDeclBaseRef);
                    }
                }

                _lookUpMembersInType(astBuilder, name, type, request, result, breadcrumbPtr);
            }
            else
            {
                // The default case is when the scope doesn't represent a
                // type or `extension` declaration, so we can look up members
                // in that scope much more simply.
                //
                _lookUpDirectAndTransparentMembers(
                    astBuilder,
                    name,
                    containerDeclRef.getDecl(),
                    containerDeclRef,
                    request,
                    result,
                    nullptr);
            }

            // Before we proceed up to the next outer scope to perform lookup
            // again, we need to consider what the current scope tells us
            // about how to interpret uses of implicit `this` or `This`. For
            // example, if we are inside a `[mutating]` method, then the implicit
            // `this` that we use for lookup should be an l-value.
            //
            // Similarly, if we look up a member in a type from the scope
            // of some nested type, then there shouldn't be an implicit `this`
            // expression for the outer type, but instead an implicit `This`.
            //
            if (containerDeclRef.is<ConstructorDecl>())
            {
                // In the context of an `__init` declaration, the members of
                // the surrounding type are accessible through a mutable `this`.
                //
                thisParameterMode = LookupResultItem::Breadcrumb::ThisParameterMode::MutableValue;
            }
            else if (containerDeclRef.is<SetterDecl>())
            {
                // In the context of a `set` accessor, the members of the
                // surrounding type are accessible through a mutable `this`.
                //
                // TODO: At some point we may want a way to opt out of this
                // behavior; it is possible to have a setter on a `struct`
                // that actually just sets data into a buffer that is
                // referenced by one of the `struct`'s fields.
                //
                thisParameterMode = LookupResultItem::Breadcrumb::ThisParameterMode::MutableValue;
            }
            else if (auto funcDeclRef = containerDeclRef.as<FunctionDeclBase>())
            {
                // The implicit `this`/`This` for a function-like declaration
                // depends on modifiers attached to the declaration.
                //
                if (isEffectivelyStatic(funcDeclRef.getDecl()))
                {
                    // A `static` method only has access to an implicit `This`,
                    // and does not have a `this` expression available.
                    //
                    thisParameterMode = LookupResultItem::Breadcrumb::ThisParameterMode::Type;
                }
                else if (funcDeclRef.getDecl()->hasModifier<MutatingAttribute>())
                {
                    // In a non-`static` method marked `[mutating]` there is
                    // an implicit `this` parameter that is mutable.
                    //
                    thisParameterMode =
                        LookupResultItem::Breadcrumb::ThisParameterMode::MutableValue;
                }
                else if (funcDeclRef.getDecl()->hasModifier<RefAttribute>())
                {
                    // In a non-`static` method marked `[ref]` there is
                    // an implicit `this` parameter that is mutable.
                    //
                    thisParameterMode =
                        LookupResultItem::Breadcrumb::ThisParameterMode::MutableValue;
                }
                else
                {
                    // In all other cases, there is an implicit `this` parameter
                    // that is immutable.
                    //
                    thisParameterMode =
                        LookupResultItem::Breadcrumb::ThisParameterMode::ImmutableValue;
                }
            }
            else if (containerDeclRef.as<AggTypeDeclBase>())
            {
                // When lookup moves from a nested typed declaration to an
                // outer scope, there is no ability to use an implicit `this`
                // expression, and we have only the `This` type available.
                //
                thisParameterMode = LookupResultItem::Breadcrumb::ThisParameterMode::Type;
            }
        }

        if (result.isValid())
        {
            // If it's overloaded or the decl we have is of an overloadable type, or if we are
            // looking up for completion suggestions then we just keep going
            if (result.isOverloaded() || _isDeclOverloadable(result.item.declRef.getDecl()) ||
                ((int32_t)request.options & (int32_t)LookupOptions::Completion) != 0)
            {
                continue;
            }

            // If we've found a result in this scope (and it's not overloadable), then there
            // is no reason to look further up (for now).
            break;
        }
    }

    // If we run out of scopes, then we are done.
}

LookupResult lookUp(
    ASTBuilder* astBuilder,
    SemanticsVisitor* semantics,
    Name* name,
    Scope* scope,
    LookupMask mask,
    bool considerAllLocalNamesInScope,
    Decl* declToExclude,
    bool ignoreTransparentMembers)
{
    LookupResult result;
    const auto options =
        (LookupOptions)((int)(considerAllLocalNamesInScope
                                  ? LookupOptions::ConsiderAllLocalNamesInScope
                                  : LookupOptions::None) |
                        (int)(ignoreTransparentMembers ? LookupOptions::IgnoreTransparentMembers
                                                       : LookupOptions::None));
    LookupRequest request = initLookupRequest(semantics, name, mask, options, scope, declToExclude);
    _lookUpInScopes(astBuilder, name, request, result);
    return result;
}

LookupResult lookUpMember(
    ASTBuilder* astBuilder,
    SemanticsVisitor* semantics,
    Name* name,
    Type* type,
    Scope* sourceScope,
    LookupMask mask,
    LookupOptions options)
{
    LookupResult result;
    LookupRequest request = initLookupRequest(semantics, name, mask, options, sourceScope, nullptr);
    _lookUpMembersInType(astBuilder, name, type, request, result, nullptr);
    return result;
}

} // namespace Slang
