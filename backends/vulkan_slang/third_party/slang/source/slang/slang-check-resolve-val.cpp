// slang-check-resolve-val.cpp

// Logic for resolving/simplifying Types and DeclRefs.

#include "slang-ast-dispatch.h"
#include "slang-ast-forward-declarations.h"
#include "slang-ast-synthesis.h"
#include "slang-check-impl.h"
#include "slang-lookup.h"
#include "slang-syntax.h"

namespace Slang
{

Type* Type::createCanonicalType()
{
    SLANG_AST_NODE_VIRTUAL_CALL(Type, createCanonicalType, ());
}

Val* Type::_resolveImplOverride()
{
    Val* resolvedVal = createCanonicalType();
    return resolvedVal;
}

DeclRefBase* _resolveAsDeclRef(DeclRefBase* declRefToResolve);

Type* DeclRefType::_createCanonicalTypeOverride()
{
    auto astBuilder = getCurrentASTBuilder();

    // A declaration reference is already canonical
    auto resolvedDeclRef = getDeclRef();
    resolvedDeclRef = _resolveAsDeclRef(getDeclRef().declRefBase);
    if (auto satisfyingVal =
            _tryLookupConcreteAssociatedTypeFromThisTypeSubst(astBuilder, resolvedDeclRef))
        return as<Type>(satisfyingVal);
    if (resolvedDeclRef != getDeclRef())
        return DeclRefType::create(astBuilder, resolvedDeclRef);
    return this;
}


Val* SubtypeWitness::_resolveImplOverride()
{
    return as<SubtypeWitness>(defaultResolveImpl());
}

ConversionCost SubtypeWitness::_getOverloadResolutionCostOverride()
{
    return kConversionCost_None;
}

ConversionCost SubtypeWitness::getOverloadResolutionCost()
{
    SLANG_AST_NODE_VIRTUAL_CALL(SubtypeWitness, getOverloadResolutionCost, ());
}

} // namespace Slang
