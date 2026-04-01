#ifndef SLANG_LOOKUP_H_INCLUDED
#define SLANG_LOOKUP_H_INCLUDED

#include "slang-syntax.h"

namespace Slang
{

struct SemanticsVisitor;

// Take an existing lookup result and refine it to only include
// results that pass the given `LookupMask`.
LookupResult refineLookup(LookupResult const& inResult, LookupMask mask);

// Look up a name in the given scope, proceeding up through
// parent scopes as needed.
LookupResult lookUp(
    ASTBuilder* astBuilder,
    SemanticsVisitor* semantics,
    Name* name,
    Scope* scope,
    LookupMask mask = LookupMask::Default,
    bool considerAllLocalNamesInScope = false,
    Decl* declToExclude = nullptr,
    bool ignoreTransparentMembers = false);

// Perform member lookup in the context of a type
LookupResult lookUpMember(
    ASTBuilder* astBuilder,
    SemanticsVisitor* semantics,
    Name* name,
    Type* type,
    Scope* sourceScope,
    LookupMask mask = LookupMask::Default,
    LookupOptions options = LookupOptions::None);

/// Perform "direct" lookup in a container declaration
LookupResult lookUpDirectAndTransparentMembers(
    ASTBuilder* astBuilder,
    SemanticsVisitor* semantics,
    Name* name,
    ContainerDecl* containerDecl,
    DeclRef<Decl> parentDeclRef, // The parent of the resulting declref.
    LookupMask mask = LookupMask::Default,
    Decl* declToExclude = nullptr);

// TODO: this belongs somewhere else

QualType getTypeForDeclRef(
    ASTBuilder* astBuilder,
    SemanticsVisitor* sema,
    DiagnosticSink* sink,
    DeclRef<Decl> declRef,
    Type** outTypeResult,
    SourceLoc loc);

QualType getTypeForDeclRef(ASTBuilder* astBuilder, DeclRef<Decl> declRef, SourceLoc loc);

/// Add a found item to a lookup result
void AddToLookupResult(LookupResult& result, LookupResultItem item);
void AddToLookupResult(LookupResult& result, const LookupResult& items);
} // namespace Slang

#endif
