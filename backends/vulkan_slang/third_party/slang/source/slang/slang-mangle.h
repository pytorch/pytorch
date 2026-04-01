#ifndef SLANG_MANGLE_H_INCLUDED
#define SLANG_MANGLE_H_INCLUDED

// This file implements the name mangling scheme for the Slang language.

#include "../core/slang-basic.h"
#include "slang-syntax.h"

namespace Slang
{
struct IRSpecialize;

void emitNameForLinkage(StringBuilder& sb, UnownedStringSlice str);

String getMangledName(ASTBuilder* astBuilder, Decl* decl);
String getMangledName(ASTBuilder* astBuilder, DeclRefBase* declRef);
String getMangledNameFromNameString(const UnownedStringSlice& name);

String getHashedName(const UnownedStringSlice& mangledName);

String getMangledNameForConformanceWitness(ASTBuilder* astBuilder, Type* sub, Type* sup);
String getMangledNameForConformanceWitness(
    ASTBuilder* astBuilder,
    Type* sub,
    Type* sup,
    IROp subOp);
String getMangledNameForConformanceWitness(
    ASTBuilder* astBuilder,
    DeclRef<Decl> sub,
    DeclRef<Decl> sup);
String getMangledNameForConformanceWitness(ASTBuilder* astBuilder, DeclRef<Decl> sub, Type* sup);
String getMangledTypeName(ASTBuilder* astBuilder, Type* type);
} // namespace Slang

#endif
