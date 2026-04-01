// slang-serialize-ast.h
#ifndef SLANG_SERIALIZE_AST_H
#define SLANG_SERIALIZE_AST_H

#include "../core/slang-riff.h"
#include "slang-ast-all.h"
#include "slang-ast-builder.h"
#include "slang-ast-support-types.h"
#include "slang-serialize-source-loc.h"
#include "slang-serialize.h"

namespace Slang
{
void writeSerializedModuleAST(
    Encoder* encoder,
    ModuleDecl* moduleDecl,
    SerialSourceLocWriter* sourceLocWriter);

ModuleDecl* readSerializedModuleAST(
    Linkage* linkage,
    ASTBuilder* astBuilder,
    DiagnosticSink* sink,
    RiffContainer::Chunk* chunk,
    SerialSourceLocReader* sourceLocReader,
    SourceLoc requestingSourceLoc);

} // namespace Slang

#endif
