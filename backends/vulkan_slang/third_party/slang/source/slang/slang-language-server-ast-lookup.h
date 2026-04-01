#pragma once

#include "slang-ast-all.h"
#include "slang-workspace-version.h"

namespace Slang
{
struct ASTLookupResult
{
    List<SyntaxNode*> path;
};
enum class ASTLookupType
{
    Decl,
    Invoke,
};

struct Loc
{
    Int line;
    Int col;
    bool operator<(const Loc& other)
    {
        return line < other.line || line == other.line && col < other.col;
    }
    bool operator<=(const Loc& other)
    {
        return line < other.line || line == other.line && col <= other.col;
    }
    static Loc fromSourceLoc(SourceManager* manager, SourceLoc loc, String* outFileName = nullptr);
};
List<ASTLookupResult> findASTNodesAt(
    DocumentVersion* doc,
    SourceManager* sourceManager,
    ModuleDecl* moduleDecl,
    ASTLookupType findType,
    UnownedStringSlice fileName,
    Int line,
    Int col);

} // namespace Slang
