// slang-ast-dump.h
#ifndef SLANG_AST_DUMP_H
#define SLANG_AST_DUMP_H

#include "slang-emit-source-writer.h"
#include "slang-syntax.h"

namespace Slang
{

struct ASTDumpAccess;

struct ASTDumpUtil
{
    enum class Style
    {
        Hierachical,
        Flat,
    };

    typedef uint32_t Flags;
    struct Flag
    {
        enum Enum : Flags
        {
            HideSourceLoc = 0x1,
            HideScope = 0x2,
        };
    };

    static void dump(NodeBase* node, Style style, Flags flags, SourceWriter* writer);
};

} // namespace Slang

#endif
