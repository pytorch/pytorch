#pragma once

#include "../core/slang-basic.h"
#include "slang-ast-all.h"
#include "slang-compiler.h"
#include "slang-syntax.h"
#include "slang-workspace-version.h"
#include "slang.h"

namespace Slang
{
enum class SemanticTokenType
{
    Type,
    EnumMember,
    Variable,
    Parameter,
    Function,
    Property,
    Namespace,
    Keyword,
    Macro,
    String,
    NormalText
};
extern const char* kSemanticTokenTypes[(int)SemanticTokenType::NormalText];

struct SemanticToken
{
    int line;
    int col;
    int length;
    SemanticTokenType type;
    bool operator<(const SemanticToken& other) const
    {
        if (line < other.line)
            return true;
        if (line == other.line)
            return col < other.col;
        return false;
    }
};
List<SemanticToken> getSemanticTokens(
    Linkage* linkage,
    Module* module,
    UnownedStringSlice fileName,
    DocumentVersion* doc);
List<uint32_t> getEncodedTokens(List<SemanticToken>& tokens);

} // namespace Slang
