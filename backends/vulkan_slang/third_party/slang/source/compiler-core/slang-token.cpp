// slang-token.cpp
#include "slang-token.h"

// #include <assert.h>

namespace Slang
{

char const* TokenTypeToString(TokenType type)
{
    switch (type)
    {
    default:
        SLANG_ASSERT(!"unexpected");
        return "<uknown>";

#define TOKEN(NAME, DESC) \
    case TokenType::NAME: \
        return DESC;
#include "slang-token-defs.h"
    }
}

} // namespace Slang
