// slang-ast-forward-declarations.h
#pragma once

namespace Slang
{

enum class ASTNodeType
{
#if 0 // FIDDLE TEMPLATE:
%for _, T in ipairs(Slang.NodeBase.subclasses) do
        $T,
%end
#else // FIDDLE OUTPUT:
#define FIDDLE_GENERATED_OUTPUT_ID 0
#include "slang-ast-forward-declarations.h.fiddle"
#endif // FIDDLE END
    CountOf
};

#if 0 // FIDDLE TEMPLATE:
%for _, T in ipairs(Slang.NodeBase.subclasses) do
    class $T;
%end
#else // FIDDLE OUTPUT:
#define FIDDLE_GENERATED_OUTPUT_ID 1
#include "slang-ast-forward-declarations.h.fiddle"
#endif // FIDDLE END

} // namespace Slang
