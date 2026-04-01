#ifndef SLANG_REFLECTION_JSON_H
#define SLANG_REFLECTION_JSON_H

#include "../compiler-core/slang-pretty-writer.h"
#include "slang.h"

namespace Slang
{

void emitReflectionJSON(
    SlangCompileRequest* request,
    SlangReflection* reflection,
    PrettyWriter& writer);

}

#endif
