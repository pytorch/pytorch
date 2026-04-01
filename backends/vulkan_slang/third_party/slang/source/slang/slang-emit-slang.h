#ifndef SLANG_EMIT_SLANG_H
#define SLANG_EMIT_SLANG_H

#include "slang-emit-base.h"
#include "slang-ir-link.h"
#include "slang-vm-bytecode.h"

namespace Slang
{
SlangResult emitSlangDeclarationsForEntryPoints(
    CodeGenContext* codeGenContext,
    LinkedIR& linkedIR,
    String& outSlangDeclaration);
}

#endif
