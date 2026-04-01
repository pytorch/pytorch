#ifndef SLANG_LLVM_COMPILER_UTIL_H
#define SLANG_LLVM_COMPILER_UTIL_H

#include "../core/slang-platform.h"
#include "slang-downstream-compiler-util.h"

namespace Slang
{

struct LLVMDownstreamCompilerUtil
{
    static SlangResult locateCompilers(
        const String& path,
        ISlangSharedLibraryLoader* loader,
        DownstreamCompilerSet* set);
};

} // namespace Slang

#endif
