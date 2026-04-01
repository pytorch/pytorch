#pragma once

#include "../core/slang-platform.h"
#include "slang-downstream-compiler-util.h"

namespace Slang
{

struct TintDownstreamCompilerUtil
{
    static SlangResult locateCompilers(
        const String& path,
        ISlangSharedLibraryLoader* loader,
        DownstreamCompilerSet* set);
};

} // namespace Slang
