// slang-llvm-compiler.cpp
#include "slang-llvm-compiler.h"

#include "../core/slang-common.h"

namespace Slang
{

/* static */ SlangResult LLVMDownstreamCompilerUtil::locateCompilers(
    const String& path,
    ISlangSharedLibraryLoader* loader,
    DownstreamCompilerSet* set)
{
    ComPtr<ISlangSharedLibrary> library;

    SLANG_RETURN_ON_FAIL(
        DownstreamCompilerUtil::loadSharedLibrary(path, loader, nullptr, "slang-llvm", library));

    SLANG_ASSERT(library);
    if (!library)
    {
        return SLANG_FAIL;
    }

    typedef SlangResult (
        *CreateDownstreamCompilerFunc)(const Guid& intf, IDownstreamCompiler** outCompiler);

    ComPtr<IDownstreamCompiler> downstreamCompiler;

    // Only accept V4, so we can update IArtifact without breaking anything
    if (auto fnV4 = (CreateDownstreamCompilerFunc)library->findFuncByName(
            "createLLVMDownstreamCompiler_V4"))
    {
        SLANG_RETURN_ON_FAIL(
            fnV4(IDownstreamCompiler::getTypeGuid(), downstreamCompiler.writeRef()));
    }
    else
    {
        return SLANG_FAIL;
    }

    set->addSharedLibrary(library);
    set->addCompiler(downstreamCompiler);
    return SLANG_OK;
}

} // namespace Slang
