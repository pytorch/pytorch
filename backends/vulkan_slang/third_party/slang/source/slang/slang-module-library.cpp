// slang-module-library.cpp
#include "slang-module-library.h"

#include "../core/slang-blob.h"
#include "../core/slang-riff.h"
#include "../core/slang-type-text-util.h"

#include <assert.h>

// Serialization
#include "slang-serialize-container.h"
#include "slang-serialize-ir.h"

namespace Slang
{

void* ModuleLibrary::getInterface(const Guid& uuid)
{
    if (uuid == ISlangUnknown::getTypeGuid() || uuid == ICastable::getTypeGuid() ||
        uuid == IArtifactRepresentation::getTypeGuid() || uuid == IModuleLibrary::getTypeGuid())
    {
        return static_cast<IModuleLibrary*>(this);
    }
    return nullptr;
}

void* ModuleLibrary::getObject(const Guid& uuid)
{
    return uuid == getTypeGuid() ? this : nullptr;
}

void* ModuleLibrary::castAs(const Guid& guid)
{
    if (auto intf = getInterface(guid))
    {
        return intf;
    }
    return getObject(guid);
}

SlangResult loadModuleLibrary(
    const Byte* inBytes,
    size_t bytesCount,
    String path,
    EndToEndCompileRequest* req,
    ComPtr<IModuleLibrary>& outLibrary)
{
    SLANG_UNUSED(path);

    auto library = new ModuleLibrary;
    ComPtr<IModuleLibrary> scopeLibrary(library);

    // Load up the module
    MemoryStreamBase memoryStream(FileAccess::Read, inBytes, bytesCount);

    RiffContainer riffContainer;
    SLANG_RETURN_ON_FAIL(RiffUtil::read(&memoryStream, riffContainer));

    auto linkage = req->getLinkage();
    auto sink = req->getSink();
    auto namePool = req->getNamePool();

    auto container = ContainerChunkRef::find(&riffContainer);

    for (auto moduleChunk : container.getModules())
    {
        auto loadedModule = linkage->findOrLoadSerializedModuleForModuleLibrary(moduleChunk, sink);
        if (!loadedModule)
            return SLANG_FAIL;

        library->m_modules.add(loadedModule);
    }

    for (auto entryPointChunk : container.getEntryPoints())
    {
        FrontEndCompileRequest::ExtraEntryPointInfo entryPointInfo;
        entryPointInfo.mangledName = entryPointChunk.getMangledName();
        entryPointInfo.name = namePool->getName(entryPointChunk.getName());
        entryPointInfo.profile = entryPointChunk.getProfile();

        library->m_entryPoints.add(entryPointInfo);
    }

    outLibrary.swap(scopeLibrary);
    return SLANG_OK;
}

SlangResult loadModuleLibrary(
    ArtifactKeep keep,
    IArtifact* artifact,
    String path,
    EndToEndCompileRequest* req,
    ComPtr<IModuleLibrary>& outLibrary)
{
    if (auto foundLibrary = findRepresentation<IModuleLibrary>(artifact))
    {
        outLibrary = foundLibrary;
        return SLANG_OK;
    }

    // Load the blob
    ComPtr<ISlangBlob> blob;
    SLANG_RETURN_ON_FAIL(artifact->loadBlob(getIntermediateKeep(keep), blob.writeRef()));

    // Load the module
    ComPtr<IModuleLibrary> library;
    SLANG_RETURN_ON_FAIL(loadModuleLibrary(
        (const Byte*)blob->getBufferPointer(),
        blob->getBufferSize(),
        path,
        req,
        library));

    if (canKeep(keep))
    {
        artifact->addRepresentation(library);
    }

    outLibrary.swap(library);
    return SLANG_OK;
}

} // namespace Slang
