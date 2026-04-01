#include "slang-shared-library.h"

#include "slang-com-ptr.h"
#include "slang-io.h"
#include "slang-string-util.h"

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__) || defined(SLANG_OSX)
#include <dlfcn.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>
#ifndef _WIN32
#include <unistd.h>
#endif

namespace Slang
{

/* !!!!!!!!!!!!!!!!!!!!!!!!!! DefaultSharedLibraryLoader !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

/* static */ DefaultSharedLibraryLoader DefaultSharedLibraryLoader::s_singleton;

ISlangUnknown* DefaultSharedLibraryLoader::getInterface(const Guid& guid)
{
    return (guid == ISlangUnknown::getTypeGuid() ||
            guid == ISlangSharedLibraryLoader::getTypeGuid())
               ? static_cast<ISlangSharedLibraryLoader*>(this)
               : nullptr;
}

SlangResult DefaultSharedLibraryLoader::loadSharedLibrary(
    const char* path,
    ISlangSharedLibrary** outSharedLibrary)
{
    *outSharedLibrary = nullptr;
    // Try loading
    SharedLibrary::Handle handle;
    SLANG_RETURN_ON_FAIL(SharedLibrary::load(path, handle));
    *outSharedLibrary = ComPtr<ISlangSharedLibrary>(new DefaultSharedLibrary(handle)).detach();
    return SLANG_OK;
}

SlangResult DefaultSharedLibraryLoader::loadPlatformSharedLibrary(
    const char* path,
    ISlangSharedLibrary** outSharedLibrary)
{
    *outSharedLibrary = nullptr;
    // Try loading
    SharedLibrary::Handle handle;
    SLANG_RETURN_ON_FAIL(SharedLibrary::loadWithPlatformPath(path, handle));
    *outSharedLibrary = ComPtr<ISlangSharedLibrary>(new DefaultSharedLibrary(handle)).detach();
    return SLANG_OK;
}

/* static */ SlangResult DefaultSharedLibraryLoader::load(
    ISlangSharedLibraryLoader* loader,
    const String& path,
    const String& name,
    ISlangSharedLibrary** outLibrary)
{
    if (path.getLength())
    {
        String combinedPath = Path::combine(path, name);
        return loader->loadSharedLibrary(combinedPath.getBuffer(), outLibrary);
    }
    else
    {
        return loader->loadSharedLibrary(name.getBuffer(), outLibrary);
    }
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!! ScopeSharedLibrary !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

ScopeSharedLibrary::~ScopeSharedLibrary()
{
    if (m_sharedLibraryHandle)
    {
        // We have to unload if we want to be able to remove
        SharedLibrary::unload(m_sharedLibraryHandle);
        m_sharedLibraryHandle = nullptr;
    }
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!! DefaultSharedLibrary !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

DefaultSharedLibrary::~DefaultSharedLibrary()
{
    if (m_sharedLibraryHandle)
    {
        SharedLibrary::unload(m_sharedLibraryHandle);
    }
}

void* DefaultSharedLibrary::findSymbolAddressByName(char const* name)
{
    return SharedLibrary::findSymbolAddressByName(m_sharedLibraryHandle, name);
}

void* DefaultSharedLibrary::castAs(const SlangUUID& guid)
{
    if (auto intf = getInterface(guid))
    {
        return intf;
    }
    return getObject(guid);
}

void* DefaultSharedLibrary::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == ICastable::getTypeGuid() ||
        guid == ISlangSharedLibrary::getTypeGuid())
    {
        return static_cast<ISlangSharedLibrary*>(this);
    }
    return nullptr;
}

void* DefaultSharedLibrary::getObject(const Guid& guid)
{
    if (guid == DefaultSharedLibrary::getTypeGuid())
    {
        return this;
    }
    return nullptr;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!! SharedLibraryUtils !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

String SharedLibraryUtils::getSharedLibraryFileName(void* symbolInLib)
{
#if defined(_WIN32)
    HMODULE moduleHandle;
    GetModuleHandleExA(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCSTR)symbolInLib,
        &moduleHandle);
    const int maxLength = 1024;
    wchar_t filenameBuffer[maxLength];
    auto length = GetModuleFileNameW(moduleHandle, filenameBuffer, maxLength);
    if (length == maxLength)
    {
        // Insufficient buffer, return empty.
        return String();
    }
    return String::fromWString(filenameBuffer);

#elif defined(__linux__) || defined(SLANG_OSX)
    Dl_info dllInfo;
    if (!dladdr(symbolInLib, &dllInfo))
    {
        return String();
    }
    return dllInfo.dli_fname;

#else
    return String();
#endif
}

uint64_t SharedLibraryUtils::getSharedLibraryTimestamp(void* symbolInLib)
{
    auto fileName = getSharedLibraryFileName(symbolInLib);
    if (fileName.getLength() == 0)
        return 0;
    struct stat result;
    if (stat(fileName.getBuffer(), &result) == 0)
    {
        auto mod_time = result.st_mtime;
        return (uint64_t)mod_time;
    }
    return 0;
}

} // namespace Slang
