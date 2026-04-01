#include "slang-rt.h"

#include "../core/slang-basic.h"
#include "../core/slang-shared-library.h"

#if SLANG_WINDOWS_FAMILY
#include <windows.h>
#endif

using namespace Slang;
Dictionary<String, ComPtr<ISlangSharedLibrary>> slangRT_loadedLibraries;

extern "C"
{
    SLANG_RT_API void SLANG_MCALL _slang_rt_abort(Slang::String errorMessage)
    {
        fprintf(stderr, "%s", errorMessage.getBuffer());
#if SLANG_WINDOWS_FAMILY
        MessageBoxA(0, errorMessage.getBuffer(), "Slang Runtime Error", MB_ICONERROR);
#endif
        abort();
    }

    SLANG_RT_API void* SLANG_MCALL _slang_rt_load_dll(Slang::String modulePath)
    {
        ComPtr<ISlangSharedLibrary> lib;
        if (!slangRT_loadedLibraries.tryGetValue(modulePath, lib))
        {
            if (DefaultSharedLibraryLoader::getSingleton()->loadSharedLibrary(
                    modulePath.getBuffer(),
                    lib.writeRef()) != SLANG_OK)
            {
                _slang_rt_abort("Failed to load DLL \"" + modulePath + "\"");
            }
            slangRT_loadedLibraries[modulePath] = lib;
        }
        return lib.get();
    }

    SLANG_RT_API void* SLANG_MCALL
    _slang_rt_load_dll_func(void* moduleHandle, Slang::String funcName, uint32_t argSize)
    {
        if (moduleHandle == nullptr)
        {
            moduleHandle = _slang_rt_load_dll("");
        }
        auto lib = static_cast<ISlangSharedLibrary*>(moduleHandle);
        auto funcPtr = lib->findFuncByName(funcName.getBuffer());
        if (!funcPtr)
        {
            // If failed, try stdcall mangled name.
            StringBuilder sb;
            sb << "_" << funcName << "@" << argSize;
            funcPtr = lib->findFuncByName(sb.toString().getBuffer());
        }
        if (!funcPtr)
        {
            _slang_rt_abort("Cannot find function \"" + funcName + "\" in loaded library.");
        }
        return (void*)funcPtr;
    }
}
