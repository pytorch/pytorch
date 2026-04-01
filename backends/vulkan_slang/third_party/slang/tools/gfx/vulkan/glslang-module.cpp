// glslang-module.cpp
#include "glslang-module.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#if SLANG_WINDOWS_FAMILY
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "../renderer-shared.h"

namespace gfx
{
using namespace Slang;

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! GlslangModule
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Slang::Result GlslangModule::init()
{
    if (isInitialized())
    {
        destroy();
    }

    const char* dynamicLibraryName = "Unknown";

#if SLANG_WINDOWS_FAMILY
    dynamicLibraryName = "slang-glslang.dll";
    HMODULE module = ::LoadLibraryA(dynamicLibraryName);
    m_module = (void*)module;
#elif SLANG_APPLE_FAMILY
    dynamicLibraryName = "libslang_glslang.dylib";
    m_module = dlopen(dynamicLibraryName, RTLD_NOW | RTLD_GLOBAL);
#else
    dynamicLibraryName = "libslang_glslang.so";
    m_module = dlopen(dynamicLibraryName, RTLD_NOW);
#endif

    if (!m_module)
    {
        return SLANG_FAIL;
    }

    // Load functions
#if SLANG_WINDOWS_FAMILY
    m_linkSPIRVFunc = (glslang_LinkSPIRVFunc)GetProcAddress((HMODULE)m_module, "glslang_linkSPIRV");
#else
    m_linkSPIRVFunc = (glslang_LinkSPIRVFunc)dlsym(m_module, "glslang_linkSPIRV");
#endif
    if (!m_linkSPIRVFunc)
    {
        return SLANG_FAIL;
    }

    return SLANG_OK;
}

void GlslangModule::destroy()
{
    if (!isInitialized())
    {
        return;
    }

#if SLANG_WINDOWS_FAMILY
    ::FreeLibrary((HMODULE)m_module);
#else
    dlclose(m_module);
#endif
    m_module = nullptr;
}

ComPtr<ISlangBlob> GlslangModule::linkSPIRV(List<ComPtr<ISlangBlob>> spirvModules)
{

    if (!m_linkSPIRVFunc)
    {
        return nullptr;
    }

    glslang_LinkRequest request = {};

    std::vector<const uint32_t*> moduleCodePtrs(spirvModules.getCount());
    std::vector<uint32_t> moduleSizes(spirvModules.getCount());
    for (Index i = 0; i < spirvModules.getCount(); ++i)
    {
        moduleCodePtrs[i] = (const uint32_t*)spirvModules[i]->getBufferPointer();
        moduleSizes[i] = spirvModules[i]->getBufferSize() / sizeof(uint32_t);
        SLANG_ASSERT(spirvModules[i]->getBufferSize() % sizeof(uint32_t) == 0);
    }
    request.modules = moduleCodePtrs.data();
    request.moduleSizes = moduleSizes.data();
    request.moduleCount = spirvModules.getCount();
    request.linkResult = nullptr;

    m_linkSPIRVFunc(&request);

    ComPtr<ISlangBlob> linkedSPIRV;
    linkedSPIRV = RawBlob::create(request.linkResult, request.linkResultSize * sizeof(uint32_t));
    return linkedSPIRV;
}

} // namespace gfx
