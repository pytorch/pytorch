#include "../core/slang-array-view.h"
#include "../core/slang-basic.h"
#include "../core/slang-blob.h"

static const uint8_t g_glslModule[] = {
#include "slang-glsl-module-generated.h"
};

static Slang::StaticBlob g_glslModuleBlob((const void*)g_glslModule, sizeof(g_glslModule));

extern "C"
{
    SLANG_DLL_EXPORT ISlangBlob* slang_getEmbeddedModule()
    {
        return &g_glslModuleBlob;
    }
}
