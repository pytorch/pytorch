
#include "slang-type-convert-util.h"

namespace Slang
{

/* static */ SlangSourceLanguage TypeConvertUtil::getSourceLanguageFromTarget(
    SlangCompileTarget target)
{
    switch (target)
    {
    case SLANG_GLSL:
        {
            return SLANG_SOURCE_LANGUAGE_GLSL;
        }
    case SLANG_HLSL:
        return SLANG_SOURCE_LANGUAGE_HLSL;
    case SLANG_C_SOURCE:
        return SLANG_SOURCE_LANGUAGE_C;
    case SLANG_CPP_SOURCE:
        return SLANG_SOURCE_LANGUAGE_CPP;
    case SLANG_CPP_PYTORCH_BINDING:
        return SLANG_SOURCE_LANGUAGE_CPP;
    case SLANG_HOST_CPP_SOURCE:
        return SLANG_SOURCE_LANGUAGE_CPP;
    case SLANG_CUDA_SOURCE:
        return SLANG_SOURCE_LANGUAGE_CUDA;
    case SLANG_WGSL:
        return SLANG_SOURCE_LANGUAGE_WGSL;
    default:
        break;
    }
    return SLANG_SOURCE_LANGUAGE_UNKNOWN;
}

/* static */ SlangCompileTarget TypeConvertUtil::getCompileTargetFromSourceLanguage(
    SlangSourceLanguage lang)
{
    switch (lang)
    {
    case SLANG_SOURCE_LANGUAGE_GLSL:
        return SLANG_GLSL;
    case SLANG_SOURCE_LANGUAGE_HLSL:
        return SLANG_HLSL;
    case SLANG_SOURCE_LANGUAGE_C:
        return SLANG_C_SOURCE;
    case SLANG_SOURCE_LANGUAGE_CPP:
        return SLANG_CPP_SOURCE;
    case SLANG_SOURCE_LANGUAGE_CUDA:
        return SLANG_CUDA_SOURCE;
    }

    return SLANG_TARGET_UNKNOWN;
}

} // namespace Slang
