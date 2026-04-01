#ifndef SLANG_CORE_TYPE_CONVERT_UTIL_H
#define SLANG_CORE_TYPE_CONVERT_UTIL_H

#include "slang.h"

namespace Slang
{

/// Utility class for simple conversions between types
struct TypeConvertUtil
{
    /// Convert a target into it's equivalent language if ones available. If not returns
    /// SOURCE_LANGUAGE_UNKNOWN
    static SlangSourceLanguage getSourceLanguageFromTarget(SlangCompileTarget target);

    /// Convert a language into the equivalent target. If not available returns SLANG_TARGET_UNKNOWN
    static SlangCompileTarget getCompileTargetFromSourceLanguage(SlangSourceLanguage lang);
};

} // namespace Slang

#endif // SLANG_CORE_TYPE_TEXT_UTIL_H
