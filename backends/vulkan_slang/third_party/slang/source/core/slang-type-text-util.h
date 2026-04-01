#ifndef SLANG_CORE_TYPE_TEXT_UTIL_H
#define SLANG_CORE_TYPE_TEXT_UTIL_H

#include "slang-array-view.h"
#include "slang-name-value.h"
#include "slang-string.h"
#include "slang.h"

namespace Slang
{

/// Utility class to allow conversion of types (such as enums) to and from text types
struct TypeTextUtil
{
    enum class FileSystemType
    {
        Default,
        LoadFile,
        Os,
    };

    struct CompileTargetInfo
    {
        SlangCompileTarget target; ///< The target
        const char* extensions; ///< Comma delimited list of extensions associated with the target
        const char* names;      ///< Comma delimited list of names associated with the target. NOTE!
                                ///< First name is taken as the normal display name.
        const char* description = nullptr; ///< Description, can be null
    };

    /// Get the compile target infos
    static ConstArrayView<CompileTargetInfo> getCompileTargetInfos();

    /// Get the language infos
    static ConstArrayView<NamesDescriptionValue> getLanguageInfos();
    /// Get the compiler infos
    static ConstArrayView<NamesDescriptionValue> getCompilerInfos();
    /// Get the archive type infos
    static ConstArrayView<NamesDescriptionValue> getArchiveTypeInfos();
    /// Get the debug format types
    static ConstArrayView<NamesDescriptionValue> getDebugInfoFormatInfos();
    /// Get the debug levels
    static ConstArrayView<NamesDescriptionValue> getDebugLevelInfos();
    /// Get the floating point modes
    static ConstArrayView<NamesDescriptionValue> getFloatingPointModeInfos();
    // Get the line directive infos
    static ConstArrayView<NamesDescriptionValue> getLineDirectiveInfos();
    /// Get the optimization level info
    static ConstArrayView<NamesDescriptionValue> getOptimizationLevelInfos();
    /// Get the file system type infos
    static ConstArrayView<NamesDescriptionValue> getFileSystemTypeInfos();

    /// Get the scalar type as text.
    static Slang::UnownedStringSlice getScalarTypeName(
        slang::TypeReflection::ScalarType scalarType);

    // Converts text to scalar type. Returns 'none' if not determined
    static slang::TypeReflection::ScalarType findScalarType(const Slang::UnownedStringSlice& text);

    /// Given a slice finds the associated debug info format
    static SlangResult findDebugInfoFormat(
        const Slang::UnownedStringSlice& text,
        SlangDebugInfoFormat& out);

    /// Get the text name for a format
    static UnownedStringSlice getDebugInfoFormatName(SlangDebugInfoFormat format);

    /// As human readable text
    static UnownedStringSlice getPassThroughAsHumanText(SlangPassThrough type);

    /// Given a source language name returns a source language. Name here is distinct from extension
    static SlangSourceLanguage findSourceLanguage(const UnownedStringSlice& text);

    /// Given a name returns the pass through
    static SlangPassThrough findPassThrough(const UnownedStringSlice& slice);
    static SlangResult findPassThrough(
        const UnownedStringSlice& slice,
        SlangPassThrough& outPassThrough);

    /// Get the compilers name
    static UnownedStringSlice getPassThroughName(SlangPassThrough passThru);

    /// Given a file extension determines suitable target
    /// If doesn't match any target will return SLANG_TARGET_UNKNOWN
    static SlangCompileTarget findCompileTargetFromExtension(const UnownedStringSlice& slice);

    /// Given a name suitable target
    /// If doesn't match any target will return SLANG_TARGET_UNKNOWN
    static SlangCompileTarget findCompileTargetFromName(const UnownedStringSlice& slice);

    /// Given a target returns the associated name.
    static UnownedStringSlice getCompileTargetName(SlangCompileTarget target);

    /// Returns SLANG_ARCHIVE_TYPE_UNKNOWN if a match is not found
    static SlangArchiveType findArchiveType(const UnownedStringSlice& slice);
};

} // namespace Slang

#endif // SLANG_CORE_TYPE_TEXT_UTIL_H
