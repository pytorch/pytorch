#ifndef SLANG_DOWNSTREAM_COMPILER_UTIL_H
#define SLANG_DOWNSTREAM_COMPILER_UTIL_H

#include "slang-downstream-compiler-set.h"
#include "slang-downstream-compiler.h"

namespace Slang
{

typedef SlangResult (*DownstreamCompilerLocatorFunc)(
    const String& path,
    ISlangSharedLibraryLoader* loader,
    DownstreamCompilerSet* set);

struct DownstreamCompilerInfo
{
    typedef DownstreamCompilerInfo This;
    typedef uint32_t SourceLanguageFlags;
    struct SourceLanguageFlag
    {
        enum Enum : SourceLanguageFlags
        {
            Unknown = SourceLanguageFlags(1) << SLANG_SOURCE_LANGUAGE_UNKNOWN,
            Slang = SourceLanguageFlags(1) << SLANG_SOURCE_LANGUAGE_SLANG,
            HLSL = SourceLanguageFlags(1) << SLANG_SOURCE_LANGUAGE_HLSL,
            GLSL = SourceLanguageFlags(1) << SLANG_SOURCE_LANGUAGE_GLSL,
            C = SourceLanguageFlags(1) << SLANG_SOURCE_LANGUAGE_C,
            CPP = SourceLanguageFlags(1) << SLANG_SOURCE_LANGUAGE_CPP,
            CUDA = SourceLanguageFlags(1) << SLANG_SOURCE_LANGUAGE_CUDA,
            SPIRV = SourceLanguageFlags(1) << SLANG_SOURCE_LANGUAGE_SPIRV,
        };
    };

    /// Get info for a compiler type
    static const This& getInfo(SlangPassThrough compiler);
    /// True if this compiler can compile the specified language
    static bool canCompile(SlangPassThrough compiler, SlangSourceLanguage sourceLanguage);

    DownstreamCompilerInfo()
        : sourceLanguageFlags(0)
    {
    }

    DownstreamCompilerInfo(SourceLanguageFlags inSourceLanguageFlags)
        : sourceLanguageFlags(inSourceLanguageFlags)
    {
    }
    SourceLanguageFlags sourceLanguageFlags;
};


// Combination of a downstream compiler type (pass through) and
// a match version.
struct DownstreamCompilerMatchVersion
{
    DownstreamCompilerMatchVersion(SlangPassThrough inType, MatchSemanticVersion inMatchVersion)
        : type(inType), matchVersion(inMatchVersion)
    {
    }

    DownstreamCompilerMatchVersion()
        : type(SLANG_PASS_THROUGH_NONE)
    {
    }

    SlangPassThrough type;             ///< The type of the compiler
    MatchSemanticVersion matchVersion; ///< The match version
};

struct DownstreamCompilerUtil : public DownstreamCompilerUtilBase
{
    enum class MatchType
    {
        MinGreaterEqual,
        MinAbsolute,
        Newest,
    };

    /// Find a compiler
    static IDownstreamCompiler* findCompiler(
        const DownstreamCompilerSet* set,
        MatchType matchType,
        const DownstreamCompilerDesc& desc);
    static IDownstreamCompiler* findCompiler(
        const List<IDownstreamCompiler*>& compilers,
        MatchType matchType,
        const DownstreamCompilerDesc& desc);

    static IDownstreamCompiler* findCompiler(
        const List<IDownstreamCompiler*>& compilers,
        SlangPassThrough type,
        const SemanticVersion& version);
    static IDownstreamCompiler* findCompiler(
        const List<IDownstreamCompiler*>& compilers,
        const DownstreamCompilerDesc& desc);

    /// Find all the compilers with the version
    static void findVersions(
        const List<IDownstreamCompiler*>& compilers,
        SlangPassThrough compiler,
        List<SemanticVersion>& versions);


    /// Find the compiler closest to the desc
    static IDownstreamCompiler* findClosestCompiler(
        const List<IDownstreamCompiler*>& compilers,
        const DownstreamCompilerMatchVersion& version);
    static IDownstreamCompiler* findClosestCompiler(
        const DownstreamCompilerSet* set,
        const DownstreamCompilerMatchVersion& version);

    /// Get the information on the compiler used to compile this source
    static DownstreamCompilerMatchVersion getCompiledVersion();

    static void updateDefault(DownstreamCompilerSet* set, SlangSourceLanguage sourceLanguage);
    static void updateDefaults(DownstreamCompilerSet* set);

    static void setDefaultLocators(
        DownstreamCompilerLocatorFunc outFuncs[int(SLANG_PASS_THROUGH_COUNT_OF)]);

    /// Attempts to determine what 'path' is and load appropriately. Is it a path to a shared
    /// library? Is it a directory holding the libraries? Some downstream shared libraries need
    /// other shared libraries to be loaded before the main shared library, such that they are in
    /// the same directory otherwise the shared library could come from some unwanted location.
    /// dependentNames names shared libraries which should be attempted to be loaded in the path of
    /// the main shared library. The list is optional (nullptr can be passed in), and the list is
    /// terminated by nullptr.
    static SlangResult loadSharedLibrary(
        const String& path,
        ISlangSharedLibraryLoader* loader,
        const char* const* dependantNames,
        const char* libraryName,
        ComPtr<ISlangSharedLibrary>& outSharedLib);

    /// Append the desc as text
    static void appendAsText(const DownstreamCompilerDesc& desc, StringBuilder& out);
};

} // namespace Slang

#endif
