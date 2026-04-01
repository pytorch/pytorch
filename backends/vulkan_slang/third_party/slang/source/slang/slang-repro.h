// slang-repro.h
#ifndef SLANG_REPRO_H_INCLUDED
#define SLANG_REPRO_H_INCLUDED

#include "../core/slang-riff.h"
#include "../core/slang-stable-hash.h"
#include "../core/slang-string.h"

// For TranslationUnitRequest
#include "../core/slang-file-system.h"
#include "../core/slang-offset-container.h"
#include "slang-compiler.h"

namespace Slang
{

/* Facilities to be able to save and load the full state of a compilation, including source files,
and all compilation options into 'slang-repro' files. Repro is short for 'reproducible' and it's
main purposes is to make compilations easily reproducible as everything that is needed from
a compilation environment is packaged up into a single file. The single file can be used to
repeat the compilation, or extracted such that everything that was specified in the compilation
can be inspected and modified. */
struct ReproUtil
{
    enum
    {
        kMajorVersion = 0,
        kMinorVersion = 0,
        kPatchVersion = 0,
    };

    static const uint32_t kSlangStateFourCC =
        SLANG_FOUR_CC('S', 'L', 'S', 'T'); ///< Holds all the slang specific chunks
    static const RiffSemanticVersion g_semanticVersion;

    struct Header
    {
        RiffHeader m_chunk;                    ///< The chunk
        RiffSemanticVersion m_semanticVersion; ///< The semantic version
        StableHashCode32
            m_typeHash; ///< A hash based on the binary representation. If doesn't match then not
                        ///< binary compatible (extra check over semantic versioning)
    };

    struct FileState
    {
        Offset32Ptr<OffsetString> uniqueIdentity; ///< The unique identity for the file (from
                                                  ///< ISlangFileSystem), or nullptr
        Offset32Ptr<OffsetString> contents;       ///< The contents of this file
        Offset32Ptr<OffsetString> canonicalPath;  ///< The canonical name of this file (or nullptr)
        Offset32Ptr<OffsetString> foundPath;      ///< The 'found' path

        Offset32Ptr<OffsetString> uniqueName; ///< A generated unique name (not used by slang, but
                                              ///< used as mechanism to replace files)
    };

    struct PathInfoState
    {
        typedef CacheFileSystem::CompressedResult CompressedResult;

        SlangPathType pathType = SLANG_PATH_TYPE_FILE;
        CompressedResult loadFileResult = CompressedResult::Uninitialized;
        CompressedResult getPathTypeResult = CompressedResult::Uninitialized;
        CompressedResult getCanonicalPathResult = CompressedResult::Uninitialized;

        Offset32Ptr<FileState> file; ///< File contents
    };

    struct PathAndPathInfo
    {
        Offset32Ptr<OffsetString> path;
        Offset32Ptr<PathInfoState> pathInfo;
    };

    struct OutputState
    {
        int32_t entryPointIndex;
        Offset32Ptr<OffsetString> outputPath;
    };

    // spSetCodeGenTarget/spAddCodeGenTarget
    // spSetTargetProfile
    // spSetTargetFlags
    // spSetTargetFloatingPointMode
    // spSetTargetMatrixLayoutMode
    struct TargetRequestState
    {
        Profile profile;
        CodeGenTarget target;
        SlangTargetFlags targetFlags;
        FloatingPointMode floatingPointMode;

        Offset32Array<OutputState> outputStates;
    };

    struct StringPair
    {
        Offset32Ptr<OffsetString> first;
        Offset32Ptr<OffsetString> second;
    };

    struct SourceFileState
    {
        PathInfo::Type type;                 ///< The type of this file
        Offset32Ptr<OffsetString> foundPath; ///< The Path this was found along
        Offset32Ptr<FileState> file;         ///< The file contents
    };

    // spAddTranslationUnit
    struct TranslationUnitRequestState
    {
        SourceLanguage language;

        Offset32Ptr<OffsetString> moduleName;

        // spTranslationUnit_addPreprocessorDefine
        Offset32Array<StringPair> preprocessorDefinitions;

        Offset32Array<Offset32Ptr<SourceFileState>> sourceFiles;
    };

    struct EntryPointState
    {
        Offset32Ptr<OffsetString> name;
        Profile profile;
        uint32_t translationUnitIndex;
        Offset32Array<Offset32Ptr<OffsetString>> specializationArgStrings;
    };

    struct RequestState
    {
        Offset32Array<Offset32Ptr<FileState>> files; ///< All of the files
        Offset32Array<Offset32Ptr<SourceFileState>>
            sourceFiles; ///< All of the source files (from source manager)

        // spSetCompileFlags
        SlangCompileFlags compileFlags;
        // spSetDumpIntermediates
        bool shouldDumpIntermediates;
        // spSetLineDirectiveMode
        LineDirectiveMode lineDirectiveMode;

        Offset32Array<TargetRequestState> targetRequests;

        // spSetDebugInfoLevel
        DebugInfoLevel debugInfoLevel;
        // spSetOptimizationLevel
        OptimizationLevel optimizationLevel;
        // spSetOutputContainerFormat
        ContainerFormat containerFormat;
        // spSetPassThrough
        PassThroughMode passThroughMode;

        // spAddSearchPath
        Offset32Array<Offset32Ptr<OffsetString>> searchPaths;

        // spAddPreprocessorDefine
        Offset32Array<StringPair> preprocessorDefinitions;

        bool useUnknownImageFormatAsDefault = false;
        bool obfuscateCode = false;

        Offset32Array<PathAndPathInfo> pathInfoMap; ///< Stores all the accesses to the file system

        Offset32Array<TranslationUnitRequestState> translationUnits;

        Offset32Array<EntryPointState> entryPoints;

        SlangMatrixLayoutMode defaultMatrixLayoutMode;
    };

    static SlangResult store(
        EndToEndCompileRequest* request,
        OffsetContainer& inOutContainer,
        Offset32Ptr<RequestState>& outRequest);

    static SlangResult saveState(EndToEndCompileRequest* request, const String& filename);

    static SlangResult saveState(EndToEndCompileRequest* request, Stream* stream);

    /// Create a cache file system that uses contents of the request state.
    /// The passed in fileSystem is used for accessing any file accesses not found in the cache
    static SlangResult loadFileSystem(
        OffsetBase& base,
        RequestState* requestState,
        ISlangFileSystem* fileSystem,
        ComPtr<ISlangFileSystemExt>& outFileSystem);

    /// Load the requestState into request
    /// The overrideFileSystem is optional and can be passed as nullptr. If set, as each file is
    /// loaded it will attempt to load from fileSystem the *uniqueName*
    static SlangResult load(
        OffsetBase& base,
        RequestState* requestState,
        ISlangFileSystem* overrideFileSystem,
        EndToEndCompileRequest* request);

    static SlangResult loadState(
        const String& filename,
        DiagnosticSink* sink,
        List<uint8_t>& outBuffer);
    static SlangResult loadState(Stream* stream, DiagnosticSink* sink, List<uint8_t>& outBuffer);
    static SlangResult loadState(
        const uint8_t* data,
        size_t size,
        DiagnosticSink* sink,
        List<uint8_t>& outBuffer);

    static RequestState* getRequest(const List<uint8_t>& inBuffer);

    static SlangResult extractFilesToDirectory(const String& file, DiagnosticSink* sink);

    static SlangResult extractFiles(
        OffsetBase& base,
        RequestState* requestState,
        ISlangMutableFileSystem* fileSystem);

    /// Given the repo file work out a suitable path
    static SlangResult calcDirectoryPathFromFilename(const String& filename, String& outPath);

    /// Given a request trys to determine a suitable dump file name, that is unique.
    static SlangResult findUniqueReproDumpStream(
        EndToEndCompileRequest* request,
        String& outFileName,
        RefPtr<Stream>& outStream);
};

} // namespace Slang

#endif
