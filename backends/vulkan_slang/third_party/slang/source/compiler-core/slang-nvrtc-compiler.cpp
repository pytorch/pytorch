// slang-nvrtc-compiler.cpp
#include "slang-nvrtc-compiler.h"

#include "../core/slang-blob.h"
#include "../core/slang-char-util.h"
#include "../core/slang-common.h"
#include "../core/slang-io.h"
#include "../core/slang-semantic-version.h"
#include "../core/slang-shared-library.h"
#include "../core/slang-string-slice-pool.h"
#include "../core/slang-string-util.h"
#include "slang-artifact-associated-impl.h"
#include "slang-artifact-desc-util.h"
#include "slang-artifact-diagnostic-util.h"
#include "slang-artifact-util.h"
#include "slang-com-helper.h"

namespace nvrtc
{

typedef enum
{
    NVRTC_SUCCESS = 0,
    NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4,
    NVRTC_ERROR_INVALID_OPTION = 5,
    NVRTC_ERROR_COMPILATION = 6,
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    NVRTC_ERROR_INTERNAL_ERROR = 11
} nvrtcResult;

typedef struct _nvrtcProgram* nvrtcProgram;

// clang-format off
#define SLANG_NVRTC_FUNCS(x) \
    x(const char*, nvrtcGetErrorString, (nvrtcResult result)) \
    x(nvrtcResult, nvrtcVersion, (int *major, int *minor)) \
    x(nvrtcResult, nvrtcCreateProgram, (nvrtcProgram *prog, const char *src, const char *name, int numHeaders, const char * const *headers, const char * const *includeNames)) \
    x(nvrtcResult, nvrtcDestroyProgram, (nvrtcProgram *prog)) \
    x(nvrtcResult, nvrtcCompileProgram, (nvrtcProgram prog, int numOptions, const char * const *options)) \
    x(nvrtcResult, nvrtcGetPTXSize, (nvrtcProgram prog, size_t *ptxSizeRet)) \
    x(nvrtcResult, nvrtcGetPTX, (nvrtcProgram prog, char *ptx)) \
    x(nvrtcResult, nvrtcGetProgramLogSize, (nvrtcProgram prog, size_t *logSizeRet)) \
    x(nvrtcResult, nvrtcGetProgramLog, (nvrtcProgram prog, char *log))\
    x(nvrtcResult, nvrtcAddNameExpression, (nvrtcProgram prog, const char * const name_expression)) \
    x(nvrtcResult, nvrtcGetLoweredName, (nvrtcProgram prog, const char *const name_expression, const char** lowered_name))
// clang-format on

} // namespace nvrtc

namespace Slang
{
using namespace nvrtc;

static SlangResult _asResult(nvrtcResult res)
{
    switch (res)
    {
    case NVRTC_SUCCESS:
        {
            return SLANG_OK;
        }
    case NVRTC_ERROR_OUT_OF_MEMORY:
        {
            return SLANG_E_OUT_OF_MEMORY;
        }
    case NVRTC_ERROR_PROGRAM_CREATION_FAILURE:
    case NVRTC_ERROR_INVALID_INPUT:
    case NVRTC_ERROR_INVALID_PROGRAM:
        {
            return SLANG_FAIL;
        }
    case NVRTC_ERROR_INVALID_OPTION:
        {
            return SLANG_E_INVALID_ARG;
        }
    case NVRTC_ERROR_COMPILATION:
    case NVRTC_ERROR_BUILTIN_OPERATION_FAILURE:
    case NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
    case NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
    case NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
        {
            return SLANG_FAIL;
        }
    case NVRTC_ERROR_INTERNAL_ERROR:
        {
            return SLANG_E_INTERNAL_FAIL;
        }
    default:
        return SLANG_FAIL;
    }
}

class NVRTCDownstreamCompiler : public DownstreamCompilerBase
{
public:
    typedef DownstreamCompilerBase Super;

    // IDownstreamCompiler
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    compile(const CompileOptions& options, IArtifact** outArtifact) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW bool SLANG_MCALL isFileBased() SLANG_OVERRIDE { return false; }
    virtual SLANG_NO_THROW bool SLANG_MCALL
    canConvert(const ArtifactDesc& from, const ArtifactDesc& to) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    convert(IArtifact* from, const ArtifactDesc& to, IArtifact** outArtifact) SLANG_OVERRIDE;

    /// Must be called before use
    SlangResult init(ISlangSharedLibrary* library);

    NVRTCDownstreamCompiler() {}

protected:
    struct ScopeProgram
    {
        ScopeProgram(NVRTCDownstreamCompiler* compiler, nvrtcProgram program)
            : m_compiler(compiler), m_program(program)
        {
        }
        ~ScopeProgram() { m_compiler->m_nvrtcDestroyProgram(&m_program); }
        NVRTCDownstreamCompiler* m_compiler;
        nvrtcProgram m_program;
    };

    SlangResult _findCUDAIncludePath(String& outIncludePath);
    SlangResult _getCUDAIncludePath(String& outIncludePath);

    SlangResult _findOptixIncludePath(String& outIncludePath);
    SlangResult _getOptixIncludePath(String& outIncludePath);

    SlangResult _maybeAddHalfSupport(const CompileOptions& options, CommandLine& ioCmdLine);
    SlangResult _maybeAddOptixSupport(const CompileOptions& options, CommandLine& ioCmdLine);

#define SLANG_NVTRC_MEMBER_FUNCS(ret, name, params) ret(*m_##name) params;

    SLANG_NVRTC_FUNCS(SLANG_NVTRC_MEMBER_FUNCS);

    // Holds list of paths passed in where cuda_fp16.h is found. Does *NOT* include cuda_fp16.h.
    List<String> m_cudaFp16FoundPaths;

    bool m_cudaIncludeSearched = false;
    // Holds location of where include (for cuda_fp16.h) is found.
    String m_cudaIncludePath;

    // Holds list of paths passed in where optix.h is found. Does *NOT* include optix.h.
    List<String> m_optixFoundPaths;

    bool m_optixIncludeSearched = false;
    // Holds location of where include (for optix.h) is found.
    String m_optixIncludePath;

    ComPtr<ISlangSharedLibrary> m_sharedLibrary;
};

#define SLANG_NVRTC_RETURN_ON_FAIL(x) \
    {                                 \
        nvrtcResult _res = x;         \
        if (_res != NVRTC_SUCCESS)    \
            return _asResult(_res);   \
    }

SlangResult NVRTCDownstreamCompiler::init(ISlangSharedLibrary* library)
{
#define SLANG_NVTRC_GET_FUNC(ret, name, params)               \
    m_##name = (ret(*) params)library->findFuncByName(#name); \
    if (m_##name == nullptr)                                  \
        return SLANG_FAIL;

    SLANG_NVRTC_FUNCS(SLANG_NVTRC_GET_FUNC)

    m_sharedLibrary = library;

    m_desc.type = SLANG_PASS_THROUGH_NVRTC;

    int major, minor;
    m_nvrtcVersion(&major, &minor);
    m_desc.version.set(major, minor);
    return SLANG_OK;
}

static SlangResult _parseLocation(
    SliceAllocator& allocator,
    const UnownedStringSlice& in,
    ArtifactDiagnostic& outDiagnostic)
{
    const Index startIndex = in.indexOf('(');

    if (startIndex >= 0)
    {
        outDiagnostic.filePath = allocator.allocate(in.begin(), in.begin() + startIndex);
        UnownedStringSlice remaining(in.begin() + startIndex + 1, in.end());
        const Int endIndex = remaining.indexOf(')');

        UnownedStringSlice lineText =
            UnownedStringSlice(remaining.begin(), remaining.begin() + endIndex);

        Int line;
        SLANG_RETURN_ON_FAIL(StringUtil::parseInt(lineText, line));
        outDiagnostic.location.line = line;
    }
    else
    {
        outDiagnostic.location.line = 0;
        outDiagnostic.filePath = allocator.allocate(in);
    }
    return SLANG_OK;
}

static bool _isDriveLetter(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

static bool _hasDriveLetter(const UnownedStringSlice& line)
{
    return line.getLength() > 2 && line[1] == ':' && _isDriveLetter(line[0]);
}

static SlangResult _parseNVRTCLine(
    SliceAllocator& allocator,
    const UnownedStringSlice& line,
    ArtifactDiagnostic& outDiagnostic)
{
    typedef ArtifactDiagnostic Diagnostic;
    typedef ArtifactDiagnostic::Severity Severity;

    outDiagnostic.stage = Diagnostic::Stage::Compile;

    List<UnownedStringSlice> split;
    if (_hasDriveLetter(line))
    {
        // The drive letter has :, which confuses things, so skip that and then fix up first entry
        UnownedStringSlice lineWithoutDrive(line.begin() + 2, line.end());
        StringUtil::split(lineWithoutDrive, ':', split);
        split[0] = UnownedStringSlice(line.begin(), split[0].end());
    }
    else
    {
        StringUtil::split(line, ':', split);
    }

    if (split.getCount() >= 3)
    {
        // tests/cuda/cuda-compile.cu(7): warning: variable "c" is used before its value is set
        const auto split1 = split[1].trim();

        Severity severity = Severity::Unknown;

        if (split1 == toSlice("error") || split1 == toSlice("catastrophic error"))
        {
            severity = Severity::Error;
        }
        else if (split1 == toSlice("warning"))
        {
            severity = Severity::Warning;
        }
        else
        {
            // Fall back position to try and determine if this really is some kind of
            // error/warning without succeeding when it's due to some other property
            // of the output diagnostics.
            //
            // Anything ending with " warning:" or " error:" in effect.

            // We can expand to include character after as this is split1, as must be followed by at
            // a minimum : (as the split has at least 3 parts).
            const UnownedStringSlice expandSplit1(split1.begin(), split1.end() + 1);

            if (expandSplit1.endsWith(toSlice(" error:")))
            {
                severity = Severity::Error;
            }
            else if (expandSplit1.endsWith(toSlice(" warning:")))
            {
                severity = Severity::Warning;
            }
        }

        if (severity != Severity::Unknown)
        {
            // The text is everything following the : after the warning.
            UnownedStringSlice text(split[2].begin(), split.getLast().end());

            // Trim whitespace at start and end
            text = text.trim();

            // Set the diagnostic
            outDiagnostic.severity = severity;
            outDiagnostic.text = allocator.allocate(text);
            SLANG_RETURN_ON_FAIL(_parseLocation(allocator, split[0], outDiagnostic));

            return SLANG_OK;
        }

        // TODO(JS): Note here if it's not possible to determine a line as being the main
        // diagnostics we fall through to it potentially being a note.
        //
        // That could mean a valid diagnostic (from NVRTCs point of view) is ignored/noted, because
        // this code can't parse it. Ideally that situation would lead to an error such that we can
        // detect and things will fail.
        //
        // So we might want to revisit this determination in the future.
    }

    // There isn't a diagnostic on this line
    if (line.getLength() == 0 || line.trim().getLength() == 0)
    {
        return SLANG_E_NOT_FOUND;
    }

    // We'll assume it's info, associated with a previous line
    outDiagnostic.severity = Severity::Info;
    outDiagnostic.text = allocator.allocate(line);

    return SLANG_OK;
}

/* An implementation of Path::Visitor that can be used for finding NVRTC shared library
 * installations. */
struct NVRTCPathVisitor : Path::Visitor
{
    struct Candidate
    {
        typedef Candidate ThisType;

        bool operator==(const ThisType& rhs) const
        {
            return path == rhs.path && version == rhs.version;
        }
        bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }

        static Candidate make(const String& path, const SemanticVersion& version)
        {
            Candidate can;
            can.version = version;
            can.path = path;
            return can;
        }
        String path;
        SemanticVersion version;
    };

    Index findVersion(const SemanticVersion& version) const
    {
        const Index count = m_candidates.getCount();
        for (Index i = 0; i < count; ++i)
        {
            if (m_candidates[i].version == version)
            {
                return i;
            }
        }
        return -1;
    }

    static bool _orderCandiate(const Candidate& a, const Candidate& b)
    {
        return a.version < b.version;
    }
    void sortCandidates() { m_candidates.sort(_orderCandiate); }


#if SLANG_WINDOWS_FAMILY
    SlangResult getVersion(const UnownedStringSlice& filename, SemanticVersion& outVersion)
    {
        // Versions on windows of the form
        // nvrtc64_110_2.dll
        //          11 - Major
        //           0 Minor
        //           2 Patch
        Index endIndex = filename.indexOf('.');
        endIndex = (endIndex < 0) ? filename.getLength() : endIndex;

        // If we have a version slice, split it
        UnownedStringSlice versionSlice = UnownedStringSlice(
            filename.begin() + m_prefix.getLength(),
            filename.begin() + endIndex);

        if (versionSlice.getLength() <= 0)
        {
            return SLANG_E_NOT_FOUND;
        }
        Int patch = 0;
        UnownedStringSlice majorMinorSlice;
        {
            List<UnownedStringSlice> slices;
            StringUtil::split(versionSlice, '_', slices);
            if (slices.getCount() >= 2)
            {
                // We don't bother checking for error here, if it's not parsable, it will be 0
                StringUtil::parseInt(slices[1], patch);
            }
            majorMinorSlice = slices[0];
        }

        if (majorMinorSlice.getLength() < 2)
        {
            // Must be a major and minor
            return SLANG_FAIL;
        }

        UnownedStringSlice majorSlice = majorMinorSlice.head(majorMinorSlice.getLength() - 1);
        UnownedStringSlice minorSlice =
            majorMinorSlice.subString(majorMinorSlice.getLength() - 1, 1);

        Int major;
        Int minor;

        SLANG_RETURN_ON_FAIL(StringUtil::parseInt(majorSlice, major));
        SLANG_RETURN_ON_FAIL(StringUtil::parseInt(minorSlice, minor));

        outVersion = SemanticVersion(int(major), int(minor), int(patch));
        return SLANG_OK;
    }
#else
    // How the path is constructed depends on platform
    // https://docs.nvidia.com/cuda/nvrtc/index.html
    // TODO(JS): Handle version number depending on the platform - it's different for
    // Windows/OSX/Linux
    SlangResult getVersion(const UnownedStringSlice& filename, SemanticVersion& outVersion)
    {
        SLANG_UNUSED(filename);
        SLANG_UNUSED(outVersion);
        return SLANG_E_NOT_IMPLEMENTED;
    }

#endif

    void accept(Path::Type type, const UnownedStringSlice& filename) SLANG_OVERRIDE
    {
        // Lets make sure it start's with nvrtc, but not worry about case
        if (type == Path::Type::File)
        {
            // If there is a defined extension, make sure it has it
            if (m_postfix.getLength() && filename.getLength() >= m_postfix.getLength())
            {
                // We test without case - really for windows
                UnownedStringSlice filenamePostfix =
                    filename.tail(filename.getLength() - m_postfix.getLength());
                if (!filenamePostfix.caseInsensitiveEquals(m_postfix.getUnownedSlice()))
                {
                    return;
                }
            }


            if (filename.getLength() >= m_prefix.getLength() &&
                filename.subString(0, m_prefix.getLength())
                    .caseInsensitiveEquals(m_prefix.getUnownedSlice()))
            {
                SemanticVersion version;
                // If it produces an error, just use 0.0.0
                if (SLANG_FAILED(getVersion(filename, version)))
                {
                    version = SemanticVersion();
                }

                // We may want to add multiple versions, if they are in different locations - as
                // there may be multiple entries in the PATH, and only one works. We'll only know
                // which works by loading

#if 0
                // We already found this version, so let's not add it again
                if (findVersion(version) >= 0)
                {
                    return;
                }
#endif

                // Strip to make a shared library name
                UnownedStringSlice sharedLibraryName =
                    filename.tail(m_prefix.getLength() - m_sharedLibraryStem.getLength());
                sharedLibraryName = filename.head(filename.getLength() - m_postfix.getLength());

                auto candidate =
                    Candidate::make(Path::combine(m_basePath, sharedLibraryName), version);

                // If we already have this candidate, then skip
                if (m_candidates.indexOf(candidate) >= 0)
                {
                    return;
                }

                // Add to the list of candidates
                m_candidates.add(candidate);
            }
        }
    }

    SlangResult findInDirectory(const String& path)
    {
        m_basePath = path;
        return Path::find(path, nullptr, this);
    }

    bool hasCandidates() const { return m_candidates.getCount() > 0; }

    NVRTCPathVisitor(const UnownedStringSlice& sharedLibraryStem)
        : m_sharedLibraryStem(sharedLibraryStem)
    {
        // Work out the prefix and postfix of the shader
        StringBuilder buf;
        SharedLibrary::appendPlatformFileName(sharedLibraryStem, buf);
        const Index index = buf.indexOf(sharedLibraryStem);
        SLANG_ASSERT(index >= 0);

        m_prefix = buf.getUnownedSlice().head(index + sharedLibraryStem.getLength());
        m_postfix = buf.getUnownedSlice().tail(index + sharedLibraryStem.getLength());
    }

    String m_prefix;
    String m_postfix;
    String m_basePath;
    String m_sharedLibraryStem;

    List<Candidate> m_candidates;
};

template<typename T>
SLANG_FORCE_INLINE static void _unusedFunction(const T& func)
{
    SLANG_UNUSED(func);
}

#define SLANG_UNUSED_FUNCTION(x) _unusedFunction(x)

static UnownedStringSlice _getNVRTCBaseName()
{
#if SLANG_WINDOWS_FAMILY && SLANG_PTR_IS_64
    return UnownedStringSlice::fromLiteral("nvrtc64_");
#else
    return UnownedStringSlice::fromLiteral("nvrtc");
#endif
}

// Candidates are in m_candidates list. Will be ordered from the oldest to newest (in version
// number)
static SlangResult _findNVRTC(NVRTCPathVisitor& visitor)
{
    // First try the instance path (if supported on platform)
    {
        StringBuilder instancePath;
        if (SLANG_SUCCEEDED(PlatformUtil::getInstancePath(instancePath)))
        {
            visitor.findInDirectory(instancePath);
        }
    }

    // If we don't have a candidate try CUDA_PATH
    if (!visitor.hasCandidates())
    {
        StringBuilder buf;
        if (!SLANG_SUCCEEDED(PlatformUtil::getEnvironmentVariable(
                UnownedStringSlice::fromLiteral("CUDA_PATH"),
                buf)))
        {
            // Look for candidates in the directory
            visitor.findInDirectory(Path::combine(buf, "bin"));
        }
    }

    // If we haven't we go searching through PATH
    if (!visitor.hasCandidates())
    {
        List<UnownedStringSlice> splitPath;

        StringBuilder buf;
        if (SLANG_SUCCEEDED(
                PlatformUtil::getEnvironmentVariable(UnownedStringSlice::fromLiteral("PATH"), buf)))
        {
            // Split so we get individual paths
            List<UnownedStringSlice> paths;
            StringUtil::split(buf.getUnownedSlice(), ';', paths);

            // We use a pool to make sure we only check each path once
            StringSlicePool pool(StringSlicePool::Style::Empty);

            // We are going to search the paths in order
            for (const auto& path : paths)
            {
                // PATH can have the same path multiple times. If we have already searched this
                // path, we don't need to again
                if (!pool.has(path))
                {
                    pool.add(path);

                    Path::split(path, splitPath);

                    // We could search every path, but here we restrict to paths that look like CUDA
                    // installations. It's a path that contains a CUDA directory and has bin
                    if (splitPath.indexOf("CUDA") >= 0 &&
                        splitPath[splitPath.getCount() - 1].caseInsensitiveEquals(
                            UnownedStringSlice::fromLiteral("bin")))
                    {
                        // Okay lets search it
                        visitor.findInDirectory(path);
                    }
                }
            }
        }
    }

    // Put into version order with oldest first.
    visitor.sortCandidates();

    return SLANG_OK;
}

static const UnownedStringSlice g_fp16HeaderName = UnownedStringSlice::fromLiteral("cuda_fp16.h");
static const UnownedStringSlice g_optixHeaderName = UnownedStringSlice::fromLiteral("optix.h");


SlangResult _findFileInIncludePath(
    const String& path,
    const UnownedStringSlice& filename,
    String& outPath)
{
    if (File::exists(Path::combine(path, filename)))
    {
        outPath = path;
        return SLANG_OK;
    }

    {
        String includePath = Path::combine(path, "include");
        if (File::exists(Path::combine(includePath, filename)))
        {
            outPath = includePath;
            return SLANG_OK;
        }
    }

    {
        String cudaIncludePath = Path::combine(path, "CUDA/include");
        if (File::exists(Path::combine(cudaIncludePath, filename)))
        {
            outPath = cudaIncludePath;
            return SLANG_OK;
        }
    }

    return SLANG_E_NOT_FOUND;
}

SlangResult NVRTCDownstreamCompiler::_findCUDAIncludePath(String& outPath)
{
    outPath = String();

    // Try looking up from a symbol. This will work as long as the nvrtc is loaded somehow from a
    // dll/sharedlibrary And the header is included from there
    {
        String libPath = SharedLibraryUtils::getSharedLibraryFileName((void*)m_nvrtcCreateProgram);
        if (libPath.getLength())
        {
            String parentPath = Path::getParentDirectory(libPath);

            if (SLANG_SUCCEEDED(_findFileInIncludePath(parentPath, g_fp16HeaderName, outPath)))
            {
                return SLANG_OK;
            }

            // See if the shared library is in the SDK, as if so we know how to find the includes
            // TODO(JS):
            // This directory structure is correct for windows perhaps could be different elsewhere.
            {
                List<UnownedStringSlice> pathSlices;
                Path::split(parentPath.getUnownedSlice(), pathSlices);

                // This -2 split holds the version number.
                const auto pathSplitCount = pathSlices.getCount();
                if (pathSplitCount >= 3 && pathSlices[pathSplitCount - 1] == toSlice("bin") &&
                    pathSlices[pathSplitCount - 3] == toSlice("CUDA"))
                {
                    // We want to make sure that one of these paths is CUDA...
                    const auto sdkPath = Path::getParentDirectory(parentPath);

                    if (SLANG_SUCCEEDED(_findFileInIncludePath(sdkPath, g_fp16HeaderName, outPath)))
                    {
                        return SLANG_OK;
                    }
                }
            }
        }
    }

    // Try CUDA_PATH environment variable
    {
        StringBuilder buf;
        if (SLANG_SUCCEEDED(PlatformUtil::getEnvironmentVariable(
                UnownedStringSlice::fromLiteral("CUDA_PATH"),
                buf)))
        {
            String includePath = Path::combine(buf, "include");

            if (File::exists(Path::combine(includePath, g_fp16HeaderName)))
            {
                outPath = includePath;
                return SLANG_OK;
            }
        }
    }

#if SLANG_LINUX_FAMILY
    // Try /usr/include
    {
        String includePath = "/usr/include";

        if (File::exists(Path::combine(includePath, g_fp16HeaderName)))
        {
            outPath = includePath;
            return SLANG_OK;
        }
    }
#endif

    return SLANG_E_NOT_FOUND;
}

SlangResult NVRTCDownstreamCompiler::_getCUDAIncludePath(String& outPath)
{
    if (!m_cudaIncludeSearched)
    {
        m_cudaIncludeSearched = true;

        SLANG_ASSERT(m_cudaIncludePath.getLength() == 0);

        _findCUDAIncludePath(m_cudaIncludePath);
    }

    outPath = m_cudaIncludePath;
    return m_cudaIncludePath.getLength() ? SLANG_OK : SLANG_E_NOT_FOUND;
}

SlangResult NVRTCDownstreamCompiler::_findOptixIncludePath(String& outPath)
{
    outPath = String();

    List<String> rootPaths;

#if SLANG_WINDOWS_FAMILY
    const char* searchPattern = "OptiX SDK *";
    StringBuilder builder;
    if (SLANG_SUCCEEDED(PlatformUtil::getEnvironmentVariable(
            UnownedStringSlice::fromLiteral("PROGRAMDATA"),
            builder)))
    {
        rootPaths.add(Path::combine(builder, "NVIDIA Corporation"));
    }
#else
    const char* searchPattern = "NVIDIA-OptiX-SDK-*";
    StringBuilder builder;
    if (SLANG_SUCCEEDED(
            PlatformUtil::getEnvironmentVariable(UnownedStringSlice::fromLiteral("HOME"), builder)))
    {
        rootPaths.add(builder);
    }
#endif

    struct OptixHeaders
    {
        String path;
        SemanticVersion version;
    };

    // Visitor to find Optix headers.
    struct Visitor : public Path::Visitor
    {
        const String& rootPath;
        List<OptixHeaders>& optixPaths;
        Visitor(const String& rootPath, List<OptixHeaders>& optixPaths)
            : rootPath(rootPath), optixPaths(optixPaths)
        {
        }
        void accept(Path::Type type, const UnownedStringSlice& path) SLANG_OVERRIDE
        {
            if (type != Path::Type::Directory)
                return;

            OptixHeaders optixPath;
#if SLANG_WINDOWS_FAMILY
            // Paths are expected to look like ".\OptiX SDK X.X.X"
            auto versionString = path.subString(path.lastIndexOf(' ') + 1, path.getLength());
#else
            // Paths are expected to look like "./NVIDIA-OptiX-SDK-X.X.X-suffix"
            auto versionString = path.subString(0, path.lastIndexOf('-'));
            versionString =
                versionString.subString(path.lastIndexOf('-') + 1, versionString.getLength());
#endif
            if (SLANG_SUCCEEDED(SemanticVersion::parse(versionString, '.', optixPath.version)))
            {
                optixPath.path = Path::combine(Path::combine(rootPath, path), "include");
                String optixHeader = Path::combine(optixPath.path, g_optixHeaderName);
                if (File::exists(optixHeader))
                {
                    optixPaths.add(optixPath);
                }
            }
        }
    };

    List<OptixHeaders> optixPaths;

    for (const String& rootPath : rootPaths)
    {
        Visitor visitor(rootPath, optixPaths);
        Path::find(rootPath, searchPattern, &visitor);
    }

    // Find newest version
    const OptixHeaders* newest = nullptr;
    for (Index i = 0; i < optixPaths.getCount(); ++i)
    {
        if (!newest || optixPaths[i].version > newest->version)
        {
            newest = &optixPaths[i];
        }
    }

    if (newest)
    {
        outPath = newest->path;
        return SLANG_OK;
    }

    return SLANG_E_NOT_FOUND;
}

SlangResult NVRTCDownstreamCompiler::_getOptixIncludePath(String& outPath)
{
    if (!m_optixIncludeSearched)
    {
        m_optixIncludeSearched = true;

        SLANG_ASSERT(m_optixIncludePath.getLength() == 0);

        _findOptixIncludePath(m_optixIncludePath);
    }

    outPath = m_optixIncludePath;
    return m_optixIncludePath.getLength() ? SLANG_OK : SLANG_E_NOT_FOUND;
}

SlangResult NVRTCDownstreamCompiler::_maybeAddHalfSupport(
    const DownstreamCompileOptions& options,
    CommandLine& ioCmdLine)
{
    if ((options.flags & DownstreamCompileOptions::Flag::EnableFloat16) == 0)
    {
        return SLANG_OK;
    }

    // First check if we know if one of the include paths contains cuda_fp16.h
    for (const auto& includePath : options.includePaths)
    {
        if (m_cudaFp16FoundPaths.indexOf(includePath) >= 0)
        {
            // Okay we have an include path that we know works.
            // Just need to enable HALF in prelude
            ioCmdLine.addArg("-DSLANG_CUDA_ENABLE_HALF");
            return SLANG_OK;
        }
    }

    // Let's see if one of the paths finds cuda_fp16.h
    for (const auto& curIncludePath : options.includePaths)
    {
        const String includePath = asString(curIncludePath);
        const String checkPath = Path::combine(includePath, g_fp16HeaderName);
        if (File::exists(checkPath))
        {
            m_cudaFp16FoundPaths.add(includePath);
            // Just need to enable HALF in prelude
            ioCmdLine.addArg("-DSLANG_CUDA_ENABLE_HALF");
            return SLANG_OK;
        }
    }

    String includePath;
    SLANG_RETURN_ON_FAIL(_getCUDAIncludePath(includePath));

    // Add the found include path
    ioCmdLine.addArg("-I");
    ioCmdLine.addArg(includePath);

    ioCmdLine.addArg("-DSLANG_CUDA_ENABLE_HALF");

    return SLANG_OK;
}

SlangResult NVRTCDownstreamCompiler::_maybeAddOptixSupport(
    const DownstreamCompileOptions& options,
    CommandLine& ioCmdLine)
{
    // First check if we know if one of the include paths contains optix.h
    for (const auto& includePath : options.includePaths)
    {
        if (m_optixFoundPaths.indexOf(includePath) >= 0)
        {
            // Okay we have an include path that we know works.
            // Just need to enable OptiX in prelude
            ioCmdLine.addArg("-DSLANG_CUDA_ENABLE_OPTIX");
            return SLANG_OK;
        }
    }

    // Let's see if one of the paths finds optix.h
    for (const auto& curIncludePath : options.includePaths)
    {
        const String includePath = asString(curIncludePath);
        const String checkPath = Path::combine(includePath, g_optixHeaderName);
        if (File::exists(checkPath))
        {
            m_optixFoundPaths.add(includePath);
            // Just need to enable OptiX in prelude
            ioCmdLine.addArg("-DSLANG_CUDA_ENABLE_OPTIX");
            return SLANG_OK;
        }
    }

    String includePath;
    SLANG_RETURN_ON_FAIL(_getOptixIncludePath(includePath));

    // Add the found include path
    ioCmdLine.addArg("-I");
    ioCmdLine.addArg(includePath);

    ioCmdLine.addArg("-DSLANG_CUDA_ENABLE_OPTIX");

    return SLANG_OK;
}

SlangResult NVRTCDownstreamCompiler::compile(
    const DownstreamCompileOptions& inOptions,
    IArtifact** outArtifact)
{
    if (!isVersionCompatible(inOptions))
    {
        // Not possible to compile with this version of the interface.
        return SLANG_E_NOT_IMPLEMENTED;
    }

    CompileOptions options = getCompatibleVersion(&inOptions);

    // This compiler can only deal with a single artifact
    if (options.sourceArtifacts.count != 1)
    {
        return SLANG_FAIL;
    }

    IArtifact* sourceArtifact = options.sourceArtifacts[0];

    CommandLine cmdLine;

    // --dopt option is only available in CUDA 11.7 and later
    bool hasDoptOption = m_desc.version >= SemanticVersion(11, 7);

    switch (options.debugInfoType)
    {
    case DebugInfoType::None:
        {
            break;
        }
    default:
        {
            cmdLine.addArg("--device-debug");
            if (hasDoptOption)
            {
                cmdLine.addArg("--dopt=on");
            }
            break;
        }
    case DebugInfoType::Maximal:
        {
            cmdLine.addArg("--device-debug");
            cmdLine.addArg("--generate-line-info");
            if (hasDoptOption)
            {
                cmdLine.addArg("--dopt=on");
            }
            break;
        }
    }

    // Don't seem to have such a control, so ignore for now
    // switch (options.optimizationLevel)
    //{
    //    default: break;
    //}

    switch (options.floatingPointMode)
    {
    case FloatingPointMode::Default:
        break;
    case FloatingPointMode::Precise:
        {
            break;
        }
    case FloatingPointMode::Fast:
        {
            cmdLine.addArg("--use_fast_math");
            break;
        }
    }

    // Add defines
    for (const auto& define : options.defines)
    {
        StringBuilder builder;
        builder << "-D";
        builder << asStringSlice(define.nameWithSig);
        if (define.value.count)
        {
            builder << "=" << asStringSlice(define.value);
        }

        cmdLine.addArg(builder);
    }

    // Add includes
    for (const auto& include : options.includePaths)
    {
        cmdLine.addArg("-I");
        cmdLine.addArg(asString(include));
    }

    SLANG_RETURN_ON_FAIL(_maybeAddHalfSupport(options, cmdLine));

    // Neither of these options are strictly required, for general use of nvrtc,
    // but are enabled to make use withing Slang work more smoothly
    {
        // Require c++17, the default at the time of writing, since we share
        // some functionality between slang itself and the compiled code
        cmdLine.addArg("-std=c++17");

        // Disable all warnings
        // This is arguably too much - but nvrtc does not appear to have a mechanism to switch off
        // individual warnings. I tried the -Xcudafe mechanism but that does not appear to work for
        // nvrtc
        cmdLine.addArg("-w");
    }

    {
        // The lowest supported CUDA architecture version supported
        // by any version of NVRTC we support is `compute_30`.
        //
        SemanticVersion version(3);

        // Newer releases of NVRTC only support newer CUDA architectures.
        if (m_desc.version.m_major >= 12)
        {
            // NVRTC in CUDA 12 only supports `compute_50` and up
            // (with everything before `compute_52` being deprecated).
            version = SemanticVersion(5, 0);
        }
        else if (m_desc.version.m_major == 11)
        {
            // NVRTC in CUDA 11 only supports `compute_35` and up
            // (with everything before `compute_52` being deprecated).
            version = SemanticVersion(3, 5);
        }

        // If constructs used in the code to be compield require
        // a higher architecture version than the minimum, then
        // we will set the version to the highest version listed
        // among the requirements.
        //
        for (const auto& capabilityVersion : options.requiredCapabilityVersions)
        {
            if (capabilityVersion.kind == DownstreamCompileOptions::CapabilityVersion::Kind::CUDASM)
            {
                if (capabilityVersion.version > version)
                {
                    version = capabilityVersion.version;
                }
            }
        }

        StringBuilder builder;
        builder << "-arch=compute_";
        builder << version.m_major;

        SLANG_ASSERT(version.m_minor >= 0 && version.m_minor <= 9);
        builder << char('0' + version.m_minor);

        cmdLine.addArg(builder);
    }

    List<const char*> headers;
    List<const char*> headerIncludeNames;

    // If compiling for OptiX, we need to add the appropriate search paths to the command line.
    //
    if (options.pipelineType == PipelineType::RayTracing)
    {
        SLANG_RETURN_ON_FAIL(_maybeAddOptixSupport(options, cmdLine));
    }

    // Add any compiler specific options
    // NOTE! If these clash with any previously set options (as set via other flags)
    // compilation might fail.
    if (options.compilerSpecificArguments.count > 0)
    {
        for (auto compilerSpecificArg : options.compilerSpecificArguments)
        {
            const char* const arg = compilerSpecificArg;
            cmdLine.addArg(arg);
        }
    }

    SLANG_ASSERT(headers.getCount() == headerIncludeNames.getCount());

    ComPtr<ISlangBlob> sourceBlob;
    SLANG_RETURN_ON_FAIL(sourceArtifact->loadBlob(ArtifactKeep::Yes, sourceBlob.writeRef()));

    auto sourcePath = ArtifactUtil::findPath(sourceArtifact);

    StringBuilder storage;
    auto sourceContents = SliceUtil::toTerminatedCharSlice(storage, sourceBlob);

    nvrtcProgram program = nullptr;
    nvrtcResult res = m_nvrtcCreateProgram(
        &program,
        sourceContents,
        String(sourcePath).getBuffer(),
        (int)headers.getCount(),
        headers.getBuffer(),
        headerIncludeNames.getBuffer());
    if (res != NVRTC_SUCCESS)
    {
        return _asResult(res);
    }
    ScopeProgram scope(this, program);

    List<const char*> dstOptions;
    dstOptions.setCount(cmdLine.m_args.getCount());
    for (Index i = 0; i < cmdLine.m_args.getCount(); ++i)
    {
        dstOptions[i] = cmdLine.m_args[i].getBuffer();
    }

    res = m_nvrtcCompileProgram(program, int(dstOptions.getCount()), dstOptions.getBuffer());

    auto artifact = ArtifactUtil::createArtifactForCompileTarget(options.targetType);
    auto diagnostics = ArtifactDiagnostics::create();

    ArtifactUtil::addAssociated(artifact, diagnostics);

    ComPtr<ISlangBlob> blob;

    diagnostics->setResult(_asResult(res));

    {
        String rawDiagnostics;

        size_t logSize = 0;
        SLANG_NVRTC_RETURN_ON_FAIL(m_nvrtcGetProgramLogSize(program, &logSize));

        if (logSize)
        {
            char* dst = rawDiagnostics.prepareForAppend(Index(logSize));
            SLANG_NVRTC_RETURN_ON_FAIL(m_nvrtcGetProgramLog(program, dst));

            // If there is a terminating zero remove it, as the rawDiagnostics
            // string will already contain one.
            logSize -= size_t(logSize > 0 && dst[logSize - 1] == 0);

            rawDiagnostics.appendInPlace(dst, Index(logSize));

            diagnostics->setRaw(SliceUtil::asCharSlice(rawDiagnostics));
        }

        SliceAllocator allocator;

        // Get all of the lines
        List<UnownedStringSlice> lines;
        StringUtil::calcLines(rawDiagnostics.getUnownedSlice(), lines);

        // Remove any trailing empty lines
        while (lines.getCount() && lines.getLast().getLength() == 0)
        {
            lines.removeLast();
        }

        // Find the index searching from last line, that is blank
        // indicating the end of the output
        Index lastIndex = lines.getCount();

        // Look for the first blank line after this point.
        // We'll assume any information after that blank line to the end of the diagnostic
        // is compilation summary information.
        for (Index i = lastIndex - 1; i >= 0; --i)
        {
            if (lines[i].getLength() == 0)
            {
                lastIndex = i;
                break;
            }
        }

        // Parse the diagnostics here
        for (auto line : makeConstArrayView(lines.getBuffer(), lastIndex))
        {
            ArtifactDiagnostic diagnostic;
            SlangResult lineRes = _parseNVRTCLine(allocator, line, diagnostic);

            if (SLANG_SUCCEEDED(lineRes))
            {
                // We only allow info diagnostics after a 'regular' diagnostic.
                if (diagnostic.severity == ArtifactDiagnostic::Severity::Info &&
                    diagnostics->getCount() == 0)
                {
                    continue;
                }

                diagnostics->add(diagnostic);
            }
            else if (lineRes != SLANG_E_NOT_FOUND)
            {
                // If there is an error exit
                // But if SLANG_E_NOT_FOUND that just means this line couldn't be parsed, so ignore.
                return lineRes;
            }
        }

        // If it has a compilation error.. and there isn't already an error set
        // set as failed.
        if (SLANG_SUCCEEDED(diagnostics->getResult()) &&
            diagnostics->hasOfAtLeastSeverity(ArtifactDiagnostic::Severity::Error))
        {
            diagnostics->setResult(SLANG_FAIL);
        }
    }

    if (res == nvrtc::NVRTC_SUCCESS)
    {
        // We should parse the log to set up the diagnostics
        size_t ptxSize;
        SLANG_NVRTC_RETURN_ON_FAIL(m_nvrtcGetPTXSize(program, &ptxSize));

        List<uint8_t> ptx;
        ptx.setCount(Index(ptxSize));

        SLANG_NVRTC_RETURN_ON_FAIL(m_nvrtcGetPTX(program, (char*)ptx.getBuffer()));

        artifact->addRepresentationUnknown(ListBlob::moveCreate(ptx));
    }

    *outArtifact = artifact.detach();
    return SLANG_OK;
}

bool NVRTCDownstreamCompiler::canConvert(const ArtifactDesc& from, const ArtifactDesc& to)
{
    return ArtifactDescUtil::isDisassembly(from, to) || ArtifactDescUtil::isDisassembly(to, from);
}

SlangResult NVRTCDownstreamCompiler::convert(
    IArtifact* from,
    const ArtifactDesc& to,
    IArtifact** outArtifact)
{
    if (!canConvert(from->getDesc(), to))
    {
        return SLANG_FAIL;
    }

    // PTX is 'binary like' and 'assembly like' so we allow conversion either way
    // We do it by just getting as a blob and sharing that blob.
    // A more sophisticated implementation could proxy to the original artifact, but this
    // is simpler, and probably fine in most scenarios.
    ComPtr<ISlangBlob> blob;
    SLANG_RETURN_ON_FAIL(from->loadBlob(ArtifactKeep::Yes, blob.writeRef()));

    auto artifact = ArtifactUtil::createArtifact(to);
    artifact->addRepresentationUnknown(blob);

    *outArtifact = artifact.detach();
    return SLANG_OK;
}

static SlangResult _findAndLoadNVRTC(
    ISlangSharedLibraryLoader* loader,
    ComPtr<ISlangSharedLibrary>& outLibrary)
{
#if SLANG_WINDOWS_FAMILY && SLANG_PTR_IS_64

    // We only need to search 64 bit versions on windows
    NVRTCPathVisitor visitor(_getNVRTCBaseName());
    SLANG_RETURN_ON_FAIL(_findNVRTC(visitor));

    // We want to start with the newest version...
    for (Index i = visitor.m_candidates.getCount() - 1; i >= 0; --i)
    {
        const auto& candidate = visitor.m_candidates[i];
        if (SLANG_SUCCEEDED(
                loader->loadSharedLibrary(candidate.path.getBuffer(), outLibrary.writeRef())))
        {
            return SLANG_OK;
        }
    }

#else
    SLANG_UNUSED(loader);
    SLANG_UNUSED(outLibrary);

    SLANG_UNUSED_FUNCTION(_getNVRTCBaseName);
    SLANG_UNUSED_FUNCTION(_findNVRTC);
#endif

    // This is an official-ish list of versions is here:
    // https://developer.nvidia.com/cuda-toolkit-archive

    // Filenames for NVRTC
    // https://docs.nvidia.com/cuda/nvrtc/index.html
    //
    // From this it appears on platforms other than windows the SharedLibrary name
    // should be nvrtc which is already tried, so we can give up now.
    return SLANG_E_NOT_FOUND;
}

/* static */ SlangResult NVRTCDownstreamCompilerUtil::locateCompilers(
    const String& path,
    ISlangSharedLibraryLoader* loader,
    DownstreamCompilerSet* set)
{
    ComPtr<ISlangSharedLibrary> library;

    // If the user supplies a path to their preferred version of NVRTC,
    // we just use this.
    if (path.getLength() != 0)
    {
        SLANG_RETURN_ON_FAIL(loader->loadSharedLibrary(path.getBuffer(), library.writeRef()));
    }
    else
    {
        // As a catch-all for non-Windows platforms, we search for
        // a library simply named `nvrtc` (well, `libnvrtc`) which
        // is expected to match whatever the user has installed.
        //
        // On Windows an installation could place the version of nvrtc it uses in the same directory
        // as the slang binary, such that it's loaded.
        // Using this name also allows a ISlangSharedLibraryLoader to easily identify what is
        // required and perhaps load a specific version
        if (SLANG_FAILED(loader->loadSharedLibrary("nvrtc", library.writeRef())))
        {
            // Try something more sophisticated to locate NVRTC
            SLANG_RETURN_ON_FAIL(_findAndLoadNVRTC(loader, library));
        }
    }

    SLANG_ASSERT(library);
    if (!library)
    {
        return SLANG_FAIL;
    }

    auto compiler = new NVRTCDownstreamCompiler;
    ComPtr<IDownstreamCompiler> compilerIntf(compiler);
    SLANG_RETURN_ON_FAIL(compiler->init(library));

    set->addCompiler(compilerIntf);
    return SLANG_OK;
}

} // namespace Slang
