// slang-dxc-compiler.cpp
#include "slang-dxc-compiler.h"

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
#include "slang-include-system.h"
#include "slang-source-loc.h"

// Enable DXIL by default unless told not to
#ifndef SLANG_ENABLE_DXIL_SUPPORT
#if SLANG_APPLE_FAMILY
#define SLANG_ENABLE_DXIL_SUPPORT 0
#else
#define SLANG_ENABLE_DXIL_SUPPORT 1
#endif
#endif

// Enable calling through to  `dxc` to
// generate code on Windows.
#if SLANG_ENABLE_DXIL_SUPPORT

#ifdef _WIN32
#include <unknwn.h>
#include <windows.h>
#endif

#include "../../external/dxc/dxcapi.h"

#ifndef _WIN32
#ifdef __uuidof
// DXC's WinAdapter.h defines __uuidof(T) over types, but the existing
// usage in this file is over values (both are accepted on MSVC.)
// We also need to decay through Slang::ComPtr, hence the helper struct
template<typename T>
struct StripSlangComPtr
{
    using type = T;
};
template<typename T>
struct StripSlangComPtr<Slang::ComPtr<T>>
{
    using type = T;
};
#undef __uuidof
#define __uuidof(x) __emulated_uuidof<StripSlangComPtr<std::decay_t<decltype(x)>>::type>()
#endif
#endif
#endif

namespace Slang
{

#if SLANG_ENABLE_DXIL_SUPPORT

static UnownedStringSlice _getSlice(IDxcBlob* blob)
{
    return StringUtil::getSlice((ISlangBlob*)blob);
}

// IDxcIncludeHandler
// 7f61fc7d-950d-467f-b3e3-3c02fb49187c
static const Guid IID_IDxcIncludeHandler =
    {0x7f61fc7d, 0x950d, 0x467f, {0x3c, 0x02, 0xfb, 0x49, 0x18, 0x7c}};

static UnownedStringSlice _addName(const UnownedStringSlice& inSlice, StringSlicePool& pool)
{
    UnownedStringSlice slice = inSlice;
    if (slice.getLength() == 0)
    {
        slice = UnownedStringSlice::fromLiteral("unnamed");
    }

    StringBuilder buf;
    const Index length = slice.getLength();
    buf << slice;

    for (Index i = 0;; ++i)
    {
        buf.reduceLength(length);

        if (i > 0)
        {
            buf << "_" << i;
        }

        StringSlicePool::Handle handle;
        if (!pool.findOrAdd(buf.getUnownedSlice(), handle))
        {
            return pool.getSlice(handle);
        }
    }
}

static UnownedStringSlice _addName(IArtifact* artifact, StringSlicePool& pool)
{
    return _addName(ArtifactUtil::findName(artifact), pool);
}

class DxcIncludeHandler : public IDxcIncludeHandler
{
public:
    // Implement IUnknown
    SLANG_NO_THROW HRESULT SLANG_MCALL QueryInterface(const IID& uuid, void** out) override
    {
        ISlangUnknown* intf = getInterface(reinterpret_cast<const Guid&>(uuid));
        if (intf)
        {
            *out = intf;
            return SLANG_OK;
        }
        return SLANG_E_NO_INTERFACE;
    }
    SLANG_NO_THROW ULONG SLANG_MCALL AddRef() SLANG_OVERRIDE { return 1; }
    SLANG_NO_THROW ULONG SLANG_MCALL Release() SLANG_OVERRIDE { return 1; }

    // Implement IDxcIncludeHandler
    virtual HRESULT SLANG_MCALL LoadSource(LPCWSTR inFilename, IDxcBlob** outSource) SLANG_OVERRIDE
    {
        // Hmm DXC does something a bit odd - when it sees a path, it just passes that in with ./ in
        // front!! NOTE! It doesn't make any difference if it is "" or <> quoted.

        // So we just do a work around where we strip if we see a path starting with ./
        String filePath = String::fromWString(inFilename);

        // If it starts with ./ then attempt to strip it
        if (filePath.startsWith("./"))
        {
            const String remaining = filePath.getUnownedSlice().tail(2);

            // Okay if we strip ./ and what we have is absolute, then it's the absolute path that we
            // care about, otherwise we just leave as is.
            if (Path::isAbsolute(remaining))
            {
                filePath = remaining;
            }
        }

        ComPtr<ISlangBlob> blob;
        PathInfo pathInfo;
        SlangResult res = m_system.findAndLoadFile(filePath, String(), pathInfo, blob);

        // NOTE! This only works because ISlangBlob is *binary compatible* with IDxcBlob, if either
        // change things could go boom
        *outSource = (IDxcBlob*)blob.detach();
        return res;
    }

    DxcIncludeHandler(
        SearchDirectoryList* searchDirectories,
        ISlangFileSystemExt* fileSystemExt,
        SourceManager* sourceManager = nullptr)
        : m_system(searchDirectories, fileSystemExt, sourceManager)
    {
    }

protected:
    // Used by QueryInterface for casting
    ISlangUnknown* getInterface(const Guid& guid)
    {
        if (guid == ISlangUnknown::getTypeGuid() || guid == IID_IDxcIncludeHandler)
        {
            return (ISlangUnknown*)(static_cast<IDxcIncludeHandler*>(this));
        }
        return nullptr;
    }

    IncludeSystem m_system;
};

class DXCDownstreamCompiler : public DownstreamCompilerBase
{
public:
    typedef DownstreamCompilerBase Super;

    // IDownstreamCompiler
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    compile(const CompileOptions& options, IArtifact** outArtifact) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW bool SLANG_MCALL
    canConvert(const ArtifactDesc& from, const ArtifactDesc& to) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    convert(IArtifact* from, const ArtifactDesc& to, IArtifact** outArtifact) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW bool SLANG_MCALL isFileBased() SLANG_OVERRIDE { return false; }
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getVersionString(slang::IBlob** outVersionString)
        SLANG_OVERRIDE;

    /// Must be called before use
    SlangResult init(ISlangSharedLibrary* library);

    DXCDownstreamCompiler() {}

protected:
    DxcCreateInstanceProc m_createInstance = nullptr;

    /// The commit hash associated with the DXC dll used
    /// If 0 length, no hash was found
    String m_commitHash;
    /// The commit count. 0 if not set
    uint32_t m_commitCount = 0;

    ComPtr<ISlangSharedLibrary> m_sharedLibrary;
};

static String _moveTaskMemAllocatedToString(char* chars)
{
    if (chars)
    {
        const String str(chars);
        ::CoTaskMemFree(chars);
        return str;
    }
    return String();
}

SlangResult DXCDownstreamCompiler::init(ISlangSharedLibrary* library)
{
    m_sharedLibrary = library;

    m_createInstance = (DxcCreateInstanceProc)library->findFuncByName("DxcCreateInstance");
    if (!m_createInstance)
    {
        return SLANG_FAIL;
    }

    // Must be able to create the compiler. We inly do this here, because we want to get the
    // compiler version.
    ComPtr<IDxcCompiler> dxcCompiler;
    SLANG_RETURN_ON_FAIL(m_createInstance(
        CLSID_DxcCompiler,
        __uuidof(dxcCompiler),
        (LPVOID*)dxcCompiler.writeRef()));

    uint32_t major = 0;
    uint32_t minor = 0;
    uint32_t patch = 0;

    // Get the version info
    {
        ComPtr<IDxcVersionInfo> versionInfo;
        if (SLANG_SUCCEEDED(dxcCompiler->QueryInterface(versionInfo.writeRef())))
        {
            versionInfo->GetVersion(&major, &minor);
        }
    }

    // Get the commit hash
    {

        ComPtr<IDxcVersionInfo2> versionInfo;
        if (SLANG_SUCCEEDED(dxcCompiler->QueryInterface(versionInfo.writeRef())))
        {
            char* commitHash = nullptr;
            versionInfo->GetCommitInfo(&m_commitCount, &commitHash);
            m_commitHash = _moveTaskMemAllocatedToString(commitHash);
        }
    }

    // Try and get the custom build string, as we can potentially get the patch version from that.
    if (patch == 0)
    {
        ComPtr<IDxcVersionInfo3> versionInfo;

        if (SLANG_SUCCEEDED(dxcCompiler->QueryInterface(versionInfo.writeRef())))
        {
            char* customVersionCString = nullptr;
            versionInfo->GetCustomVersionString(&customVersionCString);

            const String customVersionString = _moveTaskMemAllocatedToString(customVersionCString);

            SemanticVersion semanticVersion(int(major), int(minor), 0);
            StringBuilder buf;
            semanticVersion.append(buf);

            if (customVersionString.startsWith(buf) &&
                customVersionString.getLength() > buf.getLength() + 2 &&
                customVersionString[buf.getLength()] == '.')
            {
                // Get the patch slice
                UnownedStringSlice patchSlice =
                    StringUtil::getAtInSplit(customVersionString.getUnownedSlice(), '.', 2);

                Int patchValue;
                if (SLANG_SUCCEEDED(StringUtil::parseInt(patchSlice, patchValue)) && patchValue > 0)
                {
                    patch = uint32_t(patchValue);
                }
            }
        }
    }

    m_desc = Desc(SLANG_PASS_THROUGH_DXC, SemanticVersion(int(major), int(minor), int(patch)));

    return SLANG_OK;
}

static SlangResult _parseDiagnosticLine(
    SliceAllocator& allocator,
    const UnownedStringSlice& line,
    List<UnownedStringSlice>& lineSlices,
    IArtifactDiagnostics::Diagnostic& outDiagnostic)
{
    /* tests/diagnostics/syntax-error-intrinsic.slang:14:2: error: expected expression */
    if (lineSlices.getCount() < 5)
    {
        return SLANG_FAIL;
    }

    outDiagnostic.filePath = allocator.allocate(lineSlices[0]);

    SLANG_RETURN_ON_FAIL(StringUtil::parseInt(lineSlices[1], outDiagnostic.location.line));

    // Int lineCol;
    // SLANG_RETURN_ON_FAIL(StringUtil::parseInt(lineSlices[2], lineCol));

    UnownedStringSlice severitySlice = lineSlices[3].trim();

    outDiagnostic.severity = ArtifactDiagnostic::Severity::Error;
    if (severitySlice == UnownedStringSlice::fromLiteral("warning"))
    {
        outDiagnostic.severity = ArtifactDiagnostic::Severity::Warning;
    }

    // The rest of the line
    outDiagnostic.text = allocator.allocate(lineSlices[4].begin(), line.end());
    return SLANG_OK;
}

static SlangResult _handleOperationResult(
    IDxcOperationResult* dxcResult,
    IArtifactDiagnostics* diagnostics,
    ComPtr<IDxcBlob>& outBlob)
{
    // Retrieve result.
    HRESULT resultCode = S_OK;
    SLANG_RETURN_ON_FAIL(dxcResult->GetStatus(&resultCode));

    // Note: it seems like the dxcompiler interface
    // doesn't support querying diagnostic output
    // *unless* the compile failed (no way to get
    // warnings out!?).

    if (SLANG_SUCCEEDED(diagnostics->getResult()))
    {
        diagnostics->setResult(resultCode);
    }

    // Try getting the error/diagnostics blob
    ComPtr<IDxcBlobEncoding> dxcErrorBlob;
    dxcResult->GetErrorBuffer(dxcErrorBlob.writeRef());

    if (dxcErrorBlob)
    {
        const UnownedStringSlice diagnosticsSlice = _getSlice(dxcErrorBlob);
        if (diagnosticsSlice.getLength())
        {
            diagnostics->appendRaw(asCharSlice(diagnosticsSlice));

            SliceAllocator allocator;
            List<IArtifactDiagnostics::Diagnostic> parsedDiagnostics;
            SlangResult diagnosticParseRes = ArtifactDiagnosticUtil::parseColonDelimitedDiagnostics(
                allocator,
                diagnosticsSlice,
                0,
                _parseDiagnosticLine,
                diagnostics);

            SLANG_UNUSED(diagnosticParseRes);
            SLANG_ASSERT(SLANG_SUCCEEDED(diagnosticParseRes));
        }
    }

    // If it failed, make sure we have an error in the diagnostics
    if (SLANG_FAILED(resultCode))
    {
        // In case the parsing failed, we still have an error -> so require there is one in the
        // diagnostics
        diagnostics->requireErrorDiagnostic();
    }
    else
    {
        // Okay, the compile supposedly succeeded, so we
        // just need to grab the buffer with the output DXIL.
        SLANG_RETURN_ON_FAIL(dxcResult->GetResult(outBlob.writeRef()));
    }

    return SLANG_OK;
}

SlangResult DXCDownstreamCompiler::compile(const CompileOptions& inOptions, IArtifact** outArtifact)
{
    if (!isVersionCompatible(inOptions))
    {
        // Not possible to compile with this version of the interface.
        return SLANG_E_NOT_IMPLEMENTED;
    }

    CompileOptions options = getCompatibleVersion(&inOptions);

    // This compiler can only deal at most, a single source code artifact
    // Should be okay to link together multiple libraries without any source artifacts (assuming
    // that means source code)
    if (options.sourceArtifacts.count > 1)
    {
        return SLANG_FAIL;
    }

    bool hasSource = options.sourceArtifacts.count > 0;

    IArtifact* sourceArtifact = hasSource ? options.sourceArtifacts[0] : nullptr;

    if (hasSource)
    {
        if (options.sourceLanguage != SLANG_SOURCE_LANGUAGE_HLSL ||
            options.targetType != SLANG_DXIL)
        {
            SLANG_ASSERT(!"Can only compile HLSL to DXIL");
            return SLANG_FAIL;
        }
    }

    // Find all of the libraries
    List<IArtifact*> libraries;
    for (IArtifact* library : options.libraries)
    {
        const auto desc = library->getDesc();

        if (desc.kind == ArtifactKind::Library && desc.payload == ArtifactPayload::DXIL)
        {
            // Make sure they all have blobs
            ComPtr<ISlangBlob> libraryBlob;
            SLANG_RETURN_ON_FAIL(library->loadBlob(ArtifactKeep::Yes, libraryBlob.writeRef()));

            libraries.add(library);
        }
    }

    ComPtr<IDxcCompiler> dxcCompiler;
    SLANG_RETURN_ON_FAIL(m_createInstance(
        CLSID_DxcCompiler,
        __uuidof(dxcCompiler),
        (LPVOID*)dxcCompiler.writeRef()));
    ComPtr<IDxcLibrary> dxcLibrary;
    SLANG_RETURN_ON_FAIL(
        m_createInstance(CLSID_DxcLibrary, __uuidof(dxcLibrary), (LPVOID*)dxcLibrary.writeRef()));

    ComPtr<IDxcBlobEncoding> dxcSourceBlob = nullptr;
    ComPtr<ISlangBlob> sourceBlob;
    if (hasSource)
    {
        SLANG_RETURN_ON_FAIL(sourceArtifact->loadBlob(ArtifactKeep::Yes, sourceBlob.writeRef()));

        // Create blob from the string
        SLANG_RETURN_ON_FAIL(dxcLibrary->CreateBlobWithEncodingFromPinned(
            (LPBYTE)sourceBlob->getBufferPointer(),
            (UINT32)sourceBlob->getBufferSize(),
            0,
            dxcSourceBlob.writeRef()));
    }

    List<const WCHAR*> args;

    // Add all compiler specific options
    List<OSString> compilerSpecific;
    compilerSpecific.setCount(options.compilerSpecificArguments.count);

    for (Index i = 0; i < options.compilerSpecificArguments.count; ++i)
    {
        compilerSpecific[i] = asString(options.compilerSpecificArguments[i]).toWString();
        args.add(compilerSpecific[i]);
    }

    bool enablePAQs = options.enablePAQ;
    if (!enablePAQs)
        args.add(L"-disable-payload-qualifiers");
    else
        args.add(L"-enable-payload-qualifiers");

    // TODO: deal with
    bool treatWarningsAsErrors = false;
    if (treatWarningsAsErrors)
    {
        args.add(L"-WX");
    }

    switch (options.matrixLayout)
    {
    default:
        break;

    case SLANG_MATRIX_LAYOUT_ROW_MAJOR:
        args.add(L"-Zpr");
        break;
    }

    switch (options.floatingPointMode)
    {
    default:
        break;

    case FloatingPointMode::Precise:
        args.add(L"-Gis"); // "force IEEE strictness"
        break;
    }


    switch (options.optimizationLevel)
    {
    default:
        break;

    case OptimizationLevel::None:
        args.add(L"-Od");
        break;
    case OptimizationLevel::Default:
        args.add(L"-O1");
        break;
    case OptimizationLevel::High:
        args.add(L"-O2");
        break;
    case OptimizationLevel::Maximal:
        args.add(L"-O3");
        break;
    }

    switch (options.debugInfoType)
    {
    case DebugInfoType::None:
        break;

    default:
        args.add(L"-Zi");
        break;
    }

    // Slang strives to produce correct code, and by default
    // we do not show the user warnings produced by a downstream
    // compiler. When the downstream compiler *does* produce an
    // error, then we dump its entire diagnostic log, which can
    // include many distracting spurious warnings that have nothing
    // to do with the user's code, and just relate to the idiomatic
    // way that Slang outputs HLSL.
    //
    // It would be nice to use fine-grained flags to disable specific
    // warnings here, so that we keep ourselves honest (e.g., only
    // use `-Wno-parentheses` to eliminate that class of false positives),
    // but alas dxc doesn't support these options even though they
    // work on mainline Clang. Thus the only option we have available
    // is the big hammer of turning off *all* warnings coming from dxc.
    //
    args.add(L"-no-warnings");

    String profileName = asString(options.profileName);
    // If we are going to link we have to compile in the lib profile style
    if (libraries.getCount() && hasSource)
    {
        if (!profileName.startsWith("lib"))
        {
            const Index index = profileName.indexOf('_');
            if (index < 0)
            {
                profileName = "lib_6_3";
            }
            else
            {
                StringBuilder buf;
                buf << "lib" << profileName.getUnownedSlice().tail(index);
                profileName = buf;
            }
        }
    }

    OSString wideEntryPointName = asString(options.entryPointName).toWString();
    OSString wideProfileName = profileName.toWString();

    if (options.flags & CompileOptions::Flag::EnableFloat16)
    {
        args.add(L"-enable-16bit-types");
    }

    SearchDirectoryList searchDirectories;
    for (const auto& includePath : options.includePaths)
    {
        searchDirectories.searchDirectories.add(asString(includePath));
    }

    {
        // Specify -HV 2021 when using a DXC version that supports the newer language model.
        const SemanticVersion firstHlsl2021Version(1, 7);

        if (m_desc.version >= firstHlsl2021Version)
        {
            args.add(L"-HV");
            args.add(L"2021");
        }
    }

    String sourcePath;
    ComPtr<IDxcBlob> dxcResultBlob = nullptr;
    auto diagnostics = ArtifactDiagnostics::create();
    ComPtr<IDxcOperationResult> dxcOperationResult = nullptr;
    if (hasSource)
    {
        sourcePath = ArtifactUtil::findPath(sourceArtifact);
        OSString wideSourcePath = sourcePath.toWString();

        DxcIncludeHandler includeHandler(
            &searchDirectories,
            options.fileSystemExt,
            options.sourceManager);

        SLANG_RETURN_ON_FAIL(dxcCompiler->Compile(
            dxcSourceBlob,
            wideSourcePath.begin(),
            wideEntryPointName.begin(),
            wideProfileName.begin(),
            args.getBuffer(),
            UINT32(args.getCount()),
            nullptr,         // `#define`s
            0,               // `#define` count
            &includeHandler, // `#include` handler
            dxcOperationResult.writeRef()));

        SLANG_RETURN_ON_FAIL(
            _handleOperationResult(dxcOperationResult, diagnostics, dxcResultBlob));
    }

    // If we have libraries then we need to link...
    if (libraries.getCount())
    {
        ComPtr<IDxcLinker> linker;
        SLANG_RETURN_ON_FAIL(
            m_createInstance(CLSID_DxcLinker, __uuidof(linker), (void**)linker.writeRef()));

        StringSlicePool pool(StringSlicePool::Style::Default);

        List<ComPtr<ISlangBlob>> libraryBlobs;
        List<OSString> libraryNames;

        for (IArtifact* library : libraries)
        {
            ComPtr<ISlangBlob> blob;
            SLANG_RETURN_ON_FAIL(library->loadBlob(ArtifactKeep::Yes, blob.writeRef()));

            libraryBlobs.add(blob);
            libraryNames.add(String(_addName(library, pool)).toWString());
        }

        if (hasSource)
        {
            // Add the compiled blob name
            String name;
            if (options.modulePath.count)
            {
                name = Path::getFileNameWithoutExt(asString(options.modulePath));
            }
            else if (sourcePath.getLength())
            {
                name = Path::getFileNameWithoutExt(sourcePath);
            }

            // Add the blob with name
            {
                auto blob = (ISlangBlob*)dxcResultBlob.get();
                libraryBlobs.add(ComPtr<ISlangBlob>(blob));
                libraryNames.add(String(_addName(name.getUnownedSlice(), pool)).toWString());
            }
        }

        const Index librariesCount = libraryNames.getCount();
        SLANG_ASSERT(libraryBlobs.getCount() == librariesCount);
        SLANG_ASSERT(libraryNames.getCount() == librariesCount);

        List<const wchar_t*> linkLibraryNames;

        linkLibraryNames.setCount(librariesCount);

        for (Index i = 0; i < librariesCount; ++i)
        {
            linkLibraryNames[i] = libraryNames[i].begin();

            // Register the library
            SLANG_RETURN_ON_FAIL(
                linker->RegisterLibrary(linkLibraryNames[i], (IDxcBlob*)libraryBlobs[i].get()));
        }

        // Use the original profile name
        wideProfileName = asString(options.profileName).toWString();

        ComPtr<IDxcOperationResult> linkDxcResult;
        SLANG_RETURN_ON_FAIL(linker->Link(
            wideEntryPointName.begin(),
            wideProfileName.begin(),
            linkLibraryNames.getBuffer(),
            UINT32(librariesCount),
            nullptr,
            0,
            linkDxcResult.writeRef()));

        ComPtr<IDxcBlob> linkedBlob;
        SLANG_RETURN_ON_FAIL(_handleOperationResult(linkDxcResult, diagnostics, linkedBlob));

        // When we've linked we make that the overall operation result
        // As presumably it can contain pdb and perhaps other information
        dxcOperationResult = linkDxcResult;

        // Set the result blob
        dxcResultBlob = linkedBlob;
    }

    auto artifact = ArtifactUtil::createArtifactForCompileTarget(options.targetType);

    ArtifactUtil::addAssociated(artifact, diagnostics);

    if (dxcResultBlob)
    {
        artifact->addRepresentationUnknown((ISlangBlob*)dxcResultBlob.get());
    }

    // If asking for PDB extract it.
    if (options.m_debugInfoFormat == SLANG_DEBUG_INFO_FORMAT_PDB)
    {
        ComPtr<IDxcResult> dxcResult;
        if (SLANG_SUCCEEDED(dxcOperationResult->QueryInterface(dxcResult.writeRef())))
        {
            if (dxcResult->HasOutput(DXC_OUT_PDB))
            {
                ComPtr<IDxcBlob> pdbBlob;
                ComPtr<IDxcBlobWide> nameBlob;

                if (SLANG_SUCCEEDED(dxcResult->GetOutput(
                        DXC_OUT_PDB,
                        __uuidof(pdbBlob),
                        (void**)pdbBlob.writeRef(),
                        nameBlob.writeRef())))
                {
                    auto pdbArtifact = ArtifactUtil::createArtifact(ArtifactDesc::make(
                        ArtifactDesc::Kind::BinaryFormat,
                        ArtifactDesc::Payload::PdbDebugInfo));

                    if (nameBlob)
                    {
                        const auto wideName = (const WCHAR*)nameBlob->GetBufferPointer();

                        const auto name = String::fromWString(wideName);
                        if (name.getLength())
                        {
                            // Set the name on the artifact. This is the name that must be used for
                            // the PDB to be loadable as a file by other tooling.
                            pdbArtifact->setName(name.getBuffer());
                        }
                    }

                    pdbArtifact->addRepresentationUnknown((ISlangBlob*)pdbBlob.get());

                    // Associate it
                    artifact->addAssociated(pdbArtifact);
                }
            }
        }
    }

    *outArtifact = artifact.detach();
    return SLANG_OK;
}

bool DXCDownstreamCompiler::canConvert(const ArtifactDesc& from, const ArtifactDesc& to)
{
    return ArtifactDescUtil::isDisassembly(from, to) && from.payload == ArtifactPayload::DXIL;
}

SlangResult DXCDownstreamCompiler::convert(
    IArtifact* from,
    const ArtifactDesc& to,
    IArtifact** outArtifact)
{
    // Can only disassemble blobs that are DXIL
    if (!canConvert(from->getDesc(), to))
    {
        return SLANG_FAIL;
    }

    ComPtr<ISlangBlob> dxilBlob;
    SLANG_RETURN_ON_FAIL(from->loadBlob(ArtifactKeep::No, dxilBlob.writeRef()));

    ComPtr<IDxcCompiler> dxcCompiler;
    SLANG_RETURN_ON_FAIL(m_createInstance(
        CLSID_DxcCompiler,
        __uuidof(dxcCompiler),
        (LPVOID*)dxcCompiler.writeRef()));
    ComPtr<IDxcLibrary> dxcLibrary;
    SLANG_RETURN_ON_FAIL(
        m_createInstance(CLSID_DxcLibrary, __uuidof(dxcLibrary), (LPVOID*)dxcLibrary.writeRef()));

    // Create blob from the input data
    ComPtr<IDxcBlobEncoding> dxcSourceBlob;
    SLANG_RETURN_ON_FAIL(dxcLibrary->CreateBlobWithEncodingFromPinned(
        (LPBYTE)dxilBlob->getBufferPointer(),
        (UINT32)dxilBlob->getBufferSize(),
        0,
        dxcSourceBlob.writeRef()));

    ComPtr<IDxcBlobEncoding> dxcResultBlob;
    SLANG_RETURN_ON_FAIL(dxcCompiler->Disassemble(dxcSourceBlob, dxcResultBlob.writeRef()));

    auto artifact = ArtifactUtil::createArtifact(to);

    // Is compatible with ISlangBlob
    ISlangBlob* disassemblyBlob = (ISlangBlob*)dxcResultBlob.get();
    artifact->addRepresentationUnknown(disassemblyBlob);

    *outArtifact = artifact.detach();
    return SLANG_OK;
}

SlangResult DXCDownstreamCompiler::getVersionString(slang::IBlob** outVersionString)
{
    StringBuilder versionString;
    // Append the version
    m_desc.version.append(versionString);

    if (m_commitHash.getLength())
    {
        versionString << "#" << m_commitHash;
    }
    else
    {
        // If we don't have the commitHash, we use the library timestamp, to uniquely identify.
        versionString << " "
                      << SharedLibraryUtils::getSharedLibraryTimestamp(
                             reinterpret_cast<void*>(m_createInstance));
    }

    *outVersionString = StringBlob::moveCreate(versionString).detach();
    return SLANG_OK;
}

/* static */ SlangResult DXCDownstreamCompilerUtil::locateCompilers(
    const String& path,
    ISlangSharedLibraryLoader* loader,
    DownstreamCompilerSet* set)
{
    ComPtr<ISlangSharedLibrary> library;

    const char* dependentNames[] = {"dxil", nullptr};
    SLANG_RETURN_ON_FAIL(DownstreamCompilerUtil::loadSharedLibrary(
        path,
        loader,
        dependentNames,
        "dxcompiler",
        library));

    SLANG_ASSERT(library);
    if (!library)
    {
        return SLANG_FAIL;
    }

    auto compiler = new DXCDownstreamCompiler;
    ComPtr<IDownstreamCompiler> compilerIntf(compiler);
    SLANG_RETURN_ON_FAIL(compiler->init(library));

    set->addCompiler(compilerIntf);
    return SLANG_OK;
}

#else // SLANG_ENABLE_DXIL_SUPPORT

/* static */ SlangResult DXCDownstreamCompilerUtil::locateCompilers(
    const String& path,
    ISlangSharedLibraryLoader* loader,
    DownstreamCompilerSet* set)
{
    SLANG_UNUSED(path);
    SLANG_UNUSED(loader);
    SLANG_UNUSED(set);
    return SLANG_E_NOT_AVAILABLE;
}

#endif // SLANG_ENABLE_DXIL_SUPPORT

} // namespace Slang
