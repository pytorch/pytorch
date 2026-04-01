// slang-fxc-compiler.cpp
#include "slang-fxc-compiler.h"

#if SLANG_ENABLE_DXBC_SUPPORT

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
#include "slang-com-helper.h"
#include "slang-include-system.h"
#include "slang-source-loc.h"

// Enable calling through to `fxc` or `dxc` to
// generate code on Windows.
#ifdef _WIN32
#include <d3dcompiler.h>
#include <windows.h>
#endif

// Some of the `D3DCOMPILE_*` constants aren't available in all
// versions of `d3dcompiler.h`, so we define them here just in case
#ifndef D3DCOMPILE_ENABLE_UNBOUNDED_DESCRIPTOR_TABLES
#define D3DCOMPILE_ENABLE_UNBOUNDED_DESCRIPTOR_TABLES (1 << 20)
#endif

#ifndef D3DCOMPILE_ALL_RESOURCES_BOUND
#define D3DCOMPILE_ALL_RESOURCES_BOUND (1 << 21)
#endif

#endif // SLANG_ENABLE_DXBC_SUPPORT

namespace Slang
{

#if SLANG_ENABLE_DXBC_SUPPORT

static UnownedStringSlice _getSlice(ID3DBlob* blob)
{
    return StringUtil::getSlice((ISlangBlob*)blob);
}

struct FxcIncludeHandler : ID3DInclude
{

    STDMETHOD(Open)
    (D3D_INCLUDE_TYPE includeType,
     LPCSTR fileName,
     LPCVOID parentData,
     LPCVOID* outData,
     UINT* outSize) override
    {
        SLANG_UNUSED(includeType);
        // NOTE! The pParentData means the *text* of any previous include.
        // In order to work out what *path* that came from, we need to seach which source file it
        // came from, and use it's path

        // Assume the root pathInfo initially
        PathInfo includedFromPathInfo = m_rootPathInfo;

        // Lets try and find the parent source if there is any
        if (parentData)
        {
            SourceFile* foundSourceFile =
                m_system.getSourceManager()->findSourceFileByContentRecursively(
                    (const char*)parentData);
            if (foundSourceFile)
            {
                includedFromPathInfo = foundSourceFile->getPathInfo();
            }
        }

        String path(fileName);
        PathInfo pathInfo;
        ComPtr<ISlangBlob> blob;

        SLANG_RETURN_ON_FAIL(
            m_system.findAndLoadFile(path, includedFromPathInfo.foundPath, pathInfo, blob));

        // Return the data
        *outData = blob->getBufferPointer();
        *outSize = (UINT)blob->getBufferSize();

        return S_OK;
    }

    STDMETHOD(Close)(LPCVOID pData) override
    {
        SLANG_UNUSED(pData);
        return S_OK;
    }
    FxcIncludeHandler(
        SearchDirectoryList* searchDirectories,
        ISlangFileSystemExt* fileSystemExt,
        SourceManager* sourceManager)
        : m_system(searchDirectories, fileSystemExt, sourceManager)
    {
    }

    PathInfo m_rootPathInfo;
    IncludeSystem m_system;
};

class FXCDownstreamCompiler : public DownstreamCompilerBase
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

    /// Must be called before use
    SlangResult init(ISlangSharedLibrary* library);

    FXCDownstreamCompiler() {}

protected:
    pD3DCompile m_compile = nullptr;
    pD3DDisassemble m_disassemble = nullptr;

    ComPtr<ISlangSharedLibrary> m_sharedLibrary;
};

SlangResult FXCDownstreamCompiler::init(ISlangSharedLibrary* library)
{
    m_compile = (pD3DCompile)library->findFuncByName("D3DCompile");
    m_disassemble = (pD3DDisassemble)library->findFuncByName("D3DDisassemble");

    if (!m_compile || !m_disassemble)
    {
        return SLANG_FAIL;
    }

    m_sharedLibrary = library;

    // It's not clear how to query for a version, but we can get a version number from the header
    m_desc = Desc(SLANG_PASS_THROUGH_FXC, D3D_COMPILER_VERSION);

    return SLANG_OK;
}

static SlangResult _parseDiagnosticLine(
    SliceAllocator& allocator,
    const UnownedStringSlice& line,
    List<UnownedStringSlice>& lineSlices,
    ArtifactDiagnostic& outDiagnostic)
{
    /* tests/diagnostics/syntax-error-intrinsic.slang(14,2): error X3000: syntax error: unexpected
     * token '@' */
    if (lineSlices.getCount() < 3)
    {
        return SLANG_FAIL;
    }

    SLANG_RETURN_ON_FAIL(
        ArtifactDiagnosticUtil::splitPathLocation(allocator, lineSlices[0], outDiagnostic));

    {
        const UnownedStringSlice severityAndCodeSlice = lineSlices[1].trim();
        const UnownedStringSlice severitySlice =
            StringUtil::getAtInSplit(severityAndCodeSlice, ' ', 0);

        outDiagnostic.code =
            allocator.allocate(StringUtil::getAtInSplit(severityAndCodeSlice, ' ', 1));

        outDiagnostic.severity = ArtifactDiagnostic::Severity::Error;
        if (severitySlice == "warning")
        {
            outDiagnostic.severity = ArtifactDiagnostic::Severity::Warning;
        }
    }

    outDiagnostic.text = allocator.allocate(lineSlices[2].begin(), line.end());
    return SLANG_OK;
}

SlangResult FXCDownstreamCompiler::compile(const CompileOptions& inOptions, IArtifact** outArtifact)
{
    if (!isVersionCompatible(inOptions))
    {
        // Not possible to compile with this version of the interface.
        return SLANG_E_NOT_IMPLEMENTED;
    }

    CompileOptions options = getCompatibleVersion(&inOptions);

    // This compiler can only deal with a single source artifact
    if (options.sourceArtifacts.count != 1)
    {
        return SLANG_FAIL;
    }

    IArtifact* sourceArtifact = options.sourceArtifacts[0];

    if (options.sourceLanguage != SLANG_SOURCE_LANGUAGE_HLSL || options.targetType != SLANG_DXBC)
    {
        SLANG_ASSERT(!"Can only compile HLSL to DXBC");
        return SLANG_FAIL;
    }

    // If we have been invoked in a pass-through mode, then we need to make sure
    // that the downstream compiler sees whatever options were passed to Slang
    // via the command line or API.
    //
    // TODO: more pieces of information should be added here as needed.
    //

    SearchDirectoryList searchDirectories;
    for (const auto& includePath : options.includePaths)
    {
        searchDirectories.searchDirectories.add(asString(includePath));
    }

    const auto sourcePath = ArtifactUtil::findPath(sourceArtifact);

    // Use the default fileSystemExt is not set
    ID3DInclude* includeHandler = nullptr;

    FxcIncludeHandler fxcIncludeHandlerStorage(
        &searchDirectories,
        options.fileSystemExt,
        options.sourceManager);
    if (options.fileSystemExt)
    {

        if (sourcePath.getLength() > 0)
        {
            fxcIncludeHandlerStorage.m_rootPathInfo = PathInfo::makePath(sourcePath);
        }
        includeHandler = &fxcIncludeHandlerStorage;
    }

    List<D3D_SHADER_MACRO> dxMacrosStorage;
    D3D_SHADER_MACRO const* dxMacros = nullptr;

    if (options.defines.count > 0)
    {
        for (const auto& define : options.defines)
        {
            D3D_SHADER_MACRO dxMacro;
            dxMacro.Name = define.nameWithSig;
            dxMacro.Definition = define.value;
            dxMacrosStorage.add(dxMacro);
        }
        D3D_SHADER_MACRO nullTerminator = {0, 0};
        dxMacrosStorage.add(nullTerminator);

        dxMacros = dxMacrosStorage.getBuffer();
    }

    DWORD flags = 0;

    switch (options.floatingPointMode)
    {
    default:
        break;

    case FloatingPointMode::Precise:
        flags |= D3DCOMPILE_IEEE_STRICTNESS;
        break;
    }

    flags |= D3DCOMPILE_ENABLE_STRICTNESS;
    flags |= D3DCOMPILE_ENABLE_UNBOUNDED_DESCRIPTOR_TABLES;

    switch (options.optimizationLevel)
    {
    default:
        break;

    case OptimizationLevel::None:
        flags |= D3DCOMPILE_OPTIMIZATION_LEVEL0;
        break;
    case OptimizationLevel::Default:
        flags |= D3DCOMPILE_OPTIMIZATION_LEVEL1;
        break;
    case OptimizationLevel::High:
        flags |= D3DCOMPILE_OPTIMIZATION_LEVEL2;
        break;
    case OptimizationLevel::Maximal:
        flags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
        break;
    }

    switch (options.debugInfoType)
    {
    case DebugInfoType::None:
        break;

    default:
        flags |= D3DCOMPILE_DEBUG;
        break;
    }

    ComPtr<ISlangBlob> sourceBlob;
    SLANG_RETURN_ON_FAIL(sourceArtifact->loadBlob(ArtifactKeep::Yes, sourceBlob.writeRef()));

    ComPtr<ID3DBlob> codeBlob;
    ComPtr<ID3DBlob> diagnosticsBlob;
    HRESULT hr = m_compile(
        sourceBlob->getBufferPointer(),
        sourceBlob->getBufferSize(),
        String(sourcePath).getBuffer(),
        dxMacros,
        includeHandler,
        options.entryPointName,
        options.profileName,
        flags,
        0, // unused: effect flags
        codeBlob.writeRef(),
        diagnosticsBlob.writeRef());

    auto diagnostics = ArtifactDiagnostics::create();

    // HRESULT is compatible with SlangResult
    diagnostics->setResult(hr);

    SliceAllocator allocator;

    if (diagnosticsBlob)
    {
        UnownedStringSlice diagnosticText = _getSlice(diagnosticsBlob);
        diagnostics->setRaw(asCharSlice(diagnosticText));

        SlangResult diagnosticParseRes = ArtifactDiagnosticUtil::parseColonDelimitedDiagnostics(
            allocator,
            diagnosticText,
            0,
            _parseDiagnosticLine,
            diagnostics);
        SLANG_UNUSED(diagnosticParseRes);
        SLANG_ASSERT(SLANG_SUCCEEDED(diagnosticParseRes));
    }

    // If FXC failed, make sure we have an error in the diagnostics
    if (FAILED(hr))
    {
        diagnostics->requireErrorDiagnostic();
    }

    auto artifact = ArtifactUtil::createArtifactForCompileTarget(options.targetType);

    ArtifactUtil::addAssociated(artifact, diagnostics);

    if (codeBlob)
    {
        // ID3DBlob is compatible with ISlangBlob, so just cast away...
        artifact->addRepresentationUnknown((ISlangBlob*)codeBlob.get());
    }

    *outArtifact = artifact.detach();
    return SLANG_OK;
}

bool FXCDownstreamCompiler::canConvert(const ArtifactDesc& from, const ArtifactDesc& to)
{
    // Can only disassemble blobs that are DXBC
    return ArtifactDescUtil::isDisassembly(from, to) && from.payload == ArtifactPayload::DXBC;
}

SlangResult FXCDownstreamCompiler::convert(
    IArtifact* from,
    const ArtifactDesc& to,
    IArtifact** outArtifact)
{
    if (!canConvert(from->getDesc(), to))
    {
        return SLANG_FAIL;
    }

    ComPtr<ISlangBlob> dxbcBlob;
    SLANG_RETURN_ON_FAIL(from->loadBlob(ArtifactKeep::No, dxbcBlob.writeRef()));

    ComPtr<ID3DBlob> disassemblyBlob;
    SLANG_RETURN_ON_FAIL(m_disassemble(
        dxbcBlob->getBufferPointer(),
        dxbcBlob->getBufferSize(),
        0,
        nullptr,
        disassemblyBlob.writeRef()));

    auto artifact = ArtifactUtil::createArtifact(to);
    // ISlangBlob is compatible with ID3DBlob
    artifact->addRepresentationUnknown((ISlangBlob*)disassemblyBlob.get());

    *outArtifact = artifact.detach();
    return SLANG_OK;
}

/* static */ SlangResult FXCDownstreamCompilerUtil::locateCompilers(
    const String& path,
    ISlangSharedLibraryLoader* loader,
    DownstreamCompilerSet* set)
{
    ComPtr<ISlangSharedLibrary> library;

    const char* const libName = "d3dcompiler_47";
    SLANG_RETURN_ON_FAIL(
        DownstreamCompilerUtil::loadSharedLibrary(path, loader, nullptr, libName, library));

    SLANG_ASSERT(library);
    if (!library)
    {
        return SLANG_FAIL;
    }

    auto compiler = new FXCDownstreamCompiler;
    ComPtr<IDownstreamCompiler> compilerInft(compiler);

    SLANG_RETURN_ON_FAIL(compiler->init(library));

    set->addCompiler(compilerInft);
    return SLANG_OK;
}

#else // SLANG_ENABLE_DXBC_SUPPORT

/* static */ SlangResult FXCDownstreamCompilerUtil::locateCompilers(
    const String& path,
    ISlangSharedLibraryLoader* loader,
    DownstreamCompilerSet* set)
{
    SLANG_UNUSED(path);
    SLANG_UNUSED(loader);
    SLANG_UNUSED(set);
    return SLANG_E_NOT_AVAILABLE;
}

#endif // else SLANG_ENABLE_DXBC_SUPPORT

} // namespace Slang
