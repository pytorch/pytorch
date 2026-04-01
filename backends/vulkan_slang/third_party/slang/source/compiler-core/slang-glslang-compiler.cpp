// slang-glslang-compiler.cpp
#include "slang-glslang-compiler.h"

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
#include "slang-com-helper.h"
#include "slang-include-system.h"
#include "slang-source-loc.h"

// Enable calling through to `glslang` on
// all platforms.
#ifndef SLANG_ENABLE_GLSLANG_SUPPORT
#define SLANG_ENABLE_GLSLANG_SUPPORT 1
#endif

#if SLANG_ENABLE_GLSLANG_SUPPORT
#include "../slang-glslang/slang-glslang.h"
#endif

namespace Slang
{

#if SLANG_ENABLE_GLSLANG_SUPPORT

class GlslangDownstreamCompiler : public DownstreamCompilerBase
{
public:
    typedef DownstreamCompilerBase Super;

    // IDownstreamCompiler
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    compile(const CompileOptions& options, IArtifact** outResult) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW bool SLANG_MCALL
    canConvert(const ArtifactDesc& from, const ArtifactDesc& to) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    convert(IArtifact* from, const ArtifactDesc& to, IArtifact** outArtifact) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW bool SLANG_MCALL isFileBased() SLANG_OVERRIDE { return false; }
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getVersionString(slang::IBlob** outVersionString)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    validate(const uint32_t* contents, int contentsSize) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    disassemble(const uint32_t* contents, int contentsSize) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL disassembleWithResult(
        const uint32_t* contents,
        int contentsSize,
        String& outString) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW int SLANG_MCALL link(
        const uint32_t** modules,
        const uint32_t* moduleSizes,
        const uint32_t moduleCount,
        IArtifact** outArtifact) SLANG_OVERRIDE;

    /// Must be called before use
    SlangResult init(ISlangSharedLibrary* library);

    GlslangDownstreamCompiler(SlangPassThrough compilerType)
        : m_compilerType(compilerType)
    {
    }

protected:
    SlangResult _invoke(glslang_CompileRequest_1_2& request);

    glslang_CompileFunc_1_0 m_compile_1_0 = nullptr;
    glslang_CompileFunc_1_1 m_compile_1_1 = nullptr;
    glslang_CompileFunc_1_2 m_compile_1_2 = nullptr;
    glslang_ValidateSPIRVFunc m_validate = nullptr;
    glslang_DisassembleSPIRVFunc m_disassemble = nullptr;
    glslang_DisassembleSPIRVWithResultFunc m_disassembleWithResult = nullptr;
    glslang_LinkSPIRVFunc m_link = nullptr;

    ComPtr<ISlangSharedLibrary> m_sharedLibrary;

    SlangPassThrough m_compilerType;
};

SlangResult GlslangDownstreamCompiler::init(ISlangSharedLibrary* library)
{
    m_compile_1_0 = (glslang_CompileFunc_1_0)library->findFuncByName("glslang_compile");
    m_compile_1_1 = (glslang_CompileFunc_1_1)library->findFuncByName("glslang_compile_1_1");
    m_compile_1_2 = (glslang_CompileFunc_1_2)library->findFuncByName("glslang_compile_1_2");
    m_validate = (glslang_ValidateSPIRVFunc)library->findFuncByName("glslang_validateSPIRV");
    m_disassemble =
        (glslang_DisassembleSPIRVFunc)library->findFuncByName("glslang_disassembleSPIRV");
    m_disassembleWithResult = (glslang_DisassembleSPIRVWithResultFunc)library->findFuncByName(
        "glslang_disassembleSPIRVWithResult");
    m_link = (glslang_LinkSPIRVFunc)library->findFuncByName("glslang_linkSPIRV");

    if (m_compile_1_0 == nullptr && m_compile_1_1 == nullptr && m_compile_1_2 == nullptr)
    {
        return SLANG_FAIL;
    }

    m_sharedLibrary = library;

    // It's not clear how to query for a version, but we can get a version number from the header
    m_desc = Desc(m_compilerType);

    Slang::String filename;
    if (m_compile_1_2)
    {
        filename = Slang::SharedLibraryUtils::getSharedLibraryFileName((void*)m_compile_1_2);
    }
    else if (m_compile_1_1)
    {
        filename = Slang::SharedLibraryUtils::getSharedLibraryFileName((void*)m_compile_1_1);
    }
    else if (m_compile_1_0)
    {
        filename = Slang::SharedLibraryUtils::getSharedLibraryFileName((void*)m_compile_1_0);
    }
    else
    {
        return SLANG_FAIL;
    }

    return SLANG_OK;
}

SlangResult GlslangDownstreamCompiler::_invoke(glslang_CompileRequest_1_2& request)
{
    int err = 1;
    if (m_compile_1_2)
    {
        err = m_compile_1_2(&request);
    }
    else if (m_compile_1_1)
    {
        glslang_CompileRequest_1_1 request_1_1;
        memcpy(&request_1_1, &request, sizeof(request_1_1));
        request_1_1.sizeInBytes = sizeof(request_1_1);
        err = m_compile_1_1(&request_1_1);
    }
    else if (m_compile_1_0)
    {
        glslang_CompileRequest_1_1 request_1_1;
        memcpy(&request_1_1, &request, sizeof(request_1_1));
        request_1_1.sizeInBytes = sizeof(request_1_1);
        glslang_CompileRequest_1_0 request_1_0;
        request_1_0.set(request_1_1);
        err = m_compile_1_0(&request_1_0);
    }

    return err ? SLANG_FAIL : SLANG_OK;
}

static SlangResult _parseDiagnosticLine(
    SliceAllocator& allocator,
    const UnownedStringSlice& line,
    List<UnownedStringSlice>& lineSlices,
    ArtifactDiagnostic& outDiagnostic)
{
    /* ERROR: tests/diagnostics/syntax-error-intrinsic.slang:13: '@' : unexpected token */

    if (lineSlices.getCount() < 4)
    {
        return SLANG_FAIL;
    }
    {
        const UnownedStringSlice severitySlice = lineSlices[0].trim();

        outDiagnostic.severity = ArtifactDiagnostic::Severity::Error;
        if (severitySlice.caseInsensitiveEquals(UnownedStringSlice::fromLiteral("warning")))
        {
            outDiagnostic.severity = ArtifactDiagnostic::Severity::Warning;
        }
    }

    outDiagnostic.filePath = allocator.allocate(lineSlices[1]);

    SLANG_RETURN_ON_FAIL(StringUtil::parseInt(lineSlices[2], outDiagnostic.location.line));
    outDiagnostic.text = allocator.allocate(lineSlices[3].begin(), line.end());
    return SLANG_OK;
}

SlangResult GlslangDownstreamCompiler::compile(
    const CompileOptions& inOptions,
    IArtifact** outArtifact)
{
    if (!isVersionCompatible(inOptions))
    {
        // Not possible to compile with this version of the interface.
        return SLANG_E_NOT_IMPLEMENTED;
    }

    CompileOptions options = getCompatibleVersion(&inOptions);

    // This compiler can only handle a single artifact
    if (options.sourceArtifacts.count != 1)
    {
        return SLANG_FAIL;
    }

    IArtifact* sourceArtifact = options.sourceArtifacts[0];

    if (options.targetType != SLANG_SPIRV)
    {
        SLANG_ASSERT(!"Can only compile to SPIR-V");
        return SLANG_FAIL;
    }

    StringBuilder diagnosticOutput;
    auto diagnosticOutputFunc = [](void const* data, size_t size, void* userData)
    { (*(StringBuilder*)userData).append((char const*)data, (char const*)data + size); };
    List<uint8_t> spirv;
    auto outputFunc = [](void const* data, size_t size, void* userData)
    { ((List<uint8_t>*)userData)->addRange((uint8_t*)data, size); };

    ComPtr<ISlangBlob> sourceBlob;
    SLANG_RETURN_ON_FAIL(sourceArtifact->loadBlob(ArtifactKeep::Yes, sourceBlob.writeRef()));

    String sourcePath = ArtifactUtil::findPath(sourceArtifact);

    glslang_CompileRequest_1_2 request;
    memset(&request, 0, sizeof(request));
    request.sizeInBytes = sizeof(request);

    switch (options.sourceLanguage)
    {
    case SLANG_SOURCE_LANGUAGE_GLSL:
        request.action = GLSLANG_ACTION_COMPILE_GLSL_TO_SPIRV;
        break;
    case SLANG_SOURCE_LANGUAGE_SPIRV:
        request.action = GLSLANG_ACTION_OPTIMIZE_SPIRV;
        break;
    default:
        SLANG_ASSERT(!"Can only handle GLSL or SPIR-V as input.");
        return SLANG_FAIL;
    }

    request.sourcePath = sourcePath.getBuffer();

    request.slangStage = options.stage;

    const char* inputBegin = (const char*)sourceBlob->getBufferPointer();
    request.inputBegin = inputBegin;
    request.inputEnd = inputBegin + sourceBlob->getBufferSize();

    // Find the SPIR-V version if set
    SemanticVersion spirvVersion;
    for (const auto& capabilityVersion : options.requiredCapabilityVersions)
    {
        if (capabilityVersion.kind == DownstreamCompileOptions::CapabilityVersion::Kind::SPIRV)
        {
            if (capabilityVersion.version > spirvVersion)
            {
                spirvVersion = capabilityVersion.version;
            }
        }
    }

    request.spirvVersion.major = spirvVersion.m_major;
    request.spirvVersion.minor = spirvVersion.m_minor;
    request.spirvVersion.patch = spirvVersion.m_patch;

    request.outputFunc = outputFunc;
    request.outputUserData = &spirv;

    request.diagnosticFunc = diagnosticOutputFunc;
    request.diagnosticUserData = &diagnosticOutput;

    request.optimizationLevel = (unsigned)options.optimizationLevel;
    request.debugInfoType = (unsigned)options.debugInfoType;

    request.entryPointName = options.entryPointName.begin();

    const SlangResult invokeResult = _invoke(request);

    auto artifact = ArtifactUtil::createArtifactForCompileTarget(options.targetType);

    auto diagnostics = ArtifactDiagnostics::create();

    // Set the diagnostics result
    diagnostics->setResult(invokeResult);

    ArtifactUtil::addAssociated(artifact, diagnostics);

    if (SLANG_FAILED(invokeResult))
    {
        diagnostics->setRaw(SliceUtil::asCharSlice(diagnosticOutput));

        SliceAllocator allocator;

        SlangResult diagnosticParseRes = ArtifactDiagnosticUtil::parseColonDelimitedDiagnostics(
            allocator,
            diagnosticOutput.getUnownedSlice(),
            1,
            _parseDiagnosticLine,
            diagnostics);
        SLANG_UNUSED(diagnosticParseRes);

        diagnostics->requireErrorDiagnostic();
    }
    else
    {
        artifact->addRepresentationUnknown(ListBlob::moveCreate(spirv));
    }

    *outArtifact = artifact.detach();
    return SLANG_OK;
}

SlangResult GlslangDownstreamCompiler::validate(const uint32_t* contents, int contentsSize)
{
    if (m_validate == nullptr)
    {
        return SLANG_FAIL;
    }

    if (m_validate(contents, contentsSize))
    {
        return SLANG_OK;
    }
    return SLANG_FAIL;
}

SlangResult GlslangDownstreamCompiler::disassembleWithResult(
    const uint32_t* contents,
    int contentsSize,
    String& outString)
{
    if (m_disassembleWithResult == nullptr)
    {
        return SLANG_FAIL;
    }

    char* resultString = nullptr;
    if (m_disassembleWithResult(contents, contentsSize, &resultString))
    {
        if (resultString)
        {
            outString = String(resultString);
            return SLANG_OK;
        }
    }
    return SLANG_FAIL;
}

SlangResult GlslangDownstreamCompiler::disassemble(const uint32_t* contents, int contentsSize)
{
    if (m_disassemble == nullptr)
    {
        return SLANG_FAIL;
    }

    if (m_disassemble(contents, contentsSize))
    {
        return SLANG_OK;
    }
    return SLANG_FAIL;
}

SlangResult GlslangDownstreamCompiler::link(
    const uint32_t** modules,
    const uint32_t* moduleSizes,
    const uint32_t moduleCount,
    IArtifact** outArtifact)
{
    glslang_LinkRequest request;
    memset(&request, 0, sizeof(request));

    request.modules = modules;
    request.moduleSizes = moduleSizes;
    request.moduleCount = moduleCount;

    if (!m_link(&request))
    {
        return SLANG_FAIL;
    }

    auto artifact = ArtifactUtil::createArtifactForCompileTarget(SLANG_SPIRV);
    artifact->addRepresentationUnknown(
        Slang::RawBlob::create(request.linkResult, request.linkResultSize * sizeof(uint32_t)));

    *outArtifact = artifact.detach();
    return SLANG_OK;
}

bool GlslangDownstreamCompiler::canConvert(const ArtifactDesc& from, const ArtifactDesc& to)
{
    // Can only disassemble blobs that are SPIR-V
    return ArtifactDescUtil::isDisassembly(from, to) &&
           ((from.payload == ArtifactPayload::SPIRV) ||
            (from.payload == ArtifactPayload::WGSL_SPIRV));
}

SlangResult GlslangDownstreamCompiler::convert(
    IArtifact* from,
    const ArtifactDesc& to,
    IArtifact** outArtifact)
{
    if (!canConvert(from->getDesc(), to))
    {
        return SLANG_FAIL;
    }

    ComPtr<ISlangBlob> blob;
    SLANG_RETURN_ON_FAIL(from->loadBlob(ArtifactKeep::No, blob.writeRef()));

    StringBuilder builder;

    auto outputFunc = [](void const* data, size_t size, void* userData)
    { (*(StringBuilder*)userData).append((char const*)data, (char const*)data + size); };

    glslang_CompileRequest_1_2 request;
    memset(&request, 0, sizeof(request));
    request.sizeInBytes = sizeof(request);

    request.action = GLSLANG_ACTION_DISSASSEMBLE_SPIRV;

    request.sourcePath = nullptr;

    char* blobData = (char*)blob->getBufferPointer();

    request.inputBegin = blobData;
    request.inputEnd = blobData + blob->getBufferSize();

    request.outputFunc = outputFunc;
    request.outputUserData = &builder;

    SLANG_RETURN_ON_FAIL(_invoke(request));

    auto disassemblyBlob = StringBlob::moveCreate(builder);

    auto artifact = ArtifactUtil::createArtifact(to);
    artifact->addRepresentationUnknown(disassemblyBlob);

    *outArtifact = artifact.detach();

    return SLANG_OK;
}

SlangResult GlslangDownstreamCompiler::getVersionString(slang::IBlob** outVersionString)
{
    uint64_t timestamp;
    if (m_compile_1_1)
    {
        timestamp = SharedLibraryUtils::getSharedLibraryTimestamp((void*)m_compile_1_1);
    }
    else if (m_compile_1_0)
    {
        timestamp = SharedLibraryUtils::getSharedLibraryTimestamp((void*)m_compile_1_0);
    }
    else
    {
        return SLANG_FAIL;
    }

    auto timestampString = String(timestamp);
    ComPtr<ISlangBlob> version = StringBlob::create(timestampString.getBuffer());
    *outVersionString = version.detach();
    return SLANG_OK;
}

static SlangResult locateGlslangSpirvDownstreamCompiler(
    const String& path,
    ISlangSharedLibraryLoader* loader,
    DownstreamCompilerSet* set,
    SlangPassThrough compilerType)
{
    ComPtr<ISlangSharedLibrary> library;

#if SLANG_UNIX_FAMILY
    // On unix systems we need to ensure pthread is loaded first.
    // TODO(JS):
    // There is an argument that this should be performed through the loader....
    // NOTE! We don't currently load through a dependent library, as it is *assumed* something as
    // core as 'ptheads' isn't going to be distributed with the shader compiler.
    ComPtr<ISlangSharedLibrary> pthreadLibrary;
    DefaultSharedLibraryLoader::load(loader, path, "pthread", pthreadLibrary.writeRef());
    if (!pthreadLibrary.get())
    {
        DefaultSharedLibraryLoader::load(
            loader,
            path,
            "libpthread.so.0",
            pthreadLibrary.writeRef());
    }

#endif

    SLANG_RETURN_ON_FAIL(
        DownstreamCompilerUtil::loadSharedLibrary(path, loader, nullptr, "slang-glslang", library));

    SLANG_ASSERT(library);
    if (!library)
    {
        return SLANG_FAIL;
    }

    auto compiler = new GlslangDownstreamCompiler(compilerType);
    ComPtr<IDownstreamCompiler> compilerIntf(compiler);
    SLANG_RETURN_ON_FAIL(compiler->init(library));

    set->addCompiler(compilerIntf);
    return SLANG_OK;
}

SlangResult GlslangDownstreamCompilerUtil::locateCompilers(
    const String& path,
    ISlangSharedLibraryLoader* loader,
    DownstreamCompilerSet* set)
{
    return locateGlslangSpirvDownstreamCompiler(path, loader, set, SLANG_PASS_THROUGH_GLSLANG);
}

SlangResult SpirvOptDownstreamCompilerUtil::locateCompilers(
    const String& path,
    ISlangSharedLibraryLoader* loader,
    DownstreamCompilerSet* set)
{
    return locateGlslangSpirvDownstreamCompiler(path, loader, set, SLANG_PASS_THROUGH_SPIRV_OPT);
}

SlangResult SpirvDisDownstreamCompilerUtil::locateCompilers(
    const String& path,
    ISlangSharedLibraryLoader* loader,
    DownstreamCompilerSet* set)
{
    return locateGlslangSpirvDownstreamCompiler(path, loader, set, SLANG_PASS_THROUGH_SPIRV_DIS);
}

SlangResult SpirvLinkDownstreamCompilerUtil::locateCompilers(
    const String& path,
    ISlangSharedLibraryLoader* loader,
    DownstreamCompilerSet* set)
{
    return locateGlslangSpirvDownstreamCompiler(path, loader, set, SLANG_PASS_THROUGH_SPIRV_LINK);
}

#else // SLANG_ENABLE_GLSLANG_SUPPORT

/* static */ SlangResult GlslangDownstreamCompilerUtil::locateCompilers(
    const String& path,
    ISlangSharedLibraryLoader* loader,
    DownstreamCompilerSet* set)
{
    SLANG_UNUSED(path);
    SLANG_UNUSED(loader);
    SLANG_UNUSED(set);
    return SLANG_E_NOT_AVAILABLE;
}

#endif // SLANG_ENABLE_GLSLANG_SUPPORT

} // namespace Slang
