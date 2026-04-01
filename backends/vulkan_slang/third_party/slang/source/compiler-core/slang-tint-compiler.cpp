#include "slang-tint-compiler.h"

#include "../../external/slang-tint-headers/slang-tint.h"
#include "slang-artifact-associated-impl.h"

namespace Slang
{

class TintDownstreamCompiler : public DownstreamCompilerBase
{

public:
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

    SlangResult compile(IArtifact* const sourceArtifact, IArtifact** outArtifact);

    SlangResult init(ISlangSharedLibrary* library);

protected:
    ComPtr<ISlangSharedLibrary> m_sharedLibrary;

private:
    tint_CompileFunc m_compile;
    tint_FreeResultFunc m_freeResult;
};

SlangResult TintDownstreamCompiler::init(ISlangSharedLibrary* library)
{
    tint_CompileFunc compile = (tint_CompileFunc)library->findFuncByName("tint_compile");
    if (compile == nullptr)
    {
        return SLANG_FAIL;
    }

    tint_FreeResultFunc freeResult =
        (tint_FreeResultFunc)library->findFuncByName("tint_free_result");
    if (freeResult == nullptr)
    {
        return SLANG_FAIL;
    }

    m_sharedLibrary = library;
    m_desc = Desc(SLANG_PASS_THROUGH_TINT);
    m_compile = compile;
    m_freeResult = freeResult;
    return SLANG_OK;
}

SlangResult TintDownstreamCompilerUtil::locateCompilers(
    const String& path,
    ISlangSharedLibraryLoader* loader,
    DownstreamCompilerSet* set)
{
    ComPtr<ISlangSharedLibrary> library;
    SLANG_RETURN_ON_FAIL(
        DownstreamCompilerUtil::loadSharedLibrary(path, loader, nullptr, "slang-tint", library));
    SLANG_ASSERT(library);

    ComPtr<IDownstreamCompiler> compiler =
        ComPtr<IDownstreamCompiler>(new TintDownstreamCompiler());
    SLANG_RETURN_ON_FAIL(static_cast<TintDownstreamCompiler*>(compiler.get())->init(library));

    set->addCompiler(compiler);
    return SLANG_OK;
}

SlangResult TintDownstreamCompiler::compile(const CompileOptions& options, IArtifact** outArtifact)
{
    IArtifact* sourceArtifact = options.sourceArtifacts[0];
    return compile(sourceArtifact, outArtifact);
}

SlangResult TintDownstreamCompiler::compile(
    IArtifact* const sourceArtifact,
    IArtifact** outArtifact)
{
    tint_CompileRequest req = {};

    if (sourceArtifact == nullptr)
        return SLANG_FAIL;

    ComPtr<ISlangBlob> sourceBlob;
    SLANG_RETURN_FALSE_ON_FAIL(sourceArtifact->loadBlob(ArtifactKeep::Yes, sourceBlob.writeRef()));

    String wgslCode(
        (char*)sourceBlob->getBufferPointer(),
        (char*)sourceBlob->getBufferPointer() + sourceBlob->getBufferSize());
    req.wgslCode = wgslCode.begin();
    req.wgslCodeLength = wgslCode.getLength();

    tint_CompileResult result = {};
    SLANG_DEFER(m_freeResult(&result));
    bool compileSucceeded = m_compile(&req, &result) == 0;

    ComPtr<ISlangBlob> spirvBlob = RawBlob::create(result.buffer, result.bufferSize);
    result.buffer = nullptr;

    ComPtr<IArtifact> resultArtifact =
        ArtifactUtil::createArtifactForCompileTarget(SlangCompileTarget::SLANG_WGSL_SPIRV);
    auto diagnostics = ArtifactDiagnostics::create();
    diagnostics->setResult(compileSucceeded ? SLANG_OK : SLANG_FAIL);
    ArtifactUtil::addAssociated(resultArtifact, diagnostics);
    if (compileSucceeded)
    {
        resultArtifact->addRepresentationUnknown(spirvBlob);
    }
    else
    {
        diagnostics->setRaw(CharSlice(result.error));
        diagnostics->requireErrorDiagnostic();
    }

    *outArtifact = resultArtifact.detach();
    return SLANG_OK;
}

bool TintDownstreamCompiler::canConvert(const ArtifactDesc& from, const ArtifactDesc& to)
{
    return (from.payload == ArtifactPayload::WGSL) && (to.payload == ArtifactPayload::SPIRV);
}

SlangResult TintDownstreamCompiler::convert(
    IArtifact* from,
    const ArtifactDesc& to,
    IArtifact** outArtifact)
{
    if (!canConvert(from->getDesc(), to))
        return SLANG_FAIL;
    return compile(from, outArtifact);
}

SlangResult TintDownstreamCompiler::getVersionString(slang::IBlob** /* outVersionString */)
{
    // We just use Tint at whatever version is in our Dawn fork, so nobody should
    // depend on the particular version at the moment.
    return SLANG_FAIL;
}

} // namespace Slang
