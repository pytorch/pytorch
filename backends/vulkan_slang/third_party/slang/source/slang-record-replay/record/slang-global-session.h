#ifndef SLANG_GLOBAL_SESSION_H
#define SLANG_GLOBAL_SESSION_H

#include "../../core/slang-smart-pointer.h"
#include "record-manager.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang.h"

namespace SlangRecord
{
using namespace Slang;

class GlobalSessionRecorder : public RefObject, public slang::IGlobalSession
{
public:
    explicit GlobalSessionRecorder(
        const SlangGlobalSessionDesc* desc,
        slang::IGlobalSession* session);

    SLANG_REF_OBJECT_IUNKNOWN_ADD_REF
    SLANG_REF_OBJECT_IUNKNOWN_RELEASE

    SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(SlangUUID const& uuid, void** outObject)
        SLANG_OVERRIDE;

    // slang::IGlobalSession
    SLANG_NO_THROW SlangResult SLANG_MCALL
    createSession(slang::SessionDesc const& desc, slang::ISession** outSession) override;
    SLANG_NO_THROW SlangProfileID SLANG_MCALL findProfile(char const* name) override;
    SLANG_NO_THROW void SLANG_MCALL
    setDownstreamCompilerPath(SlangPassThrough passThrough, char const* path) override;
    SLANG_NO_THROW void SLANG_MCALL
    setDownstreamCompilerPrelude(SlangPassThrough inPassThrough, char const* prelude) override;
    SLANG_NO_THROW void SLANG_MCALL
    getDownstreamCompilerPrelude(SlangPassThrough inPassThrough, ISlangBlob** outPrelude) override;
    SLANG_NO_THROW const char* SLANG_MCALL getBuildTagString() override;
    SLANG_NO_THROW SlangResult SLANG_MCALL setDefaultDownstreamCompiler(
        SlangSourceLanguage sourceLanguage,
        SlangPassThrough defaultCompiler) override;
    SLANG_NO_THROW SlangPassThrough SLANG_MCALL
    getDefaultDownstreamCompiler(SlangSourceLanguage sourceLanguage) override;

    SLANG_NO_THROW void SLANG_MCALL
    setLanguagePrelude(SlangSourceLanguage inSourceLanguage, char const* prelude) override;
    SLANG_NO_THROW void SLANG_MCALL
    getLanguagePrelude(SlangSourceLanguage inSourceLanguage, ISlangBlob** outPrelude) override;

    SLANG_NO_THROW SlangResult SLANG_MCALL
    createCompileRequest(slang::ICompileRequest** outCompileRequest) override;

    SLANG_NO_THROW void SLANG_MCALL
    addBuiltins(char const* sourcePath, char const* sourceString) override;
    SLANG_NO_THROW void SLANG_MCALL
    setSharedLibraryLoader(ISlangSharedLibraryLoader* loader) override;
    SLANG_NO_THROW ISlangSharedLibraryLoader* SLANG_MCALL getSharedLibraryLoader() override;
    SLANG_NO_THROW SlangResult SLANG_MCALL
    checkCompileTargetSupport(SlangCompileTarget target) override;
    SLANG_NO_THROW SlangResult SLANG_MCALL
    checkPassThroughSupport(SlangPassThrough passThrough) override;

    SLANG_NO_THROW SlangResult SLANG_MCALL
    compileCoreModule(slang::CompileCoreModuleFlags flags) override;
    SLANG_NO_THROW SlangResult SLANG_MCALL
    loadCoreModule(const void* coreModule, size_t coreModuleSizeInBytes) override;
    SLANG_NO_THROW SlangResult SLANG_MCALL
    saveCoreModule(SlangArchiveType archiveType, ISlangBlob** outBlob) override;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL compileBuiltinModule(
        slang::BuiltinModuleName module,
        slang::CompileCoreModuleFlags flags) override;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL loadBuiltinModule(
        slang::BuiltinModuleName module,
        const void* moduleData,
        size_t sizeInBytes) override;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL saveBuiltinModule(
        slang::BuiltinModuleName module,
        SlangArchiveType archiveType,
        ISlangBlob** outBlob) override;

    SLANG_NO_THROW SlangCapabilityID SLANG_MCALL findCapability(char const* name) override;

    SLANG_NO_THROW void SLANG_MCALL setDownstreamCompilerForTransition(
        SlangCompileTarget source,
        SlangCompileTarget target,
        SlangPassThrough compiler) override;
    SLANG_NO_THROW SlangPassThrough SLANG_MCALL getDownstreamCompilerForTransition(
        SlangCompileTarget source,
        SlangCompileTarget target) override;
    SLANG_NO_THROW void SLANG_MCALL
    getCompilerElapsedTime(double* outTotalTime, double* outDownstreamTime) override;

    SLANG_NO_THROW SlangResult SLANG_MCALL setSPIRVCoreGrammar(char const* jsonPath) override;

    SLANG_NO_THROW SlangResult SLANG_MCALL parseCommandLineArguments(
        int argc,
        const char* const* argv,
        slang::SessionDesc* outSessionDesc,
        ISlangUnknown** outAllocation) override;

    SLANG_NO_THROW SlangResult SLANG_MCALL
    getSessionDescDigest(slang::SessionDesc* sessionDesc, ISlangBlob** outBlob) override;

    RecordManager* getRecordManager() { return m_recordManager.get(); }

private:
    SLANG_FORCE_INLINE slang::IGlobalSession* asExternal(GlobalSessionRecorder* session)
    {
        return static_cast<slang::IGlobalSession*>(session);
    }

    Slang::ComPtr<slang::IGlobalSession> m_actualGlobalSession;

    // we will create one record file per IGlobalSession.
    // We don't try to reproduce the user application's threading model, because it requires lots of
    // effort and it's not necessary. Instead, we record all the compilation jobs associated with
    // the session in the same record file, so that during replay, those jobs will be executed
    // sequentially. This might violate the user application's threading model, because those jobs
    // might be executed in different threads. But it's not a big problem, because slang doesn't
    // allow multiple threads to access the same session at the same time. So even if there is one
    // session used by multiple threads, those threads will execute the compile jobs sequentially.
    Slang::RefPtr<RecordManager> m_recordManager;
    uint64_t m_globalSessionHandle = 0;
};
} // namespace SlangRecord

#endif
