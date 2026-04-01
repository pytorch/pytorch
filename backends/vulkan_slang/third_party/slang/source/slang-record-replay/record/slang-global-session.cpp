#include "slang-global-session.h"

#include "../../slang/slang-compiler.h"
#include "../util/record-utility.h"
#include "slang-filesystem.h"
#include "slang-session.h"

namespace SlangRecord
{
// constructor is called in slang_createGlobalSession
GlobalSessionRecorder::GlobalSessionRecorder(
    const SlangGlobalSessionDesc* desc,
    slang::IGlobalSession* session)
    : m_actualGlobalSession(session)
{
    SLANG_RECORD_ASSERT(m_actualGlobalSession != nullptr);

    m_globalSessionHandle =
        reinterpret_cast<SlangRecord::AddressFormat>(m_actualGlobalSession.get());
    m_recordManager = new RecordManager(m_globalSessionHandle);

    // We will use the address of the global session as the filename for the record manager
    // to make it unique for each global session.
    // record slang::createGlobalSession

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::CreateGlobalSession,
            g_globalFunctionHandle);
        recorder->recordStruct(*desc);
        recorder = m_recordManager->endMethodRecord();
    }

    recorder->recordAddress(m_actualGlobalSession);
    m_recordManager->apendOutput();
}

SLANG_NO_THROW SlangResult SLANG_MCALL
GlobalSessionRecorder::queryInterface(SlangUUID const& uuid, void** outObject)
{
    if (uuid == Session::getTypeGuid())
    {
        // no add-ref here, the query will cause the inner session to handle the add-ref.
        this->m_actualGlobalSession->queryInterface(uuid, outObject);
        return SLANG_OK;
    }

    if (uuid == ISlangUnknown::getTypeGuid() && uuid == IGlobalSession::getTypeGuid())
    {
        addReference();
        *outObject = static_cast<slang::IGlobalSession*>(this);
        return SLANG_OK;
    }

    return SLANG_E_NO_INTERFACE;
}

SLANG_NO_THROW SlangResult SLANG_MCALL
GlobalSessionRecorder::createSession(slang::SessionDesc const& desc, slang::ISession** outSession)
{
    setLogLevel();
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    slang::ISession* actualSession = nullptr;

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_createSession,
            m_globalSessionHandle);
        recorder->recordStruct(desc);
        recorder = m_recordManager->endMethodRecord();
    }

    SlangResult res = m_actualGlobalSession->createSession(desc, &actualSession);

    { // record output
        recorder->recordAddress(actualSession);
        m_recordManager->apendOutput();
    }

    if (actualSession != nullptr)
    {
        // reset the file system to our record file system. After createSession() call,
        // the Linkage will set to user provided file system or slang default file system.
        // We need to reset it to our record file system
        Slang::Linkage* linkage = static_cast<Linkage*>(actualSession);
        FileSystemRecorder* fileSystemRecord =
            new FileSystemRecorder(linkage->getFileSystemExt(), m_recordManager.get());

        Slang::ComPtr<FileSystemRecorder> resultFileSystemRecorder(fileSystemRecord);
        linkage->setFileSystem(resultFileSystemRecorder.detach());

        SessionRecorder* sessionRecord = new SessionRecorder(actualSession, m_recordManager.get());
        Slang::ComPtr<SessionRecorder> result(sessionRecord);
        *outSession = result.detach();
    }

    return res;
}

SLANG_NO_THROW SlangProfileID SLANG_MCALL GlobalSessionRecorder::findProfile(char const* name)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_findProfile,
            m_globalSessionHandle);
        recorder->recordString(name);
        recorder = m_recordManager->endMethodRecord();
    }

    SlangProfileID profileId = m_actualGlobalSession->findProfile(name);
    return profileId;
}

SLANG_NO_THROW void SLANG_MCALL
GlobalSessionRecorder::setDownstreamCompilerPath(SlangPassThrough passThrough, char const* path)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_setDownstreamCompilerPath,
            m_globalSessionHandle);
        recorder->recordEnumValue(passThrough);
        recorder->recordString(path);
        m_recordManager->endMethodRecord();
    }

    m_actualGlobalSession->setDownstreamCompilerPath(passThrough, path);
}

SLANG_NO_THROW void SLANG_MCALL GlobalSessionRecorder::setDownstreamCompilerPrelude(
    SlangPassThrough inPassThrough,
    char const* prelude)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_setDownstreamCompilerPrelude,
            m_globalSessionHandle);
        recorder->recordEnumValue(inPassThrough);
        recorder->recordString(prelude);
        m_recordManager->endMethodRecord();
    }

    m_actualGlobalSession->setDownstreamCompilerPrelude(inPassThrough, prelude);
}

SLANG_NO_THROW void SLANG_MCALL GlobalSessionRecorder::getDownstreamCompilerPrelude(
    SlangPassThrough inPassThrough,
    ISlangBlob** outPrelude)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_getDownstreamCompilerPrelude,
            m_globalSessionHandle);
        recorder->recordEnumValue(inPassThrough);
        recorder = m_recordManager->endMethodRecord();
    }

    m_actualGlobalSession->getDownstreamCompilerPrelude(inPassThrough, outPrelude);

    {
        recorder->recordAddress(*outPrelude);
        m_recordManager->apendOutput();
    }
}

SLANG_NO_THROW const char* SLANG_MCALL GlobalSessionRecorder::getBuildTagString()
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    // No need to record this function. It's just a query function and it won't impact the internal
    // state.
    const char* resStr = m_actualGlobalSession->getBuildTagString();
    return resStr;
}

SLANG_NO_THROW SlangResult SLANG_MCALL GlobalSessionRecorder::setDefaultDownstreamCompiler(
    SlangSourceLanguage sourceLanguage,
    SlangPassThrough defaultCompiler)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_setDefaultDownstreamCompiler,
            m_globalSessionHandle);
        recorder->recordEnumValue(sourceLanguage);
        recorder->recordEnumValue(defaultCompiler);
        recorder = m_recordManager->endMethodRecord();
    }

    SlangResult res =
        m_actualGlobalSession->setDefaultDownstreamCompiler(sourceLanguage, defaultCompiler);
    return res;
}

SLANG_NO_THROW SlangPassThrough SLANG_MCALL
GlobalSessionRecorder::getDefaultDownstreamCompiler(SlangSourceLanguage sourceLanguage)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_getDefaultDownstreamCompiler,
            m_globalSessionHandle);
        recorder->recordEnumValue(sourceLanguage);
        recorder = m_recordManager->endMethodRecord();
    }

    SlangPassThrough passThrough =
        m_actualGlobalSession->getDefaultDownstreamCompiler(sourceLanguage);
    return passThrough;
}

SLANG_NO_THROW void SLANG_MCALL
GlobalSessionRecorder::setLanguagePrelude(SlangSourceLanguage inSourceLanguage, char const* prelude)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_setLanguagePrelude,
            m_globalSessionHandle);
        recorder->recordEnumValue(inSourceLanguage);
        recorder->recordString(prelude);
        recorder = m_recordManager->endMethodRecord();
    }

    m_actualGlobalSession->setLanguagePrelude(inSourceLanguage, prelude);
}

SLANG_NO_THROW void SLANG_MCALL GlobalSessionRecorder::getLanguagePrelude(
    SlangSourceLanguage inSourceLanguage,
    ISlangBlob** outPrelude)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_getLanguagePrelude,
            m_globalSessionHandle);
        recorder->recordEnumValue(inSourceLanguage);
        recorder = m_recordManager->endMethodRecord();
    }

    m_actualGlobalSession->getLanguagePrelude(inSourceLanguage, outPrelude);

    {
        recorder->recordAddress(*outPrelude);
        m_recordManager->apendOutput();
    }
}

SLANG_NO_THROW SlangResult SLANG_MCALL
GlobalSessionRecorder::createCompileRequest(slang::ICompileRequest** outCompileRequest)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_createCompileRequest,
            m_globalSessionHandle);
        recorder = m_recordManager->endMethodRecord();
    }

    SLANG_ALLOW_DEPRECATED_BEGIN
    SlangResult res = m_actualGlobalSession->createCompileRequest(outCompileRequest);
    SLANG_ALLOW_DEPRECATED_END

    {
        recorder->recordAddress(*outCompileRequest);
        m_recordManager->apendOutput();
    }

    return res;
}

SLANG_NO_THROW void SLANG_MCALL
GlobalSessionRecorder::addBuiltins(char const* sourcePath, char const* sourceString)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);
    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_addBuiltins,
            m_globalSessionHandle);
        recorder->recordString(sourcePath);
        recorder->recordString(sourceString);
        recorder = m_recordManager->endMethodRecord();
    }

    m_actualGlobalSession->addBuiltins(sourcePath, sourceString);
}

SLANG_NO_THROW void SLANG_MCALL
GlobalSessionRecorder::setSharedLibraryLoader(ISlangSharedLibraryLoader* loader)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);
    // TODO: Not sure if we need to record this function. Because this functions is something like
    // the file system override, it's provided by user code. So capturing it makes no sense. The
    // only way is to wrapper this interface by our own implementation, and record it there.
    m_actualGlobalSession->setSharedLibraryLoader(loader);
}

SLANG_NO_THROW ISlangSharedLibraryLoader* SLANG_MCALL
GlobalSessionRecorder::getSharedLibraryLoader()
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_getSharedLibraryLoader,
            m_globalSessionHandle);
        recorder = m_recordManager->endMethodRecord();
    }

    ISlangSharedLibraryLoader* loader = m_actualGlobalSession->getSharedLibraryLoader();

    {
        recorder->recordAddress(loader);
        m_recordManager->apendOutput();
    }
    return loader;
}

SLANG_NO_THROW SlangResult SLANG_MCALL
GlobalSessionRecorder::checkCompileTargetSupport(SlangCompileTarget target)
{
    // No need to record this function. It's just a query function and it won't impact the internal
    // state.
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);
    SlangResult res = m_actualGlobalSession->checkCompileTargetSupport(target);
    return res;
}

SLANG_NO_THROW SlangResult SLANG_MCALL
GlobalSessionRecorder::checkPassThroughSupport(SlangPassThrough passThrough)
{
    // No need to record this function. It's just a query function and it won't impact the internal
    // state.
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);
    SlangResult res = m_actualGlobalSession->checkPassThroughSupport(passThrough);
    return res;
}

SLANG_NO_THROW SlangResult SLANG_MCALL
GlobalSessionRecorder::compileCoreModule(slang::CompileCoreModuleFlags flags)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_compileCoreModule,
            m_globalSessionHandle);
        recorder->recordEnumValue(flags);
        m_recordManager->endMethodRecord();
    }

    SlangResult res = m_actualGlobalSession->compileCoreModule(flags);
    return res;
}

SLANG_NO_THROW SlangResult SLANG_MCALL
GlobalSessionRecorder::loadCoreModule(const void* coreModule, size_t coreModuleSizeInBytes)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_loadCoreModule,
            m_globalSessionHandle);
        recorder->recordPointer(coreModule, false, coreModuleSizeInBytes);
        m_recordManager->endMethodRecord();
    }

    SlangResult res = m_actualGlobalSession->loadCoreModule(coreModule, coreModuleSizeInBytes);
    return res;
}

SLANG_NO_THROW SlangResult SLANG_MCALL
GlobalSessionRecorder::saveCoreModule(SlangArchiveType archiveType, ISlangBlob** outBlob)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_saveCoreModule,
            m_globalSessionHandle);
        recorder->recordEnumValue(archiveType);
        recorder = m_recordManager->endMethodRecord();
    }

    SlangResult res = m_actualGlobalSession->saveCoreModule(archiveType, outBlob);

    {
        recorder->recordAddress(*outBlob);
        m_recordManager->apendOutput();
    }
    return res;
}

SLANG_NO_THROW SlangResult SLANG_MCALL GlobalSessionRecorder::compileBuiltinModule(
    slang::BuiltinModuleName module,
    slang::CompileCoreModuleFlags flags)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_compileBuiltinModule,
            m_globalSessionHandle);
        recorder->recordEnumValue(module);
        recorder->recordEnumValue(flags);
        m_recordManager->endMethodRecord();
    }

    SlangResult res = m_actualGlobalSession->compileBuiltinModule(module, flags);
    return res;
}

SLANG_NO_THROW SlangResult SLANG_MCALL GlobalSessionRecorder::loadBuiltinModule(
    slang::BuiltinModuleName module,
    const void* moduleData,
    size_t sizeInBytes)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_loadBuiltinModule,
            m_globalSessionHandle);
        recorder->recordEnumValue(module);
        recorder->recordPointer(moduleData, false, sizeInBytes);
        m_recordManager->endMethodRecord();
    }

    SlangResult res = m_actualGlobalSession->loadBuiltinModule(module, moduleData, sizeInBytes);
    return res;
}
SLANG_NO_THROW SlangResult SLANG_MCALL GlobalSessionRecorder::saveBuiltinModule(
    slang::BuiltinModuleName module,
    SlangArchiveType archiveType,
    ISlangBlob** outBlob)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_saveBuiltinModule,
            m_globalSessionHandle);
        recorder->recordEnumValue(module);
        recorder->recordEnumValue(archiveType);
        recorder = m_recordManager->endMethodRecord();
    }
    SlangResult res = m_actualGlobalSession->saveBuiltinModule(module, archiveType, outBlob);
    {
        recorder->recordAddress(*outBlob);
        m_recordManager->apendOutput();
    }
    return res;
}

SLANG_NO_THROW SlangCapabilityID SLANG_MCALL GlobalSessionRecorder::findCapability(char const* name)
{
    // No need to record this function. It's just a query function and it won't impact the internal
    // state.
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);
    SlangCapabilityID capId = m_actualGlobalSession->findCapability(name);
    return capId;
}

SLANG_NO_THROW void SLANG_MCALL GlobalSessionRecorder::setDownstreamCompilerForTransition(
    SlangCompileTarget source,
    SlangCompileTarget target,
    SlangPassThrough compiler)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_setDownstreamCompilerForTransition,
            m_globalSessionHandle);
        recorder->recordEnumValue(source);
        recorder->recordEnumValue(target);
        recorder->recordEnumValue(compiler);
        m_recordManager->endMethodRecord();
    }

    m_actualGlobalSession->setDownstreamCompilerForTransition(source, target, compiler);
}

SLANG_NO_THROW SlangPassThrough SLANG_MCALL
GlobalSessionRecorder::getDownstreamCompilerForTransition(
    SlangCompileTarget source,
    SlangCompileTarget target)
{
    // No need to record this function. It's just a query function and it won't impact the internal
    // state.
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);
    SlangPassThrough passThrough =
        m_actualGlobalSession->getDownstreamCompilerForTransition(source, target);
    return passThrough;
}

SLANG_NO_THROW void SLANG_MCALL
GlobalSessionRecorder::getCompilerElapsedTime(double* outTotalTime, double* outDownstreamTime)
{
    // No need to record this function. It's just a query function and it won't impact the internal
    // state.
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);
    m_actualGlobalSession->getCompilerElapsedTime(outTotalTime, outDownstreamTime);
}

SLANG_NO_THROW SlangResult SLANG_MCALL
GlobalSessionRecorder::setSPIRVCoreGrammar(char const* jsonPath)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_setSPIRVCoreGrammar,
            m_globalSessionHandle);
        recorder->recordString(jsonPath);
        m_recordManager->endMethodRecord();
    }

    SlangResult res = m_actualGlobalSession->setSPIRVCoreGrammar(jsonPath);
    return res;
}

SLANG_NO_THROW SlangResult SLANG_MCALL GlobalSessionRecorder::parseCommandLineArguments(
    int argc,
    const char* const* argv,
    slang::SessionDesc* outSessionDesc,
    ISlangUnknown** outAllocation)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_parseCommandLineArguments,
            m_globalSessionHandle);
        recorder->recordInt32(argc);
        recorder->recordStringArray(argv, argc);
        recorder = m_recordManager->endMethodRecord();
    }

    SlangResult res =
        m_actualGlobalSession->parseCommandLineArguments(argc, argv, outSessionDesc, outAllocation);

    {
        recorder->recordAddress(outSessionDesc);
        recorder->recordAddress(*outAllocation);
        m_recordManager->apendOutput();
    }
    return res;
}

SLANG_NO_THROW SlangResult SLANG_MCALL
GlobalSessionRecorder::getSessionDescDigest(slang::SessionDesc* sessionDesc, ISlangBlob** outBlob)
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualGlobalSession.get(), __PRETTY_FUNCTION__);

    ParameterRecorder* recorder{};
    {
        recorder = m_recordManager->beginMethodRecord(
            ApiCallId::IGlobalSession_getSessionDescDigest,
            m_globalSessionHandle);
        recorder->recordStruct(*sessionDesc);
        recorder = m_recordManager->endMethodRecord();
    }

    SlangResult res = m_actualGlobalSession->getSessionDescDigest(sessionDesc, outBlob);

    {
        recorder->recordAddress(*outBlob);
        m_recordManager->apendOutput();
    }
    return res;
}
} // namespace SlangRecord
