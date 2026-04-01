#include "replay-consumer.h"

#include "../../core/slang-string-util.h"

namespace SlangRecord
{

#define OutputObjectSanityCheck(objId)                        \
    do                                                        \
    {                                                         \
        if (m_objectMap.tryGetValue((objId)))                 \
        {                                                     \
            slangRecordLog(                                   \
                LogLevel::Error,                              \
                "Output object 0x%X already exists! %s:%d\n", \
                objId,                                        \
                __PRETTY_FUNCTION__,                          \
                __LINE__);                                    \
            std::abort();                                     \
        }                                                     \
    } while (0)

#define InputObjectSanityCheck(objId)                                    \
    do                                                                   \
    {                                                                    \
        if (!m_objectMap.tryGetValue((objId)))                           \
        {                                                                \
            slangRecordLog(                                              \
                LogLevel::Error,                                         \
                "Input object 0x%X has not been allocated yet! %s:%d\n", \
                objId,                                                   \
                __PRETTY_FUNCTION__,                                     \
                __LINE__);                                               \
            std::abort();                                                \
        }                                                                \
    } while (0)

#define FAIL_WITH_LOG(funcName)                              \
    do                                                       \
    {                                                        \
        if (SLANG_FAILED(res))                               \
        {                                                    \
            slangRecordLog(                                  \
                LogLevel::Error,                             \
                #funcName " fails, ret: 0x%X, this: 0x%X\n", \
                res,                                         \
                objectId);                                   \
        }                                                    \
    } while (0)

SlangResult CommonInterfaceReplayer::getLayout(
    ObjectID objectId,
    SlangInt targetIndex,
    ObjectID outDiagnosticsId,
    ObjectID retProgramLayoutId)
{
    InputObjectSanityCheck(objectId);

    slang::IComponentType* pObj = getObjectPointer(objectId);
    slang::IBlob* outDiagnostics{};
    slang::ProgramLayout* outProgramLayout{};

    SlangResult res = SLANG_OK;
    outProgramLayout = pObj->getLayout(targetIndex, &outDiagnostics);

    if (outProgramLayout)
    {
        m_objectMap.addIfNotExists(retProgramLayoutId, outProgramLayout);
    }
    else
    {
        res = SLANG_FAIL;
    }
    return res;
}

SlangResult CommonInterfaceReplayer::getEntryPointCode(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outCodeId,
    ObjectID outDiagnosticsId)
{
    InputObjectSanityCheck(objectId);

    slang::IComponentType* pObj = getObjectPointer(objectId);
    slang::IBlob* outCode{};
    slang::IBlob* outDiagnostics{};

    SlangResult res =
        pObj->getEntryPointCode(entryPointIndex, targetIndex, &outCode, &outDiagnostics);

    if (outCode && SLANG_SUCCEEDED(res))
    {
        m_objectMap.addIfNotExists(outCodeId, outCode);
    }

    ReplayConsumer::printDiagnosticMessage(outDiagnostics);
    return res;
}

SlangResult CommonInterfaceReplayer::getTargetCode(
    ObjectID objectId,
    SlangInt targetIndex,
    ObjectID outCodeId,
    ObjectID outDiagnosticsId)
{
    InputObjectSanityCheck(objectId);

    slang::IComponentType* pObj = getObjectPointer(objectId);
    slang::IBlob* outCode{};
    slang::IBlob* outDiagnostics{};

    SlangResult res = pObj->getTargetCode(targetIndex, &outCode, &outDiagnostics);

    if (outCode && SLANG_SUCCEEDED(res))
    {
        m_objectMap.addIfNotExists(outCodeId, outCode);
    }

    ReplayConsumer::printDiagnosticMessage(outDiagnostics);
    return res;
}

SlangResult CommonInterfaceReplayer::getResultAsFileSystem(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outFileSystemId)
{
    InputObjectSanityCheck(objectId);
    OutputObjectSanityCheck(outFileSystemId);

    slang::IComponentType* pObj = getObjectPointer(objectId);
    ISlangMutableFileSystem* outFileSystem{};
    SlangResult res = pObj->getResultAsFileSystem(entryPointIndex, targetIndex, &outFileSystem);

    if (outFileSystem && SLANG_SUCCEEDED(res))
    {
        m_objectMap.add(outFileSystemId, outFileSystem);
    }
    return res;
}

SlangResult CommonInterfaceReplayer::getEntryPointHash(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outHashId)
{
    InputObjectSanityCheck(objectId);

    SlangResult res = SLANG_OK;
    slang::IComponentType* pObj = getObjectPointer(objectId);
    slang::IBlob* outHash{};
    pObj->getEntryPointHash(entryPointIndex, targetIndex, &outHash);

    if (outHash)
    {
        if (outHash->getBufferSize())
        {
            uint8_t* buffer = (uint8_t*)outHash->getBufferPointer();
            Slang::StringBuilder strBuilder;
            strBuilder << "callIdx: " << m_globalCounter << ", entrypoint: " << entryPointIndex
                       << ", target: " << targetIndex << ", hash: ";
            m_globalCounter++;

            for (size_t i = 0; i < outHash->getBufferSize(); i++)
            {
                strBuilder << Slang::StringUtil::makeStringWithFormat("%.2X", buffer[i]);
            }
            slangRecordLog(LogLevel::Verbose, "%s\n", strBuilder.begin());
        }
        else
        {
            res = SLANG_FAIL;
        }
    }
    else
    {
        res = SLANG_FAIL;
    }
    return res;
}

SlangResult CommonInterfaceReplayer::specialize(
    ObjectID objectId,
    slang::SpecializationArg const* specializationArgs,
    SlangInt specializationArgCount,
    ObjectID outSpecializedComponentTypeId,
    ObjectID outDiagnosticsId)
{
    InputObjectSanityCheck(objectId);

    for (SlangInt i = 0; i < specializationArgCount; i++)
    {
        if (specializationArgs[i].type != nullptr)
        {
            slangRecordLog(
                LogLevel::Error,
                "We only support nullptr for 'type' as reflection is not supported yet, %s:%d\n",
                objectId,
                __PRETTY_FUNCTION__,
                __LINE__);
            return SLANG_FAIL;
        }
    }

    slang::IComponentType* pObj = getObjectPointer(objectId);
    slang::IComponentType* outSpecializedComponentType{};
    slang::IBlob* outDiagnostics{};

    SlangResult res = pObj->specialize(
        specializationArgs,
        specializationArgCount,
        &outSpecializedComponentType,
        &outDiagnostics);
    if (outSpecializedComponentType && SLANG_SUCCEEDED(res))
    {
        m_objectMap.addIfNotExists(outSpecializedComponentTypeId, outSpecializedComponentType);
    }

    ReplayConsumer::printDiagnosticMessage(outDiagnostics);
    return res;
}

SlangResult CommonInterfaceReplayer::link(
    ObjectID objectId,
    ObjectID outLinkedComponentTypeId,
    ObjectID outDiagnosticsId)
{
    InputObjectSanityCheck(objectId);

    slang::IComponentType* pObj = getObjectPointer(objectId);
    slang::IComponentType* outLinkedComponentType{};
    slang::IBlob* outDiagnostics{};

    SlangResult res = pObj->link(&outLinkedComponentType, &outDiagnostics);

    if (outLinkedComponentType && SLANG_SUCCEEDED(res))
    {
        m_objectMap.addIfNotExists(outLinkedComponentTypeId, outLinkedComponentType);
    }

    ReplayConsumer::printDiagnosticMessage(outDiagnostics);
    return res;
}

SlangResult CommonInterfaceReplayer::getEntryPointHostCallable(
    ObjectID objectId,
    int entryPointIndex,
    int targetIndex,
    ObjectID outSharedLibraryId,
    ObjectID outDiagnosticsId)
{
    InputObjectSanityCheck(objectId);

    slang::IComponentType* pObj = getObjectPointer(objectId);
    ISlangSharedLibrary* outSharedLib{};
    slang::IBlob* outDiagnosticsBlob{};

    SlangResult res = pObj->getEntryPointHostCallable(
        entryPointIndex,
        targetIndex,
        &outSharedLib,
        &outDiagnosticsBlob);

    if (outSharedLib && SLANG_SUCCEEDED(res))
    {
        m_objectMap.addIfNotExists(outSharedLibraryId, outSharedLib);
    }

    ReplayConsumer::printDiagnosticMessage(outDiagnosticsBlob);
    return res;
}

SlangResult CommonInterfaceReplayer::renameEntryPoint(
    ObjectID objectId,
    const char* newName,
    ObjectID outEntryPointId)
{
    InputObjectSanityCheck(objectId);

    slang::IComponentType* pObj = getObjectPointer(objectId);
    slang::IComponentType* outEntryPoint{};
    SlangResult res = pObj->renameEntryPoint(newName, &outEntryPoint);

    if (outEntryPoint && SLANG_SUCCEEDED(res))
    {
        m_objectMap.addIfNotExists(outEntryPointId, outEntryPoint);
    }
    return res;
}

SlangResult CommonInterfaceReplayer::linkWithOptions(
    ObjectID objectId,
    ObjectID outLinkedComponentTypeId,
    uint32_t compilerOptionEntryCount,
    slang::CompilerOptionEntry* compilerOptionEntries,
    ObjectID outDiagnosticsId)
{
    InputObjectSanityCheck(objectId);

    slang::IComponentType* pObj = getObjectPointer(objectId);
    slang::IComponentType* outLinkedComponentType{};
    slang::IBlob* outDiagnostics{};

    SlangResult res = pObj->linkWithOptions(
        &outLinkedComponentType,
        compilerOptionEntryCount,
        compilerOptionEntries,
        &outDiagnostics);

    if (outLinkedComponentType && SLANG_SUCCEEDED(res))
    {
        m_objectMap.addIfNotExists(outLinkedComponentTypeId, outLinkedComponentType);
    }

    ReplayConsumer::printDiagnosticMessage(outDiagnostics);
    return res;
}


void ReplayConsumer::printDiagnosticMessage(slang::IBlob* diagnosticsBlob)
{
    if (diagnosticsBlob)
    {
        const char* diagnostics = (const char*)diagnosticsBlob->getBufferPointer();
        slangRecordLog(LogLevel::Error, "Replayer: %s\n", diagnostics);
    }
}

void ReplayConsumer::CreateGlobalSession(
    SlangGlobalSessionDesc const& desc,
    ObjectID outGlobalSessionId)
{
    OutputObjectSanityCheck(outGlobalSessionId);

    slang::IGlobalSession* outGlobalSession{};
    slang::createGlobalSession(&desc, &outGlobalSession);

    if (outGlobalSession)
    {
        m_objectMap.add(outGlobalSessionId, outGlobalSession);
        slangRecordLog(
            LogLevel::Verbose,
            "createGlobalSession, capture: 0x%X, new: 0x%X\n",
            outGlobalSessionId,
            reinterpret_cast<AddressFormat>(outGlobalSession));
    }
    else
    {
        slangRecordLog(
            LogLevel::Error,
            "createGlobalSession fails, outGlobalSessionId: 0x%X\n",
            outGlobalSessionId);
    }
}

void ReplayConsumer::IGlobalSession_createSession(
    ObjectID objectId,
    slang::SessionDesc const& desc,
    ObjectID outSessionId)
{
    InputObjectSanityCheck(objectId);
    OutputObjectSanityCheck(outSessionId);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    slang::ISession* outSession{};
    SlangResult res = globalSession->createSession(desc, &outSession);

    if (outSession && SLANG_SUCCEEDED(res))
    {
        m_objectMap.add(outSessionId, outSession);
    }
    else
    {
        slangRecordLog(
            LogLevel::Error,
            "IGlobalSession::createSession fails, ret: 0x%X, this: 0x%X\n",
            res,
            objectId);
    }
}


void ReplayConsumer::IGlobalSession_findProfile(ObjectID objectId, char const* name)
{
    InputObjectSanityCheck(objectId);
    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    globalSession->findProfile(name);
}


void ReplayConsumer::IGlobalSession_setDownstreamCompilerPath(
    ObjectID objectId,
    SlangPassThrough passThrough,
    char const* path)
{
    InputObjectSanityCheck(objectId);
    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    globalSession->setDownstreamCompilerPath(passThrough, path);
}


void ReplayConsumer::IGlobalSession_setDownstreamCompilerPrelude(
    ObjectID objectId,
    SlangPassThrough inPassThrough,
    char const* prelude)
{
    InputObjectSanityCheck(objectId);
    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    globalSession->setDownstreamCompilerPrelude(inPassThrough, prelude);
}


void ReplayConsumer::IGlobalSession_getDownstreamCompilerPrelude(
    ObjectID objectId,
    SlangPassThrough inPassThrough,
    ObjectID outPreludeId)
{
    InputObjectSanityCheck(objectId);
    OutputObjectSanityCheck(outPreludeId);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);

    ISlangBlob* outPrelude{};
    globalSession->getDownstreamCompilerPrelude(inPassThrough, &outPrelude);
    if (outPrelude)
    {
        m_objectMap.add(outPreludeId, outPrelude);
    }
}


void ReplayConsumer::IGlobalSession_setDefaultDownstreamCompiler(
    ObjectID objectId,
    SlangSourceLanguage sourceLanguage,
    SlangPassThrough defaultCompiler)
{
    InputObjectSanityCheck(objectId);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    SlangResult res = globalSession->setDefaultDownstreamCompiler(sourceLanguage, defaultCompiler);

    if (SLANG_FAILED(res))
    {
        slangRecordLog(
            LogLevel::Error,
            "IGlobalSession::setDefaultDownstreamCompiler fails, ret: 0x%X, this: 0x%X\n",
            res,
            objectId);
    }
}


void ReplayConsumer::IGlobalSession_getDefaultDownstreamCompiler(
    ObjectID objectId,
    SlangSourceLanguage sourceLanguage)
{
    InputObjectSanityCheck(objectId);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    globalSession->getDefaultDownstreamCompiler(sourceLanguage);
}


void ReplayConsumer::IGlobalSession_setLanguagePrelude(
    ObjectID objectId,
    SlangSourceLanguage inSourceLanguage,
    char const* prelude)
{
    InputObjectSanityCheck(objectId);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    globalSession->setLanguagePrelude(inSourceLanguage, prelude);
}


void ReplayConsumer::IGlobalSession_getLanguagePrelude(
    ObjectID objectId,
    SlangSourceLanguage inSourceLanguage,
    ObjectID outPreludeId)
{
    InputObjectSanityCheck(objectId);
    OutputObjectSanityCheck(outPreludeId);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);

    ISlangBlob* outPrelude{};
    globalSession->getLanguagePrelude(inSourceLanguage, &outPrelude);
    if (outPrelude)
    {
        m_objectMap.add(outPreludeId, outPrelude);
    }
}


void ReplayConsumer::IGlobalSession_createCompileRequest(
    ObjectID objectId,
    ObjectID outCompileRequest)
{
    InputObjectSanityCheck(objectId);
    OutputObjectSanityCheck(outCompileRequest);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    slang::ICompileRequest* outRequest{};
    SLANG_ALLOW_DEPRECATED_BEGIN
    SlangResult res = globalSession->createCompileRequest(&outRequest);
    SLANG_ALLOW_DEPRECATED_END

    if (outRequest && SLANG_SUCCEEDED(res))
    {
        m_objectMap.add(outCompileRequest, outRequest);
    }
    else
    {
        slangRecordLog(
            LogLevel::Error,
            "IGlobalSession::createCompileRequest fails, ret: 0x%X, this: 0x%X\n",
            res,
            objectId);
    }
}


void ReplayConsumer::IGlobalSession_addBuiltins(
    ObjectID objectId,
    char const* sourcePath,
    char const* sourceString)
{
    InputObjectSanityCheck(objectId);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    globalSession->addBuiltins(sourcePath, sourceString);
}


void ReplayConsumer::IGlobalSession_setSharedLibraryLoader(ObjectID objectId, ObjectID loaderId)
{
    InputObjectSanityCheck(objectId);
    InputObjectSanityCheck(loaderId);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    ISlangSharedLibraryLoader* loader = getObjectPointer<ISlangSharedLibraryLoader>(loaderId);
    globalSession->setSharedLibraryLoader(loader);
}


void ReplayConsumer::IGlobalSession_getSharedLibraryLoader(ObjectID objectId, ObjectID outLoaderId)
{
    InputObjectSanityCheck(objectId);
    OutputObjectSanityCheck(outLoaderId);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    ISlangSharedLibraryLoader* loader = globalSession->getSharedLibraryLoader();
    if (loader)
    {
        m_objectMap.add(outLoaderId, loader);
    }
    else
    {
        slangRecordLog(
            LogLevel::Error,
            "IGlobalSession::getSharedLibraryLoader fails, this: 0x%X\n",
            objectId);
    }
}


void ReplayConsumer::IGlobalSession_checkCompileTargetSupport(
    ObjectID objectId,
    SlangCompileTarget target)
{
    InputObjectSanityCheck(objectId);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    SlangResult res = globalSession->checkCompileTargetSupport(target);

    if (SLANG_FAILED(res))
    {
        slangRecordLog(
            LogLevel::Error,
            "IGlobalSession::checkCompileTargetSupport fails, ret: 0x%X, this: 0x%X\n",
            res,
            objectId);
    }
}


void ReplayConsumer::IGlobalSession_checkPassThroughSupport(
    ObjectID objectId,
    SlangPassThrough passThrough)
{
    InputObjectSanityCheck(objectId);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    SlangResult res = globalSession->checkPassThroughSupport(passThrough);

    if (SLANG_FAILED(res))
    {
        slangRecordLog(
            LogLevel::Error,
            "IGlobalSession::checkPassThroughSupport fails, ret: 0x%X, this: 0x%X\n",
            res,
            objectId);
    }
}


void ReplayConsumer::IGlobalSession_compileCoreModule(
    ObjectID objectId,
    slang::CompileCoreModuleFlags flags)
{
    InputObjectSanityCheck(objectId);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    SlangResult res = globalSession->compileCoreModule(flags);

    if (SLANG_FAILED(res))
    {
        slangRecordLog(
            LogLevel::Error,
            "IGlobalSession::compileCoreModule fails, ret: 0x%X, this: 0x%X\n",
            res,
            objectId);
    }
}


void ReplayConsumer::IGlobalSession_loadCoreModule(
    ObjectID objectId,
    const void* coreModule,
    size_t coreModuleSizeInBytes)
{
    InputObjectSanityCheck(objectId);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    SlangResult res = globalSession->loadCoreModule(coreModule, coreModuleSizeInBytes);

    if (SLANG_FAILED(res))
    {
        slangRecordLog(
            LogLevel::Error,
            "IGlobalSession::loadCoreModule fails, ret: 0x%X, this: 0x%X\n",
            res,
            objectId);
    }
}


void ReplayConsumer::IGlobalSession_saveCoreModule(
    ObjectID objectId,
    SlangArchiveType archiveType,
    ObjectID outBlobId)
{
    InputObjectSanityCheck(objectId);
    OutputObjectSanityCheck(outBlobId);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    ISlangBlob* outBlob{};
    SlangResult res = globalSession->saveCoreModule(archiveType, &outBlob);

    if (outBlob && SLANG_SUCCEEDED(res))
    {
        m_objectMap.add(outBlobId, outBlob);
    }
    else
    {
        slangRecordLog(
            LogLevel::Error,
            "IGlobalSession::saveCoreModule fails, ret: 0x%X, this: 0x%X\n",
            res,
            objectId);
    }
}


void ReplayConsumer::IGlobalSession_findCapability(ObjectID objectId, char const* name)
{
    InputObjectSanityCheck(objectId);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    globalSession->findCapability(name);
}


void ReplayConsumer::IGlobalSession_setDownstreamCompilerForTransition(
    ObjectID objectId,
    SlangCompileTarget source,
    SlangCompileTarget target,
    SlangPassThrough compiler)
{
    InputObjectSanityCheck(objectId);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    globalSession->setDownstreamCompilerForTransition(source, target, compiler);
}


void ReplayConsumer::IGlobalSession_getDownstreamCompilerForTransition(
    ObjectID objectId,
    SlangCompileTarget source,
    SlangCompileTarget target)
{
    InputObjectSanityCheck(objectId);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    globalSession->getDownstreamCompilerForTransition(source, target);
}


void ReplayConsumer::IGlobalSession_setSPIRVCoreGrammar(ObjectID objectId, char const* jsonPath)
{
    InputObjectSanityCheck(objectId);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    SlangResult res = globalSession->setSPIRVCoreGrammar(jsonPath);

    if (SLANG_FAILED(res))
    {
        slangRecordLog(
            LogLevel::Error,
            "IGlobalSession::setSPIRVCoreGrammar fails, ret: 0x%X, this: 0x%X\n",
            res,
            objectId);
    }
}


void ReplayConsumer::IGlobalSession_parseCommandLineArguments(
    ObjectID objectId,
    int argc,
    const char* const* argv,
    ObjectID outSessionDescId,
    ObjectID outAllocationId)
{
    InputObjectSanityCheck(objectId);
    OutputObjectSanityCheck(outAllocationId);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    slang::SessionDesc sessionDesc{};
    ISlangUnknown* allocation{};

    // Note: we don't save the sessionDesc here, because it's an output to user, but we do need to
    // save the allocation, because it holds the auxiliary data for the sessionDesc. So if use
    // provide the same sessionDesc to slang, we won't hit any segfault.
    SlangResult res =
        globalSession->parseCommandLineArguments(argc, argv, &sessionDesc, &allocation);

    if (SLANG_SUCCEEDED(res))
    {
        m_objectMap.add(outAllocationId, allocation);
    }
    else
    {
        slangRecordLog(
            LogLevel::Debug,
            "IGlobalSession::parseCommandLineArguments fails, ret: 0x%X, this: 0x%X\n",
            res,
            objectId);
    }
}


void ReplayConsumer::IGlobalSession_getSessionDescDigest(
    ObjectID objectId,
    slang::SessionDesc* sessionDesc,
    ObjectID outBlobId)
{
    InputObjectSanityCheck(objectId);
    OutputObjectSanityCheck(outBlobId);

    slang::IGlobalSession* globalSession = getObjectPointer<slang::IGlobalSession>(objectId);
    ISlangBlob* outBlob{};
    SlangResult res = globalSession->getSessionDescDigest(sessionDesc, &outBlob);

    if (outBlob && SLANG_SUCCEEDED(res))
    {
        m_objectMap.add(outBlobId, outBlob);
    }
    else
    {
        slangRecordLog(
            LogLevel::Error,
            "IGlobalSession::getSessionDescDigest fails, ret:0x%X, this: 0x%X\n",
            res,
            objectId);
    }
}


// ISession
void ReplayConsumer::ISession_getGlobalSession(ObjectID objectId, ObjectID outGlobalSessionId)
{
    // No need to replay this function
}


void ReplayConsumer::ISession_loadModule(
    ObjectID objectId,
    const char* moduleName,
    ObjectID outDiagnosticsId,
    ObjectID outModuleId)
{
    InputObjectSanityCheck(objectId);

    slang::ISession* session = getObjectPointer<slang::ISession>(objectId);
    slang::IModule* outModule{};
    slang::IBlob* outDiagnostics{};

    // loadModule could return a new module or an existing module, so can't use
    // OutputObjectSanityCheck here
    outModule = session->loadModule(moduleName, &outDiagnostics);

    if (outModule)
    {
        // Save the module if it's not being tracked
        m_objectMap.addIfNotExists(outModuleId, outModule);
    }
    else
    {
        slangRecordLog(
            LogLevel::Error,
            "ISession::loadModule returns nullptr, this: 0x%X\n",
            objectId);
    }

    printDiagnosticMessage(outDiagnostics);
}


void ReplayConsumer::ISession_loadModuleFromIRBlob(
    ObjectID objectId,
    const char* moduleName,
    const char* path,
    slang::IBlob* source,
    ObjectID outDiagnosticsId,
    ObjectID outModuleId)
{
    InputObjectSanityCheck(objectId);

    slang::ISession* session = getObjectPointer<slang::ISession>(objectId);
    slang::IModule* outModule{};
    slang::IBlob* outDiagnostics{};

    outModule = session->loadModuleFromIRBlob(moduleName, path, source, &outDiagnostics);

    if (outModule)
    {
        // Save the module if it's not being tracked
        m_objectMap.addIfNotExists(outModuleId, outModule);
    }
    else
    {
        slangRecordLog(
            LogLevel::Error,
            "ISession::loadModule returns nullptr, this: 0x%X\n",
            objectId);
    }

    printDiagnosticMessage(outDiagnostics);
}


void ReplayConsumer::ISession_loadModuleFromSource(
    ObjectID objectId,
    const char* moduleName,
    const char* path,
    slang::IBlob* source,
    ObjectID outDiagnosticsId,
    ObjectID outModuleId)
{
    slang::ISession* session = getObjectPointer<slang::ISession>(objectId);
    slang::IModule* outModule{};
    slang::IBlob* outDiagnostics{};

    outModule = session->loadModuleFromSource(moduleName, path, source, &outDiagnostics);

    if (outModule)
    {
        // Save the module if it's not being tracked
        m_objectMap.addIfNotExists(outModuleId, outModule);
    }
    else
    {
        slangRecordLog(
            LogLevel::Error,
            "ISession::loadModule returns nullptr, this: 0x%X\n",
            objectId);
    }

    printDiagnosticMessage(outDiagnostics);
}


void ReplayConsumer::ISession_loadModuleFromSourceString(
    ObjectID objectId,
    const char* moduleName,
    const char* path,
    const char* string,
    ObjectID outDiagnosticsId,
    ObjectID outModuleId)
{
    slang::ISession* session = getObjectPointer<slang::ISession>(objectId);
    slang::IModule* outModule{};
    slang::IBlob* outDiagnostics{};

    outModule = session->loadModuleFromSourceString(moduleName, path, string, &outDiagnostics);

    if (outModule)
    {
        // Save the module if it's not being tracked
        m_objectMap.addIfNotExists(outModuleId, outModule);
    }
    else
    {
        slangRecordLog(
            LogLevel::Error,
            "ISession::loadModule returns nullptr, this: 0x%X\n",
            objectId);
    }

    printDiagnosticMessage(outDiagnostics);
}


void ReplayConsumer::ISession_createCompositeComponentType(
    ObjectID objectId,
    ObjectID* componentTypeIds,
    SlangInt componentTypeCount,
    ObjectID outCompositeComponentTypeId,
    ObjectID outDiagnosticsId)
{
    InputObjectSanityCheck(objectId);
    for (SlangInt i = 0; i < componentTypeCount; i++)
    {
        InputObjectSanityCheck(componentTypeIds[i]);
    }

    // We don't need to check existence of outCompositeComponentTypeId, because it could be the same
    // object as the input one

    slang::ISession* session = getObjectPointer<slang::ISession>(objectId);

    Slang::List<slang::IComponentType*> componentTypes;
    componentTypes.reserve(componentTypeCount);

    for (SlangInt i = 0; i < componentTypeCount; i++)
    {
        componentTypes.add(getObjectPointer<slang::IComponentType>(componentTypeIds[i]));
    }

    slang::IComponentType* outCompositeComponentType{};
    slang::IBlob* outDiagnostics{};

    SlangResult res = session->createCompositeComponentType(
        componentTypes.getBuffer(),
        componentTypeCount,
        &outCompositeComponentType,
        &outDiagnostics);

    if (outCompositeComponentType && SLANG_SUCCEEDED(res))
    {
        m_objectMap.addIfNotExists(outCompositeComponentTypeId, outCompositeComponentType);
    }
    else
    {
        slangRecordLog(
            LogLevel::Error,
            "ISession::createCompositeComponentType fails, ret: 0x%X, this: 0x%X\n",
            objectId);
    }

    printDiagnosticMessage(outDiagnostics);
}

// TODO: implement those functions related to TypeReflection
void ReplayConsumer::ISession_specializeType(
    ObjectID objectId,
    ObjectID typeId,
    slang::SpecializationArg const* specializationArgs,
    SlangInt specializationArgCount,
    ObjectID outDiagnosticsId,
    ObjectID outTypeReflectionId)
{
}

void ReplayConsumer::ISession_getTypeLayout(
    ObjectID objectId,
    ObjectID typeId,
    SlangInt targetIndex,
    slang::LayoutRules rules,
    ObjectID outDiagnosticsId,
    ObjectID outTypeLayoutReflection)
{
}

void ReplayConsumer::ISession_getContainerType(
    ObjectID objectId,
    ObjectID elementType,
    slang::ContainerType containerType,
    ObjectID outDiagnosticsId,
    ObjectID outTypeReflectionId)
{
}

void ReplayConsumer::ISession_getDynamicType(ObjectID objectId, ObjectID outTypeReflectionId) {}

void ReplayConsumer::ISession_getTypeRTTIMangledName(
    ObjectID objectId,
    ObjectID typeId,
    ObjectID outNameBlobId)
{
}

void ReplayConsumer::ISession_getTypeConformanceWitnessMangledName(
    ObjectID objectId,
    ObjectID typeId,
    ObjectID interfaceTypeId,
    ObjectID outNameBlobId)
{
}


void ReplayConsumer::ISession_getTypeConformanceWitnessSequentialID(
    ObjectID objectId,
    ObjectID typeId,
    ObjectID interfaceTypeId,
    uint32_t outId)
{
}

void ReplayConsumer::ISession_createTypeConformanceComponentType(
    ObjectID objectId,
    ObjectID typeId,
    ObjectID interfaceTypeId,
    ObjectID outConformanceId,
    SlangInt conformanceIdOverride,
    ObjectID outDiagnosticsId)
{
}
// End of TODO


void ReplayConsumer::ISession_createCompileRequest(ObjectID objectId, ObjectID outCompileRequestId)
{
    InputObjectSanityCheck(objectId);
    OutputObjectSanityCheck(outCompileRequestId);

    slang::ISession* session = getObjectPointer<slang::ISession>(objectId);
    slang::ICompileRequest* outRequest{};
    SlangResult res = session->createCompileRequest(&outRequest);

    if (outRequest && SLANG_SUCCEEDED(res))
    {
        m_objectMap.add(outCompileRequestId, outRequest);
    }
    else
    {
        slangRecordLog(
            LogLevel::Error,
            "ISession::createCompileRequest fails, ret:0x%X, this: 0x%X\n",
            res,
            objectId);
    }
}


void ReplayConsumer::ISession_getLoadedModule(
    ObjectID objectId,
    SlangInt index,
    ObjectID outModuleId)
{
    InputObjectSanityCheck(objectId);

    slang::ISession* session = getObjectPointer<slang::ISession>(objectId);
    slang::IModule* outModule = session->getLoadedModule(index);

    if (!m_objectMap.tryGetValue(outModuleId))
    {
        // This module should already be tracked during loadModule() call, if not this should be a
        // bug in the replayer or record.
        slangRecordLog(
            LogLevel::Error,
            "ISession::getLoadedModule: this: 0x%X, module 0x%X is not tracked \n",
            objectId,
            outModuleId);
        m_objectMap.add(outModuleId, outModule);
    }

    if (!outModule)
    {
        slangRecordLog(
            LogLevel::Error,
            "ISession::getLoadedModule returns nullptr, this: 0x%X\n",
            objectId);
    }
}


// IModule
void ReplayConsumer::IModule_findEntryPointByName(
    ObjectID objectId,
    char const* name,
    ObjectID outEntryPointId)
{
    InputObjectSanityCheck(objectId);

    slang::IModule* module = getObjectPointer<slang::IModule>(objectId);
    slang::IEntryPoint* outEntryPoint{};
    SlangResult res = module->findEntryPointByName(name, &outEntryPoint);

    if (outEntryPoint && SLANG_SUCCEEDED(res))
    {
        m_objectMap.addIfNotExists(outEntryPointId, outEntryPoint);
    }
    else
    {
        slangRecordLog(
            LogLevel::Error,
            "IModule::findEntryPointByName fails, ret: 0x%X, this: 0x%X\n",
            objectId);
    }
}


void ReplayConsumer::IModule_getDefinedEntryPoint(
    ObjectID objectId,
    SlangInt32 index,
    ObjectID outEntryPointId)
{
    InputObjectSanityCheck(objectId);

    slang::IModule* module = getObjectPointer<slang::IModule>(objectId);
    slang::IEntryPoint* outEntryPoint{};
    SlangResult res = module->getDefinedEntryPoint(index, &outEntryPoint);

    if (outEntryPoint && SLANG_SUCCEEDED(res))
    {
        m_objectMap.addIfNotExists(outEntryPointId, outEntryPoint);
    }
    else
    {
        slangRecordLog(
            LogLevel::Error,
            "IModule::getDefinedEntryPoint returns nullptr, this: 0x%X\n",
            objectId);
    }
}


void ReplayConsumer::IModule_serialize(ObjectID objectId, ObjectID outSerializedBlobId)
{
    InputObjectSanityCheck(objectId);
    OutputObjectSanityCheck(outSerializedBlobId);

    slang::IModule* module = getObjectPointer<slang::IModule>(objectId);
    slang::IBlob* outBlob{};
    SlangResult res = module->serialize(&outBlob);

    if (outBlob && SLANG_SUCCEEDED(res))
    {
        m_objectMap.add(outSerializedBlobId, outBlob);
    }
    else
    {
        slangRecordLog(
            LogLevel::Error,
            "IModule::serialize fails, ret: 0x%X, this: 0x%X\n",
            res,
            objectId);
    }
}


void ReplayConsumer::IModule_writeToFile(ObjectID objectId, char const* fileName)
{
    InputObjectSanityCheck(objectId);

    slang::IModule* module = getObjectPointer<slang::IModule>(objectId);
    SlangResult res = module->writeToFile(fileName);

    if (SLANG_FAILED(res))
    {
        slangRecordLog(
            LogLevel::Error,
            "IModule::writeToFile fails, ret: 0x%X, this: 0x%X\n",
            res,
            objectId);
    }
}


void ReplayConsumer::IModule_findAndCheckEntryPoint(
    ObjectID objectId,
    char const* name,
    SlangStage stage,
    ObjectID outEntryPointId,
    ObjectID outDiagnosticsId)
{
    InputObjectSanityCheck(objectId);

    slang::IModule* module = getObjectPointer<slang::IModule>(objectId);
    slang::IEntryPoint* outEntryPoint{};
    slang::IBlob* outDiagnostics{};

    SlangResult res = module->findAndCheckEntryPoint(name, stage, &outEntryPoint, &outDiagnostics);

    if (outEntryPoint && SLANG_SUCCEEDED(res))
    {
        m_objectMap.addIfNotExists(outEntryPointId, outEntryPoint);
    }
    else
    {
        slangRecordLog(
            LogLevel::Error,
            "IModule::findAndCheckEntryPoint fails, ret: 0x%X, this: 0x%X\n",
            res,
            objectId);
    }

    printDiagnosticMessage(outDiagnostics);
}


void ReplayConsumer::IModule_getSession(ObjectID objectId, ObjectID outSessionId)
{
    // No need to replay this function
}

void ReplayConsumer::IModule_getLayout(
    ObjectID objectId,
    SlangInt targetIndex,
    ObjectID outDiagnosticsId,
    ObjectID retProgramLayoutId)
{
    SlangResult res =
        m_commonReplayer.getLayout(objectId, targetIndex, outDiagnosticsId, retProgramLayoutId);
    FAIL_WITH_LOG(IModule::getLayout);
}


void ReplayConsumer::IModule_getEntryPointCode(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outCodeId,
    ObjectID outDiagnosticsId)
{
    SlangResult res =
        m_commonReplayer
            .getEntryPointCode(objectId, entryPointIndex, targetIndex, outCodeId, outDiagnosticsId);
    FAIL_WITH_LOG(IModule::getEntryPointCode);
}


void ReplayConsumer::IModule_getTargetCode(
    ObjectID objectId,
    SlangInt targetIndex,
    ObjectID outCodeId,
    ObjectID outDiagnosticsId)
{
    SlangResult res =
        m_commonReplayer.getTargetCode(objectId, targetIndex, outCodeId, outDiagnosticsId);
    FAIL_WITH_LOG(IModule::getTargetCode);
}


void ReplayConsumer::IModule_getResultAsFileSystem(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outFileSystemId)
{
    SlangResult res = m_commonReplayer.getResultAsFileSystem(
        objectId,
        entryPointIndex,
        targetIndex,
        outFileSystemId);
    FAIL_WITH_LOG(IModule::getResultAsFileSystem);
}


void ReplayConsumer::IModule_getEntryPointHash(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outHashId)
{
    SlangResult res =
        m_commonReplayer.getEntryPointHash(objectId, entryPointIndex, targetIndex, outHashId);
    FAIL_WITH_LOG(IModule::getEntryPointHash);
}


void ReplayConsumer::IModule_specialize(
    ObjectID objectId,
    slang::SpecializationArg const* specializationArgs,
    SlangInt specializationArgCount,
    ObjectID outSpecializedComponentTypeId,
    ObjectID outDiagnosticsId)
{
    SlangResult res = m_commonReplayer.specialize(
        objectId,
        specializationArgs,
        specializationArgCount,
        outSpecializedComponentTypeId,
        outDiagnosticsId);
    FAIL_WITH_LOG(IModule::specialize);
}


void ReplayConsumer::IModule_link(
    ObjectID objectId,
    ObjectID outLinkedComponentTypeId,
    ObjectID outDiagnosticsId)
{
    SlangResult res = m_commonReplayer.link(objectId, outLinkedComponentTypeId, outDiagnosticsId);
    FAIL_WITH_LOG(IModule::link);
}


void ReplayConsumer::IModule_getEntryPointHostCallable(
    ObjectID objectId,
    int entryPointIndex,
    int targetIndex,
    ObjectID outSharedLibrary,
    ObjectID outDiagnostics)
{
    SlangResult res = m_commonReplayer.getEntryPointHostCallable(
        objectId,
        entryPointIndex,
        targetIndex,
        outSharedLibrary,
        outDiagnostics);
    FAIL_WITH_LOG(IModule::getEntryPointHostCallable);
}


void ReplayConsumer::IModule_renameEntryPoint(
    ObjectID objectId,
    const char* newName,
    ObjectID outEntryPointId)
{
    SlangResult res = m_commonReplayer.renameEntryPoint(objectId, newName, outEntryPointId);
    FAIL_WITH_LOG(IModule::renameEntryPoint);
}


void ReplayConsumer::IModule_linkWithOptions(
    ObjectID objectId,
    ObjectID outLinkedComponentTypeId,
    uint32_t compilerOptionEntryCount,
    slang::CompilerOptionEntry* compilerOptionEntries,
    ObjectID outDiagnosticsId)
{
    SlangResult res = m_commonReplayer.linkWithOptions(
        objectId,
        outLinkedComponentTypeId,
        compilerOptionEntryCount,
        compilerOptionEntries,
        outDiagnosticsId);
    FAIL_WITH_LOG(IModule::linkWithOptions);
}


// IEntryPoint
void ReplayConsumer::IEntryPoint_getSession(ObjectID objectId, ObjectID outSessionId) {}


void ReplayConsumer::IEntryPoint_getLayout(
    ObjectID objectId,
    SlangInt targetIndex,
    ObjectID outDiagnosticsId,
    ObjectID retProgramLayoutId)
{
    SlangResult res =
        m_commonReplayer.getLayout(objectId, targetIndex, outDiagnosticsId, retProgramLayoutId);
    FAIL_WITH_LOG(IEntryPoint::getLayout);
}


void ReplayConsumer::IEntryPoint_getEntryPointCode(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outCode,
    ObjectID outDiagnostics)
{
    SlangResult res =
        m_commonReplayer
            .getEntryPointCode(objectId, entryPointIndex, targetIndex, outCode, outDiagnostics);
    FAIL_WITH_LOG(IEntryPoint::getEntryPointCode);
}


void ReplayConsumer::IEntryPoint_getTargetCode(
    ObjectID objectId,
    SlangInt targetIndex,
    ObjectID outCode,
    ObjectID outDiagnostics)
{
    SlangResult res =
        m_commonReplayer.getTargetCode(objectId, targetIndex, outCode, outDiagnostics);
    FAIL_WITH_LOG(IEntryPoint::getTargetCode);
}


void ReplayConsumer::IEntryPoint_getResultAsFileSystem(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outFileSystem)
{
    SlangResult res = m_commonReplayer.getResultAsFileSystem(
        objectId,
        entryPointIndex,
        targetIndex,
        outFileSystem);
    FAIL_WITH_LOG(IEntryPoint::getResultAsFileSystem);
}


void ReplayConsumer::IEntryPoint_getEntryPointHash(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outHashId)
{
    SlangResult res =
        m_commonReplayer.getEntryPointHash(objectId, entryPointIndex, targetIndex, outHashId);
    FAIL_WITH_LOG(IEntryPoint::getEntryPointHash);
}


void ReplayConsumer::IEntryPoint_specialize(
    ObjectID objectId,
    slang::SpecializationArg const* specializationArgs,
    SlangInt specializationArgCount,
    ObjectID outSpecializedComponentTypeId,
    ObjectID outDiagnosticsId)
{
    SlangResult res = m_commonReplayer.specialize(
        objectId,
        specializationArgs,
        specializationArgCount,
        outSpecializedComponentTypeId,
        outDiagnosticsId);
    FAIL_WITH_LOG(IModule::specialize);
}


void ReplayConsumer::IEntryPoint_link(
    ObjectID objectId,
    ObjectID outLinkedComponentTypeId,
    ObjectID outDiagnosticsId)
{
    SlangResult res = m_commonReplayer.link(objectId, outLinkedComponentTypeId, outDiagnosticsId);
    FAIL_WITH_LOG(IEntryPoint::link);
}


void ReplayConsumer::IEntryPoint_getEntryPointHostCallable(
    ObjectID objectId,
    int entryPointIndex,
    int targetIndex,
    ObjectID outSharedLibrary,
    ObjectID outDiagnostics)
{
    SlangResult res = m_commonReplayer.getEntryPointHostCallable(
        objectId,
        entryPointIndex,
        targetIndex,
        outSharedLibrary,
        outDiagnostics);
    FAIL_WITH_LOG(IEntryPoint::getEntryPointHostCallable);
}


void ReplayConsumer::IEntryPoint_renameEntryPoint(
    ObjectID objectId,
    const char* newName,
    ObjectID outEntryPointId)
{
    SlangResult res = m_commonReplayer.renameEntryPoint(objectId, newName, outEntryPointId);
    FAIL_WITH_LOG(IEntryPoint::renameEntryPoint);
}


void ReplayConsumer::IEntryPoint_linkWithOptions(
    ObjectID objectId,
    ObjectID outLinkedComponentTypeId,
    uint32_t compilerOptionEntryCount,
    slang::CompilerOptionEntry* compilerOptionEntries,
    ObjectID outDiagnosticsId)
{
    SlangResult res = m_commonReplayer.linkWithOptions(
        objectId,
        outLinkedComponentTypeId,
        compilerOptionEntryCount,
        compilerOptionEntries,
        outDiagnosticsId);
    FAIL_WITH_LOG(IEntryPoint::linkWithOptions);
}


// ICompositeComponentType
void ReplayConsumer::ICompositeComponentType_getSession(ObjectID objectId, ObjectID outSessionId) {}


void ReplayConsumer::ICompositeComponentType_getLayout(
    ObjectID objectId,
    SlangInt targetIndex,
    ObjectID outDiagnosticsId,
    ObjectID retProgramLayoutId)
{
    SlangResult res =
        m_commonReplayer.getLayout(objectId, targetIndex, outDiagnosticsId, retProgramLayoutId);
    FAIL_WITH_LOG(ICompositeComponentType::getLayout);
}


void ReplayConsumer::ICompositeComponentType_getEntryPointCode(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outCode,
    ObjectID outDiagnostics)
{
    SlangResult res =
        m_commonReplayer
            .getEntryPointCode(objectId, entryPointIndex, targetIndex, outCode, outDiagnostics);
    FAIL_WITH_LOG(ICompositeComponentType::getEntryPointCode);
}


void ReplayConsumer::ICompositeComponentType_getTargetCode(
    ObjectID objectId,
    SlangInt targetIndex,
    ObjectID outCode,
    ObjectID outDiagnostics)
{
    SlangResult res =
        m_commonReplayer.getTargetCode(objectId, targetIndex, outCode, outDiagnostics);
    FAIL_WITH_LOG(ICompositeComponentType::getTargetCode);
}


void ReplayConsumer::ICompositeComponentType_getResultAsFileSystem(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outFileSystem)
{
    SlangResult res = m_commonReplayer.getResultAsFileSystem(
        objectId,
        entryPointIndex,
        targetIndex,
        outFileSystem);
    FAIL_WITH_LOG(ICompositeComponentType::getResultAsFileSystem);
}


void ReplayConsumer::ICompositeComponentType_getEntryPointHash(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outHashId)
{
    SlangResult res =
        m_commonReplayer.getEntryPointHash(objectId, entryPointIndex, targetIndex, outHashId);
    FAIL_WITH_LOG(ICompositeComponentType::getEntryPointHash);
}


void ReplayConsumer::ICompositeComponentType_specialize(
    ObjectID objectId,
    slang::SpecializationArg const* specializationArgs,
    SlangInt specializationArgCount,
    ObjectID outSpecializedComponentTypeId,
    ObjectID outDiagnosticsId)
{
    SlangResult res = m_commonReplayer.specialize(
        objectId,
        specializationArgs,
        specializationArgCount,
        outSpecializedComponentTypeId,
        outDiagnosticsId);
    FAIL_WITH_LOG(IModule::specialize);
}


void ReplayConsumer::ICompositeComponentType_link(
    ObjectID objectId,
    ObjectID outLinkedComponentTypeId,
    ObjectID outDiagnosticsId)
{
    SlangResult res = m_commonReplayer.link(objectId, outLinkedComponentTypeId, outDiagnosticsId);
    FAIL_WITH_LOG(ICompositeComponentType::link);
}


void ReplayConsumer::ICompositeComponentType_getEntryPointHostCallable(
    ObjectID objectId,
    int entryPointIndex,
    int targetIndex,
    ObjectID outSharedLibrary,
    ObjectID outDiagnostics)
{
    SlangResult res = m_commonReplayer.getEntryPointHostCallable(
        objectId,
        entryPointIndex,
        targetIndex,
        outSharedLibrary,
        outDiagnostics);
    FAIL_WITH_LOG(ICompositeComponentType::getEntryPointHostCallable);
}


void ReplayConsumer::ICompositeComponentType_renameEntryPoint(
    ObjectID objectId,
    const char* newName,
    ObjectID outEntryPointId)
{
    SlangResult res = m_commonReplayer.renameEntryPoint(objectId, newName, outEntryPointId);
    FAIL_WITH_LOG(ICompositeComponentType::renameEntryPoint);
}


void ReplayConsumer::ICompositeComponentType_linkWithOptions(
    ObjectID objectId,
    ObjectID outLinkedComponentTypeId,
    uint32_t compilerOptionEntryCount,
    slang::CompilerOptionEntry* compilerOptionEntries,
    ObjectID outDiagnosticsId)
{
    SlangResult res = m_commonReplayer.linkWithOptions(
        objectId,
        outLinkedComponentTypeId,
        compilerOptionEntryCount,
        compilerOptionEntries,
        outDiagnosticsId);
    FAIL_WITH_LOG(ICompositeComponentType::linkWithOptions);
}


// ITypeConformance
void ReplayConsumer::ITypeConformance_getSession(ObjectID objectId, ObjectID outSessionId) {}


void ReplayConsumer::ITypeConformance_getLayout(
    ObjectID objectId,
    SlangInt targetIndex,
    ObjectID outDiagnosticsId,
    ObjectID retProgramLayoutId)
{
    SlangResult res =
        m_commonReplayer.getLayout(objectId, targetIndex, outDiagnosticsId, retProgramLayoutId);
    FAIL_WITH_LOG(ITypeConformance::getLayout);
}


void ReplayConsumer::ITypeConformance_getEntryPointCode(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outCode,
    ObjectID outDiagnostics)
{
    SlangResult res =
        m_commonReplayer
            .getEntryPointCode(objectId, entryPointIndex, targetIndex, outCode, outDiagnostics);
    FAIL_WITH_LOG(ITypeConformance::getEntryPointCode);
}


void ReplayConsumer::ITypeConformance_getTargetCode(
    ObjectID objectId,
    SlangInt targetIndex,
    ObjectID outCode,
    ObjectID outDiagnostics)
{
    SlangResult res =
        m_commonReplayer.getTargetCode(objectId, targetIndex, outCode, outDiagnostics);
    FAIL_WITH_LOG(ITypeConformance::getTargetCode);
}


void ReplayConsumer::ITypeConformance_getResultAsFileSystem(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outFileSystem)
{
    SlangResult res = m_commonReplayer.getResultAsFileSystem(
        objectId,
        entryPointIndex,
        targetIndex,
        outFileSystem);
    FAIL_WITH_LOG(ITypeConformance::getResultAsFileSystem);
}


void ReplayConsumer::ITypeConformance_getEntryPointHash(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outHashId)
{
    SlangResult res =
        m_commonReplayer.getEntryPointHash(objectId, entryPointIndex, targetIndex, outHashId);
    FAIL_WITH_LOG(ITypeConformance::getEntryPointHash);
}


void ReplayConsumer::ITypeConformance_specialize(
    ObjectID objectId,
    slang::SpecializationArg const* specializationArgs,
    SlangInt specializationArgCount,
    ObjectID outSpecializedComponentTypeId,
    ObjectID outDiagnosticsId)
{
    SlangResult res = m_commonReplayer.specialize(
        objectId,
        specializationArgs,
        specializationArgCount,
        outSpecializedComponentTypeId,
        outDiagnosticsId);
    FAIL_WITH_LOG(IModule::specialize);
}


void ReplayConsumer::ITypeConformance_link(
    ObjectID objectId,
    ObjectID outLinkedComponentTypeId,
    ObjectID outDiagnosticsId)
{
    SlangResult res = m_commonReplayer.link(objectId, outLinkedComponentTypeId, outDiagnosticsId);
    FAIL_WITH_LOG(ITypeConformance::link);
}


void ReplayConsumer::ITypeConformance_getEntryPointHostCallable(
    ObjectID objectId,
    int entryPointIndex,
    int targetIndex,
    ObjectID outSharedLibrary,
    ObjectID outDiagnostics)
{
    SlangResult res = m_commonReplayer.getEntryPointHostCallable(
        objectId,
        entryPointIndex,
        targetIndex,
        outSharedLibrary,
        outDiagnostics);
    FAIL_WITH_LOG(ITypeConformance::getEntryPointHostCallable);
}


void ReplayConsumer::ITypeConformance_renameEntryPoint(
    ObjectID objectId,
    const char* newName,
    ObjectID outEntryPointId)
{
    SlangResult res = m_commonReplayer.renameEntryPoint(objectId, newName, outEntryPointId);
    FAIL_WITH_LOG(ITypeConformance::renameEntryPoint);
}


void ReplayConsumer::ITypeConformance_linkWithOptions(
    ObjectID objectId,
    ObjectID outLinkedComponentTypeId,
    uint32_t compilerOptionEntryCount,
    slang::CompilerOptionEntry* compilerOptionEntries,
    ObjectID outDiagnosticsId)
{
    SlangResult res = m_commonReplayer.linkWithOptions(
        objectId,
        outLinkedComponentTypeId,
        compilerOptionEntryCount,
        compilerOptionEntries,
        outDiagnosticsId);
    FAIL_WITH_LOG(ITypeConformance::linkWithOptions);
}


}; // namespace SlangRecord
